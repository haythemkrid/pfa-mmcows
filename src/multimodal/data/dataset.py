"""
src.multimodal.data.dataset
===========================

PyTorch Dataset that pairs a sensor window with the corresponding
UWB-synced camera frames for the MBT.

GPU utilisation fix
-------------------
The original implementation opened and decoded a JPEG on every
``__getitem__`` call.  With 4 DataLoader workers this created a
CPU→GPU pipeline bubble: the GPU finished each batch in ~50 ms but
waited 200-400 ms for the next one, resulting in ~19% GPU utilisation.

The fix is a two-level image cache:

1. **``ImageCache``** — decodes every unique frame the dataset will ever
   need *once* at construction time, stores it as a uint8 (H, W, 3)
   numpy array.  Because many windows share the same UWB-synced frame
   (one frame per 15 s, many windows per frame), the unique frame count
   is small (~5,760 for a 24-hour day at 15-s intervals across 4
   cameras) while window count is ~95k.  All disk I/O moves out of the
   training loop entirely.

2. **Transforms stay in ``__getitem__``** — random augmentations are
   applied *after* the cache lookup, so augmentation diversity is
   preserved.

Memory estimate: 5,760 frames × 256×256×3 bytes ≈ 1.1 GB.
Well within the 25 GB host RAM visible in nvtop.

Set ``data.cache_images: false`` in mbt_default.yaml to fall back to
per-call disk reads if RAM is tight.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────

def build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Frame index
# ─────────────────────────────────────────────────────────────────────────────

class FrameIndex:
    """O(log N) nearest-timestamp lookup for UWB-synced JPEG frames.

    Build once per run, share across all folds and datasets.
    """

    def __init__(
        self,
        visual_data_dir: str | Path,
        date:            str,
        cameras:         list[int],
    ) -> None:
        root = Path(visual_data_dir) / "images" / date

        pairs: list[tuple[int, Path]] = []
        for cam in cameras:
            cam_dir = root / f"cam_{cam}"
            if not cam_dir.exists():
                logger.warning("Camera directory not found: %s", cam_dir)
                continue
            for p in sorted(cam_dir.glob("*.jpg")):
                ts = int(p.stem.split("_")[0])
                pairs.append((ts, p))

        pairs.sort(key=lambda x: x[0])
        self._ts_list:   list[int]  = [ts for ts, _ in pairs]
        self._path_list: list[Path] = [p  for _, p  in pairs]
        logger.info("FrameIndex: %d frames indexed", len(pairs))

    def nearest(self, timestamp: int) -> Path:
        idx = bisect.bisect_left(self._ts_list, timestamp)
        idx = min(idx, len(self._ts_list) - 1)
        if idx > 0 and abs(self._ts_list[idx - 1] - timestamp) < abs(self._ts_list[idx] - timestamp):
            idx -= 1
        return self._path_list[idx]

    def all_paths(self) -> list[Path]:
        return self._path_list


# ─────────────────────────────────────────────────────────────────────────────
# Image cache
# ─────────────────────────────────────────────────────────────────────────────

class ImageCache:
    """Decode every unique frame once into RAM as uint8 numpy arrays.

    Parameters
    ----------
    frame_index:
        The FrameIndex whose paths will be cached.
    resize_to:
        Resize the shorter edge to this value before caching.
        256 is slightly larger than the 224 final crop so that
        RandomResizedCrop has room to work.  Set 0 to skip resize.
    log_every:
        Progress log frequency (frames).
    """

    def __init__(
        self,
        frame_index: FrameIndex,
        resize_to:   int = 256,
        log_every:   int = 500,
    ) -> None:
        paths = frame_index.all_paths()
        logger.info("ImageCache: decoding %d frames into RAM "
                    "(resize_to=%d)...", len(paths), resize_to)

        self._cache: dict[Path, np.ndarray] = {}
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            if resize_to > 0:
                w, h  = img.size
                scale = resize_to / min(w, h)
                img   = img.resize(
                    (int(w * scale), int(h * scale)), Image.BILINEAR
                )
            self._cache[p] = np.array(img, dtype=np.uint8)
            if (i + 1) % log_every == 0 or (i + 1) == len(paths):
                logger.info("  cached %d / %d", i + 1, len(paths))

        total_mb = sum(a.nbytes for a in self._cache.values()) / 1e6
        logger.info("ImageCache ready: %.0f MB in RAM", total_mb)

    def get(self, path: Path) -> np.ndarray:
        return self._cache[path]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MBTDataset(Dataset):
    """Paired sensor-window + UWB-synced image dataset.

    Parameters
    ----------
    X_sensor:         (N, T_S, C) float32 sensor windows.
    y:                (N,) int labels (1-indexed, 1-7).
    cow_ids:          (N,) int.
    start_timestamps: (N,) int64 Unix timestamps.
    frame_index:      Pre-built FrameIndex.
    image_cache:      Pre-built ImageCache, or None for disk fallback.
    image_size:       Square crop size in pixels.
    train:            Apply training augmentations.
    """

    def __init__(
        self,
        X_sensor:         np.ndarray,
        y:                np.ndarray,
        cow_ids:          np.ndarray,
        start_timestamps: np.ndarray,
        frame_index:      FrameIndex,
        image_cache:      ImageCache | None = None,
        image_size:       int  = 224,
        train:            bool = True,
    ) -> None:
        assert len(X_sensor) == len(y) == len(cow_ids) == len(start_timestamps)
        self.X_sensor         = X_sensor
        self.y                = y
        self.cow_ids          = cow_ids
        self.start_timestamps = start_timestamps
        self.frame_index      = frame_index
        self.image_cache      = image_cache
        self.transform        = build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        sensor = torch.from_numpy(self.X_sensor[idx])    # (T_S, C)
        label  = int(self.y[idx]) - 1                    # 0-indexed
        cow_id = int(self.cow_ids[idx])
        ts     = int(self.start_timestamps[idx])

        img_path = self.frame_index.nearest(ts)

        if self.image_cache is not None:
            # Fast path: RAM → PIL → transform (no disk I/O)
            arr   = self.image_cache.get(img_path)
            image = self.transform(Image.fromarray(arr))
        else:
            # Slow fallback: disk → PIL → transform
            image = self.transform(Image.open(img_path).convert("RGB"))

        return {
            "sensor":    sensor,
            "image":     image,
            "label":     torch.tensor(label, dtype=torch.long),
            "cow_id":    cow_id,
            "timestamp": ts,
        }
