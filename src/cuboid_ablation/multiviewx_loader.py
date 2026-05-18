"""
src.cuboid_ablation.multiviewx_loader
======================================

Loads all data needed to run the cuboid reconstruction ablation on MultiviewX:

  - Per-camera projection matrices  P = K @ [R | t]   (3×4, metric units in m)
  - Optical centres  C = –R⁻¹ t   (3,) in metres
  - Per-frame pedestrian annotations: positionID → pixel bboxes per camera

MultiviewX coordinate system
-----------------------------
  - World unit: metre
  - Ground plane: z = 0
  - Grid: 640 rows × 1000 cols,  1 cell = 0.025 m
  - World origin is at grid (0,0),  coord_x = grid_x * 0.025,  coord_y = grid_y * 0.025
  - Camera XML: rvec / tvec in metres via OpenCV convention

So P operates in metres.  The reconstructor operates in cm.  We therefore
expose a flag ``to_cm=True`` (default) that scales P and C to centimetres,
matching the interface expected by CuboidReconstructor.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── camera file names ────────────────────────────────────────────────────────
_INTR_FMT = "intr_Camera{}.xml"
_EXTR_FMT = "extr_Camera{}.xml"
NUM_CAMS = 6
IMG_H, IMG_W = 1080, 1920


# ─────────────────────────────────────────────────────────────────────────────
# Calibration loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_intrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (camera_matrix 3×3, dist_coeffs 1×5)."""
    print("Intrinsic Path:", path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return K.astype(np.float64), dist.astype(np.float64)


def _load_extrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (rvec 3, tvec 3) in metres."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat().squeeze().astype(np.float64)
    tvec = fs.getNode("tvec").mat().squeeze().astype(np.float64)
    fs.release()
    return rvec, tvec


def _build_projection(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Return P = K @ [R | t]  (3×4)."""
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack([R, tvec.reshape(3, 1)])  # (3, 4)
    return (K @ Rt).astype(np.float64)


def _optical_centre(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Return optical centre C = –Rᵀ t  (3,) in the same unit as tvec."""
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec.reshape(3, 1)).squeeze()


@dataclass
class MultiviewXCalib:
    """Calibration data for all cameras, optionally converted to cm.

    Attributes
    ----------
    proj_mats  : dict  cam_name → (3, 4) ndarray
    cam_centres: dict  cam_name → (3,)  ndarray
    intrinsics : dict  cam_name → (3, 3) ndarray   (K, original pixels)
    dist_coeffs: dict  cam_name → (1×5) ndarray
    cam_names  : list[str]     ordered list e.g. ["cam_0", ..., "cam_5"]
    scale      : float         1.0 (metres) or 100.0 (centimetres)
    """

    proj_mats: dict[str, np.ndarray]
    cam_centres: dict[str, np.ndarray]
    intrinsics: dict[str, np.ndarray]
    dist_coeffs: dict[str, np.ndarray]
    cam_names: list[str]
    scale: float = 1.0  # metres by default; 100.0 when to_cm=True


def load_calibration(dataset_root: str | Path, to_cm: bool = True) -> MultiviewXCalib:
    """Load all camera calibrations from a MultiviewX dataset tree.

    Parameters
    ----------
    dataset_root : path to the MultiviewX root (contains calibrations/)
    to_cm        : if True, scale P and C so world units = centimetres.
                   MultiviewX native unit is metres.

    Returns
    -------
    MultiviewXCalib
    """
    root = Path(dataset_root)
    intr_dir = root / "calibrations" / "intrinsic"
    extr_dir = root / "calibrations" / "extrinsic"

    scale = 100.0 if to_cm else 1.0

    proj_mats: dict[str, np.ndarray] = {}
    cam_centres: dict[str, np.ndarray] = {}
    intrinsics: dict[str, np.ndarray] = {}
    dist_coeffs: dict[str, np.ndarray] = {}

    for i in range(1, NUM_CAMS + 1):
        cam_name = f"cam_{i - 1}"   # 0-indexed names: cam_0 … cam_5

        K, dist = _load_intrinsic(intr_dir / _INTR_FMT.format(i))
        rvec, tvec = _load_extrinsic(extr_dir / _EXTR_FMT.format(i))

        # Scale: P_cm = K @ [R | t_cm]  where t_cm = tvec * 100
        #   (K stays in pixels; only the translation magnitude changes)
        tvec_scaled = tvec * scale
        P = _build_projection(K, rvec, tvec_scaled)
        C = _optical_centre(rvec, tvec_scaled)

        proj_mats[cam_name] = P
        cam_centres[cam_name] = C
        intrinsics[cam_name] = K
        dist_coeffs[cam_name] = dist

    cam_names = [f"cam_{i}" for i in range(NUM_CAMS)]
    logger.info(
        "Loaded calibration for %d cameras (unit=%s).",
        NUM_CAMS,
        "cm" if to_cm else "m",
    )
    return MultiviewXCalib(
        proj_mats=proj_mats,
        cam_centres=cam_centres,
        intrinsics=intrinsics,
        dist_coeffs=dist_coeffs,
        cam_names=cam_names,
        scale=scale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonAnnotation:
    """Single-pedestrian annotation for one frame.

    Attributes
    ----------
    person_id  : int
    position_id: int   (MultiviewX positionID = grid_x + grid_y * 1000)
    world_x_cm : float  ground-plane x in cm  (= grid_x / 40 * 100)
    world_y_cm : float  ground-plane y in cm  (= grid_y / 40 * 100)
    bboxes     : dict  cam_name → (x1, y1, x2, y2) pixel bbox,
                        or None if the person is not visible in that camera.
    """

    person_id: int
    position_id: int
    world_x_cm: float
    world_y_cm: float
    bboxes: dict[str, Optional[tuple[int, int, int, int]]]


def _positionid_to_worldcoord_cm(position_id: int) -> tuple[float, float]:
    """Convert MultiviewX positionID → world (x, y) in cm.

    positionID = grid_x + grid_y * 1000
    world_coord = grid / 40   (metres)
    → multiply by 100 for cm.
    """
    grid_x = position_id % 1000
    grid_y = position_id // 1000
    return grid_x / 40 * 100.0, grid_y / 40 * 100.0


def load_frame_annotations(
    dataset_root: str | Path,
    frame_idx: int,
    min_cameras_visible: int = 2,
    min_bbox_area: int = 50,
) -> list[PersonAnnotation]:
    """Load all annotated pedestrians for one frame.

    Parameters
    ----------
    dataset_root       : MultiviewX root directory
    frame_idx          : 0-based frame index (files are 0000.json … 0399.json)
    min_cameras_visible: discard persons visible in fewer cameras than this
    min_bbox_area      : discard degenerate bboxes smaller than this in pixels²

    Returns
    -------
    list[PersonAnnotation]
    """
    root = Path(dataset_root)
    ann_path = root / "annotations_positions" / f"{frame_idx:05d}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open() as fh:
        raw = json.load(fh)

    persons: list[PersonAnnotation] = []
    for entry in raw:
        bboxes: dict[str, Optional[tuple[int, int, int, int]]] = {}
        n_visible = 0

        for view in entry["views"]:
            cam_idx = view["viewNum"]   # 0-based in MultiviewX
            cam_name = f"cam_{cam_idx}"
            x1, y1, x2, y2 = view["xmin"], view["ymin"], view["xmax"], view["ymax"]

            # sentinel for invisible: all four –1 or bbox degenerate
            if x1 == -1 and x2 == -1 and y1 == -1 and y2 == -1:
                bboxes[cam_name] = None
                continue
            if x2 <= x1 or y2 <= y1:
                bboxes[cam_name] = None
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < min_bbox_area:
                bboxes[cam_name] = None
                continue

            # Clip to image bounds
            x1 = max(0, min(x1, IMG_W - 1))
            x2 = max(0, min(x2, IMG_W - 1))
            y1 = max(0, min(y1, IMG_H - 1))
            y2 = max(0, min(y2, IMG_H - 1))
            bboxes[cam_name] = (x1, y1, x2, y2)
            n_visible += 1

        if n_visible < min_cameras_visible:
            continue

        wx, wy = _positionid_to_worldcoord_cm(entry["positionID"])
        persons.append(
            PersonAnnotation(
                person_id=entry["personID"],
                position_id=entry["positionID"],
                world_x_cm=wx,
                world_y_cm=wy,
                bboxes=bboxes,
            )
        )

    return persons


def list_frame_indices(dataset_root: str | Path) -> list[int]:
    """Return sorted list of frame indices found in annotations_positions/."""
    ann_dir = Path(dataset_root) / "annotations_positions"
    indices = sorted(
        int(p.stem)
        for p in ann_dir.glob("*.json")
        if p.stem.isdigit()
    )
    return indices
