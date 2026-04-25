"""
MmCows — Data Loaders
=====================

One loader class per sensor modality, plus a multi-modal fusion loader.

Design principles
-----------------
* **Pure DataFrames out.**  Every ``load_*`` method returns a tidy
  ``pd.DataFrame`` so that callers are not locked into any particular
  ML framework.
* **Fail loudly, skip gracefully.**  A missing file for one cow raises a
  warning and skips that cow rather than crashing the whole load.
* **No side-effects.**  Loaders are stateless; they do not cache results
  or write to disk.
* **Single source of truth for paths.**  All path construction is
  delegated to ``mmcows.utils.io_utils``.

Behavior label filtering
--------------------------
By default every loader drops rows where ``behavior == 0`` (unknown).
Pass ``drop_unknown=False`` to retain them.

Column naming conventions
--------------------------
All loaders guarantee the following columns in their output:

    ``timestamp``   — int64 Unix timestamp
    ``cow_id``      — int  (1-based, matching folder index)
    ``behavior``   — int  (7-class label, 0 = unknown)

Modality-specific feature columns are documented in each class's
``load()`` docstring.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mmcows.utils.constants import (
    ALL_COW_IDS,
    ANNOTATED_DATE,
    CAMERA_IDS,
    SENSOR_COW_IDS,
)
from mmcows.utils.io_utils import (
    behavior_label_path,
    cow_name,
    image_dir_path,
    sensor_path,
    tag_name,
    visual_location_path,
)
from mmcows.utils.time_utils import floor_to_second

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_csv_safe(path: Path, label: str = "") -> pd.DataFrame | None:
    """Read a CSV, returning ``None`` (with a warning) if the file is missing."""
    if not path.exists():
        warnings.warn(f"File not found — skipping {label or path}", stacklevel=3)
        return None
    return pd.read_csv(path)


def _load_behavior_labels(
    sensor_data_dir: str | Path,
    cow_id: int,
    date: str,
    *,
    drop_unknown: bool = True,
) -> pd.DataFrame | None:
    """Load the 1-Hz behavior label CSV for one cow on one date."""
    path = behavior_label_path(sensor_data_dir, cow_name=cow_name(cow_id), date=date)
    df = _read_csv_safe(path, label=f"behavior labels cow {cow_id} date {date}")
    if df is None:
        return None

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["cow_id"] = cow_id

    if drop_unknown:
        df = df[df["behavior"] != 0]

    return df[["timestamp", "cow_id", "behavior"]].reset_index(drop=True)


def _align_labels_to_uwb(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Inner-join feature rows to label timestamps, dropping unmatched rows.

    UWB and its dependent modalities are sampled every 15 s; labels are
    1 Hz.  We keep only the UWB timestamps that have a matching label.
    """
    valid_ts = set(label_df[timestamp_col].astype(np.int64))
    mask = feature_df[timestamp_col].astype(np.int64).isin(valid_ts)
    feature_df = feature_df.loc[mask].copy()
    return feature_df.merge(
        label_df[[timestamp_col, "behavior"]],
        on=timestamp_col,
        how="inner",
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# IMU Loader
# ─────────────────────────────────────────────────────────────────────────────


class IMULoader:
    """Load and align IMU (acceleration + head-direction) data with behavior labels.

    The IMU is recorded at **10 Hz**.  Labels are at **1 Hz**.  This loader
    bins IMU timestamps to the nearest second, then aligns with labels.

    Output columns
    --------------
    ``timestamp``, ``cow_id``,
    ``accel_x_mps2``, ``accel_y_mps2``, ``accel_z_mps2``,
    ``relative_angle``,
    ``behavior``
    """

    def __init__(
        self,
        sensor_data_dir: str | Path,
        *,
        drop_unknown: bool = True,
    ) -> None:
        self.sensor_data_dir = Path(sensor_data_dir)
        self.drop_unknown = drop_unknown

    def load(
        self,
        cow_ids: Sequence[int] = SENSOR_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load IMU data for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers (must be in 1–10 for sensor data).
        date:
            Date string in ``MMDD`` format (default ``"0725"``).

        Returns
        -------
        A tidy ``pd.DataFrame`` with one row per 1-Hz sample × cow.
        """
        frames: list[pd.DataFrame] = []

        for cid in cow_ids:
            tname = tag_name(cid)

            # --- acceleration (10 Hz) ---
            accel_path = sensor_path(
                self.sensor_data_dir,
                sub="main_data/immu",
                tag_or_cow=tname,
                date=date,
            )
            accel_df = _read_csv_safe(accel_path, label=f"IMU accel {tname}")
            if accel_df is None:
                continue

            # Bin 10-Hz timestamps to 1-Hz integers
            accel_df["timestamp"] = floor_to_second(accel_df["timestamp"])
            accel_df = accel_df[
                ["timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]
            ]

            # --- head direction (10 Hz) → downsample to 1 Hz by mean ---
            head_path = sensor_path(
                self.sensor_data_dir,
                sub="sub_data/head_direction",
                tag_or_cow=tname,
                date=date,
            )
            head_df = _read_csv_safe(head_path, label=f"head direction {tname}")
            if head_df is None:
                # head direction is optional; proceed without it
                accel_df["relative_angle"] = np.nan
                merged = accel_df.copy()
            else:
                head_df["timestamp"] = floor_to_second(head_df["timestamp"])
                head_1hz = (
                    head_df.groupby("timestamp")[["relative_angle"]]
                    .mean()
                    .reset_index()
                )
                merged = accel_df.merge(head_1hz, on="timestamp", how="left")

            # Downsample to 1-Hz by averaging within each integer second
            merged = (
                merged.groupby("timestamp")
                .mean(numeric_only=True)
                .reset_index()
            )

            # --- behavior labels ---
            label_df = _load_behavior_labels(
                self.sensor_data_dir, cid, date, drop_unknown=self.drop_unknown
            )
            if label_df is None:
                continue

            # Align
            merged = _align_labels_to_uwb(merged, label_df)
            merged["cow_id"] = cid
            merged = merged.ffill()
            frames.append(merged)

        if not frames:
            logger.warning("IMULoader: no data loaded for cow_ids=%s date=%s", cow_ids, date)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["timestamp"] = result["timestamp"].astype(np.int64)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# UWB Loader
# ─────────────────────────────────────────────────────────────────────────────


class UWBLoader:
    """Load UWB 3-D neck-location data aligned with behavior labels.

    UWB is sampled every **15 seconds**.  Labels are at **1 Hz**.  We keep
    only UWB timestamps that have a matching behavior label.

    Output columns
    --------------
    ``timestamp``, ``cow_id``,
    ``coord_x_cm``, ``coord_y_cm``, ``coord_z_cm``,
    ``behavior``
    """

    def __init__(
        self,
        sensor_data_dir: str | Path,
        *,
        drop_unknown: bool = True,
    ) -> None:
        self.sensor_data_dir = Path(sensor_data_dir)
        self.drop_unknown = drop_unknown

    def load(
        self,
        cow_ids: Sequence[int] = SENSOR_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load UWB data for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers.
        date:
            Date string in ``MMDD`` format (default ``"0725"``).

        Returns
        -------
        A tidy ``pd.DataFrame`` with one row per UWB sample × cow.
        """
        frames: list[pd.DataFrame] = []

        for cid in cow_ids:
            tname = tag_name(cid)

            uwb_path = sensor_path(
                self.sensor_data_dir,
                sub="main_data/uwb",
                tag_or_cow=tname,
                date=date,
            )
            uwb_df = _read_csv_safe(uwb_path, label=f"UWB {tname}")
            if uwb_df is None:
                continue

            uwb_df["timestamp"] = uwb_df["timestamp"].astype(np.int64)
            uwb_df = uwb_df.drop(columns=["datetime"], errors="ignore")

            label_df = _load_behavior_labels(
                self.sensor_data_dir, cid, date, drop_unknown=self.drop_unknown
            )
            if label_df is None:
                continue

            merged = _align_labels_to_uwb(uwb_df, label_df)
            merged["cow_id"] = cid
            merged = merged.dropna()
            frames.append(merged)

        if not frames:
            logger.warning("UWBLoader: no data loaded for cow_ids=%s date=%s", cow_ids, date)
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["timestamp"] = result["timestamp"].astype(np.int64)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# UWB + Head-Direction Loader
# ─────────────────────────────────────────────────────────────────────────────


class UWBHeadDirectionLoader:
    """Load UWB locations fused with downsampled head-direction data.

    Head direction is recorded at 10 Hz and is downsampled to 1 Hz
    (by within-second mean), then further to 1/15 Hz by alignment with
    UWB timestamps.

    Output columns
    --------------
    ``timestamp``, ``cow_id``,
    ``coord_x_cm``, ``coord_y_cm``, ``coord_z_cm``,
    ``roll``, ``pitch``, ``yaw``   (head orientation, degrees),
    ``behavior``

    Note: ``relative_angle`` and ``accel_norm`` columns from the raw
    head-direction file are dropped (not useful for behavior classification).
    """

    _DROP_COLS = {"datetime", "accel_norm", "relative_angle"}

    def __init__(
        self,
        sensor_data_dir: str | Path,
        *,
        drop_unknown: bool = True,
    ) -> None:
        self.sensor_data_dir = Path(sensor_data_dir)
        self.drop_unknown = drop_unknown

    def load(
        self,
        cow_ids: Sequence[int] = SENSOR_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load UWB + head-direction data for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers.
        date:
            Date string in ``MMDD`` format.

        Returns
        -------
        A tidy ``pd.DataFrame`` with one row per UWB sample × cow.
        """
        frames: list[pd.DataFrame] = []

        for cid in cow_ids:
            tname = tag_name(cid)

            # --- UWB ---
            uwb_path = sensor_path(
                self.sensor_data_dir, sub="main_data/uwb", tag_or_cow=tname, date=date
            )
            uwb_df = _read_csv_safe(uwb_path, label=f"UWB {tname}")
            if uwb_df is None:
                continue
            uwb_df["timestamp"] = uwb_df["timestamp"].astype(np.int64)
            uwb_df = uwb_df.drop(columns=list(self._DROP_COLS), errors="ignore")

            # --- head direction ---
            head_path = sensor_path(
                self.sensor_data_dir,
                sub="sub_data/head_direction",
                tag_or_cow=tname,
                date=date,
            )
            head_df = _read_csv_safe(head_path, label=f"head direction {tname}")
            if head_df is not None:
                # Downsample: 10 Hz → 1 Hz
                head_df["timestamp"] = floor_to_second(head_df["timestamp"])
                head_1hz = (
                    head_df.drop(columns=list(self._DROP_COLS), errors="ignore")
                    .groupby("timestamp")
                    .mean(numeric_only=True)
                    .reset_index()
                )
                # Align to UWB grid (1/15 Hz)
                head_aligned = head_1hz[
                    head_1hz["timestamp"].isin(uwb_df["timestamp"])
                ].copy()
                merged = uwb_df.merge(head_aligned, on="timestamp", how="left")
            else:
                merged = uwb_df.copy()

            # --- behavior labels ---
            label_df = _load_behavior_labels(
                self.sensor_data_dir, cid, date, drop_unknown=self.drop_unknown
            )
            if label_df is None:
                continue

            merged = _align_labels_to_uwb(merged, label_df)
            merged["cow_id"] = cid
            merged = merged.dropna()
            frames.append(merged)

        if not frames:
            logger.warning(
                "UWBHeadDirectionLoader: no data loaded for cow_ids=%s date=%s",
                cow_ids, date,
            )
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["timestamp"] = result["timestamp"].astype(np.int64)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal Sensor Loader  (UWB + Head-Direction + Ankle)
# ─────────────────────────────────────────────────────────────────────────────


class MultimodalSensorLoader:
    """Load UWB + head-direction + ankle data aligned with behavior labels.

    Ankle data is recorded at 1-minute intervals.  It is upsampled to the
    15-second UWB grid via nearest-neighbour ``merge_asof``.

    Output columns
    --------------
    All columns from :class:`UWBHeadDirectionLoader`, plus ankle columns
    (typically a ``lying`` binary flag and/or raw ankle acceleration stats).
    """

    _DROP_COLS = {"datetime", "accel_norm", "relative_angle"}

    def __init__(
        self,
        sensor_data_dir: str | Path,
        *,
        drop_unknown: bool = True,
    ) -> None:
        self.sensor_data_dir = Path(sensor_data_dir)
        self.drop_unknown = drop_unknown
        self._uwb_hd_loader = UWBHeadDirectionLoader(
            sensor_data_dir, drop_unknown=drop_unknown
        )

    def load(
        self,
        cow_ids: Sequence[int] = SENSOR_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load full multimodal sensor data for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers.
        date:
            Date string in ``MMDD`` format.

        Returns
        -------
        A tidy ``pd.DataFrame`` with one row per UWB sample × cow.
        """
        # Start from UWB + head direction
        base_df = self._uwb_hd_loader.load(cow_ids=cow_ids, date=date)
        if base_df.empty:
            return base_df

        augmented_frames: list[pd.DataFrame] = []

        for cid in cow_ids:
            cow_df = base_df[base_df["cow_id"] == cid].copy()
            if cow_df.empty:
                continue

            # --- ankle ---
            cname = cow_name(cid)
            ankle_path = sensor_path(
                self.sensor_data_dir,
                sub="main_data/ankle",
                tag_or_cow=cname,
                date=date,
            )
            ankle_df = _read_csv_safe(ankle_path, label=f"ankle {cname}")
            if ankle_df is not None:
                ankle_df = (
                    ankle_df
                    .drop(columns=list(self._DROP_COLS), errors="ignore")
                    .astype({"timestamp": np.int64})
                    .sort_values("timestamp")
                )
                cow_df = cow_df.sort_values("timestamp")
                cow_df = pd.merge_asof(
                    cow_df,
                    ankle_df,
                    on="timestamp",
                    direction="nearest",
                    tolerance=60,       # 60-second tolerance for ankle data
                    suffixes=("", "_ankle"),
                )

            augmented_frames.append(cow_df)

        if not augmented_frames:
            return pd.DataFrame()

        result = pd.concat(augmented_frames, ignore_index=True).dropna()
        result["timestamp"] = result["timestamp"].astype(np.int64)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Visual Location Loader
# ─────────────────────────────────────────────────────────────────────────────


class VisualLocationLoader:
    """Load visual 3-D body locations (derived from multi-view bounding boxes).

    These locations are available for **all 16 cows** on the annotated day
    (July 25th), unlike sensor data which covers only cows 1–10.

    Output columns
    --------------
    ``timestamp``, ``cow_id``,
    ``loc_x_cm``, ``loc_y_cm``, ``loc_z_cm``
    """

    def __init__(self, visual_data_dir: str | Path) -> None:
        self.visual_data_dir = Path(visual_data_dir)

    def load(
        self,
        cow_ids: Sequence[int] = ALL_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load visual-location data for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers (1–16 supported).
        date:
            Date string in ``MMDD`` format.

        Returns
        -------
        A tidy ``pd.DataFrame`` with one row per 15-second sample × cow.
        """
        frames: list[pd.DataFrame] = []

        for cid in cow_ids:
            cname = cow_name(cid)
            path = visual_location_path(
                self.visual_data_dir, cow_name=cname, date=date
            )
            df = _read_csv_safe(path, label=f"visual location {cname}")
            if df is None:
                continue

            df["timestamp"] = df["timestamp"].astype(np.int64)
            df["cow_id"] = cid
            frames.append(df)

        if not frames:
            logger.warning(
                "VisualLocationLoader: no data loaded for cow_ids=%s date=%s",
                cow_ids, date,
            )
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Image + Visual-Location Loader
# ─────────────────────────────────────────────────────────────────────────────


class ImageLocationLoader:
    """Pair camera-image paths with their corresponding 3-D visual locations.

    This is the foundation for any RGB-based or UWB+RGB fusion pipeline.
    The returned DataFrame contains image file paths (strings) so that a
    PyTorch ``Dataset`` can open images on demand without loading everything
    into memory.

    Output columns
    --------------
    ``timestamp``, ``cow_id``, ``camera``,
    ``image_path`` (absolute path string),
    ``loc_x_cm``, ``loc_y_cm``, ``loc_z_cm``
    """

    def __init__(
        self,
        visual_data_dir: str | Path,
        *,
        cameras: Sequence[str] = CAMERA_IDS,
        timestamp_tolerance_s: int = 1,
    ) -> None:
        self.visual_data_dir = Path(visual_data_dir)
        self.cameras = list(cameras)
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self._vis_loader = VisualLocationLoader(visual_data_dir)

    def load(
        self,
        cow_ids: Sequence[int] = ALL_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load image–location pairs for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers.
        date:
            Date string in ``MMDD`` format.

        Returns
        -------
        A ``pd.DataFrame`` where each row links one image file to one
        cow's 3-D body location.
        """
        location_df = self._vis_loader.load(cow_ids=cow_ids, date=date)
        if location_df.empty:
            return pd.DataFrame()

        records: list[dict] = []

        for camera in self.cameras:
            img_dir = image_dir_path(self.visual_data_dir, date=date, camera=camera)
            if not img_dir.is_dir():
                warnings.warn(f"Image directory not found: {img_dir}", stacklevel=2)
                continue

            image_files = sorted(img_dir.glob("*.jpg"))
            if not image_files:
                warnings.warn(f"No JPEG images in {img_dir}", stacklevel=2)
                continue

            for img_file in image_files:
                try:
                    img_timestamp = int(img_file.stem.split("_")[0])
                except ValueError:
                    logger.debug("Could not parse timestamp from filename: %s", img_file)
                    continue

                # Match to visual locations within tolerance
                tol = self.timestamp_tolerance_s
                matches = location_df[
                    (location_df["timestamp"] - img_timestamp).abs() <= tol
                ]

                for _, row in matches.iterrows():
                    records.append(
                        {
                            "timestamp":  img_timestamp,
                            "cow_id":     int(row["cow_id"]),
                            "camera":     camera,
                            "image_path": str(img_file),
                            "loc_x_cm":   row.get("loc_x_cm", np.nan),
                            "loc_y_cm":   row.get("loc_y_cm", np.nan),
                            "loc_z_cm":   row.get("loc_z_cm", np.nan),
                        }
                    )

        if not records:
            logger.warning(
                "ImageLocationLoader: no records assembled for cow_ids=%s date=%s",
                cow_ids, date,
            )
            return pd.DataFrame()

        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# UWB + RGB Fusion Loader
# ─────────────────────────────────────────────────────────────────────────────


class UWBRGBFusionLoader:
    """Synchronise UWB location data with camera images and visual locations.

    Both UWB and the camera images are sampled every 15 seconds, so
    temporal alignment is straightforward.  ``merge_asof`` with a
    configurable tolerance handles any residual jitter.

    Output columns
    --------------
    ``timestamp``, ``cow_id``, ``camera``,
    ``coord_x_cm``, ``coord_y_cm``, ``coord_z_cm``  (UWB neck),
    ``image_path``,
    ``loc_x_cm``, ``loc_y_cm``, ``loc_z_cm``        (visual body)
    """

    def __init__(
        self,
        sensor_data_dir: str | Path,
        visual_data_dir: str | Path,
        *,
        cameras: Sequence[str] = CAMERA_IDS,
        time_tolerance_s: int = 15,
    ) -> None:
        self.sensor_data_dir = Path(sensor_data_dir)
        self.visual_data_dir = Path(visual_data_dir)
        self._uwb_loader = UWBLoader(sensor_data_dir, drop_unknown=False)
        self._img_loader = ImageLocationLoader(
            visual_data_dir, cameras=cameras, timestamp_tolerance_s=1
        )
        self.time_tolerance_s = time_tolerance_s

    def load(
        self,
        cow_ids: Sequence[int] = SENSOR_COW_IDS,
        date: str = ANNOTATED_DATE,
    ) -> pd.DataFrame:
        """Load synchronised UWB + RGB data for the given cow IDs and date.

        Parameters
        ----------
        cow_ids:
            Iterable of 1-based cow integers (must have sensor data, i.e. 1–10).
        date:
            Date string in ``MMDD`` format.

        Returns
        -------
        A ``pd.DataFrame`` with one row per (cow, timestamp, camera) triple
        where both a UWB reading and an image exist within the tolerance window.
        """
        uwb_df = self._uwb_loader.load(cow_ids=cow_ids, date=date)
        img_df = self._img_loader.load(cow_ids=list(cow_ids), date=date)

        if uwb_df.empty or img_df.empty:
            logger.warning("UWBRGBFusionLoader: one or both sources are empty.")
            return pd.DataFrame()

        merged_parts: list[pd.DataFrame] = []

        for cid in cow_ids:
            uwb_cow = uwb_df[uwb_df["cow_id"] == cid].sort_values("timestamp")
            img_cow = img_df[img_df["cow_id"] == cid].sort_values("timestamp")

            if uwb_cow.empty or img_cow.empty:
                continue

            uwb_cow = uwb_cow.astype({"timestamp": np.int64})
            img_cow = img_cow.astype({"timestamp": np.int64})

            merged = pd.merge_asof(
                uwb_cow,
                img_cow.drop(columns=["cow_id"]),
                on="timestamp",
                direction="nearest",
                tolerance=self.time_tolerance_s,
                suffixes=("_uwb", "_vis"),
            )
            merged_parts.append(merged)

        if not merged_parts:
            return pd.DataFrame()

        result = (
            pd.concat(merged_parts, ignore_index=True)
            .dropna(subset=["image_path"])
        )
        result["timestamp"] = result["timestamp"].astype(np.int64)
        return result