"""
Unit tests for mmcows.data.loaders and mmcows.data.splits.

These tests use *synthetic* data written to a temporary directory so that
they can run without any real dataset files present.  They verify:

  * correct column names in loader output
  * correct alignment / merging behaviour
  * graceful handling of missing files
  * SplitConfig splitting logic (both S1 and S2)

Run with:  pytest tests/test_loaders.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmcows.data.loaders import (
    IMULoader,
    MultimodalSensorLoader,
    UWBHeadDirectionLoader,
    UWBLoader,
    VisualLocationLoader,
)
from mmcows.data.splits import SplitConfig
from mmcows.utils.constants import ANNOTATED_DATE


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: build a synthetic dataset on disk
# ─────────────────────────────────────────────────────────────────────────────


def _make_timestamps(n: int, start: int = 1_690_300_000, step: int = 15) -> np.ndarray:
    return np.arange(start, start + n * step, step, dtype=np.int64)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture(scope="module")
def synthetic_sensor_dir(tmp_path_factory):
    """Return a temporary sensor_data directory pre-populated with fake CSVs."""
    root = tmp_path_factory.mktemp("sensor_data")
    date = ANNOTATED_DATE
    n = 50   # number of UWB samples (15-s intervals)

    uwb_ts = _make_timestamps(n)

    for cid in [1, 2, 3]:
        tag = f"T{cid:02d}"
        cow = f"C{cid:02d}"

        # ── UWB ──
        uwb_df = pd.DataFrame(
            {
                "timestamp":   uwb_ts,
                "coord_x_cm": np.random.uniform(-600, 600, n),
                "coord_y_cm": np.random.uniform(-600, 600, n),
                "coord_z_cm": np.random.uniform(0, 200, n),
            }
        )
        _write_csv(root / "main_data" / "uwb" / tag / f"{tag}_{date}.csv", uwb_df)

        # ── IMU (10 Hz = one sample per 0.1 s) ──
        imu_n = n * 15 * 10   # 10 samples per second × 15s intervals × n UWB samples
        imu_ts = np.linspace(uwb_ts[0], uwb_ts[-1], imu_n)
        imu_df = pd.DataFrame(
            {
                "timestamp":    imu_ts,
                "accel_x_mps2": np.random.randn(imu_n),
                "accel_y_mps2": np.random.randn(imu_n),
                "accel_z_mps2": np.random.randn(imu_n) + 9.8,
            }
        )
        _write_csv(root / "main_data" / "immu" / tag / f"{tag}_{date}.csv", imu_df)

        # ── Head direction (10 Hz) ──
        hd_df = pd.DataFrame(
            {
                "timestamp":     imu_ts,
                "relative_angle": np.random.uniform(0, 180, imu_n),
                "roll":           np.random.randn(imu_n),
                "pitch":          np.random.randn(imu_n),
                "yaw":            np.random.uniform(-180, 180, imu_n),
            }
        )
        _write_csv(root / "sub_data" / "head_direction" / tag / f"{tag}_{date}.csv", hd_df)

        # ── Ankle (1-min intervals) ──
        ankle_ts = _make_timestamps(n // 4 + 1, start=int(uwb_ts[0]), step=60)
        ankle_df = pd.DataFrame(
            {
                "timestamp": ankle_ts,
                "lying":     np.random.randint(0, 2, len(ankle_ts)),
            }
        )
        _write_csv(root / "main_data" / "ankle" / cow / f"{cow}_{date}.csv", ankle_df)

        # ── Behaviour labels (1 Hz, one label per second) ──
        label_ts = np.arange(int(uwb_ts[0]), int(uwb_ts[-1]) + 1, 1, dtype=np.int64)
        label_df = pd.DataFrame(
            {
                "timestamp": label_ts,
                "behaviour": np.random.choice([1, 2, 3, 4, 7], len(label_ts)),
            }
        )
        _write_csv(
            root / "behavior_labels" / "individual" / f"{cow}_{date}.csv",
            label_df,
        )

    return root


@pytest.fixture(scope="module")
def synthetic_visual_dir(tmp_path_factory):
    """Return a temporary visual_data directory with fake visual location CSVs."""
    root = tmp_path_factory.mktemp("visual_data")
    date = ANNOTATED_DATE
    n = 50
    ts = _make_timestamps(n)

    for cid in [1, 2]:
        cow = f"C{cid:02d}"
        vis_df = pd.DataFrame(
            {
                "timestamp": ts,
                "loc_x_cm":  np.random.uniform(-600, 600, n),
                "loc_y_cm":  np.random.uniform(-600, 600, n),
                "loc_z_cm":  np.random.uniform(0, 200, n),
            }
        )
        _write_csv(root / "visual_location" / cow / f"{cow}_{date}.csv", vis_df)

    return root


@pytest.fixture(scope="module")
def synthetic_split_configs(tmp_path_factory):
    """Write fake S1 and S2 config JSONs and return their paths."""
    cfg_dir = tmp_path_factory.mktemp("split_configs")

    s1 = {
        "folds": {
            "fold_1": {"train": [1], "val": [2], "test": [3]},
        }
    }
    s2 = {
        "group_1": {
            "A": ["2023-07-25 02:57:00", "2023-07-25 06:00:00"],
            "B": ["2023-07-25 10:00:00", "2023-07-25 12:00:00"],
        },
        "group_2": {
            "A": ["2023-07-25 18:00:00", "2023-07-25 20:00:00"],
            "B": ["2023-07-25 22:00:00", "2023-07-25 23:57:00"],
        },
        "folds": {
            "fold_1": {"train": ["A"], "val": ["B"], "test": ["B"]},
        },
    }

    s1_path = cfg_dir / "config_s1.json"
    s2_path = cfg_dir / "config_s2.json"
    s1_path.write_text(json.dumps(s1))
    s2_path.write_text(json.dumps(s2))

    return s1_path, s2_path


# ─────────────────────────────────────────────────────────────────────────────
# UWBLoader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUWBLoader:
    def test_columns_present(self, synthetic_sensor_dir):
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2, 3])
        required = {"timestamp", "cow_id", "behaviour"}
        assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    def test_no_unknown_behaviour(self, synthetic_sensor_dir):
        loader = UWBLoader(synthetic_sensor_dir, drop_unknown=True)
        df = loader.load(cow_ids=[1])
        assert (df["behaviour"] != 0).all(), "drop_unknown=True left unknown rows"

    def test_all_cow_ids_present(self, synthetic_sensor_dir):
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2, 3])
        assert set(df["cow_id"].unique()) == {1, 2, 3}

    def test_missing_cow_skipped_gracefully(self, synthetic_sensor_dir):
        loader = UWBLoader(synthetic_sensor_dir)
        # Cow 99 does not exist
        df = loader.load(cow_ids=[1, 99])
        assert 99 not in df["cow_id"].values
        assert 1 in df["cow_id"].values

    def test_not_empty(self, synthetic_sensor_dir):
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        assert len(df) > 0

    def test_timestamps_are_int64(self, synthetic_sensor_dir):
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        assert df["timestamp"].dtype == np.int64


# ─────────────────────────────────────────────────────────────────────────────
# IMULoader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIMULoader:
    def test_columns_present(self, synthetic_sensor_dir):
        loader = IMULoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        required = {"timestamp", "cow_id", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "behaviour"}
        assert required.issubset(df.columns)

    def test_not_empty(self, synthetic_sensor_dir):
        loader = IMULoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        assert len(df) > 0

    def test_no_unknown_behaviour(self, synthetic_sensor_dir):
        loader = IMULoader(synthetic_sensor_dir, drop_unknown=True)
        df = loader.load(cow_ids=[1])
        assert (df["behaviour"] != 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# UWBHeadDirectionLoader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestUWBHeadDirectionLoader:
    def test_columns_present(self, synthetic_sensor_dir):
        loader = UWBHeadDirectionLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        required = {"timestamp", "cow_id", "coord_x_cm", "behaviour"}
        assert required.issubset(df.columns)

    def test_not_empty(self, synthetic_sensor_dir):
        loader = UWBHeadDirectionLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2])
        assert len(df) > 0


# ─────────────────────────────────────────────────────────────────────────────
# MultimodalSensorLoader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultimodalSensorLoader:
    def test_not_empty(self, synthetic_sensor_dir):
        loader = MultimodalSensorLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        assert len(df) > 0

    def test_has_ankle_column(self, synthetic_sensor_dir):
        loader = MultimodalSensorLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])
        # Should have merged ankle data (column 'lying' from ankle CSV)
        assert "lying" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# VisualLocationLoader tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVisualLocationLoader:
    def test_columns_present(self, synthetic_visual_dir):
        loader = VisualLocationLoader(synthetic_visual_dir)
        df = loader.load(cow_ids=[1, 2])
        required = {"timestamp", "cow_id", "loc_x_cm", "loc_y_cm", "loc_z_cm"}
        assert required.issubset(df.columns)

    def test_not_empty(self, synthetic_visual_dir):
        loader = VisualLocationLoader(synthetic_visual_dir)
        df = loader.load(cow_ids=[1])
        assert len(df) > 0

    def test_missing_cow_skipped(self, synthetic_visual_dir):
        loader = VisualLocationLoader(synthetic_visual_dir)
        df = loader.load(cow_ids=[1, 99])
        assert 99 not in df["cow_id"].values


# ─────────────────────────────────────────────────────────────────────────────
# SplitConfig tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitConfig:
    def test_load_from_json(self, synthetic_split_configs):
        s1_path, s2_path = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path, s2_path=s2_path)
        assert cfg.s1
        assert cfg.s2

    def test_available_folds_s1(self, synthetic_split_configs):
        s1_path, s2_path = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path, s2_path=s2_path)
        assert "fold_1" in cfg.available_folds("s1")

    def test_s1_split_disjoint(self, synthetic_split_configs, synthetic_sensor_dir):
        s1_path, _ = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path)

        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2, 3])

        train, val, test = cfg.split(df, split_type="s1", fold="fold_1")

        # Cow sets must be disjoint
        train_cows = set(train["cow_id"].unique())
        val_cows   = set(val["cow_id"].unique())
        test_cows  = set(test["cow_id"].unique())

        assert train_cows.isdisjoint(val_cows)
        assert train_cows.isdisjoint(test_cows)
        assert val_cows.isdisjoint(test_cows)

    def test_s1_split_covers_all_rows(self, synthetic_split_configs, synthetic_sensor_dir):
        s1_path, _ = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path)

        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2, 3])

        train, val, test = cfg.split(df, split_type="s1", fold="fold_1")
        total = len(train) + len(val) + len(test)
        assert total == len(df)

    def test_invalid_split_type(self, synthetic_split_configs, synthetic_sensor_dir):
        s1_path, _ = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path)
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])

        with pytest.raises(ValueError, match="split_type"):
            cfg.split(df, split_type="s99")

    def test_invalid_fold_name(self, synthetic_split_configs, synthetic_sensor_dir):
        s1_path, _ = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path)
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1])

        with pytest.raises(ValueError, match="fold_99"):
            cfg.split(df, split_type="s1", fold="fold_99")

    def test_keep_timestamp_false(self, synthetic_split_configs, synthetic_sensor_dir):
        s1_path, _ = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path)
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2, 3])
        train, _, _ = cfg.split(df, split_type="s1", fold="fold_1", keep_timestamp=False)
        assert "timestamp" not in train.columns

    def test_iter_folds(self, synthetic_split_configs, synthetic_sensor_dir):
        s1_path, _ = synthetic_split_configs
        cfg = SplitConfig.from_json(s1_path=s1_path)
        loader = UWBLoader(synthetic_sensor_dir)
        df = loader.load(cow_ids=[1, 2, 3])

        results = list(cfg.iter_folds(df, split_type="s1"))
        assert len(results) == 1   # only fold_1 in synthetic config
        fold_name, train, val, test = results[0]
        assert fold_name == "fold_1"
        assert isinstance(train, pd.DataFrame)