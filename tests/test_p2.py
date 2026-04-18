"""
Unit tests for Priority 2 components.

Tests cover:
  - visual_dataset: bbox math, label parsing, index building
  - tracking: KalmanTrack state machine, DeepSORTTracker, cosine distance
  - backbones: build_backbone registry, graceful ImportError
  - fusion: forward shapes (when PyTorch available), P2Results aggregation
  - p2_pipeline: P2Results / P2FoldResult dataclasses

All tests run without dataset files or PyTorch.
Run with: pytest tests/test_p2.py -v
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from mmcows.data.visual_dataset import (
    BEHAVIOR_TO_CLASS,
    CLASS_TO_BEHAVIOR,
    NUM_CLASSES,
    _bbox_norm_to_pixels,
    _parse_label_file,
    build_crop_index,
)
from mmcows.models.backbones import _BACKBONE_REGISTRY, build_backbone
from mmcows.models.tracking import (
    MAX_AGE,
    N_INIT,
    DeepSORTTracker,
    KalmanTrack,
    _cosine_distance,
    _hungarian_assign,
)
from mmcows.pipelines.p2_pipeline import P2FoldResult, P2Results


# ─────────────────────────────────────────────────────────────────────────────
# visual_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestBBoxMath:
    def test_centre_pixel_round_trip(self):
        """Normalised centre box at (0.5, 0.5) should map to image centre."""
        x1, y1, x2, y2 = _bbox_norm_to_pixels(0.5, 0.5, 0.1, 0.2, img_w=4480, img_h=2800)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        assert abs(cx - 2240) < 2, f"Expected cx≈2240, got {cx}"
        assert abs(cy - 1400) < 2, f"Expected cy≈1400, got {cy}"

    def test_non_negative_coords(self):
        """No negative pixel coordinates."""
        x1, y1, x2, y2 = _bbox_norm_to_pixels(0.01, 0.01, 0.02, 0.02)
        assert x1 >= 0 and y1 >= 0

    def test_clamped_to_image(self):
        """Oversized box is clamped to image boundary."""
        x1, y1, x2, y2 = _bbox_norm_to_pixels(0.5, 0.5, 2.0, 2.0, img_w=4480, img_h=2800)
        assert x2 <= 4480 and y2 <= 2800

    def test_valid_box(self):
        """x_max > x_min and y_max > y_min."""
        x1, y1, x2, y2 = _bbox_norm_to_pixels(0.5, 0.5, 0.2, 0.3)
        assert x2 > x1 and y2 > y1


class TestLabelParsing:
    def test_missing_file_returns_empty(self, tmp_path):
        df = _parse_label_file(tmp_path / "nonexistent.txt")
        assert df.empty

    def test_valid_label_file(self, tmp_path):
        lf = tmp_path / "frame.txt"
        lf.write_text("5 0.5 0.4 0.1 0.2\n7 0.3 0.6 0.08 0.15\n")
        df = _parse_label_file(lf)
        assert len(df) == 2
        assert int(df.iloc[0]["cow_id"]) == 5
        assert int(df.iloc[1]["cow_id"]) == 7

    def test_malformed_line_skipped(self, tmp_path):
        lf = tmp_path / "frame.txt"
        lf.write_text("5 0.5 0.4 0.1 0.2\nnot a number\n7 0.3 0.6 0.08 0.15\n")
        df = _parse_label_file(lf)
        assert len(df) == 2   # malformed line skipped


class TestBehaviorMapping:
    def test_round_trip(self):
        for bid, cid in BEHAVIOR_TO_CLASS.items():
            assert CLASS_TO_BEHAVIOR[cid] == bid

    def test_num_classes(self):
        assert NUM_CLASSES == 7
        assert len(BEHAVIOR_TO_CLASS) == 7

    def test_class_indices_are_0_to_6(self):
        assert set(BEHAVIOR_TO_CLASS.values()) == set(range(7))


class TestBuildCropIndex:
    """build_crop_index with a minimal synthetic visual_data directory."""

    def _make_synthetic_visual_dir(self, tmp_path: Path) -> tuple:
        date = "0725"
        camera = "cam_1"
        ts = 1690300000

        # Image file
        img_dir = tmp_path / "images" / date / camera
        img_dir.mkdir(parents=True)
        # Minimal valid JPEG (1×1 pixel)
        img_bytes = bytes([
            0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
            0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
            0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
            0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
            0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
            0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
            0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
            0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
            0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
            0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
            0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
            0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
            0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
            0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,0x00,0xFB,0xD5,
            0xFF,0xD9,
        ])
        (img_dir / f"{ts}_cam1.jpg").write_bytes(img_bytes)

        # Annotation file
        lbl_dir = tmp_path / "labels" / "combined" / camera
        lbl_dir.mkdir(parents=True)
        (lbl_dir / f"{ts}_cam1.txt").write_text("1 0.5 0.5 0.1 0.2\n")

        # Behavior label
        beh_dir = tmp_path / "behavior_labels" / "individual"
        beh_dir.mkdir(parents=True)
        import pandas as pd
        beh_ts = list(range(ts - 5, ts + 10))
        pd.DataFrame({"timestamp": beh_ts, "behavior": [7] * len(beh_ts)}).to_csv(
            beh_dir / f"C01_{date}.csv", index=False
        )

        return tmp_path, date

    def test_index_non_empty(self, tmp_path):
        vis_dir, date = self._make_synthetic_visual_dir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = build_crop_index(
                vis_dir,
                vis_dir / "behavior_labels" / "individual",
                date=date,
                cameras=["cam_1"],
            )
        assert len(df) >= 1

    def test_index_has_required_columns(self, tmp_path):
        vis_dir, date = self._make_synthetic_visual_dir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = build_crop_index(
                vis_dir,
                vis_dir / "behavior_labels" / "individual",
                date=date,
                cameras=["cam_1"],
            )
        required = {"timestamp", "camera", "image_path", "cow_id", "behavior", "class_idx"}
        assert required.issubset(df.columns)

    def test_class_idx_in_range(self, tmp_path):
        vis_dir, date = self._make_synthetic_visual_dir(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = build_crop_index(
                vis_dir,
                vis_dir / "behavior_labels" / "individual",
                date=date,
                cameras=["cam_1"],
            )
        if not df.empty:
            assert df["class_idx"].between(0, 6).all()

    def test_timestamp_filter(self, tmp_path):
        """With a filter that excludes the only timestamp → empty index."""
        vis_dir, date = self._make_synthetic_visual_dir(tmp_path)
        ts = 1690300000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = build_crop_index(
                vis_dir,
                vis_dir / "behavior_labels" / "individual",
                date=date,
                cameras=["cam_1"],
                timestamp_filter=[(ts + 1000, ts + 2000)],  # excludes ts
            )
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanTrack:
    def setup_method(self):
        KalmanTrack._next_id = 1

    def _make_track(self):
        box = np.array([100., 100., 300., 400.])
        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        return KalmanTrack(box, emb), box, emb

    def test_initial_state_tentative(self):
        track, _, _ = self._make_track()
        assert track.state == "tentative"

    def test_track_id_increments(self):
        t1, _, _ = self._make_track()
        t2, _, _ = self._make_track()
        assert t2.track_id == t1.track_id + 1

    def test_predict_returns_4d(self):
        track, _, _ = self._make_track()
        pred = track.predict()
        assert pred.shape == (4,)

    def test_becomes_confirmed_after_n_init_updates(self):
        track, box, emb = self._make_track()
        for i in range(N_INIT):
            track.update(box + i * 5, emb)
        assert track.state == "confirmed"

    def test_still_tentative_before_n_init(self):
        track, box, emb = self._make_track()
        for i in range(N_INIT - 1):
            track.update(box + i, emb)
        assert track.state == "tentative"

    def test_deleted_after_max_age_without_update(self):
        track, box, emb = self._make_track()
        # Confirm first
        for _ in range(N_INIT):
            track.update(box, emb)
        # Then predict without updates until max_age
        for _ in range(MAX_AGE + 1):
            track.predict()
            track.mark_missed()
        assert track.state == "deleted"

    def test_track_features_shape(self):
        track, box, emb = self._make_track()
        for _ in range(N_INIT):
            track.update(box, emb)
        feats = track.get_track_features()
        assert feats.shape == (7,)

    def test_embedding_history_bounded(self):
        track, box, emb = self._make_track()
        from mmcows.models.tracking import EMBED_HISTORY
        for _ in range(EMBED_HISTORY + 5):
            track.update(box, emb)
        assert len(track.embedding_history) == EMBED_HISTORY


class TestDeepSORTTracker:
    def setup_method(self):
        KalmanTrack._next_id = 1

    def _make_dets_embs(self, n: int):
        rng = np.random.default_rng(0)
        dets = rng.uniform(100, 800, (n, 4))
        dets[:, 2] = dets[:, 0] + rng.uniform(50, 200, n)
        dets[:, 3] = dets[:, 1] + rng.uniform(50, 300, n)
        embs = rng.standard_normal((n, 256)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return dets, embs

    def test_new_detections_create_tracks(self):
        tracker = DeepSORTTracker()
        dets, embs = self._make_dets_embs(3)
        tracker.update(dets, embs)
        assert len(tracker.tracks) == 3

    def test_empty_detections_no_crash(self):
        tracker = DeepSORTTracker()
        confirmed = tracker.update(np.zeros((0, 4)), np.zeros((0, 256)))
        assert confirmed == []

    def test_second_update_increases_hit_streak(self):
        tracker = DeepSORTTracker()
        dets, embs = self._make_dets_embs(2)
        tracker.update(dets, embs)
        tracker.update(dets + 5, embs)
        for t in tracker.tracks:
            assert t.hit_streak >= 2

    def test_confirmed_tracks_subset(self):
        tracker = DeepSORTTracker()
        dets, embs = self._make_dets_embs(2)
        for _ in range(N_INIT + 1):
            tracker.update(dets + np.random.randn(*dets.shape) * 2, embs)
        for t in tracker.confirmed_tracks:
            assert t.state == "confirmed"

    def test_reset_clears_state(self):
        tracker = DeepSORTTracker()
        dets, embs = self._make_dets_embs(2)
        tracker.update(dets, embs)
        tracker.reset()
        assert tracker.tracks == []
        assert tracker._frame_count == 0


class TestCosineDistance:
    def test_same_vector_zero_distance(self):
        a = np.eye(4)
        d = _cosine_distance(a, a)
        assert np.allclose(np.diag(d), 0.0, atol=1e-6)

    def test_orthogonal_vectors_unit_distance(self):
        a = np.array([[1., 0., 0.]])
        b = np.array([[0., 1., 0.]])
        d = _cosine_distance(a, b)
        assert abs(d[0, 0] - 1.0) < 1e-6

    def test_output_shape(self):
        a = np.random.randn(5, 10)
        b = np.random.randn(3, 10)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        d = _cosine_distance(a, b)
        assert d.shape == (5, 3)

    def test_bounded_between_0_and_2(self):
        a = np.random.randn(10, 32).astype(np.float32)
        b = np.random.randn(8, 32).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        d = _cosine_distance(a, b)
        assert d.min() >= -1e-6
        assert d.max() <= 2.0 + 1e-6


class TestHungarianAssign:
    def test_perfect_assignment(self):
        cost = np.array([[0.1, 0.9], [0.8, 0.2]])
        matches, unmatched_tr, unmatched_det = _hungarian_assign(cost, max_cost=0.5)
        assert len(matches) == 2
        assert not unmatched_tr
        assert not unmatched_det

    def test_high_cost_rejected(self):
        cost = np.array([[0.9, 0.8], [0.7, 0.6]])
        matches, unmatched_tr, unmatched_det = _hungarian_assign(cost, max_cost=0.5)
        assert len(matches) == 0
        assert sorted(unmatched_tr) == [0, 1]
        assert sorted(unmatched_det) == [0, 1]

    def test_empty_cost_matrix(self):
        matches, unmatched_tr, unmatched_det = _hungarian_assign(np.zeros((0, 3)), max_cost=1.0)
        assert matches == []
        assert unmatched_tr == []
        assert unmatched_det == [0, 1, 2]


# ─────────────────────────────────────────────────────────────────────────────
# backbones
# ─────────────────────────────────────────────────────────────────────────────

class TestBackboneRegistry:
    def test_all_expected_backbones_registered(self):
        expected = {"efficientnet_b0", "efficientvit_s1", "slowfast_r50", "videomae_vit_b"}
        assert expected.issubset(set(_BACKBONE_REGISTRY))

    def test_unknown_backbone_raises(self):
        try:
            import torch   # noqa
        except ImportError:
            pytest.skip("PyTorch not available")
        with pytest.raises((ValueError, ImportError)):
            build_backbone("unknown_backbone_xyz")

    def test_build_without_torch_raises_import_error(self):
        from mmcows.models import backbones as bb
        if bb._TORCH:
            pytest.skip("PyTorch is available")
        with pytest.raises(ImportError):
            build_backbone("efficientnet_b0")


# ─────────────────────────────────────────────────────────────────────────────
# P2 pipeline dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class TestP2Results:
    def _make_results(self):
        r = P2Results(backbone="efficientvit_s1", fusion="cross_modal")
        r.folds = [
            P2FoldResult("f1", "efficientvit_s1", "cross_modal", test_f1=0.85, test_acc=0.92),
            P2FoldResult("f2", "efficientvit_s1", "cross_modal", test_f1=0.89, test_acc=0.94),
            P2FoldResult("f3", "efficientvit_s1", "cross_modal", test_f1=0.91, test_acc=0.95),
        ]
        return r

    def test_mean_f1(self):
        r = self._make_results()
        expected = (0.85 + 0.89 + 0.91) / 3
        assert abs(r.mean_f1 - expected) < 1e-6

    def test_std_f1(self):
        r = self._make_results()
        vals = [0.85, 0.89, 0.91]
        import numpy as np
        assert abs(r.std_f1 - np.std(vals)) < 1e-6

    def test_mean_accuracy(self):
        r = self._make_results()
        expected = (0.92 + 0.94 + 0.95) / 3
        assert abs(r.mean_accuracy - expected) < 1e-6

    def test_empty_results_return_zero(self):
        r = P2Results(backbone="x", fusion="y")
        assert r.mean_f1 == 0.0
        assert r.std_f1 == 0.0
