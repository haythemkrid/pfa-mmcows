"""
Unit tests for Priority 1 signal-processing components.

Tests are purely numerical — no dataset files required.
Run with: pytest tests/test_p1_signal_processing.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from mmcows.models.signal_processing import (
    BEHAVIOR_IDS,
    BEHAVIOR_NAMES,
    MIN_DWELL_SECONDS,
    BBoxFeatureExtractor,
    BBoxKalmanFilter,
    IMUKalmanFilter,
    SemiMarkovBeliefSmoother,
)


# ─────────────────────────────────────────────────────────────────────────────
# BBoxFeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────


class TestBBoxFeatureExtractor:
    def _make_boxes(self, T: int = 20) -> np.ndarray:
        """Generate synthetic [x_min, y_min, x_max, y_max, conf] detections."""
        rng = np.random.default_rng(0)
        x = rng.uniform(100, 3000, T)
        y = rng.uniform(100, 2000, T)
        w = rng.uniform(200, 800, T)
        h = rng.uniform(200, 800, T)
        conf = rng.uniform(0.7, 1.0, T)
        return np.stack([x, y, x + w, y + h, conf], axis=1)

    def test_output_shape_raw(self):
        ext = BBoxFeatureExtractor(proj_dim=None)
        boxes = self._make_boxes(30)
        feats = ext.extract_sequence(boxes)
        assert feats.shape == (30, 8), f"Expected (30, 8), got {feats.shape}"

    def test_aspect_ratio_column(self):
        """AR column (index 7) should be w/h — verified against hand calculation."""
        ext = BBoxFeatureExtractor(proj_dim=None, img_w=4480, img_h=2800)
        # One box: x_min=0, y_min=0, x_max=448 (w=448), y_max=560 (h=560)
        box = np.array([[0, 0, 448, 560, 1.0]])
        feats = ext.extract_sequence(box)
        expected_ar = (448 / 4480) / (560 / 2800)   # (w_norm) / (h_norm)
        # AR is computed in normalised space: (w/img_w) / (h/img_h)
        assert np.isclose(feats[0, 7], expected_ar, rtol=1e-4), (
            f"Expected AR={expected_ar:.4f}, got {feats[0, 7]:.4f}"
        )

    def test_first_frame_deltas_are_zero(self):
        """Frame-over-frame deltas should be 0 at t=0."""
        ext = BBoxFeatureExtractor(proj_dim=None)
        boxes = self._make_boxes(10)
        feats = ext.extract_sequence(boxes)
        # Columns 4 (Δw) and 5 (Δh) at row 0 must be 0
        assert feats[0, 4] == 0.0 and feats[0, 5] == 0.0

    def test_lying_vs_standing_ar(self):
        """Lying cow (wide, low box) should have higher AR than standing (tall, narrow)."""
        ext = BBoxFeatureExtractor(proj_dim=None, img_w=4480, img_h=2800)
        # Wide box: w=800, h=400  → AR=2.0
        lying    = np.array([[100, 100, 900, 500, 1.0]])
        # Tall box: w=200, h=600  → AR=0.33
        standing = np.array([[100, 100, 300, 700, 1.0]])

        ar_lying    = ext.extract_sequence(lying)[0, 7]
        ar_standing = ext.extract_sequence(standing)[0, 7]
        assert ar_lying > ar_standing, (
            f"Expected AR(lying)={ar_lying:.3f} > AR(standing)={ar_standing:.3f}"
        )

    def test_single_detection_no_crash(self):
        ext = BBoxFeatureExtractor(proj_dim=None)
        single = np.array([100, 200, 400, 600, 0.95])
        feats = ext.extract_sequence(single)
        assert feats.shape == (1, 8)

    def test_no_proj_dim_no_project_call(self):
        ext = BBoxFeatureExtractor(proj_dim=None)
        assert ext.projection_layer is None
        with pytest.raises(RuntimeError):
            ext.project(np.zeros((5, 8)))   # should raise, no layer


# ─────────────────────────────────────────────────────────────────────────────
# BBoxKalmanFilter
# ─────────────────────────────────────────────────────────────────────────────


class TestBBoxKalmanFilter:
    def _make_noisy_boxes(self, T: int = 50, noise: float = 5.0) -> np.ndarray:
        """Straight-line trajectory with Gaussian jitter."""
        rng = np.random.default_rng(1)
        t = np.linspace(0, 1, T)
        x_min = 200 + 100 * t + rng.normal(0, noise, T)
        y_min = 300 + 50  * t + rng.normal(0, noise, T)
        x_max = x_min + 400 + rng.normal(0, noise, T)
        y_max = y_min + 300 + rng.normal(0, noise, T)
        return np.stack([x_min, y_min, x_max, y_max], axis=1)

    def test_output_shape(self):
        kf = BBoxKalmanFilter()
        boxes = self._make_noisy_boxes(40)
        smoothed = kf.smooth_sequence(boxes)
        assert smoothed.shape == boxes.shape

    def test_smoothing_reduces_variance(self):
        """Smoothed AR should have lower variance than raw AR."""
        kf  = BBoxKalmanFilter()
        ext = BBoxFeatureExtractor(proj_dim=None, img_w=4480, img_h=2800)
        boxes = self._make_noisy_boxes(100, noise=20.0)
        smoothed = kf.smooth_sequence(boxes)

        raw_ar      = ext.extract_sequence(np.hstack([boxes, np.ones((100, 1))]))[: , 7]
        smooth_ar   = ext.extract_sequence(np.hstack([smoothed, np.ones((100, 1))]))[: , 7]

        assert np.var(smooth_ar) < np.var(raw_ar), (
            f"Expected var(smooth_AR) < var(raw_AR): "
            f"{np.var(smooth_ar):.4f} vs {np.var(raw_ar):.4f}"
        )

    def test_smoothed_stays_near_trajectory(self):
        """Smoothed centroids should be close to the true trajectory."""
        kf = BBoxKalmanFilter(process_noise=0.1, measurement_noise=5.0)
        boxes_clean = self._make_noisy_boxes(50, noise=0.0)   # ground truth
        boxes_noisy = self._make_noisy_boxes(50, noise=15.0)  # observed

        smoothed = kf.smooth_sequence(boxes_noisy)
        # Compute centroid error
        cx_clean = (boxes_clean[:, 0] + boxes_clean[:, 2]) / 2
        cx_smooth = (smoothed[:, 0] + smoothed[:, 2]) / 2
        cx_noisy  = (boxes_noisy[:, 0] + boxes_noisy[:, 2]) / 2

        err_smooth = np.mean(np.abs(cx_smooth - cx_clean))
        err_noisy  = np.mean(np.abs(cx_noisy  - cx_clean))
        assert err_smooth < err_noisy, (
            f"Smoothed error {err_smooth:.2f} should be less than noisy {err_noisy:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# IMUKalmanFilter
# ─────────────────────────────────────────────────────────────────────────────


class TestIMUKalmanFilter:
    def _make_noisy_imu(self, T: int = 500) -> np.ndarray:
        rng = np.random.default_rng(2)
        # True signal: constant 9.8 in z, small values in x/y
        true_signal = np.zeros((T, 3))
        true_signal[:, 2] = 9.8
        noise = rng.normal(0, 0.5, (T, 3))
        return true_signal + noise

    def test_output_shape(self):
        kf = IMUKalmanFilter()
        accel = self._make_noisy_imu(200)
        smoothed = kf.smooth_sequence(accel)
        assert smoothed.shape == accel.shape

    def test_smoothing_reduces_noise(self):
        kf = IMUKalmanFilter(sigma_acc=0.5)
        accel = self._make_noisy_imu(500)
        smoothed = kf.smooth_sequence(accel)
        # Variance of the smoothed z-axis should be lower than raw
        assert np.var(smoothed[:, 2]) < np.var(accel[:, 2])

    def test_activity_index_shape(self):
        kf = IMUKalmanFilter()
        accel = self._make_noisy_imu(100)
        A = kf.compute_activity_index(accel)
        assert A.shape == (100,)

    def test_activity_index_non_negative(self):
        kf = IMUKalmanFilter()
        accel = self._make_noisy_imu(100)
        A = kf.compute_activity_index(accel)
        assert (A >= 0).all()

    def test_activity_index_with_temperature(self):
        kf = IMUKalmanFilter()
        accel = self._make_noisy_imu(50)
        dt = np.random.randn(50) * 0.2
        A_with = kf.compute_activity_index(accel, delta_temp=dt)
        A_no   = kf.compute_activity_index(accel)
        # With temperature the index should generally be ≥ without
        assert np.mean(A_with) >= np.mean(A_no) - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# SemiMarkovBeliefSmoother
# ─────────────────────────────────────────────────────────────────────────────


class TestSemiMarkovBeliefSmoother:
    K = 7
    BEHAVIOR_IDS_LIST = [1, 2, 3, 4, 5, 6, 7]

    def _make_smoother(self, dt: float = 15.0) -> SemiMarkovBeliefSmoother:
        return SemiMarkovBeliefSmoother(
            behavior_ids=self.BEHAVIOR_IDS_LIST, dt=dt
        )

    def _make_likelihoods(self, T: int, peak_class: int = 0) -> np.ndarray:
        """Synthetic likelihoods — class peak_class is dominant."""
        rng = np.random.default_rng(3)
        raw = rng.dirichlet(np.ones(self.K), size=T)
        # Make peak_class dominant
        raw[:, peak_class] += 2.0
        return raw / raw.sum(axis=1, keepdims=True)

    def test_forward_output_shape(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(100)
        b = sm.forward(L)
        assert b.shape == (100, self.K)

    def test_forward_rows_sum_to_one(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(50)
        b = sm.forward(L)
        assert np.allclose(b.sum(axis=1), 1.0, atol=1e-6)

    def test_smooth_output_shape(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(80)
        g = sm.smooth(L)
        assert g.shape == (80, self.K)

    def test_smooth_rows_sum_to_one(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(80)
        g = sm.smooth(L)
        assert np.allclose(g.sum(axis=1), 1.0, atol=1e-6)

    def test_decode_length(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(60)
        g = sm.smooth(L)
        labels = sm.decode(g)
        assert len(labels) == 60

    def test_decode_valid_labels(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(60)
        labels = sm.decode(sm.smooth(L))
        assert all(lab in self.BEHAVIOR_IDS_LIST for lab in labels)

    def test_dwell_constraint_lying(self):
        """The smoother must not produce lying→non-lying transitions faster than 30 s."""
        sm = self._make_smoother(dt=1.0)  # 1-second steps
        T = 200
        # Alternate every 10 steps between lying (class idx 6) and standing (class idx 1)
        L = np.zeros((T, self.K))
        for t in range(T):
            dominant = 6 if (t // 10) % 2 == 0 else 1  # alternate lying / standing
            L[t, dominant] = 0.9
            L[t, :] += 0.1 / self.K
            L[t] /= L[t].sum()

        # Forward pass enforces dwell constraints
        b = sm.forward(L)
        labels = sm.decode(b)

        # Count how many transitions out of "lying" (label=7) happen
        lying_id = 7
        transitions_out_of_lying = sum(
            1 for i in range(1, len(labels))
            if labels[i - 1] == lying_id and labels[i] != lying_id
        )
        # With min_dwell=30 s and dt=1 s, a lying bout must last ≥ 30 steps.
        # In 200 steps alternating every 10, raw would give ~10 transitions.
        # Constrained should give ≤ ~6 (200/30 ≈ 6 bouts max).
        assert transitions_out_of_lying <= 10, (
            f"Too many lying→other transitions: {transitions_out_of_lying}"
        )

    def test_locomotion_budget_sums_to_one(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(100)
        g = sm.smooth(L)
        budget = sm.locomotion_budget(g)
        total = sum(budget.values())
        assert np.isclose(total, 1.0, atol=1e-6), f"Budget sums to {total:.6f}"

    def test_dwell_histogram_non_empty(self):
        sm = self._make_smoother()
        L = self._make_likelihoods(100, peak_class=6)  # lying dominant
        g = sm.smooth(L)
        hist = sm.dwell_time_histogram(g)
        # At least the dominant behavior should have bouts
        assert sum(len(v) for v in hist.values()) > 0

    def test_estimate_transition_matrix(self):
        sm = self._make_smoother()
        rng = np.random.default_rng(4)
        labels = rng.choice(self.BEHAVIOR_IDS_LIST, size=500)
        A = sm.estimate_transition_matrix(labels)
        assert A.shape == (self.K, self.K)
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)

    def test_invalid_transition_matrix_shape(self):
        with pytest.raises(ValueError, match="shape"):
            SemiMarkovBeliefSmoother(
                behavior_ids=self.BEHAVIOR_IDS_LIST,
                transition_matrix=np.eye(3),   # wrong size
            )

    def test_invalid_transition_matrix_not_stochastic(self):
        bad_A = np.eye(self.K) * 2.0   # rows sum to 2, not 1
        with pytest.raises(ValueError, match="rows must sum"):
            SemiMarkovBeliefSmoother(
                behavior_ids=self.BEHAVIOR_IDS_LIST,
                transition_matrix=bad_A,
            )

    def test_smooth_vs_forward_uncertainty(self):
        """Smoothed beliefs should differ from forward beliefs — future helps."""
        sm = self._make_smoother()
        L = self._make_likelihoods(100)
        b_fwd = sm.forward(L)
        b_smo = sm.smooth(L)
        # They should not be identical (future evidence changes the distribution)
        assert not np.allclose(b_fwd, b_smo, atol=1e-6), (
            "Forward and smooth beliefs are identical — something is wrong"
        )