"""
src.cuboid_ablation.metrics
=============================

Evaluation metrics for the cuboid reconstruction ablation.

All distances in centimetres.  Thresholds for success-rate curves are in cm.

Metrics computed per-frame per-person
--------------------------------------
  pos_err_cm     – Euclidean distance between reconstructed (cx, cy) and
                   GT ground position (world_x_cm, world_y_cm).
  height_err_cm  – |h_reconstructed – PERSON_HEIGHT_CM|
  depth_err_cm   – |d_reconstructed – PERSON_DEPTH_CM|
  rms_px         – per-side RMS pixel residual (from the reconstructor itself)
  success_25     – indicator pos_err_cm < 25 cm
  success_50     – indicator pos_err_cm < 50 cm
  success_100    – indicator pos_err_cm < 100 cm

Aggregate metrics
-----------------
  mean / median of each per-frame metric across all (frame, person) pairs.
  success rate at each threshold.
  RMSE of position error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SUCCESS_THRESHOLDS_CM = [25.0, 50.0, 100.0]


@dataclass
class FrameResult:
    """Metrics for one (frame, person) reconstruction attempt."""

    frame_idx: int
    person_id: int
    condition_name: str

    # reconstruction output
    cx: float
    cy: float
    d_cm: float
    h_cm: float
    rms_px: float
    n_cams_used: int

    # ground truth
    gt_x_cm: float
    gt_y_cm: float

    # derived
    pos_err_cm: float = 0.0
    height_err_cm: float = 0.0
    depth_err_cm: float = 0.0

    # success indicators (filled by compute_metrics)
    success_25: bool = False
    success_50: bool = False
    success_100: bool = False

    failed: bool = False      # True if the reconstructor threw an exception
    fail_reason: str = ""


def make_frame_result(
    frame_idx: int,
    person_id: int,
    condition_name: str,
    cx: float,
    cy: float,
    d_cm: float,
    h_cm: float,
    rms_px: float,
    n_cams_used: int,
    gt_x_cm: float,
    gt_y_cm: float,
    gt_height_cm: float,
    gt_depth_cm: float,
) -> FrameResult:
    pos_err = float(np.hypot(cx - gt_x_cm, cy - gt_y_cm))
    return FrameResult(
        frame_idx=frame_idx,
        person_id=person_id,
        condition_name=condition_name,
        cx=cx,
        cy=cy,
        d_cm=d_cm,
        h_cm=h_cm,
        rms_px=rms_px,
        n_cams_used=n_cams_used,
        gt_x_cm=gt_x_cm,
        gt_y_cm=gt_y_cm,
        pos_err_cm=pos_err,
        height_err_cm=abs(h_cm - gt_height_cm),
        depth_err_cm=abs(d_cm - gt_depth_cm),
        success_25=pos_err < 25.0,
        success_50=pos_err < 50.0,
        success_100=pos_err < 100.0,
    )


def make_failed_result(
    frame_idx: int,
    person_id: int,
    condition_name: str,
    n_cams_used: int,
    gt_x_cm: float,
    gt_y_cm: float,
    reason: str = "",
) -> FrameResult:
    return FrameResult(
        frame_idx=frame_idx,
        person_id=person_id,
        condition_name=condition_name,
        cx=float("nan"),
        cy=float("nan"),
        d_cm=float("nan"),
        h_cm=float("nan"),
        rms_px=float("nan"),
        n_cams_used=n_cams_used,
        gt_x_cm=gt_x_cm,
        gt_y_cm=gt_y_cm,
        pos_err_cm=float("nan"),
        height_err_cm=float("nan"),
        depth_err_cm=float("nan"),
        failed=True,
        fail_reason=reason,
    )


@dataclass
class AggregateMetrics:
    """Aggregate statistics for one ablation condition over all frames."""

    condition_name: str
    n_total: int
    n_failed: int

    mean_pos_err_cm: float
    median_pos_err_cm: float
    rmse_pos_cm: float
    p90_pos_err_cm: float

    mean_height_err_cm: float
    mean_depth_err_cm: float

    mean_rms_px: float
    median_rms_px: float

    success_rate_25: float
    success_rate_50: float
    success_rate_100: float

    failure_rate: float

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def as_mlflow_metrics(self) -> dict:
        return {
            f"metrics/{k}": v
            for k, v in self.as_dict().items()
            if isinstance(v, (int, float)) and k != "condition_name"
        }


def aggregate(results: Sequence[FrameResult], condition_name: str) -> AggregateMetrics:
    """Compute aggregate metrics from a list of FrameResult objects.

    Failed frames are counted in n_failed and failure_rate but excluded from
    all other statistics so that a bad condition doesn't trivially win by
    returning small errors on only the easy frames.
    """
    n_total = len(results)
    failed = [r for r in results if r.failed]
    valid = [r for r in results if not r.failed]
    n_failed = len(failed)

    def _stat(values: list[float], fn):
        if not values:
            return float("nan")
        return float(fn(values))

    pos_errs  = [r.pos_err_cm for r in valid]
    h_errs    = [r.height_err_cm for r in valid]
    d_errs    = [r.depth_err_cm for r in valid]
    rms_vals  = [r.rms_px for r in valid]

    success_25  = sum(r.success_25 for r in valid)
    success_50  = sum(r.success_50 for r in valid)
    success_100 = sum(r.success_100 for r in valid)
    denom = n_total  # denominator includes failures (they count as non-success)

    return AggregateMetrics(
        condition_name=condition_name,
        n_total=n_total,
        n_failed=n_failed,
        mean_pos_err_cm=_stat(pos_errs, np.mean),
        median_pos_err_cm=_stat(pos_errs, np.median),
        rmse_pos_cm=_stat(pos_errs, lambda x: np.sqrt(np.mean(np.array(x) ** 2))),
        p90_pos_err_cm=_stat(pos_errs, lambda x: float(np.percentile(x, 90))),
        mean_height_err_cm=_stat(h_errs, np.mean),
        mean_depth_err_cm=_stat(d_errs, np.mean),
        mean_rms_px=_stat(rms_vals, np.mean),
        median_rms_px=_stat(rms_vals, np.median),
        success_rate_25=success_25 / denom if denom else float("nan"),
        success_rate_50=success_50 / denom if denom else float("nan"),
        success_rate_100=success_100 / denom if denom else float("nan"),
        failure_rate=n_failed / n_total if n_total else float("nan"),
    )
