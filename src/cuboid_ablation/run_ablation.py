"""
src.cuboid_ablation.run_ablation
==================================

Entry point for the cuboid reconstruction ablation study on MultiviewX.

Usage
-----
    python -m src.cuboid_ablation.run_ablation \\
        --dataset-root ~/store/data/raw/MultiviewX \\
        --output-dir   logs/cuboid_ablation \\
        [--frames all|test|0-39]   (default: all 400 frames)
        [--seed 42]
        [--no-mlflow]
        [--sanity-only]

Resilience
----------
  Each condition writes results to:
    logs/cuboid_ablation/checkpoints/<condition_name>.csv        (geometry)
    logs/cuboid_ablation/downstream/<condition_name>_ds.csv      (classifier)
  A condition whose checkpoint CSV already covers all frame indices is skipped.
  Re-running therefore resumes exactly where it stopped.

Sanity check
------------
  Before the main loop: baseline on 2 frames, figures saved to sanity/.
  A failure here aborts before the full run starts.

Output structure
----------------
  logs/cuboid_ablation/
  ├── sanity/                   diagnostic figures (2 frames)
  ├── checkpoints/              per-condition geometry CSV
  ├── downstream/               per-condition classifier metrics CSV
  ├── figures/                  summary plots
  ├── label_bank.csv            cached GT zone+density labels (built once)
  ├── label_bank.kmeans_centres.npy
  ├── ablation_results.csv      geometry aggregate table
  └── downstream_results.csv    classifier aggregate table
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── MLflow ────────────────────────────────────────────────────────────────────
try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False
    print("[WARN] mlflow not found — logging disabled.")

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.cuboid_ablation.cuboid_reconstruction import (
    _hard_cost,
    _build_result,
    _build_obs_array,
    _cuboid_homogeneous,
    _LBFGS_OPTS,
    _NM_OPTS,
    _RESTART_ANGLES,
)

from src.cuboid_ablation.multiviewx_loader import (
    load_calibration,
    load_frame_annotations,
    list_frame_indices,
    PersonAnnotation,
    MultiviewXCalib,
)
from src.cuboid_ablation.ablation_engine import (
    AblationCondition,
    build_ofat_conditions,
    select_diverse_cameras,
    apply_bbox_noise,
    ground_contact_point,
    PERSON_HEIGHT_CM,
    PERSON_WIDTH_CM,
    PERSON_DEPTH_CM,
)
from src.cuboid_ablation.metrics import (
    FrameResult,
    AggregateMetrics,
    make_frame_result,
    make_failed_result,
    aggregate,
)
from src.cuboid_ablation.downstream_labels import DownstreamLabelBank
from src.cuboid_ablation.downstream_classifier import (
    DownstreamResult,
    TaskResult,
    evaluate_downstream,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cuboid_ablation")

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]

DIMENSION_LABELS = {
    "baseline":            "Baseline",
    "A_num_cameras":       "(A) Number of Cameras",
    "B_bbox_noise":        "(B) BBox Noise (px σ)",
    "C_ground_contact":    "(C) Ground Contact Point",
    "D_prior_sensitivity": "(D) Prior Scale Factor",
    "E_lse_alpha":         "(E) LSE Sharpness α",
    "F_warm_start":        "(F) Warm-Start Strategy",
}


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction with ablation overrides
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_with_ablation(
    calib: MultiviewXCalib,
    active_cams: list[str],
    bbox_dict: dict[str, tuple],
    condition: AblationCondition,
    rng: np.random.Generator,
):
    """Full 3-stage reconstruction pipeline with ablation parameters injected."""

    # Scaled priors
    w0 = PERSON_WIDTH_CM  * condition.prior_factor
    d0 = PERSON_DEPTH_CM  * condition.prior_factor
    h0 = PERSON_HEIGHT_CM * condition.prior_factor

    # Bbox noise
    noisy: dict[str, tuple] = {
        cam: apply_bbox_noise(bbox_dict[cam], condition.noise_sigma, rng)
        for cam in active_cams
    }

    P_stack = np.stack([calib.proj_mats[c] for c in active_cams])    # (N,3,4)
    C0_arr  = np.stack([calib.cam_centres[c] for c in active_cams])  # (N,3)
    obs_arr = _build_obs_array(active_cams, noisy)                    # (N,4)
    N = len(active_cams)
    w = w0   # width is fixed throughout

    # ── Warm start ────────────────────────────────────────────────────────
    if condition.warm_start == "origin":
        cx0, cy0 = 800.0, 1250.0   # scene centre (cm)
    else:
        gpts = []
        for P, C0, row in zip(P_stack, C0_arr, obs_arr):
            u_min, u_max, v_min, v_max = row
            # obs_arr is [umin,umax,vmin,vmax]; ground_contact_point expects (x1,y1,x2,y2)
            uc, vc = ground_contact_point(
                (u_min, v_min, u_max, v_max), condition.ground_contact
            )
            Pp  = P.T @ np.linalg.inv(P @ P.T)
            Xh  = Pp @ np.array([uc, vc, 1.0])
            pt  = Xh[:3] / Xh[3]
            ray = pt - C0
            nrm = np.linalg.norm(ray)
            if nrm < 1e-12:
                continue
            ray /= nrm
            if abs(ray[2]) < 1e-9:
                continue
            t  = -C0[2] / ray[2]
            gp = C0 + t * ray
            if abs(gp[0]) < 10_000 and abs(gp[1]) < 10_000:
                gpts.append(gp[:2])
        if gpts:
            cx0, cy0 = np.median(np.array(gpts), axis=0)
        else:
            cx0, cy0 = 800.0, 1250.0

    x0   = np.array([cx0, cy0, d0, h0, 0.0])
    bnds = [
        (cx0 - 800, cx0 + 800),
        (cy0 - 800, cy0 + 800),
        (10.0, 600.0),
        (50.0, 250.0),
        (None, None),
    ]

    alpha = condition.alpha
    args  = (P_stack, obs_arr, w)

    # ── LSE cost with ablation alpha ──────────────────────────────────────
    def smooth_cost(x5, P_stack, obs_arr, w):
        cx, cy, d, h, th = x5
        if d < 5 or h < 5:
            return 1e12
        Xh = _cuboid_homogeneous(cx, cy, d, h, th, w)
        ph = P_stack @ Xh
        pu = ph[:, 0, :] / ph[:, 2, :]
        pv = ph[:, 1, :] / ph[:, 2, :]

        def smax(a):
            m = a.max(1, keepdims=True)
            return np.log(np.exp(alpha * (a - m)).sum(1)) / alpha + m[:, 0]

        diff = np.stack([
            -smax(-pu) - obs_arr[:, 0],
             smax( pu) - obs_arr[:, 1],
            -smax(-pv) - obs_arr[:, 2],
             smax( pv) - obs_arr[:, 3],
        ], axis=1)
        return float((diff ** 2).sum())

    # ── Stage 1: L-BFGS-B ────────────────────────────────────────────────
    res = minimize(smooth_cost, x0, args=args,
                   method="L-BFGS-B", bounds=bnds, options=_LBFGS_OPTS)
    hc = _hard_cost(res.x, *args)

    # ── Stage 2: Nelder-Mead polish ───────────────────────────────────────
    r2 = minimize(_hard_cost, res.x, args=args,
                  method="Nelder-Mead", options=_NM_OPTS)
    if r2.fun < hc:
        res, hc = r2, r2.fun

    # ── Stage 3: Rotated restarts if poor ────────────────────────────────
    if np.sqrt(hc / (N * 4)) > 300.0:
        for dth in _RESTART_ANGLES:
            xi = x0.copy(); xi[4] += dth
            rf = minimize(smooth_cost, xi, args=args,
                          method="L-BFGS-B", bounds=bnds, options=_LBFGS_OPTS)
            hcf = _hard_cost(rf.x, *args)
            if hcf < hc:
                res, hc = rf, hcf
            if np.sqrt(hc / (N * 4)) < 300.0:
                break

    return _build_result(res.x, hc, w, N, P_stack, obs_arr)


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition geometry evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_condition(
    condition: AblationCondition,
    calib: MultiviewXCalib,
    frame_indices: list[int],
    dataset_root: str,
    output_dir: Path,
    seed: int = 42,
) -> list[FrameResult]:
    """Reconstruct all (frame, person) pairs for one condition.

    Results are appended line-by-line to a checkpoint CSV so the run
    can be safely interrupted and resumed.
    """
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{condition.name}.csv"

    # Resume: skip if checkpoint already covers all frames
    if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
        df_ckpt = pd.read_csv(ckpt_path)
        completed = set(df_ckpt["frame_idx"].unique())
        if set(frame_indices).issubset(completed):
            logger.info("  [SKIP] %s — checkpoint complete (%d frames).",
                        condition.name, len(completed))
            return _load_results_from_csv(ckpt_path)
        logger.info("  [RESUME] %s — %d/%d frames done.",
                    condition.name, len(completed), len(frame_indices))
    else:
        completed = set()

    active_cams = select_diverse_cameras(
        calib.cam_names, calib.cam_centres, condition.num_cameras
    )
    logger.info("  %s | cams(%d)=%s", condition.name, len(active_cams), active_cams)

    rng = np.random.default_rng(seed)
    results: list[FrameResult] = []
    fieldnames = list(FrameResult.__dataclass_fields__.keys())

    with ckpt_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if ckpt_path.stat().st_size == 0:
            writer.writeheader()

        for frame_idx in frame_indices:
            if frame_idx in completed:
                continue

            try:
                persons = load_frame_annotations(
                    dataset_root, frame_idx,
                    min_cameras_visible=condition.num_cameras,
                )
            except Exception as exc:
                logger.warning("Frame %d load failed: %s", frame_idx, exc)
                continue

            for person in persons:
                bbox_dict = {
                    cam: bb
                    for cam in active_cams
                    if (bb := person.bboxes.get(cam)) is not None
                }
                n_used = len(bbox_dict)

                if n_used < 2:
                    r = make_failed_result(
                        frame_idx, person.person_id, condition.name,
                        n_used, person.world_x_cm, person.world_y_cm,
                        reason="insufficient_cameras",
                    )
                else:
                    try:
                        result = _reconstruct_with_ablation(
                            calib, list(bbox_dict.keys()), bbox_dict,
                            condition, rng,
                        )
                        r = make_frame_result(
                            frame_idx=frame_idx,
                            person_id=person.person_id,
                            condition_name=condition.name,
                            cx=result.cx, cy=result.cy,
                            d_cm=result.d, h_cm=result.h,
                            rms_px=result.rms_px,
                            n_cams_used=n_used,
                            gt_x_cm=person.world_x_cm,
                            gt_y_cm=person.world_y_cm,
                            gt_height_cm=PERSON_HEIGHT_CM,
                            gt_depth_cm=PERSON_DEPTH_CM,
                        )
                    except Exception as exc:
                        logger.debug("  Frame %d p%d: %s",
                                     frame_idx, person.person_id, exc)
                        r = make_failed_result(
                            frame_idx, person.person_id, condition.name,
                            n_used, person.world_x_cm, person.world_y_cm,
                            reason=str(exc)[:120],
                        )

                results.append(r)
                writer.writerow(asdict(r))
            fh.flush()

    return results


def _load_results_from_csv(path: Path) -> list[FrameResult]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        d = row.to_dict()
        for col in ("success_25", "success_50", "success_100", "failed"):
            d[col] = bool(d[col])
        out.append(FrameResult(**d))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sanity figures
# ─────────────────────────────────────────────────────────────────────────────

def save_sanity_figure(
    frame_idx: int,
    persons: list[PersonAnnotation],
    results: list[FrameResult],
    output_dir: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"SANITY CHECK — Frame {frame_idx:04d}  |  Baseline  |  MultiviewX",
        fontsize=12, fontweight="bold",
    )

    # Left: GT vs reconstructed ground positions
    ax = axes[0]
    ax.set_facecolor("#f4f4f4")
    ax.set_title("GT (▲) vs Reconstructed (×)", fontsize=10)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.35)

    ax.scatter(
        [p.world_x_cm for p in persons],
        [p.world_y_cm for p in persons],
        marker="^", s=110, color="royalblue", label="GT", zorder=5,
    )

    for r in results:
        if r.failed or np.isnan(r.cx):
            ax.scatter([r.gt_x_cm], [r.gt_y_cm], marker="x", s=60,
                       color="gray", linewidths=1.2, zorder=4, alpha=0.5)
            continue
        col = ("green" if r.pos_err_cm < 50
               else "orange" if r.pos_err_cm < 150 else "crimson")
        ax.scatter([r.cx], [r.cy], marker="x", s=110, color=col,
                   linewidths=2, zorder=6)
        ax.plot([r.gt_x_cm, r.cx], [r.gt_y_cm, r.cy],
                color="gray", lw=0.7, alpha=0.45)
        ax.text(r.cx, r.cy + 18,
                f"{r.pos_err_cm:.0f}cm / {r.rms_px:.0f}px",
                fontsize=6.5, ha="center", color=col)

    ax.legend(handles=[
        mpatches.Patch(color="royalblue", label="GT position"),
        mpatches.Patch(color="green",     label="Recon err < 50 cm"),
        mpatches.Patch(color="orange",    label="Recon err 50–150 cm"),
        mpatches.Patch(color="crimson",   label="Recon err > 150 cm"),
        mpatches.Patch(color="gray",      label="Failed"),
    ], fontsize=7, loc="upper right")

    # Right: per-person error bars
    ax2 = axes[1]
    ax2.set_facecolor("#f4f4f4")
    ax2.set_title("Per-person position error (cm)", fontsize=10)
    ax2.set_xlabel("Person"); ax2.set_ylabel("cm")
    ax2.grid(axis="y", alpha=0.35)
    valid_r = [r for r in results if not r.failed and not np.isnan(r.pos_err_cm)]
    if valid_r:
        lbls   = [f"p{r.person_id}" for r in valid_r]
        errs   = [r.pos_err_cm for r in valid_r]
        cols   = ["green" if e < 50 else ("orange" if e < 150 else "crimson")
                  for e in errs]
        ax2.bar(range(len(lbls)), errs, color=cols, alpha=0.85, edgecolor="white")
        ax2.set_xticks(range(len(lbls)))
        ax2.set_xticklabels(lbls, rotation=45, ha="right", fontsize=8)
        mean_e = np.mean(errs)
        ax2.axhline(mean_e, color="navy", ls="--", lw=1.5,
                    label=f"mean = {mean_e:.1f} cm")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No valid reconstructions", ha="center",
                 va="center", transform=ax2.transAxes, fontsize=10, color="gray")

    fig.tight_layout()
    out = output_dir / "sanity" / f"sanity_frame{frame_idx:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Sanity figure: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation summary figures
# ─────────────────────────────────────────────────────────────────────────────

def save_ablation_figures(
    agg_list: list[AggregateMetrics],
    downstream_list: list[DownstreamResult],
    conditions: list[AblationCondition],
    output_dir: Path,
):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dim_map = {c.name: c.dimension for c in conditions}
    ds_map  = {d.condition_name: d for d in downstream_list}
    baseline_agg = next((a for a in agg_list if a.condition_name == "baseline"), None)
    baseline_ds  = ds_map.get("baseline")

    from collections import defaultdict
    dim_to_agg: dict[str, list[AggregateMetrics]] = defaultdict(list)
    for a in agg_list:
        dim_to_agg[dim_map.get(a.condition_name, "unknown")].append(a)

    # ── 1. Per-dimension panels (geometry + both downstream tasks) ────────
    for dim, agg_items in dim_to_agg.items():
        names = [a.condition_name for a in agg_items]
        xs    = list(range(len(names)))

        def _ds_f1(a, task, model_lower):
            dr = ds_map.get(a.condition_name)
            tr = getattr(dr, f"{task}_{model_lower}", None) if dr else None
            return tr.mean_f1_macro if tr else float("nan")

        def _ds_auc(a, task, model_lower):
            dr = ds_map.get(a.condition_name)
            tr = getattr(dr, f"{task}_{model_lower}", None) if dr else None
            return tr.mean_auc if tr else float("nan")

        panels = [
            # Geometry
            ([a.mean_pos_err_cm   for a in agg_items], "Mean Pos. Error (cm)",     "cm",   False),
            ([a.success_rate_50*100 for a in agg_items],"Success Rate @50cm",       "%",    True),
            ([a.mean_rms_px       for a in agg_items], "Mean RMS Residual (px)",   "px",   False),
            ([a.failure_rate*100  for a in agg_items], "Failure Rate (%)",         "%",    False),
            # Zone task
            ([_ds_f1(a,"zone","mlp")  for a in agg_items], "Zone F1-macro (MLP)",  "F1",   True),
            ([_ds_f1(a,"zone","rf")   for a in agg_items], "Zone F1-macro (RF)",   "F1",   True),
            # Density task
            ([_ds_f1(a,"density","mlp")  for a in agg_items], "Density F1-macro (MLP)", "F1", True),
            ([_ds_f1(a,"density","rf")   for a in agg_items], "Density F1-macro (RF)",  "F1", True),
            ([_ds_auc(a,"density","mlp") for a in agg_items], "Density AUC-ROC (MLP)",  "",  True),
            ([_ds_auc(a,"density","rf")  for a in agg_items], "Density AUC-ROC (RF)",   "",  True),
        ]

        ncols = 5
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle(DIMENSION_LABELS.get(dim, dim), fontsize=13, fontweight="bold")

        for ax_i, (vals, title, ylabel, higher_better) in enumerate(panels):
            ax = axes[ax_i]
            bar_cols = [COLORS[i % len(COLORS)] for i in range(len(names))]
            ax.bar(xs, vals, color=bar_cols, alpha=0.85, edgecolor="white")

            # Baseline reference
            if baseline_agg is not None and dim != "baseline":
                ref_geo = {
                    "Mean Pos. Error (cm)":  baseline_agg.mean_pos_err_cm,
                    "Success Rate @50cm":    baseline_agg.success_rate_50 * 100,
                    "Mean RMS Residual (px)": baseline_agg.mean_rms_px,
                    "Failure Rate (%)":      baseline_agg.failure_rate * 100,
                }
                if title in ref_geo:
                    ax.axhline(ref_geo[title], color="black", ls="--",
                               lw=1.3, label="baseline")
                    ax.legend(fontsize=7)
                elif baseline_ds:
                    for task, ml, lbl in [
                        ("zone", "mlp", "Zone F1-macro (MLP)"),
                        ("zone", "rf",  "Zone F1-macro (RF)"),
                        ("density", "mlp", "Density F1-macro (MLP)"),
                        ("density", "rf",  "Density F1-macro (RF)"),
                        ("density", "mlp", "Density AUC-ROC (MLP)"),
                        ("density", "rf",  "Density AUC-ROC (RF)"),
                    ]:
                        if title == lbl:
                            tr = getattr(baseline_ds, f"{task}_{ml}", None)
                            if tr:
                                ref = tr.mean_auc if "AUC" in lbl else tr.mean_f1_macro
                                ax.axhline(ref, color="black", ls="--",
                                           lw=1.3, label="baseline")
                                ax.legend(fontsize=7)

            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=32, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        for ax_i in range(len(panels), len(axes)):
            axes[ax_i].set_visible(False)

        fig.tight_layout()
        out = fig_dir / f"dim_{dim}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Figure: %s", out)

    # ── 2. Geometry heatmap ───────────────────────────────────────────────
    geo_spec = [
        ("mean_pos_err_cm",   "Pos err (cm)"),
        ("rmse_pos_cm",       "RMSE (cm)"),
        ("p90_pos_err_cm",    "P90 (cm)"),
        ("success_rate_50",   "SR@50cm"),
        ("success_rate_100",  "SR@100cm"),
        ("mean_rms_px",       "RMS (px)"),
        ("mean_height_err_cm","H err (cm)"),
        ("mean_depth_err_cm", "D err (cm)"),
        ("failure_rate",      "Fail rate"),
    ]
    _heatmap(
        row_labels=[a.condition_name for a in agg_list],
        col_labels=[s[1] for s in geo_spec],
        data=np.array([[getattr(a, k, float("nan")) for k, _ in geo_spec]
                       for a in agg_list], dtype=float),
        title="Geometry Metrics — All Conditions",
        path=fig_dir / "heatmap_geometry.png",
        cmap="RdYlGn_r",
    )

    # ── 3. Downstream heatmap ─────────────────────────────────────────────
    ds_spec = [
        ("zone_mlp",    "mean_f1_macro",  "Zone F1 MLP"),
        ("zone_rf",     "mean_f1_macro",  "Zone F1 RF"),
        ("zone_mlp",    "mean_accuracy",  "Zone Acc MLP"),
        ("zone_rf",     "mean_accuracy",  "Zone Acc RF"),
        ("density_mlp", "mean_f1_macro",  "Den F1 MLP"),
        ("density_rf",  "mean_f1_macro",  "Den F1 RF"),
        ("density_mlp", "mean_auc",       "Den AUC MLP"),
        ("density_rf",  "mean_auc",       "Den AUC RF"),
        ("density_mlp", "mean_accuracy",  "Den Acc MLP"),
        ("density_rf",  "mean_accuracy",  "Den Acc RF"),
    ]

    def _ds_val(cond_name, attr, metric):
        dr = ds_map.get(cond_name)
        tr = getattr(dr, attr, None) if dr else None
        return getattr(tr, metric, float("nan")) if tr else float("nan")

    _heatmap(
        row_labels=[a.condition_name for a in agg_list],
        col_labels=[s[2] for s in ds_spec],
        data=np.array([
            [_ds_val(a.condition_name, attr, metric) for attr, metric, _ in ds_spec]
            for a in agg_list
        ], dtype=float),
        title="Downstream Classifier Metrics — All Conditions",
        path=fig_dir / "heatmap_downstream.png",
        cmap="RdYlGn",
    )

    # ── 4. Success-rate curves ────────────────────────────────────────────
    thresholds = np.arange(5, 301, 5)
    ckpt_dir   = output_dir / "checkpoints"
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlabel("Position error threshold (cm)", fontsize=11)
    ax.set_ylabel("Cumulative success rate", fontsize=11)
    ax.set_title("Success-Rate Curves by Ablation Condition", fontsize=12)
    ax.set_xlim(0, 300); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3)

    for ci, cond in enumerate(conditions):
        csv_p = ckpt_dir / f"{cond.name}.csv"
        if not csv_p.exists():
            continue
        df_c  = pd.read_csv(csv_p)
        errs  = df_c[~df_c["failed"].astype(bool)]["pos_err_cm"].dropna().values
        if len(errs) == 0:
            continue
        srs = [np.mean(errs < t) for t in thresholds]
        lw  = 2.8 if cond.name == "baseline" else 1.2
        ax.plot(thresholds, srs, label=cond.name,
                color=COLORS[ci % len(COLORS)], lw=lw,
                alpha=1.0 if cond.name == "baseline" else 0.65)

    ax.legend(fontsize=6.5, ncol=3, loc="lower right")
    fig.tight_layout()
    out = fig_dir / "success_rate_curves.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure: %s", out)

    # ── 5. Scatter: geometric error vs downstream F1 ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Position Error vs Downstream Classification F1 (RF)",
                 fontsize=12, fontweight="bold")

    for ax, (task, label_y) in zip(axes, [
        ("zone",    "Zone F1-macro (RF)"),
        ("density", "Density F1-macro (RF)"),
    ]):
        xs_s, ys_s, lbls_s = [], [], []
        for a in agg_list:
            dr = ds_map.get(a.condition_name)
            tr = getattr(dr, f"{task}_rf", None) if dr else None
            if tr is None or np.isnan(tr.mean_f1_macro):
                continue
            xs_s.append(a.mean_pos_err_cm)
            ys_s.append(tr.mean_f1_macro)
            lbls_s.append(a.condition_name)

        scatter_cols = [COLORS[i % len(COLORS)] for i in range(len(xs_s))]
        ax.scatter(xs_s, ys_s, s=90, color=scatter_cols, zorder=5, alpha=0.9)
        for px, py, lb in zip(xs_s, ys_s, lbls_s):
            ax.annotate(lb, (px, py), xytext=(5, 4),
                        textcoords="offset points", fontsize=6.5, alpha=0.85)
        ax.set_xlabel("Mean Position Error (cm)", fontsize=10)
        ax.set_ylabel(label_y, fontsize=10)
        ax.set_title(f"Pos. Error vs {label_y}", fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = fig_dir / "scatter_geo_vs_downstream.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Figure: %s", out)


def _heatmap(row_labels, col_labels, data, title, path, cmap="RdYlGn_r"):
    fig, ax = plt.subplots(
        figsize=(max(10, len(col_labels) * 1.3), max(5, len(row_labels) * 0.40))
    )
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=38, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    norm = plt.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = data[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "N/A"
            nv  = norm(v) if not np.isnan(v) else 0.5
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if (nv < 0.25 or nv > 0.75) else "black")
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Heatmap: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Downstream cache reload helper
# ─────────────────────────────────────────────────────────────────────────────

def _downstream_from_flat(d: dict, condition_name: str) -> DownstreamResult:
    """Reconstruct a DownstreamResult from a flat cached-metrics dict."""
    dr = DownstreamResult(condition_name=condition_name)

    def _make(task, ml_lower, ml_upper) -> TaskResult | None:
        acc = d.get(f"{task}_{ml_lower}_acc", float("nan"))
        f1  = d.get(f"{task}_{ml_lower}_f1macro", float("nan"))
        auc = d.get(f"{task}_{ml_lower}_auc", float("nan"))
        if np.isnan(acc) and np.isnan(f1):
            return None
        return TaskResult(
            task=task, model=ml_upper, condition_name=condition_name,
            n_samples=0, n_classes=0,
            mean_accuracy=float(acc), std_accuracy=0.0,
            mean_f1_macro=float(f1), std_f1_macro=0.0,
            mean_f1_weighted=float(f1),
            mean_auc=float(auc),
        )

    dr.zone_mlp    = _make("zone",    "mlp", "MLP")
    dr.zone_rf     = _make("zone",    "rf",  "RF")
    dr.density_mlp = _make("density", "mlp", "MLP")
    dr.density_rf  = _make("density", "rf",  "RF")
    return dr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cuboid reconstruction ablation on MultiviewX"
    )
    p.add_argument("--dataset-root", type=str,
                   default=os.path.expanduser("~/store/data/raw/MultiviewX"))
    p.add_argument("--output-dir",   type=str, default="logs/cuboid_ablation")
    p.add_argument("--frames",       type=str, default="all",
                   help="'all', 'test' (last 40 frames), or '0-39'")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--no-mlflow",    action="store_true")
    p.add_argument("--mlflow-uri",   type=str,
                   default="sqlite:///logs/mlflow/mlflow.db")
    p.add_argument("--sanity-only",  action="store_true")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_mlflow = _MLFLOW and not args.no_mlflow

    # ── Calibration ──────────────────────────────────────────────────────
    logger.info("Loading calibration from: %s", args.dataset_root)
    calib = load_calibration(args.dataset_root, to_cm=True)
    for cam in calib.cam_names:
        C = calib.cam_centres[cam]
        logger.info("  %s  centre=(%.0f, %.0f, %.0f) cm", cam, C[0], C[1], C[2])

    # ── Frame range ──────────────────────────────────────────────────────
    all_frames = list_frame_indices(args.dataset_root)
    logger.info("Total annotation frames found: %d", len(all_frames))
    if args.frames == "all":
        frame_indices = all_frames
    elif args.frames == "test":
        frame_indices = all_frames[-40:]
    else:
        lo, hi = args.frames.split("-")
        frame_indices = [f for f in all_frames if int(lo) <= f <= int(hi)]
    logger.info("Running on %d frames.", len(frame_indices))

    # ── Label bank (built once, cached) ──────────────────────────────────
    label_bank_cache = output_dir / "label_bank.csv"
    logger.info("Building / loading downstream label bank …")
    label_bank = DownstreamLabelBank.build(
        args.dataset_root,
        seed=args.seed,
        cache_path=label_bank_cache,
    )
    logger.info(
        "Label bank: %d records  zones=%s  density=%s",
        label_bank.n_persons(),
        label_bank.zone_counts(),
        label_bank.density_counts(),
    )

    # ── Conditions ───────────────────────────────────────────────────────
    conditions = build_ofat_conditions()
    logger.info("Ablation conditions: %d total", len(conditions))
    for c in conditions:
        logger.info("  %-40s [%s]", c.name, c.dimension)

    # ══════════════════════════════════════════════════════════════════════
    # SANITY CHECK — 2 frames, baseline
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 62)
    logger.info("SANITY CHECK — baseline condition on 2 frames")
    logger.info("=" * 62)

    baseline_cond = conditions[0]
    sanity_ok     = True

    for fi in frame_indices[:2]:
        try:
            persons = load_frame_annotations(
                args.dataset_root, fi, min_cameras_visible=2
            )
            logger.info("  Frame %04d: %d persons visible in ≥2 cams", fi, len(persons))
            if not persons:
                logger.warning("  Frame %04d: no persons — skipping.", fi)
                continue

            fig_results: list[FrameResult] = []
            for person in persons[:8]:
                bbox_dict = {
                    cam: bb for cam in calib.cam_names
                    if (bb := person.bboxes.get(cam)) is not None
                }
                if len(bbox_dict) < 2:
                    logger.warning("    p%d: only %d cams — skip",
                                   person.person_id, len(bbox_dict))
                    continue
                try:
                    res = _reconstruct_with_ablation(
                        calib, list(bbox_dict.keys()), bbox_dict,
                        baseline_cond,
                        np.random.default_rng(args.seed),
                    )
                    pos_err = np.hypot(
                        res.cx - person.world_x_cm,
                        res.cy - person.world_y_cm,
                    )
                    logger.info(
                        "    p%d  GT=(%.0f,%.0f)  Recon=(%.0f,%.0f)  "
                        "err=%.1fcm  rms=%.1fpx",
                        person.person_id,
                        person.world_x_cm, person.world_y_cm,
                        res.cx, res.cy, pos_err, res.rms_px,
                    )
                    fig_results.append(make_frame_result(
                        fi, person.person_id, "sanity",
                        res.cx, res.cy, res.d, res.h, res.rms_px,
                        len(bbox_dict),
                        person.world_x_cm, person.world_y_cm,
                        PERSON_HEIGHT_CM, PERSON_DEPTH_CM,
                    ))
                except Exception as exc:
                    logger.warning("    p%d FAILED: %s", person.person_id, exc)
                    fig_results.append(make_failed_result(
                        fi, person.person_id, "sanity", len(bbox_dict),
                        person.world_x_cm, person.world_y_cm, str(exc)[:80],
                    ))

            save_sanity_figure(fi, persons[:8], fig_results, output_dir)

        except Exception as exc:
            logger.error("SANITY frame %04d FAILED: %s", fi, exc, exc_info=True)
            sanity_ok = False

    if not sanity_ok:
        logger.error("Sanity check failed — fix errors above before the full run.")
        sys.exit(1)

    logger.info("Sanity check PASSED.  Figures in: %s/sanity/", output_dir)
    if args.sanity_only:
        logger.info("--sanity-only — exiting.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # FULL ABLATION
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 62)
    logger.info("FULL ABLATION — %d conditions × %d frames",
                len(conditions), len(frame_indices))
    logger.info("=" * 62)

    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("cuboid_ablation_multiviewx")

    all_agg: list[AggregateMetrics]  = []
    all_ds:  list[DownstreamResult]  = []
    ds_ckpt_dir = output_dir / "downstream"
    ds_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ci, condition in enumerate(conditions):
        logger.info(
            "[%d/%d]  %-42s  dim=%s",
            ci + 1, len(conditions), condition.name, condition.dimension,
        )
        t0 = time.perf_counter()

        if use_mlflow:
            mlflow.start_run(run_name=condition.name)
            mlflow.log_params(condition.mlflow_params())
            mlflow.log_params({
                "dataset":  "MultiviewX",
                "n_frames": len(frame_indices),
                "seed":     args.seed,
            })

        # ── Geometry ──────────────────────────────────────────────────────
        results = evaluate_condition(
            condition=condition,
            calib=calib,
            frame_indices=frame_indices,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            seed=args.seed,
        )
        agg = aggregate(results, condition.name)
        all_agg.append(agg)

        # ── Downstream classifiers ─────────────────────────────────────────
        ds_cache = ds_ckpt_dir / f"{condition.name}_ds.csv"
        if ds_cache.exists() and ds_cache.stat().st_size > 0:
            logger.info("  [SKIP downstream] %s — cache hit.", condition.name)
            ds_df  = pd.read_csv(ds_cache)
            ds_row = ds_df.iloc[0].to_dict() if len(ds_df) > 0 else {}
            ds_res = _downstream_from_flat(ds_row, condition.name)
        else:
            results_df = pd.read_csv(
                output_dir / "checkpoints" / f"{condition.name}.csv"
            )
            ds_res = evaluate_downstream(
                results_df=results_df,
                label_bank=label_bank,
                condition_name=condition.name,
                seed=args.seed,
                n_cv_splits=5,
            )
            pd.DataFrame([ds_res.as_flat_dict()]).to_csv(ds_cache, index=False)

        all_ds.append(ds_res)

        elapsed = time.perf_counter() - t0
        z_f1  = ds_res.zone_rf.mean_f1_macro    if ds_res.zone_rf    else float("nan")
        d_f1  = ds_res.density_rf.mean_f1_macro  if ds_res.density_rf else float("nan")
        d_auc = ds_res.density_rf.mean_auc        if ds_res.density_rf else float("nan")
        logger.info(
            "  → pos_err=%.1fcm  SR@50=%.2f  rms=%.1fpx  fail=%.2f  "
            "zone_F1(RF)=%.3f  den_F1(RF)=%.3f  den_AUC=%.3f  [%.0fs]",
            agg.mean_pos_err_cm, agg.success_rate_50, agg.mean_rms_px,
            agg.failure_rate, z_f1, d_f1, d_auc, elapsed,
        )

        if use_mlflow:
            mlflow.log_metrics(agg.as_mlflow_metrics())
            mlflow.log_metrics(ds_res.as_mlflow_metrics())
            mlflow.log_metric("runtime_s", elapsed)
            mlflow.end_run()

    # ── Save result tables ────────────────────────────────────────────────
    geo_path = output_dir / "ablation_results.csv"
    pd.DataFrame([a.as_dict() for a in all_agg]).to_csv(geo_path, index=False)
    logger.info("Geometry results saved: %s", geo_path)

    ds_path = output_dir / "downstream_results.csv"
    pd.DataFrame([d.as_flat_dict() for d in all_ds]).to_csv(ds_path, index=False)
    logger.info("Downstream results saved: %s", ds_path)

    # ── Figures ───────────────────────────────────────────────────────────
    logger.info("Generating summary figures …")
    save_ablation_figures(all_agg, all_ds, conditions, output_dir)

    logger.info("=" * 62)
    logger.info("ABLATION COMPLETE — output: %s", output_dir)
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
