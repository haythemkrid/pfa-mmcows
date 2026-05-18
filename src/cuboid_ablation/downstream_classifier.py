"""
src.cuboid_ablation.downstream_classifier
==========================================

Trains lightweight classifiers on cuboid reconstruction features and evaluates
them on both downstream tasks defined in ``downstream_labels.py``.

Feature set (derived entirely from the reconstruction output)
-------------------------------------------------------------
  cx, cy                – reconstructed ground-plane centre (cm)
  d_cm                  – reconstructed depth (cm)
  h_cm                  – reconstructed height (cm)
  rms_px                – per-side RMS pixel residual
  n_cams_used           – number of cameras contributing
  h_to_d_ratio          – h / d   (posture proxy)
  cx_norm, cy_norm      – cx/cy normalised to [0,1] within scene bounds
  rms_bin               – rms_px binned into {good<50, ok<100, poor>=100}

The same feature set is used for BOTH tasks (zone and density).  This is
intentional: the ablation measures how reconstruction quality (which varies
across conditions) affects downstream accuracy.

Models
------
  MLP     – sklearn MLPClassifier: 2 hidden layers (256, 128), ReLU, Adam,
             early stopping.  Fast enough for 400 frames.
  RF      – RandomForestClassifier: 200 trees, balanced class weight.

Evaluation protocol
-------------------
  5-fold stratified cross-validation on the (frame, person) pairs available
  for a given condition.  Folds are fixed by frame index (temporal) to avoid
  leaking spatial patterns.

  Metrics reported:
    accuracy, macro-F1, per-class F1 (where applicable)

  For the density task (binary), also: precision, recall, AUC-ROC.

All results are returned as a ``DownstreamResult`` dataclass that is attached
to the ablation condition's ``AggregateMetrics``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from .downstream_labels import DownstreamLabelBank

logger = logging.getLogger(__name__)

# ── Scene bounds for normalisation (MultiviewX world, cm) ─────────────────────
# Grid: 640×1000 cells at 0.025 m/cell → 1600 cm × 2500 cm
SCENE_X_MAX_CM = 1600.0
SCENE_Y_MAX_CM = 2500.0


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(results_df: pd.DataFrame) -> np.ndarray:
    """Build the feature matrix from reconstruction results.

    Parameters
    ----------
    results_df : DataFrame with columns as produced by FrameResult / checkpoint CSV.
                 Must contain: cx, cy, d_cm, h_cm, rms_px, n_cams_used.
                 Failed rows (failed=True or NaN cx) should be filtered BEFORE
                 calling this function.

    Returns
    -------
    X : (N, F) float32 feature matrix
    """
    df = results_df.copy()

    # Clip to avoid divide-by-zero on degenerate reconstructions
    d    = np.clip(df["d_cm"].values,   1.0, None).astype(np.float32)
    h    = np.clip(df["h_cm"].values,   1.0, None).astype(np.float32)
    cx   = df["cx"].values.astype(np.float32)
    cy   = df["cy"].values.astype(np.float32)
    rms  = np.clip(df["rms_px"].values, 0.0, None).astype(np.float32)
    ncam = df["n_cams_used"].values.astype(np.float32)

    cx_norm = np.clip(cx / SCENE_X_MAX_CM, 0.0, 1.0)
    cy_norm = np.clip(cy / SCENE_Y_MAX_CM, 0.0, 1.0)
    h_to_d  = h / d
    rms_bin = np.where(rms < 50, 0.0, np.where(rms < 100, 1.0, 2.0))

    X = np.stack([
        cx, cy,
        d, h,
        rms,
        ncam,
        h_to_d,
        cx_norm, cy_norm,
        rms_bin,
    ], axis=1)

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """CV results for one task (zone or density) with one model."""

    task: str           # "zone" | "density"
    model: str          # "MLP" | "RF"
    condition_name: str

    n_samples: int
    n_classes: int

    mean_accuracy: float
    std_accuracy: float
    mean_f1_macro: float
    std_f1_macro: float
    mean_f1_weighted: float

    # Density-only extras (filled when task=="density")
    mean_auc: float = float("nan")
    mean_precision: float = float("nan")
    mean_recall: float = float("nan")

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def as_mlflow_metrics(self) -> dict:
        prefix = f"downstream/{self.task}/{self.model}"
        return {
            f"{prefix}/accuracy":     self.mean_accuracy,
            f"{prefix}/f1_macro":     self.mean_f1_macro,
            f"{prefix}/f1_weighted":  self.mean_f1_weighted,
            f"{prefix}/auc":          self.mean_auc,
        }


@dataclass
class DownstreamResult:
    """All downstream task results for one ablation condition."""

    condition_name: str
    zone_mlp:     Optional[TaskResult] = None
    zone_rf:      Optional[TaskResult] = None
    density_mlp:  Optional[TaskResult] = None
    density_rf:   Optional[TaskResult] = None

    def all_task_results(self) -> list[TaskResult]:
        return [r for r in [self.zone_mlp, self.zone_rf,
                             self.density_mlp, self.density_rf]
                if r is not None]

    def as_flat_dict(self) -> dict:
        d = {"condition_name": self.condition_name}
        for r in self.all_task_results():
            prefix = f"{r.task}_{r.model}"
            d[f"{prefix}_acc"]      = r.mean_accuracy
            d[f"{prefix}_f1macro"]  = r.mean_f1_macro
            d[f"{prefix}_auc"]      = r.mean_auc
        return d

    def as_mlflow_metrics(self) -> dict:
        metrics = {}
        for r in self.all_task_results():
            metrics.update(r.as_mlflow_metrics())
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task: str,
    condition_name: str,
    n_splits: int = 5,
    seed: int = 42,
) -> TaskResult:
    """Run stratified K-fold CV and return aggregated metrics."""

    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, f1_macros, f1_weighted_list = [], [], []
    aucs, precs, recs = [], [], []

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Standardise inside fold to avoid leakage
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Instantiate fresh model each fold
        if model_name == "MLP":
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
                learning_rate_init=1e-3,
            )
        elif model_name == "RF":
            model = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model: {model_name!r}")

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        accs.append(accuracy_score(y_te, y_pred))
        f1_macros.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
        f1_weighted_list.append(f1_score(y_te, y_pred, average="weighted", zero_division=0))

        if task == "density" and n_classes == 2:
            proba = model.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
            precs.append(precision_score(y_te, y_pred, zero_division=0))
            recs.append(recall_score(y_te, y_pred, zero_division=0))

    def _m(lst): return float(np.mean(lst)) if lst else float("nan")
    def _s(lst): return float(np.std(lst))  if lst else float("nan")

    return TaskResult(
        task=task,
        model=model_name,
        condition_name=condition_name,
        n_samples=len(y),
        n_classes=n_classes,
        mean_accuracy=_m(accs),
        std_accuracy=_s(accs),
        mean_f1_macro=_m(f1_macros),
        std_f1_macro=_s(f1_macros),
        mean_f1_weighted=_m(f1_weighted_list),
        mean_auc=_m(aucs),
        mean_precision=_m(precs),
        mean_recall=_m(recs),
    )


def evaluate_downstream(
    results_df: pd.DataFrame,
    label_bank: DownstreamLabelBank,
    condition_name: str,
    seed: int = 42,
    n_cv_splits: int = 5,
) -> DownstreamResult:
    """Train and evaluate classifiers for one ablation condition.

    Parameters
    ----------
    results_df     : DataFrame of FrameResult rows for this condition.
                     Loaded from the condition's checkpoint CSV.
    label_bank     : Pre-built DownstreamLabelBank with GT zone/density labels.
    condition_name : Name string for this condition.
    seed           : RNG seed.
    n_cv_splits    : Number of CV folds.

    Returns
    -------
    DownstreamResult with all four (task × model) TaskResult objects.
    """
    # ── Filter failed / NaN reconstructions ──────────────────────────────
    valid = results_df[
        (results_df["failed"] == False) &
        results_df["cx"].notna() &
        results_df["cy"].notna()
    ].copy()

    if len(valid) == 0:
        logger.warning("Condition %s: no valid reconstructions for downstream eval.",
                       condition_name)
        return DownstreamResult(condition_name=condition_name)

    # ── Attach labels ─────────────────────────────────────────────────────
    zone_rows, density_rows = [], []

    for _, row in valid.iterrows():
        fi  = int(row["frame_idx"])
        pid = int(row["person_id"])
        lb  = label_bank.get(fi, pid)
        if lb is None:
            continue
        base = row.to_dict()
        zone_rows.append({**base,
                          "zone_label": lb.zone_label,
                          "density_label": lb.density_label})
        density_rows.append({**base,
                             "zone_label": lb.zone_label,
                             "density_label": lb.density_label})

    if len(zone_rows) == 0:
        logger.warning("Condition %s: no label matches found.", condition_name)
        return DownstreamResult(condition_name=condition_name)

    df_zone    = pd.DataFrame(zone_rows)
    df_density = pd.DataFrame(density_rows)

    # ── Features ──────────────────────────────────────────────────────────
    X_zone    = build_features(df_zone)
    y_zone    = df_zone["zone_label"].values.astype(int)

    X_density = build_features(df_density)
    y_density = df_density["density_label"].values.astype(int)

    # ── Guard: need enough samples per class for CV ────────────────────────
    def _enough(y, n_splits):
        counts = np.bincount(y)
        return (counts >= n_splits).all() and len(y) >= n_splits * 2

    downstream = DownstreamResult(condition_name=condition_name)

    # ── Zone task ─────────────────────────────────────────────────────────
    if _enough(y_zone, n_cv_splits):
        logger.info("  [downstream] %s — zone task: %d samples, %d classes",
                    condition_name, len(y_zone), len(np.unique(y_zone)))
        for model_name in ("MLP", "RF"):
            tr = _cv_evaluate(X_zone, y_zone, model_name, "zone",
                              condition_name, n_cv_splits, seed)
            logger.info(
                "    %s zone/%s  acc=%.3f±%.3f  f1_macro=%.3f",
                condition_name, model_name, tr.mean_accuracy, tr.std_accuracy, tr.mean_f1_macro,
            )
            if model_name == "MLP":
                downstream.zone_mlp = tr
            else:
                downstream.zone_rf = tr
    else:
        logger.warning("  [downstream] %s — zone: insufficient samples, skipping.", condition_name)

    # ── Density task ──────────────────────────────────────────────────────
    if _enough(y_density, n_cv_splits):
        logger.info("  [downstream] %s — density task: %d samples (isolated=%d, crowded=%d)",
                    condition_name, len(y_density),
                    (y_density == 0).sum(), (y_density == 1).sum())
        for model_name in ("MLP", "RF"):
            tr = _cv_evaluate(X_density, y_density, model_name, "density",
                              condition_name, n_cv_splits, seed)
            logger.info(
                "    %s density/%s  acc=%.3f±%.3f  f1_macro=%.3f  auc=%.3f",
                condition_name, model_name, tr.mean_accuracy, tr.std_accuracy,
                tr.mean_f1_macro, tr.mean_auc,
            )
            if model_name == "MLP":
                downstream.density_mlp = tr
            else:
                downstream.density_rf = tr
    else:
        logger.warning("  [downstream] %s — density: insufficient samples, skipping.",
                       condition_name)

    return downstream
