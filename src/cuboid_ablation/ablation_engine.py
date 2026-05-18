"""
src.cuboid_ablation.ablation_engine
=====================================

Defines every ablation dimension applied to the cuboid reconstruction method.

Ablation dimensions
-------------------
A  NUM_CAMERAS         – how many cameras are used (2 → 6).
                         Camera subsets are chosen by a greedy max-baseline
                         heuristic (most spatially diverse first) so the
                         comparison is fair, not random.

B  BBOX_NOISE          – additive Gaussian noise on every bbox edge
                         (σ ∈ {0, 5, 10, 20, 40} px).

C  GROUND_CONTACT      – where on the bbox bottom edge the ground ray is cast:
                         "bottom_centre"  (default),
                         "bottom_left", "bottom_right"  (±25% of width),
                         "midpoint"       (vertical centre of bbox).

D  PRIOR_SENSITIVITY   – the width prior w₀ is varied ±{0%, ±25%, ±50%} of
                         the GT bounding-box width estimate.  depth/height
                         priors are similarly perturbed independently.

E  LSE_ALPHA           – LogSumExp sharpness α ∈ {5, 10, 20, 50, 100}.
                         Default is 20 (px scale).

F  WARM_START_QUALITY  – compare the ground-plane back-projection warm-start
                         (default) vs. a naive origin warm-start (cx=0, cy=0).

Each dimension is independent; the baseline uses:
   all cameras,  σ_noise=0,  bottom_centre,  nominal priors,  α=20,
   ground-plane warm-start.

The full factorial would be huge, so we run a one-factor-at-a-time (OFAT)
ablation: vary each dimension while keeping all others at baseline.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .multiviewx_loader import PersonAnnotation, MultiviewXCalib

logger = logging.getLogger(__name__)

# ─── Baseline values ──────────────────────────────────────────────────────────
BASELINE_ALPHA = 20.0
BASELINE_GROUND_CONTACT = "bottom_centre"
BASELINE_NOISE_SIGMA = 0.0
BASELINE_WARM_START = "backproject"
# priors for pedestrians in MultiviewX (PersonX models, metre-scale scene)
# PersonX: ~1.75 m tall, ~0.5 m wide, ~0.3 m deep (shoulder width)
# In cm:
PERSON_HEIGHT_CM = 175.0
PERSON_WIDTH_CM  = 50.0
PERSON_DEPTH_CM  = 30.0

# ─── Ablation grids ──────────────────────────────────────────────────────────
ABLATION_NUM_CAMERAS    = [2, 3, 4, 5, 6]
ABLATION_NOISE_SIGMA    = [0.0, 5.0, 10.0, 20.0, 40.0]
ABLATION_GROUND_CONTACT = ["bottom_centre", "bottom_left", "bottom_right", "midpoint"]
ABLATION_PRIOR_FACTOR   = [0.5, 0.75, 1.0, 1.25, 1.5]   # multiplicative on w₀/d₀/h₀
ABLATION_ALPHA          = [5.0, 10.0, 20.0, 50.0, 100.0]
ABLATION_WARM_START     = ["backproject", "origin"]


# ─────────────────────────────────────────────────────────────────────────────
# Camera subset selection
# ─────────────────────────────────────────────────────────────────────────────

def _camera_baseline(cam_centres: dict[str, np.ndarray]) -> float:
    """Return maximum pairwise distance between camera centres (a diversity score)."""
    names = list(cam_centres.keys())
    centres = np.stack([cam_centres[n] for n in names])
    max_d = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            max_d = max(max_d, np.linalg.norm(centres[i] - centres[j]))
    return max_d


def select_diverse_cameras(
    all_cam_names: list[str],
    cam_centres: dict[str, np.ndarray],
    n: int,
) -> list[str]:
    """Greedy max-diversity camera subset of size n.

    Starts with the pair that has the largest baseline, then at each step
    adds the camera that maximises the *minimum* distance to all already-
    selected cameras (max-min criterion, a standard diversity heuristic).

    Returns
    -------
    list of n camera names (deterministic, reproducible).
    """
    if n >= len(all_cam_names):
        return list(all_cam_names)

    centres = {c: cam_centres[c] for c in all_cam_names}

    # Seed: pair with largest baseline
    best_pair, best_dist = None, -1.0
    for i, ci in enumerate(all_cam_names):
        for cj in all_cam_names[i + 1:]:
            d = np.linalg.norm(centres[ci] - centres[cj])
            if d > best_dist:
                best_dist, best_pair = d, [ci, cj]

    selected = list(best_pair)
    while len(selected) < n:
        remaining = [c for c in all_cam_names if c not in selected]
        sel_centres = np.stack([centres[c] for c in selected])
        best_cam, best_score = None, -1.0
        for c in remaining:
            cc = centres[c]
            min_d = float(np.min(np.linalg.norm(sel_centres - cc, axis=1)))
            if min_d > best_score:
                best_score, best_cam = min_d, c
        selected.append(best_cam)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box perturbations
# ─────────────────────────────────────────────────────────────────────────────

def apply_bbox_noise(
    bbox: tuple[int, int, int, int],
    sigma: float,
    rng: np.random.Generator,
    img_h: int = 1080,
    img_w: int = 1920,
) -> tuple[float, float, float, float]:
    """Add independent Gaussian noise to each edge of the bbox.

    Parameters
    ----------
    bbox  : (x1, y1, x2, y2)
    sigma : std dev in pixels
    rng   : numpy random generator (seeded externally for reproducibility)

    Returns
    -------
    Noisy (x1, y1, x2, y2) clipped to image bounds.
    The result is float so downstream code (the reconstructor) sees real values.
    """
    if sigma <= 0.0:
        return tuple(float(v) for v in bbox)

    x1, y1, x2, y2 = bbox
    noise = rng.normal(0.0, sigma, 4)
    x1n = float(np.clip(x1 + noise[0], 0, img_w - 1))
    y1n = float(np.clip(y1 + noise[1], 0, img_h - 1))
    x2n = float(np.clip(x2 + noise[2], 0, img_w - 1))
    y2n = float(np.clip(y2 + noise[3], 0, img_h - 1))

    # Guarantee x2 > x1 and y2 > y1 even after noise
    if x2n <= x1n:
        x2n = x1n + 1.0
    if y2n <= y1n:
        y2n = y1n + 1.0

    return (x1n, y1n, x2n, y2n)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-contact point perturbation
# ─────────────────────────────────────────────────────────────────────────────

def ground_contact_point(
    bbox: tuple,
    mode: str,
) -> tuple[float, float]:
    """Return the (u_c, v_c) pixel used as the ground-contact back-projection.

    Parameters
    ----------
    bbox : (x1, y1, x2, y2) — possibly float after noise
    mode : one of
        "bottom_centre"  – midpoint of bottom edge  (default)
        "bottom_left"    – 25% from left on bottom edge
        "bottom_right"   – 75% from left on bottom edge
        "midpoint"       – centre of the full bounding box (height midpoint)

    Returns
    -------
    (u_c, v_c) in pixels
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = x2 - x1
    if mode == "bottom_centre":
        return (x1 + w / 2, y2)
    elif mode == "bottom_left":
        return (x1 + w * 0.25, y2)
    elif mode == "bottom_right":
        return (x1 + w * 0.75, y2)
    elif mode == "midpoint":
        return (x1 + w / 2, (y1 + y2) / 2)
    else:
        raise ValueError(f"Unknown ground_contact mode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation condition descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationCondition:
    """One row in the ablation table.

    Parameters
    ----------
    name          : human-readable label for MLflow / CSV
    dimension     : which ablation axis (A–F)
    num_cameras   : how many cameras to use
    noise_sigma   : std dev of bbox noise (px)
    ground_contact: ground contact mode string
    prior_factor  : multiplicative factor on w₀, d₀, h₀
    alpha         : LogSumExp sharpness
    warm_start    : "backproject" | "origin"
    """

    name: str
    dimension: str

    num_cameras: int       = 6
    noise_sigma: float     = 0.0
    ground_contact: str    = BASELINE_GROUND_CONTACT
    prior_factor: float    = 1.0
    alpha: float           = BASELINE_ALPHA
    warm_start: str        = BASELINE_WARM_START

    def mlflow_params(self) -> dict:
        return {
            "condition/name":           self.name,
            "condition/dimension":      self.dimension,
            "condition/num_cameras":    self.num_cameras,
            "condition/noise_sigma_px": self.noise_sigma,
            "condition/ground_contact": self.ground_contact,
            "condition/prior_factor":   self.prior_factor,
            "condition/alpha":          self.alpha,
            "condition/warm_start":     self.warm_start,
        }


def build_ofat_conditions() -> list[AblationCondition]:
    """Build the full one-factor-at-a-time (OFAT) ablation grid.

    Returns a list of AblationCondition objects, including the baseline.
    The baseline appears exactly once (first entry).
    """
    conditions: list[AblationCondition] = []

    # ── Baseline ─────────────────────────────────────────────────────────────
    baseline = AblationCondition(
        name="baseline",
        dimension="baseline",
        num_cameras=6,
        noise_sigma=0.0,
        ground_contact="bottom_centre",
        prior_factor=1.0,
        alpha=20.0,
        warm_start="backproject",
    )
    conditions.append(baseline)

    # ── A: Number of cameras ──────────────────────────────────────────────────
    for nc in ABLATION_NUM_CAMERAS:
        if nc == 6:
            continue   # that's the baseline
        conditions.append(AblationCondition(
            name=f"num_cams_{nc}",
            dimension="A_num_cameras",
            num_cameras=nc,
        ))

    # ── B: BBox noise ─────────────────────────────────────────────────────────
    for sigma in ABLATION_NOISE_SIGMA:
        if sigma == 0.0:
            continue
        conditions.append(AblationCondition(
            name=f"bbox_noise_sigma{int(sigma)}px",
            dimension="B_bbox_noise",
            noise_sigma=sigma,
        ))

    # ── C: Ground-contact point ───────────────────────────────────────────────
    for gc in ABLATION_GROUND_CONTACT:
        if gc == BASELINE_GROUND_CONTACT:
            continue
        conditions.append(AblationCondition(
            name=f"ground_contact_{gc}",
            dimension="C_ground_contact",
            ground_contact=gc,
        ))

    # ── D: Prior sensitivity ──────────────────────────────────────────────────
    for pf in ABLATION_PRIOR_FACTOR:
        if pf == 1.0:
            continue
        conditions.append(AblationCondition(
            name=f"prior_factor_{pf:.2f}",
            dimension="D_prior_sensitivity",
            prior_factor=pf,
        ))

    # ── E: LSE alpha ──────────────────────────────────────────────────────────
    for alpha in ABLATION_ALPHA:
        if alpha == BASELINE_ALPHA:
            continue
        conditions.append(AblationCondition(
            name=f"alpha_{int(alpha)}",
            dimension="E_lse_alpha",
            alpha=alpha,
        ))

    # ── F: Warm-start quality ─────────────────────────────────────────────────
    conditions.append(AblationCondition(
        name="warmstart_origin",
        dimension="F_warm_start",
        warm_start="origin",
    ))

    return conditions
