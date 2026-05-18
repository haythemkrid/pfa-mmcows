"""
MmCows — Multi-View Cuboid Reconstruction Engine (v3)
======================================================

Fits a 3-D axis-aligned-bottom cuboid to per-camera pixel bounding boxes
using a two-stage optimiser (L-BFGS-B → Nelder-Mead polish) with an optional
rotated-restart fallback.

Design choices
--------------
- **World unit = cm** (camera centres are ~1190 cm from pen centre).
- **Pixel-space cost** — vertices are projected through ``P`` directly;
  no UV conversion needed.
- **Width fixed** (``prior_w``) — lateral width is ambiguous from 4 corner
  cameras looking roughly along the body axis.
- **5-DOF optimisation**: ``(cx, cy, depth, height, θ)``.
- **Batched projection**: ``(N, 3, 4) @ (4, 8)`` matrix multiply covers all
  cameras simultaneously, eliminating Python loops in the hot path.

Typical runtime: ~100 ms per frame with a good warm start.

Usage example
-------------
::

    from mmcows.data.visual_loaders import VisualDataConfig, VisualLoader
    from mmcows.pipelines.cuboid_reconstruction import (
        CuboidPriors,
        CuboidReconstructor,
    )

    cfg     = VisualDataConfig(dataset_root="/data/mmcows", date="0725")
    loader  = VisualLoader(cfg)
    proj_mats, cam_centres = loader.load_projection_matrices()

    priors      = CuboidPriors()           # default Holstein cow priors
    reconstructor = CuboidReconstructor(
        proj_mats=proj_mats,
        cam_centres=cam_centres,
        priors=priors,
    )

    labels = loader.load_labels("1690271846")
    bboxes = {
        cam: loader.yolo_to_pixel_bbox(
            labels[cam].iloc[0].x_c,
            labels[cam].iloc[0].y_c,
            labels[cam].iloc[0].w,
            labels[cam].iloc[0].h,
        )
        for cam in labels
        if not labels[cam].empty
    }

    result = reconstructor.reconstruct(
        active_cams=list(bboxes.keys()),
        bbox_dict=bboxes,
    )
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ── Optimiser hyper-parameters ────────────────────────────────────────────────
# These match the v3 notebook settings.  Adjust only if you change the cost scale.
_ALPHA = 20.0  # LogSumExp sharpness (px scale; original sim used 400 for UV scale)
_LBFGS_OPTS = dict(maxiter=300, ftol=1e-15, gtol=1e-11)
_NM_OPTS = dict(maxiter=150, xatol=1.0, fatol=1.0, adaptive=True)

# Rotation offsets (rad) tried when the primary fit is poor.
_RESTART_ANGLES = [
    np.pi / 4, -np.pi / 4,
    np.pi / 6, -np.pi / 6,
    np.pi / 3, -np.pi / 3,
]


# ─────────────────────────────────────────────────────────────────────────────
# Priors dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CuboidPriors:
    """Species-level size priors for the cuboid warm-start (cm).

    Attributes
    ----------
    width:
        Lateral (left–right) width — kept **fixed** throughout optimisation
        because it is ambiguous from corner cameras looking along the body axis.
    depth:
        Fore-aft depth — used as the warm-start initial guess.
    height:
        Vertical height — used as the warm-start initial guess.
    restart_rms_threshold:
        Pixel RMS above which rotated restarts are triggered.  With a good
        warm start this fires rarely, keeping runtime ~100 ms.
    """

    width: float = 60.0
    depth: float = 180.0
    height: float = 140.0
    restart_rms_threshold: float = 200.0


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CuboidResult:
    """Output of a single cuboid reconstruction.

    All lengths in **cm**.

    Attributes
    ----------
    cx, cy:
        Ground-plane centre coordinates (cm).
    w:
        Lateral width (cm) — equal to ``priors.width``.
    d:
        Fore-aft depth (cm).
    h:
        Height (cm).
    rot_deg:
        Yaw angle in [0, 180) degrees.
    theta_rad:
        Raw optimised yaw angle (radians).
    residual_px2:
        Sum-of-squared pixel residuals across all camera sides.
    rms_px:
        RMS residual per bbox side (px).
    vertices:
        ``(8, 3)`` array of cuboid corner coordinates (cm).
        Vertices 0–3 are bottom; 4–7 are top.
    per_cam_deltas:
        List of ``[Δu_min, Δu_max, Δv_min, Δv_max]`` (px) per active camera.
    """

    cx: float
    cy: float
    w: float
    d: float
    h: float
    rot_deg: float
    theta_rad: float
    residual_px2: float
    rms_px: float
    vertices: np.ndarray
    per_cam_deltas: list[list[float]]


# ─────────────────────────────────────────────────────────────────────────────
# Reconstructor
# ─────────────────────────────────────────────────────────────────────────────


class CuboidReconstructor:
    """Multi-view cuboid reconstructor.

    Parameters
    ----------
    proj_mats:
        Dict ``{cam_name: (3, 4) projection matrix}``.
    cam_centres:
        Dict ``{cam_name: (3,) optical centre (cm)}``.
    priors:
        Size priors and optimiser thresholds.
    """

    def __init__(
        self,
        proj_mats: dict[str, np.ndarray],
        cam_centres: dict[str, np.ndarray],
        priors: Optional[CuboidPriors] = None,
    ) -> None:
        self.proj_mats = proj_mats
        self.cam_centres = cam_centres
        self.priors = priors or CuboidPriors()

    def reconstruct(
        self,
        active_cams: list[str],
        bbox_dict: dict[str, tuple[int, int, int, int]],
        fixed_width: Optional[float] = None,
    ) -> CuboidResult:
        """Fit a cuboid to the observed pixel bounding boxes.

        Parameters
        ----------
        active_cams:
            Ordered list of camera names that have observations for this frame.
            Must be a subset of the cameras passed to the constructor.
        bbox_dict:
            ``{cam: (x1, y1, x2, y2)}`` pixel bounding boxes.
        fixed_width:
            Override the species-prior lateral width (cm).  Defaults to
            ``priors.width``.

        Returns
        -------
        :class:`CuboidResult`

        Raises
        ------
        ValueError
            If fewer than 2 cameras are active.
        """
        if len(active_cams) < 2:
            raise ValueError(
                f"At least 2 active cameras are required, got {len(active_cams)}."
            )

        w = fixed_width if fixed_width is not None else self.priors.width

        P_stack = np.stack([self.proj_mats[c] for c in active_cams])    # (N,3,4)
        C0_arr = np.stack([self.cam_centres[c] for c in active_cams])   # (N,3)
        obs_arr = _build_obs_array(active_cams, bbox_dict)               # (N,4)
        N = len(active_cams)
        args = (P_stack, obs_arr, w)

        # ── Warm start ────────────────────────────────────────────────────
        x0 = _warm_start(P_stack, obs_arr, C0_arr, self.priors)
        cx0, cy0 = x0[0], x0[1]
        logger.debug("Warm-start: cx=%.0f cy=%.0f cm", cx0, cy0)

        bnds = [
            (cx0 - 600, cx0 + 600),
            (cy0 - 600, cy0 + 600),
            (40, 350),        # depth
            (50, 220),        # height
            (None, None),     # yaw (unbounded)
        ]

        # ── Stage 1: L-BFGS-B on smooth (LogSumExp) cost ─────────────────
        res = minimize(
            _smooth_cost, x0, args=args,
            method="L-BFGS-B", bounds=bnds, options=_LBFGS_OPTS,
        )
        hc = _hard_cost(res.x, *args)

        # ── Stage 2: Nelder-Mead polish on exact cost ─────────────────────
        r2 = minimize(
            _hard_cost, res.x, args=args,
            method="Nelder-Mead", options=_NM_OPTS,
        )
        if r2.fun < hc:
            res, hc = r2, r2.fun

        # ── Stage 3: rotated restarts (only when fit is very poor) ────────
        rms = np.sqrt(hc / (N * 4))
        if rms > self.priors.restart_rms_threshold:
            logger.debug(
                "RMS=%.0f px > threshold (%.0f px) — trying rotated restarts",
                rms, self.priors.restart_rms_threshold,
            )
            for dth in _RESTART_ANGLES:
                xi = x0.copy()
                xi[4] += dth
                rf = minimize(
                    _smooth_cost, xi, args=args,
                    method="L-BFGS-B", bounds=bnds, options=_LBFGS_OPTS,
                )
                hcf = _hard_cost(rf.x, *args)
                if hcf < hc:
                    res, hc = rf, hcf
                if np.sqrt(hc / (N * 4)) < self.priors.restart_rms_threshold:
                    break

        # ── Post-process ──────────────────────────────────────────────────
        return _build_result(res.x, hc, w, N, P_stack, obs_arr)


# ─────────────────────────────────────────────────────────────────────────────
# Cost functions  (private)
# ─────────────────────────────────────────────────────────────────────────────


def _build_obs_array(
    cam_list: list[str],
    bbox_dict: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    """Stack per-camera bboxes into ``(N, 4)`` array ``[u_min, u_max, v_min, v_max]``."""
    return np.array(
        [
            (bbox_dict[c][0], bbox_dict[c][2],  # u_min, u_max  (x1, x2)
             bbox_dict[c][1], bbox_dict[c][3])  # v_min, v_max  (y1, y2)
            for c in cam_list
        ],
        dtype=np.float64,
    )


def _cuboid_homogeneous(cx: float, cy: float, d: float, h: float, th: float, w: float):
    """Return ``(4, 8)`` homogeneous vertex matrix for the cuboid."""
    hw, hd = w / 2, d / 2
    c, s = np.cos(th), np.sin(th)
    sx = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=np.float64) * hw
    sy = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=np.float64) * hd
    sz = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64) * h
    return np.array(
        [c * sx - s * sy + cx,
         s * sx + c * sy + cy,
         sz,
         np.ones(8)],
    )


def _smooth_cost(
    x5: np.ndarray,
    P_stack: np.ndarray,
    obs_arr: np.ndarray,
    w: float,
) -> float:
    """Fully-batched differentiable LogSumExp cost.

    Smooth approximation to the hard bbox residual, enabling gradient-based
    optimisation via L-BFGS-B.

    Parameters
    ----------
    x5:
        ``[cx, cy, d, h, θ]`` — 5-DOF parameter vector.
    P_stack:
        ``(N, 3, 4)`` stacked projection matrices.
    obs_arr:
        ``(N, 4)`` observed ``[u_min, u_max, v_min, v_max]`` per camera.
    w:
        Fixed lateral width (cm).
    """
    cx, cy, d, h, th = x5
    if d < 5 or h < 5:
        return 1e12

    Xh = _cuboid_homogeneous(cx, cy, d, h, th, w)   # (4, 8)
    ph = P_stack @ Xh                                 # (N, 3, 8)
    pu = ph[:, 0, :] / ph[:, 2, :]                   # (N, 8)
    pv = ph[:, 1, :] / ph[:, 2, :]

    def _smax(a: np.ndarray) -> np.ndarray:
        """Row-wise soft maximum via LogSumExp."""
        m = a.max(1, keepdims=True)
        return np.log(np.exp(_ALPHA * (a - m)).sum(1)) / _ALPHA + m[:, 0]

    def _smin(a: np.ndarray) -> np.ndarray:
        return -_smax(-a)

    diff = np.stack(
        [
            _smin(pu) - obs_arr[:, 0],
            _smax(pu) - obs_arr[:, 1],
            _smin(pv) - obs_arr[:, 2],
            _smax(pv) - obs_arr[:, 3],
        ],
        axis=1,
    )  # (N, 4)
    return float((diff ** 2).sum())


def _hard_cost(
    x5: np.ndarray,
    P_stack: np.ndarray,
    obs_arr: np.ndarray,
    w: float,
) -> float:
    """Exact (non-smooth) pixel residual.

    Used for convergence checks and Nelder-Mead polish (which does not need
    gradients).
    """
    cx, cy, d, h, th = x5
    if d < 5 or h < 5:
        return 1e12

    Xh = _cuboid_homogeneous(cx, cy, d, h, th, w)
    ph = P_stack @ Xh
    pu = ph[:, 0, :] / ph[:, 2, :]
    pv = ph[:, 1, :] / ph[:, 2, :]
    diff = np.stack(
        [
            pu.min(1) - obs_arr[:, 0],
            pu.max(1) - obs_arr[:, 1],
            pv.min(1) - obs_arr[:, 2],
            pv.max(1) - obs_arr[:, 3],
        ],
        axis=1,
    )
    return float((diff ** 2).sum())


def _warm_start(
    P_stack: np.ndarray,
    obs_arr: np.ndarray,
    C0_arr: np.ndarray,
    priors: CuboidPriors,
) -> np.ndarray:
    """Estimate initial ``[cx, cy, depth, height, yaw]`` from back-projection.

    Back-projects the bottom-centre of each bounding box to the ground plane
    (z = 0) via the pseudo-inverse of P.  The median of the resulting ground
    points becomes ``(cx, cy)``; depth, height, and yaw are initialised from
    the species priors.
    """
    gpts: list[np.ndarray] = []
    for P, C0, (u_min, u_max, v_min, v_max) in zip(P_stack, C0_arr, obs_arr):
        Pp = P.T @ np.linalg.inv(P @ P.T)             # (4, 3) pseudo-inverse
        uc = (u_min + u_max) / 2.0
        vc = float(v_max)                              # bottom edge → near ground
        Xh = Pp @ np.array([uc, vc, 1.0])             # (4,) homogeneous point
        pt = Xh[:3] / Xh[3]
        ray = pt - C0
        ray /= np.linalg.norm(ray)
        if abs(ray[2]) < 1e-9:
            continue
        t = -C0[2] / ray[2]                           # intersect z = 0
        gp = C0 + t * ray
        if abs(gp[0]) < 5000 and abs(gp[1]) < 5000:
            gpts.append(gp[:2])

    if not gpts:
        logger.warning("Warm-start back-projection failed — using origin.")
        return np.array([0.0, 0.0, priors.depth, priors.height, 0.0])

    cx0, cy0 = np.median(np.array(gpts), axis=0)
    return np.array([cx0, cy0, priors.depth, priors.height, 0.0])


def _build_result(
    x5: np.ndarray,
    hc: float,
    w: float,
    N: int,
    P_stack: np.ndarray,
    obs_arr: np.ndarray,
) -> CuboidResult:
    """Assemble a :class:`CuboidResult` from optimiser output."""
    cx, cy, d, h, theta = x5
    d, h = abs(d), abs(h)

    hw, hd = w / 2, d / 2
    c, s = np.cos(theta), np.sin(theta)
    sx = np.array([-1, 1, 1, -1, -1, 1, 1, -1], float) * hw
    sy = np.array([-1, -1, 1, 1, -1, -1, 1, 1], float) * hd
    sz = np.array([0, 0, 0, 0, 1, 1, 1, 1], float) * h
    vertices = np.stack([c * sx - s * sy + cx, s * sx + c * sy + cy, sz], axis=1)

    Xh_v = np.vstack([vertices.T, np.ones((1, 8))])   # (4, 8)
    per_cam: list[list[float]] = []
    for P, (u1, u2, v1, v2) in zip(P_stack, obs_arr):
        ph = P @ Xh_v
        pu = ph[0, :] / ph[2, :]
        pv = ph[1, :] / ph[2, :]
        per_cam.append([
            float(pu.min() - u1), float(pu.max() - u2),
            float(pv.min() - v1), float(pv.max() - v2),
        ])

    rms_final = float(np.sqrt(hc / (N * 4)))

    return CuboidResult(
        cx=float(cx), cy=float(cy), w=float(w), d=float(d), h=float(h),
        rot_deg=float(np.degrees(theta) % 180),
        theta_rad=float(theta),
        residual_px2=float(hc),
        rms_px=rms_final,
        vertices=vertices,
        per_cam_deltas=per_cam,
    )
