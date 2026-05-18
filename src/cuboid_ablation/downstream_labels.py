"""
src.cuboid_ablation.downstream_labels
=======================================

Generates two sets of pseudo-labels from MultiviewX ground-truth annotations
for use in the downstream classification evaluation.

Task A — Spatial Zone Classification
--------------------------------------
  K-means clustering (K=8) on GT ground-plane positions (world_x_cm, world_y_cm)
  across the entire dataset.  Each pedestrian instance is assigned the zone of
  its GT position.  The reconstructor never sees the GT position; it only sees
  pixel bounding boxes.

  Rationale: if reconstruction is accurate, the recovered (cx, cy) should carry
  enough positional information to predict which spatial zone the person is in.
  Degraded reconstruction (noise, fewer cameras, wrong priors) should increase
  zone classification error.

Task B — Density Label (Binary)
---------------------------------
  For each (frame, person) pair, compute the distance to the nearest other
  person in that frame using GT ground positions.  If the distance is below a
  threshold (DENSITY_THRESHOLD_CM = 150 cm ≈ 1.5 m), the person is "crowded"
  (label=1), otherwise "isolated" (label=0).

  Rationale: a person that is reconstructed with large positional error is more
  likely to be misclassified as isolated/crowded because the apparent distance
  to neighbours is wrong.

Both labels are computed once from GT and stored.  They are then applied to
every reconstruction result by joining on (frame_idx, person_id).

Design
------
  - Labels are computed on the FULL dataset (all 400 frames) so that K-means
    zones cover the full scene extent.
  - Label assignment is deterministic given the seed.
  - The module exports a single ``DownstreamLabelBank`` object that the
    training module queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from .multiviewx_loader import (
    load_frame_annotations,
    list_frame_indices,
    PersonAnnotation,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
N_ZONES = 8                  # number of spatial zones
DENSITY_THRESHOLD_CM = 150.0 # nearest-neighbour distance threshold (cm)
MIN_CROWD_FRAMES = 2         # a frame needs ≥ this many persons for density to be meaningful


@dataclass
class PersonLabel:
    """Ground-truth labels for one (frame, person) pair."""
    frame_idx: int
    person_id: int
    world_x_cm: float
    world_y_cm: float
    zone_label: int           # 0 … N_ZONES-1
    density_label: int        # 0 = isolated, 1 = crowded
    nn_dist_cm: float         # distance to nearest neighbour (cm); inf if alone


class DownstreamLabelBank:
    """Pre-computed label bank for all (frame, person) pairs.

    Usage
    -----
    ::

        bank = DownstreamLabelBank.build(dataset_root, seed=42)
        zone   = bank.zone(frame_idx, person_id)
        density = bank.density(frame_idx, person_id)
        df = bank.as_dataframe()
    """

    def __init__(self, labels: list[PersonLabel], kmeans: KMeans):
        self._labels = labels
        self._kmeans = kmeans
        self._index: dict[tuple[int, int], PersonLabel] = {
            (l.frame_idx, l.person_id): l for l in labels
        }

    # ── Lookups ───────────────────────────────────────────────────────────────

    def zone(self, frame_idx: int, person_id: int) -> Optional[int]:
        key = (frame_idx, person_id)
        return self._index[key].zone_label if key in self._index else None

    def density(self, frame_idx: int, person_id: int) -> Optional[int]:
        key = (frame_idx, person_id)
        return self._index[key].density_label if key in self._index else None

    def get(self, frame_idx: int, person_id: int) -> Optional[PersonLabel]:
        return self._index.get((frame_idx, person_id))

    def predict_zone_from_position(self, cx: float, cy: float) -> int:
        """Assign a zone label to a reconstructed position using the fitted K-means."""
        return int(self._kmeans.predict(np.array([[cx, cy]]))[0])

    # ── Bulk accessors ────────────────────────────────────────────────────────

    def as_dataframe(self) -> pd.DataFrame:
        rows = []
        for lb in self._labels:
            rows.append({
                "frame_idx":     lb.frame_idx,
                "person_id":     lb.person_id,
                "world_x_cm":   lb.world_x_cm,
                "world_y_cm":   lb.world_y_cm,
                "zone_label":   lb.zone_label,
                "density_label": lb.density_label,
                "nn_dist_cm":   lb.nn_dist_cm,
            })
        return pd.DataFrame(rows)

    def n_persons(self) -> int:
        return len(self._labels)

    def zone_counts(self) -> dict[int, int]:
        from collections import Counter
        return dict(Counter(lb.zone_label for lb in self._labels))

    def density_counts(self) -> dict[int, int]:
        from collections import Counter
        return dict(Counter(lb.density_label for lb in self._labels))

    def kmeans_centres(self) -> np.ndarray:
        """Return (N_ZONES, 2) array of zone centroids in cm."""
        return self._kmeans.cluster_centers_

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        dataset_root: str | Path,
        seed: int = 42,
        n_zones: int = N_ZONES,
        density_threshold_cm: float = DENSITY_THRESHOLD_CM,
        cache_path: Optional[str | Path] = None,
    ) -> "DownstreamLabelBank":
        """Compute all labels for the full MultiviewX dataset.

        Parameters
        ----------
        dataset_root       : path to MultiviewX root
        seed               : random seed for K-means
        n_zones            : number of spatial clusters
        density_threshold_cm: nearest-neighbour threshold for crowded label
        cache_path         : if given, save/load the label CSV from this path

        Returns
        -------
        DownstreamLabelBank
        """
        # ── Try loading from cache ────────────────────────────────────────
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading downstream label bank from cache: %s", cache_path)
                return cls._from_cache(cache_path, n_zones, seed)

        # ── Load all GT positions ────────────────────────────────────────
        logger.info("Building downstream label bank (this runs once) …")
        frame_indices = list_frame_indices(dataset_root)

        # Collect all (frame, person, x, y) records
        records: list[dict] = []
        frame_persons: dict[int, list[tuple[int, float, float]]] = {}  # frame → [(pid, x, y)]

        for fi in frame_indices:
            try:
                persons = load_frame_annotations(dataset_root, fi, min_cameras_visible=1)
            except Exception as exc:
                logger.warning("Frame %d annotation load failed: %s", fi, exc)
                continue

            fp_list = []
            for p in persons:
                records.append({
                    "frame_idx": fi,
                    "person_id": p.person_id,
                    "world_x_cm": p.world_x_cm,
                    "world_y_cm": p.world_y_cm,
                })
                fp_list.append((p.person_id, p.world_x_cm, p.world_y_cm))
            frame_persons[fi] = fp_list

        logger.info("Collected %d (frame, person) pairs from %d frames.",
                    len(records), len(frame_indices))

        df = pd.DataFrame(records)
        coords = df[["world_x_cm", "world_y_cm"]].values

        # ── Fit K-means zones ────────────────────────────────────────────
        logger.info("Fitting K-means with K=%d …", n_zones)
        kmeans = KMeans(n_clusters=n_zones, random_state=seed, n_init=20, max_iter=500)
        zone_labels = kmeans.fit_predict(coords)
        df["zone_label"] = zone_labels.astype(int)
        logger.info("Zone sizes: %s",
                    dict(zip(*np.unique(zone_labels, return_counts=True))))

        # ── Compute density labels ────────────────────────────────────────
        nn_dists = np.full(len(df), np.inf)
        density  = np.zeros(len(df), dtype=int)

        for fi, fp_list in frame_persons.items():
            if len(fp_list) < 2:
                # Only one (or zero) person: infinite distance, isolated
                continue
            pids  = [x[0] for x in fp_list]
            poses = np.array([(x[1], x[2]) for x in fp_list])

            # Pairwise distances
            for i, (pid_i, xi, yi) in enumerate(fp_list):
                others = np.delete(poses, i, axis=0)
                dists  = np.linalg.norm(others - np.array([xi, yi]), axis=1)
                nn     = float(dists.min())
                # Find rows in df for this (frame, person)
                mask   = (df["frame_idx"] == fi) & (df["person_id"] == pid_i)
                idx    = df.index[mask]
                if len(idx) > 0:
                    nn_dists[idx[0]] = nn
                    density[idx[0]]  = int(nn < density_threshold_cm)

        df["nn_dist_cm"]    = nn_dists
        df["density_label"] = density

        logger.info(
            "Density: isolated=%d  crowded=%d  (threshold=%.0f cm)",
            (density == 0).sum(), (density == 1).sum(), density_threshold_cm,
        )

        # ── Save cache ────────────────────────────────────────────────────
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(str(cache_path), index=False)
            # Also save kmeans centres alongside
            centres_path = cache_path.with_suffix(".kmeans_centres.npy")
            np.save(str(centres_path), kmeans.cluster_centers_)
            logger.info("Label bank cached to: %s", cache_path)

        # ── Build label objects ───────────────────────────────────────────
        labels = []
        for _, row in df.iterrows():
            labels.append(PersonLabel(
                frame_idx=int(row["frame_idx"]),
                person_id=int(row["person_id"]),
                world_x_cm=float(row["world_x_cm"]),
                world_y_cm=float(row["world_y_cm"]),
                zone_label=int(row["zone_label"]),
                density_label=int(row["density_label"]),
                nn_dist_cm=float(row["nn_dist_cm"]),
            ))

        return cls(labels, kmeans)

    @classmethod
    def _from_cache(
        cls,
        cache_path: Path,
        n_zones: int,
        seed: int,
    ) -> "DownstreamLabelBank":
        df = pd.read_csv(str(cache_path))
        coords = df[["world_x_cm", "world_y_cm"]].values

        # Re-fit K-means on cached positions to restore the model
        # (or load centres if saved)
        centres_path = cache_path.with_suffix(".kmeans_centres.npy")
        if centres_path.exists():
            centres = np.load(str(centres_path))
            kmeans = KMeans(n_clusters=n_zones, random_state=seed, n_init=1, max_iter=1)
            kmeans.fit(centres)   # one-step fit on centres to initialise internals
            kmeans.cluster_centers_ = centres
        else:
            logger.warning("K-means centres not found — re-fitting.")
            kmeans = KMeans(n_clusters=n_zones, random_state=seed, n_init=20)
            kmeans.fit(coords)

        labels = []
        for _, row in df.iterrows():
            labels.append(PersonLabel(
                frame_idx=int(row["frame_idx"]),
                person_id=int(row["person_id"]),
                world_x_cm=float(row["world_x_cm"]),
                world_y_cm=float(row["world_y_cm"]),
                zone_label=int(row.get("zone_label", -1)),
                density_label=int(row.get("density_label", 0)),
                nn_dist_cm=float(row.get("nn_dist_cm", np.inf)),
            ))
        return cls(labels, kmeans)
