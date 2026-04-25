"""
MmCows — Train / Val / Test Split Logic
========================================

Two split strategies match the official benchmark:

* **Object-wise split (OS / S1)** — cows are partitioned into disjoint
  train / val / test groups.  This tests generalisation to *unseen animals*.
* **Temporal split (TS / S2)** — the single annotated day is sliced into
  time windows.  This tests generalisation to *unseen times of day*.

Both strategies are loaded from the JSON config files in the official
MmCows repository (``configs/config_s1.json``, ``configs/config_s2.json``).

Usage example
-------------
::

    from mmcows.data.splits import SplitConfig
    from mmcows.data.loaders import UWBLoader

    config = SplitConfig.from_json(
        s1_path="path/to/mmcows/configs/config_s1.json",
        s2_path="path/to/mmcows/configs/config_s2.json",
    )

    loader = UWBLoader(sensor_data_dir="/data/sensor_data")
    full_df = loader.load(cow_ids=list(range(1, 11)), date="0725")

    train, val, test = config.split(full_df, split_type="s1", fold="fold_1")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mmcows.utils.time_utils import cdt_str_to_unix

logger = logging.getLogger(__name__)

SplitType = Literal["s1", "s2"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_s2_config(raw: dict) -> dict:
    """Convert S2 datetime strings → Unix timestamps in-place and return."""
    for group_key in ("group_1", "group_2"):
        for window_key, bounds in raw.get(group_key, {}).items():
            raw[group_key][window_key] = [
                cdt_str_to_unix(bounds[0]),
                cdt_str_to_unix(bounds[1]),
            ]
    return raw


def _extract_s2_rows(
    df: pd.DataFrame,
    window_keys: list[str],
    group_1: dict[str, list[int]],
    group_2: dict[str, list[int]],
) -> pd.DataFrame:
    """Select rows from *df* that fall inside any of the given time windows."""
    masks: list[pd.Series] = []
    for key in window_keys:
        for bounds in (group_1.get(key), group_2.get(key)):
            if bounds is None:
                continue
            start, end = bounds
            masks.append(
                (df["timestamp"] >= start) & (df["timestamp"] < end)
            )

    if not masks:
        return pd.DataFrame(columns=df.columns)

    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = combined_mask | m

    return df.loc[combined_mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SplitConfig:
    """Holds the parsed S1 and S2 split configurations.

    Attributes
    ----------
    s1 : dict
        Parsed object-wise split config (cow ID groups per fold).
    s2 : dict
        Parsed temporal split config (time windows per fold, converted
        to Unix timestamps).
    """

    s1: dict = field(default_factory=dict)
    s2: dict = field(default_factory=dict)

    # ─── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_json(
        cls,
        s1_path: str | Path | None = None,
        s2_path: str | Path | None = None,
    ) -> "SplitConfig":
        """Load split configs from the official JSON files.

        Either or both paths can be ``None`` (the corresponding split will
        be unavailable).

        Parameters
        ----------
        s1_path:
            Path to ``config_s1.json`` from the official MmCows repository.
        s2_path:
            Path to ``config_s2.json`` from the official MmCows repository.

        Returns
        -------
        A :class:`SplitConfig` instance.
        """
        s1, s2 = {}, {}

        if s1_path is not None:
            p = Path(s1_path)
            if not p.exists():
                raise FileNotFoundError(f"S1 config not found: {p}")
            with p.open() as fh:
                s1 = json.load(fh)
            logger.info("Loaded S1 config from %s — folds: %s", p, list(s1.get("folds", {}).keys()))

        if s2_path is not None:
            p = Path(s2_path)
            if not p.exists():
                raise FileNotFoundError(f"S2 config not found: {p}")
            with p.open() as fh:
                raw = json.load(fh)
            s2 = _parse_s2_config(raw)
            logger.info("Loaded S2 config from %s — folds: %s", p, list(s2.get("folds", {}).keys()))

        return cls(s1=s1, s2=s2)

    # ─── Available folds ──────────────────────────────────────────────────

    def available_folds(self, split_type: SplitType = "s1") -> list[str]:
        """Return the list of fold names available for *split_type*.

        Parameters
        ----------
        split_type:
            ``"s1"`` (object-wise) or ``"s2"`` (temporal).
        """
        cfg = self.s1 if split_type == "s1" else self.s2
        return list(cfg.get("folds", {}).keys())

    # ─── Core splitting method ────────────────────────────────────────────

    def split(
        self,
        df: pd.DataFrame,
        *,
        split_type: SplitType = "s1",
        fold: str = "fold_1",
        timestamp_col: str = "timestamp",
        cow_id_col: str = "cow_id",
        keep_timestamp: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split *df* into (train, val, test) according to the configuration.

        Parameters
        ----------
        df:
            A DataFrame produced by any loader.  Must contain
            *timestamp_col* and *cow_id_col*.
        split_type:
            ``"s1"`` for object-wise split; ``"s2"`` for temporal split.
        fold:
            Fold name, e.g. ``"fold_1"``.  See :meth:`available_folds`.
        timestamp_col:
            Name of the Unix timestamp column (default ``"timestamp"``).
        cow_id_col:
            Name of the cow-ID column (default ``"cow_id"``).
        keep_timestamp:
            Whether to retain the timestamp column in the output DataFrames
            (default ``True``).  Set to ``False`` to match the original
            repo's loader behavior.

        Returns
        -------
        ``(train_df, val_df, test_df)`` — three DataFrames.  Any subset can
        be empty if the split configuration yields no matching rows.

        Raises
        ------
        ValueError
            If *split_type* is invalid or *fold* is not in the config.
        RuntimeError
            If the requested config was not loaded (path was ``None``).
        """
        if split_type not in ("s1", "s2"):
            raise ValueError(f"split_type must be 's1' or 's2', got {split_type!r}")

        cfg = self.s1 if split_type == "s1" else self.s2
        if not cfg:
            raise RuntimeError(
                f"Config for split_type={split_type!r} was not loaded. "
                "Pass the corresponding path to SplitConfig.from_json()."
            )

        if fold not in cfg.get("folds", {}):
            available = list(cfg.get("folds", {}).keys())
            raise ValueError(
                f"Fold {fold!r} not found in {split_type} config. "
                f"Available folds: {available}"
            )

        fold_cfg = cfg["folds"][fold]

        if split_type == "s1":
            train_df, val_df, test_df = self._split_s1(
                df, fold_cfg, cow_id_col=cow_id_col
            )
        else:
            train_df, val_df, test_df = self._split_s2(
                df,
                fold_cfg,
                group_1=cfg.get("group_1", {}),
                group_2=cfg.get("group_2", {}),
                timestamp_col=timestamp_col,
            )

        if not keep_timestamp:
            for part in (train_df, val_df, test_df):
                part.drop(columns=[timestamp_col], inplace=True, errors="ignore")

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # ─── Internal split implementations ──────────────────────────────────

    @staticmethod
    def _split_s1(
        df: pd.DataFrame,
        fold_cfg: dict,
        *,
        cow_id_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Object-wise split: partition by cow_id."""
        train_ids = {int(x) for x in fold_cfg["train"]}
        val_ids   = {int(x) for x in fold_cfg["val"]}
        test_ids  = {int(x) for x in fold_cfg["test"]}

        col = df[cow_id_col].astype(int)
        train_df = df[col.isin(train_ids)]
        val_df   = df[col.isin(val_ids)]
        test_df  = df[col.isin(test_ids)]

        logger.debug(
            "S1 split — train cows: %s, val: %s, test: %s",
            sorted(train_ids), sorted(val_ids), sorted(test_ids),
        )
        return train_df, val_df, test_df

    @staticmethod
    def _split_s2(
        df: pd.DataFrame,
        fold_cfg: dict,
        *,
        group_1: dict,
        group_2: dict,
        timestamp_col: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal split: partition by time windows."""
        train_df = _extract_s2_rows(df, fold_cfg["train"], group_1, group_2)
        val_df   = _extract_s2_rows(df, fold_cfg["val"],   group_1, group_2)
        test_df  = _extract_s2_rows(df, fold_cfg["test"],  group_1, group_2)

        logger.debug(
            "S2 split — train: %d rows, val: %d rows, test: %d rows",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    # ─── Convenience: all folds iterator ──────────────────────────────────

    def iter_folds(
        self,
        df: pd.DataFrame,
        *,
        split_type: SplitType = "s1",
        **split_kwargs,
    ):
        """Iterate over all folds, yielding ``(fold_name, train, val, test)``.

        Parameters
        ----------
        df:
            Source DataFrame.
        split_type:
            ``"s1"`` or ``"s2"``.
        **split_kwargs:
            Additional keyword arguments forwarded to :meth:`split`.

        Yields
        ------
        ``(fold_name, train_df, val_df, test_df)``
        """
        for fold_name in self.available_folds(split_type):
            train, val, test = self.split(
                df, split_type=split_type, fold=fold_name, **split_kwargs
            )
            yield fold_name, train, val, test