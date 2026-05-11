"""
mmcows.data.windowing
=====================

Sliding-window construction and feature extraction for sensor time series.

Two public functions cover the full Increment-1 pipeline:

``make_windows``
    Slide a window of ``W`` seconds over a per-cow DataFrame and return
    raw ``(N, T, C)`` arrays with majority-vote labels.

``window_to_flat_features``
    Convert raw ``(N, T, C)`` windows to a ``(N, F)`` flat feature matrix
    using per-channel statistics (Increment-1 baseline).
    Increment-2 will replace / extend this with the full time-domain +
    frequency-domain + physical feature set.

Usage
-----
::

    from mmcows.data.windowing import make_windows, window_to_flat_features

    X_raw, y, cow_ids, start_ts = make_windows(
        df,
        feature_cols=["coord_x_cm", "coord_y_cm", "coord_z_cm"],
        window_size_s=15,
        target_rate_hz=2.0,
        overlap=0.5,
    )
    X_feat = window_to_flat_features(X_raw)   # (N, C*6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size_s: int,
    target_rate_hz: float,
    overlap: float = 0.5,
    unknown_threshold: float = 0.25,
    label_col: str = "behavior",
    cow_id_col: str = "cow_id",
    timestamp_col: str = "timestamp",
    inference: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over a (possibly multi-cow) aligned DataFrame.

    Parameters
    ----------
    df:
        DataFrame produced by any loader + ``resample_to_target``.
        Must contain ``timestamp_col``, ``cow_id_col``, ``label_col``,
        and all ``feature_cols``.
    feature_cols:
        Sensor channels to include in X.
    window_size_s:
        Window length in seconds.
    target_rate_hz:
        Samples per second present in ``df`` (must match what was used
        in ``resample_to_target``).
    overlap:
        Fractional step overlap (0 = non-overlapping, 0.5 = 50% overlap).
        Ignored when ``inference=True`` (always 0).
    unknown_threshold:
        Discard windows where more than this fraction of labels are
        unknown (behavior == 0).  Default 0.25.
    label_col:
        Name of the integer behaviour label column.
    cow_id_col:
        Name of the cow-ID column.
    timestamp_col:
        Name of the Unix timestamp column.
    inference:
        When ``True`` forces non-overlapping windows to avoid label
        leakage during evaluation.

    Returns
    -------
    X:
        ``(N, T, C)`` float32 array of raw sensor windows.
    y:
        ``(N,)`` int array of majority behaviour labels.
    cow_ids:
        ``(N,)`` int array mapping each window to its source cow.
    start_ts:
        ``(N,)`` int64 array of Unix timestamps at the start of each window.
    """
    T    = int(window_size_s * target_rate_hz)
    step = max(1, int(T * (1.0 - (0.0 if inference else overlap))))

    X_list:   list[np.ndarray] = []
    y_list:   list[int]        = []
    cow_list: list[int]        = []
    ts_list:  list[int]        = []

    for cow_id, cow_df in df.groupby(cow_id_col, sort=False):
        cow_df = cow_df.sort_values(timestamp_col).reset_index(drop=True)
        values     = cow_df[feature_cols].values.astype(np.float32)
        labels     = cow_df[label_col].values
        timestamps = cow_df[timestamp_col].values

        for start in range(0, len(values) - T + 1, step):
            end           = start + T
            window_labels = labels[start:end]

            # Discard windows that are mostly unknown
            if np.mean(window_labels == 0) > unknown_threshold:
                continue

            # Majority label, ignoring unknowns
            valid_labels = window_labels[window_labels != 0]
            if len(valid_labels) == 0:
                continue
            uniq, counts  = np.unique(valid_labels, return_counts=True)
            majority_label = int(uniq[np.argmax(counts)])

            X_list.append(values[start:end])
            y_list.append(majority_label)
            cow_list.append(int(cow_id))
            ts_list.append(int(timestamps[start]))

    C = len(feature_cols)
    if not X_list:
        return (
            np.empty((0, T, C), dtype=np.float32),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list,   dtype=int),
        np.array(cow_list, dtype=int),
        np.array(ts_list,  dtype=np.int64),
    )


def window_to_flat_features(X: np.ndarray) -> np.ndarray:
    """Convert a ``(N, T, C)`` window array to a ``(N, C*6)`` feature matrix.

    Features computed **per channel**: mean, std, min, max, median, range.

    This is the Increment-1 baseline feature set.  Increment-2 will
    extend this to the full time-domain + frequency-domain + physical
    feature pipeline (~300 dimensions before selection).

    Parameters
    ----------
    X:
        Raw window array of shape ``(N, T, C)``.

    Returns
    -------
    ``(N, C * 6)`` float32 feature matrix.
    """
    return np.concatenate([
        X.mean(axis=1),                     # mean     (N, C)
        X.std(axis=1),                      # std      (N, C)
        X.min(axis=1),                      # min      (N, C)
        X.max(axis=1),                      # max      (N, C)
        np.median(X, axis=1),               # median   (N, C)
        X.max(axis=1) - X.min(axis=1),      # range    (N, C)
    ], axis=1).astype(np.float32)
