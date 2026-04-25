"""
mmcows.data.sync
================

Synchronisation and resampling utilities for heterogeneous sensor streams.

All sensors in MMCows run at different rates:

    UWB       1/15 Hz  (one reading every 15 seconds)
    IMMU      10  Hz
    Ankle     1/60 Hz  (one reading per minute)
    CBT       1/60 Hz

``resample_to_target`` brings any of these to a common rate ``f_t`` on a
regular time axis, ready for windowing.

Strategy
--------
* **Decimation** (source > target): Butterworth low-pass anti-aliasing
  filter (order 4, cutoff f_t/2), then cubic-spline interpolation onto
  the new grid.
* **Upsampling** (source < target): cubic-spline interpolation directly.
* **Drop-outs**: forward-fill NaNs that arise from extrapolation outside
  the original measurement range; a companion ``<col>_valid`` binary
  column (1 = original reading, 0 = gap-filled) is added per channel.

Usage
-----
::

    from mmcows.data.sync import resample_to_target

    cow_df_2hz = resample_to_target(
        cow_df,
        feature_cols=["coord_x_cm", "coord_y_cm", "coord_z_cm"],
        source_rate_hz=1 / 15,
        target_rate_hz=2.0,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _butterworth_lowpass(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter applied along axis 0."""
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.999)   # clamp below Nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def resample_to_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    source_rate_hz: float,
    target_rate_hz: float,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample a single-cow sensor DataFrame to a regular ``target_rate_hz`` grid.

    Parameters
    ----------
    df:
        Sorted DataFrame for a single cow.  Must contain ``timestamp_col``
        and all columns in ``feature_cols``.
    feature_cols:
        Columns to resample.  Non-feature columns (e.g. ``cow_id``,
        ``behavior``) are NOT carried over — merge them back via
        ``pd.merge_asof`` after calling this function.
    source_rate_hz:
        Nominal sampling rate of the input signal (Hz).
        For UWB use ``1/15``; for IMMU use ``10``; for ankle/CBT use ``1/60``.
    target_rate_hz:
        Desired output rate (Hz).  Typical choices: 1, 2, 5.
    timestamp_col:
        Name of the Unix timestamp column (default ``"timestamp"``).

    Returns
    -------
    A new DataFrame with columns:

        ``timestamp``          — int64, regular grid at ``target_rate_hz``
        ``<col>``              — resampled signal values (float32)
        ``<col>_valid``        — int8, 1 = original reading, 0 = gap-filled
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    t_orig = df[timestamp_col].values.astype(float)

    step  = 1.0 / target_rate_hz
    t_new = np.arange(t_orig[0], t_orig[-1] + step * 0.5, step)

    result: dict[str, np.ndarray] = {timestamp_col: t_new.astype(np.int64)}

    for col in feature_cols:
        signal = df[col].values.astype(float)

        if source_rate_hz > target_rate_hz:
            # Decimate: anti-alias then interpolate onto new grid
            filtered = _butterworth_lowpass(signal, target_rate_hz / 2.0, source_rate_hz)
            cs = CubicSpline(t_orig, filtered, extrapolate=False)
        else:
            # Upsample or same rate: spline directly
            cs = CubicSpline(t_orig, signal, extrapolate=False)

        resampled = cs(t_new)
        valid     = (~np.isnan(resampled)).astype(np.int8)

        # Forward-fill NaNs at boundaries
        series = pd.Series(resampled).ffill()
        result[col]              = series.values.astype(np.float32)
        result[f"{col}_valid"]   = valid

    return pd.DataFrame(result)
