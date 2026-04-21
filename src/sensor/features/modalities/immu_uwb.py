import numpy as np
import pandas as pd


def _safe_skew(values):
    std = np.std(values)
    if std < 1e-12:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 3))


def _safe_kurtosis(values):
    std = np.std(values)
    if std < 1e-12:
        return 0.0
    centered = values - np.mean(values)
    return float(np.mean((centered / std) ** 4) - 3.0)


def _wrap_angle_delta(values):
    return (values + np.pi) % (2 * np.pi) - np.pi


def infer_position_columns(data_df, position_cols=None):
    if position_cols is not None:
        if len(position_cols) != 3:
            raise ValueError("position_cols must contain exactly 3 column names")
        missing = [col for col in position_cols if col not in data_df.columns]
        if missing:
            raise ValueError(f"Missing provided position columns: {missing}")
        return list(position_cols)

    candidates = [
        ["coord_x_cm", "coord_y_cm", "coord_z_cm"],
        ["coord_x", "coord_y", "coord_z"],
        ["x", "y", "z"],
        ["x_m", "y_m", "z_m"],
        ["pos_x", "pos_y", "pos_z"],
        ["location_x", "location_y", "location_z"],
    ]

    for cols in candidates:
        if all(col in data_df.columns for col in cols):
            return cols

    ignored = {"timestamp", "behavior", "datetime", "id", "tag_id", "cow_id"}
    numeric_cols = [
        col
        for col in data_df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(data_df[col])
    ]
    if len(numeric_cols) < 3:
        raise ValueError("Could not infer three numeric position columns from UWB dataframe")

    return numeric_cols[:3]


def _assign_sequence_ids(timestamps, max_gap_s):
    ts = np.asarray(timestamps, dtype=float)
    if ts.size == 0:
        return np.array([], dtype=int)

    seq_ids = np.zeros(ts.shape[0], dtype=int)
    diffs = np.diff(ts)
    break_mask = (diffs <= 0) | (diffs > max_gap_s)
    seq_ids[1:] = np.cumsum(break_mask)
    return seq_ids


def build_engineered_feature_frame(
    data_df,
    position_cols=None,
    lag_steps=(1, 2, 3, 4),
    roll_windows=(3, 5),
    max_gap_s=35.0,
    grid_bins=4,
):
    """Build fused UWB (low-rate) + IMMU (window features) engineered features.

    Input is expected to be aligned to UWB timestamps, with IMMU window features
    already present as columns prefixed by `immu_`.

    Returns
    -------
    X_df : pd.DataFrame
        Tabular engineered features for classification.
    y : np.ndarray
        Behavior labels aligned with X_df.
    """

    if "timestamp" not in data_df.columns or "behavior" not in data_df.columns:
        raise ValueError("Input dataframe must contain timestamp and behavior columns")

    pos_cols = infer_position_columns(data_df, position_cols)
    x_col, y_col, z_col = pos_cols

    immu_cols = [
        col
        for col in data_df.columns
        if col.startswith("immu_") and pd.api.types.is_numeric_dtype(data_df[col])
    ]

    keep_cols = ["timestamp", x_col, y_col, z_col, "behavior"] + immu_cols

    df = data_df[keep_cols].copy().dropna().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), np.array([], dtype=int)

    df["sequence_id"] = _assign_sequence_ids(df["timestamp"].values, max_gap_s=max_gap_s)

    feat_df = pd.DataFrame(index=df.index)
    feat_df["sequence_id"] = df["sequence_id"].astype(int)
    feat_df["timestamp"] = df["timestamp"].astype(float)
    feat_df["behavior"] = df["behavior"].astype(int)

    feat_df["uwb_x"] = df[x_col].astype(float)
    feat_df["uwb_y"] = df[y_col].astype(float)
    feat_df["uwb_z"] = df[z_col].astype(float)

    # Pass-through IMMU window features (already aligned to UWB timestamps).
    for col in immu_cols:
        feat_df[col] = df[col].astype(float)

    grouped = feat_df.groupby("sequence_id", sort=False)

    # UWB kinematics
    feat_df["dt"] = grouped["timestamp"].diff().fillna(15.0).clip(lower=1e-3)

    feat_df["uwb_dx"] = grouped["uwb_x"].diff().fillna(0.0)
    feat_df["uwb_dy"] = grouped["uwb_y"].diff().fillna(0.0)
    feat_df["uwb_dz"] = grouped["uwb_z"].diff().fillna(0.0)

    feat_df["uwb_disp_xy"] = np.sqrt(feat_df["uwb_dx"] ** 2 + feat_df["uwb_dy"] ** 2)
    feat_df["uwb_disp_xyz"] = np.sqrt(
        feat_df["uwb_dx"] ** 2 + feat_df["uwb_dy"] ** 2 + feat_df["uwb_dz"] ** 2
    )

    feat_df["uwb_speed_xy"] = feat_df["uwb_disp_xy"] / feat_df["dt"]
    feat_df["uwb_speed_xyz"] = feat_df["uwb_disp_xyz"] / feat_df["dt"]

    feat_df["uwb_accel"] = grouped["uwb_speed_xyz"].diff().fillna(0.0) / feat_df["dt"]
    feat_df["uwb_jerk"] = grouped["uwb_accel"].diff().fillna(0.0) / feat_df["dt"]

    feat_df["uwb_heading"] = np.arctan2(feat_df["uwb_dy"], feat_df["uwb_dx"])
    heading_delta = grouped["uwb_heading"].diff().fillna(0.0).to_numpy(dtype=float)
    feat_df["uwb_turn_rate"] = _wrap_angle_delta(heading_delta) / feat_df["dt"].to_numpy(dtype=float)

    feat_df["uwb_curvature"] = np.abs(feat_df["uwb_turn_rate"]) / (feat_df["uwb_speed_xy"] + 1e-6)

    feat_df["uwb_speed_abs"] = np.abs(feat_df["uwb_speed_xyz"])
    feat_df["uwb_speed_sq"] = feat_df["uwb_speed_xyz"] ** 2
    feat_df["uwb_accel_abs"] = np.abs(feat_df["uwb_accel"])
    feat_df["uwb_jerk_abs"] = np.abs(feat_df["uwb_jerk"])

    # Lags (UWB-only to keep feature count bounded)
    lag_base_cols = [
        "uwb_x",
        "uwb_y",
        "uwb_z",
        "uwb_speed_xy",
        "uwb_speed_xyz",
        "uwb_accel",
        "uwb_jerk",
        "uwb_turn_rate",
    ]
    for lag in lag_steps:
        for base_col in lag_base_cols:
            feat_df[f"{base_col}_lag_{lag}"] = grouped[base_col].shift(lag)

    # Rolling stats (UWB-only)
    roll_base_cols = [
        "uwb_speed_xy",
        "uwb_speed_xyz",
        "uwb_accel",
        "uwb_turn_rate",
        "uwb_disp_xy",
    ]
    for win in roll_windows:
        for base_col in roll_base_cols:
            feat_df[f"{base_col}_roll_mean_{win}"] = grouped[base_col].transform(
                lambda s: s.rolling(win, min_periods=1).mean()
            )
            feat_df[f"{base_col}_roll_std_{win}"] = grouped[base_col].transform(
                lambda s: s.rolling(win, min_periods=1).std().fillna(0.0)
            )
            feat_df[f"{base_col}_roll_max_{win}"] = grouped[base_col].transform(
                lambda s: s.rolling(win, min_periods=1).max()
            )

    # Spatial bins from UWB
    try:
        feat_df["uwb_zone_x"] = pd.qcut(
            feat_df["uwb_x"], q=grid_bins, labels=False, duplicates="drop"
        )
        feat_df["uwb_zone_y"] = pd.qcut(
            feat_df["uwb_y"], q=grid_bins, labels=False, duplicates="drop"
        )
    except ValueError:
        feat_df["uwb_zone_x"] = 0
        feat_df["uwb_zone_y"] = 0

    feat_df["uwb_zone_x"] = feat_df["uwb_zone_x"].fillna(0).astype(int)
    feat_df["uwb_zone_y"] = feat_df["uwb_zone_y"].fillna(0).astype(int)
    feat_df["uwb_zone_id"] = feat_df["uwb_zone_y"] * grid_bins + feat_df["uwb_zone_x"]

    zone_dummies = pd.get_dummies(feat_df["uwb_zone_id"], prefix="uwb_zone", dtype=float)
    feat_df = pd.concat([feat_df, zone_dummies], axis=1)

    # A couple of simple cross-modal interactions (only if the IMMU feature exists)
    if "immu_mag_energy" in feat_df.columns:
        feat_df["int_uwb_immu_speed_energy"] = feat_df["uwb_speed_xy"] * feat_df["immu_mag_energy"]
    if "immu_mag_rms" in feat_df.columns:
        feat_df["int_uwb_immu_speed_rms"] = feat_df["uwb_speed_xy"] * feat_df["immu_mag_rms"]

    # Sequence-level descriptors
    feat_df["seq_uwb_speed_mean"] = grouped["uwb_speed_xyz"].transform("mean")
    feat_df["seq_uwb_speed_std"] = grouped["uwb_speed_xyz"].transform("std").fillna(0.0)
    feat_df["seq_uwb_speed_skew"] = grouped["uwb_speed_xyz"].transform(
        lambda s: _safe_skew(s.to_numpy(dtype=float))
    )
    feat_df["seq_uwb_speed_kurtosis"] = grouped["uwb_speed_xyz"].transform(
        lambda s: _safe_kurtosis(s.to_numpy(dtype=float))
    )

    feature_cols = [
        col
        for col in feat_df.columns
        if col
        not in {
            "sequence_id",
            "timestamp",
            "behavior",
            "uwb_zone_x",
            "uwb_zone_y",
            "uwb_zone_id",
        }
    ]

    model_df = feat_df[["behavior"] + feature_cols].replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna().reset_index(drop=True)

    if model_df.empty:
        return pd.DataFrame(), np.array([], dtype=int)

    X_df = model_df.drop(columns=["behavior"])
    y = model_df["behavior"].to_numpy(dtype=int)
    return X_df, y

from src.sensor.features.base_selector import FeatureSelectionCascade

class ImmuUwbFeatureSelector(FeatureSelectionCascade):
    pass
