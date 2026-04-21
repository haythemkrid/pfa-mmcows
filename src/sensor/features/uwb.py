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


def _add_stat_features(feat_df, col_name, prefix):
    values = feat_df[col_name].to_numpy(dtype=float)
    feat_df[f"{prefix}_abs"] = np.abs(values)
    feat_df[f"{prefix}_sq"] = values ** 2


def build_engineered_feature_frame(
    data_df,
    position_cols=None,
    lag_steps=(1, 2, 3, 4),
    roll_windows=(3, 5),
    max_gap_s=35.0,
    grid_bins=4,
):
    """Build advanced UWB engineered features at each timestamp.

    Returns
    -------
    X_df : pd.DataFrame
        Tabular engineered features for classification.
    y : np.ndarray
        Behavior class labels aligned with X_df.
    """
    if "timestamp" not in data_df.columns or "behavior" not in data_df.columns:
        raise ValueError("Input dataframe must contain timestamp and behavior columns")

    pos_cols = infer_position_columns(data_df, position_cols)
    x_col, y_col, z_col = pos_cols

    df = data_df[["timestamp", x_col, y_col, z_col, "behavior"]].copy().reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), np.array([], dtype=int)

    df["sequence_id"] = _assign_sequence_ids(df["timestamp"].values, max_gap_s=max_gap_s)

    feat_df = pd.DataFrame(index=df.index)
    feat_df["sequence_id"] = df["sequence_id"].astype(int)
    feat_df["timestamp"] = df["timestamp"].astype(float)
    feat_df["behavior"] = df["behavior"].astype(int)

    feat_df["x"] = df[x_col].astype(float)
    feat_df["y"] = df[y_col].astype(float)
    feat_df["z"] = df[z_col].astype(float)

    grouped = feat_df.groupby("sequence_id", sort=False)

    feat_df["dt"] = grouped["timestamp"].diff().fillna(15.0)
    feat_df["dt"] = feat_df["dt"].clip(lower=1e-3)

    feat_df["dx"] = grouped["x"].diff().fillna(0.0)
    feat_df["dy"] = grouped["y"].diff().fillna(0.0)
    feat_df["dz"] = grouped["z"].diff().fillna(0.0)

    feat_df["disp_xy"] = np.sqrt(feat_df["dx"] ** 2 + feat_df["dy"] ** 2)
    feat_df["disp_xyz"] = np.sqrt(feat_df["dx"] ** 2 + feat_df["dy"] ** 2 + feat_df["dz"] ** 2)

    feat_df["speed_xy"] = feat_df["disp_xy"] / feat_df["dt"]
    feat_df["speed_xyz"] = feat_df["disp_xyz"] / feat_df["dt"]
    feat_df["vertical_speed"] = feat_df["dz"] / feat_df["dt"]

    feat_df["accel"] = grouped["speed_xyz"].diff().fillna(0.0) / feat_df["dt"]

    feat_df["jerk"] = grouped["accel"].diff().fillna(0.0) / feat_df["dt"]

    feat_df["heading"] = np.arctan2(feat_df["dy"], feat_df["dx"])
    heading_delta = grouped["heading"].diff().fillna(0.0).to_numpy(dtype=float)
    feat_df["turn_rate"] = _wrap_angle_delta(heading_delta) / feat_df["dt"].to_numpy(dtype=float)

    feat_df["curvature"] = np.abs(feat_df["turn_rate"]) / (feat_df["speed_xy"] + 1e-6)

    _add_stat_features(feat_df, "speed_xy", "speed_xy")
    _add_stat_features(feat_df, "speed_xyz", "speed_xyz")
    _add_stat_features(feat_df, "accel", "accel")
    _add_stat_features(feat_df, "jerk", "jerk")
    _add_stat_features(feat_df, "turn_rate", "turn")

    for lag in lag_steps:
        for base_col in ["x", "y", "z", "speed_xy", "speed_xyz", "accel", "jerk", "turn_rate"]:
            feat_df[f"{base_col}_lag_{lag}"] = grouped[base_col].shift(lag)

    for win in roll_windows:
        for base_col in ["x", "y", "z", "speed_xy", "speed_xyz", "accel", "jerk", "turn_rate", "disp_xy"]:
            feat_df[f"{base_col}_roll_mean_{win}"] = grouped[base_col].transform(
                lambda s: s.rolling(win, min_periods=1).mean()
            )
            feat_df[f"{base_col}_roll_std_{win}"] = grouped[base_col].transform(
                lambda s: s.rolling(win, min_periods=1).std().fillna(0.0)
            )

    # Grid-zone features (spatial context)
    try:
        feat_df["zone_x"] = pd.qcut(feat_df["x"], q=grid_bins, labels=False, duplicates="drop")
        feat_df["zone_y"] = pd.qcut(feat_df["y"], q=grid_bins, labels=False, duplicates="drop")
    except ValueError:
        feat_df["zone_x"] = 0
        feat_df["zone_y"] = 0

    feat_df["zone_x"] = feat_df["zone_x"].fillna(0).astype(int)
    feat_df["zone_y"] = feat_df["zone_y"].fillna(0).astype(int)
    feat_df["zone_id"] = feat_df["zone_y"] * grid_bins + feat_df["zone_x"]

    zone_dummies = pd.get_dummies(feat_df["zone_id"], prefix="zone", dtype=float)
    feat_df = pd.concat([feat_df, zone_dummies], axis=1)

    # Sequence-level descriptors repeated per row
    feat_df["seq_speed_mean"] = grouped["speed_xyz"].transform("mean")
    feat_df["seq_speed_std"] = grouped["speed_xyz"].transform("std").fillna(0.0)
    feat_df["seq_speed_skew"] = grouped["speed_xyz"].transform(
        lambda s: _safe_skew(s.to_numpy(dtype=float))
    )
    feat_df["seq_speed_kurtosis"] = grouped["speed_xyz"].transform(
        lambda s: _safe_kurtosis(s.to_numpy(dtype=float))
    )

    feature_cols = [
        col
        for col in feat_df.columns
        if col not in {"sequence_id", "timestamp", "behavior", "zone_x", "zone_y", "zone_id"}
    ]

    model_df = feat_df[["behavior"] + feature_cols].replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna().reset_index(drop=True)

    if model_df.empty:
        return pd.DataFrame(), np.array([], dtype=int)

    X_df = model_df.drop(columns=["behavior"])
    y = model_df["behavior"].to_numpy(dtype=int)

    return X_df, y

from .base_selector import FeatureSelectionCascade

class UwbFeatureSelector(FeatureSelectionCascade):
    pass
