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


def _assign_sequence_ids(timestamps, max_gap_s):
    ts = np.asarray(timestamps, dtype=float)
    if ts.size == 0:
        return np.array([], dtype=int)

    seq_ids = np.zeros(ts.shape[0], dtype=int)
    diffs = np.diff(ts)
    break_mask = (diffs <= 0) | (diffs > max_gap_s)
    seq_ids[1:] = np.cumsum(break_mask)
    return seq_ids


def _infer_position_columns(data_df, position_cols=None):
    if position_cols is not None:
        if len(position_cols) != 3:
            raise ValueError("position_cols must contain exactly 3 columns")
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
    ]
    for cols in candidates:
        if all(col in data_df.columns for col in cols):
            return cols

    raise ValueError("Could not infer UWB position columns")


def _infer_hd_angle_column(data_df, angle_col=None):
    if angle_col is not None:
        if angle_col not in data_df.columns:
            raise ValueError(f"Provided angle_col is missing: {angle_col}")
        return angle_col

    candidates = [
        "relative_angle",
        "head_direction",
        "heading",
        "head_angle",
        "yaw",
        "angle",
    ]
    for col in candidates:
        if col in data_df.columns:
            return col

    ignored = {"timestamp", "behavior", "datetime", "id", "tag_id", "cow_id"}
    numeric_cols = [
        col
        for col in data_df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(data_df[col])
    ]
    token_cols = [
        col
        for col in numeric_cols
        if ("angle" in col.lower()) or ("head" in col.lower()) or ("yaw" in col.lower())
    ]
    if token_cols:
        return token_cols[0]

    raise ValueError("Could not infer HD angle column")


def _infer_ankle_channels(data_df, ankle_axis_cols=None):
    if ankle_axis_cols is not None:
        if len(ankle_axis_cols) != 3:
            raise ValueError("ankle_axis_cols must contain exactly 3 columns")
        missing = [col for col in ankle_axis_cols if col not in data_df.columns]
        if missing:
            raise ValueError(f"Missing provided ankle axis columns: {missing}")
        return {
            "mode": "triaxial",
            "ax": ankle_axis_cols[0],
            "ay": ankle_axis_cols[1],
            "az": ankle_axis_cols[2],
            "mag": None,
        }

    axis_candidates = [
        ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"],
        ["accel_x", "accel_y", "accel_z"],
        ["ankle_accel_x", "ankle_accel_y", "ankle_accel_z"],
    ]
    for cols in axis_candidates:
        if all(col in data_df.columns for col in cols):
            return {
                "mode": "triaxial",
                "ax": cols[0],
                "ay": cols[1],
                "az": cols[2],
                "mag": "accel_norm" if "accel_norm" in data_df.columns else None,
            }

    scalar_candidates = ["accel_norm", "lying", "ankle_signal"]
    for col in scalar_candidates:
        if col in data_df.columns:
            return {
                "mode": "scalar",
                "ax": None,
                "ay": None,
                "az": None,
                "mag": col,
            }

    ignored = {
        "timestamp",
        "behavior",
        "datetime",
        "id",
        "tag_id",
        "cow_id",
        "coord_x_cm",
        "coord_y_cm",
        "coord_z_cm",
        "coord_x",
        "coord_y",
        "coord_z",
        "roll",
        "pitch",
        "yaw",
        "relative_angle",
    }
    numeric_cols = [
        col
        for col in data_df.columns
        if col not in ignored and pd.api.types.is_numeric_dtype(data_df[col])
    ]
    token_cols = [
        col
        for col in numeric_cols
        if ("ankle" in col.lower()) or ("acc" in col.lower()) or ("imu" in col.lower())
    ]
    if token_cols:
        return {
            "mode": "scalar",
            "ax": None,
            "ay": None,
            "az": None,
            "mag": token_cols[0],
        }

    raise ValueError("Could not infer AKL channels")


def _to_radians(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    max_abs = np.max(np.abs(finite))
    # If values look like degrees, convert to radians.
    if max_abs > (2.0 * np.pi + 0.25):
        return np.deg2rad(arr)
    return arr


def build_engineered_feature_frame(
    data_df,
    position_cols=None,
    angle_col=None,
    ankle_axis_cols=None,
    lag_steps=(1, 2, 3, 4),
    roll_windows=(3, 5),
    max_gap_s=35.0,
    grid_bins=4,
):
    if "timestamp" not in data_df.columns or "behavior" not in data_df.columns:
        raise ValueError("Input dataframe must include timestamp and behavior columns")

    x_col, y_col, z_col = _infer_position_columns(data_df, position_cols)
    hd_col = _infer_hd_angle_column(data_df, angle_col)
    akl_info = _infer_ankle_channels(data_df, ankle_axis_cols)
    accel_norm_col = "accel_norm" if "accel_norm" in data_df.columns else None

    keep_cols = ["timestamp", x_col, y_col, z_col, hd_col, "behavior"]
    if akl_info["mode"] == "triaxial":
        keep_cols.extend([akl_info["ax"], akl_info["ay"], akl_info["az"]])
        if akl_info["mag"] is not None:
            keep_cols.append(akl_info["mag"])
    else:
        keep_cols.append(akl_info["mag"])

    df = data_df[keep_cols].copy().dropna().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), np.array([], dtype=int)

    df["sequence_id"] = _assign_sequence_ids(df["timestamp"].values, max_gap_s=max_gap_s)

    feat_df = pd.DataFrame(index=df.index)
    feat_df["sequence_id"] = df["sequence_id"].astype(int)
    feat_df["timestamp"] = df["timestamp"].astype(float)
    feat_df["behavior"] = df["behavior"].astype(int)

    # UWB base coordinates
    feat_df["uwb_x"] = df[x_col].astype(float)
    feat_df["uwb_y"] = df[y_col].astype(float)
    feat_df["uwb_z"] = df[z_col].astype(float)

    # HD base angle
    hd_theta = _to_radians(df[hd_col].to_numpy(dtype=float))
    feat_df["hd_theta"] = hd_theta
    feat_df["hd_sin"] = np.sin(hd_theta)
    feat_df["hd_cos"] = np.cos(hd_theta)

    # AKL base signals
    if akl_info["mode"] == "triaxial":
        feat_df["akl_ax"] = df[akl_info["ax"]].astype(float)
        feat_df["akl_ay"] = df[akl_info["ay"]].astype(float)
        feat_df["akl_az"] = df[akl_info["az"]].astype(float)
        akl_mag_from_axes = np.sqrt(
            feat_df["akl_ax"] ** 2 + feat_df["akl_ay"] ** 2 + feat_df["akl_az"] ** 2
        )
        if akl_info["mag"] is not None:
            raw_norm = df[akl_info["mag"]].astype(float)
            feat_df["akl_mag"] = raw_norm.where(np.isfinite(raw_norm), akl_mag_from_axes)
            feat_df["akl_mag"] = feat_df["akl_mag"].fillna(akl_mag_from_axes)
        else:
            feat_df["akl_mag"] = akl_mag_from_axes
    else:
        feat_df["akl_mag"] = df[akl_info["mag"]].astype(float)
        feat_df["akl_ax"] = feat_df["akl_mag"]
        feat_df["akl_ay"] = 0.0
        feat_df["akl_az"] = 0.0

    grouped = feat_df.groupby("sequence_id", sort=False)

    feat_df["dt"] = grouped["timestamp"].diff().fillna(15.0).clip(lower=1e-3)

    # UWB kinematics
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

    feat_df["uwb_speed_abs"] = np.abs(feat_df["uwb_speed_xyz"])
    feat_df["uwb_speed_sq"] = feat_df["uwb_speed_xyz"] ** 2
    feat_df["uwb_accel_abs"] = np.abs(feat_df["uwb_accel"])
    feat_df["uwb_jerk_abs"] = np.abs(feat_df["uwb_jerk"])

    # HD dynamics
    hd_delta = grouped["hd_theta"].diff().fillna(0.0).to_numpy(dtype=float)
    feat_df["hd_dtheta"] = _wrap_angle_delta(hd_delta)
    feat_df["hd_turn_rate"] = feat_df["hd_dtheta"] / feat_df["dt"]
    feat_df["hd_turn_abs"] = np.abs(feat_df["hd_turn_rate"])
    feat_df["hd_turn_sq"] = feat_df["hd_turn_rate"] ** 2

    # AKL dynamics
    for col in ["akl_ax", "akl_ay", "akl_az", "akl_mag"]:
        feat_df[f"{col}_jerk"] = grouped[col].diff().fillna(0.0) / feat_df["dt"]

    feat_df["akl_jerk_mag"] = np.sqrt(
        feat_df["akl_ax_jerk"] ** 2 + feat_df["akl_ay_jerk"] ** 2 + feat_df["akl_az_jerk"] ** 2
    )

    # Detrended magnitude proxy
    feat_df["akl_mag_roll_mean_5"] = grouped["akl_mag"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    feat_df["akl_dyn_mag"] = feat_df["akl_mag"] - feat_df["akl_mag_roll_mean_5"]

    # Lags
    lag_base_cols = [
        "uwb_x",
        "uwb_y",
        "uwb_z",
        "uwb_speed_xy",
        "uwb_speed_xyz",
        "uwb_accel",
        "uwb_jerk",
        "uwb_turn_rate",
        "hd_sin",
        "hd_cos",
        "hd_turn_rate",
        "akl_mag",
        "akl_dyn_mag",
        "akl_jerk_mag",
    ]
    for lag in lag_steps:
        for base_col in lag_base_cols:
            feat_df[f"{base_col}_lag_{lag}"] = grouped[base_col].shift(lag)

    # Rolling stats
    roll_base_cols = [
        "uwb_speed_xy",
        "uwb_speed_xyz",
        "uwb_accel",
        "uwb_turn_rate",
        "hd_turn_rate",
        "hd_turn_abs",
        "akl_mag",
        "akl_dyn_mag",
        "akl_jerk_mag",
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

    # AKL axis coupling proxies
    feat_df["akl_xy_dot"] = feat_df["akl_ax"] * feat_df["akl_ay"]
    feat_df["akl_xz_dot"] = feat_df["akl_ax"] * feat_df["akl_az"]
    feat_df["akl_yz_dot"] = feat_df["akl_ay"] * feat_df["akl_az"]

    # Spatial bins from UWB
    try:
        feat_df["uwb_zone_x"] = pd.qcut(feat_df["uwb_x"], q=grid_bins, labels=False, duplicates="drop")
        feat_df["uwb_zone_y"] = pd.qcut(feat_df["uwb_y"], q=grid_bins, labels=False, duplicates="drop")
    except ValueError:
        feat_df["uwb_zone_x"] = 0
        feat_df["uwb_zone_y"] = 0

    feat_df["uwb_zone_x"] = feat_df["uwb_zone_x"].fillna(0).astype(int)
    feat_df["uwb_zone_y"] = feat_df["uwb_zone_y"].fillna(0).astype(int)
    feat_df["uwb_zone_id"] = feat_df["uwb_zone_y"] * grid_bins + feat_df["uwb_zone_x"]

    zone_dummies = pd.get_dummies(feat_df["uwb_zone_id"], prefix="uwb_zone", dtype=float)
    feat_df = pd.concat([feat_df, zone_dummies], axis=1)

    # Cross-modal interactions
    feat_df["int_uwb_hd_speed_turn"] = feat_df["uwb_speed_xy"] * np.abs(feat_df["hd_turn_rate"])
    feat_df["int_uwb_akl_speed_mag"] = feat_df["uwb_speed_xy"] * feat_df["akl_mag"]
    feat_df["int_hd_akl_turn_mag"] = np.abs(feat_df["hd_turn_rate"]) * feat_df["akl_mag"]
    feat_df["int_hd_akl_turn_jerk"] = feat_df["hd_turn_rate"] * feat_df["akl_jerk_mag"]

    # Sequence-level descriptors
    feat_df["seq_uwb_speed_mean"] = grouped["uwb_speed_xyz"].transform("mean")
    feat_df["seq_uwb_speed_std"] = grouped["uwb_speed_xyz"].transform("std").fillna(0.0)
    feat_df["seq_hd_turn_abs_mean"] = grouped["hd_turn_abs"].transform("mean")
    feat_df["seq_akl_mag_mean"] = grouped["akl_mag"].transform("mean")
    feat_df["seq_akl_mag_std"] = grouped["akl_mag"].transform("std").fillna(0.0)
    feat_df["seq_akl_mag_skew"] = grouped["akl_mag"].transform(
        lambda s: _safe_skew(s.to_numpy(dtype=float))
    )
    feat_df["seq_akl_mag_kurtosis"] = grouped["akl_mag"].transform(
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
    model_df = model_df.bfill().ffill().fillna(0).reset_index(drop=True)

    if model_df.empty:
        return pd.DataFrame(), np.array([], dtype=int)

    X_df = model_df.drop(columns=["behavior"])
    y = model_df["behavior"].to_numpy(dtype=int)
    return X_df, y

from src.sensor.features.base_selector import FeatureSelectionCascade

class UwbHdAklFeatureSelector(FeatureSelectionCascade):
    pass
