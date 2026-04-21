import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
from .base_selector import FeatureSelectionCascade

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None

def _safe_skew(values):
    from scipy.stats import skew
    return float(skew(values, bias=False, nan_policy="omit")) if len(values) > 2 else 0.0

def _safe_kurtosis(values):
    from scipy.stats import kurtosis
    return float(kurtosis(values, bias=False, nan_policy="omit")) if len(values) > 3 else 0.0

def _zero_crossing_rate(values):
    centered = values - np.mean(values)
    return float(np.sum(np.abs(np.diff(np.signbit(centered))))) / max(1, len(values) - 1)

def _spectral_entropy(power):
    prob_dist = power / (np.sum(power) + 1e-12)
    prob_dist = prob_dist[prob_dist > 0]
    return float(-np.sum(prob_dist * np.log2(prob_dist))) if prob_dist.size > 0 else 0.0

def _band_power(freqs, power, low_f, high_f):
    mask = (freqs >= low_f) & (freqs <= high_f)
    return float(np.sum(power[mask])) if np.any(mask) else 0.0

def _safe_corr(x, y):
    if np.var(x) < 1e-12 or np.var(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def _time_features(values, prefix):
    """Mean, std, min, max, median, IQR, RMS, energy, skew, kurtosis, ZCR"""
    if len(values) == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_energy": float(np.sum(values**2)),
        f"{prefix}_skew": _safe_skew(values),
        f"{prefix}_kurtosis": _safe_kurtosis(values),
        f"{prefix}_zcr": _zero_crossing_rate(values),
    }

def _frequency_features(values, prefix, sample_rate):
    """Spectral entropy, dominant frequency, spectral centroid, band powers"""
    if len(values) == 0:
         return {}
    centered = values - np.mean(values)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(values.size, d=1.0 / sample_rate)
    
    return {
        f"{prefix}_spec_entropy": _spectral_entropy(power[1:]),
        f"{prefix}_dom_freq": float(freqs[np.argmax(power[1:])]) if power.size > 1 else 0.0,
        f"{prefix}_spec_centroid": float(np.sum(freqs[1:] * power[1:]) / np.sum(power[1:])) if power.size > 1 and np.sum(power[1:]) > 0 else 0.0,
        f"{prefix}_band_0_05": _band_power(freqs[1:], power[1:], 0.0, 0.5),
        f"{prefix}_band_05_15": _band_power(freqs[1:], power[1:], 0.5, 1.5),
        f"{prefix}_band_15_30": _band_power(freqs[1:], power[1:], 1.5, 3.0),
        f"{prefix}_band_30_50": _band_power(freqs[1:], power[1:], 3.0, 5.0),
    }

def _extract_window_features(window, sample_rate):
    """Compute time + frequency domain features per window"""
    if window.shape[0] == 0:
        return {}
        
    acc_x, acc_y, acc_z = window[:, 0], window[:, 1], window[:, 2]
    rel_angle = window[:, 3] if window.shape[1] > 3 else np.zeros(window.shape[0])
    
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    tilt = np.degrees(np.arctan2(acc_z, np.sqrt(acc_x**2 + acc_y**2) + 1e-8))
    jerk = np.diff(magnitude, prepend=magnitude[0]) * sample_rate
    
    feature_map = {}
    
    for series, prefix in [(acc_x, "ax"), (acc_y, "ay"), (acc_z, "az"), 
                           (magnitude, "mag"), (tilt, "tilt"), (jerk, "jerk")]:
        feature_map.update(_time_features(series, prefix))
        feature_map.update(_frequency_features(series, prefix, sample_rate))
    
    path_length = float(np.sum(np.sqrt(np.diff(acc_x)**2 + np.diff(acc_y)**2 + np.diff(acc_z)**2)))
    net_disp = float(np.sqrt((acc_x[-1]-acc_x[0])**2 + (acc_y[-1]-acc_y[0])**2 + (acc_z[-1]-acc_z[0])**2))
    
    feature_map.update({
        "corr_xy": _safe_corr(acc_x, acc_y),
        "corr_xz": _safe_corr(acc_x, acc_z),
        "corr_yz": _safe_corr(acc_y, acc_z),
        "path_length": path_length,
        "net_displacement": net_disp,
        "path_tortuosity": path_length / (net_disp + 1e-8),
    })
    
    return feature_map

def _build_window_indices_and_labels(data_df, window_size, overlap):
     if len(data_df) == 0:
         return np.array([]), [], np.array([])
         
     step_size = int(window_size * (1 - overlap))
     if step_size < 1: step_size = 1
     
     values = data_df[['accel_x_mps2', 'accel_y_mps2', 'accel_z_mps2', 'relative_angle']].to_numpy() if 'relative_angle' in data_df.columns else data_df[['accel_x_mps2', 'accel_y_mps2', 'accel_z_mps2']].to_numpy()
     labels = data_df['behavior'].to_numpy()
     
     start_indices = list(range(0, len(data_df) - window_size + 1, step_size))
     
     y = []
     for idx in start_indices:
         window_labels = labels[idx:idx+window_size]
         # Use majority voting for label
         unique, counts = np.unique(window_labels, return_counts=True)
         y.append(unique[np.argmax(counts)])
         
     return values, start_indices, np.array(y)

def build_engineered_feature_frame(
    data_df: pd.DataFrame,
    window_size: int = 100,
    overlap: float = 0.5,
    sample_rate: float = 10.0,
    n_jobs: int = 1,
    parallel_backend: str = "loky",
) -> Tuple[pd.DataFrame, np.ndarray]:
    
    values, start_indices, y = _build_window_indices_and_labels(
        data_df, window_size=window_size, overlap=overlap
    )
    
    if len(start_indices) == 0:
        return pd.DataFrame(), y
    
    if n_jobs != 1 and Parallel is not None:
        rows = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
            delayed(_extract_window_features)(
                values[start_idx:start_idx + window_size, :], 
                sample_rate=sample_rate
            ) for start_idx in start_indices
        )
    else:
        rows = [_extract_window_features(
            values[start_idx:start_idx + window_size, :], 
            sample_rate=sample_rate
        ) for start_idx in tqdm(start_indices, desc="Extracting features")]
    
    X_df = pd.DataFrame(rows)
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_df, y

class ImmuFeatureSelector(FeatureSelectionCascade):
    pass
