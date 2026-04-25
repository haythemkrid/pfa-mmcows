from typing import Any, Tuple, Dict
import pandas as pd

class FeatureFactory:
    """Factory to retrieve specific feature builders and selectors based on modality."""
    
    @classmethod
    def _get_immu_features(cls, train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict) -> Tuple:
        from src.sensor.features.modalities.immu import build_engineered_feature_frame, ImmuFeatureSelector
        
        window_size = config["features"].get("window_size", 100)
        overlap = config["features"].get("overlap", 0.5)
        sample_rate = config["features"].get("sample_rate", 10.0)
        
        X_train, y_train = build_engineered_feature_frame(train_df, window_size, overlap, sample_rate)
        X_test, y_test = build_engineered_feature_frame(test_df, window_size, overlap, sample_rate)
        return X_train, y_train, X_test, y_test, ImmuFeatureSelector
    
    @classmethod
    def _get_uwb_features(cls, train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict) -> Tuple:
        from src.sensor.features.modalities.uwb import build_engineered_feature_frame, UwbFeatureSelector
        return cls._build_uwb_base(build_engineered_feature_frame, UwbFeatureSelector, train_df, test_df, config)

    @classmethod
    def _get_immu_uwb_features(cls, train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict) -> Tuple:
        from src.sensor.features.modalities.immu_uwb import build_engineered_feature_frame, ImmuUwbFeatureSelector
        return cls._build_uwb_base(build_engineered_feature_frame, ImmuUwbFeatureSelector, train_df, test_df, config)
        
    @classmethod
    def _get_uwb_hd_akl_features(cls, train_df: pd.DataFrame, test_df: pd.DataFrame, config: Dict) -> Tuple:
        from src.sensor.features.modalities.uwb_hd_akl import build_engineered_feature_frame, UwbHdAklFeatureSelector
        return cls._build_uwb_base(build_engineered_feature_frame, UwbHdAklFeatureSelector, train_df, test_df, config)
        
    @classmethod
    def _build_uwb_base(cls, build_func, selector_cls, train_df, test_df, config):
        lag_steps = tuple(config["features"].get("lag_steps", (1, 2, 3, 4)))
        roll_windows = tuple(config["features"].get("roll_windows", (3, 5)))
        
        X_train, y_train = build_func(train_df, lag_steps=lag_steps, roll_windows=roll_windows)
        X_test, y_test = build_func(test_df, lag_steps=lag_steps, roll_windows=roll_windows)
        return X_train, y_train, X_test, y_test, selector_cls

    @classmethod
    def create(cls, modality: str, config: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any]:
        """Returns the pre-processed train/test datasets and the initialized Selector object."""
        
        # HOW TO ADD A NEW SENSOR:
        # 1. Copy `modalities/_template_sensor.py` to `modalities/new_sensor.py` and implement logic.
        # 2. Add a new @classmethod here (e.g., `_get_new_sensor_features`) to map config args to the build function.
        # 3. Add the string key (e.g., 'new_sensor') to the dictionary mapping below.

        builders = {
            "immu": cls._get_immu_features,
            "uwb": cls._get_uwb_features,
            "immu_uwb": cls._get_immu_uwb_features,
            "uwb_hd_akl": cls._get_uwb_hd_akl_features
        }
        
        if modality not in builders:
            raise NotImplementedError(f"Modality {modality} feature builder not implemented yet.")
        
        X_train, y_train, X_test, y_test, SelectorClass = builders[modality](train_df, test_df, config)
        
        pearson_threshold = config["selection"].get("pearson_threshold", 0.9)
        rf_n_estimators = config["selection"].get("rf_n_estimators", 200)
        
        selector = SelectorClass(
            pearson_threshold=pearson_threshold,
            rf_n_estimators=rf_n_estimators
        )
        
        return X_train, y_train, X_test, y_test, selector
