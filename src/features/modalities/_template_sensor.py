import pandas as pd
from typing import Tuple

def build_engineered_feature_frame(
    df: pd.DataFrame, 
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features from raw sensor DataFrame.
    
    Args:
        df (pd.DataFrame): Raw sensor data.
        **kwargs: Extra parameters from config.
    
    Returns:
        X (pd.DataFrame): Engineered feature matrix (n_samples, n_features).
        y (pd.Series): Target labels (n_samples,).
    """
    # TODO: Implement feature engineering logic here
    X = pd.DataFrame()
    y = pd.Series(dtype=int)
    return X, y

class TemplateFeatureSelector:
    """
    Dummy template for feature selection.
    Any new sensor feature selector must implement `fit`, `get_score_table`, and `select_top_k`.
    """
    def __init__(self, pearson_threshold: float = 0.9, rf_n_estimators: int = 200, **kwargs):
        self.pearson_threshold = pearson_threshold
        self.rf_n_estimators = rf_n_estimators
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the feature selection algorithms to the training data."""
        # TODO: Implement selector fitting (e.g., correlation filter, random forest importances)
        pass

    def get_score_table(self) -> pd.DataFrame:
        """Return the feature importance scores."""
        # TODO: Return a structured DataFrame containing feature names and their objective scores
        return pd.DataFrame()

    def select_top_k(self, X: pd.DataFrame, k: int) -> pd.DataFrame:
        """Filter the dataset down to the top K features."""
        # TODO: Filter X down to optimal subsets
        return X
