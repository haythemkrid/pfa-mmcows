import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any
from sklearn.feature_selection import VarianceThreshold, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

class FeatureSelectionCascade:
    """3-stage feature selection: Variance -> Pearson correlation -> Scoring"""
    
    def __init__(
        self,
        pearson_threshold: float = 0.9,
        variance_threshold: float = 1e-10,
        random_state: int = 42,
        rf_n_estimators: int = 300,
    ):
        self.pearson_threshold = pearson_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.rf_n_estimators = rf_n_estimators
        self.score_table_: pd.DataFrame = None
        
        self.variance_selector_ = None
        self.variance_columns_: List[str] = []
        self.pearson_drop_columns_: List[str] = []
        self.selected_columns_: List[str] = []
        self.input_columns_: List[str] = []
    
    def fit(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]) -> 'FeatureSelectionCascade':
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        
        self.input_columns_ = X_train.columns.tolist()
        y_train = np.asarray(y_train, dtype=int)
        
        # STAGE 1: Variance Threshold
        self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
        X_var_values = self.variance_selector_.fit_transform(X_train)
        variance_mask = self.variance_selector_.get_support()
        self.variance_columns_ = list(np.array(self.input_columns_)[variance_mask])
        
        X_var = pd.DataFrame(X_var_values, columns=self.variance_columns_, index=X_train.index)
        
        # STAGE 2: Pearson Correlation Pruning
        corr = X_var.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.pearson_drop_columns_ = [
            col for col in upper.columns 
            if np.any(upper[col] > self.pearson_threshold)
        ]
        
        X_pruned = X_var.drop(columns=self.pearson_drop_columns_, errors="ignore")
        self.selected_columns_ = X_pruned.columns.tolist()
        X_pruned_values = X_pruned.to_numpy(dtype=float)
        
        if len(self.selected_columns_) == 0:
            print("Warning: All features dropped.")
            self.score_table_ = pd.DataFrame()
            return self

        # STAGE 3: Score Features
        scaled_for_chi2 = MinMaxScaler().fit_transform(X_pruned_values)
        chi2_scores, _ = chi2(scaled_for_chi2, y_train)
        
        anova_scores, _ = f_classif(X_pruned_values, y_train)
        
        rf_selector = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        rf_selector.fit(X_pruned, y_train)
        mdi_scores = rf_selector.feature_importances_
        
        for scores in [chi2_scores, anova_scores, mdi_scores]:
            np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        
        def _descending_rank(scores):
            order = np.argsort(-scores)
            rank = np.empty_like(order, dtype=float)
            rank[order] = np.arange(1, len(scores) + 1, dtype=float)
            return rank
        
        rank_chi2 = _descending_rank(chi2_scores)
        rank_anova = _descending_rank(anova_scores)
        rank_mdi = _descending_rank(mdi_scores)
        aggregate_rank = rank_chi2 + rank_anova + rank_mdi
        
        score_table = pd.DataFrame({
            "feature": self.selected_columns_,
            "chi2_score": chi2_scores,
            "anova_score": anova_scores,
            "mdi_score": mdi_scores,
            "rank_chi2": rank_chi2,
            "rank_anova": rank_anova,
            "rank_mdi": rank_mdi,
            "aggregate_rank": aggregate_rank,
        })
        
        self.score_table_ = score_table.sort_values(
            by=["aggregate_rank", "rank_mdi", "mdi_score"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply variance threshold and pearson pruning"""
        X_var = X[self.variance_columns_]
        return X_var.drop(columns=self.pearson_drop_columns_, errors="ignore")

    def select_top_k(self, X: pd.DataFrame, k: Union[int, str]) -> pd.DataFrame:
        """Select top k features by aggregate rank"""
        if self.score_table_ is None or self.score_table_.empty:
            return pd.DataFrame()
            
        X_base = self.transform(X)
        if isinstance(k, str) and k.lower() == "all":
            return X_base
        
        k_int = int(k)
        selected_features = self.get_ranked_features()[:k_int]
        return X_base.reindex(columns=selected_features, fill_value=0.0)
    
    def get_ranked_features(self) -> List[str]:
         if self.score_table_ is None or self.score_table_.empty:
            return []
         return self.score_table_["feature"].tolist()
    
    def get_score_table(self) -> pd.DataFrame:
         if self.score_table_ is None:
             return pd.DataFrame()
         return self.score_table_.copy()
