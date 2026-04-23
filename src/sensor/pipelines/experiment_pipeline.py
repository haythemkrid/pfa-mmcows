from src.shared.utils.logger import logger
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.shared.base.pipeline import BasePipeline
from src.shared.utils.config import load_config
from src.shared.utils.mlflow_logger import MLflowLogger

from src.sensor.data.loader import get_splits
from src.sensor.features.factory import FeatureFactory

class ExperimentPipeline(BasePipeline):
    """Core orchestrator for running machine learning feature selection experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mlflow_logger = MLflowLogger(config)
        self.modality = config.get("modality", "immu")
        
    def _evaluate_metrics(self, y_pred, y_test) -> Tuple[Dict, Dict, Dict, Dict]:
        """Runs scikit-learn metrics for evaluation."""
        classes = np.unique(y_test)
        acc_dict = {cls: accuracy_score(y_test[y_test == cls], y_pred[y_test == cls]) for cls in classes}
        prec_dict = {cls: val for cls, val in zip(classes, precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0))}
        recal_dict = {cls: val for cls, val in zip(classes, recall_score(y_test, y_pred, average=None, labels=classes, zero_division=0))}
        f1_dict = {cls: val for cls, val in zip(classes, f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0))}
        return acc_dict, prec_dict, recal_dict, f1_dict

    def run(self) -> None:
        """Executes the full pipeline: load data, build features, select features, and evaluate."""
        self.mlflow_logger.start_run()
        
        try:
            # 1. LOG PARAMETERS IMMEDIATELY (Keeps DB awake & saves config)
            logger.info("Logging initial experiment parameters to MLflow...")
            self.mlflow_logger.log_params({
                "modality": self.modality,
                "date": self.config["data"].get("date", "0725"),
                "window_size": self.config.get("features", {}).get("window_size", "unknown"),
                "overlap": self.config.get("features", {}).get("overlap", "unknown"),
                "rf_n_estimators": self.config.get("selection", {}).get("rf_n_estimators", 200)
            })
            
            # 2. START HEAVY LIFTING
            logger.info("Loading data for sensor...")
            data_dir = self.config["data"]["sensor_data_dir"]
            fold_config_raw = self.config["data"]["fold_config"]
            
            if isinstance(fold_config_raw, str):
                loaded_json = load_config(fold_config_raw)
                if "folds" in loaded_json and "fold_1" in loaded_json["folds"]:
                    fold_config = loaded_json["folds"]["fold_1"]
                    logger.info(f"Loaded fold_1 from {fold_config_raw}")
                else:
                    raise ValueError(f"Could not find 'folds' -> 'fold_1' in {fold_config_raw}")
            else:
                fold_config = fold_config_raw

            date = self.config["data"].get("date", "0725")
            
            train_df, val_df, test_df = get_splits(
                data_dir=data_dir,
                fold_config=fold_config,
                date=date,
                pre_loader_func_name=self.config["data"]["pre_loader"],
                module_name="src.sensor.data.loader"
            )

            logger.info(f"Building engineered features for {self.modality}... (This may take a while)")
            X_train, y_train, X_test, y_test, selector = FeatureFactory.create(
                self.modality, self.config, train_df, test_df
            )
            
            logger.info("Running feature selection cascade...")
            selector.fit(X_train, y_train)
            score_df = selector.get_score_table()
            
            if score_df.empty:
                logger.info("No features selected.")
                return

            metric_out_dir = self.config.get("output_dir", "outputs/sensor/metrics")
            os.makedirs(metric_out_dir, exist_ok=True)
            score_path = os.path.join(metric_out_dir, f"{self.modality}_score_table.csv")
            score_df.to_csv(score_path, index=False)
            
            self.mlflow_logger.log_artifact(score_path)

            k_values = self.config["evaluation"].get("k_values", [20, 50, 100, 'all'])
            logger.info(f"Evaluating top k features: {k_values}")
            
            for k in k_values:
                X_train_k = selector.select_top_k(X_train, k)
                X_test_k = selector.select_top_k(X_test, k)
                
                clf = RandomForestClassifier(
                    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
                )
                clf.fit(X_train_k, y_train)
                y_pred = clf.predict(X_test_k)
                
                macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                macro_acc = accuracy_score(y_test, y_pred)
                # Ensure _evaluate_metrics is defined in your class
                acc_dict, prec_dict, recal_dict, f1_dict = self._evaluate_metrics(y_pred, y_test) 
                
                logger.info(f"  k={k}: Macro F1={macro_f1:.4f}, Accuracy={macro_acc:.4f}")
                
                self.mlflow_logger.log_metrics({f"f1_k_{k}": macro_f1, f"acc_k_{k}": macro_acc})

                # NEW: Log individual class F1 scores
                class_metrics = {}
                for cls_name, f1_val in f1_dict.items():
                    class_metrics[f"class_{cls_name}_f1_k_{k}"] = f1_val
                self.mlflow_logger.log_metrics(class_metrics)
                
        finally:
            self.mlflow_logger.end_run()