import argparse
import yaml
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sensor.data.loader import get_splits
from sensor.features.immu import build_engineered_feature_frame as immu_build_features
from sensor.features.immu import ImmuFeatureSelector

# Placeholder for mlflow if installed
try:
    import mlflow
    import mlflow.sklearn  # <--- ADD THIS LINE HERE
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use YAML or JSON.")

def cmb_eval(y_pred, y_test):
    classes = np.unique(y_test)
    acc_dict = {cls: accuracy_score(y_test[y_test == cls], y_pred[y_test == cls]) for cls in classes}
    prec_dict = {cls: val for cls, val in zip(classes, precision_score(y_test, y_pred, average=None, labels=classes, zero_division=0))}
    recal_dict = {cls: val for cls, val in zip(classes, recall_score(y_test, y_pred, average=None, labels=classes, zero_division=0))}
    f1_dict = {cls: val for cls, val in zip(classes, f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0))}
    return acc_dict, prec_dict, recal_dict, f1_dict

def main():
    parser = argparse.ArgumentParser(description="Run Feature Selection Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # MLflow Setup
    if MLFLOW_AVAILABLE and config.get("mlflow", {}).get("enabled", False):
        # Default to the background server running on your shared machine
        default_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", default_uri))
        experiment_name = config["mlflow"].get("experiment_name", "Feature_Selection")
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run(run_name=config.get("modality", "unknown"))
        mlflow.log_params(config)
    else:
        active_run = None
    
    try:
        data_dir = config["data"]["sensor_data_dir"]
        fold_config = config["data"]["fold_config"]
        date = config["data"].get("date", "0725")
        
        # Load Data
        print("Loading data...")
        train_df, val_df, test_df = get_splits(
            data_dir=data_dir,
            fold_config=fold_config,
            date=date,
            pre_loader_func_name=config["data"]["pre_loader"],
            module_name="sensor.data.loader"
        )
        
        modality = config.get("modality", "immu")
        
        # Build Features
        print(f"Building engineered features for {modality}...")
        if modality == "immu":
            from sensor.features.immu import build_engineered_feature_frame as build_features
            from sensor.features.immu import ImmuFeatureSelector as SelectorClass
            
            window_size = config["features"].get("window_size", 100)
            overlap = config["features"].get("overlap", 0.5)
            sample_rate = config["features"].get("sample_rate", 10.0)
            
            X_train, y_train = build_features(train_df, window_size, overlap, sample_rate)
            X_test, y_test = build_features(test_df, window_size, overlap, sample_rate)
            
        elif modality == "uwb":
            from sensor.features.uwb import build_engineered_feature_frame as build_features
            from sensor.features.uwb import UwbFeatureSelector as SelectorClass
            
            lag_steps = tuple(config["features"].get("lag_steps", (1, 2, 3, 4)))
            roll_windows = tuple(config["features"].get("roll_windows", (3, 5)))
            
            X_train, y_train = build_features(train_df, lag_steps=lag_steps, roll_windows=roll_windows)
            X_test, y_test = build_features(test_df, lag_steps=lag_steps, roll_windows=roll_windows)
        
        elif modality == "immu_uwb":
            from sensor.features.immu_uwb import build_engineered_feature_frame as build_features
            from sensor.features.immu_uwb import ImmuUwbFeatureSelector as SelectorClass
            
            lag_steps = tuple(config["features"].get("lag_steps", (1, 2, 3, 4)))
            roll_windows = tuple(config["features"].get("roll_windows", (3, 5)))
            
            X_train, y_train = build_features(train_df, lag_steps=lag_steps, roll_windows=roll_windows)
            X_test, y_test = build_features(test_df, lag_steps=lag_steps, roll_windows=roll_windows)

        elif modality == "uwb_hd_akl":
            from sensor.features.uwb_hd_akl import build_engineered_feature_frame as build_features
            from sensor.features.uwb_hd_akl import UwbHdAklFeatureSelector as SelectorClass
            
            lag_steps = tuple(config["features"].get("lag_steps", (1, 2, 3, 4)))
            roll_windows = tuple(config["features"].get("roll_windows", (3, 5)))
            
            X_train, y_train = build_features(train_df, lag_steps=lag_steps, roll_windows=roll_windows)
            X_test, y_test = build_features(test_df, lag_steps=lag_steps, roll_windows=roll_windows)
        else:
             print(f"Modality {modality} feature builder not implemented yet.")
             return

        selector = SelectorClass(
            pearson_threshold=config["selection"].get("pearson_threshold", 0.9),
            rf_n_estimators=config["selection"].get("rf_n_estimators", 200)
        )

        # Fit Selector
        print("Running feature selection cascade...")
        selector.fit(X_train, y_train)
        score_df = selector.get_score_table()
        
        if score_df.empty:
            print("No features selected.")
            return

        # Save score table
       # Route CSVs and tables to the metrics folder
        metric_out_dir = config.get("output_dir", "outputs/sensor/metrics")
        os.makedirs(metric_out_dir, exist_ok=True)

        score_path = os.path.join(metric_out_dir, f"{modality}_score_table.csv")
        res_path = os.path.join(metric_out_dir, f"{modality}_evaluation.csv")
        
        if MLFLOW_AVAILABLE and active_run:
            mlflow.log_artifact(score_path)

        # Evaluation
        k_values = config["evaluation"].get("k_values", [20, 50, 100, 'all'])
        print(f"Evaluating top k features: {k_values}")
        
        results = []
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

            acc_dict, prec_dict, recal_dict, f1_dict = cmb_eval(y_pred, y_test)
            
            print(f"  k={k}: Macro F1={macro_f1:.4f}, Accuracy={macro_acc:.4f}")
            results.append({"k": k, "macro_f1": macro_f1, "accuracy": macro_acc})
            
            if MLFLOW_AVAILABLE and active_run:
                # Log global metrics
                mlflow.log_metrics({f"f1_k_{k}": macro_f1, f"acc_k_{k}": macro_acc})
                
                # Log per-class F1 scores
                for cls, val in f1_dict.items():
                    mlflow.log_metric(f"f1_class_{cls}_k_{k}", val)
            
            if MLFLOW_AVAILABLE and active_run:
                # This saves the trained Random Forest into the MLflow artifact store
                mlflow.sklearn.log_model(clf, f"random_forest_model_k_{k}")

        # Re-use the metric_out_dir we defined earlier!
        res_path = os.path.join(metric_out_dir, f"{modality}_evaluation.csv")
        res_df.to_csv(res_path, index=False)
        
        if MLFLOW_AVAILABLE and active_run:
            mlflow.log_artifact(res_path)
            
    finally:
        if MLFLOW_AVAILABLE and active_run:
            mlflow.end_run()

if __name__ == "__main__":
    main()
