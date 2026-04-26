import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class MLflowLogger:
    """Handles MLflow initialization and logging of parameters, metrics, and artifacts."""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = MLFLOW_AVAILABLE and config.get("mlflow", {}).get("enabled", False)
        self.config = config
        self.active_run = None
        load_dotenv()
        
        if self.enabled:
            # Resolve tracking URI with a remote-only default.
            default_uri = "https://dagshub.com/haythemkrid/pfa-mmcows.mlflow"
            tracking_uri = (
                config.get("mlflow", {}).get("tracking_uri")
                or os.environ.get("MLFLOW_TRACKING_URI")
                or default_uri
            )

            # Guard against local MLflow stores; this project uses remote tracking only.
            local_markers = ("file:", "sqlite:", "runs/mlflow", "./mlruns", "mlruns")
            if str(tracking_uri).startswith(local_markers) or str(tracking_uri).startswith("/"):
                raise ValueError(
                    "Local MLflow tracking URI is not allowed in this project. "
                    "Use your remote tracking URI (e.g. DagsHub)."
                )
            
            # (Optional) A quick sanity check to warn you if you forgot to export your tokens in the terminal!
            if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
                print("⚠️ WARNING: MLflow username or password environment variables are missing. DagsHub might reject the connection!")
            
            # 2. Set the Tracking URI
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
            
            # 3. Set the Experiment Name
            experiment_name = config.get("mlflow", {}).get("experiment_name", "Feature_Selection")
            mlflow.set_experiment(experiment_name)
            print(f"MLflow experiment set to: {experiment_name}")

    @staticmethod
    def _flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flattens nested dictionaries for MLflow parameter logging."""
        flat: Dict[str, Any] = {}
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
            if isinstance(value, dict):
                flat.update(MLflowLogger._flatten_dict(value, new_key, sep))
            else:
                flat[new_key] = value
        return flat

    def start_run(self, run_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        """Starts an MLflow run and optionally logs flattened parameters."""
        if self.enabled:
            default_name = self.config.get("modality", "unknown")
            self.active_run = mlflow.start_run(run_name=run_name or default_name)

            to_log = params if params is not None else self.config
            self.log_params(to_log)
        return self.active_run

    def log_artifact(self, local_path: str):
        """Logs an artifact such as a file to MLflow."""
        if self.enabled and self.active_run:
            mlflow.log_artifact(local_path)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Logs multiple metrics to MLflow, optionally at a specific step."""
        if self.enabled and self.active_run:
            if step is None:
                mlflow.log_metrics(metrics)
            else:
                mlflow.log_metrics(metrics, step=step)
            
    def end_run(self):
        """Ends the active MLflow run."""
        if self.enabled and self.active_run:
            mlflow.end_run()
    
    def log_params(self, params: dict) -> None:
        """Logs flattened parameters to MLflow."""
        if self.enabled and self.active_run:
            flat_params = self._flatten_dict(params)
            filtered: Dict[str, Any] = {}

            for key, value in flat_params.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    filtered[key] = value
                else:
                    filtered[key] = str(value)

            if filtered:
                mlflow.log_params(filtered)