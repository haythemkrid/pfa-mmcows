import os
from typing import Dict, Any
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

    def start_run(self):
        """Starts an MLflow run and logs configuration parameters."""
        if self.enabled:
            run_name = self.config.get("modality", "unknown")
            self.active_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(self.config)
        return self.active_run

    def log_artifact(self, local_path: str):
        """Logs an artifact such as a file to MLflow."""
        if self.enabled and self.active_run:
            mlflow.log_artifact(local_path)

    def log_metrics(self, metrics: Dict[str, Any]):
        """Logs multiple metrics to MLflow."""
        if self.enabled and self.active_run:
            mlflow.log_metrics(metrics)
            
    def end_run(self):
        """Ends the active MLflow run."""
        if self.enabled and self.active_run:
            mlflow.end_run()
    
    def log_params(self, params: dict) -> None:
        """Logs a dictionary of parameters to MLflow."""
        if self.enabled:
            mlflow.log_params(params)