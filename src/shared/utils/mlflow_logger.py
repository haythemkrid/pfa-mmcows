import os
from typing import Dict, Any

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
        
        if self.enabled:
            default_uri = os.environ.get("MLFLOW_DATABASE_URL", "http://127.0.0.1:5000")
            mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", default_uri))
            experiment_name = config["mlflow"].get("experiment_name", "Feature_Selection")
            mlflow.set_experiment(experiment_name)

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
        import mlflow
        mlflow.log_params(params)
