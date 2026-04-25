import pandas as pd
from typing import Dict, Any

from src.shared.base.pipeline import BasePipeline
from src.shared.utils.logger import logger
from src.shared.utils.mlflow_logger import MLflowLogger
from src.sensor.features.factory import FeatureFactory

class TemplatePipeline(BasePipeline):
    """A template showing how to build a standard ML pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Initialize MLflow logger using shared generic utilities
        self.mlflow_logger = MLflowLogger(config)
        self.modality = config.get("modality", "template_modality")

    def run(self) -> None:
        logger.info("Initializing Template Pipeline...")
        # Start MLflow run
        self.mlflow_logger.start_run()
        
        try:
            # 1. Load Data
            # This is where you would call your dedicated data loader
            logger.info("Loading dataset splits...")
            train_df, test_df = pd.DataFrame(), pd.DataFrame() # Dummy
            
            # 2. Extract Features via Factory
            # We defer all logic dynamically handling the 'modality' to FeatureFactory
            logger.info(f"Building features for {self.modality}...")
            
            # Example call once factory is implemented properly:
            # X_train, y_train, X_test, y_test, selector = FeatureFactory.create(
            #     self.modality, self.config, train_df, test_df
            # )
            
            # 3. Train a dummy model
            logger.info("Training model...")
            dummy_metric = 0.96
            
            # 4. Log results and Metrics using standard logger
            logger.info(f"Evaluation complete. Accuracy = {dummy_metric}")
            self.mlflow_logger.log_metrics({"accuracy": dummy_metric})
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Always close the run to prevent hanging MLflow processes
            self.mlflow_logger.end_run()
            logger.info("Template Pipeline finished.")
