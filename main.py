import argparse
import sys
import os
from dotenv import load_dotenv

from src.shared.utils.config import load_config
from src.shared.utils.logger import logger

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Multi-Modal ML Framework Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")
    
    # Sensor command
    sensor_parser = subparsers.add_parser("sensor", help="Run sensor feature selection experiment")
    sensor_parser.add_argument("--config", type=str, required=True, help="Path to config file for sensor")
    
    # Visual command
    visual_parser = subparsers.add_parser("visual", help="Run visual model training")
    visual_parser.add_argument("--config", type=str, required=True, help="Path to config file for visual")

    # Fusion command
    fusion_parser = subparsers.add_parser("fusion", help="Run multi-modal fusion")
    fusion_parser.add_argument("--config", type=str, required=True, help="Path to config file for fusion")
    
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
        
    if args.command == "sensor":
        logger.info("Initializing Sensor ExperimentPipeline...")
        from src.sensor.pipelines.experiment_pipeline import ExperimentPipeline
        pipeline = ExperimentPipeline(config)
        pipeline.run()
        
    elif args.command == "visual":
        logger.info("Initializing VisualTrainingPipeline...")
        from src.visual.pipelines.training_pipeline import VisualTrainingPipeline
        pipeline = VisualTrainingPipeline(config)
        pipeline.run()

    elif args.command == "fusion":
        logger.info("Initializing Fusion pipeline...")
        # from src.fusion.pipelines.fusion_pipeline import FusionPipeline
        # pipeline = FusionPipeline(config)
        # pipeline.run()
        logger.info("Fusion pipeline is not yet implemented.")

if __name__ == "__main__":
    main()
