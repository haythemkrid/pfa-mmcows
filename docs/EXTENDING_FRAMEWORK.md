# Extending `pfa-mmcows` Framework
Welcome to the multi-modal machine learning framework for MMCows. This framework is explicitly designed with the **Single Responsibility Principle (SRP)** and high modularity in mind.

To extend the framework securely, use the boilerplate templates included in the codebase.

## 1. Adding a New Sensor Modality
We use a factory pattern `src.sensor.features.factory.FeatureFactory` to handle all new features. Follow these steps:

1. **Copy the Template**
   Copy `src/sensor/features/modalities/_template_sensor.py` and rename it to your new modality: 
   ```bash
   cp src/sensor/features/modalities/_template_sensor.py src/sensor/features/modalities/new_sensor.py
   ```
2. **Implement Interfaces**
   Inside `new_sensor.py`, replace the `TODO` placeholders in:
   - `build_engineered_feature_frame(df, ...)`: Extract X and Y from your data split.
   - `NewSensorFeatureSelector` Class: Supply specific feature filtering implementations.
3. **Register in Factory**
   In `src/sensor/features/factory.py`, follow the commented `HOW TO ADD A NEW SENSOR` directions to hook up your file to the factory dictionary parser.
4. **Update Configuration**
   In your new config (e.g. `configs/new_sensor.yaml`), set the `modality:` parameter to your new string key.

## 2. Integrating a New Pipeline Workflow
All ML logic loops are built from `src.shared.base.pipeline.BasePipeline` extending robust centralized logging metrics.

1. **Use the Pipeline Template**
   Copy `src/sensor/pipelines/_template_pipeline.py`.
2. **Override the `.run()` Method**
   Add your initialization rules. Leverage `src.shared.utils.logger.logger` and `src.shared.utils.mlflow_logger.MLflowLogger` to ensure outputs match standard formatting.
3. **Hook into `main.py`**
   Add an `argparse` command at the root of `main.py` to map custom configurations sequentially to your newly instantiated pipeline framework.
