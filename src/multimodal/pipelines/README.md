# Training pipeline template

This folder contains a small, documented template for training pipelines and
conventions to follow when implementing new training flows.

Files
- `src/multimodal/pipelines/pipeline_template.py` — A minimal scaffold showing
  the recommended methods and responsibilities for a pipeline.

Purpose
- Provide a simple, testable structure to implement training logic.
- Encourage consistency across pipelines (prepare data, build components,
  run epochs, checkpointing, logging).

Recommended pipeline shape
- Public entrypoint: `__init__(cfg)` and `run()`
- Protected hooks to implement:
  - `_prepare_data() -> (train_loader, val_loader)`
  - `_build_components() -> (model, optimizer, scheduler)`
  - `_train_epoch(model, loader, optimizer) -> Dict[str, float]`
  - `_save_checkpoint(path, state)` and `_load_checkpoint(path, device)`
- Keep MLflow / external integrations behind small helper methods so they
  can be mocked or disabled in tests.

Config expectations
- `cfg.data.*` — data locations, batch size, workers, augmentations
- `cfg.model.*` — model architecture params
- `cfg.training.epochs`, `cfg.training.lr`, `cfg.training.batch_size`, etc.
- `cfg.output.run_dir` — base run directory

Usage example (Hydra)

Run a pipeline that subclasses the template via Hydra:

```bash
python -m src.multimodal.pipelines.my_pipeline_config
# Or with hydra override:
python -m src.multimodal.pipelines.my_pipeline_config --config-name=my_config
```

Notes
- The template is intentionally small — extend it with utilities already in
  `shared.utils` for logging and mlflow handling.
- When adding a new pipeline, copy `pipeline_template.py` and implement the
  protected methods. Add unit tests for `_train_epoch` to cover training logic.

Next steps
- Replace or subclass `TemplateTrainingPipeline` for concrete training code.
- Add a small example implementation in this folder if you want a runnable
  example that trains on a tiny synthetic dataset.
