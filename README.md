# pfa-mmcows

Projet ML structure pour MMCows (sensor + visual + fusion).

# pfa-mmcows

Project for multimodal cow behaviour analysis (sensor + visual + fusion).

This repository contains data processing, models, and training pipelines
for the MMCows multimodal project. It includes:

- sensor data loaders and preprocessing (UWB/IMU)
- visual dataset indexing and YOLO-based visual pipelines
- multimodal fusion models (Multimodal Bottleneck Transformer — MBT)
- reusable training pipeline scaffolds and Hydra-based configs
- utilities for logging, MLflow integration and resumable training

This README is organised by concern so you can quickly find what you
need to run experiments, implement new pipelines, or extend the codebase.

Contents
- [Prerequisites](#prerequisites)
- [Repository layout](#repository-layout)
- [Pipeline template & conventions](#pipeline-template--conventions)
- [Quickstart: Visual pipeline](#quickstart-visual-pipeline)
- [Quickstart: MBT multimodal training](#quickstart-mbt-multimodal-training)
- [Hydra usage & common overrides](#hydra-usage--common-overrides)
- [Resuming, checkpoints and MLflow](#resuming-checkpoints-and-mlflow)
- [Development: testing and examples](#development-testing-and-examples)


## Prerequisites

1. Create and activate your Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Some pipelines use additional requirements. See individual
   folders or `requirements_mbt.txt` for MBT-specific deps.


## Repository layout

Top-level highlights:

- `src/` — code for data loaders, models, pipelines and utilities
- `configs/` — Hydra / experiment configs
- `store/` — data storage (raw, clean, features)
- `logs/` — MLflow and pipeline run outputs
- `scripts/`, `Makefile` — convenient helpers and Make targets


## Pipeline template & conventions

We provide a small scaffold to keep training pipelines consistent and
testable: see `src/multimodal/pipelines/pipeline_template.py` which defines a
`TemplateTrainingPipeline` with explicit hooks to implement:

- `_prepare_data()` — return `(train_loader, val_loader)`
- `_build_components()` — build and return `(model, optimizer, scheduler)`
- `_train_epoch(model, loader, optimizer)` — run a single epoch and return metrics
- `_save_checkpoint(path, state)` and `_load_checkpoint(path, device)`

Guidelines:
- Keep `run()` thin: orchestration only; heavy logic should be in protected methods
- Keep MLflow / external integrations behind small helper methods so they can be mocked
- Save `last.pt` every epoch and `best.pt` when validation improves; mark completed runs

Follow the template by copying `pipeline_template.py` and implementing the
protected methods for your new pipeline. A README copy is located at
`src/multimodal/pipelines/README.md` with additional tips.


## Quickstart: Visual pipeline

Examples for the visual (YOLO) pipelines are provided under `src/visual`.

Typical commands (Make targets wrap these):

```bash
# build dataset index
make visual-index

# prepare YOLO artifacts (train/val/test splits and YAML)
make visual-yolo

# train YOLO (uses configs in configs/)
make visual-train

# run the whole visual pipeline
make visual-pipeline
```

Data layout expected for visual assets:

```
store/data/raw/visual_data/
    images/...
    labels/...
```


## Quickstart: MBT multimodal training

The main MBT training entrypoint is `src/multimodal/pipelines/training_pipeline.py`.

Install MBT-specific dependencies if required:

```bash
pip install -r requirements_mbt.txt
```

Run a default training run (Hydra-managed config):

```bash
python -m src.multimodal.pipelines.training_pipeline
```

Run a quick smoke-test (CPU, 2 epochs):

```bash
python -m src.multimodal.pipelines.training_pipeline \
    training.epochs=2 \
    training.batch_size=4 \
    data.num_workers=0 \
    output.use_mlflow=false \
    model.visual_pretrained=false \
    experiment.run_name=smoke_test
```

Long-running runs: run inside `tmux` or `nohup` so disconnects do not kill the job.


## Hydra usage & common overrides

All pipelines are configurable via Hydra. Override config values on the
command-line using the `key=value` syntax. Examples:

```bash
python -m src.multimodal.pipelines.training_pipeline \
    training.lr=3e-4 \
    model.num_bottleneck_tokens=8 \
    training.batch_size=16

python -m src.multimodal.pipelines.training_pipeline \
    data.split_type=s2

python -m src.multimodal.pipelines.training_pipeline \
    output.use_mlflow=false \
    training.epochs=2
```


## Resuming, checkpoints and MLflow

- The MBT pipeline saves `last.pt` each epoch and `best.pt` when validation
  improves. `best.pt` is annotated with `completed: true` when a fold finishes.
- To resume, re-run with the same `experiment.run_name` — the pipeline will
  skip folds already completed and resume from the appropriate `last.pt`.
- MLflow integration is optional and controlled by `output.use_mlflow` in the
  config. Credentials can be provided via a `.env` file or environment vars.

Monitor logs at `logs/mbt_runs/<run_name>/train.log`; MLflow UI can be served
from `logs/mlflow` with `mlflow ui --backend-store-uri logs/mlflow`.


## Development: testing and examples

- Add unit tests for `_train_epoch` implementations to cover core training
  behaviour (loss/backprop, metric calculation, gradient clipping).
- Consider adding a small runnable example pipeline in
  `src/multimodal/pipelines/examples/` that trains on synthetic data for CI.


## Contributing

1. Open an issue describing the change.
2. Create a feature branch and a clear PR, referencing the issue.
3. Add/adjust tests where appropriate.


---

If you'd like, I can now:

1. Replace the pipeline-folder README into the main README (migrate content),
2. Add a concrete example pipeline that subclasses the template and runs on synthetic data,
3. Add a small test that runs the smoke-test command in CI.

Tell me which of the above you'd like me to do next.
    if r.get('event') == 'epoch':
