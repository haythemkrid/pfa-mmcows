# pfa-mmcows

Projet ML structure pour MMCows (sensor + visual + fusion).

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

## Migration visual depuis mmcows-visual

Le travail visual a ete migre dans ce repo, sans modifier le repo source.

Scripts principaux migres:
- src/visual/data/a1_build_index.py
- src/visual/data/a2_yolo_binary.py
- src/visual/data/remap_labels_single_class.py
- src/visual/models/train.py
- src/visual/pipelines/training_pipeline.py

## Prerequis

1. Installer les dependances:

```bash
pip install -r requirements.txt
```

2. Verifier que les donnees visual sont sous:

```text
store/data/raw/visual_data/
	images/0725/cam_1 ... cam_4
	labels/0725/cam_1 ... cam_4
```

## Execution rapide

1. Construire index dataset:

```bash
make visual-index
```

2. Generer artefacts YOLO (train/val/test + yaml):

```bash
make visual-yolo
```

3. Entrainer YOLOv8:

```bash
make visual-train
```

Ou tout en une commande:

```bash
make visual-pipeline
```

## DVC

Le pipeline DVC inclut 3 stages visual:
- visual_build_index
- visual_prepare_yolo
- visual_train

Lancer:

```bash
dvc repro
```




# Running the MBT Training Pipeline

## 1. Install dependencies

```bash
cd pfa-mmcows
pip install -r requirements_mbt.txt
```

---

## 2. Protect your session (critical for long runs)

The training takes multiple days.  Wrap it in **tmux** so that a
disconnect never kills the process.

```bash
# Start a named session
tmux new -s mbt_train

# Inside the tmux session, run the pipeline (see step 4).
# To detach without killing the process:  Ctrl+B  then  D
# To reattach from any terminal later:
tmux attach -t mbt_train
```

If tmux is unavailable, use `nohup`:
```bash
nohup python -m src.multimodal.pipelines.training_pipeline \
    > logs/mbt_runs/nohup.out 2>&1 &
echo $!   # save this PID — use  kill <PID>  to stop gracefully
```

---

## 3. Directory layout after the pipeline runs

```
pfa-mmcows/
└── logs/
    ├── mlflow/                        ← MLflow artefact store
    └── mbt_runs/
        └── 20250424_093012/           ← one directory per run
            ├── config.yaml            ← exact config used (for reproducibility)
            ├── train.log              ← structured JSON log (one event per line)
            ├── fold_1/
            │   ├── best.pt            ← best-F1 checkpoint for this fold
            │   └── last.pt            ← last-epoch checkpoint (for resuming)
            ├── fold_2/
            │   ├── best.pt
            │   └── last.pt
            ...
```

---

## 4. First run

```bash
# From the repo root
python -m src.multimodal.pipelines.training_pipeline
```

Hydra writes its own output to `outputs/` by default.  All training
artefacts go to `logs/mbt_runs/<timestamp>/`.

---

## 5. Override any config value on the command line

```bash
# Change learning rate and number of bottleneck tokens
python -m src.multimodal.pipelines.training_pipeline \
    training.lr=3e-4 \
    model.num_bottleneck_tokens=8 \
    training.batch_size=16

# Use temporal split instead of object-wise
python -m src.multimodal.pipelines.training_pipeline \
    data.split_type=s2

# Disable MLflow (useful for quick debug runs)
python -m src.multimodal.pipelines.training_pipeline \
    output.use_mlflow=false \
    training.epochs=2
```

---

## 6. Resume after a crash or disconnect

The pipeline automatically resumes.  Just re-run with the **same run_name**:

```bash
python -m src.multimodal.pipelines.training_pipeline \
    experiment.run_name=20250424_093012
```

What happens:
- Folds whose `best.pt` has `completed: true` are **skipped entirely**.
- The interrupted fold resumes from `last.pt` (saved every epoch).
- All log entries are **appended** to `train.log` (never overwritten).

---

## 7. Monitor training in real time

```bash
# Stream the structured log and pretty-print each JSON line
tail -f logs/mbt_runs/<run_name>/train.log | python -m json.tool

# Filter only epoch events
tail -f logs/mbt_runs/<run_name>/train.log | grep '"event": "epoch"'

# Watch val F1 per fold
tail -f logs/mbt_runs/<run_name>/train.log \
    | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if r.get('event') == 'epoch':
        print(f\"fold={r['fold']}  epoch={r['epoch']:3d}  \
val_f1={r['val_f1']:.4f}  val_loss={r['val_loss']:.5f}\")
"
```

Launch the MLflow UI (in a separate tmux pane):
```bash
mlflow ui --backend-store-uri logs/mlflow --port 5000
# Then open http://localhost:5000 in VS Code's port-forward tab
```

---

## 8. Load a saved checkpoint for inference

```python
import torch
from src.multimodal.models.mbt import MultimodalBottleneckTransformer
from omegaconf import OmegaConf

cfg = OmegaConf.load("logs/mbt_runs/<run_name>/config.yaml")
model = MultimodalBottleneckTransformer.from_config(cfg.model)

ckpt = torch.load("logs/mbt_runs/<run_name>/fold_1/best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Loaded best model from fold_1  (val F1 = {ckpt['best_f1']:.4f})")
```

---

## 9. Quick smoke-test (no data needed)

Run for 2 epochs on CPU to verify the full pipeline compiles and saves
checkpoints correctly:

```bash
python -m src.multimodal.pipelines.training_pipeline \
    training.epochs=2 \
    training.batch_size=4 \
    data.num_workers=0 \
    output.use_mlflow=false \
    model.visual_pretrained=false \
    experiment.run_name=smoke_test
```

---

## 10. Run multiple MBT experiments from one command

You can now orchestrate single or batch multimodal runs through `main.py`
and Make targets.

### Single experiment

With `main.py`:

```bash
python main.py multimodal \
    --run-name mbt_single \
    --overrides training.epochs=2 output.use_mlflow=false
```

With Make:

```bash
make mbt-run \
    MBT_RUN_NAME=mbt_single \
    MBT_OVERRIDES="training.epochs=2 output.use_mlflow=false"
```

### Batch experiments (comma-separated run names)

With `main.py`:

```bash
python main.py multimodal-batch \
    --experiments mbt_baseline,mbt_s2,mbt_lr3e4 \
    --base-overrides training.epochs=5 output.use_mlflow=true
```

With Make:

```bash
make mbt-multi \
    MBT_BASE_OVERRIDES="training.epochs=5 output.use_mlflow=true"
```

By default, `make mbt-multi` loads experiments from
`configs/mbt_experiments.yaml`.

If you want comma-separated names instead, clear `MBT_EXPERIMENTS_FILE`:

```bash
make mbt-multi \
    MBT_EXPERIMENTS_FILE= \
    MBT_EXPERIMENTS=mbt_baseline,mbt_s2,mbt_lr3e4 \
    MBT_BASE_OVERRIDES="training.epochs=5 output.use_mlflow=true"
```

### Batch experiments from YAML/JSON file

Command:

```bash
python main.py multimodal-batch \
    --experiments-file configs/mbt_experiments.yaml \
    --base-overrides training.batch_size=16 \
    --stop-on-error
```

Accepted file shapes:

```yaml
experiments:
    - name: mbt_baseline
    - name: mbt_s2
        overrides:
            data.split_type: s2
    - name: mbt_lr3e4
        overrides:
            training.lr: 3e-4
```

or:

```json
[
    {"name": "mbt_baseline"},
    {"name": "mbt_s2", "overrides": ["data.split_type=s2"]},
    {"name": "mbt_lr3e4", "overrides": {"training.lr": "3e-4"}}
]
```
