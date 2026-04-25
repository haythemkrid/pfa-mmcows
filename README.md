# pfa-mmcows

Projet ML structure pour MMCows (sensor + visual + fusion).

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
