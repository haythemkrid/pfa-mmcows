"""
A3 — YOLOv8-nano Evaluation
Test set evaluation + global loss column.
"""
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

# === PATHS ===
DATA_YAML = "/home/oussema/pfa-mmcows/store/data/clean/yolo_dataset/dataset.yaml"
WEIGHTS = "/home/oussema/pfa-mmcows/runs/detect/yolov8n_cows/weights/best.pt"
RESULTS_CSV = Path("/home/oussema/pfa-mmcows/runs/detect/yolov8n_cows/results.csv")

# === TEST SET EVALUATION ===
model = YOLO(WEIGHTS)
metrics = model.val(data=DATA_YAML, split="test", device=0)

print("=" * 40)
print("TEST SET RESULTS")
print("=" * 40)
print(f"mAP@0.5:    {metrics.box.map50:.4f}  (target >= 0.80)")
print(f"mAP@0.5-95: {metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")

# === ADD GLOBAL LOSS ===
df_r = pd.read_csv(RESULTS_CSV)
df_r.columns = df_r.columns.str.strip()
df_r["train/global_loss"] = df_r["train/box_loss"] + df_r["train/cls_loss"] + df_r["train/dfl_loss"]
df_r["val/global_loss"] = df_r["val/box_loss"] + df_r["val/cls_loss"] + df_r["val/dfl_loss"]
df_r.to_csv(RESULTS_CSV, index=False)
print(f"\nGlobal loss column added to {RESULTS_CSV}")