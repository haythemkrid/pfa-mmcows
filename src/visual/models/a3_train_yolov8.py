"""
A3 — YOLOv8-nano Fine-tuning
Train binary cow detector on merged 4-camera dataset.
Optimized for RTX 5060 Ti (8 GB VRAM).
"""
from ultralytics import YOLO

# === PATHS ===
DATA_YAML = "/home/oussema/pfa-mmcows/store/data/clean/yolo_dataset/dataset.yaml"
PROJECT = "/home/oussema/pfa-mmcows/runs/detect"
NAME = "yolov8n_cows"

# === TRAIN ===
model = YOLO("yolov8n.pt")

model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    amp=True,
    project=PROJECT,
    name=NAME,
    patience=10,
    save=True,
    plots=True,
)