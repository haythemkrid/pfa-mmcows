"""Train a YOLOv8 model for visual cow detection."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train YOLOv8 on the MMCows visual dataset.")
	parser.add_argument(
		"--data-yaml",
		type=Path,
		default=Path("store/data/raw/visual_data/yolo_nano/mmcows_binary.yaml"),
		help="Path to dataset YAML file.",
	)
	parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model checkpoint.")
	parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
	parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
	parser.add_argument("--batch", type=int, default=16, help="Batch size.")
	parser.add_argument("--device", type=str, default="cpu", help="Training device (cpu, 0, 0,1...).")
	parser.add_argument("--workers", type=int, default=4, help="Data loader worker count.")
	parser.add_argument(
		"--project",
		type=Path,
		default=Path("store/models/visual"),
		help="Directory where YOLO run outputs are saved.",
	)
	parser.add_argument("--name", type=str, default="cow_detection_v8n", help="Run name.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	try:
		from ultralytics import YOLO
	except ImportError as exc:
		raise ImportError("ultralytics is required. Install dependencies from requirements.txt") from exc

	data_yaml = args.data_yaml.resolve()
	if not data_yaml.exists():
		raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

	project_dir = args.project.resolve()
	project_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(args.model)
	print(f"[OK] Starting training with model={args.model}, data={data_yaml}")
	model.train(
		data=str(data_yaml),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		workers=args.workers,
		project=str(project_dir),
		name=args.name,
	)


if __name__ == "__main__":
	main()
