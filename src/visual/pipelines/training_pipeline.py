"""Run the end-to-end visual pipeline: index -> YOLO prep -> train."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run full visual training pipeline.")
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path("store/data/raw/visual_data"),
		help="Root of visual dataset.",
	)
	parser.add_argument("--date-folder", type=str, default="0725")
	parser.add_argument("--cameras", type=str, default="cam_1,cam_2,cam_3,cam_4")
	parser.add_argument("--split-ratios", type=str, default="0.70,0.15,0.15")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--remap-labels-in-place", action="store_true")
	parser.add_argument("--model", type=str, default="yolov8n.pt")
	parser.add_argument("--epochs", type=int, default=50)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--run-name", type=str, default="cow_detection_v8n")
	return parser.parse_args()


def run_command(command: list[str]) -> None:
	print("[RUN]", " ".join(command))
	result = subprocess.run(command, check=False)
	if result.returncode != 0:
		raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main() -> None:
	args = parse_args()
	dataset_root = args.dataset_root.resolve()

	a1_script = Path(__file__).resolve().parents[1] / "data" / "a1_build_index.py"
	a2_script = Path(__file__).resolve().parents[1] / "data" / "a2_yolo_binary.py"
	train_script = Path(__file__).resolve().parents[1] / "models" / "train.py"

	run_command(
		[
			sys.executable,
			str(a1_script),
			"--dataset-root",
			str(dataset_root),
			"--date-folder",
			args.date_folder,
			"--cameras",
			args.cameras,
			"--split-ratios",
			args.split_ratios,
			"--seed",
			str(args.seed),
		]
	)

	a2_cmd = [
		sys.executable,
		str(a2_script),
		"--dataset-root",
		str(dataset_root),
	]
	if args.remap_labels_in_place:
		a2_cmd.append("--remap-labels-in-place")
	run_command(a2_cmd)

	run_command(
		[
			sys.executable,
			str(train_script),
			"--data-yaml",
			str((dataset_root / "yolo_nano" / "mmcows_binary.yaml").resolve()),
			"--model",
			args.model,
			"--epochs",
			str(args.epochs),
			"--imgsz",
			str(args.imgsz),
			"--batch",
			str(args.batch),
			"--device",
			args.device,
			"--workers",
			str(args.workers),
			"--name",
			args.run_name,
		]
	)

	print("[OK] Visual training pipeline completed")


if __name__ == "__main__":
	main()
