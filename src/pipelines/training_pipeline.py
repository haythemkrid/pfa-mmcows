from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from src.shared.base.pipeline import BasePipeline
from src.shared.utils.logger import logger
from src.shared.utils.mlflow_logger import MLflowLogger


def _run_step(step_name: str, command: list[str]) -> None:
    logger.info("[%s] Running: %s", step_name, " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the visual training pipeline (index -> yolo -> train).")
    parser.add_argument("--dataset-root", type=Path, default=Path("store/data/raw/visual_data"))
    parser.add_argument("--date-folder", type=str, default="0725")
    parser.add_argument("--cameras", type=str, default="cam_1,cam_2,cam_3,cam_4")
    parser.add_argument("--split-ratios", type=str, default="0.70,0.15,0.15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="cow_detection_v8n")
    parser.add_argument("--experiment-name", type=str, default="Visual_Training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    return parser.parse_args()


def run_visual_pipeline(
    dataset_root: Path,
    date_folder: str,
    cameras: str,
    split_ratios: str,
    seed: int,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    run_name: str,
) -> None:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Visual dataset root not found: {dataset_root}")

    yolo_dir = dataset_root / "yolo_nano"
    data_yaml = yolo_dir / "mmcows_binary.yaml"

    _run_step(
        "visual_build_index",
        [
            sys.executable,
            "-m",
            "src.visual.data.a1_build_index",
            "--dataset-root",
            str(dataset_root),
            "--date-folder",
            date_folder,
            "--cameras",
            cameras,
            "--split-ratios",
            split_ratios,
            "--seed",
            str(seed),
        ],
    )

    _run_step(
        "visual_prepare_yolo",
        [
            sys.executable,
            "-m",
            "src.visual.data.a2_yolo_binary",
            "--dataset-root",
            str(dataset_root),
            "--remap-labels-in-place",
        ],
    )

    _run_step(
        "visual_train",
        [
            sys.executable,
            "-m",
            "src.visual.models.train",
            "--data-yaml",
            str(data_yaml),
            "--model",
            model,
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--device",
            device,
            "--workers",
            str(workers),
            "--project",
            "store/models/visual",
            "--name",
            run_name,
        ],
    )


class VisualTrainingPipeline(BasePipeline):
    """Pipeline for visual model training."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mlflow_logger = MLflowLogger(config)

    def run(self) -> None:
        self.mlflow_logger.start_run()
        try:
            logger.info("Running Visual Training Pipeline...")
            run_visual_pipeline(
                dataset_root=Path(self.config.get("dataset_root", "store/data/raw/visual_data")),
                date_folder=self.config.get("date_folder", "0725"),
                cameras=self.config.get("cameras", "cam_1,cam_2,cam_3,cam_4"),
                split_ratios=self.config.get("split_ratios", "0.70,0.15,0.15"),
                seed=int(self.config.get("seed", 42)),
                model=self.config.get("model", "yolov8n.pt"),
                epochs=int(self.config.get("epochs", 50)),
                imgsz=int(self.config.get("imgsz", 640)),
                batch=int(self.config.get("batch", 16)),
                device=self.config.get("device", "cpu"),
                workers=int(self.config.get("workers", 4)),
                run_name=self.config.get("run_name", "cow_detection_v8n"),
            )
        finally:
            self.mlflow_logger.end_run()


def main() -> None:
    args = parse_args()
    mlflow_cfg: Dict[str, Any] = {
        "enabled": True,
        "experiment_name": args.experiment_name,
    }
    if args.mlflow_tracking_uri:
        mlflow_cfg["tracking_uri"] = args.mlflow_tracking_uri

    config: Dict[str, Any] = {
        "modality": "visual",
        "dataset_root": str(args.dataset_root),
        "date_folder": args.date_folder,
        "cameras": args.cameras,
        "split_ratios": args.split_ratios,
        "seed": args.seed,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "run_name": args.run_name,
        "mlflow": mlflow_cfg,
    }

    pipeline = VisualTrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
