import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
import torch
from dotenv import dotenv_values
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multimodal.models.mbt import MultimodalBottleneckTransformer


def _load_env() -> None:
    values = dotenv_values(ROOT / ".env")
    for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "MLFLOW_TRACKING_URI"):
        if values.get(key):
            os.environ[key] = str(values[key])


def _select_target_run_id(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
    target_run_id: Optional[str],
) -> str:
    if target_run_id:
        return target_run_id

    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=50,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise RuntimeError(f"No remote run found with runName={run_name}")

    finished = next((run for run in runs if run.info.status == "FINISHED"), None)
    return finished.info.run_id if finished else runs[0].info.run_id


def _find_best_checkpoint(run_dir: Path) -> Path:
    best_ckpt = None
    best_f1 = float("-inf")

    for fold_dir in sorted(run_dir.glob("fold_*")):
        ckpt_path = fold_dir / "best.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        f1 = float(ckpt.get("best_f1", -1.0))
        if f1 > best_f1:
            best_f1 = f1
            best_ckpt = ckpt_path

    if best_ckpt is None:
        raise RuntimeError(f"No fold best.pt found under {run_dir}")

    print(f"Best checkpoint: {best_ckpt} (best_f1={best_f1:.6f})")
    return best_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLflow artifacts/model for an existing local MBT run")
    parser.add_argument("--run-name", default="run1", help="Local/remote run name to backfill")
    parser.add_argument("--experiment-name", default="mbt_uwb_rgb", help="MLflow experiment name")
    parser.add_argument("--model-name", default="mmcows-multimodal-mbt", help="MLflow registered model name")
    parser.add_argument("--target-run-id", default=None, help="Optional explicit target MLflow run_id")
    parser.add_argument("--register", dest="register", action="store_true", help="Register model version")
    parser.add_argument("--no-register", dest="register", action="store_false", help="Skip model registration")
    parser.set_defaults(register=True)
    args = parser.parse_args()

    _load_env()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/haythemkrid/pfa-mmcows.mlflow")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found: {args.experiment_name}")

    target_run_id = _select_target_run_id(
        client=client,
        experiment_id=experiment.experiment_id,
        run_name=args.run_name,
        target_run_id=args.target_run_id,
    )
    print(f"Target run_id: {target_run_id}")

    run_dir = ROOT / "logs" / "mbt_runs" / args.run_name
    if not run_dir.exists():
        raise RuntimeError(f"Local run directory not found: {run_dir}")

    for path in (run_dir / "train.log", run_dir / "config.yaml"):
        if path.exists():
            client.log_artifact(target_run_id, str(path))
            print(f"Logged artifact: {path}")

    best_ckpt = _find_best_checkpoint(run_dir)
    client.log_artifact(target_run_id, str(best_ckpt))
    print(f"Logged artifact: {best_ckpt}")

    cfg = OmegaConf.load(run_dir / "config.yaml")
    model = MultimodalBottleneckTransformer.from_config(cfg.model)
    checkpoint = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with tempfile.TemporaryDirectory() as temp_dir:
        local_model_dir = Path(temp_dir) / "backfilled_best"
        mlflow.pytorch.save_model(
            pytorch_model=model,
            path=str(local_model_dir),
            metadata=checkpoint.get("model_metadata", {}),
        )
        client.log_artifacts(target_run_id, str(local_model_dir), artifact_path="model/backfilled_best")
        print("Logged MLflow model artifact to model/backfilled_best")

    if args.register:
        model_uri = f"runs:/{target_run_id}/model/backfilled_best"
        registration = mlflow.register_model(model_uri=model_uri, name=args.model_name)
        print(f"Registered model version: {getattr(registration, 'version', None)}")

    print("Backfill completed.")


if __name__ == "__main__":
    main()
