"""Training pipeline for multimodal MBT models with resumable CV and MLflow tracking."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import dotenv_values, load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from shared.base.pipeline import BasePipeline
from shared.utils.logger import setup_logger
from shared.utils.mlflow_logger import MLflowLogger
from src.multimodal.data.dataset import FrameIndex, ImageCache, MBTDataset
from src.multimodal.models.mbt import MultimodalBottleneckTransformer
from src.sensor.data.loaders import UWBLoader
from src.sensor.data.splits import SplitConfig
from src.sensor.data.sync import resample_to_target
from src.sensor.data.windowing import make_windows

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.models import infer_signature

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    infer_signature = None
    MLFLOW_AVAILABLE = False


def _is_local_tracking_uri(uri: str) -> bool:
    local_markers = ("file:", "sqlite:", "runs/mlflow", "./mlruns", "mlruns")
    return str(uri).startswith(local_markers) or str(uri).startswith("/")


class MultimodalTrainingPipeline(BasePipeline):
    """Cross-validated training pipeline for the multimodal MBT model."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError("Hydra config could not be converted to a dictionary.")

        super().__init__(config=config_dict)

        self.project_root = Path(get_original_cwd())
        self.run_name = self.cfg.experiment.run_name or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = (self.project_root / Path(self.cfg.output.run_dir) / self.run_name).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.run_dir / "train.log"
        self.config_path = self.run_dir / "config.yaml"
        self.config_path.write_text(OmegaConf.to_yaml(self.cfg), encoding="utf-8")

        self.logger = setup_logger(
            name="mmcows.multimodal.training",
            log_file=str(self.log_path),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mlflow_logger: Optional[MLflowLogger] = None
        self.mlflow_run = None
        self.best_model_info: Optional[Dict[str, Any]] = None

    def _log_event(self, event: str, **payload: Any) -> None:
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            **payload,
        }
        self.logger.info(json.dumps(record, default=str))

    def _ensure_mlflow_credentials(self) -> None:
        env_path = self.project_root / ".env"
        if env_path.exists():
            file_values = dotenv_values(env_path)
            load_dotenv(dotenv_path=env_path, override=False)
        else:
            file_values = {}
            load_dotenv(override=False)

        for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
            if not os.environ.get(key) and file_values.get(key):
                os.environ[key] = str(file_values[key])

        if not self.cfg.output.use_mlflow:
            return

        missing = [
            key
            for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
            if not os.environ.get(key)
        ]
        if missing:
            raise RuntimeError(
                "Missing MLflow credentials. Set the following keys in .env or env vars: "
                + ", ".join(missing)
            )

    def _resolve_tracking_uri(self) -> str:
        default_uri = "https://dagshub.com/haythemkrid/pfa-mmcows.mlflow"
        candidates = [
            self.config.get("mlflow", {}).get("tracking_uri"),
            os.environ.get("MLFLOW_TRACKING_URI"),
            self.config.get("output", {}).get("mlflow_tracking_uri"),
            default_uri,
        ]

        for candidate in candidates:
            if candidate and not _is_local_tracking_uri(str(candidate)):
                return str(candidate)
        return default_uri

    def _build_mlflow_config(self) -> Dict[str, Any]:
        mlflow_cfg = dict(self.config)
        mlflow_section = dict(mlflow_cfg.get("mlflow", {}))
        mlflow_section.update(
            {
                "enabled": bool(self.cfg.output.use_mlflow),
                "tracking_uri": self._resolve_tracking_uri(),
                "experiment_name": self.cfg.experiment.name,
            }
        )
        mlflow_cfg["mlflow"] = mlflow_section
        return mlflow_cfg

    def _start_mlflow(self) -> None:
        if not self.cfg.output.use_mlflow:
            self._log_event(event="mlflow_disabled")
            return

        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow is enabled in config but mlflow package is not installed.")

        self._ensure_mlflow_credentials()
        mlflow_cfg = self._build_mlflow_config()

        self.mlflow_logger = MLflowLogger(mlflow_cfg)
        flat_cfg = MLflowLogger._flatten_dict(self.config)
        self.mlflow_run = self.mlflow_logger.start_run(run_name=self.run_name, params=flat_cfg)
        self._log_event(
            event="mlflow_started",
            tracking_uri=mlflow.get_tracking_uri() if mlflow else None,
            experiment=self.cfg.experiment.name,
        )

    @staticmethod
    def _save_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
        epoch: int,
        best_f1: float,
        model_metadata: Optional[Dict[str, Any]] = None,
        completed: bool = False,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_f1": best_f1,
                "completed": completed,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model_metadata": model_metadata or {},
            },
            path,
        )

    @staticmethod
    def _load_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
        device: torch.device,
    ) -> Tuple[int, float]:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        return int(ckpt["epoch"]) + 1, float(ckpt.get("best_f1", 0.0))

    def _build_signature_metadata(
        self, model: nn.Module, dataset: MBTDataset
    ) -> Tuple[Optional[Any], Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
        if not MLFLOW_AVAILABLE or infer_signature is None or len(dataset) == 0:
            return None, {}, None

        sample = dataset[0]
        sensor = sample["sensor"].unsqueeze(0).to(self.device)
        image = sample["image"].unsqueeze(0).to(self.device)

        model.eval()
        with torch.no_grad():
            logits = model(sensor, image)

        input_example = {
            "sensor": sensor.detach().cpu().numpy(),
            "image": image.detach().cpu().numpy(),
        }
        output_example = logits.detach().cpu().numpy()
        signature = infer_signature(input_example, output_example)

        metadata = {
            "input_schema": str(signature.inputs),
            "output_schema": str(signature.outputs),
            "input_example_shapes": {
                "sensor": list(input_example["sensor"].shape),
                "image": list(input_example["image"].shape),
            },
            "output_example_shape": list(output_example.shape),
        }
        return signature, metadata, input_example

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[AdamW],
        scaler: Optional[GradScaler],
        train: bool,
    ) -> Tuple[float, float]:
        model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for batch in loader:
                sensor = batch["sensor"].to(self.device, non_blocking=True)
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                use_amp = bool(self.cfg.training.mixed_precision and self.device.type == "cuda")
                with autocast("cuda", enabled=use_amp):
                    logits = model(sensor, images)
                    loss = criterion(logits, labels)

                if train and optimizer is not None:
                    if scaler is not None and use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip)
                        optimizer.step()

                total_loss += float(loss.item()) * labels.size(0)
                all_preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

        mean_loss = total_loss / max(len(all_labels), 1)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return mean_loss, float(macro_f1)

    def _prepare_sensor_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._log_event(event="data_load", stage="sensor_start")
        uwb_df = UWBLoader(self.cfg.data.sensor_data_dir, drop_unknown=True).load(
            cow_ids=list(self.cfg.data.sensor_cow_ids),
            date=self.cfg.data.annotated_date,
        )

        parts = []
        for cid, cow_df in uwb_df.groupby("cow_id"):
            rdf = resample_to_target(
                cow_df,
                feature_cols=list(self.cfg.data.uwb_feature_cols),
                source_rate_hz=self.cfg.data.uwb_source_rate_hz,
                target_rate_hz=self.cfg.data.target_rate_hz,
            )
            labels = (
                cow_df[["timestamp", "behavior", "cow_id"]]
                .astype({"timestamp": np.int64})
                .sort_values("timestamp")
            )
            rdf = pd.merge_asof(
                rdf.sort_values("timestamp"),
                labels,
                on="timestamp",
                direction="nearest",
                tolerance=int(1 / self.cfg.data.uwb_source_rate_hz),
            )
            rdf["cow_id"] = cid
            rdf["behavior"] = rdf["behavior"].fillna(0).astype(int)
            parts.append(rdf)

        uwb_resampled = pd.concat(parts, ignore_index=True)
        X_raw, y, cow_ids, ts = make_windows(
            uwb_resampled,
            feature_cols=list(self.cfg.data.uwb_feature_cols),
            window_size_s=self.cfg.data.window_size_s,
            target_rate_hz=self.cfg.data.target_rate_hz,
            overlap=self.cfg.data.overlap,
        )

        self._log_event(
            event="data_load",
            stage="sensor_done",
            windows=int(X_raw.shape[0]),
            shape=list(X_raw.shape),
        )
        return X_raw, y, cow_ids, ts

    def _train_fold(
        self,
        fold: str,
        fold_dir: Path,
        train_ds: MBTDataset,
        val_ds: MBTDataset,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        best_ckpt = fold_dir / "best.pt"
        last_ckpt = fold_dir / "last.pt"

        if best_ckpt.exists():
            meta = torch.load(best_ckpt, map_location="cpu")
            if meta.get("completed", False):
                self._log_event(
                    event="fold_skip",
                    fold=fold,
                    reason="already_completed",
                    best_f1=float(meta.get("best_f1", 0.0)),
                )
                return float(meta.get("best_f1", 0.0)), None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
        )

        model = MultimodalBottleneckTransformer.from_config(self.cfg.model).to(self.device)
        optimizer = AdamW(
            model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.training.epochs,
            eta_min=self.cfg.training.lr * 1e-2,
        )
        scaler = GradScaler("cuda") if self.cfg.training.mixed_precision and self.device.type == "cuda" else None
        criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.training.label_smoothing)

        signature, signature_meta, input_example = self._build_signature_metadata(model, val_ds)

        start_epoch, best_f1 = 0, 0.0
        if last_ckpt.exists():
            start_epoch, best_f1 = self._load_checkpoint(
                last_ckpt,
                model,
                optimizer,
                scheduler,
                self.device,
            )
            self._log_event(event="resume", fold=fold, start_epoch=start_epoch, best_f1=best_f1)

        warmup_epochs = int(self.cfg.training.warmup_epochs)
        base_lr = float(self.cfg.training.lr)
        patience_counter = 0
        fold_best_info: Optional[Dict[str, Any]] = None

        def _apply_warmup(epoch: int) -> None:
            if epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr * (epoch + 1) / max(warmup_epochs, 1)

        for epoch in range(start_epoch, self.cfg.training.epochs):
            _apply_warmup(epoch)
            t0 = time.perf_counter()

            tr_loss, tr_f1 = self._run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
            if epoch >= warmup_epochs:
                scheduler.step()
            val_loss, val_f1 = self._run_epoch(model, val_loader, criterion, None, None, train=False)

            elapsed = time.perf_counter() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            self._log_event(
                event="epoch",
                fold=fold,
                epoch=epoch,
                tr_loss=round(tr_loss, 5),
                tr_f1=round(tr_f1, 5),
                val_loss=round(val_loss, 5),
                val_f1=round(val_f1, 5),
                lr=round(lr_now, 8),
                elapsed_s=round(elapsed, 1),
            )

            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics(
                    {
                        f"{fold}/tr_loss": tr_loss,
                        f"{fold}/tr_f1": tr_f1,
                        f"{fold}/val_loss": val_loss,
                        f"{fold}/val_f1": val_f1,
                        f"{fold}/lr": lr_now,
                    },
                    step=epoch,
                )

            self._save_checkpoint(
                last_ckpt,
                model,
                optimizer,
                scheduler,
                epoch,
                best_f1,
                model_metadata=signature_meta,
            )

            if val_f1 > best_f1 + self.cfg.training.early_stop_min_delta:
                best_f1 = val_f1
                patience_counter = 0
                self._save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_f1,
                    model_metadata=signature_meta,
                )
                self._log_event(event="best_saved", fold=fold, epoch=epoch, best_f1=round(best_f1, 5))

                fold_best_info = {
                    "fold": fold,
                    "checkpoint_path": str(best_ckpt),
                    "best_f1": float(best_f1),
                    "signature": signature,
                    "signature_meta": signature_meta,
                    "input_example": input_example,
                }
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.training.early_stop_patience:
                self._log_event(event="early_stop", fold=fold, epoch=epoch, best_f1=round(best_f1, 5))
                break

        if best_ckpt.exists():
            meta = torch.load(best_ckpt, map_location="cpu")
            meta["completed"] = True
            torch.save(meta, best_ckpt)

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()
        return float(best_f1), fold_best_info

    def _register_best_model_if_qualified(self) -> None:
        if self.mlflow_logger is None or self.mlflow_run is None:
            return
        if self.best_model_info is None:
            self._log_event(event="registry_skip", reason="no_best_model")
            return

        threshold = float(self.config.get("output", {}).get("model_registry_threshold", 0.0))
        best_f1 = float(self.best_model_info["best_f1"])
        if best_f1 < threshold:
            self._log_event(
                event="registry_skip",
                reason="below_threshold",
                best_f1=best_f1,
                threshold=threshold,
            )
            return

        ckpt_path = Path(self.best_model_info["checkpoint_path"])
        if not ckpt_path.exists():
            self._log_event(event="registry_skip", reason="checkpoint_missing", path=str(ckpt_path))
            return

        if not MLFLOW_AVAILABLE or mlflow is None:
            self._log_event(event="registry_skip", reason="mlflow_unavailable")
            return

        model = MultimodalBottleneckTransformer.from_config(self.cfg.model).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        artifact_path = f"model/{self.best_model_info['fold']}_best"
        input_example = self.best_model_info.get("input_example")
        signature = self.best_model_info.get("signature")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            metadata=ckpt.get("model_metadata", {}),
        )

        model_name = str(
            self.config.get("output", {}).get(
                "registered_model_name",
                "mmcows-multimodal-mbt",
            )
        )
        model_uri = f"runs:/{self.mlflow_run.info.run_id}/{artifact_path}"
        registration = mlflow.register_model(model_uri=model_uri, name=model_name)

        self._log_event(
            event="model_registered",
            model_name=model_name,
            model_version=getattr(registration, "version", None),
            best_f1=best_f1,
            threshold=threshold,
            checkpoint=str(ckpt_path),
        )

    def _log_run_artifacts(self) -> None:
        if self.mlflow_logger is None:
            return

        self.mlflow_logger.log_artifact(str(self.log_path))

        hydra_config = Path.cwd() / ".hydra" / "config.yaml"
        if hydra_config.exists():
            self.mlflow_logger.log_artifact(str(hydra_config))
        elif self.config_path.exists():
            self.mlflow_logger.log_artifact(str(self.config_path))

    def run(self) -> None:
        self._log_event(
            event="start",
            run_dir=str(self.run_dir),
            run_name=self.run_name,
            device=str(self.device),
        )

        self._start_mlflow()

        try:
            split_config = SplitConfig.from_json(
                s1_path=self.cfg.data.config_s1_path,
                s2_path=self.cfg.data.config_s2_path,
            )
            folds = split_config.available_folds(self.cfg.data.split_type)
            self._log_event(event="folds", folds=folds, split_type=self.cfg.data.split_type)

            X_sensor, y, cow_ids, ts = self._prepare_sensor_data()
            frame_index = FrameIndex(
                visual_data_dir=self.cfg.data.visual_data_dir,
                date=self.cfg.data.annotated_date,
                cameras=list(self.cfg.data.cameras),
            )

            image_cache: Optional[ImageCache] = None
            if self.cfg.data.cache_images:
                self._log_event(event="image_cache", stage="start")
                image_cache = ImageCache(
                    frame_index=frame_index,
                    resize_to=self.cfg.data.cache_resize_to,
                )
                self._log_event(event="image_cache", stage="done")

            all_best_f1s: list[float] = []
            for fold in folds:
                fold_dir = self.run_dir / fold
                fold_dir.mkdir(exist_ok=True)
                self._log_event(event="fold_start", fold=fold)

                index_df = pd.DataFrame(
                    {
                        "timestamp": ts,
                        "cow_id": cow_ids,
                        "behavior": y,
                        "_idx": np.arange(len(y)),
                    }
                )
                train_df, val_df, _ = split_config.split(
                    index_df,
                    split_type=self.cfg.data.split_type,
                    fold=fold,
                )

                def _make_ds(df: pd.DataFrame, is_train: bool) -> MBTDataset:
                    idx = df["_idx"].values
                    return MBTDataset(
                        X_sensor=X_sensor[idx],
                        y=y[idx],
                        cow_ids=cow_ids[idx],
                        start_timestamps=ts[idx],
                        frame_index=frame_index,
                        image_cache=image_cache,
                        image_size=self.cfg.data.image_size,
                        train=is_train,
                    )

                train_ds, val_ds = _make_ds(train_df, True), _make_ds(val_df, False)
                self._log_event(event="fold_sizes", fold=fold, train=len(train_ds), val=len(val_ds))

                best_f1, fold_best_info = self._train_fold(fold, fold_dir, train_ds, val_ds)
                all_best_f1s.append(best_f1)

                if fold_best_info is not None and (
                    self.best_model_info is None
                    or float(fold_best_info["best_f1"]) > float(self.best_model_info["best_f1"])
                ):
                    self.best_model_info = fold_best_info

                self._log_event(event="fold_end", fold=fold, best_f1=round(best_f1, 5))

                del train_ds, val_ds
                gc.collect()

            mean_f1 = float(np.mean(all_best_f1s)) if all_best_f1s else 0.0
            std_f1 = float(np.std(all_best_f1s)) if all_best_f1s else 0.0
            self._log_event(
                event="done",
                per_fold_f1=all_best_f1s,
                mean_f1=round(mean_f1, 5),
                std_f1=round(std_f1, 5),
            )

            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics({"cv_mean_f1": mean_f1, "cv_std_f1": std_f1})
                self._register_best_model_if_qualified()
                self._log_run_artifacts()

            print(f"\nCV macro-F1 = {mean_f1:.4f} +- {std_f1:.4f}")
        finally:
            if self.mlflow_logger is not None:
                self.mlflow_logger.end_run()


@hydra.main(
    config_path="../../multimodal/configs",
    config_name="mbt_default",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    pipeline = MultimodalTrainingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()