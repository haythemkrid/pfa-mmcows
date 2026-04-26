"""
src.multimodal.pipelines.training_pipeline
==========================================

Robust, resumable training pipeline for the Multimodal Bottleneck
Transformer (MBT).

Resilience guarantees
---------------------
* **tmux / nohup safe** — the process lives entirely in the remote
  terminal.  A network disconnect does not kill it.
* **Epoch-level resume** — ``last.pt`` is overwritten every epoch with
  the complete training state (weights, optimiser, scheduler, epoch
  counter, best F1).  Re-running the script with the same
  ``experiment.run_name`` picks up exactly where it stopped.
* **Fold-level skip** — once a fold writes ``completed: true`` into
  ``best.pt``, it is skipped entirely on subsequent runs.
* **Append-mode log** — ``train.log`` is never truncated; all events
  from a resumed run are appended after the previous ones.

Structured log format
---------------------
Every event is a JSON object on its own line::

    {"time": "2025-04-24T09:30:12", "event": "epoch", "fold": "fold_1",
     "epoch": 5, "tr_loss": 1.234, "val_f1": 0.712, ...}

Stream it live::

    tail -f logs/mbt_runs/<run>/train.log | python -m json.tool
"""

from __future__ import annotations

# ── Compat shim must be first ─────────────────────────────────────────────────
# Injects fake mmcows.utils.* into sys.modules before anything in
# src.sensor.data tries to import them.
from src.sensor.data._compat import _install
_install()
# ─────────────────────────────────────────────────────────────────────────────

import gc
import json
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.sensor.data.loaders   import UWBLoader
from src.sensor.data.splits    import SplitConfig
from src.sensor.data.sync      import resample_to_target
from src.sensor.data.windowing import make_windows
from src.shared.utils.logger import logger
from src.multimodal.data.dataset import FrameIndex, ImageCache, MBTDataset
from src.multimodal.models.mbt   import MultimodalBottleneckTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Structured logger
# ─────────────────────────────────────────────────────────────────────────────

class StructuredLogger:
    """One JSON object per line, appended to file and echoed to console."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = log_path.open("a", buffering=1)

    def log(self, **kwargs) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **kwargs}
        line   = json.dumps(record)
        self._fh.write(line + "\n")
        logger.info(line)

    def close(self) -> None:
        self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1, completed=False):
    torch.save({
        "epoch": epoch, "best_f1": best_f1, "completed": completed,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)


def _load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt["best_f1"]


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, scaler, device, cfg, train):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            sensor = batch["sensor"].to(device, non_blocking=True)
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = cfg.training.mixed_precision and device.type == "cuda"
            with autocast("cuda", enabled=use_amp):
                logits = model(sensor, images)
                loss   = criterion(logits, labels)

            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    mean_loss = total_loss / len(all_labels)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return mean_loss, macro_f1


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold training
# ─────────────────────────────────────────────────────────────────────────────

def _train_fold(fold, fold_dir, train_ds, val_ds, cfg, device, slog, mlflow_run):
    best_ckpt = fold_dir / "best.pt"
    last_ckpt = fold_dir / "last.pt"

    if best_ckpt.exists():
        meta = torch.load(best_ckpt, map_location="cpu")
        if meta.get("completed", False):
            slog.log(event="fold_skip", fold=fold,
                     reason="already completed", best_f1=meta["best_f1"])
            return meta["best_f1"]

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size * 2, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    model     = MultimodalBottleneckTransformer.from_config(cfg.model).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 1e-2)
    scaler    = GradScaler("cuda") if cfg.training.mixed_precision and device.type == "cuda" else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    start_epoch, best_f1 = 0, 0.0
    if last_ckpt.exists():
        start_epoch, best_f1 = _load_checkpoint(last_ckpt, model, optimizer, scheduler, device)
        slog.log(event="resume", fold=fold, start_epoch=start_epoch, best_f1=best_f1)

    warmup_epochs = cfg.training.warmup_epochs
    base_lr       = cfg.training.lr

    def _apply_warmup(ep):
        if ep < warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * (ep + 1) / warmup_epochs

    patience_counter = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        _apply_warmup(epoch)
        t0 = time.perf_counter()

        tr_loss,  tr_f1  = _run_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, True)
        if epoch >= warmup_epochs:
            scheduler.step()
        val_loss, val_f1 = _run_epoch(model, val_loader, criterion, None, None, device, cfg, False)

        elapsed = time.perf_counter() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        slog.log(
            event="epoch", fold=fold, epoch=epoch,
            tr_loss=round(tr_loss, 5),  tr_f1=round(tr_f1, 5),
            val_loss=round(val_loss, 5), val_f1=round(val_f1, 5),
            lr=round(lr_now, 8), elapsed_s=round(elapsed, 1),
        )
        if mlflow_run is not None:
            mlflow.log_metrics({
                f"{fold}/tr_loss": tr_loss, f"{fold}/tr_f1": tr_f1,
                f"{fold}/val_loss": val_loss, f"{fold}/val_f1": val_f1,
                f"{fold}/lr": lr_now,
            }, step=epoch)

        _save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, best_f1)

        if val_f1 > best_f1 + cfg.training.early_stop_min_delta:
            best_f1, patience_counter = val_f1, 0
            _save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, best_f1)
            slog.log(event="best_saved", fold=fold, epoch=epoch, best_f1=round(best_f1, 5))
        else:
            patience_counter += 1

        if patience_counter >= cfg.training.early_stop_patience:
            slog.log(event="early_stop", fold=fold, epoch=epoch, best_f1=round(best_f1, 5))
            break

    if best_ckpt.exists():
        meta = torch.load(best_ckpt, map_location="cpu")
        meta["completed"] = True
        torch.save(meta, best_ckpt)

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_sensor_data(cfg, slog):
    slog.log(event="data_load", stage="sensor_start")
    uwb_df = UWBLoader(cfg.data.sensor_data_dir, drop_unknown=True).load(
        cow_ids=list(cfg.data.sensor_cow_ids), date=cfg.data.annotated_date,
    )
    parts = []
    for cid, cow_df in uwb_df.groupby("cow_id"):
        rdf = resample_to_target(
            cow_df,
            feature_cols   = list(cfg.data.uwb_feature_cols),
            source_rate_hz = cfg.data.uwb_source_rate_hz,
            target_rate_hz = cfg.data.target_rate_hz,
        )
        labels = (
            cow_df[["timestamp", "behavior", "cow_id"]]
            .astype({"timestamp": np.int64})
            .sort_values("timestamp")
        )
        rdf = pd.merge_asof(
            rdf.sort_values("timestamp"), labels,
            on="timestamp", direction="nearest",
            tolerance=int(1 / cfg.data.uwb_source_rate_hz),
        )
        rdf["cow_id"]   = cid
        rdf["behavior"] = rdf["behavior"].fillna(0).astype(int)
        parts.append(rdf)

    uwb_resampled = pd.concat(parts, ignore_index=True)
    X_raw, y, cow_ids, ts = make_windows(
        uwb_resampled,
        feature_cols   = list(cfg.data.uwb_feature_cols),
        window_size_s  = cfg.data.window_size_s,
        target_rate_hz = cfg.data.target_rate_hz,
        overlap        = cfg.data.overlap,
    )
    slog.log(event="data_load", stage="sensor_done",
             windows=int(X_raw.shape[0]), shape=list(X_raw.shape))
    return X_raw, y, cow_ids, ts


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(
    config_path  = "../../multimodal/configs",
    config_name  = "mbt_default",
    version_base = "1.3",
)
def main(cfg: DictConfig) -> None:
    run_name = cfg.experiment.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(cfg.output.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    slog   = StructuredLogger(run_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slog.log(event="start", run_dir=str(run_dir), run_name=run_name, device=str(device))

    mlflow_run = None
    if cfg.output.use_mlflow:
        mlflow.set_tracking_uri(cfg.output.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.experiment.name)
        mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

    split_config = SplitConfig.from_json(
        s1_path=cfg.data.config_s1_path, s2_path=cfg.data.config_s2_path,
    )
    folds = split_config.available_folds(cfg.data.split_type)
    slog.log(event="folds", folds=folds, split_type=cfg.data.split_type)

    X_sensor, y, cow_ids, ts = _prepare_sensor_data(cfg, slog)
    frame_index = FrameIndex(
        visual_data_dir=cfg.data.visual_data_dir,
        date=cfg.data.annotated_date,
        cameras=list(cfg.data.cameras),
    )

    # Build the image cache once — eliminates all disk I/O from training loop.
    # Workers share it via fork; no serialisation overhead.
    image_cache: ImageCache | None = None
    if cfg.data.cache_images:
        slog.log(event="image_cache", stage="start")
        image_cache = ImageCache(
            frame_index = frame_index,
            resize_to   = cfg.data.cache_resize_to,
        )
        slog.log(event="image_cache", stage="done")

    all_best_f1s: list[float] = []

    for fold in folds:
        fold_dir = run_dir / fold
        fold_dir.mkdir(exist_ok=True)
        slog.log(event="fold_start", fold=fold)

        index_df = pd.DataFrame({
            "timestamp": ts, "cow_id": cow_ids,
            "behavior":  y,  "_idx":   np.arange(len(y)),
        })
        train_df, val_df, _ = split_config.split(
            index_df, split_type=cfg.data.split_type, fold=fold
        )

        def _make_ds(df, is_train):
            idx = df["_idx"].values
            return MBTDataset(
                X_sensor=X_sensor[idx], y=y[idx],
                cow_ids=cow_ids[idx], start_timestamps=ts[idx],
                frame_index=frame_index,
                image_cache=image_cache,
                image_size=cfg.data.image_size, train=is_train,
            )

        train_ds, val_ds = _make_ds(train_df, True), _make_ds(val_df, False)
        slog.log(event="fold_sizes", fold=fold, train=len(train_ds), val=len(val_ds))

        best_f1 = _train_fold(fold, fold_dir, train_ds, val_ds, cfg, device, slog, mlflow_run)
        all_best_f1s.append(best_f1)
        slog.log(event="fold_end", fold=fold, best_f1=round(best_f1, 5))

        del train_ds, val_ds
        gc.collect()

    mean_f1 = float(np.mean(all_best_f1s))
    std_f1  = float(np.std(all_best_f1s))
    slog.log(event="done", per_fold_f1=all_best_f1s,
             mean_f1=round(mean_f1, 5), std_f1=round(std_f1, 5))

    if mlflow_run is not None:
        mlflow.log_metrics({"cv_mean_f1": mean_f1, "cv_std_f1": std_f1})
        mlflow.end_run()

    slog.close()
    print(f"\n✅  CV macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}")


if __name__ == "__main__":
    main()
