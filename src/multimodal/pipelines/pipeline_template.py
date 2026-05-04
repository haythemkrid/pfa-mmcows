"""A minimal, documented training pipeline template for multimodal projects.

This file is a lightweight scaffold showing the recommended structure and
hooks for a training pipeline in this repository. It is designed to be
readable and easy to extend. Copy or subclass `TemplateTrainingPipeline`
when creating new pipelines.

Guidelines:
- Keep a small public surface: `__init__`, `run`, and a few well-named
  protected methods (_prepare_data, _build_model, _train_epoch, _save)
- Implement deterministic checkpoint saving and resumable training
- Keep logging and external integrations (MLflow) isolated behind
  small adapter methods
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.base.pipeline import BasePipeline
from shared.utils.logger import setup_logger


class TemplateTrainingPipeline(BasePipeline):
    """Minimal, well-documented training pipeline scaffold.

    Responsibilities:
    - Load and prepare datasets in `_prepare_data()`
    - Build model/optimizer/scheduler in `_build_components()`
    - Run training loop in `_train_epoch()` and validation
    - Save/load checkpoints via `_save_checkpoint()` and `_load_checkpoint()`
    - Expose a single `run()` entrypoint used by CLI/Hydra
    """

    def __init__(self, cfg: Dict[str, Any]):
        # Convert or validate cfg as needed (Hydra will normally pass a DictConfig)
        super().__init__(config=cfg)
        self.cfg = cfg

        self.root = Path.cwd()
        self.run_name = self.cfg.get("experiment", {}).get("run_name") or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = (self.root / self.cfg.get("output", {}).get("run_dir", "runs") / self.run_name).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(name="mmcows.template.pipeline", log_file=str(self.run_dir / "run.log"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _log_event(self, event: str, **payload: Any) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **payload}
        self.logger.info(json.dumps(record, default=str))

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load / prepare datasets and return (train_loader, val_loader).

        Implementations should honor `cfg.data.*` settings such as
        `batch_size`, `num_workers`, `pin_memory`, and augmentation flags.
        """
        raise NotImplementedError("_prepare_data must be implemented by subclasses")

    def _build_components(self) -> Tuple[nn.Module, Any, Any]:
        """Build and return (model, optimizer, scheduler).

        Keep model creation in a single method so it's easy to re-create and
        to load state for evaluation or registration.
        """
        raise NotImplementedError("_build_components must be implemented by subclasses")

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer]) -> Dict[str, float]:
        """Run a single epoch (train or eval depending on optimizer).

        Returns a dict of summary metrics (loss, f1, etc.). Keep this method
        small and focused so unit tests can exercise it.
        """
        raise NotImplementedError("_train_epoch must be implemented by subclasses")

    @staticmethod
    def _save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
        torch.save(state, path)

    @staticmethod
    def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
        return torch.load(path, map_location=device)

    def run(self) -> None:
        """High-level entrypoint for the pipeline.

        This method should remain thin: orchestrate preparation, training across
        folds/epochs, logging, and finalization. Business logic lives in
        protected methods so unit tests can target behaviour precisely.
        """
        self._log_event("start", run_dir=str(self.run_dir), device=str(self.device))

        # Example orchestration (override / extend as needed):
        train_loader, val_loader = self._prepare_data()
        model, optimizer, scheduler = self._build_components()

        for epoch in range(int(self.cfg.get("training", {}).get("epochs", 1))):
            metrics = self._train_epoch(model, train_loader, optimizer)
            val_metrics = self._train_epoch(model, val_loader, None)
            self._log_event("epoch", epoch=epoch, **metrics, **{f"val_{k}": v for k, v in val_metrics.items()})

        # Final save
        self._save_checkpoint(self.run_dir / "final.pt", {"cfg": self.cfg})
        self._log_event("done")


__all__ = ["TemplateTrainingPipeline"]
