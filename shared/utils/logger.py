"""Shared logging utilities for MMCOWS pipelines."""

from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logger(
    name: str = "mmcows",
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or return a configured application logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    logger.info("Logger initialized at %s", datetime.now().isoformat(timespec="seconds"))
    return logger


logger = setup_logger()
