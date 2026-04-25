"""
src.sensor.data._compat
=======================

Compatibility shim that replaces ``mmcows.utils.*`` imports so that
``loaders.py``, ``splits.py``, ``sync.py``, and ``windowing.py`` work
when imported as ``src.sensor.data.*`` — i.e. without requiring the
``mmcows`` package to be installed or on ``sys.path``.

This module is an *internal* implementation detail.  External code should
never import from it directly.

How it works
------------
``loaders.py`` and ``splits.py`` contain::

    from mmcows.utils.constants import ALL_COW_IDS, ...
    from mmcows.utils.io_utils   import sensor_path, tag_name, ...
    from mmcows.utils.time_utils import floor_to_second, cdt_str_to_unix

We intercept those imports via ``sys.modules`` injection (see
``_install()`` below) so that Python resolves them to the implementations
defined here, without touching the original files.

Call ``_install()`` once at the top of any module that then imports the
data-layer files.  The pipeline's ``__init__.py`` does this automatically.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# mmcows.utils.constants
# ─────────────────────────────────────────────────────────────────────────────

SENSOR_COW_IDS: list[int] = list(range(1, 11))   # cows 1-10 have wearables
ALL_COW_IDS:    list[int] = list(range(1, 17))   # all 16 cows
ANNOTATED_DATE: str       = "0725"
CAMERA_IDS:     list[int] = [1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
# mmcows.utils.io_utils
# ─────────────────────────────────────────────────────────────────────────────

def tag_name(cow_id: int) -> str:
    """Return the tag folder name for a given cow ID, e.g. 3 → 'T03'."""
    return f"T{cow_id:02d}"


def cow_name(cow_id: int) -> str:
    """Return the cow folder name for a given cow ID, e.g. 3 → 'C03'."""
    return f"C{cow_id:02d}"


def sensor_path(
    sensor_data_dir: str | Path,
    sub: str,
    tag_or_cow: str,
    date: str,
) -> Path:
    """Build the path to a per-tag/per-cow daily CSV.

    Layout (matching the repo's ``store/data/raw/sensor_data/`` tree)::

        <sensor_data_dir>/main_data/<sub>/<tag_or_cow>/<tag_or_cow>_<date>.csv
        <sensor_data_dir>/sub_data/<sub>/<tag_or_cow>/<tag_or_cow>_<date>.csv

    The ``sub`` argument already includes the ``main_data/`` or ``sub_data/``
    prefix (e.g. ``"main_data/uwb"`` or ``"sub_data/head_direction"``).
    """
    return Path(sensor_data_dir) / sub / tag_or_cow / f"{tag_or_cow}_{date}.csv"


def behavior_label_path(
    sensor_data_dir: str | Path,
    cow_name: str,
    date: str,
) -> Path:
    """Path to the per-cow daily behaviour label CSV."""
    return (
        Path(sensor_data_dir)
        / "behavior_labels"
        / "individual"
        / f"{cow_name}_{date}.csv"
    )


def image_dir_path(
    visual_data_dir: str | Path,
    date: str,
    camera_id: int,
) -> Path:
    """Path to the image directory for one camera on one date."""
    return Path(visual_data_dir) / "images" / date / f"cam_{camera_id}"


def visual_location_path(
    sensor_data_dir: str | Path,
    cow_name: str,
    date: str,
) -> Path:
    """Path to the per-cow daily visual-location CSV."""
    return (
        Path(sensor_data_dir)
        / "sub_data"
        / "visual_location"
        / cow_name
        / f"{cow_name}_{date}.csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# mmcows.utils.time_utils
# ─────────────────────────────────────────────────────────────────────────────

def floor_to_second(ts_series):
    """Floor a float-second Unix timestamp Series to integer seconds."""
    import pandas as pd
    return ts_series.astype(float).astype(np.int64)


def cdt_str_to_unix(dt_str: str) -> int:
    """Convert a CDT datetime string to a UTC Unix timestamp (int).

    Expected format: ``"YYYY-MM-DD HH:MM:SS"`` in US Central time
    (UTC-5 in winter / UTC-6 in summer).  The dataset was collected in
    Wisconsin in July 2023, so CDT = UTC-5.
    """
    CDT_OFFSET_SECONDS = -5 * 3600   # CDT = UTC-5
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp()) - CDT_OFFSET_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# sys.modules injection
# ─────────────────────────────────────────────────────────────────────────────

def _make_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install() -> None:
    """Inject fake ``mmcows.utils.*`` modules into ``sys.modules``.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    if "mmcows" in sys.modules:
        return   # already installed (or the real package is present)

    # Stub package hierarchy
    mmcows         = _make_module("mmcows")
    mmcows_utils   = _make_module("mmcows.utils")
    mmcows_data    = _make_module("mmcows.data")   # needed by splits.py docstring

    constants_mod = _make_module(
        "mmcows.utils.constants",
        ALL_COW_IDS    = ALL_COW_IDS,
        ANNOTATED_DATE = ANNOTATED_DATE,
        CAMERA_IDS     = CAMERA_IDS,
        SENSOR_COW_IDS = SENSOR_COW_IDS,
    )
    io_utils_mod = _make_module(
        "mmcows.utils.io_utils",
        behavior_label_path  = behavior_label_path,
        cow_name             = cow_name,
        image_dir_path       = image_dir_path,
        sensor_path          = sensor_path,
        tag_name             = tag_name,
        visual_location_path = visual_location_path,
    )
    time_utils_mod = _make_module(
        "mmcows.utils.time_utils",
        cdt_str_to_unix = cdt_str_to_unix,
        floor_to_second = floor_to_second,
    )

    sys.modules["mmcows"]                  = mmcows
    sys.modules["mmcows.utils"]            = mmcows_utils
    sys.modules["mmcows.data"]             = mmcows_data
    sys.modules["mmcows.utils.constants"]  = constants_mod
    sys.modules["mmcows.utils.io_utils"]   = io_utils_mod
    sys.modules["mmcows.utils.time_utils"] = time_utils_mod
