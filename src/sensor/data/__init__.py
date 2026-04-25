"""
src.sensor.data
===============

Data-layer package: loaders, splits, synchronisation, and windowing.

Importing this package automatically installs the ``mmcows.utils.*``
compatibility shim so that ``loaders.py`` and ``splits.py`` work without
the ``mmcows`` package being installed.
"""

# Must be first — injects fake mmcows.utils.* before any submodule loads.
from src.sensor.data._compat import _install
_install()

from src.sensor.data.loaders   import (   # noqa: E402  (after _install)
    IMULoader,
    MultimodalSensorLoader,
    UWBHeadDirectionLoader,
    UWBLoader,
    VisualLocationLoader,
)
from src.sensor.data.splits    import SplitConfig       # noqa: E402
from src.sensor.data.sync      import resample_to_target # noqa: E402
from src.sensor.data.windowing import make_windows, window_to_flat_features  # noqa: E402

__all__ = [
    "IMULoader",
    "MultimodalSensorLoader",
    "UWBHeadDirectionLoader",
    "UWBLoader",
    "VisualLocationLoader",
    "SplitConfig",
    "resample_to_target",
    "make_windows",
    "window_to_flat_features",
]
