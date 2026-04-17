"""Compatibility shim for US-specific parity calibration."""

from __future__ import annotations

try:
    from microplex_us.unified_calibration import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.unified_calibration moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.unified_calibration`."
    ) from exc
