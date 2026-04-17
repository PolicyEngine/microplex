"""Compatibility shim for US-specific calibration harness helpers."""

from __future__ import annotations

try:
    from microplex_us.calibration_harness import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.calibration_harness moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.calibration_harness`."
    ) from exc
