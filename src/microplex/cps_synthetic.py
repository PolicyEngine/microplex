"""Compatibility shim for CPS-specific synthetic-data helpers."""

from __future__ import annotations

try:
    from microplex_us.cps_synthetic import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.cps_synthetic moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.cps_synthetic`."
    ) from exc
