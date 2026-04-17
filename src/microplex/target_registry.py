"""Compatibility shim for US-specific target registry helpers."""

from __future__ import annotations

try:
    from microplex_us.target_registry import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.target_registry moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.target_registry`."
    ) from exc
