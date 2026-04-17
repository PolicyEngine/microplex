"""Compatibility shim for US-specific CPS data helpers."""

from __future__ import annotations

try:
    from microplex_us.data import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.data moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.data`."
    ) from exc
