"""Compatibility shims for US-specific validation helpers."""

from __future__ import annotations

try:
    from microplex_us.validation import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.validation moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.validation`."
    ) from exc
