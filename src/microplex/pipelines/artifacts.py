"""Compatibility shim for US artifact persistence helpers."""

from __future__ import annotations

try:
    from microplex_us.pipelines.artifacts import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.pipelines.artifacts moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.pipelines.artifacts`."
    ) from exc
