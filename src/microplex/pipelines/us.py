"""Compatibility shim for the US pipeline package."""

from __future__ import annotations

try:
    from microplex_us.pipelines.us import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.pipelines.us moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.pipelines.us`."
    ) from exc
