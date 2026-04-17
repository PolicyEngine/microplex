"""Compatibility shims for country-specific data-source adapters."""

from __future__ import annotations

try:
    from microplex_us.data_sources import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.data_sources moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.data_sources`."
    ) from exc
