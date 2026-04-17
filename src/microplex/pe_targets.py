"""Compatibility shim for US PolicyEngine target loaders."""

from __future__ import annotations

try:
    from microplex_us.pe_targets import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.pe_targets moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.pe_targets`."
    ) from exc
