"""Compatibility shim for US-specific RAC mappings."""

from __future__ import annotations

try:
    from microplex_us.targets.rac_mapping import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.targets.rac_mapping moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.targets.rac_mapping`."
    ) from exc
