"""Compatibility shim for US PolicyEngine helpers."""

from __future__ import annotations

try:
    from microplex_us.policyengine.us import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name != "microplex_us":
        raise
    raise ModuleNotFoundError(
        "microplex.policyengine.us moved to the separate `microplex-us` package. "
        "Install or add `microplex-us`, then import `microplex_us.policyengine.us`."
    ) from exc
