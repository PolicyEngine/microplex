"""Compatibility shim for the legacy US targets database models."""

from __future__ import annotations

from typing import Any


def _raise_missing_us_targets_database() -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "Legacy US targets database models moved to the separate `microplex-us` "
        "package. Install or add `microplex-us`, then import "
        "`microplex_us.targets_database`."
    )


try:
    from microplex_us.targets_database import (  # noqa: F401
        Target,
        TargetCategory,
        TargetsDatabase,
    )
except ModuleNotFoundError:
    class _MissingUSTargetDatabase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise _raise_missing_us_targets_database()

    class TargetCategory(_MissingUSTargetDatabase):  # type: ignore[no-redef]
        pass

    class Target(_MissingUSTargetDatabase):  # type: ignore[no-redef]
        pass

    class TargetsDatabase(_MissingUSTargetDatabase):  # type: ignore[no-redef]
        pass
