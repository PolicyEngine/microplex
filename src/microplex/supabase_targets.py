"""Compatibility shim for US Supabase calibration targets.

US-specific Supabase target loading now lives in `microplex-us`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "microplex.supabase_targets is deprecated; use "
    "microplex_us.supabase_targets instead.",
    DeprecationWarning,
    stacklevel=2,
)

_moved_loader_import_error: ImportError | None = None

try:
    from microplex_us.supabase_targets import SupabaseTargetLoader
except ImportError as _import_error:
    _moved_loader_import_error = _import_error

    class SupabaseTargetLoader:  # type: ignore[no-redef]
        """Placeholder that explains how to access the moved US loader."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise ImportError(
                "SupabaseTargetLoader moved to microplex-us. Install "
                "`microplex-us` and import "
                "`microplex_us.supabase_targets.SupabaseTargetLoader`."
            ) from _moved_loader_import_error


__all__ = ["SupabaseTargetLoader"]
