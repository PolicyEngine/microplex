"""Compatibility tests for the US Supabase target loader shim."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_supabase_targets_module():
    src_path = Path(__file__).parent.parent / "src" / "microplex"
    spec = importlib.util.spec_from_file_location(
        "supabase_targets",
        src_path / "supabase_targets.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with pytest.warns(DeprecationWarning, match="microplex_us.supabase_targets"):
        spec.loader.exec_module(module)
    return module


def test_supabase_target_loader_is_compatibility_shim() -> None:
    module = _load_supabase_targets_module()

    assert module.__all__ == ["SupabaseTargetLoader"]


def test_missing_microplex_us_loader_raises_actionable_import_error() -> None:
    module = _load_supabase_targets_module()

    if module.SupabaseTargetLoader.__module__.startswith("microplex_us"):
        pytest.skip("microplex-us is installed; shim resolved to the moved loader")

    with pytest.raises(ImportError, match="microplex-us"):
        module.SupabaseTargetLoader()
