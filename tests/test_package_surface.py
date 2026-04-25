"""Package-surface regression tests for the core engine."""

from __future__ import annotations

import microplex


def test_top_level_package_does_not_export_us_specific_helpers() -> None:
    assert not hasattr(microplex, "load_cps_asec")
    assert not hasattr(microplex, "load_cps_for_synthesis")
    assert not hasattr(microplex, "create_sample_data")
    assert not hasattr(microplex, "get_data_info")
    assert not hasattr(microplex, "CPSSummaryStats")
    assert not hasattr(microplex, "CPSSyntheticGenerator")
    assert not hasattr(microplex, "validate_synthetic")
    assert not hasattr(microplex, "SupabaseTargetLoader")
    assert not hasattr(microplex, "BlockGeography")
    assert not hasattr(microplex, "load_block_probabilities")
    assert not hasattr(microplex, "derive_geographies")
