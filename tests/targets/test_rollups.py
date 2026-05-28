"""Tests for generic tabular rollup target providers."""

from __future__ import annotations

import pandas as pd

from microplex.core import EntityType
from microplex.targets import (
    TabularRollupSpec,
    TabularRollupTargetProvider,
    TargetAggregation,
    TargetFilter,
    TargetQuery,
    as_string_tuple,
    build_tabular_rollup_targets,
)

ROLLUPS = {
    "national": TabularRollupSpec(
        geo_level="national",
        source_column=None,
        filter_feature=None,
        group_name="people_national",
        name_prefix="people_national",
    ),
    "region": TabularRollupSpec(
        geo_level="region",
        source_column="region_code",
        filter_feature="region",
        group_name="people_region",
        name_prefix="people_region",
    ),
}


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_code": ["01", "01", "02", None],
            "population": [10, 5, 7, 3],
        }
    )


def test_build_tabular_rollup_targets_groups_and_filters() -> None:
    targets = build_tabular_rollup_targets(
        _rows(),
        rollups=ROLLUPS,
        value_column="population",
        variable="person_count",
        entity=EntityType.PERSON,
        period=2026,
        source="publisher",
        units="persons",
        geo_levels=("national", "region"),
        geographic_ids=("01",),
        base_metadata={"source_year": 2024},
    )

    assert [target.name for target in targets] == ["people_region_01"]
    target = targets[0]
    assert target.value == 15
    assert target.entity is EntityType.PERSON
    assert target.aggregation is TargetAggregation.COUNT
    assert target.filters == (
        TargetFilter(feature="region", operator="==", value="01"),
    )
    assert target.metadata["variable"] == "person_count"
    assert target.metadata["geo_level"] == "region"
    assert target.metadata["geographic_id"] == "01"
    assert target.metadata["target_group"] == "people_region"
    assert target.metadata["tabular_rollup"] is True
    assert target.metadata["source_year"] == 2024


def test_tabular_rollup_provider_honors_query_and_variable_aliases() -> None:
    provider = TabularRollupTargetProvider(
        _rows(),
        rollups=ROLLUPS,
        value_column="population",
        variable="person_count",
        variable_aliases=("population",),
        entity=EntityType.PERSON,
        period=2026,
        default_geo_levels=("region",),
    )

    target_set = provider.load_target_set(
        TargetQuery(
            provider_filters={
                "variables": ["population"],
                "geographic_levels": ["region"],
                "geographic_ids": ["02"],
            }
        )
    )

    assert [target.name for target in target_set.targets] == ["people_region_02"]
    assert target_set.targets[0].value == 7


def test_tabular_rollup_provider_returns_empty_for_unrelated_variable() -> None:
    provider = TabularRollupTargetProvider(
        _rows(),
        rollups=ROLLUPS,
        value_column="population",
        variable="person_count",
        entity=EntityType.PERSON,
        period=2026,
    )

    target_set = provider.load_target_set(
        TargetQuery(provider_filters={"variables": ["household_count"]})
    )

    assert target_set.targets == []


def test_tabular_rollup_targets_keep_zero_values_by_default() -> None:
    targets = build_tabular_rollup_targets(
        pd.DataFrame({"region_code": ["03"], "population": [0]}),
        rollups=ROLLUPS,
        value_column="population",
        variable="person_count",
        entity=EntityType.PERSON,
        period=2026,
        geo_levels=("region",),
    )

    assert [target.value for target in targets] == [0]


def test_as_string_tuple_accepts_scalar_provider_filters() -> None:
    assert as_string_tuple(None) == ()
    assert as_string_tuple("state") == ("state",)
    assert as_string_tuple(6) == ("6",)
    assert as_string_tuple(["01", 2]) == ("01", "2")
