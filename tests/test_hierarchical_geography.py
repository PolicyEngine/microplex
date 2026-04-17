"""Focused tests for generic geography assignment in hierarchical synthesis."""

from __future__ import annotations

import pandas as pd

from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyAssignmentPlan,
    StaticGeographyProvider,
    nearest_numeric_partition_key,
)
from microplex.hierarchical import HierarchicalSynthesizer


def test_apply_geography_assignment_adds_atomic_and_materialized_columns() -> None:
    crosswalk = AtomicGeographyCrosswalk(
        data=pd.DataFrame(
            {
                "cell_id": ["north-a", "north-b", "south-a"],
                "region_code": ["north", "north", "south"],
                "district_id": ["N-1", "N-2", "S-1"],
                "prob": [0.25, 0.75, 1.0],
            }
        ),
        atomic_id_column="cell_id",
        geography_columns=("region_code", "district_id"),
        probability_column="prob",
    )
    provider = StaticGeographyProvider(
        crosswalk=crosswalk,
        default_partition_columns=("region_code",),
    )
    plan = GeographyAssignmentPlan(
        partition_columns=("region_code",),
        atomic_id_column="cell_id",
        geography_columns=("district_id",),
    )
    synthesizer = HierarchicalSynthesizer(
        geography_provider=provider,
        geography_assignment=plan,
        random_state=42,
    )
    households = pd.DataFrame({"region_code": ["north", "south"]})

    result = synthesizer._apply_geography_assignment(households)

    assert set(result.columns) == {"region_code", "cell_id", "district_id"}
    assert result.loc[0, "cell_id"].startswith("north-")
    assert result.loc[1, "cell_id"] == "south-a"
    assert result.loc[0, "district_id"].startswith("N-")
    assert result.loc[1, "district_id"] == "S-1"


def test_apply_geography_assignment_refreshes_partition_columns_from_crosswalk() -> None:
    crosswalk = AtomicGeographyCrosswalk(
        data=pd.DataFrame(
            {
                "cell_id": ["01-a", "02-a"],
                "region_code": ["01", "02"],
                "district_id": ["north", "south"],
                "prob": [1.0, 1.0],
            }
        ),
        atomic_id_column="cell_id",
        geography_columns=("region_code", "district_id"),
        probability_column="prob",
    )
    provider = StaticGeographyProvider(crosswalk=crosswalk)
    plan = GeographyAssignmentPlan(
        partition_columns=("region_code",),
        atomic_id_column="cell_id",
        geography_columns=("district_id",),
        partition_normalizers={
            "region_code": lambda value: str(int(round(float(value)))).zfill(2),
        },
        fallback_resolver=nearest_numeric_partition_key,
    )
    synthesizer = HierarchicalSynthesizer(
        geography_provider=provider,
        geography_assignment=plan,
        random_state=7,
    )
    households = pd.DataFrame({"region_code": [1.4, 1.8]})

    result = synthesizer._apply_geography_assignment(households)

    assert result["region_code"].tolist() == ["01", "02"]
    assert result["cell_id"].tolist() == ["01-a", "02-a"]
    assert result["district_id"].tolist() == ["north", "south"]
