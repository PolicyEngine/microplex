"""Compatibility coverage for moved hierarchical helpers."""

from __future__ import annotations

import pandas as pd

from microplex.hierarchical import prepare_cps_for_hierarchical


def test_prepare_cps_for_hierarchical_delegates_to_country_package() -> None:
    cps_data = pd.DataFrame(
        {
            "household_id": [1, 1, 2],
            "age": [40, 12, 35],
            "state_fips": [6, 6, 48],
        }
    )

    households, persons = prepare_cps_for_hierarchical(cps_data)

    assert list(households["n_persons"]) == [2, 1]
    assert list(households["n_children"]) == [1, 0]
    assert len(persons) == 3
