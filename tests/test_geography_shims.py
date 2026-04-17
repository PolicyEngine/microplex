"""Compatibility coverage for moved US geography helpers."""

from __future__ import annotations

import pandas as pd

from microplex.geography import BlockGeography, derive_geographies


def test_block_geography_from_data_delegates_to_country_package() -> None:
    geography = BlockGeography.from_data(
        pd.DataFrame(
            {
                "geoid": ["060010001001001", "360610001001001"],
                "state_fips": ["06", "36"],
                "county": ["001", "061"],
                "tract": ["000100", "000100"],
                "tract_geoid": ["06001000100", "36061000100"],
                "cd_id": ["CA-13", "NY-12"],
                "prob": [0.6, 1.0],
                "national_prob": [0.4, 0.6],
            }
        )
    )

    assigned = geography.assign(pd.DataFrame({"state_fips": ["06", "36"]}), random_state=1)

    assert "block_geoid" in assigned.columns
    assert assigned["block_geoid"].str.startswith(("06", "36")).all()


def test_derive_geographies_delegates_to_country_package() -> None:
    result = derive_geographies(["060010001001001", "360610001001001"])

    assert list(result["state_fips"]) == ["06", "36"]
    assert list(result["county_fips"]) == ["06001", "36061"]
