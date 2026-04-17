"""Generic atomic-geography tests for the core microplex engine."""

from __future__ import annotations

import pandas as pd
import pytest

from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyProvider,
    GeographyQuery,
    ProbabilisticAtomicGeographyAssigner,
    StaticGeographyProvider,
    materialize_geographies,
    nearest_numeric_partition_key,
)


class TestAtomicGeographyCrosswalk:
    """Test generic atomic-geography crosswalk utilities."""

    @pytest.fixture
    def sample_crosswalk(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "atomic_id": ["A-1", "A-2", "B-1"],
                "region_code": ["01", "01", "02"],
                "state_fips": ["06", "06", "36"],
                "district_id": ["CA-01", "CA-02", "NY-01"],
                "prob": [0.25, 0.75, 1.0],
            }
        )

    def test_materialize_geographies_adds_requested_columns(
        self,
        sample_crosswalk: pd.DataFrame,
    ) -> None:
        crosswalk = AtomicGeographyCrosswalk(
            data=sample_crosswalk,
            atomic_id_column="atomic_id",
            geography_columns=("state_fips", "district_id"),
            probability_column="prob",
        )
        frame = pd.DataFrame({"atomic_id": ["A-2", "B-1"]})

        result = crosswalk.materialize(frame, columns=("district_id",))

        assert list(result.columns) == ["atomic_id", "district_id"]
        assert result["district_id"].tolist() == ["CA-02", "NY-01"]

    def test_materialize_can_overwrite_existing_columns(
        self,
        sample_crosswalk: pd.DataFrame,
    ) -> None:
        crosswalk = AtomicGeographyCrosswalk(
            data=sample_crosswalk,
            atomic_id_column="atomic_id",
            geography_columns=("region_code", "district_id"),
            probability_column="prob",
        )
        frame = pd.DataFrame(
            {
                "atomic_id": ["A-2"],
                "region_code": ["stale"],
                "district_id": ["stale-district"],
            }
        )

        result = crosswalk.materialize(
            frame,
            columns=("region_code", "district_id"),
            overwrite=True,
        )

        assert result.loc[0, "region_code"] == "01"
        assert result.loc[0, "district_id"] == "CA-02"

    def test_materialize_geographies_function_uses_all_columns_by_default(
        self,
        sample_crosswalk: pd.DataFrame,
    ) -> None:
        crosswalk = AtomicGeographyCrosswalk(
            data=sample_crosswalk,
            atomic_id_column="atomic_id",
            geography_columns=("region_code", "state_fips", "district_id"),
            probability_column="prob",
        )
        frame = pd.DataFrame({"atomic_id": ["A-1"]})

        result = materialize_geographies(frame, crosswalk)

        assert set(result.columns) == {
            "atomic_id",
            "region_code",
            "state_fips",
            "district_id",
        }
        assert result.loc[0, "state_fips"] == "06"
        assert result.loc[0, "district_id"] == "CA-01"

    def test_duplicate_atomic_ids_raise(self, sample_crosswalk: pd.DataFrame) -> None:
        duplicated = pd.concat(
            [sample_crosswalk, sample_crosswalk.iloc[[0]]],
            ignore_index=True,
        )

        with pytest.raises(ValueError, match="unique atomic ids"):
            AtomicGeographyCrosswalk(
                data=duplicated,
                atomic_id_column="atomic_id",
                geography_columns=("state_fips",),
            )


class TestProbabilisticAtomicGeographyAssigner:
    """Test grouped probabilistic assignment at the atomic geography level."""

    @pytest.fixture
    def sample_crosswalk(self) -> AtomicGeographyCrosswalk:
        return AtomicGeographyCrosswalk(
            data=pd.DataFrame(
                {
                    "atomic_id": ["A-1", "A-2", "B-1"],
                    "region_code": ["01", "01", "02"],
                    "county_code": ["01001", "01001", "02003"],
                    "prob": [0.2, 0.8, 1.0],
                }
            ),
            atomic_id_column="atomic_id",
            geography_columns=("region_code", "county_code"),
            probability_column="prob",
        )

    def test_assign_adds_atomic_ids_grouped_by_partition(
        self,
        sample_crosswalk: AtomicGeographyCrosswalk,
    ) -> None:
        assigner = ProbabilisticAtomicGeographyAssigner(
            crosswalk=sample_crosswalk,
            partition_columns=("region_code",),
        )
        frame = pd.DataFrame({"region_code": ["01", "02", "01"]})

        result = assigner.assign(frame, random_state=42)

        assert "atomic_id" in result.columns
        assert len(result) == 3
        assert result["atomic_id"].str.startswith(("A", "B")).all()
        assert result.loc[result["region_code"] == "02", "atomic_id"].iloc[0] == "B-1"

    def test_assign_is_reproducible(
        self,
        sample_crosswalk: AtomicGeographyCrosswalk,
    ) -> None:
        assigner = ProbabilisticAtomicGeographyAssigner(
            crosswalk=sample_crosswalk,
            partition_columns=("region_code",),
        )
        frame = pd.DataFrame({"region_code": ["01", "01", "02", "01"]})

        result_a = assigner.assign(frame, random_state=7)
        result_b = assigner.assign(frame, random_state=7)

        pd.testing.assert_series_equal(result_a["atomic_id"], result_b["atomic_id"])

    def test_assign_uses_fallback_resolver_for_missing_partition(
        self,
        sample_crosswalk: AtomicGeographyCrosswalk,
    ) -> None:
        assigner = ProbabilisticAtomicGeographyAssigner(
            crosswalk=sample_crosswalk,
            partition_columns=("region_code",),
            fallback_resolver=nearest_numeric_partition_key,
        )
        frame = pd.DataFrame({"region_code": [1.8]})

        result = assigner.assign(frame, random_state=1)

        assert result["atomic_id"].iloc[0] == "B-1"

    def test_assigner_exposes_public_partition_support_api(
        self,
        sample_crosswalk: AtomicGeographyCrosswalk,
    ) -> None:
        assigner = ProbabilisticAtomicGeographyAssigner(
            crosswalk=sample_crosswalk,
            partition_columns=("region_code",),
        )

        assert assigner.normalize_partition_key(("01",)) == ("01",)
        assert ("01",) in assigner.available_partition_keys
        assert assigner.supports_partition_key(("01",)) is True
        assert assigner.supports_partition_key(("99",)) is False

    def test_nearest_numeric_partition_key(self) -> None:
        resolved = nearest_numeric_partition_key(
            ("47",),
            (("06",), ("36",), ("48",)),
        )

        assert resolved == ("48",)


class TestGeographyProviders:
    def test_static_geography_provider_builds_crosswalk_and_assigner(self) -> None:
        crosswalk = AtomicGeographyCrosswalk(
            data=pd.DataFrame(
                {
                    "atomic_id": ["A-1", "A-2", "B-1"],
                    "region_code": ["01", "01", "02"],
                    "county_code": ["01001", "01003", "02059"],
                    "prob": [0.25, 0.75, 1.0],
                }
            ),
            atomic_id_column="atomic_id",
            geography_columns=("region_code", "county_code"),
            probability_column="prob",
        )
        provider = StaticGeographyProvider(
            crosswalk=crosswalk,
            default_partition_columns=("region_code",),
        )

        assert isinstance(provider, GeographyProvider)
        loaded_crosswalk = provider.load_crosswalk(
            GeographyQuery(geography_columns=("county_code",))
        )
        assigner = provider.load_assigner(GeographyQuery(partition_columns=("region_code",)))

        assert loaded_crosswalk.geography_columns == ("county_code",)
        result = assigner.assign(pd.DataFrame({"region_code": ["01", "02"]}), random_state=1)
        assert result["atomic_id"].str.startswith(("A", "B")).all()
