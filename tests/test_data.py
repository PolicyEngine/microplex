"""
Tests for microplex.data module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microplex.data import (
    _prepare_household_data,
    _prepare_person_data,
    create_sample_data,
    get_data_info,
    load_cps_asec,
    load_cps_for_synthesis,
)


class TestCreateSampleData:
    """Tests for create_sample_data function."""

    def test_returns_tuple_of_dataframes(self):
        """Should return (households, persons) tuple."""
        result = create_sample_data(n_households=100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        households, persons = result
        assert isinstance(households, pd.DataFrame)
        assert isinstance(persons, pd.DataFrame)

    def test_household_count_matches_request(self):
        """Should create requested number of households."""
        households, _ = create_sample_data(n_households=500)
        assert len(households) == 500

    def test_person_count_reasonable(self):
        """Person count should be reasonable multiple of households."""
        households, persons = create_sample_data(n_households=1000, seed=42)
        # Average household size should be ~2-3 persons
        avg_size = len(persons) / len(households)
        assert 1.5 < avg_size < 4.0

    def test_household_required_columns(self):
        """Households should have all required columns."""
        households, _ = create_sample_data(n_households=100)
        required = ["household_id", "n_persons", "n_adults", "n_children",
                    "state_fips", "tenure", "hh_weight"]
        for col in required:
            assert col in households.columns

    def test_person_required_columns(self):
        """Persons should have all required columns."""
        _, persons = create_sample_data(n_households=100)
        required = ["person_id", "household_id", "age", "sex", "income",
                    "employment_status", "education", "relationship_to_head"]
        for col in required:
            assert col in persons.columns

    def test_household_ids_unique(self):
        """Household IDs should be unique."""
        households, _ = create_sample_data(n_households=500)
        assert households["household_id"].nunique() == len(households)

    def test_person_ids_unique(self):
        """Person IDs should be unique."""
        _, persons = create_sample_data(n_households=500)
        assert persons["person_id"].nunique() == len(persons)

    def test_n_persons_equals_sum_adults_children(self):
        """n_persons should equal n_adults + n_children."""
        households, _ = create_sample_data(n_households=500)
        np.testing.assert_array_equal(
            households["n_persons"],
            households["n_adults"] + households["n_children"]
        )

    def test_at_least_one_adult_per_household(self):
        """Each household should have at least one adult."""
        households, _ = create_sample_data(n_households=500)
        assert (households["n_adults"] >= 1).all()

    def test_person_ages_valid(self):
        """Person ages should be in valid range."""
        _, persons = create_sample_data(n_households=500)
        assert (persons["age"] >= 0).all()
        assert (persons["age"] <= 120).all()

    def test_adults_are_18_plus(self):
        """Adults should be age 18 or older."""
        households, persons = create_sample_data(n_households=500)
        for hh_id in households["household_id"][:50]:  # Check subset
            hh_persons = persons[persons["household_id"] == hh_id]
            adults = hh_persons[hh_persons["relationship_to_head"].isin([1, 2, 3])]
            # First few persons are adults
            assert (adults["age"] >= 18).all()

    def test_children_are_under_18(self):
        """Children should be under age 18."""
        households, persons = create_sample_data(n_households=500)
        children = persons[persons["relationship_to_head"] == 4]
        assert (children["age"] < 18).all()

    def test_income_nonnegative(self):
        """Income should be non-negative."""
        _, persons = create_sample_data(n_households=500)
        assert (persons["income"] >= 0).all()

    def test_seed_reproducibility(self):
        """Same seed should produce identical data."""
        hh1, p1 = create_sample_data(n_households=100, seed=12345)
        hh2, p2 = create_sample_data(n_households=100, seed=12345)
        pd.testing.assert_frame_equal(hh1, hh2)
        pd.testing.assert_frame_equal(p1, p2)

    def test_different_seeds_different_data(self):
        """Different seeds should produce different data."""
        hh1, _ = create_sample_data(n_households=100, seed=1)
        hh2, _ = create_sample_data(n_households=100, seed=2)
        # At least some values should differ
        assert not hh1["income" if "income" in hh1.columns else "state_fips"].equals(
            hh2["income" if "income" in hh2.columns else "state_fips"]
        )


class TestPrepareHouseholdData:
    """Tests for _prepare_household_data function."""

    def test_adds_missing_columns(self):
        """Should add missing columns with defaults."""
        df = pd.DataFrame({"household_id": [1, 2, 3]})
        result = _prepare_household_data(df)
        required = ["n_persons", "n_adults", "n_children", "state_fips", "tenure", "hh_weight"]
        for col in required:
            assert col in result.columns

    def test_preserves_existing_columns(self):
        """Should preserve existing column values."""
        df = pd.DataFrame({
            "household_id": [1, 2, 3],
            "n_persons": [2, 3, 4],
            "state_fips": [6, 36, 48],
        })
        result = _prepare_household_data(df)
        pd.testing.assert_series_equal(result["n_persons"], df["n_persons"])
        pd.testing.assert_series_equal(result["state_fips"], df["state_fips"])

    def test_n_persons_at_least_one(self):
        """n_persons should be at least 1."""
        df = pd.DataFrame({
            "household_id": [1, 2, 3],
            "n_persons": [0, -1, 2],
        })
        result = _prepare_household_data(df)
        assert (result["n_persons"] >= 1).all()


class TestPreparePersonData:
    """Tests for _prepare_person_data function."""

    def test_adds_missing_columns(self):
        """Should add missing columns with defaults."""
        df = pd.DataFrame({"person_id": [1, 2, 3]})
        result = _prepare_person_data(df)
        required = ["household_id", "age", "sex", "income", "employment_status", "education"]
        for col in required:
            assert col in result.columns

    def test_clips_age_to_valid_range(self):
        """Should clip age to 0-120."""
        df = pd.DataFrame({
            "person_id": [1, 2, 3],
            "age": [-5, 150, 50],
        })
        result = _prepare_person_data(df)
        assert (result["age"] >= 0).all()
        assert (result["age"] <= 120).all()


class TestLoadCpsAsec:
    """Tests for load_cps_asec function."""

    def test_raises_file_not_found_when_no_data(self):
        """Should raise FileNotFoundError when data files don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError) as exc_info:
                load_cps_asec(data_dir=tmpdir)
            assert "CPS ASEC data files not found" in str(exc_info.value)

    def test_loads_from_valid_directory(self):
        """Should load data when files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sample data and save
            hh, persons = create_sample_data(n_households=100)
            hh.to_parquet(tmpdir / "cps_asec_households.parquet")
            persons.to_parquet(tmpdir / "cps_asec_persons.parquet")

            # Load
            loaded_hh, loaded_persons = load_cps_asec(data_dir=tmpdir)

            assert len(loaded_hh) == 100
            assert len(loaded_persons) > 0

    def test_households_only_flag(self):
        """Should return only households when flag set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            hh, persons = create_sample_data(n_households=100)
            hh.to_parquet(tmpdir / "cps_asec_households.parquet")
            persons.to_parquet(tmpdir / "cps_asec_persons.parquet")

            result = load_cps_asec(data_dir=tmpdir, households_only=True)

            assert isinstance(result, pd.DataFrame)
            assert "household_id" in result.columns
            assert "person_id" not in result.columns

    def test_persons_only_flag(self):
        """Should return only persons when flag set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            hh, persons = create_sample_data(n_households=100)
            hh.to_parquet(tmpdir / "cps_asec_households.parquet")
            persons.to_parquet(tmpdir / "cps_asec_persons.parquet")

            result = load_cps_asec(data_dir=tmpdir, persons_only=True)

            assert isinstance(result, pd.DataFrame)
            assert "person_id" in result.columns


class TestLoadCpsForSynthesis:
    """Tests for load_cps_for_synthesis function."""

    def test_sampling_reduces_data(self):
        """Sample fraction should reduce number of households."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            hh, persons = create_sample_data(n_households=1000)
            hh.to_parquet(tmpdir / "cps_asec_households.parquet")
            persons.to_parquet(tmpdir / "cps_asec_persons.parquet")

            sampled_hh, sampled_persons = load_cps_for_synthesis(
                data_dir=tmpdir,
                sample_fraction=0.1,
                random_state=42,
            )

            # Should have approximately 10% of households
            assert 80 < len(sampled_hh) < 120

    def test_sampling_preserves_household_person_linkage(self):
        """Sampled persons should belong to sampled households."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            hh, persons = create_sample_data(n_households=500)
            hh.to_parquet(tmpdir / "cps_asec_households.parquet")
            persons.to_parquet(tmpdir / "cps_asec_persons.parquet")

            sampled_hh, sampled_persons = load_cps_for_synthesis(
                data_dir=tmpdir,
                sample_fraction=0.2,
            )

            # All person household_ids should be in sampled households
            sampled_hh_ids = set(sampled_hh["household_id"])
            person_hh_ids = set(sampled_persons["household_id"])
            assert person_hh_ids.issubset(sampled_hh_ids)


class TestGetDataInfo:
    """Tests for get_data_info function."""

    def test_returns_dict(self):
        """Should return dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = get_data_info(data_dir=tmpdir)
            assert isinstance(info, dict)

    def test_indicates_missing_files(self):
        """Should indicate files don't exist when missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = get_data_info(data_dir=tmpdir)
            assert info["households"]["exists"] is False
            assert info["persons"]["exists"] is False

    def test_provides_file_info_when_exists(self):
        """Should provide file info when files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            hh, persons = create_sample_data(n_households=100)
            hh.to_parquet(tmpdir / "cps_asec_households.parquet")
            persons.to_parquet(tmpdir / "cps_asec_persons.parquet")

            info = get_data_info(data_dir=tmpdir)

            assert info["households"]["exists"] is True
            assert info["households"]["n_records"] == 100
            assert "household_id" in info["households"]["columns"]

            assert info["persons"]["exists"] is True
            assert info["persons"]["n_records"] > 0
