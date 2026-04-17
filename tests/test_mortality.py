"""
Tests for the Mortality transition model.

TDD tests that verify mortality modeling using SSA period life tables:
1. Load age/gender-specific death probabilities (qx values)
2. Apply stochastically to simulate deaths
3. Vectorized operations for efficiency

SSA 2021 Period Life Table qx (probability of dying within year):
- Age 0: 0.005 (M), 0.004 (F)
- Age 30: 0.001 (M), 0.0006 (F)
- Age 50: 0.004 (M), 0.002 (F)
- Age 65: 0.015 (M), 0.010 (F)
- Age 75: 0.035 (M), 0.024 (F)
- Age 85: 0.10 (M), 0.07 (F)
"""

import numpy as np
import pandas as pd
import pytest


class TestMortalityInit:
    """Test Mortality class initialization."""

    def test_basic_initialization(self):
        """Should initialize with default SSA 2021 life table."""
        from microplex.transitions.mortality import Mortality

        mortality = Mortality()

        assert mortality.year == 2021
        assert mortality.qx_male is not None
        assert mortality.qx_female is not None

    def test_custom_year_parameter(self):
        """Should accept year parameter (for future extension)."""
        from microplex.transitions.mortality import Mortality

        mortality = Mortality(year=2021)

        assert mortality.year == 2021

    def test_qx_arrays_have_correct_shape(self):
        """qx arrays should cover ages 0-119."""
        from microplex.transitions.mortality import Mortality

        mortality = Mortality()

        # Should have entries for ages 0-119 (120 total)
        assert len(mortality.qx_male) == 120
        assert len(mortality.qx_female) == 120

    def test_qx_values_are_probabilities(self):
        """qx values should be valid probabilities between 0 and 1."""
        from microplex.transitions.mortality import Mortality

        mortality = Mortality()

        assert np.all(mortality.qx_male >= 0)
        assert np.all(mortality.qx_male <= 1)
        assert np.all(mortality.qx_female >= 0)
        assert np.all(mortality.qx_female <= 1)

    def test_qx_increases_with_age_generally(self):
        """qx should generally increase with age (after infancy)."""
        from microplex.transitions.mortality import Mortality

        mortality = Mortality()

        # After age 10, mortality should generally increase
        # Check that age 80 > age 50 > age 30
        assert mortality.qx_male[80] > mortality.qx_male[50]
        assert mortality.qx_male[50] > mortality.qx_male[30]

        assert mortality.qx_female[80] > mortality.qx_female[50]
        assert mortality.qx_female[50] > mortality.qx_female[30]

    def test_male_qx_higher_than_female_at_most_ages(self):
        """Male mortality should be higher at most adult ages."""
        from microplex.transitions.mortality import Mortality

        mortality = Mortality()

        # Check at key ages
        for age in [30, 50, 65, 75, 85]:
            assert mortality.qx_male[age] > mortality.qx_female[age], (
                f"Male qx should be higher at age {age}"
            )


class TestSSALifeTableValues:
    """Test that loaded qx values match SSA 2021 period life table."""

    @pytest.fixture
    def mortality(self):
        """Create Mortality instance."""
        from microplex.transitions.mortality import Mortality

        return Mortality()

    def test_age_0_qx(self, mortality):
        """Age 0 qx should match SSA values."""
        # SSA 2021: M=0.005, F=0.004 (approximately)
        np.testing.assert_allclose(mortality.qx_male[0], 0.005, rtol=0.1)
        np.testing.assert_allclose(mortality.qx_female[0], 0.004, rtol=0.1)

    def test_age_30_qx(self, mortality):
        """Age 30 qx should match SSA values."""
        # SSA 2021: M=0.001, F=0.0006
        np.testing.assert_allclose(mortality.qx_male[30], 0.001, rtol=0.2)
        np.testing.assert_allclose(mortality.qx_female[30], 0.0006, rtol=0.2)

    def test_age_50_qx(self, mortality):
        """Age 50 qx should match SSA values."""
        # SSA 2021: M=0.004, F=0.002
        np.testing.assert_allclose(mortality.qx_male[50], 0.004, rtol=0.2)
        np.testing.assert_allclose(mortality.qx_female[50], 0.002, rtol=0.2)

    def test_age_65_qx(self, mortality):
        """Age 65 qx should match SSA values."""
        # SSA 2021: M=0.015, F=0.010
        np.testing.assert_allclose(mortality.qx_male[65], 0.015, rtol=0.2)
        np.testing.assert_allclose(mortality.qx_female[65], 0.010, rtol=0.2)

    def test_age_75_qx(self, mortality):
        """Age 75 qx should match SSA values."""
        # SSA 2021: M=0.035, F=0.024
        np.testing.assert_allclose(mortality.qx_male[75], 0.035, rtol=0.2)
        np.testing.assert_allclose(mortality.qx_female[75], 0.024, rtol=0.2)

    def test_age_85_qx(self, mortality):
        """Age 85 qx should match SSA values."""
        # SSA 2021: M=0.10, F=0.07
        np.testing.assert_allclose(mortality.qx_male[85], 0.10, rtol=0.2)
        np.testing.assert_allclose(mortality.qx_female[85], 0.07, rtol=0.2)


class TestGetQx:
    """Test get_qx method for looking up death probabilities."""

    @pytest.fixture
    def mortality(self):
        """Create Mortality instance."""
        from microplex.transitions.mortality import Mortality

        return Mortality()

    def test_get_qx_single_person(self, mortality):
        """Should return qx for a single person."""
        qx = mortality.get_qx(age=50, is_male=True)

        assert isinstance(qx, (float, np.floating))
        assert 0 < qx < 1

    def test_get_qx_array(self, mortality):
        """Should accept numpy arrays for vectorized lookup."""
        ages = np.array([30, 50, 70])
        is_male = np.array([True, False, True])

        qx = mortality.get_qx(age=ages, is_male=is_male)

        assert isinstance(qx, np.ndarray)
        assert len(qx) == 3

    def test_get_qx_handles_high_ages(self, mortality):
        """Should handle ages >= 120 by using age 119 qx."""
        qx_119 = mortality.get_qx(age=119, is_male=True)
        qx_120 = mortality.get_qx(age=120, is_male=True)
        qx_150 = mortality.get_qx(age=150, is_male=True)

        # All should use age 119 value
        assert qx_120 == qx_119
        assert qx_150 == qx_119


class TestApplyMortality:
    """Test apply method for stochastic death simulation."""

    @pytest.fixture
    def mortality(self):
        """Create Mortality instance."""
        from microplex.transitions.mortality import Mortality

        return Mortality()

    @pytest.fixture
    def population(self):
        """Create sample population DataFrame."""
        np.random.seed(42)
        n = 10000

        return pd.DataFrame({
            "person_id": range(n),
            "age": np.random.choice([30, 50, 70, 85], n),
            "is_male": np.random.choice([True, False], n),
            "weight": np.ones(n),
        })

    def test_apply_returns_deaths_mask(self, mortality, population):
        """apply() should return boolean mask of deaths."""
        np.random.seed(42)

        deaths = mortality.apply(
            age=population["age"].values,
            is_male=population["is_male"].values,
        )

        assert isinstance(deaths, np.ndarray)
        assert deaths.dtype == bool
        assert len(deaths) == len(population)

    def test_apply_is_reproducible_with_seed(self, mortality, population):
        """apply() should be reproducible with random seed."""
        deaths1 = mortality.apply(
            age=population["age"].values,
            is_male=population["is_male"].values,
            seed=42,
        )

        deaths2 = mortality.apply(
            age=population["age"].values,
            is_male=population["is_male"].values,
            seed=42,
        )

        np.testing.assert_array_equal(deaths1, deaths2)

    def test_apply_different_seeds_give_different_results(self, mortality, population):
        """Different seeds should give different death patterns."""
        deaths1 = mortality.apply(
            age=population["age"].values,
            is_male=population["is_male"].values,
            seed=42,
        )

        deaths2 = mortality.apply(
            age=population["age"].values,
            is_male=population["is_male"].values,
            seed=123,
        )

        # Should not be identical
        assert not np.array_equal(deaths1, deaths2)

    def test_apply_death_rate_approximately_matches_qx(self, mortality):
        """Simulated death rate should approximately match qx."""
        # Create homogeneous population of 50-year-old males
        n = 100000
        ages = np.full(n, 50)
        is_male = np.full(n, True)

        deaths = mortality.apply(age=ages, is_male=is_male, seed=42)

        observed_rate = deaths.sum() / n
        expected_rate = mortality.qx_male[50]

        # Should be within 10% of expected (statistical variation)
        np.testing.assert_allclose(observed_rate, expected_rate, rtol=0.1)

    def test_apply_higher_death_rate_for_elderly(self, mortality):
        """Elderly should have higher death rate than young."""
        n = 50000

        # Young population (age 30)
        young_deaths = mortality.apply(
            age=np.full(n, 30),
            is_male=np.full(n, True),
            seed=42,
        )

        # Elderly population (age 85)
        elderly_deaths = mortality.apply(
            age=np.full(n, 85),
            is_male=np.full(n, True),
            seed=42,
        )

        young_rate = young_deaths.sum() / n
        elderly_rate = elderly_deaths.sum() / n

        assert elderly_rate > young_rate * 10  # Much higher for elderly


class TestSimulateYear:
    """Test simulate_year method for advancing population one year."""

    @pytest.fixture
    def mortality(self):
        """Create Mortality instance."""
        from microplex.transitions.mortality import Mortality

        return Mortality()

    @pytest.fixture
    def population(self):
        """Create sample population DataFrame."""
        np.random.seed(42)
        n = 1000

        return pd.DataFrame({
            "person_id": range(n),
            "age": np.random.randint(0, 100, n),
            "is_male": np.random.choice([True, False], n),
            "alive": np.ones(n, dtype=bool),
            "weight": np.ones(n),
        })

    def test_simulate_year_returns_dataframe(self, mortality, population):
        """simulate_year() should return updated DataFrame."""
        result = mortality.simulate_year(
            population,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            seed=42,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(population)

    def test_simulate_year_increments_age(self, mortality, population):
        """simulate_year() should increment age by 1."""
        result = mortality.simulate_year(
            population,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            seed=42,
        )

        expected_ages = population["age"] + 1
        np.testing.assert_array_equal(result["age"], expected_ages)

    def test_simulate_year_marks_deaths(self, mortality, population):
        """simulate_year() should mark some as not alive."""
        result = mortality.simulate_year(
            population,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            seed=42,
        )

        # Some should be marked as dead
        n_dead = (~result["alive"]).sum()
        assert n_dead > 0
        assert n_dead < len(population)  # Not everyone dies

    def test_simulate_year_respects_already_dead(self, mortality, population):
        """simulate_year() should not resurrect the dead."""
        # Mark some as already dead
        population = population.copy()
        population.loc[population.index[:100], "alive"] = False

        result = mortality.simulate_year(
            population,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            seed=42,
        )

        # Already dead should still be dead
        originally_dead = ~population["alive"]
        assert (~result.loc[originally_dead.values, "alive"]).all()

    def test_simulate_year_adds_death_year(self, mortality, population):
        """simulate_year() can optionally record death year."""
        result = mortality.simulate_year(
            population,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            death_year_col="death_year",
            current_year=2024,
            seed=42,
        )

        # New deaths should have death_year = 2024
        new_deaths = ~result["alive"]
        assert "death_year" in result.columns
        # Dead people should have death year
        assert result.loc[new_deaths, "death_year"].notna().all()


class TestMultiYearSimulation:
    """Test simulating multiple years."""

    @pytest.fixture
    def mortality(self):
        """Create Mortality instance."""
        from microplex.transitions.mortality import Mortality

        return Mortality()

    def test_simulate_multiple_years(self, mortality):
        """Should correctly simulate over multiple years."""
        # Start with 10000 newborns
        n = 10000
        population = pd.DataFrame({
            "person_id": range(n),
            "age": np.zeros(n, dtype=int),
            "is_male": np.random.choice([True, False], n),
            "alive": np.ones(n, dtype=bool),
        })

        # Simulate 80 years
        for year in range(80):
            population = mortality.simulate_year(
                population,
                age_col="age",
                is_male_col="is_male",
                alive_col="alive",
                seed=year,  # Different seed each year
            )

        # After 80 years, most should still be alive (life expectancy ~78)
        survival_rate = population["alive"].mean()
        assert 0.3 < survival_rate < 0.7  # Reasonable range

        # All survivors should be age 80
        survivors = population[population["alive"]]
        assert (survivors["age"] == 80).all()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mortality(self):
        """Create Mortality instance."""
        from microplex.transitions.mortality import Mortality

        return Mortality()

    def test_empty_population(self, mortality):
        """Should handle empty population gracefully."""
        empty_pop = pd.DataFrame({
            "age": [],
            "is_male": [],
            "alive": [],
        })

        result = mortality.simulate_year(
            empty_pop,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
        )

        assert len(result) == 0

    def test_single_person(self, mortality):
        """Should handle single person population."""
        single = pd.DataFrame({
            "age": [50],
            "is_male": [True],
            "alive": [True],
        })

        result = mortality.simulate_year(
            single,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            seed=42,
        )

        assert len(result) == 1
        assert result["age"].iloc[0] == 51

    def test_very_old_ages(self, mortality):
        """Should handle ages beyond 119."""
        old = pd.DataFrame({
            "age": [119, 120, 130],
            "is_male": [True, True, True],
            "alive": [True, True, True],
        })

        result = mortality.simulate_year(
            old,
            age_col="age",
            is_male_col="is_male",
            alive_col="alive",
            seed=42,
        )

        # Should complete without error
        assert len(result) == 3
        # Ages should increment
        np.testing.assert_array_equal(result["age"], [120, 121, 131])

    def test_negative_ages_raises_error(self, mortality):
        """Should raise error for negative ages."""
        invalid = pd.DataFrame({
            "age": [-1, 50],
            "is_male": [True, True],
            "alive": [True, True],
        })

        with pytest.raises(ValueError, match="negative"):
            mortality.simulate_year(
                invalid,
                age_col="age",
                is_male_col="is_male",
                alive_col="alive",
            )
