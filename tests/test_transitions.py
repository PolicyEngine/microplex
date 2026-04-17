"""
Tests for demographic transition models.

TDD tests that verify transition models for panel synthesis:
1. MarriageTransition - hazard rates for getting married
2. DivorceTransition - hazard rates for divorce
3. Vectorized application to datasets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest


class TestMarriageTransitionInit:
    """Test MarriageTransition initialization."""

    def test_basic_initialization(self):
        """Should initialize with default rates."""
        from microplex.transitions.demographic import MarriageTransition

        transition = MarriageTransition()

        assert hasattr(transition, "base_rates")
        assert hasattr(transition, "age_effects")

    def test_custom_base_rates(self):
        """Should accept custom base rates."""
        from microplex.transitions.demographic import MarriageTransition

        custom_rates = {
            "male": 0.08,
            "female": 0.10,
        }
        transition = MarriageTransition(base_rates=custom_rates)

        assert transition.base_rates["male"] == 0.08
        assert transition.base_rates["female"] == 0.10


class TestMarriageTransitionRates:
    """Test marriage hazard rate calculation."""

    @pytest.fixture
    def transition(self):
        """Create default transition model."""
        from microplex.transitions.demographic import MarriageTransition

        return MarriageTransition()

    def test_get_hazard_rate_single_person(self, transition):
        """Should compute hazard rate for a single person."""
        rate = transition.get_hazard_rate(
            age=25,
            is_male=True,
            is_married=False,
        )

        # Should return a probability between 0 and 1
        assert 0 <= rate <= 1

    def test_married_people_have_zero_rate(self, transition):
        """Already married people should have zero marriage rate."""
        rate = transition.get_hazard_rate(
            age=30,
            is_male=True,
            is_married=True,
        )

        assert rate == 0.0

    def test_young_people_have_lower_rates(self, transition):
        """Very young people should have lower marriage rates."""
        rate_18 = transition.get_hazard_rate(age=18, is_male=True, is_married=False)
        rate_28 = transition.get_hazard_rate(age=28, is_male=True, is_married=False)

        # Peak marriage ages are typically late 20s
        assert rate_18 < rate_28

    def test_old_people_have_lower_rates(self, transition):
        """Older people should have lower marriage rates."""
        rate_28 = transition.get_hazard_rate(age=28, is_male=True, is_married=False)
        rate_60 = transition.get_hazard_rate(age=60, is_male=True, is_married=False)

        assert rate_60 < rate_28

    def test_gender_affects_rates(self, transition):
        """Gender should affect marriage rates (women marry slightly earlier)."""
        rate_male = transition.get_hazard_rate(age=25, is_male=True, is_married=False)
        rate_female = transition.get_hazard_rate(age=25, is_male=False, is_married=False)

        # Rates should be different (not a strict ordering)
        # In US data, women tend to marry slightly earlier
        assert rate_male != rate_female


class TestMarriageTransitionVectorized:
    """Test vectorized application of marriage transitions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample population data."""
        np.random.seed(42)
        n = 1000

        return pd.DataFrame({
            "person_id": range(n),
            "age": np.random.randint(18, 80, n),
            "is_male": np.random.choice([True, False], n),
            "is_married": np.random.choice([True, False], n, p=[0.6, 0.4]),
        })

    def test_apply_returns_correct_shape(self, sample_data):
        """apply() should return array of same length."""
        from microplex.transitions.demographic import MarriageTransition

        transition = MarriageTransition()
        rates = transition.apply(sample_data)

        assert len(rates) == len(sample_data)

    def test_apply_returns_probabilities(self, sample_data):
        """apply() should return valid probabilities."""
        from microplex.transitions.demographic import MarriageTransition

        transition = MarriageTransition()
        rates = transition.apply(sample_data)

        assert (rates >= 0).all()
        assert (rates <= 1).all()

    def test_apply_married_have_zero_rate(self, sample_data):
        """apply() should give zero rates to married people."""
        from microplex.transitions.demographic import MarriageTransition

        transition = MarriageTransition()
        rates = transition.apply(sample_data)

        married_mask = sample_data["is_married"]
        assert (rates[married_mask] == 0).all()

    def test_simulate_transitions(self, sample_data):
        """simulate() should return transition indicators."""
        from microplex.transitions.demographic import MarriageTransition

        transition = MarriageTransition()
        np.random.seed(42)
        transitions = transition.simulate(sample_data)

        # Should be boolean array
        assert transitions.dtype == bool
        assert len(transitions) == len(sample_data)

        # Some transitions should occur
        assert transitions.sum() > 0

        # No married people should transition
        married_mask = sample_data["is_married"]
        assert (~transitions[married_mask]).all()


class TestDivorceTransitionInit:
    """Test DivorceTransition initialization."""

    def test_basic_initialization(self):
        """Should initialize with default rates."""
        from microplex.transitions.demographic import DivorceTransition

        transition = DivorceTransition()

        assert hasattr(transition, "base_rates")
        assert hasattr(transition, "duration_effects")


class TestDivorceTransitionRates:
    """Test divorce hazard rate calculation."""

    @pytest.fixture
    def transition(self):
        """Create default transition model."""
        from microplex.transitions.demographic import DivorceTransition

        return DivorceTransition()

    def test_get_hazard_rate_married_person(self, transition):
        """Should compute hazard rate for a married person."""
        rate = transition.get_hazard_rate(
            age=35,
            is_male=True,
            is_married=True,
            marriage_duration=5,
        )

        # Should return a probability between 0 and 1
        assert 0 <= rate <= 1

    def test_unmarried_people_have_zero_rate(self, transition):
        """Unmarried people should have zero divorce rate."""
        rate = transition.get_hazard_rate(
            age=30,
            is_male=True,
            is_married=False,
            marriage_duration=0,
        )

        assert rate == 0.0

    def test_short_marriages_have_higher_rates(self, transition):
        """Short-duration marriages have higher divorce rates."""
        rate_2yr = transition.get_hazard_rate(
            age=30, is_male=True, is_married=True, marriage_duration=2
        )
        rate_15yr = transition.get_hazard_rate(
            age=45, is_male=True, is_married=True, marriage_duration=15
        )

        # Early years of marriage have higher divorce risk
        assert rate_2yr > rate_15yr

    def test_age_affects_rates(self, transition):
        """Age should affect divorce rates."""
        # Young marriages tend to have higher divorce rates
        rate_young = transition.get_hazard_rate(
            age=22, is_male=True, is_married=True, marriage_duration=2
        )
        rate_older = transition.get_hazard_rate(
            age=35, is_male=True, is_married=True, marriage_duration=2
        )

        assert rate_young > rate_older


class TestDivorceTransitionVectorized:
    """Test vectorized application of divorce transitions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample population data with marriage duration."""
        np.random.seed(42)
        n = 1000

        is_married = np.random.choice([True, False], n, p=[0.6, 0.4])
        # Marriage duration only meaningful for married people
        marriage_duration = np.where(
            is_married,
            np.random.randint(1, 30, n),
            0
        )

        return pd.DataFrame({
            "person_id": range(n),
            "age": np.random.randint(20, 75, n),
            "is_male": np.random.choice([True, False], n),
            "is_married": is_married,
            "marriage_duration": marriage_duration,
        })

    def test_apply_returns_correct_shape(self, sample_data):
        """apply() should return array of same length."""
        from microplex.transitions.demographic import DivorceTransition

        transition = DivorceTransition()
        rates = transition.apply(sample_data)

        assert len(rates) == len(sample_data)

    def test_apply_returns_probabilities(self, sample_data):
        """apply() should return valid probabilities."""
        from microplex.transitions.demographic import DivorceTransition

        transition = DivorceTransition()
        rates = transition.apply(sample_data)

        assert (rates >= 0).all()
        assert (rates <= 1).all()

    def test_apply_unmarried_have_zero_rate(self, sample_data):
        """apply() should give zero rates to unmarried people."""
        from microplex.transitions.demographic import DivorceTransition

        transition = DivorceTransition()
        rates = transition.apply(sample_data)

        unmarried_mask = ~sample_data["is_married"]
        assert (rates[unmarried_mask] == 0).all()

    def test_simulate_transitions(self, sample_data):
        """simulate() should return transition indicators."""
        from microplex.transitions.demographic import DivorceTransition

        transition = DivorceTransition()
        np.random.seed(42)
        transitions = transition.simulate(sample_data)

        # Should be boolean array
        assert transitions.dtype == bool
        assert len(transitions) == len(sample_data)

        # Some transitions should occur (among married)
        married_mask = sample_data["is_married"]
        if married_mask.sum() > 100:  # Enough married people
            assert transitions[married_mask].sum() > 0

        # No unmarried people should transition
        assert (~transitions[~married_mask]).all()


class TestTransitionIntegration:
    """Test integration of transition models for panel synthesis."""

    @pytest.fixture
    def panel_data(self):
        """Create sample panel data (multiple years)."""
        np.random.seed(42)
        n = 500

        # Year 1 baseline
        return pd.DataFrame({
            "person_id": range(n),
            "year": 2020,
            "age": np.random.randint(18, 70, n),
            "is_male": np.random.choice([True, False], n),
            "is_married": np.random.choice([True, False], n, p=[0.5, 0.5]),
            "marriage_duration": np.random.randint(0, 20, n),
        })

    def test_apply_both_transitions(self, panel_data):
        """Should apply both marriage and divorce transitions."""
        from microplex.transitions.demographic import (
            DivorceTransition,
            MarriageTransition,
        )

        marriage_trans = MarriageTransition()
        divorce_trans = DivorceTransition()

        # Get transition probabilities
        marriage_rates = marriage_trans.apply(panel_data)
        divorce_rates = divorce_trans.apply(panel_data)

        # Marriage rates only for unmarried
        unmarried = ~panel_data["is_married"]
        assert (marriage_rates[~unmarried] == 0).all()

        # Divorce rates only for married
        married = panel_data["is_married"]
        assert (divorce_rates[~married] == 0).all()

    def test_simulate_year_transition(self, panel_data):
        """Should simulate a full year of transitions."""
        from microplex.transitions.demographic import (
            DivorceTransition,
            MarriageTransition,
        )

        marriage_trans = MarriageTransition()
        divorce_trans = DivorceTransition()

        np.random.seed(42)

        # Simulate transitions
        gets_married = marriage_trans.simulate(panel_data)
        gets_divorced = divorce_trans.simulate(panel_data)

        # Update marital status for next year
        next_year = panel_data.copy()
        next_year["year"] = 2021
        next_year["age"] = panel_data["age"] + 1

        # Apply marriage transitions (unmarried -> married)
        next_year.loc[gets_married, "is_married"] = True
        next_year.loc[gets_married, "marriage_duration"] = 0

        # Apply divorce transitions (married -> unmarried)
        next_year.loc[gets_divorced, "is_married"] = False
        next_year.loc[gets_divorced, "marriage_duration"] = 0

        # Increment marriage duration for those still married
        still_married = next_year["is_married"] & ~gets_married
        next_year.loc[still_married, "marriage_duration"] += 1

        # Validate counts changed
        original_married_count = panel_data["is_married"].sum()
        new_married_count = next_year["is_married"].sum()

        # Some change should have occurred
        assert original_married_count != new_married_count


class TestRealisticRates:
    """Test that transition rates match realistic CPS/SIPP ranges."""

    def test_marriage_rates_are_realistic(self):
        """Marriage rates should be in realistic range."""
        from microplex.transitions.demographic import MarriageTransition

        transition = MarriageTransition()

        # For peak marriage age (late 20s), rate should be ~5-10% annual
        rate = transition.get_hazard_rate(age=27, is_male=False, is_married=False)
        assert 0.03 <= rate <= 0.15

        # For older ages, should be lower
        rate_old = transition.get_hazard_rate(age=55, is_male=True, is_married=False)
        assert rate_old <= 0.05

    def test_divorce_rates_are_realistic(self):
        """Divorce rates should be in realistic range."""
        from microplex.transitions.demographic import DivorceTransition

        transition = DivorceTransition()

        # Overall divorce rate is ~1-3% annually
        rate = transition.get_hazard_rate(
            age=35, is_male=True, is_married=True, marriage_duration=5
        )
        assert 0.01 <= rate <= 0.05

        # Early marriages have higher rates
        rate_early = transition.get_hazard_rate(
            age=22, is_male=True, is_married=True, marriage_duration=2
        )
        assert rate_early >= rate

    def test_aggregate_rates_match_population(self):
        """Aggregate transition rates should match population patterns."""
        from microplex.transitions.demographic import (
            DivorceTransition,
            MarriageTransition,
        )

        np.random.seed(42)

        # Create representative population
        n = 10000
        age = np.random.choice(range(18, 75), n, p=np.ones(57) / 57)
        is_male = np.random.choice([True, False], n)
        is_married = np.random.choice([True, False], n, p=[0.5, 0.5])
        marriage_duration = np.where(
            is_married, np.random.randint(1, 20, n), 0
        )

        data = pd.DataFrame({
            "age": age,
            "is_male": is_male,
            "is_married": is_married,
            "marriage_duration": marriage_duration,
        })

        marriage_trans = MarriageTransition()
        divorce_trans = DivorceTransition()

        marriage_rates = marriage_trans.apply(data)
        divorce_rates = divorce_trans.apply(data)

        # Average marriage rate for unmarried should be ~5-10%
        unmarried_mask = ~data["is_married"]
        avg_marriage_rate = marriage_rates[unmarried_mask].mean()
        assert 0.02 <= avg_marriage_rate <= 0.15

        # Average divorce rate for married should be ~1-3%
        married_mask = data["is_married"]
        avg_divorce_rate = divorce_rates[married_mask].mean()
        assert 0.005 <= avg_divorce_rate <= 0.05
