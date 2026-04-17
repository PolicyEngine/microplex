"""Tests for disability transition models."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.transitions.disability import (
    DisabilityOnset,
    DisabilityRecovery,
    DisabilityTransitionModel,
)


class TestDisabilityOnset:
    """Tests for DisabilityOnset model."""

    def test_init_default_rates(self):
        """Should initialize with SSA-based default rates."""
        model = DisabilityOnset()
        assert hasattr(model, "base_rates")
        assert len(model.base_rates) > 0

    def test_probability_increases_with_age(self):
        """Disability onset probability should increase with age (working ages)."""
        model = DisabilityOnset()
        # Test only working ages where DI incidence consistently increases
        ages = np.array([25, 35, 45, 55])

        probs = model.probability(ages)

        # Each successive age should have higher probability
        for i in range(len(probs) - 1):
            assert probs[i + 1] > probs[i], (
                f"Age {ages[i+1]} should have higher probability than age {ages[i]}"
            )

    def test_probability_vectorized(self):
        """Should handle vectorized inputs efficiently."""
        model = DisabilityOnset()
        n = 10000
        ages = np.random.randint(25, 65, size=n)

        probs = model.probability(ages)

        assert len(probs) == n
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_bounds(self):
        """Probabilities should be between 0 and 1."""
        model = DisabilityOnset()
        ages = np.arange(18, 80)

        probs = model.probability(ages)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_approximate_ssa_rates(self):
        """Probabilities should be approximately correct for SSA DI rates."""
        model = DisabilityOnset()

        # Age 25-34: ~0.3% annual incidence
        age_30_prob = model.probability(np.array([30]))[0]
        assert 0.001 < age_30_prob < 0.01, f"Age 30 prob {age_30_prob} not in expected range"

        # Age 45-54: ~1.0% annual incidence
        age_50_prob = model.probability(np.array([50]))[0]
        assert 0.005 < age_50_prob < 0.02, f"Age 50 prob {age_50_prob} not in expected range"

        # Age 55-64: ~1.5% annual incidence
        age_60_prob = model.probability(np.array([60]))[0]
        assert 0.01 < age_60_prob < 0.03, f"Age 60 prob {age_60_prob} not in expected range"

    def test_gender_effect(self):
        """Should support gender-specific rates if provided."""
        model = DisabilityOnset()
        ages = np.array([40, 40])
        genders = np.array([0, 1])  # 0=female, 1=male

        probs = model.probability(ages, gender=genders)

        # Just check it runs and returns valid probabilities
        assert len(probs) == 2
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_sample_transitions(self):
        """Should sample disability onset events."""
        model = DisabilityOnset()
        n = 10000
        ages = np.full(n, 50)  # All age 50

        np.random.seed(42)
        transitions = model.sample(ages)

        # Should be binary outcomes
        assert set(np.unique(transitions)).issubset({0, 1})

        # Rate should be approximately correct (around 1% for age 50)
        transition_rate = transitions.mean()
        assert 0.005 < transition_rate < 0.02


class TestDisabilityRecovery:
    """Tests for DisabilityRecovery model."""

    def test_init_default_rates(self):
        """Should initialize with default recovery rates."""
        model = DisabilityRecovery()
        assert hasattr(model, "base_rates")

    def test_recovery_decreases_with_duration(self):
        """Recovery probability should decrease with disability duration."""
        model = DisabilityRecovery()
        ages = np.full(5, 45)  # Same age
        durations = np.array([1, 2, 3, 4, 5])  # Different durations

        probs = model.probability(ages, durations)

        # Each successive duration should have lower or equal probability
        for i in range(len(probs) - 1):
            assert probs[i + 1] <= probs[i], (
                f"Duration {durations[i+1]} should have lower recovery prob than {durations[i]}"
            )

    def test_recovery_decreases_with_age(self):
        """Recovery probability should decrease with age."""
        model = DisabilityRecovery()
        ages = np.array([30, 40, 50, 60])
        durations = np.full(4, 1)  # Same duration

        probs = model.probability(ages, durations)

        # Generally decreasing with age
        assert probs[0] >= probs[-1], "Young should have higher recovery rate than old"

    def test_probability_vectorized(self):
        """Should handle vectorized inputs."""
        model = DisabilityRecovery()
        n = 10000
        ages = np.random.randint(25, 65, size=n)
        durations = np.random.randint(1, 10, size=n)

        probs = model.probability(ages, durations)

        assert len(probs) == n
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_approximate_recovery_rates(self):
        """Recovery rates should match approximate expectations."""
        model = DisabilityRecovery()

        # Year 1: ~10% recovery rate
        year1_prob = model.probability(np.array([45]), np.array([1]))[0]
        assert 0.05 < year1_prob < 0.20, f"Year 1 recovery {year1_prob} not in expected range"

        # Year 2+: ~5% recovery rate
        year3_prob = model.probability(np.array([45]), np.array([3]))[0]
        assert 0.02 < year3_prob < 0.10, f"Year 3 recovery {year3_prob} not in expected range"

    def test_sample_recoveries(self):
        """Should sample recovery events."""
        model = DisabilityRecovery()
        n = 10000
        ages = np.full(n, 45)
        durations = np.ones(n)  # All year 1

        np.random.seed(42)
        recoveries = model.sample(ages, durations)

        # Should be binary outcomes
        assert set(np.unique(recoveries)).issubset({0, 1})

        # Rate should be approximately correct (around 10% for year 1)
        recovery_rate = recoveries.mean()
        assert 0.05 < recovery_rate < 0.20


class TestDisabilityTransitionModel:
    """Tests for combined DisabilityTransitionModel."""

    def test_init(self):
        """Should initialize with onset and recovery models."""
        model = DisabilityTransitionModel()
        assert hasattr(model, "onset_model")
        assert hasattr(model, "recovery_model")
        assert isinstance(model.onset_model, DisabilityOnset)
        assert isinstance(model.recovery_model, DisabilityRecovery)

    def test_simulate_year(self):
        """Should simulate one year of transitions."""
        model = DisabilityTransitionModel()
        n = 1000

        # Initial state: no one disabled
        ages = np.random.randint(25, 65, size=n)
        is_disabled = np.zeros(n, dtype=bool)
        disability_duration = np.zeros(n)

        np.random.seed(42)
        new_disabled, new_duration = model.simulate_year(
            ages, is_disabled, disability_duration
        )

        # Some should have become disabled
        assert new_disabled.sum() > 0
        # New disabled should have duration 1
        assert np.all(new_duration[new_disabled & ~is_disabled] == 1)
        # Non-disabled should have duration 0
        assert np.all(new_duration[~new_disabled] == 0)

    def test_simulate_year_with_recovery(self):
        """Should allow recovery for disabled individuals."""
        model = DisabilityTransitionModel()
        n = 1000

        # Initial state: everyone disabled for 1 year
        ages = np.full(n, 45)
        is_disabled = np.ones(n, dtype=bool)
        disability_duration = np.ones(n)

        np.random.seed(42)
        new_disabled, new_duration = model.simulate_year(
            ages, is_disabled, disability_duration
        )

        # Some should have recovered
        recovered = is_disabled & ~new_disabled
        assert recovered.sum() > 0

        # Those still disabled should have duration incremented
        still_disabled = is_disabled & new_disabled
        assert np.all(new_duration[still_disabled] == 2)

    def test_simulate_trajectory(self):
        """Should simulate multi-year trajectory."""
        model = DisabilityTransitionModel()
        n = 100
        years = 10

        initial_ages = np.random.randint(25, 55, size=n)
        np.random.seed(42)

        trajectory = model.simulate_trajectory(initial_ages, years)

        # Should have right shape
        assert trajectory["is_disabled"].shape == (n, years)
        assert trajectory["duration"].shape == (n, years)
        assert trajectory["age"].shape == (n, years)

        # Ages should increment each year
        for t in range(years):
            expected_age = initial_ages + t
            np.testing.assert_array_equal(trajectory["age"][:, t], expected_age)

        # Some disability should occur over 10 years
        ever_disabled = trajectory["is_disabled"].any(axis=1)
        assert ever_disabled.sum() > 0

    def test_simulate_trajectory_consistent(self):
        """Trajectories should be internally consistent."""
        model = DisabilityTransitionModel()
        n = 100
        years = 10

        initial_ages = np.full(n, 40)
        np.random.seed(42)

        trajectory = model.simulate_trajectory(initial_ages, years)

        for i in range(n):
            for t in range(1, years):
                if trajectory["is_disabled"][i, t]:
                    # If disabled now, either was disabled before or newly disabled
                    if trajectory["is_disabled"][i, t - 1]:
                        # Was disabled: duration should increment
                        assert trajectory["duration"][i, t] == trajectory["duration"][i, t - 1] + 1
                    else:
                        # Newly disabled: duration should be 1
                        assert trajectory["duration"][i, t] == 1
                else:
                    # Not disabled: duration should be 0
                    assert trajectory["duration"][i, t] == 0


class TestCustomRates:
    """Tests for using custom rate specifications."""

    def test_custom_onset_rates(self):
        """Should accept custom onset rates."""
        custom_rates = {
            (25, 34): 0.005,  # 0.5% for age 25-34
            (35, 44): 0.010,  # 1.0% for age 35-44
            (45, 54): 0.020,  # 2.0% for age 45-54
            (55, 64): 0.030,  # 3.0% for age 55-64
        }

        model = DisabilityOnset(base_rates=custom_rates)

        # Check rates match custom values
        prob_30 = model.probability(np.array([30]))[0]
        assert np.isclose(prob_30, 0.005, rtol=0.1)

    def test_custom_recovery_rates(self):
        """Should accept custom recovery rates by duration."""
        custom_rates = {
            1: 0.15,  # 15% year 1
            2: 0.08,  # 8% year 2
            3: 0.04,  # 4% year 3+
        }

        model = DisabilityRecovery(base_rates=custom_rates)

        ages = np.array([45, 45, 45])
        durations = np.array([1, 2, 5])

        probs = model.probability(ages, durations)

        assert np.isclose(probs[0], 0.15, rtol=0.1)
        assert np.isclose(probs[1], 0.08, rtol=0.1)
        # Duration 5 should use the rate for 3+ (highest key <= duration)
        assert np.isclose(probs[2], 0.04, rtol=0.1)
