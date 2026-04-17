"""
Disability transition models for panel synthesis.

Implements disability onset and recovery hazard models based on
SSA Disability Insurance (DI) incidence and recovery rates.

Key data sources:
- SSA DI incidence rates by age/gender
- SSA DI recovery/termination rates by duration

Usage:
    >>> model = DisabilityTransitionModel()
    >>> trajectory = model.simulate_trajectory(initial_ages, years=10)
"""

from dataclasses import dataclass, field

import numpy as np

# Default SSA DI incidence rates by age group (annual probability)
# Source: SSA Annual Statistical Report on SSDI
DEFAULT_ONSET_RATES = {
    (18, 24): 0.002,  # 0.2%
    (25, 34): 0.003,  # 0.3%
    (35, 44): 0.005,  # 0.5%
    (45, 54): 0.010,  # 1.0%
    (55, 64): 0.015,  # 1.5%
    (65, 99): 0.005,  # Lower rate post-retirement age (some still working)
}

# Default recovery rates by disability duration (annual probability)
# Recovery rates decline sharply with duration
DEFAULT_RECOVERY_RATES = {
    1: 0.10,  # 10% in year 1
    2: 0.05,  # 5% in year 2
    3: 0.03,  # 3% in years 3+
}

# Gender multipliers (males have slightly higher DI incidence)
GENDER_MULTIPLIERS = {
    0: 0.95,  # Female
    1: 1.05,  # Male
}

# Age effect on recovery (younger = better recovery)
# Multiplier relative to age 45
AGE_RECOVERY_EFFECT = {
    (18, 34): 1.3,  # 30% higher recovery for young
    (35, 44): 1.1,  # 10% higher
    (45, 54): 1.0,  # Baseline
    (55, 64): 0.8,  # 20% lower for older
    (65, 99): 0.6,  # 40% lower post-65
}


class DisabilityOnset:
    """
    Model for disability onset probability.

    Computes annual probability of becoming disabled based on
    age and optionally gender, using SSA DI incidence rates.

    Attributes:
        base_rates: Dict mapping (age_min, age_max) -> annual probability
    """

    def __init__(
        self,
        base_rates: dict[tuple[int, int], float] | None = None,
        gender_multipliers: dict[int, float] | None = None,
    ):
        """
        Initialize disability onset model.

        Args:
            base_rates: Custom onset rates by age group.
                        Format: {(age_min, age_max): probability, ...}
                        If None, uses SSA DI default rates.
            gender_multipliers: Custom gender multipliers.
                                Format: {0: female_mult, 1: male_mult}
                                If None, uses default multipliers.
        """
        self.base_rates = base_rates or DEFAULT_ONSET_RATES
        self.gender_multipliers = gender_multipliers or GENDER_MULTIPLIERS

        # Pre-compute age boundaries for vectorized lookup
        self._age_bounds = sorted(self.base_rates.keys())

    def _get_base_rate(self, ages: np.ndarray) -> np.ndarray:
        """Get base onset rate for each age using vectorized lookup."""
        rates = np.zeros(len(ages))

        for (age_min, age_max), rate in self.base_rates.items():
            mask = (ages >= age_min) & (ages <= age_max)
            rates[mask] = rate

        return rates

    def probability(
        self,
        ages: np.ndarray,
        gender: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute disability onset probability for each individual.

        Args:
            ages: Array of ages [n]
            gender: Optional array of gender (0=female, 1=male) [n]

        Returns:
            Array of onset probabilities [n]
        """
        ages = np.asarray(ages)
        probs = self._get_base_rate(ages)

        if gender is not None:
            gender = np.asarray(gender)
            gender_mult = np.where(
                gender == 1,
                self.gender_multipliers.get(1, 1.0),
                self.gender_multipliers.get(0, 1.0),
            )
            probs = probs * gender_mult

        # Ensure valid probability bounds
        return np.clip(probs, 0.0, 1.0)

    def sample(
        self,
        ages: np.ndarray,
        gender: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Sample disability onset events.

        Args:
            ages: Array of ages [n]
            gender: Optional array of gender [n]

        Returns:
            Binary array indicating new disability (1) or not (0) [n]
        """
        probs = self.probability(ages, gender)
        return (np.random.random(len(ages)) < probs).astype(np.int32)


class DisabilityRecovery:
    """
    Model for disability recovery probability.

    Computes annual probability of recovering from disability based on
    age and disability duration. Recovery rates decline sharply with
    duration and moderately with age.

    Attributes:
        base_rates: Dict mapping duration -> annual recovery probability
    """

    def __init__(
        self,
        base_rates: dict[int, float] | None = None,
        age_effects: dict[tuple[int, int], float] | None = None,
    ):
        """
        Initialize disability recovery model.

        Args:
            base_rates: Custom recovery rates by duration (years).
                        Format: {1: prob_year1, 2: prob_year2, 3: prob_year3+}
                        Keys should be integers; highest key used for longer durations.
                        If None, uses default rates.
            age_effects: Custom age effect multipliers.
                         Format: {(age_min, age_max): multiplier, ...}
                         If None, uses default age effects.
        """
        self.base_rates = base_rates or DEFAULT_RECOVERY_RATES
        self.age_effects = age_effects or AGE_RECOVERY_EFFECT

        # Sort duration keys for lookup
        self._duration_keys = sorted(self.base_rates.keys())

    def _get_duration_rate(self, durations: np.ndarray) -> np.ndarray:
        """Get base recovery rate for each duration."""
        rates = np.zeros(len(durations))

        for i, dur in enumerate(durations):
            # Find highest key <= duration
            rate_key = 1
            for key in self._duration_keys:
                if key <= dur:
                    rate_key = key
            rates[i] = self.base_rates[rate_key]

        return rates

    def _get_age_multiplier(self, ages: np.ndarray) -> np.ndarray:
        """Get age effect multiplier for each age."""
        multipliers = np.ones(len(ages))

        for (age_min, age_max), mult in self.age_effects.items():
            mask = (ages >= age_min) & (ages <= age_max)
            multipliers[mask] = mult

        return multipliers

    def probability(
        self,
        ages: np.ndarray,
        durations: np.ndarray,
    ) -> np.ndarray:
        """
        Compute disability recovery probability.

        Args:
            ages: Array of ages [n]
            durations: Array of disability durations in years [n]

        Returns:
            Array of recovery probabilities [n]
        """
        ages = np.asarray(ages)
        durations = np.asarray(durations)

        base_rates = self._get_duration_rate(durations)
        age_mult = self._get_age_multiplier(ages)

        probs = base_rates * age_mult

        # Ensure valid probability bounds
        return np.clip(probs, 0.0, 1.0)

    def sample(
        self,
        ages: np.ndarray,
        durations: np.ndarray,
    ) -> np.ndarray:
        """
        Sample disability recovery events.

        Args:
            ages: Array of ages [n]
            durations: Array of disability durations [n]

        Returns:
            Binary array indicating recovery (1) or continued disability (0) [n]
        """
        probs = self.probability(ages, durations)
        return (np.random.random(len(ages)) < probs).astype(np.int32)


@dataclass
class DisabilityTransitionModel:
    """
    Combined disability transition model for panel synthesis.

    Manages both onset and recovery transitions, simulating
    disability trajectories over time.

    Example:
        >>> model = DisabilityTransitionModel()
        >>> # Single year transition
        >>> new_disabled, new_duration = model.simulate_year(
        ...     ages, is_disabled, disability_duration
        ... )
        >>> # Multi-year trajectory
        >>> trajectory = model.simulate_trajectory(initial_ages, years=10)
    """

    onset_model: DisabilityOnset = field(default_factory=DisabilityOnset)
    recovery_model: DisabilityRecovery = field(default_factory=DisabilityRecovery)

    def simulate_year(
        self,
        ages: np.ndarray,
        is_disabled: np.ndarray,
        disability_duration: np.ndarray,
        gender: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate one year of disability transitions.

        For each individual:
        - If not disabled: may become disabled (onset)
        - If disabled: may recover or continue disabled (duration increases)

        Args:
            ages: Array of ages [n]
            is_disabled: Boolean array of current disability status [n]
            disability_duration: Array of disability duration in years [n]
            gender: Optional array of gender [n]

        Returns:
            Tuple of:
            - new_disabled: Boolean array of new disability status [n]
            - new_duration: Array of new disability durations [n]
        """
        ages = np.asarray(ages)
        is_disabled = np.asarray(is_disabled, dtype=bool)
        disability_duration = np.asarray(disability_duration)
        n = len(ages)

        new_disabled = np.zeros(n, dtype=bool)
        new_duration = np.zeros(n)

        # Handle currently non-disabled: check for onset
        not_disabled_mask = ~is_disabled
        if not_disabled_mask.any():
            onset_events = self.onset_model.sample(
                ages[not_disabled_mask],
                gender=gender[not_disabled_mask] if gender is not None else None,
            )
            newly_disabled = not_disabled_mask.copy()
            newly_disabled[not_disabled_mask] = onset_events.astype(bool)
            new_disabled[newly_disabled] = True
            new_duration[newly_disabled] = 1

        # Handle currently disabled: check for recovery
        disabled_mask = is_disabled
        if disabled_mask.any():
            recovery_events = self.recovery_model.sample(
                ages[disabled_mask],
                disability_duration[disabled_mask],
            )
            # Those who don't recover stay disabled
            still_disabled = disabled_mask.copy()
            still_disabled[disabled_mask] = ~recovery_events.astype(bool)
            new_disabled[still_disabled] = True
            new_duration[still_disabled] = disability_duration[still_disabled] + 1

        return new_disabled, new_duration

    def simulate_trajectory(
        self,
        initial_ages: np.ndarray,
        years: int,
        initial_disabled: np.ndarray | None = None,
        initial_duration: np.ndarray | None = None,
        gender: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Simulate multi-year disability trajectories.

        Args:
            initial_ages: Array of starting ages [n]
            years: Number of years to simulate
            initial_disabled: Optional initial disability status [n]
            initial_duration: Optional initial disability duration [n]
            gender: Optional gender array [n]

        Returns:
            Dict with:
            - "is_disabled": Boolean array [n, years]
            - "duration": Array of durations [n, years]
            - "age": Array of ages [n, years]
        """
        initial_ages = np.asarray(initial_ages)
        n = len(initial_ages)

        # Initialize state
        is_disabled = (
            np.asarray(initial_disabled, dtype=bool)
            if initial_disabled is not None
            else np.zeros(n, dtype=bool)
        )
        duration = (
            np.asarray(initial_duration)
            if initial_duration is not None
            else np.zeros(n)
        )

        # Output arrays
        disabled_trajectory = np.zeros((n, years), dtype=bool)
        duration_trajectory = np.zeros((n, years))
        age_trajectory = np.zeros((n, years), dtype=np.int32)

        for t in range(years):
            current_ages = initial_ages + t
            age_trajectory[:, t] = current_ages

            is_disabled, duration = self.simulate_year(
                current_ages,
                is_disabled,
                duration,
                gender=gender,
            )

            disabled_trajectory[:, t] = is_disabled
            duration_trajectory[:, t] = duration

        return {
            "is_disabled": disabled_trajectory,
            "duration": duration_trajectory,
            "age": age_trajectory,
        }
