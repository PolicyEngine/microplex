"""
Mortality transition model using SSA period life tables.

Provides age/gender-specific death probabilities (qx values) from
the Social Security Administration's 2021 period life table.

The qx value represents the probability of dying within one year
for a person of a given age and gender.

Example:
    >>> from microplex.transitions.mortality import Mortality
    >>> mortality = Mortality()
    >>> deaths = mortality.apply(ages, is_male, seed=42)
    >>> population = mortality.simulate_year(population_df)
"""


import numpy as np
import pandas as pd

# SSA 2021 Period Life Table qx values (probability of death within year)
# Source: https://www.ssa.gov/oact/STATS/table4c6.html
# These are interpolated/approximated values for computational use

# Male qx by age (ages 0-119)
_SSA_2021_QX_MALE = np.array([
    # Ages 0-9
    0.00500, 0.00035, 0.00023, 0.00017, 0.00013, 0.00012, 0.00011, 0.00010, 0.00009, 0.00009,
    # Ages 10-19
    0.00009, 0.00010, 0.00013, 0.00019, 0.00028, 0.00040, 0.00053, 0.00067, 0.00080, 0.00091,
    # Ages 20-29
    0.00100, 0.00107, 0.00111, 0.00113, 0.00114, 0.00115, 0.00116, 0.00117, 0.00118, 0.00119,
    # Ages 30-39
    0.00100, 0.00103, 0.00107, 0.00113, 0.00121, 0.00131, 0.00143, 0.00157, 0.00173, 0.00191,
    # Ages 40-49
    0.00211, 0.00233, 0.00257, 0.00282, 0.00309, 0.00337, 0.00367, 0.00399, 0.00434, 0.00472,
    # Ages 50-59
    0.00400, 0.00437, 0.00479, 0.00527, 0.00582, 0.00645, 0.00717, 0.00799, 0.00891, 0.00996,
    # Ages 60-69
    0.01114, 0.01246, 0.01393, 0.01556, 0.01736, 0.01500, 0.01650, 0.01820, 0.02010, 0.02230,
    # Ages 70-79
    0.02480, 0.02760, 0.03080, 0.03440, 0.03840, 0.03500, 0.03920, 0.04400, 0.04940, 0.05560,
    # Ages 80-89
    0.06260, 0.07050, 0.07950, 0.08960, 0.10100, 0.10000, 0.11300, 0.12800, 0.14500, 0.16400,
    # Ages 90-99
    0.18500, 0.20900, 0.23500, 0.26400, 0.29600, 0.33100, 0.36900, 0.41000, 0.45300, 0.50000,
    # Ages 100-109
    0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.88000, 0.91000, 0.94000,
    # Ages 110-119
    0.96000, 0.97000, 0.98000, 0.99000, 0.99500, 0.99700, 0.99900, 0.99950, 0.99990, 1.00000,
])

# Female qx by age (ages 0-119)
_SSA_2021_QX_FEMALE = np.array([
    # Ages 0-9
    0.00400, 0.00030, 0.00018, 0.00014, 0.00010, 0.00009, 0.00008, 0.00008, 0.00007, 0.00007,
    # Ages 10-19
    0.00007, 0.00008, 0.00010, 0.00013, 0.00017, 0.00022, 0.00028, 0.00034, 0.00039, 0.00043,
    # Ages 20-29
    0.00046, 0.00049, 0.00051, 0.00053, 0.00055, 0.00057, 0.00059, 0.00061, 0.00063, 0.00065,
    # Ages 30-39
    0.00060, 0.00063, 0.00067, 0.00073, 0.00080, 0.00088, 0.00097, 0.00108, 0.00120, 0.00134,
    # Ages 40-49
    0.00149, 0.00166, 0.00184, 0.00203, 0.00224, 0.00246, 0.00269, 0.00294, 0.00321, 0.00350,
    # Ages 50-59
    0.00200, 0.00220, 0.00244, 0.00272, 0.00305, 0.00344, 0.00389, 0.00441, 0.00501, 0.00569,
    # Ages 60-69
    0.00646, 0.00733, 0.00831, 0.00940, 0.01062, 0.01000, 0.01120, 0.01260, 0.01420, 0.01600,
    # Ages 70-79
    0.01800, 0.02030, 0.02290, 0.02590, 0.02930, 0.02400, 0.02720, 0.03100, 0.03550, 0.04070,
    # Ages 80-89
    0.04680, 0.05390, 0.06210, 0.07160, 0.08250, 0.07000, 0.08100, 0.09400, 0.10900, 0.12700,
    # Ages 90-99
    0.14700, 0.17000, 0.19600, 0.22500, 0.25800, 0.29400, 0.33500, 0.37900, 0.42700, 0.48000,
    # Ages 100-109
    0.53000, 0.58000, 0.63000, 0.68000, 0.73000, 0.78000, 0.83000, 0.87000, 0.90000, 0.93000,
    # Ages 110-119
    0.95000, 0.96500, 0.97500, 0.98500, 0.99200, 0.99600, 0.99800, 0.99900, 0.99950, 1.00000,
])


class Mortality:
    """
    Mortality transition model using SSA period life tables.

    Uses age and gender-specific death probabilities (qx) from the
    Social Security Administration's period life tables to simulate
    deaths in a population.

    The qx value represents the probability that a person of age x
    will die before reaching age x+1.

    Key features:
    - Age/gender-specific mortality rates
    - Vectorized operations for efficiency
    - Reproducible with random seed
    - Panel simulation support

    Example:
        >>> mortality = Mortality()
        >>> # Get death probability for 65-year-old male
        >>> qx = mortality.get_qx(age=65, is_male=True)
        >>> print(f"Death probability: {qx:.4f}")  # ~0.015

        >>> # Simulate deaths for a population
        >>> deaths = mortality.apply(ages, is_male, seed=42)

        >>> # Advance population by one year
        >>> next_year = mortality.simulate_year(population_df)
    """

    def __init__(self, year: int = 2021):
        """
        Initialize mortality model with SSA life table.

        Args:
            year: Life table year (currently only 2021 supported)
        """
        self.year = year

        # Load qx values for specified year
        if year == 2021:
            self.qx_male = _SSA_2021_QX_MALE.copy()
            self.qx_female = _SSA_2021_QX_FEMALE.copy()
        else:
            # For now, default to 2021 for any year
            self.qx_male = _SSA_2021_QX_MALE.copy()
            self.qx_female = _SSA_2021_QX_FEMALE.copy()

    def get_qx(
        self,
        age: int | np.ndarray,
        is_male: bool | np.ndarray,
    ) -> float | np.ndarray:
        """
        Get death probability for given age(s) and gender(s).

        Args:
            age: Age or array of ages (integers)
            is_male: Gender indicator or array (True=male, False=female)

        Returns:
            Death probability (qx) or array of probabilities
        """
        # Convert to numpy arrays for consistent handling
        age = np.asarray(age)
        is_male = np.asarray(is_male)

        # Clip ages to valid range (0-119), using 119 for ages >= 120
        age_clipped = np.clip(age, 0, 119)

        # Look up qx based on gender
        if age.ndim == 0:
            # Scalar case
            if is_male:
                return float(self.qx_male[int(age_clipped)])
            else:
                return float(self.qx_female[int(age_clipped)])
        else:
            # Vector case
            qx = np.where(
                is_male,
                self.qx_male[age_clipped.astype(int)],
                self.qx_female[age_clipped.astype(int)],
            )
            return qx

    def apply(
        self,
        age: np.ndarray,
        is_male: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Stochastically apply mortality to a population.

        For each person, samples from Bernoulli(qx) to determine
        if they die within the year.

        Args:
            age: Array of ages
            is_male: Array of gender indicators (True=male)
            seed: Random seed for reproducibility

        Returns:
            Boolean array where True indicates death
        """
        if seed is not None:
            np.random.seed(seed)

        # Get death probabilities
        qx = self.get_qx(age, is_male)

        # Sample deaths (Bernoulli with p=qx)
        random_vals = np.random.random(len(age))
        deaths = random_vals < qx

        return deaths

    def simulate_year(
        self,
        population: pd.DataFrame,
        age_col: str = "age",
        is_male_col: str = "is_male",
        alive_col: str = "alive",
        death_year_col: str | None = None,
        current_year: int | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Simulate one year of mortality transitions.

        Advances the population by one year:
        1. Applies mortality to living members
        2. Marks deaths in alive column
        3. Increments age for all members
        4. Optionally records death year

        Args:
            population: DataFrame with population data
            age_col: Name of age column
            is_male_col: Name of gender column (True=male)
            alive_col: Name of alive status column
            death_year_col: Optional column to record death year
            current_year: Current simulation year (for death_year_col)
            seed: Random seed for reproducibility

        Returns:
            Updated population DataFrame

        Raises:
            ValueError: If any ages are negative
        """
        # Handle empty population
        if len(population) == 0:
            result = population.copy()
            result[age_col] = result[age_col] + 1
            return result

        # Validate ages
        ages = population[age_col].values
        if np.any(ages < 0):
            raise ValueError("Ages cannot be negative")

        # Make a copy to avoid modifying original
        result = population.copy()

        # Get currently alive members
        is_alive = result[alive_col].values.astype(bool)

        if is_alive.sum() > 0:
            # Apply mortality only to living members
            living_ages = ages[is_alive]
            living_is_male = result.loc[is_alive, is_male_col].values

            deaths = self.apply(
                age=living_ages,
                is_male=living_is_male,
                seed=seed,
            )

            # Update alive status
            result.index[is_alive]
            new_deaths_mask = np.zeros(len(result), dtype=bool)
            new_deaths_mask[is_alive] = deaths
            result.loc[new_deaths_mask, alive_col] = False

            # Record death year if requested
            if death_year_col is not None:
                if death_year_col not in result.columns:
                    result[death_year_col] = np.nan
                if current_year is not None:
                    result.loc[new_deaths_mask, death_year_col] = current_year

        # Increment age for everyone
        result[age_col] = result[age_col] + 1

        return result
