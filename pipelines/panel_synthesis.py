from __future__ import annotations

"""Panel/longitudinal synthesis for the Microplex.

Architecture for generating lifetime trajectories (e.g., earnings histories
for Social Security modeling).

TRAINING DATA SOURCES:
- PSID (Panel Study of Income Dynamics): 50+ years, ~10k families
  - Registration required: https://psidonline.isr.umich.edu/
  - Best for: full career trajectories (age 18-70)

- SIPP (Survey of Income and Program Participation): 4-year panels, public
  - Available via HuggingFace: PolicyEngine/policyengine-us-data
  - Best for: short-term dynamics, program participation

ARCHITECTURE:
The key insight is generating trajectories ALL AT ONCE (not sequentially).

Sequential approach (QRF in social-security-model):
  P(earnings_25 | X), P(earnings_30 | X), ... separately
  Problem: must post-hoc smooth to ensure consistency

All-at-once approach (microplex):
  P(earnings_18:70 | X) as single 50-dimensional conditional flow
  Advantage: automatically preserves correlations across ages

The flow learns latent "trajectory types":
  Type A: Steady growth (professional career, concave shape)
  Type B: Peak-then-decline (manual labor, physical jobs)
  Type C: Volatile (self-employment, gig economy)
  Type D: Interrupted (disability, caregiving, unemployment spells)
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Import microplex
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from microplex import Synthesizer


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory synthesis."""

    # Time horizon
    start_age: int = 18
    end_age: int = 70

    # Variables to model trajectories for
    trajectory_vars: list[str] = field(default_factory=lambda: ["earnings"])

    # Conditioning variables (from cross-section)
    condition_vars: list[str] = field(
        default_factory=lambda: ["education", "gender", "birth_cohort"]
    )

    # Age granularity for trajectory (5-year intervals = fewer dimensions)
    age_interval: int = 5  # 1 = every year, 5 = every 5 years

    # Model hyperparameters
    n_layers: int = 8
    hidden_dim: int = 128
    epochs: int = 200

    @property
    def trajectory_ages(self) -> list[int]:
        """Ages at which we model earnings."""
        return list(range(self.start_age, self.end_age + 1, self.age_interval))

    @property
    def n_trajectory_dims(self) -> int:
        """Number of trajectory dimensions."""
        return len(self.trajectory_ages)


class TrajectoryModel:
    """Synthesizes lifetime trajectories for individuals.

    Generates full trajectory ALL AT ONCE using microplex's flow-based approach.
    This automatically preserves correlations across ages (people who earn more
    at 30 tend to earn more at 50).

    Training data: PSID (ideal, 50+ years) or SIPP (4-year panels, public)

    Example usage:
        # 1. Prepare wide-format panel data
        wide_panel = panel_to_wide(psid_long, id_col='person_id', age_col='age',
                                   earnings_col='earnings', condition_vars=['education', 'gender'])

        # 2. Train trajectory model
        model = TrajectoryModel(config)
        model.fit(wide_panel)

        # 3. Generate trajectories for CPS
        cps_with_trajectories = model.generate(cps_cross_section)
    """

    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.synthesizer_: Synthesizer | None = None
        self.is_fitted_ = False

    def fit(
        self,
        wide_panel: pd.DataFrame,
        weight_col: str | None = None,
        verbose: bool = True,
    ) -> TrajectoryModel:
        """Fit trajectory model on wide-format panel data.

        Args:
            wide_panel: Wide-format DataFrame with one row per person.
                Must have columns: condition_vars + earnings_age_X for each age X.
            weight_col: Optional weight column
            verbose: Whether to print progress

        Returns:
            self
        """
        # Build target column names
        target_cols = [
            f"earnings_age_{age}" for age in self.config.trajectory_ages
        ]

        # Check all required columns exist
        missing_targets = [c for c in target_cols if c not in wide_panel.columns]
        if missing_targets:
            raise ValueError(f"Missing trajectory columns: {missing_targets[:5]}...")

        missing_conds = [
            c for c in self.config.condition_vars if c not in wide_panel.columns
        ]
        if missing_conds:
            raise ValueError(f"Missing condition columns: {missing_conds}")

        if verbose:
            print("Training trajectory model...")
            print(f"  Individuals: {len(wide_panel):,}")
            print(f"  Trajectory dimensions: {len(target_cols)} ages")
            print(f"  Condition variables: {self.config.condition_vars}")

        # Create and train synthesizer
        self.synthesizer_ = Synthesizer(
            target_vars=target_cols,
            condition_vars=self.config.condition_vars,
            n_layers=self.config.n_layers,
            hidden_dim=self.config.hidden_dim,
            zero_inflated=True,  # Handle zero-earnings years
            variance_regularization=0.1,  # Prevent mode collapse
        )

        self.synthesizer_.fit(
            wide_panel,
            weight_col=weight_col,
            epochs=self.config.epochs,
            verbose=verbose,
        )

        self.is_fitted_ = True
        return self

    def generate(
        self,
        cross_section: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate trajectories for cross-sectional individuals.

        Takes a cross-section (e.g., from CPS) and generates full
        earnings histories for each person, conditioned on their demographics.

        Args:
            cross_section: Cross-sectional data with demographics.
                Must have columns matching config.condition_vars.
            seed: Random seed for reproducibility

        Returns:
            DataFrame with original columns + earnings_age_X for each age
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # Verify condition variables present
        missing = [
            c for c in self.config.condition_vars if c not in cross_section.columns
        ]
        if missing:
            raise ValueError(f"Missing condition columns: {missing}")

        # Generate trajectories CONDITIONED on the provided demographics
        # This is the key - we use synthesizer_.generate(conditions) not sample()
        conditions = cross_section[self.config.condition_vars].copy()
        synthetic = self.synthesizer_.generate(conditions, seed=seed)

        # Merge trajectories with cross-section
        result = cross_section.copy().reset_index(drop=True)
        trajectory_cols = [
            f"earnings_age_{age}" for age in self.config.trajectory_ages
        ]
        for col in trajectory_cols:
            result[col] = synthetic[col].values

        return result

    def interpolate_full_trajectory(
        self,
        wide_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Interpolate to annual trajectory from 5-year intervals.

        If model trained on 5-year intervals (ages 18, 23, 28, ...),
        this interpolates to get every year (18, 19, 20, ...).

        Args:
            wide_data: DataFrame with earnings_age_X columns at intervals

        Returns:
            DataFrame with earnings_age_X for every year
        """
        result = wide_data.copy()
        interval_ages = self.config.trajectory_ages

        for i in range(len(interval_ages) - 1):
            start_age = interval_ages[i]
            end_age = interval_ages[i + 1]
            start_col = f"earnings_age_{start_age}"
            end_col = f"earnings_age_{end_age}"

            # Linear interpolation between interval points
            for year in range(start_age + 1, end_age):
                t = (year - start_age) / (end_age - start_age)
                result[f"earnings_age_{year}"] = (
                    (1 - t) * result[start_col] + t * result[end_col]
                )

        return result


def panel_to_wide(
    panel_data: pd.DataFrame,
    id_col: str = "person_id",
    age_col: str = "age",
    earnings_col: str = "earnings",
    condition_vars: list[str] | None = None,
) -> pd.DataFrame:
    """Convert long-format panel data to wide format for trajectory modeling.

    Takes panel data with multiple rows per person (one per age observation)
    and converts to one row per person with earnings_age_X columns.

    Args:
        panel_data: Long-format panel DataFrame
        id_col: Column identifying individuals
        age_col: Column with age at each observation
        earnings_col: Column with earnings at each observation
        condition_vars: Time-invariant variables to include (e.g., education)

    Returns:
        Wide-format DataFrame with one row per person

    Example:
        Input (long format):
        | person_id | age | earnings | education |
        |-----------|-----|----------|-----------|
        | 1         | 25  | 30000    | 3         |
        | 1         | 30  | 40000    | 3         |
        | 2         | 25  | 25000    | 2         |

        Output (wide format):
        | person_id | education | earnings_age_25 | earnings_age_30 |
        |-----------|-----------|-----------------|-----------------|
        | 1         | 3         | 30000           | 40000           |
        | 2         | 2         | 25000           | NaN             |
    """
    if condition_vars is None:
        condition_vars = []

    # Pivot earnings to wide format
    earnings_wide = panel_data.pivot(
        index=id_col,
        columns=age_col,
        values=earnings_col,
    )
    earnings_wide.columns = [f"earnings_age_{int(age)}" for age in earnings_wide.columns]
    earnings_wide = earnings_wide.reset_index()

    # Get time-invariant characteristics (take first observation per person)
    if condition_vars:
        conditions = panel_data.groupby(id_col)[condition_vars].first().reset_index()
        result = earnings_wide.merge(conditions, on=id_col)
    else:
        result = earnings_wide

    return result


def create_psid_panel() -> pd.DataFrame:
    """Load and prepare PSID panel data for trajectory modeling.

    The PSID (Panel Study of Income Dynamics) is the gold standard
    for earnings trajectory modeling because it has 50+ years of
    longitudinal data on the same individuals.

    Access: https://psidonline.isr.umich.edu/
    Registration required but free for academic use.

    Returns:
        Long-format panel DataFrame
    """
    # PSID requires registration - provide instructions
    print("=" * 70)
    print("PSID DATA ACCESS")
    print("=" * 70)
    print("""
PSID is not included directly due to data use requirements.

To use PSID:
1. Register at https://psidonline.isr.umich.edu/
2. Download the Family and Individual files
3. Extract earnings variables (V-numbers vary by year)

Key variables for earnings trajectories:
- Individual ID: ER30001 (1968 ID) + ER30002 (person number)
- Earnings: Various by year (e.g., ER65349 for 2019 labor income)
- Demographics: age, sex, education, marital status

Alternative: Use SIPP for shorter 4-year panels (publicly available).
    """)

    return pd.DataFrame()


def demo_trajectory_synthesis():
    """Demonstrate the trajectory synthesis workflow."""
    print("=" * 70)
    print("TRAJECTORY SYNTHESIS DEMO")
    print("=" * 70)

    # Configuration - using 5-year age intervals for fewer dimensions
    config = TrajectoryConfig(
        start_age=25,
        end_age=65,
        age_interval=5,  # 25, 30, 35, ..., 65 = 9 dimensions
        condition_vars=["education", "gender", "birth_cohort"],
        n_layers=4,
        hidden_dim=64,
        epochs=50,
    )

    # Create mock panel data (simulating PSID)
    print("\n1. Creating mock panel data...")
    np.random.seed(42)
    n_persons = 500
    ages = config.trajectory_ages

    panel_data = []
    for person_id in range(n_persons):
        education = np.random.randint(1, 5)
        gender = np.random.randint(0, 2)
        birth_cohort = np.random.randint(1950, 1990)
        base_earnings = 20000 + education * 15000

        for age in ages:
            # Age-earnings profile: growth then decline
            if age < 50:
                growth_factor = 1.03 ** (age - 25)
            else:
                growth_factor = 1.03 ** 25 * 0.98 ** (age - 50)

            earnings = base_earnings * growth_factor * np.random.lognormal(0, 0.3)
            panel_data.append({
                "person_id": person_id,
                "age": age,
                "education": education,
                "gender": gender,
                "birth_cohort": birth_cohort,
                "earnings": max(0, earnings),
            })

    panel_df = pd.DataFrame(panel_data)
    print(f"  Created panel: {len(panel_df):,} observations")
    print(f"  {n_persons:,} persons × {len(ages)} ages")

    # Convert to wide format
    print("\n2. Converting to wide format...")
    wide_panel = panel_to_wide(
        panel_df,
        id_col="person_id",
        age_col="age",
        earnings_col="earnings",
        condition_vars=["education", "gender", "birth_cohort"],
    )
    print(f"  Wide format: {len(wide_panel):,} rows × {len(wide_panel.columns)} columns")

    # Fit trajectory model
    print("\n3. Fitting trajectory model...")
    model = TrajectoryModel(config)
    model.fit(wide_panel, verbose=True)

    # Generate trajectories for cross-section
    print("\n4. Generating trajectories for cross-section...")
    cross_section = wide_panel[["education", "gender", "birth_cohort"]].head(100)
    print(f"  Cross-section: {len(cross_section):,} individuals")

    trajectories = model.generate(cross_section, seed=42)
    print(f"  Generated {len(trajectories):,} trajectories")

    # Show sample trajectories
    print("\n5. Sample trajectories (first 3 people):")
    traj_cols = [f"earnings_age_{age}" for age in config.trajectory_ages]
    print(trajectories[["education"] + traj_cols[:5]].head(3).to_string())

    # Interpolate to annual
    print("\n6. Interpolating to annual trajectory...")
    full = model.interpolate_full_trajectory(trajectories)
    annual_cols = [c for c in full.columns if c.startswith("earnings_age_")]
    print(f"  Full trajectory: {len(annual_cols)} years (ages 25-65)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
TrajectoryModel successfully:
- Trained on {len(wide_panel):,} person-level records with {len(ages)}-point trajectories
- Generated {len(trajectories):,} synthetic trajectories
- Interpolated to annual earnings for ages 25-65

Next steps for production use:
1. Load PSID panel data (register at https://psidonline.isr.umich.edu/)
2. Train on full 50-year PSID histories
3. Apply to CPS cross-section (100k individuals)
4. Calibrate to SSA administrative targets
    """)


if __name__ == "__main__":
    demo_trajectory_synthesis()
