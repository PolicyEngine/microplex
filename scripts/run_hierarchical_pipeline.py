#!/usr/bin/env python3
"""
Full Hierarchical Calibration Pipeline

Runs the complete microplex pipeline on real CPS data:
1. Load CPS households and persons
2. Aggregate person-level features to household level
3. Build state-level and demographic targets from Census/IRS data
4. Run IPF calibration to match real US population (~331M)
5. Propagate weights to persons
6. Validate against targets
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from microplex.calibration import Calibrator

# State FIPS to abbreviation mapping
FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY", 3: "AS", 7: "CZ", 14: "GU", 43: "PR", 52: "VI",  # territories
}
STATE_TO_FIPS = {v: k for k, v in FIPS_TO_STATE.items()}


def load_cps_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPS household and person data."""
    print("=" * 70)
    print("LOADING CPS DATA")
    print("=" * 70)

    hh = pd.read_parquet(data_dir / "cps_asec_households.parquet")
    persons = pd.read_parquet(data_dir / "cps_asec_persons.parquet")

    print(f"Raw households: {len(hh):,}")
    print(f"Raw persons: {len(persons):,}")

    # Filter out territories (keep only 50 states + DC)
    # FIPS codes 1-56 excluding 3, 7, 14, 43, 52 (territories)
    territory_fips = {3, 7, 14, 43, 52}
    state_mask = ~hh['state_fips'].isin(territory_fips)
    hh = hh[state_mask].copy()
    persons = persons[persons['household_id'].isin(hh['household_id'])].copy()

    print("After filtering territories:")
    print(f"  Households: {len(hh):,}")
    print(f"  Persons: {len(persons):,}")
    print(f"  Avg HH size: {len(persons) / len(hh):.2f}")
    print(f"  Total weighted HH: {hh['hh_weight'].sum():,.0f}")

    # Show state distribution
    print("\nState distribution (top 10):")
    state_counts = hh.groupby('state_fips')['hh_weight'].sum().sort_values(ascending=False)
    for state, count in state_counts.head(10).items():
        FIPS_TO_STATE.get(count, str(count))
        print(f"  {FIPS_TO_STATE.get(state, state)}: {count:,.0f}")

    return hh, persons


def aggregate_person_features(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate person-level features to household level for calibration."""
    print("\n" + "=" * 70)
    print("AGGREGATING PERSON FEATURES TO HOUSEHOLD LEVEL")
    print("=" * 70)

    hh = hh.copy()

    # Age groups
    age_groups = [
        ("n_age_0_17", (0, 18)),
        ("n_age_18_64", (18, 65)),
        ("n_age_65_plus", (65, 200)),
    ]

    for col_name, (age_min, age_max) in age_groups:
        mask = (persons["age"] >= age_min) & (persons["age"] < age_max)
        counts = persons[mask].groupby("household_id").size()
        hh[col_name] = hh["household_id"].map(counts).fillna(0).astype(int)
        print(f"  {col_name}: {hh[col_name].sum():,} persons across {(hh[col_name] > 0).sum():,} HHs")

    # Employment status
    employed = persons[persons["employment_status"] == 1]
    employed_per_hh = employed.groupby("household_id").size()
    hh["n_employed"] = hh["household_id"].map(employed_per_hh).fillna(0).astype(int)
    print(f"  n_employed: {hh['n_employed'].sum():,} persons across {(hh['n_employed'] > 0).sum():,} HHs")

    # Total income per household
    income_per_hh = persons.groupby("household_id")["income"].sum()
    hh["hh_income"] = hh["household_id"].map(income_per_hh).fillna(0)
    print(f"  hh_income: ${hh['hh_income'].sum():,.0f} total")

    return hh


def load_official_targets(data_dir: Path) -> pd.DataFrame:
    """Load official calibration targets from Census, IRS, etc."""
    targets_path = data_dir / "targets.parquet"
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")
    return pd.read_parquet(targets_path)


def build_state_person_indicators(hh: pd.DataFrame) -> pd.DataFrame:
    """Create household-level indicators for state person counts.

    For each state, creates column n_persons_state_XX which is:
    - n_persons if household is in state XX
    - 0 otherwise

    This allows calibrating to person-level state populations using
    household-level weights: sum(hh_weight × n_persons_in_state) = state_pop
    """
    hh = hh.copy()

    # Get total persons per household (sum of age group counts)
    if 'n_persons' not in hh.columns:
        hh['n_persons'] = hh['n_age_0_17'] + hh['n_age_18_64'] + hh['n_age_65_plus']

    for fips in hh['state_fips'].unique():
        state_abbrev = FIPS_TO_STATE.get(int(fips), str(fips))
        col_name = f"n_persons_state_{state_abbrev}"
        hh[col_name] = np.where(hh['state_fips'] == fips, hh['n_persons'], 0)

    return hh


def build_targets(hh: pd.DataFrame, official_targets: pd.DataFrame) -> dict:
    """Build marginal calibration targets from official sources.

    Note: State-level targets are person populations, so we use continuous
    targets with person-count indicators rather than marginal targets.
    Returns empty dict since all state targets are handled as continuous.
    """
    print("\n" + "=" * 70)
    print("BUILDING CALIBRATION TARGETS FROM OFFICIAL SOURCES")
    print("=" * 70)
    print("(State population targets handled as continuous targets)")

    # No marginal targets - all handled as continuous
    return {}


def build_continuous_targets(hh: pd.DataFrame, official_targets: pd.DataFrame) -> dict:
    """Build continuous calibration targets from official sources.

    Uses Census age distributions and state populations.
    State populations are person-level targets using n_persons_state_XX indicators.
    """
    print("\n" + "=" * 70)
    print("BUILDING CONTINUOUS TARGETS FROM OFFICIAL SOURCES")
    print("=" * 70)

    targets = {}

    # Parse state populations from official targets
    state_pop = {}
    age_0_17 = 0
    age_18_64 = 0
    age_65_plus = 0

    for _, row in official_targets.iterrows():
        name = row['name']
        value = row['value']

        if not name.startswith('pop_') or '_age_' not in name:
            continue

        # Parse state and age from name like "pop_CA_age_0_5"
        parts = name.split('_age_')
        if len(parts) != 2:
            continue

        state_abbrev = name.split('_')[1]
        age_part = parts[1]

        # Accumulate state population
        if state_abbrev not in state_pop:
            state_pop[state_abbrev] = 0
        state_pop[state_abbrev] += value

        # Handle different age formats
        if age_part == '85_plus':
            age_min, age_max = 85, 200
        elif '_' in age_part:
            age_min, age_max = map(int, age_part.split('_'))
        else:
            continue

        # Aggregate into broad age groups
        if age_max <= 18:
            age_0_17 += value
        elif age_min >= 65:
            age_65_plus += value
        elif age_min >= 18 and age_max <= 65:
            age_18_64 += value
        elif age_min < 18 < age_max <= 65:
            pct_child = (18 - age_min) / (age_max - age_min)
            age_0_17 += value * pct_child
            age_18_64 += value * (1 - pct_child)
        elif age_min >= 18 and age_min < 65 < age_max:
            pct_working = (65 - age_min) / (age_max - age_min)
            age_18_64 += value * pct_working
            age_65_plus += value * (1 - pct_working)

    # Add age group targets
    targets["n_age_0_17"] = age_0_17
    targets["n_age_18_64"] = age_18_64
    targets["n_age_65_plus"] = age_65_plus

    print("Age targets:")
    print(f"  n_age_0_17: {targets['n_age_0_17']:,.0f}")
    print(f"  n_age_18_64: {targets['n_age_18_64']:,.0f}")
    print(f"  n_age_65_plus: {targets['n_age_65_plus']:,.0f}")

    # Add state population targets (using n_persons_state_XX indicators)
    print(f"\nState population targets: {len(state_pop)} states")
    total_pop = 0
    for state_abbrev, pop in sorted(state_pop.items(), key=lambda x: x[1], reverse=True):
        if state_abbrev in STATE_TO_FIPS:
            STATE_TO_FIPS[state_abbrev]
            # Check if this state exists in our data
            col_name = f"n_persons_state_{state_abbrev}"
            if col_name in hh.columns:
                targets[col_name] = pop
                total_pop += pop

    print(f"  Total population: {total_pop:,.0f}")

    # Show top 5 states
    top_states = sorted(
        [(k, v) for k, v in targets.items() if k.startswith('n_persons_state_')],
        key=lambda x: x[1], reverse=True
    )[:5]
    for col, pop in top_states:
        state = col.replace('n_persons_state_', '')
        print(f"    {state}: {pop:,.0f}")

    return targets


def run_calibration(
    hh: pd.DataFrame,
    marginal_targets: dict,
    continuous_targets: dict,
) -> pd.DataFrame:
    """Run IPF calibration on household data."""
    print("\n" + "=" * 70)
    print("RUNNING IPF CALIBRATION")
    print("=" * 70)

    # Initialize with original weights
    hh = hh.copy()
    hh["weight"] = hh["hh_weight"]

    calibrator = Calibrator(
        method="ipf",
        max_iter=100,
        tol=1e-6,
    )

    print(f"Marginal constraints: {len(marginal_targets)} dimensions")
    for dim, targets in marginal_targets.items():
        print(f"  {dim}: {len(targets)} categories")

    print(f"Continuous constraints: {len(continuous_targets)}")

    calibrator.fit(
        hh,
        marginal_targets=marginal_targets,
        continuous_targets=continuous_targets,
        weight_col="weight",
    )

    hh["calibrated_weight"] = calibrator.weights_

    print("\nCalibration complete!")
    print(f"  Converged: {calibrator.converged_}")
    print(f"  Iterations: {calibrator.n_iterations_}")

    # Weight statistics
    weight_ratio = hh["calibrated_weight"] / hh["weight"]
    print("\nWeight adjustment statistics:")
    print(f"  Min ratio: {weight_ratio.min():.4f}")
    print(f"  Max ratio: {weight_ratio.max():.4f}")
    print(f"  Mean ratio: {weight_ratio.mean():.4f}")
    print(f"  Std ratio: {weight_ratio.std():.4f}")

    return hh


def propagate_weights_to_persons(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """Propagate household weights to all persons."""
    print("\n" + "=" * 70)
    print("PROPAGATING WEIGHTS TO PERSONS")
    print("=" * 70)

    persons = persons.copy()
    weight_map = hh.set_index("household_id")["calibrated_weight"]
    persons["weight"] = persons["household_id"].map(weight_map)

    print(f"Total weighted persons: {persons['weight'].sum():,.0f}")

    return persons


def validate_calibration(
    hh: pd.DataFrame,
    marginal_targets: dict,
    continuous_targets: dict,
) -> dict:
    """Validate calibrated weights against targets."""
    print("\n" + "=" * 70)
    print("VALIDATING CALIBRATION")
    print("=" * 70)

    results = {}

    # Check marginal targets
    print("\nMarginal target validation:")
    for dim, targets in marginal_targets.items():
        dim_results = {}
        actual = hh.groupby(dim)["calibrated_weight"].sum()

        total_error = 0
        for cat, target in targets.items():
            actual_val = actual.get(cat, 0)
            error = abs(actual_val - target) / target if target > 0 else 0
            dim_results[cat] = {
                "target": target,
                "actual": actual_val,
                "error_pct": error * 100,
            }
            total_error += error

        avg_error = total_error / len(targets) * 100
        print(f"  {dim}: avg error = {avg_error:.4f}%")
        results[dim] = dim_results

    # Check continuous targets
    print("\nContinuous target validation:")
    for var, target in continuous_targets.items():
        actual = (hh[var] * hh["calibrated_weight"]).sum()
        error = abs(actual - target) / target * 100 if target > 0 else 0
        print(f"  {var}: target={target:,.0f}, actual={actual:,.0f}, error={error:.4f}%")
        results[var] = {"target": target, "actual": actual, "error_pct": error}

    return results


def save_outputs(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
    output_dir: Path,
):
    """Save calibrated outputs."""
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    hh_path = output_dir / "microplex_hh.parquet"
    persons_path = output_dir / "microplex_persons.parquet"

    hh.to_parquet(hh_path, index=False)
    persons.to_parquet(persons_path, index=False)

    print(f"  Saved {len(hh):,} households to {hh_path}")
    print(f"  Saved {len(persons):,} persons to {persons_path}")


def main():
    """Run full hierarchical calibration pipeline."""
    print("=" * 70)
    print("MICROPLEX HIERARCHICAL CALIBRATION PIPELINE")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"

    # Step 1: Load data
    hh, persons = load_cps_data(data_dir)

    # Step 2: Aggregate person features to household level
    hh = aggregate_person_features(hh, persons)

    # Step 3: Load official targets (Census, IRS, etc.)
    print("\n" + "=" * 70)
    print("LOADING OFFICIAL TARGETS")
    print("=" * 70)
    official_targets = load_official_targets(data_dir)
    print(f"Loaded {len(official_targets):,} targets from official sources")

    # Step 4: Build state person indicators for calibration
    print("\n" + "=" * 70)
    print("BUILDING STATE PERSON INDICATORS")
    print("=" * 70)
    hh = build_state_person_indicators(hh)
    state_cols = [c for c in hh.columns if c.startswith('n_persons_state_')]
    print(f"Created {len(state_cols)} state person indicators")

    # Step 5: Build targets from official sources
    marginal_targets = build_targets(hh, official_targets)
    continuous_targets = build_continuous_targets(hh, official_targets)

    # Step 6: Run calibration
    hh = run_calibration(hh, marginal_targets, continuous_targets)

    # Step 7: Propagate weights to persons
    persons = propagate_weights_to_persons(hh, persons)

    # Step 8: Validate
    validation = validate_calibration(hh, marginal_targets, continuous_targets)

    # Step 9: Save outputs
    save_outputs(hh, persons, data_dir)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nFinal statistics:")
    print(f"  Households: {len(hh):,}")
    print(f"  Persons: {len(persons):,}")
    print(f"  Total weighted HHs: {hh['calibrated_weight'].sum():,.0f}")
    print(f"  Total weighted persons: {persons['weight'].sum():,.0f}")

    return hh, persons, validation


if __name__ == "__main__":
    hh, persons, validation = main()
