#!/usr/bin/env python3
"""
Synthesize microdata with congressional district assignments.

The proper flow per design docs:
1. Synthesize many records (e.g., 100k+ households)
2. Assign CDs pseudorandomly during synthesis based on state CD population shares
3. Calibrate weights to match CD and other targets

This gives each CD proper representation in the synthetic population.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from microplex.calibration import Calibrator
from microplex.hierarchical import HierarchicalSynthesizer, HouseholdSchema

# State FIPS to abbreviation
FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def load_data(data_dir: Path):
    """Load CPS data and CD probability mappings."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load CPS
    hh = pd.read_parquet(data_dir / "cps_asec_households.parquet")
    persons = pd.read_parquet(data_dir / "cps_asec_persons.parquet")

    # Filter territories
    territory_fips = {3, 7, 14, 43, 52}
    hh = hh[~hh['state_fips'].isin(territory_fips)].copy()
    persons = persons[persons['household_id'].isin(hh['household_id'])].copy()

    print(f"CPS households: {len(hh):,}")
    print(f"CPS persons: {len(persons):,}")

    # Load CD probabilities
    cd_probs = pd.read_parquet(data_dir / "state_cd_probabilities.parquet")
    print(f"CD probability mappings: {len(cd_probs)}")
    print(f"  States: {cd_probs['state_fips'].nunique()}")
    print(f"  CDs: {cd_probs['cd_id'].nunique()}")

    # Load CD targets
    targets = pd.read_parquet(data_dir / "targets.parquet")
    cd_mask = targets["geography"].str.match(r"^[A-Z]{2}-\d{2}$|^[A-Z]{2}-AL$", na=False)
    cd_targets = targets[cd_mask].copy()
    print(f"CD targets: {len(cd_targets)}")

    return hh, persons, cd_probs, cd_targets


def synthesize_with_cds(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
    cd_probs: pd.DataFrame,
    n_households: int = 100_000,
    seed: int = 42,
):
    """Synthesize households with CD assignments."""
    print("\n" + "=" * 70)
    print("SYNTHESIZING WITH CD ASSIGNMENTS")
    print("=" * 70)

    # Create schema
    schema = HouseholdSchema(
        hh_vars=['n_persons', 'n_adults', 'n_children', 'state_fips', 'tenure'],
        person_vars=['age', 'sex', 'income', 'employment_status', 'education', 'relationship_to_head'],
    )

    # Initialize synthesizer with CD probabilities
    synth = HierarchicalSynthesizer(
        schema=schema,
        random_state=seed,
        cd_probabilities=cd_probs,  # This enables CD assignment during generation
    )

    # Fit on CPS data
    print("\nFitting on CPS data...")
    synth.fit(
        hh_data=hh,
        person_data=persons,
        hh_weight_col='hh_weight',
        epochs=50,
        verbose=True,
    )

    # Generate synthetic data with CD assignments
    print(f"\nGenerating {n_households:,} synthetic households with CD assignments...")
    synthetic_hh, synthetic_persons = synth.generate(
        n_households=n_households,
        verbose=True,
    )

    # Verify CD assignments
    cd_counts = synthetic_hh['cd_id'].value_counts()
    print("\nCD assignment summary:")
    print(f"  Unique CDs assigned: {len(cd_counts)}")
    print(f"  Households per CD: {cd_counts.mean():.1f} (mean)")
    print(f"  Range: {cd_counts.min()}-{cd_counts.max()}")

    return synthetic_hh, synthetic_persons


def build_cd_indicators(hh: pd.DataFrame) -> pd.DataFrame:
    """Build CD indicator columns for calibration."""
    # Get all unique CDs
    cd_ids = hh['cd_id'].dropna().unique()

    # Create indicator matrix more efficiently
    indicators = {}
    for cd_id in cd_ids:
        col_name = f"n_persons_{cd_id}"
        indicators[col_name] = np.where(hh['cd_id'] == cd_id, hh['n_persons'], 0)

    # Add all at once
    indicator_df = pd.DataFrame(indicators, index=hh.index)
    return pd.concat([hh, indicator_df], axis=1)


def calibrate_to_cd_targets(
    hh: pd.DataFrame,
    cd_targets: pd.DataFrame,
):
    """Calibrate synthetic data to CD population targets."""
    print("\n" + "=" * 70)
    print("CALIBRATING TO CD TARGETS")
    print("=" * 70)

    # Build CD indicators
    print("Building CD indicators...")
    hh = build_cd_indicators(hh)

    # Get CD target dict
    cd_target_dict = dict(zip(cd_targets['geography'], cd_targets['value']))

    # Filter to CDs in our data
    valid_cds = set(hh['cd_id'].dropna().unique())
    cd_target_dict = {k: v for k, v in cd_target_dict.items() if k in valid_cds and v > 0}
    print(f"Valid CD targets: {len(cd_target_dict)}")

    # Build continuous targets
    continuous_targets = {}
    for cd_id, target_pop in cd_target_dict.items():
        col_name = f"n_persons_{cd_id}"
        if col_name in hh.columns:
            continuous_targets[col_name] = target_pop

    print(f"Continuous targets: {len(continuous_targets)}")

    # Initialize weights
    hh['weight'] = 1.0

    # Run calibration
    print("\nRunning IPF calibration...")
    calibrator = Calibrator(method="ipf", max_iter=500, tol=1e-6)

    calibrated_hh = calibrator.fit_transform(
        hh,
        marginal_targets={},
        continuous_targets=continuous_targets,
        weight_col="weight"
    )

    print("\nCalibration results:")
    print(f"  Converged: {calibrator.converged_}")
    print(f"  Iterations: {calibrator.n_iterations_}")

    # Evaluate
    total_persons = (calibrated_hh['weight'] * calibrated_hh['n_persons']).sum()
    print(f"  Total weighted persons: {total_persons:,.0f}")

    # Calculate errors
    errors = []
    for cd_id, target_pop in cd_target_dict.items():
        col_name = f"n_persons_{cd_id}"
        if col_name in calibrated_hh.columns:
            calibrated_pop = (calibrated_hh['weight'] * calibrated_hh[col_name]).sum()
            error = abs(calibrated_pop - target_pop) / target_pop if target_pop > 0 else 0
            errors.append(error)

    print("\nCD target errors:")
    print(f"  Mean: {np.mean(errors)*100:.2f}%")
    print(f"  Max: {np.max(errors)*100:.2f}%")
    print(f"  Median: {np.median(errors)*100:.2f}%")

    return calibrated_hh


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-households", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"

    # Load data
    hh, persons, cd_probs, cd_targets = load_data(data_dir)

    # Synthesize with CD assignments
    synthetic_hh, synthetic_persons = synthesize_with_cds(
        hh, persons, cd_probs,
        n_households=args.n_households,
        seed=args.seed,
    )

    # Calibrate to CD targets
    calibrated_hh = calibrate_to_cd_targets(synthetic_hh, cd_targets)

    # Save
    output_path = data_dir / "microplex_synthetic_with_cds.parquet"
    calibrated_hh.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n" + "=" * 70)
    print("SYNTHESIS WITH CDs COMPLETE")
    print("=" * 70)
    print(f"Synthetic households: {len(calibrated_hh):,}")
    print(f"Total weighted persons: {(calibrated_hh['weight'] * calibrated_hh['n_persons']).sum():,.0f}")


if __name__ == "__main__":
    main()
