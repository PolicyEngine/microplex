#!/usr/bin/env python3
"""
Synthesize microdata with block-level geographic assignments.

This script extends the CD-based synthesis to use Census blocks as the
primary geographic unit, then derives higher-level geographies:
- block_geoid (15-digit Census block GEOID)
- tract_geoid (11-digit tract GEOID)
- county_fips (5-digit state+county FIPS)
- cd_id (Congressional district, e.g., "CA-12", "TX-AL")
- state_fips (2-digit state FIPS)

Flow:
1. Load CPS data and block probability mappings
2. Synthesize households with state from CPS distribution
3. Assign blocks within state based on population-weighted probabilities
4. Derive all higher-level geographies from block assignments
5. Build calibration indicators for CDs
6. Run IPF calibration to match CD population targets
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from microplex.calibration import Calibrator
from microplex.geography import derive_geographies
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


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CPS data and block probability mappings.

    Returns:
        Tuple of (households, persons, block_probs, cd_targets)
    """
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

    # Load block probabilities
    block_probs = pd.read_parquet(data_dir / "block_probabilities.parquet")
    print(f"\nBlock probability mappings: {len(block_probs):,}")
    print(f"  States: {block_probs['state_fips'].nunique()}")
    print(f"  CDs: {block_probs['cd_id'].nunique()}")
    print(f"  Tracts: {block_probs['tract_geoid'].nunique()}")
    print(f"  Blocks: {block_probs['geoid'].nunique()}")
    print(f"  Total population: {block_probs['population'].sum():,}")

    # Load CD targets
    targets = pd.read_parquet(data_dir / "targets.parquet")
    cd_mask = targets["geography"].str.match(r"^[A-Z]{2}-\d{2}$|^[A-Z]{2}-AL$", na=False)
    cd_targets = targets[cd_mask].copy()
    print(f"\nCD targets: {len(cd_targets)}")

    return hh, persons, block_probs, cd_targets


def build_block_lookup(block_probs: pd.DataFrame) -> dict:
    """Build lookup dict for fast block assignment by state.

    Args:
        block_probs: DataFrame with columns: geoid, state_fips, prob, and geography fields

    Returns:
        Dict mapping state_fips (int) to dict with block info arrays
    """
    lookup = {}

    # Ensure state_fips is available as int
    block_probs = block_probs.copy()
    block_probs['state_fips_int'] = block_probs['state_fips'].astype(int)

    for state_fips in block_probs['state_fips_int'].unique():
        state_blocks = block_probs[block_probs['state_fips_int'] == state_fips]

        lookup[int(state_fips)] = {
            'block_geoid': state_blocks['geoid'].values,
            'tract_geoid': state_blocks['tract_geoid'].values,
            'county': state_blocks['county'].values,
            'cd_id': state_blocks['cd_id'].values,
            'probs': state_blocks['prob'].values,
        }

    return lookup


def assign_blocks(
    hh: pd.DataFrame,
    block_lookup: dict,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Assign Census blocks to households based on state.

    Uses pseudorandom assignment weighted by block population shares within state.

    Args:
        hh: Household DataFrame with state_fips column
        block_lookup: Dict from build_block_lookup()
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with block_geoid, tract_geoid, county_fips, cd_id columns added
    """
    hh = hh.copy()
    rng = np.random.default_rng(random_state)

    # Get valid state FIPS codes
    valid_fips = np.array(list(block_lookup.keys()))

    # Initialize arrays for assignments
    len(hh)
    block_geoids = []
    tract_geoids = []
    county_codes = []
    cd_ids = []
    fixed_fips = []

    for state_fips in hh['state_fips'].values:
        # Round to nearest valid state FIPS
        state_fips_rounded = int(round(state_fips))

        # If not valid, find nearest valid FIPS
        if state_fips_rounded not in block_lookup:
            diffs = np.abs(valid_fips - state_fips)
            state_fips_rounded = valid_fips[np.argmin(diffs)]

        lookup = block_lookup[state_fips_rounded]

        # Sample a block based on population probabilities
        idx = rng.choice(len(lookup['block_geoid']), p=lookup['probs'])

        block_geoids.append(lookup['block_geoid'][idx])
        tract_geoids.append(lookup['tract_geoid'][idx])
        county_codes.append(lookup['county'][idx])
        cd_ids.append(lookup['cd_id'][idx])
        fixed_fips.append(state_fips_rounded)

    # Assign to DataFrame
    hh['block_geoid'] = block_geoids
    hh['tract_geoid'] = tract_geoids
    hh['cd_id'] = cd_ids

    # Build county_fips (5-digit: state + county)
    hh['county_fips'] = [
        f"{fips:02d}{county}" for fips, county in zip(fixed_fips, county_codes)
    ]

    # Fix state_fips to be valid integer
    hh['state_fips'] = fixed_fips

    return hh


def synthesize_with_blocks(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
    block_probs: pd.DataFrame,
    n_households: int = 100_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize households with block-level geographic assignments.

    Pipeline:
    1. Synthesize minimal dataset (only block_geoid assigned during synthesis)
    2. Derive parent geographies post-hoc (tract, county, CD, SLD)
    3. Ready for calibration

    Args:
        hh: CPS household data
        persons: CPS person data
        block_probs: Block probability mappings
        n_households: Number of households to generate
        seed: Random seed

    Returns:
        Tuple of (synthetic_households, synthetic_persons)
    """
    print("\n" + "=" * 70)
    print("SYNTHESIZING WITH BLOCK-LEVEL ASSIGNMENTS (NEW PIPELINE)")
    print("=" * 70)

    # Create schema
    schema = HouseholdSchema(
        hh_vars=['n_persons', 'n_adults', 'n_children', 'state_fips', 'tenure'],
        person_vars=['age', 'sex', 'income', 'employment_status', 'education', 'relationship_to_head'],
    )

    # Initialize synthesizer WITH block_probabilities (will assign only block_geoid)
    synth = HierarchicalSynthesizer(
        schema=schema,
        random_state=seed,
        block_probabilities=block_probs,  # Pass block probs to synthesizer
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

    # Generate synthetic data (with block_geoid assigned during synthesis)
    print(f"\nGenerating {n_households:,} synthetic households...")
    synthetic_hh, synthetic_persons = synth.generate(
        n_households=n_households,
        verbose=True,
    )

    # STEP 2: Derive parent geographies post-hoc
    print("\nDeriving parent geographies from block assignments...")
    geos = derive_geographies(
        synthetic_hh['block_geoid'],
        include_cd=True,
        include_sld=True,
        block_data=block_probs,
    )

    # Merge derived geographies into synthetic_hh
    for col in ['tract_geoid', 'county_fips', 'cd_id']:
        if col in geos.columns:
            synthetic_hh[col] = geos[col].values

    # Optionally add SLD columns
    for col in ['sldu_id', 'sldl_id']:
        if col in geos.columns:
            synthetic_hh[col] = geos[col].values

    # Print assignment summary
    print("\nGeographic assignment summary:")
    print(f"  Unique blocks assigned: {synthetic_hh['block_geoid'].nunique():,}")
    print(f"  Unique tracts assigned: {synthetic_hh['tract_geoid'].nunique():,}")
    print(f"  Unique counties assigned: {synthetic_hh['county_fips'].nunique()}")
    print(f"  Unique CDs assigned: {synthetic_hh['cd_id'].nunique()}")
    if 'sldu_id' in synthetic_hh.columns:
        print(f"  Unique SLDUs assigned: {synthetic_hh['sldu_id'].nunique()}")
    if 'sldl_id' in synthetic_hh.columns:
        print(f"  Unique SLDLs assigned: {synthetic_hh['sldl_id'].nunique()}")
    print(f"  Unique states: {synthetic_hh['state_fips'].nunique()}")

    # CD distribution
    cd_counts = synthetic_hh['cd_id'].value_counts()
    print(f"\n  Households per CD: {cd_counts.mean():.1f} (mean)")
    print(f"  Range: {cd_counts.min()}-{cd_counts.max()}")

    return synthetic_hh, synthetic_persons


def build_cd_indicators(hh: pd.DataFrame) -> pd.DataFrame:
    """Build CD indicator columns for calibration.

    Creates n_persons_{cd_id} columns for each CD.
    """
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
) -> pd.DataFrame:
    """Calibrate synthetic data to CD population targets.

    Args:
        hh: Synthetic household DataFrame with cd_id column
        cd_targets: DataFrame with CD population targets

    Returns:
        Calibrated household DataFrame with weights
    """
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


def print_geography_summary(hh: pd.DataFrame) -> None:
    """Print summary statistics about geographic coverage."""
    print("\n" + "=" * 70)
    print("GEOGRAPHIC COVERAGE SUMMARY")
    print("=" * 70)

    print(f"\nBlocks:   {hh['block_geoid'].nunique():>8,}")
    print(f"Tracts:   {hh['tract_geoid'].nunique():>8,}")
    print(f"Counties: {hh['county_fips'].nunique():>8}")
    print(f"CDs:      {hh['cd_id'].nunique():>8}")
    print(f"States:   {hh['state_fips'].nunique():>8}")

    # Show distribution by state
    print("\n--- Households by State (top 10) ---")
    state_counts = hh['state_fips'].value_counts().head(10)
    for state_fips, count in state_counts.items():
        state_abbr = FIPS_TO_STATE.get(int(state_fips), f"FIPS-{state_fips}")
        print(f"  {state_abbr}: {count:,}")

    # Show CD distribution
    print("\n--- Households by CD (top 10) ---")
    cd_counts = hh['cd_id'].value_counts().head(10)
    for cd_id, count in cd_counts.items():
        print(f"  {cd_id}: {count:,}")

    # Show weighted population by state
    if 'weight' in hh.columns:
        print("\n--- Weighted Population by State (top 10) ---")
        hh_with_wpop = hh.copy()
        hh_with_wpop['weighted_persons'] = hh_with_wpop['weight'] * hh_with_wpop['n_persons']
        state_pop = hh_with_wpop.groupby('state_fips')['weighted_persons'].sum().sort_values(ascending=False).head(10)
        for state_fips, wpop in state_pop.items():
            state_abbr = FIPS_TO_STATE.get(int(state_fips), f"FIPS-{state_fips}")
            print(f"  {state_abbr}: {wpop:,.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize microdata with block-level geographic assignments"
    )
    parser.add_argument("--n-households", type=int, default=100_000,
                        help="Number of households to generate (default: 100,000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"

    # Load data
    hh, persons, block_probs, cd_targets = load_data(data_dir)

    # Synthesize with block assignments
    synthetic_hh, synthetic_persons = synthesize_with_blocks(
        hh, persons, block_probs,
        n_households=args.n_households,
        seed=args.seed,
    )

    # Calibrate to CD targets
    calibrated_hh = calibrate_to_cd_targets(synthetic_hh, cd_targets)

    # Print geographic summary
    print_geography_summary(calibrated_hh)

    # Save
    output_path = data_dir / "microplex_synthetic_with_blocks.parquet"

    # Select output columns (exclude indicator columns for cleaner output)
    output_cols = [
        'household_id', 'n_persons', 'n_adults', 'n_children',
        'state_fips', 'tenure', 'weight',
        'block_geoid', 'tract_geoid', 'county_fips', 'cd_id',
        'sldu_id', 'sldl_id',  # State legislative districts
        'hh_income',
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in calibrated_hh.columns]

    calibrated_hh[output_cols].to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Also save persons with household geography
    persons_output_path = data_dir / "microplex_synthetic_persons_with_blocks.parquet"

    # Merge geography to persons (only new columns not already in persons)
    geo_cols = ['household_id', 'block_geoid', 'tract_geoid', 'county_fips', 'cd_id',
                'sldu_id', 'sldl_id', 'state_fips', 'weight']
    geo_cols = [c for c in geo_cols if c in calibrated_hh.columns]

    # Drop any overlapping columns from persons before merge (except household_id)
    persons_cols_to_drop = [c for c in geo_cols if c in synthetic_persons.columns and c != 'household_id']
    synthetic_persons_clean = synthetic_persons.drop(columns=persons_cols_to_drop, errors='ignore')

    synthetic_persons_with_geo = synthetic_persons_clean.merge(
        calibrated_hh[geo_cols],
        on='household_id',
        how='left'
    )
    synthetic_persons_with_geo.to_parquet(persons_output_path, index=False)
    print(f"Saved persons to {persons_output_path}")

    print("\n" + "=" * 70)
    print("BLOCK-LEVEL SYNTHESIS COMPLETE")
    print("=" * 70)
    print(f"Synthetic households: {len(calibrated_hh):,}")
    print(f"Synthetic persons: {len(synthetic_persons_with_geo):,}")
    print(f"Total weighted persons: {(calibrated_hh['weight'] * calibrated_hh['n_persons']).sum():,.0f}")


if __name__ == "__main__":
    main()
