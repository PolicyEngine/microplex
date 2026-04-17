#!/usr/bin/env python3
"""
Run calibration with synthetic expansion using normalizing flows.

1. Load CPS enhanced data with detailed income variables
2. Train Synthesizer to learn P(income_vars | demographics)
3. Generate synthetic records (expand the dataset)
4. Run L0 calibration on the combined data

This gives the calibrator more "degrees of freedom" to match targets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time

import numpy as np
import pandas as pd

from microplex import Synthesizer

# Income/benefit variables to synthesize
# Note: snap has too many NaNs (58%), so we exclude it for now
TARGET_VARS = [
    "employment_income",
    "self_employment_income",
    "dividend_income",
    "interest_income",
    "rental_income",
    "social_security",
    "ssi",
    "unemployment_compensation",
    "pension_income",
]

# Demographics to condition on
CONTEXT_VARS = [
    "age",
    "is_male",
    "state_fips",
    "is_disabled",
]


def load_cps_enhanced(data_dir: Path) -> pd.DataFrame:
    """Load CPS enhanced data with detailed income variables."""
    path = data_dir / "cps_enhanced_persons.parquet"
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # Filter to complete cases for key variables
    required_cols = TARGET_VARS + CONTEXT_VARS
    available = [c for c in required_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=available)
    after = len(df)
    if before != after:
        print(f"  Filtered to complete cases: {before:,} -> {after:,}")

    # Fill remaining NaNs with 0 for income variables
    for col in TARGET_VARS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def train_synthesizer(
    df: pd.DataFrame,
    target_vars: list,
    context_vars: list,
    epochs: int = 100,
) -> Synthesizer:
    """Train synthesizer on CPS data."""
    # Filter to available columns
    available_targets = [v for v in target_vars if v in df.columns]
    available_context = [v for v in context_vars if v in df.columns]

    print("\nTraining Synthesizer...")
    print(f"  Context vars: {available_context}")
    print(f"  Target vars: {available_targets}")

    synth = Synthesizer(
        target_vars=available_targets,
        condition_vars=available_context,
        n_layers=6,
        hidden_dim=128,
        zero_inflated=True,
        variance_regularization=0.1,
    )

    # Use person_weight if available
    weight_col = "person_weight" if "person_weight" in df.columns else None

    start = time.time()
    synth.fit(df, weight_col=weight_col, epochs=epochs, verbose=True)
    print(f"  Training time: {time.time() - start:.1f}s")

    return synth


def generate_synthetic(
    synth: Synthesizer,
    original_df: pd.DataFrame,
    n_synthetic: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic records."""
    print(f"\nGenerating {n_synthetic:,} synthetic records...")
    start = time.time()

    # Sample conditions from original data (bootstrap)
    np.random.seed(seed)
    sampled_idx = np.random.choice(len(original_df), size=n_synthetic, replace=True)
    conditions = original_df[synth.condition_vars].iloc[sampled_idx].reset_index(drop=True)

    # Generate targets
    synthetic = synth.generate(conditions, seed=seed)

    # Copy over non-target columns from sampled records
    for col in original_df.columns:
        if col not in synthetic.columns and col not in synth.target_vars:
            synthetic[col] = original_df[col].iloc[sampled_idx].values

    # Reset IDs (weights will be set in combine step for proper scaling)
    synthetic["person_id"] = range(len(original_df), len(original_df) + n_synthetic)

    print(f"  Generated {len(synthetic):,} records in {time.time() - start:.1f}s")

    return synthetic


def validate_synthetic(original: pd.DataFrame, synthetic: pd.DataFrame, target_vars: list):
    """Validate synthetic data matches original distributions."""
    print("\nValidation: Synthetic vs Original")
    print("-" * 60)

    weight_col = "person_weight" if "person_weight" in original.columns else None

    for var in target_vars:
        if var not in original.columns or var not in synthetic.columns:
            continue

        orig = original[var]
        syn = synthetic[var]

        if weight_col:
            orig_mean = np.average(orig, weights=original[weight_col])
        else:
            orig_mean = orig.mean()

        syn_mean = syn.mean()
        ratio = syn_mean / (orig_mean + 1e-6)

        (orig == 0).mean() * 100
        syn_zero = (syn == 0).mean() * 100

        print(f"{var:30} orig={orig_mean:12,.0f} syn={syn_mean:12,.0f} ratio={ratio:.2f} zero={syn_zero:.0f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-synthetic", type=int, default=200000,
                        help="Number of synthetic records to generate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs for synthesizer")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"

    print("=" * 70)
    print("EXPANDED CALIBRATION WITH SYNTHETIC DATA")
    print("=" * 70)

    # Step 1: Load original CPS data
    df = load_cps_enhanced(data_dir)

    # Step 2: Train synthesizer
    synth = train_synthesizer(
        df,
        target_vars=TARGET_VARS,
        context_vars=CONTEXT_VARS,
        epochs=args.epochs,
    )

    # Step 3: Generate synthetic records
    synthetic = generate_synthetic(synth, df, n_synthetic=args.n_synthetic)

    # Step 4: Validate
    validate_synthetic(df, synthetic, TARGET_VARS)

    # Step 5: Combine original + synthetic
    print("\nCombining datasets...")
    df["is_synthetic"] = False
    synthetic["is_synthetic"] = True

    # Set weight column for original data (needed for calibration)
    if "weight" not in df.columns:
        df["weight"] = df["person_weight"]

    # For synthetic records, set initial weight proportional to original
    # Each synthetic record represents the same fraction of population as
    # an average original record. This gives calibration a reasonable starting point.
    original_mean_weight = df["weight"].mean()
    synthetic["weight"] = original_mean_weight  # Start at same scale as original

    combined = pd.concat([df, synthetic], ignore_index=True)
    print(f"  Original: {len(df):,}")
    print(f"  Synthetic: {len(synthetic):,}")
    print(f"  Combined: {len(combined):,}")
    print(f"  Original weight sum: {df['weight'].sum():,.0f}")
    print(f"  Synthetic weight sum: {synthetic['weight'].sum():,.0f}")
    print(f"  Mean weight: {original_mean_weight:,.1f}")

    # Step 6: Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_dir / "cps_expanded_for_calibration.parquet"

    print(f"\nSaving to {output_path}...")
    combined.to_parquet(output_path, index=False)
    print(f"  Saved {len(combined):,} records")

    print("\n" + "=" * 70)
    print("READY FOR CALIBRATION")
    print("=" * 70)
    print(f"Run: python scripts/run_supabase_calibration.py --input {output_path}")

    return combined


if __name__ == "__main__":
    main()
