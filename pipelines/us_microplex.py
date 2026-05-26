"""US Microplex Pipeline.

Generates a 1:1 scale synthetic US population from survey data.

Currently uses CPS as the primary data source. Future versions will
incorporate multi-survey fusion (CPS + IRS PUF + SIPP) using masked loss.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex import Synthesizer


# Data paths
POLICYENGINE_DATA = Path("/Users/maxghenis/PolicyEngine/arch-data")
CPS_2024 = POLICYENGINE_DATA / "micro/us/cps_2024.parquet"


# Variable definitions
CONTEXT_VARS = [
    "age",
    "sex",
    "race",
    "marital_status",
    "state_fips",
    "employment_status",
]

TARGET_VARS = [
    "wage_salary_income",
    "self_employment_income",
    "interest_income",
    "dividend_income",
    "social_security_income",
    "unemployment_compensation",
]


def load_cps(path: Path = CPS_2024, sample_frac: float = 1.0) -> pd.DataFrame:
    """Load CPS data."""
    print(f"Loading CPS from {path}...")
    df = pd.read_parquet(path)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"  Sampled to {len(df):,} records")

    # Ensure all needed columns exist
    needed = CONTEXT_VARS + TARGET_VARS
    missing = [v for v in needed if v not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing}")
        needed = [v for v in needed if v in df.columns]

    # Handle weight column
    if "march_supplement_weight" in df.columns:
        df["weight"] = df["march_supplement_weight"]
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    print(f"  Loaded {len(df):,} records")
    print(f"  Total weighted population: {df['weight'].sum():,.0f}")

    return df


def train_synthesizer(
    data: pd.DataFrame,
    target_vars: List[str],
    context_vars: List[str],
    epochs: int = 100,
    verbose: bool = True,
) -> Synthesizer:
    """Train microplex synthesizer on survey data."""
    print(f"\nTraining synthesizer...")
    print(f"  Context vars: {context_vars}")
    print(f"  Target vars: {target_vars}")

    synth = Synthesizer(
        target_vars=target_vars,
        condition_vars=context_vars,
        n_layers=6,
        hidden_dim=128,
        zero_inflated=True,
        variance_regularization=0.1,
    )

    start = time.time()
    synth.fit(data, weight_col="weight", epochs=epochs, verbose=verbose)
    print(f"  Training time: {time.time() - start:.1f}s")

    return synth


def generate_microplex(
    synth: Synthesizer,
    n: int,
    batch_size: int = 100000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the Microplex - 1:1 scale synthetic population."""
    print(f"\nGenerating Microplex ({n:,} records)...")
    start = time.time()

    # Generate in batches for memory efficiency
    batches = []
    n_batches = (n + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch_n = min(batch_size, n - i * batch_size)
        batch_seed = seed + i if seed else None
        batch = synth.sample(batch_n, seed=batch_seed)
        batches.append(batch)

        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  Generated {(i + 1) * batch_size:,} / {n:,} records")

    result = pd.concat(batches, ignore_index=True)
    print(f"  Generation time: {time.time() - start:.1f}s")

    return result


def validate_microplex(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_vars: List[str],
) -> Dict:
    """Validate synthetic data against original."""
    print("\nValidating Microplex...")

    results = {}

    for var in target_vars:
        if var not in original.columns or var not in synthetic.columns:
            continue

        orig = original[var]
        syn = synthetic[var]

        # Weight original data for fair comparison
        if "weight" in original.columns:
            orig_mean = np.average(orig, weights=original["weight"])
            orig_std = np.sqrt(np.average((orig - orig_mean) ** 2, weights=original["weight"]))
        else:
            orig_mean = orig.mean()
            orig_std = orig.std()

        syn_mean = syn.mean()
        syn_std = syn.std()

        # Zero rate comparison
        orig_zero = (orig == 0).mean()
        syn_zero = (syn == 0).mean()

        results[var] = {
            "orig_mean": orig_mean,
            "syn_mean": syn_mean,
            "mean_ratio": syn_mean / (orig_mean + 1e-6),
            "orig_std": orig_std,
            "syn_std": syn_std,
            "std_ratio": syn_std / (orig_std + 1e-6),
            "orig_zero_rate": orig_zero,
            "syn_zero_rate": syn_zero,
        }

    # Print summary
    print(f"\n{'Variable':<25} {'Orig Mean':>12} {'Syn Mean':>12} {'Ratio':>8} {'Zero%':>8}")
    print("-" * 70)
    for var, stats in results.items():
        orig_z = f"{stats['orig_zero_rate']*100:.1f}%"
        syn_z = f"{stats['syn_zero_rate']*100:.1f}%"
        print(f"{var:<25} {stats['orig_mean']:>12,.0f} {stats['syn_mean']:>12,.0f} "
              f"{stats['mean_ratio']:>8.2f} {syn_z:>8}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate US Microplex")
    parser.add_argument("--sample-frac", type=float, default=0.1,
                        help="Fraction of CPS to use for training (default: 0.1)")
    parser.add_argument("--n-generate", type=int, default=100000,
                        help="Number of records to generate (default: 100000)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: don't save)")
    args = parser.parse_args()

    print("=" * 70)
    print("US MICROPLEX GENERATION")
    print("=" * 70)

    # Load data
    context_vars = [v for v in CONTEXT_VARS if True]  # Filter available
    target_vars = [v for v in TARGET_VARS if True]

    data = load_cps(sample_frac=args.sample_frac)

    # Filter to available columns
    available = list(data.columns)
    context_vars = [v for v in context_vars if v in available]
    target_vars = [v for v in target_vars if v in available]

    # Train
    synth = train_synthesizer(
        data,
        target_vars=target_vars,
        context_vars=context_vars,
        epochs=args.epochs,
    )

    # Generate
    microplex = generate_microplex(synth, n=args.n_generate)

    # Validate
    validate_microplex(data, microplex, target_vars)

    # Save
    if args.output:
        print(f"\nSaving to {args.output}...")
        microplex.to_parquet(args.output, index=False)

    print("\n" + "=" * 70)
    print("MICROPLEX GENERATION COMPLETE")
    print("=" * 70)

    return microplex


if __name__ == "__main__":
    main()
