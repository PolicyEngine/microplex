"""
Build US Microplex: Synthetic population calibrated to geographic targets.

Pipeline:
1. Load CPS ASEC microdata (seed population)
2. Synthesize additional records using normalizing flows
3. Calibrate weights to match:
   - State population totals
   - Age distribution
   - Income totals
   - (Optional) County/district targets

Usage:
    python scripts/build_us_microplex.py --n-synthetic 100000 --device mps
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# State FIPS codes
STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def load_cps_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPS ASEC microdata."""
    persons = pd.read_parquet(data_dir / "cps_asec_persons.parquet")
    households = pd.read_parquet(data_dir / "cps_asec_households.parquet")
    return persons, households


def merge_person_household(
    persons: pd.DataFrame, households: pd.DataFrame
) -> pd.DataFrame:
    """Merge person and household data."""
    # Get household-level info for each person
    hh_cols = ["household_id", "state_fips", "hh_weight", "tenure"]
    merged = persons.merge(households[hh_cols], on="household_id", how="left")

    # Create state name from FIPS
    merged["state"] = merged["state_fips"].map(STATE_FIPS)

    # Create age groups
    merged["age_group"] = pd.cut(
        merged["age"],
        bins=[0, 18, 35, 55, 65, 100],
        labels=["0-17", "18-34", "35-54", "55-64", "65+"],
    )

    # Create income brackets
    merged["income_bracket"] = pd.cut(
        merged["income"],
        bins=[-np.inf, 25000, 50000, 100000, np.inf],
        labels=["<25k", "25-50k", "50-100k", "100k+"],
    )

    return merged


def compute_targets_from_data(
    data: pd.DataFrame, weight_col: str = "hh_weight"
) -> tuple[dict, dict]:
    """Compute calibration targets from weighted data."""
    weights = data[weight_col].values

    marginal_targets = {}

    # State targets
    marginal_targets["state"] = {}
    for state in data["state"].dropna().unique():
        mask = data["state"] == state
        marginal_targets["state"][state] = float(weights[mask].sum())

    # Age group targets
    marginal_targets["age_group"] = {}
    for age_grp in data["age_group"].dropna().unique():
        mask = data["age_group"] == age_grp
        marginal_targets["age_group"][str(age_grp)] = float(weights[mask].sum())

    # Income bracket targets
    marginal_targets["income_bracket"] = {}
    for bracket in data["income_bracket"].dropna().unique():
        mask = data["income_bracket"] == bracket
        marginal_targets["income_bracket"][str(bracket)] = float(weights[mask].sum())

    # Continuous targets
    continuous_targets = {
        "income": float((weights * data["income"]).sum()),
    }

    return marginal_targets, continuous_targets


def synthesize_population(
    seed_data: pd.DataFrame,
    n_synthetic: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthesize additional population records.

    For now, uses bootstrap resampling. TODO: Use normalizing flows.
    """
    np.random.seed(seed)

    # Bootstrap resample from seed data
    indices = np.random.choice(len(seed_data), n_synthetic, replace=True)
    synthetic = seed_data.iloc[indices].copy().reset_index(drop=True)

    # Add small noise to continuous variables
    synthetic["income"] = synthetic["income"] * np.random.lognormal(0, 0.1, n_synthetic)
    synthetic["income"] = np.maximum(synthetic["income"], 0)

    # Reset IDs
    synthetic["person_id"] = range(n_synthetic)
    synthetic["household_id"] = range(n_synthetic)  # Simplification: 1 person per HH

    # Initialize uniform weights
    synthetic["weight"] = 1.0

    return synthetic


def calibrate_population(
    data: pd.DataFrame,
    marginal_targets: dict,
    continuous_targets: dict,
    method: str = "sparse",
    target_sparsity: float = 0.9,
    device: str = "cpu",
    verbose: bool = True,
) -> pd.DataFrame:
    """Calibrate synthetic population to targets."""
    from microplex.calibration import HardConcreteCalibrator, SparseCalibrator

    # Convert categorical columns to strings for calibration
    data = data.copy()
    data["age_group"] = data["age_group"].astype(str)
    data["income_bracket"] = data["income_bracket"].astype(str)

    if method == "sparse":
        calibrator = SparseCalibrator(
            target_sparsity=target_sparsity,
            max_iter=2000,
            tol=1e-6,
        )
    elif method == "hardconcrete":
        calibrator = HardConcreteCalibrator(
            lambda_l0=1e-4,
            epochs=2000,
            lr=0.1,
            device=device,
            verbose=verbose,
            verbose_freq=200,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if verbose:
        n_cat = sum(len(v) for v in marginal_targets.values())
        n_cont = len(continuous_targets)
        print(f"Calibrating {len(data):,} records to {n_cat + n_cont} targets...")

    start = time.time()
    result = calibrator.fit_transform(data, marginal_targets, continuous_targets)
    elapsed = time.time() - start

    if verbose:
        val = calibrator.validate(result)
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Sparsity: {calibrator.get_sparsity():.1%}")
        print(f"  Max error: {val['max_error']:.2%}")
        print(f"  Mean error: {val['mean_error']:.2%}")

    return result


def build_us_microplex(
    data_dir: Path,
    n_synthetic: int = 100000,
    target_sparsity: float = 0.9,
    calibration_method: str = "sparse",
    device: str = "cpu",
    output_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build US microplex: synthesize and calibrate population."""

    if verbose:
        print("=" * 60)
        print("BUILDING US MICROPLEX")
        print("=" * 60)

    # Step 1: Load CPS data
    if verbose:
        print("\n1. Loading CPS ASEC data...")
    persons, households = load_cps_data(data_dir)
    seed_data = merge_person_household(persons, households)
    if verbose:
        print(f"   Loaded {len(seed_data):,} person records from {len(households):,} households")

    # Step 2: Compute targets from weighted CPS
    if verbose:
        print("\n2. Computing calibration targets...")
    marginal_targets, continuous_targets = compute_targets_from_data(seed_data)
    if verbose:
        print(f"   States: {len(marginal_targets['state'])}")
        print(f"   Age groups: {len(marginal_targets['age_group'])}")
        print(f"   Income brackets: {len(marginal_targets['income_bracket'])}")
        print(f"   Total income target: ${continuous_targets['income']:,.0f}")

    # Step 3: Synthesize population
    if verbose:
        print(f"\n3. Synthesizing {n_synthetic:,} records...")
    synthetic = synthesize_population(seed_data, n_synthetic)
    if verbose:
        print(f"   Generated {len(synthetic):,} synthetic records")

    # Step 4: Calibrate to targets
    if verbose:
        print(f"\n4. Calibrating with {calibration_method} method...")
    calibrated = calibrate_population(
        synthetic,
        marginal_targets,
        continuous_targets,
        method=calibration_method,
        target_sparsity=target_sparsity,
        device=device,
        verbose=verbose,
    )

    # Step 5: Save output
    if output_path:
        if verbose:
            print(f"\n5. Saving to {output_path}...")
        calibrated.to_parquet(output_path)
        if verbose:
            print(f"   Saved {len(calibrated):,} records")

    if verbose:
        print("\n" + "=" * 60)
        print("US MICROPLEX COMPLETE")
        print("=" * 60)
        non_zero = (calibrated["weight"] > 1e-9).sum()
        print(f"Total records: {len(calibrated):,}")
        print(f"Non-zero weights: {non_zero:,} ({non_zero/len(calibrated):.1%})")
        print(f"Total weighted population: {calibrated['weight'].sum():,.0f}")

    return calibrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build US Microplex")
    parser.add_argument(
        "--n-synthetic", type=int, default=100000,
        help="Number of synthetic records to generate"
    )
    parser.add_argument(
        "--target-sparsity", type=float, default=0.9,
        help="Target sparsity (fraction of zero weights)"
    )
    parser.add_argument(
        "--method", choices=["sparse", "hardconcrete"], default="sparse",
        help="Calibration method"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for HardConcrete (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    output_path = Path(args.output) if args.output else None

    build_us_microplex(
        data_dir=data_dir,
        n_synthetic=args.n_synthetic,
        target_sparsity=args.target_sparsity,
        calibration_method=args.method,
        device=args.device,
        output_path=output_path,
        verbose=not args.quiet,
    )
