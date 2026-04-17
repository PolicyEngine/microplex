#!/usr/bin/env python3
"""Run reweighting method comparison benchmark on real data.

Evaluates methods on both in-sample (training) and out-of-sample (held-out)
targets. Calibrates on age_group + weight, evaluates on held-out is_male.

Usage:
    python scripts/run_reweighting_benchmark.py
    python scripts/run_reweighting_benchmark.py --methods ipf entropy hardconcrete
    python scripts/run_reweighting_benchmark.py --output benchmarks/results/reweighting_full.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(data_dir: Path, max_rows: int = 20000) -> pd.DataFrame:
    """Load stacked multi-source data for reweighting benchmark."""
    stacked_path = data_dir / "stacked_comprehensive.parquet"
    if not stacked_path.exists():
        print(f"ERROR: {stacked_path} not found")
        sys.exit(1)

    print(f"Loading {stacked_path}...")
    df = pd.read_parquet(stacked_path)
    print(f"  Total rows: {len(df):,}")

    # Subsample if needed
    if len(df) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), max_rows, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {max_rows:,} rows")

    return df


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """Build train and test calibration targets.

    Train targets: age_group (5 categories) + total weight (continuous)
    Test targets: is_male (2 categories)

    Returns:
        (df, train_marginal, train_continuous, test_marginal, test_continuous)
    """
    rng = np.random.RandomState(42)

    train_marginal = {}
    train_continuous = {}
    test_marginal = {}
    test_continuous = {}

    # Create age bins (TRAIN)
    if "age" in df.columns:
        bins = [0, 18, 35, 55, 65, 120]
        labels = ["0-17", "18-34", "35-54", "55-64", "65+"]
        df = df.copy()
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        counts = df["age_group"].value_counts(dropna=True)
        perturbed = {}
        for cat, count in counts.items():
            if pd.isna(cat):
                continue
            perturbed[str(cat)] = round(count * rng.uniform(0.7, 1.3))
        if perturbed:
            df["age_group"] = df["age_group"].astype(str)
            train_marginal["age_group"] = perturbed
            print(f"  Train target: age_group ({len(perturbed)} categories)")

    # is_male (TEST — held out during calibration)
    if "is_male" in df.columns and df["is_male"].isna().mean() < 0.05:
        counts = df["is_male"].value_counts(dropna=True)
        perturbed = {}
        for cat, count in counts.items():
            perturbed[cat] = round(count * rng.uniform(0.8, 1.2))
        test_marginal["is_male"] = perturbed
        print(f"  Test target:  is_male ({len(perturbed)} categories)")

    # Continuous: total weight (TRAIN)
    total_weight = df["weight"].sum()
    train_continuous["weight"] = round(total_weight * rng.uniform(0.9, 1.1))
    print(f"  Train target: weight (total={total_weight:,.0f} -> {train_continuous['weight']:,.0f})")

    return df, train_marginal, train_continuous, test_marginal, test_continuous


def build_methods(method_names: list[str] = None):
    """Build method instances from names."""
    from microplex.eval.reweighting_benchmark import (
        IPFMethod, EntropyMethod,
        L1SparseMethod, L0SparseMethod,
        SparseCalibratorMethod, HardConcreteMethod,
    )

    all_methods = {
        "ipf": IPFMethod(),
        "entropy": EntropyMethod(),
        "l1": L1SparseMethod(),
        "l0": L0SparseMethod(),
        "sparse": SparseCalibratorMethod(sparsity_weight=0.01),
        "hardconcrete": HardConcreteMethod(lambda_l0=1e-4, epochs=2000),
    }

    if method_names is None:
        method_names = ["ipf", "entropy", "l1", "l0", "sparse"]
        try:
            import l0
            method_names.append("hardconcrete")
        except ImportError:
            print("  (l0-python not installed, skipping HardConcrete)")

    methods = []
    for name in method_names:
        name_lower = name.lower()
        if name_lower in all_methods:
            methods.append(all_methods[name_lower])
        else:
            print(f"  WARNING: Unknown method '{name}', skipping")

    return methods


def evaluate_weights(data, weights, marginal_targets, continuous_targets):
    """Compute per-target relative errors for given weights."""
    errors = {}
    for var, var_targets in marginal_targets.items():
        for cat, target in var_targets.items():
            mask = data[var] == cat
            actual = float(weights[mask].sum())
            rel_err = abs(actual - target) / target if target > 0 else 0.0
            errors[f"{var}={cat}"] = {
                "target": target, "actual": actual, "error": rel_err,
            }
    if continuous_targets:
        for var, target in continuous_targets.items():
            if var in data.columns:
                actual = float((weights * data[var].values).sum())
                rel_err = abs(actual - target) / abs(target) if target != 0 else 0.0
                errors[var] = {"target": target, "actual": actual, "error": rel_err}
    return errors


def main():
    parser = argparse.ArgumentParser(description="Run reweighting method benchmark")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to compare (default: all available). "
             "Options: ipf, entropy, l1, l0, sparse, hardconcrete",
    )
    parser.add_argument("--output", type=str, help="Save results to JSON")
    parser.add_argument(
        "--max-rows", type=int, default=5000,
        help="Max rows (default: 5000)",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).parent.parent / "data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    df = load_data(data_dir, max_rows=args.max_rows)

    # Build train/test targets
    print("\nBuilding calibration targets...")
    df, train_marginal, train_continuous, test_marginal, test_continuous = build_targets(df)

    n_train = sum(len(v) for v in train_marginal.values()) + len(train_continuous)
    n_test = sum(len(v) for v in test_marginal.values()) + len(test_continuous)
    print(f"  Train: {n_train} targets, Test: {n_test} targets")

    if not train_marginal:
        print("ERROR: No training targets found.")
        sys.exit(1)

    # Drop NaN rows in ALL target columns (train + test)
    all_cat_cols = list(train_marginal.keys()) + list(test_marginal.keys())
    cat_cols = [c for c in all_cat_cols if c in df.columns]
    if cat_cols:
        before = len(df)
        df = df.dropna(subset=cat_cols).reset_index(drop=True)
        if len(df) < before:
            print(f"  Dropped {before - len(df)} rows with NaN ({len(df)} remaining)")

    # Build methods
    methods = build_methods(args.methods)
    print(f"\nMethods ({len(methods)}): {[m.name for m in methods]}")

    # All targets for evaluation
    all_marginal = {**train_marginal, **test_marginal}
    all_continuous = {**train_continuous, **test_continuous}

    # Run benchmark
    print(f"\n{'Method':<20} {'Train err':>10} {'Test err':>10} "
          f"{'All err':>10} {'Sparsity':>10} {'Time':>8}")
    print("-" * 78)

    results = {}
    for method in methods:
        t0 = time.time()
        try:
            # Fit on TRAIN targets only
            method.fit(df, train_marginal, train_continuous if train_continuous else None)
            elapsed = time.time() - t0
            weights = method.get_weights()

            # Evaluate on train, test, and all targets
            train_errs = evaluate_weights(df, weights, train_marginal, train_continuous)
            test_errs = evaluate_weights(df, weights, test_marginal, test_continuous)
            all_errs = evaluate_weights(df, weights, all_marginal, all_continuous)

            train_mean = np.mean([e["error"] for e in train_errs.values()])
            test_mean = np.mean([e["error"] for e in test_errs.values()]) if test_errs else 0.0
            all_mean = np.mean([e["error"] for e in all_errs.values()])
            all_max = max(e["error"] for e in all_errs.values())

            mean_w = weights.mean()
            cv = float(weights.std() / mean_w) if mean_w > 0 else 0.0
            sparsity = float((weights < 1e-9).sum() / len(weights))

            results[method.name] = {
                "method_name": method.name,
                "train_mean_error": round(train_mean, 6),
                "test_mean_error": round(test_mean, 6),
                "mean_relative_error": round(all_mean, 6),
                "max_relative_error": round(all_max, 6),
                "weight_cv": round(cv, 4),
                "sparsity": round(sparsity, 4),
                "elapsed_seconds": round(elapsed, 2),
                "train_errors": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                                     for kk, vv in v.items()}
                                for k, v in train_errs.items()},
                "test_errors": {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                                    for kk, vv in v.items()}
                               for k, v in test_errs.items()},
            }

            print(f"{method.name:<20} {train_mean:>10.2%} {test_mean:>10.2%} "
                  f"{all_mean:>10.2%} {sparsity:>10.1%} {elapsed:>7.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results[method.name] = {
                "method_name": method.name,
                "train_mean_error": float("inf"),
                "test_mean_error": float("inf"),
                "mean_relative_error": float("inf"),
                "max_relative_error": float("inf"),
                "weight_cv": 0.0,
                "sparsity": 0.0,
                "elapsed_seconds": round(elapsed, 2),
            }
            print(f"{method.name:<20} {'ERROR':>10} {'ERROR':>10} "
                  f"{'ERROR':>10} {'':>10} {elapsed:>7.1f}s  {e}")

    print("-" * 78)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "seed": args.seed,
            "methods": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_records": len(df),
            "n_train_targets": n_train,
            "n_test_targets": n_test,
            "n_marginal_targets": sum(len(v) for v in all_marginal.values()),
            "n_continuous_targets": len(all_continuous),
            "train_variables": list(train_marginal.keys()) + list(train_continuous.keys()),
            "test_variables": list(test_marginal.keys()) + list(test_continuous.keys()),
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
