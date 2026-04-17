#!/usr/bin/env python3
"""Generate reweighting frontier data: records used vs out-of-sample error.

Sweeps regularization parameters for each method, recording (n_active, test_error)
pairs. Uses the same train/test split as the paper benchmark: calibrate on
age_group + weight, evaluate on held-out is_male.

Usage:
    python scripts/run_reweighting_frontier.py
    python scripts/run_reweighting_frontier.py --output benchmarks/results/reweighting_frontier.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(data_dir: Path, max_rows: int = 5000) -> pd.DataFrame:
    stacked_path = data_dir / "stacked_comprehensive.parquet"
    if not stacked_path.exists():
        print(f"ERROR: {stacked_path} not found")
        sys.exit(1)
    df = pd.read_parquet(stacked_path)
    if len(df) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), max_rows, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def build_targets(df: pd.DataFrame):
    """Build train (age_group + weight) and test (is_male) targets."""
    rng = np.random.RandomState(42)
    df = df.copy()

    # Train: age_group
    train_marginal = {}
    bins = [0, 18, 35, 55, 65, 120]
    labels = ["0-17", "18-34", "35-54", "55-64", "65+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)
    counts = df["age_group"].value_counts(dropna=True)
    train_marginal["age_group"] = {
        str(cat): round(count * rng.uniform(0.7, 1.3))
        for cat, count in counts.items() if not pd.isna(cat)
    }

    # Train: weight (continuous)
    train_continuous = {"weight": round(df["weight"].sum() * rng.uniform(0.9, 1.1))}

    # Test: is_male
    test_marginal = {}
    counts = df["is_male"].value_counts(dropna=True)
    test_marginal["is_male"] = {
        cat: round(count * rng.uniform(0.8, 1.2))
        for cat, count in counts.items()
    }

    # Drop NaN rows
    df = df.dropna(subset=["age_group", "is_male"]).reset_index(drop=True)

    return df, train_marginal, train_continuous, test_marginal


def evaluate_test_error(df, weights, test_marginal):
    """Mean relative error on held-out test targets."""
    errors = []
    for var, var_targets in test_marginal.items():
        for cat, target in var_targets.items():
            mask = df[var] == cat
            actual = float(weights[mask].sum())
            errors.append(abs(actual - target) / target if target > 0 else 0.0)
    return np.mean(errors)


def evaluate_train_error(df, weights, train_marginal, train_continuous):
    """Mean relative error on training targets."""
    errors = []
    for var, var_targets in train_marginal.items():
        for cat, target in var_targets.items():
            mask = df[var] == cat
            actual = float(weights[mask].sum())
            errors.append(abs(actual - target) / target if target > 0 else 0.0)
    for var, target in train_continuous.items():
        actual = float((weights * df[var].values).sum())
        errors.append(abs(actual - target) / abs(target) if target != 0 else 0.0)
    return np.mean(errors)


def run_single(method_cls, df, train_m, train_c, test_m, **kwargs):
    """Run a single method config and return results dict."""
    t0 = time.time()
    try:
        method = method_cls(**kwargs)
        method.fit(df, train_m, train_c if hasattr(method, '_sparsity_weight') or
                   isinstance(method, (type(None),)) else train_c)
        weights = method.get_weights()
        elapsed = time.time() - t0

        n_active = int((weights > 1e-9).sum())
        test_err = evaluate_test_error(df, weights, test_m)
        train_err = evaluate_train_error(df, weights, train_m, train_c)

        return {
            "n_active": n_active,
            "test_error": round(test_err, 6),
            "train_error": round(train_err, 6),
            "sparsity": round(1 - n_active / len(df), 4),
            "weight_cv": round(float(weights.std() / weights.mean()), 4) if weights.mean() > 0 else 0,
            "elapsed": round(elapsed, 2),
            "params": {k: v for k, v in kwargs.items() if k != "verbose"},
        }
    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).parent.parent / "benchmarks" / "results" / "reweighting_frontier.json"))
    parser.add_argument("--data-dir", type=str,
                        default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--max-rows", type=int, default=5000)
    args = parser.parse_args()

    from microplex.eval.reweighting_benchmark import (
        IPFMethod, EntropyMethod, SparseCalibratorMethod, HardConcreteMethod,
        L1SparseMethod, L0SparseMethod,
    )

    # Load data and build targets
    df = load_data(Path(args.data_dir), args.max_rows)
    df, train_m, train_c, test_m = build_targets(df)
    n_records = len(df)
    print(f"Records: {n_records:,}")
    print(f"Train targets: {sum(len(v) for v in train_m.values()) + len(train_c)}")
    print(f"Test targets: {sum(len(v) for v in test_m.values())}")

    results = {"n_records": n_records, "methods": {}}

    # --- Dense methods (single point each) ---
    print("\n--- Dense methods ---")

    for name, method_cls, kwargs in [
        ("IPF", IPFMethod, {}),
        ("Entropy", EntropyMethod, {}),
        ("L1-Sparse", L1SparseMethod, {}),
        ("L0-Sparse", L0SparseMethod, {}),
    ]:
        print(f"  {name}...")
        method = method_cls(**kwargs)
        method.fit(df, train_m, train_c)
        weights = method.get_weights()
        n_active = int((weights > 1e-9).sum())
        test_err = evaluate_test_error(df, weights, test_m)
        train_err = evaluate_train_error(df, weights, train_m, train_c)
        results["methods"][name] = [{
            "n_active": n_active,
            "test_error": round(test_err, 6),
            "train_error": round(train_err, 6),
            "sparsity": round(1 - n_active / n_records, 4),
            "params": kwargs,
        }]
        print(f"    n_active={n_active}, test_error={test_err:.4f}")

    # --- SparseCalibrator: sweep sparsity_weight ---
    print("\n--- SparseCalibrator (sweep sparsity_weight) ---")
    sc_results = []
    for sw in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        print(f"  sparsity_weight={sw}...")
        method = SparseCalibratorMethod(sparsity_weight=sw)
        method.fit(df, train_m, train_c)
        weights = method.get_weights()
        n_active = int((weights > 1e-9).sum())
        test_err = evaluate_test_error(df, weights, test_m)
        train_err = evaluate_train_error(df, weights, train_m, train_c)
        sc_results.append({
            "n_active": n_active,
            "test_error": round(test_err, 6),
            "train_error": round(train_err, 6),
            "sparsity": round(1 - n_active / n_records, 4),
            "params": {"sparsity_weight": sw},
        })
        print(f"    n_active={n_active}, test_error={test_err:.4f}, train_error={train_err:.4f}")
    results["methods"]["SparseCalibrator"] = sc_results

    # --- HardConcrete: sweep lambda_l0 ---
    print("\n--- HardConcrete (sweep lambda_l0) ---")
    hc_results = []
    for lam in [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        print(f"  lambda_l0={lam:.0e}...")
        try:
            method = HardConcreteMethod(lambda_l0=lam, epochs=2000)
            method.fit(df, train_m, train_c)
            weights = method.get_weights()
            n_active = int((weights > 1e-9).sum())
            test_err = evaluate_test_error(df, weights, test_m)
            train_err = evaluate_train_error(df, weights, train_m, train_c)
            hc_results.append({
                "n_active": n_active,
                "test_error": round(test_err, 6),
                "train_error": round(train_err, 6),
                "sparsity": round(1 - n_active / n_records, 4),
                "params": {"lambda_l0": lam, "epochs": 2000},
            })
            print(f"    n_active={n_active}, test_error={test_err:.4f}, train_error={train_err:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
    results["methods"]["HardConcrete"] = hc_results

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
