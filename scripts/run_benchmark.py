#!/usr/bin/env python3
"""Run synthesis method comparison benchmark on real data.

Compares QRF, ZI-QRF, QDNN, ZI-QDNN, MAF, ZI-MAF (and CTGAN/TVAE if sdv installed)
on PRDC coverage outcomes against holdouts from each source.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --methods qrf zi-qrf qdnn zi-qdnn
    python scripts/run_benchmark.py --output benchmarks/results/benchmark.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_multi_source_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load stacked multi-source data, split by survey."""
    stacked_path = data_dir / "stacked_comprehensive.parquet"
    if not stacked_path.exists():
        print(f"ERROR: {stacked_path} not found")
        sys.exit(1)

    print(f"Loading {stacked_path}...")
    df = pd.read_parquet(stacked_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Sources: {df['_survey'].value_counts().to_dict()}")

    numeric_dtypes = [np.float64, np.int64, np.float32, np.int32]
    sources = {}
    for survey_name in df["_survey"].unique():
        sub = df[df["_survey"] == survey_name].drop(columns=["_survey"]).copy()
        numeric_cols = [
            col for col in sub.columns
            if sub[col].dtype in numeric_dtypes and sub[col].isna().mean() < 0.05
        ]
        sub = sub[numeric_cols].dropna().reset_index(drop=True)
        sources[survey_name] = sub
        print(f"  {survey_name}: {len(sub):,} rows, {len(sub.columns)} cols")

    return sources


def find_shared_cols(sources: dict[str, pd.DataFrame]) -> list[str]:
    """Find numeric columns present in ALL sources."""
    numeric_dtypes = [np.float64, np.int64, np.float32, np.int32]
    exclude = {"weight", "person_id", "household_id", "interview_number"}
    first_df = next(iter(sources.values()))

    shared = [
        col for col in first_df.columns
        if first_df[col].dtype in numeric_dtypes
        and col not in exclude
        and all(
            col in df.columns and df[col].isna().mean() < 0.05
            for df in sources.values()
        )
    ]
    return sorted(shared)


def build_methods(method_names: list[str] = None, fast: bool = False):
    """Build method instances from names."""
    from microplex.eval.benchmark import (
        QRFMethod, ZIQRFMethod, QDNNMethod, ZIQDNNMethod,
        MAFMethod, ZIMAFMethod, CTGANMethod, TVAEMethod,
    )

    # Fast mode: fewer estimators/epochs for quick testing
    qrf_kwargs = {"n_estimators": 20 if fast else 100}
    qdnn_kwargs = {"hidden_dim": 32 if fast else 64, "epochs": 10 if fast else 50}
    maf_kwargs = {
        "n_layers": 2 if fast else 4,
        "hidden_dim": 16 if fast else 32,
        "epochs": 10 if fast else 50,
    }
    gan_kwargs = {"epochs": 50 if fast else 300}

    all_methods = {
        "qrf": QRFMethod(**qrf_kwargs),
        "zi-qrf": ZIQRFMethod(**qrf_kwargs),
        "qdnn": QDNNMethod(**qdnn_kwargs),
        "zi-qdnn": ZIQDNNMethod(**qdnn_kwargs),
        "maf": MAFMethod(**maf_kwargs),
        "zi-maf": ZIMAFMethod(**maf_kwargs),
        "ctgan": CTGANMethod(**gan_kwargs),
        "tvae": TVAEMethod(**gan_kwargs),
    }

    if method_names is None:
        method_names = ["qrf", "zi-qrf", "qdnn", "zi-qdnn", "maf", "zi-maf"]
        # Add SDV methods if available
        try:
            import sdv
            method_names.extend(["ctgan", "tvae"])
        except ImportError:
            print("  (sdv not installed, skipping CTGAN/TVAE)")

    methods = []
    for name in method_names:
        name_lower = name.lower()
        if name_lower in all_methods:
            methods.append(all_methods[name_lower])
        else:
            print(f"  WARNING: Unknown method '{name}', skipping")

    return methods


def main():
    parser = argparse.ArgumentParser(description="Run synthesis method benchmark")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to compare (default: all available). "
             "Options: qrf, zi-qrf, qdnn, zi-qdnn, maf, zi-maf, ctgan, tvae",
    )
    parser.add_argument("--output", type=str, help="Save results to JSON")
    parser.add_argument("--k", type=int, default=5, help="k for PRDC (default: 5)")
    parser.add_argument(
        "--holdout-frac", type=float, default=0.2,
        help="Holdout fraction (default: 0.2)",
    )
    parser.add_argument(
        "--n-generate", type=int, default=None,
        help="Records to generate per method (default: sum of holdouts)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=20000,
        help="Max rows per source (default: 20000, for speed)",
    )
    parser.add_argument("--fast", action="store_true", help="Fast mode (fewer estimators/epochs)")
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).parent.parent / "data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="Number of seeds for multi-seed evaluation (default: 1, single run)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    sources = load_multi_source_data(data_dir)
    shared_cols = find_shared_cols(sources)
    print(f"\nShared columns ({len(shared_cols)}): {shared_cols}")

    # Subsample large sources
    rng = np.random.RandomState(args.seed)
    for name in list(sources.keys()):
        if len(sources[name]) > args.max_rows:
            idx = rng.choice(len(sources[name]), args.max_rows, replace=False)
            sources[name] = sources[name].iloc[idx].reset_index(drop=True)
            print(f"  Subsampled {name} to {args.max_rows:,} rows")

    # Build methods
    methods = build_methods(args.methods, fast=args.fast)
    print(f"\nMethods to compare ({len(methods)}): {[m.name for m in methods]}")

    # Run benchmark
    from microplex.eval.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(methods=methods)
    t0 = time.time()

    if args.n_seeds > 1:
        result_dict = runner.run_multi_seed(
            sources=sources,
            shared_cols=shared_cols,
            n_seeds=args.n_seeds,
            base_seed=args.seed,
            holdout_frac=args.holdout_frac,
            n_generate=args.n_generate,
            k=args.k,
        )
        total_elapsed = time.time() - t0

        print(f"\n{'='*75}")
        print(f"Multi-seed results ({args.n_seeds} seeds)")
        print(f"{'='*75}")
        for method_name, source_stats in result_dict["methods"].items():
            print(f"\n  {method_name}:")
            for source_name, stats in source_stats.items():
                print(f"    {source_name}: {stats['mean']:.1%} +/- {stats['se']:.1%} "
                      f"(n={stats['n_seeds']})")
    else:
        result = runner.run(
            sources=sources,
            shared_cols=shared_cols,
            holdout_frac=args.holdout_frac,
            n_generate=args.n_generate,
            k=args.k,
            seed=args.seed,
        )
        total_elapsed = time.time() - t0
        result_dict = result.to_dict()
        print(f"\n{result.summary()}")

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result_dict["total_elapsed_seconds"] = round(total_elapsed, 1)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
