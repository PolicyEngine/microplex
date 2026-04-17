#!/usr/bin/env python3
"""Run microplex evaluation on real data.

Evaluates:
1. Synthesis: PRDC coverage per donor source (CPS, SIPP, PSID) against holdouts
2. Reweighting: loss vs aggregate targets (SOI income, benefit spending, geography)

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --synthesis-only
    python scripts/run_eval.py --reweighting-only
    python scripts/run_eval.py --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_multi_source_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load stacked multi-source data, split by survey.

    Keeps only numeric columns with <5% NaN per source, then drops remaining NaN rows.
    """
    stacked_path = data_dir / "stacked_comprehensive.parquet"
    if not stacked_path.exists():
        print(f"ERROR: {stacked_path} not found")
        sys.exit(1)

    print(f"Loading {stacked_path}...")
    df = pd.read_parquet(stacked_path)
    print(f"  Total rows: {len(df):,}")
    print(f"  Sources: {df['_survey'].value_counts().to_dict()}")

    sources = {}
    for survey_name in df["_survey"].unique():
        sub = df[df["_survey"] == survey_name].copy()
        # Drop _survey column
        if "_survey" in sub.columns:
            sub = sub.drop(columns=["_survey"])
        # Keep only numeric columns with low NaN rate
        numeric_cols = []
        for col in sub.columns:
            if sub[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                if sub[col].isna().mean() < 0.05:
                    numeric_cols.append(col)
        sub = sub[numeric_cols].dropna().reset_index(drop=True)
        sources[survey_name] = sub
        print(f"  {survey_name}: {len(sub):,} rows, {len(sub.columns)} cols: {sorted(sub.columns.tolist())}")

    return sources


def load_weighted_data(data_dir: Path) -> pd.DataFrame:
    """Load weighted microdata for reweighting eval."""
    # Prefer enhanced CPS (real weights summing to ~330M US population)
    # NOT expanded (which has synthetic records with inflated weights)
    enhanced_path = data_dir / "cps_enhanced_persons.parquet"

    for path in [enhanced_path]:
        if path.exists():
            print(f"Loading {path.name}...")
            df = pd.read_parquet(path)
            # Ensure weight column exists
            if "weight" not in df.columns and "person_weight" in df.columns:
                df["weight"] = df["person_weight"]
            print(f"  Shape: {df.shape}")
            return df

    print("ERROR: No weighted CPS data found")
    sys.exit(1)


def find_shared_cols(sources: dict[str, pd.DataFrame]) -> list[str]:
    """Find numeric columns present in ALL sources with <5% NaN."""
    all_source_names = list(sources.keys())
    shared = []
    for col in sources[all_source_names[0]].columns:
        # Must be numeric
        if sources[all_source_names[0]][col].dtype not in [
            np.float64, np.int64, np.float32, np.int32
        ]:
            continue
        # Must be in all sources with low NaN rate
        in_all = True
        for name, df in sources.items():
            if col not in df.columns:
                in_all = False
                break
            if df[col].isna().mean() > 0.05:
                in_all = False
                break
        if in_all:
            shared.append(col)

    # Exclude weight/id columns from synthesis eval
    exclude = {"weight", "person_id", "household_id", "interview_number"}
    shared = [c for c in shared if c not in exclude]
    return sorted(shared)


def run_synthesis_eval(
    sources: dict[str, pd.DataFrame],
    shared_cols: list[str],
    k: int = 5,
) -> dict:
    """Run per-source synthesis evaluation.

    For each donor source:
    1. Split 80/20 train/holdout
    2. Train PopulationDGP on train portion
    3. Generate synthetic records
    4. Compute PRDC against holdout (multivariate coverage)
    """
    from microplex.dgp import PopulationDGP, Survey, compute_prdc
    from microplex.eval.harness import SourceCoverage, SynthesisEvalResult

    print("\n" + "=" * 70)
    print("SYNTHESIS EVALUATION (per-source holdout)")
    print("=" * 70)

    rng = np.random.RandomState(42)
    coverages = []
    total_synthetic = 0

    for name, df in sources.items():
        # Only use numeric columns, exclude weight/id
        exclude = {"weight", "person_id", "household_id", "interview_number"}
        eval_cols = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        if len(eval_cols) < 3:
            print(f"\n  {name}: skipping (only {len(eval_cols)} numeric cols)")
            continue

        # Subsample large sources for speed (cap at 20K)
        max_n = 20000
        if len(df) > max_n:
            sub_idx = rng.choice(len(df), max_n, replace=False)
            df = df.iloc[sub_idx].reset_index(drop=True)

        print(f"\n  {name}: {len(df):,} rows, {len(eval_cols)} eval cols")
        print(f"    Cols: {eval_cols}")

        # Split
        n = len(df)
        n_holdout = int(n * 0.2)
        perm = rng.permutation(n)
        holdout = df[eval_cols].iloc[perm[:n_holdout]].reset_index(drop=True)
        train = df[eval_cols].iloc[perm[n_holdout:]].reset_index(drop=True)

        # Pick conditioning columns (first 2-3 columns) and target columns
        # Use demographics (age, is_male, etc.) as conditions, income/econ as targets
        demo_cols = [c for c in ["age", "is_male", "race", "hispanic", "marital_status"]
                     if c in eval_cols]
        target_cols = [c for c in eval_cols if c not in demo_cols]

        if not demo_cols or not target_cols:
            # Fallback: first 2 cols as shared, rest as targets
            demo_cols = eval_cols[:2]
            target_cols = eval_cols[2:]

        print(f"    Condition cols: {demo_cols}")
        print(f"    Target cols: {target_cols}")

        # Train DGP on this source alone
        t0 = time.time()
        dgp = PopulationDGP(n_estimators=50, random_state=42)
        survey = Survey(name, train, columns=eval_cols)
        try:
            dgp.fit([survey], shared_cols=demo_cols)
        except Exception as e:
            print(f"    FIT ERROR: {e}")
            continue

        # Generate
        n_gen = n_holdout * 3  # Generate 3x holdout size
        try:
            synthetic = dgp.generate(n=n_gen, seed=42)
        except (IndexError, ValueError) as e:
            # DGP bug: single-class zero-inflation classifier
            print(f"    GENERATE ERROR: {e}")
            continue
        elapsed = time.time() - t0

        # Compute PRDC on all eval columns present in synthetic
        common_cols = [c for c in eval_cols if c in synthetic.columns]
        holdout_vals = holdout[common_cols].dropna().values.astype(float)
        synth_vals = synthetic[common_cols].dropna().values.astype(float)

        # Standardize before PRDC
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        holdout_scaled = scaler.fit_transform(holdout_vals)
        synth_scaled = scaler.transform(synth_vals)

        prdc = compute_prdc(holdout_scaled, synth_scaled, k=k)
        total_synthetic += n_gen

        sc = SourceCoverage(
            source_name=name,
            precision=prdc["precision"],
            recall=prdc["recall"],
            density=prdc["density"],
            coverage=prdc["coverage"],
            n_holdout=len(holdout_vals),
            n_synthetic=len(synth_vals),
            columns_evaluated=common_cols,
        )
        coverages.append(sc)

        print(f"    Coverage={sc.coverage:.1%}  Precision={sc.precision:.1%}  "
              f"Recall={sc.recall:.1%}  Density={sc.density:.2f}  ({elapsed:.1f}s)")

    result = SynthesisEvalResult(
        source_coverages=coverages,
        n_synthetic=total_synthetic,
    )
    print(f"\n{result.summary()}")

    return result.to_dict()


def run_reweighting_eval(data: pd.DataFrame) -> dict:
    """Run reweighting evaluation against aggregate targets."""
    from microplex.eval.harness import EvalHarness

    print("\n" + "=" * 70)
    print("REWEIGHTING EVALUATION")
    print("=" * 70)
    print(f"Data: {data.shape[0]:,} records, {data.shape[1]} columns")

    weight_col = "weight" if "weight" in data.columns else "person_weight"
    print(f"Weight column: {weight_col}")
    print(f"Total weighted population: {data[weight_col].sum():,.0f}")

    harness = EvalHarness()

    t0 = time.time()
    result = harness.evaluate_reweighting(data=data, weight_col=weight_col)
    elapsed = time.time() - t0

    print(f"\n{result.summary()}")
    print(f"\nElapsed: {elapsed:.1f}s")

    result_dict = result.to_dict()
    result_dict["elapsed_seconds"] = round(elapsed, 1)
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Run microplex evaluation")
    parser.add_argument("--synthesis-only", action="store_true")
    parser.add_argument("--reweighting-only", action="store_true")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--k", type=int, default=5, help="k for PRDC (default: 5)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    run_synth = not args.reweighting_only
    run_rw = not args.synthesis_only

    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if run_synth:
        sources = load_multi_source_data(data_dir)
        shared_cols = find_shared_cols(sources)
        results["synthesis"] = run_synthesis_eval(sources, shared_cols, k=args.k)

    if run_rw:
        weighted = load_weighted_data(data_dir)
        results["reweighting"] = run_reweighting_eval(weighted)

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
