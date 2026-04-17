"""Multi-source fusion experiment with real SIPP and CPS ASEC data.

Loads:
- SIPP panel data with income trajectories
- CPS ASEC cross-sectional data with broader income components

Trains FusedSynthesizer on both sources and evaluates coverage.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_loaders import load_sipp
from experiments.sipp_inspect_holdouts import prepare_sipp_panel
from experiments.fusion_synthesizer import FusedSynthesizer, evaluate_coverage


def load_cps_from_policyengine(sample_frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Load CPS ASEC data from PolicyEngine.

    Returns cross-sectional data with income components.
    """
    print("Loading CPS from PolicyEngine...")

    from policyengine_us_data import CPS_2024
    from policyengine_us import Microsimulation

    sim = Microsimulation(dataset=CPS_2024)

    # Get person-level variables
    variables_to_load = [
        'age',
        'employment_income',
        'self_employment_income',
        'interest_income',
        'dividend_income',
        'social_security_income',
        'unemployment_compensation',
        'person_weight',
    ]

    data = {}
    for var in variables_to_load:
        try:
            values = sim.calculate(var, 2024)
            data[var] = values.values
        except Exception as e:
            print(f"  Warning: Could not load {var}: {e}")

    df = pd.DataFrame(data)

    # Sample if needed
    if sample_frac < 1.0:
        np.random.seed(seed)
        n_sample = int(len(df) * sample_frac)
        df = df.sample(n=n_sample, random_state=seed)

    # Rename columns to common names
    df = df.rename(columns={
        'employment_income': 'wage_income',
    })

    # Ensure non-negative
    income_cols = ['wage_income', 'self_employment_income', 'interest_income',
                   'dividend_income', 'social_security_income', 'unemployment_compensation']
    for col in income_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Add person_id (just index for cross-sectional)
    df['person_id'] = range(len(df))
    df['period'] = 0  # Cross-sectional = period 0 only

    print(f"  Loaded {len(df):,} CPS records")
    print(f"  Columns: {list(df.columns)}")

    return df


def load_and_prepare_sipp(sample_frac: float = 0.3) -> pd.DataFrame:
    """Load and prepare SIPP panel data.

    IMPORTANT: SIPP income values are MONTHLY. CPS values are ANNUAL.
    We annualize SIPP incomes by multiplying by 12 to match CPS scale.
    """
    print("Loading SIPP panel data...")
    sipp_raw = load_sipp(sample_frac=sample_frac)
    sipp = prepare_sipp_panel(sipp_raw)

    # Rename to common variable names for better comparison with CPS
    sipp = sipp.rename(columns={
        'total_income': 'wage_income',  # SIPP's total income maps roughly to CPS wages
    })

    # CRITICAL: Annualize SIPP incomes to match CPS annual scale
    # SIPP incomes are monthly, CPS incomes are annual
    income_cols = ['wage_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    for col in income_cols:
        if col in sipp.columns:
            sipp[col] = sipp[col] * 12  # Monthly -> Annual

    print(f"  Loaded {sipp['person_id'].nunique():,} SIPP persons with {len(sipp):,} records")
    print(f"  NOTE: SIPP incomes annualized (multiplied by 12) to match CPS annual scale")

    return sipp


def identify_common_and_unique_variables(sipp_df: pd.DataFrame, cps_df: pd.DataFrame):
    """Identify common and unique variables between surveys."""

    # Core variables we care about
    potential_vars = [
        'age', 'wage_income', 'self_employment_income', 'interest_income',
        'dividend_income', 'social_security_income', 'unemployment_compensation',
        'job1_income', 'job2_income', 'job3_income', 'tip_income'
    ]

    sipp_vars = [v for v in potential_vars if v in sipp_df.columns]
    cps_vars = [v for v in potential_vars if v in cps_df.columns]

    common_vars = [v for v in potential_vars if v in sipp_df.columns and v in cps_df.columns]
    sipp_only = [v for v in sipp_vars if v not in cps_vars]
    cps_only = [v for v in cps_vars if v not in sipp_vars]

    all_vars = list(set(sipp_vars + cps_vars))
    # Sort for consistent ordering
    all_vars = [v for v in potential_vars if v in all_vars]

    print("\nVariable Analysis:")
    print(f"  Common to both: {common_vars}")
    print(f"  SIPP only: {sipp_only}")
    print(f"  CPS only: {cps_only}")
    print(f"  All variables: {all_vars}")

    return all_vars, common_vars, sipp_only, cps_only


def run_fusion_experiment():
    """Run the multi-source fusion experiment."""
    print("=" * 70)
    print("MULTI-SOURCE FUSION: REAL SIPP + CPS DATA")
    print("=" * 70)

    # Load data
    sipp = load_and_prepare_sipp(sample_frac=0.3)
    cps = load_cps_from_policyengine(sample_frac=0.1)

    # Identify variables
    all_vars, common_vars, sipp_only, cps_only = identify_common_and_unique_variables(sipp, cps)

    # Filter to 6-period complete panels for SIPP
    n_periods = 6

    def filter_complete_panels(df, n_periods):
        """Filter to persons with at least n_periods of data."""
        periods_per_person = df.groupby('person_id')['period'].nunique()
        complete = periods_per_person[periods_per_person >= n_periods].index
        df = df[df['person_id'].isin(complete)]
        df = df.sort_values(['person_id', 'period']).groupby('person_id').head(n_periods)
        return df

    sipp = filter_complete_panels(sipp, n_periods)

    # Split SIPP into train/holdout by person
    sipp_persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(sipp_persons)
    n_train = int(len(sipp_persons) * 0.8)
    train_sipp_persons = sipp_persons[:n_train]
    holdout_sipp_persons = sipp_persons[n_train:]

    train_sipp = sipp[sipp['person_id'].isin(train_sipp_persons)]
    holdout_sipp = sipp[sipp['person_id'].isin(holdout_sipp_persons)]

    # Split CPS into train/holdout
    cps_persons = cps['person_id'].unique()
    np.random.shuffle(cps_persons)
    n_train_cps = int(len(cps_persons) * 0.8)
    train_cps_persons = cps_persons[:n_train_cps]
    holdout_cps_persons = cps_persons[n_train_cps:]

    train_cps = cps[cps['person_id'].isin(train_cps_persons)]
    holdout_cps = cps[cps['person_id'].isin(holdout_cps_persons)]

    print(f"\nData splits:")
    print(f"  SIPP train: {train_sipp['person_id'].nunique():,} persons")
    print(f"  SIPP holdout: {holdout_sipp['person_id'].nunique():,} persons")
    print(f"  CPS train: {len(train_cps):,} records")
    print(f"  CPS holdout: {len(holdout_cps):,} records")

    # Create DataFrames for fusion with only relevant columns
    # For SIPP: Keep panel structure
    sipp_vars_present = ['person_id', 'period'] + [v for v in all_vars if v in train_sipp.columns]
    train_sipp_clean = train_sipp[sipp_vars_present].copy()
    holdout_sipp_clean = holdout_sipp[sipp_vars_present].copy()

    # For CPS: Keep cross-sectional structure
    cps_vars_present = ['person_id', 'period'] + [v for v in all_vars if v in train_cps.columns]
    train_cps_clean = train_cps[cps_vars_present].copy()
    holdout_cps_clean = holdout_cps[cps_vars_present].copy()

    # Show data distributions
    print("\n" + "=" * 70)
    print("DATA DISTRIBUTIONS")
    print("=" * 70)

    print("\nSIPP training data (initial period only):")
    sipp_initial = train_sipp[train_sipp['period'] == train_sipp.groupby('person_id')['period'].transform('min')]
    for var in all_vars:
        if var in sipp_initial.columns:
            vals = sipp_initial[var]
            print(f"  {var}: mean={vals.mean():.0f}, median={vals.median():.0f}, zero%={(vals==0).mean()*100:.1f}%")

    print("\nCPS training data:")
    for var in all_vars:
        if var in train_cps.columns:
            vals = train_cps[var]
            print(f"  {var}: mean={vals.mean():.0f}, median={vals.median():.0f}, zero%={(vals==0).mean()*100:.1f}%")

    # Train fused model
    print("\n" + "=" * 70)
    print("TRAINING FUSED SYNTHESIZER")
    print("=" * 70)

    synthesizer = FusedSynthesizer(all_vars, hidden_dim=256)
    synthesizer.fit(
        surveys={
            'sipp': train_sipp_clean,
            'cps': train_cps_clean,
        },
        epochs=150,
        lr=1e-3,
        verbose=True,
    )

    # Generate synthetic data
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 70)

    n_synth = 2000
    synth_df = synthesizer.generate(n_synth, n_periods, seed=42)
    print(f"\nGenerated {n_synth} synthetic trajectories")
    print(f"Variables: {list(synth_df.columns)}")

    # Show sample
    print("\nSample synthetic record (period 0):")
    sample_cols = ['person_id', 'period'] + all_vars
    sample_cols = [c for c in sample_cols if c in synth_df.columns]
    print(synth_df[synth_df['person_id'] == 0].head(1)[sample_cols].to_string())

    # Evaluate coverage on SIPP holdout (using SIPP variables)
    print("\n" + "=" * 70)
    print("EVALUATING COVERAGE")
    print("=" * 70)

    # SIPP panel coverage
    coverage = evaluate_coverage(
        synth_df,
        {
            'sipp_panel': holdout_sipp_clean,
        },
        all_vars,
        n_periods=n_periods,
        include_zero_indicators=True,
    )

    # CPS cross-sectional coverage (special handling for period=0)
    cps_vars = [v for v in all_vars if v in holdout_cps_clean.columns]
    if cps_vars:
        print("\n  Evaluating CPS cross-sectional coverage...")
        synth_period0 = synth_df[synth_df['period'] == 0].copy()

        # Compute coverage for CPS
        def compute_cross_sectional_coverage(synth_df, holdout_df, vars):
            """Compute coverage for cross-sectional data."""
            existing_vars = [v for v in vars if v in synth_df.columns and v in holdout_df.columns]
            if not existing_vars:
                return float('nan')

            # Add zero indicators for income-like variables
            zero_indicator_vars = ['wage_income', 'self_employment_income', 'interest_income',
                                   'dividend_income', 'unemployment_compensation']

            holdout_aug = holdout_df.copy()
            synth_aug = synth_df.copy()
            eval_vars = list(existing_vars)

            for var in existing_vars:
                if var in zero_indicator_vars:
                    holdout_aug[f'{var}_nz'] = (holdout_aug[var] > 0).astype(float)
                    synth_aug[f'{var}_nz'] = (synth_aug[var] > 0).astype(float)
                    eval_vars.append(f'{var}_nz')

            holdout_mat = holdout_aug[eval_vars].values
            synth_mat = synth_aug[eval_vars].values

            scaler = StandardScaler().fit(holdout_mat)
            holdout_scaled = scaler.transform(holdout_mat)
            synth_scaled = scaler.transform(synth_mat)

            nn_model = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
            distances, _ = nn_model.kneighbors(holdout_scaled)
            return float(np.mean(distances))

        cps_coverage = compute_cross_sectional_coverage(synth_period0, holdout_cps_clean, cps_vars)
        coverage['cps_crosssection'] = cps_coverage
        print(f"  cps_crosssection: coverage = {cps_coverage:.4f} ({len(cps_vars)} vars, {len(holdout_cps_clean)} holdout records)")

    # Train single-source baseline on SIPP only
    print("\n" + "=" * 70)
    print("BASELINE: SIPP-ONLY TRAINING")
    print("=" * 70)

    sipp_only_vars = [v for v in all_vars if v in train_sipp.columns]
    synth_sipp_only = FusedSynthesizer(sipp_only_vars, hidden_dim=256)
    synth_sipp_only.fit(
        {'sipp': train_sipp[['person_id', 'period'] + sipp_only_vars]},
        epochs=150,
        verbose=False,
    )
    synth_df_sipp_only = synth_sipp_only.generate(n_synth, n_periods, seed=42)

    coverage_sipp_only = evaluate_coverage(
        synth_df_sipp_only,
        {'sipp_panel': holdout_sipp[['person_id', 'period'] + sipp_only_vars]},
        sipp_only_vars,
        n_periods=n_periods,
        include_zero_indicators=True,
    )

    # Train single-source baseline on CPS only
    print("\n" + "=" * 70)
    print("BASELINE: CPS-ONLY TRAINING")
    print("=" * 70)

    cps_only_vars = [v for v in all_vars if v in train_cps.columns]
    synth_cps_only = FusedSynthesizer(cps_only_vars, hidden_dim=256)
    synth_cps_only.fit(
        {'cps': train_cps[['person_id', 'period'] + cps_only_vars]},
        epochs=150,
        verbose=False,
    )
    synth_df_cps_only = synth_cps_only.generate(n_synth, n_periods, seed=42)

    # Evaluate CPS-only on CPS holdout
    cps_only_coverage = compute_cross_sectional_coverage(
        synth_df_cps_only[synth_df_cps_only['period'] == 0],
        holdout_cps_clean,
        cps_only_vars
    )
    coverage_cps_only = {'cps_crosssection': cps_only_coverage}
    print(f"  cps_crosssection: coverage = {cps_only_coverage:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nFused model (SIPP + CPS):")
    for name, cov in coverage.items():
        print(f"  {name}: coverage = {cov:.4f}")

    print(f"\nSIPP-only baseline:")
    for name, cov in coverage_sipp_only.items():
        print(f"  {name}: coverage = {cov:.4f}")

    print(f"\nCPS-only baseline:")
    for name, cov in coverage_cps_only.items():
        print(f"  {name}: coverage = {cov:.4f}")

    # Diagnostics: Zero rates
    print("\n" + "=" * 70)
    print("DIAGNOSTICS: ZERO RATES")
    print("=" * 70)

    print(f"\n{'Variable':<25} {'SIPP Holdout':<15} {'CPS Holdout':<15} {'Synth (fused)':<15}")
    print("-" * 70)

    for var in all_vars:
        sipp_zero = "N/A"
        cps_zero = "N/A"
        synth_zero = "N/A"

        if var in holdout_sipp.columns:
            sipp_zero = f"{(holdout_sipp[var] == 0).mean()*100:.1f}%"
        if var in holdout_cps.columns:
            cps_zero = f"{(holdout_cps[var] == 0).mean()*100:.1f}%"
        if var in synth_df.columns:
            synth_zero = f"{(synth_df[var] == 0).mean()*100:.1f}%"

        print(f"{var:<25} {sipp_zero:<15} {cps_zero:<15} {synth_zero:<15}")

    # Diagnostics: Mean nonzero values
    print("\n" + "=" * 70)
    print("DIAGNOSTICS: MEAN NONZERO VALUES")
    print("=" * 70)

    print(f"\n{'Variable':<25} {'SIPP Holdout':<15} {'CPS Holdout':<15} {'Synth (fused)':<15}")
    print("-" * 70)

    for var in all_vars:
        def get_mean_nz(df, col):
            if col not in df.columns:
                return "N/A"
            nz = df[df[col] > 0][col]
            if len(nz) == 0:
                return "N/A"
            return f"{nz.mean():,.0f}"

        sipp_mean = get_mean_nz(holdout_sipp, var)
        cps_mean = get_mean_nz(holdout_cps, var)
        synth_mean = get_mean_nz(synth_df, var)

        print(f"{var:<25} {sipp_mean:<15} {cps_mean:<15} {synth_mean:<15}")

    # Save model
    model_path = Path(__file__).parent / "real_fusion_synthesizer.pt"
    synthesizer.save(str(model_path))

    # Save results
    results = {
        'fused_coverage': coverage,
        'sipp_only_coverage': coverage_sipp_only,
        'all_vars': all_vars,
        'common_vars': common_vars,
        'sipp_only_vars': sipp_only,
        'cps_only_vars': cps_only,
    }

    results_path = Path(__file__).parent / "real_fusion_results.csv"
    pd.DataFrame([
        {'model': 'fused', 'holdout': k, 'coverage': v}
        for k, v in coverage.items()
    ] + [
        {'model': 'sipp_only', 'holdout': k, 'coverage': v}
        for k, v in coverage_sipp_only.items()
    ] + [
        {'model': 'cps_only', 'holdout': k, 'coverage': v}
        for k, v in coverage_cps_only.items()
    ]).to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    # Print final summary table
    print("\n" + "=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"\n{'Model':<25} {'SIPP Panel':<20} {'CPS Cross-Section':<20}")
    print("-" * 65)
    print(f"{'Fused (SIPP + CPS)':<25} {coverage.get('sipp_panel', float('nan')):<20.4f} {coverage.get('cps_crosssection', float('nan')):<20.4f}")
    print(f"{'SIPP-only':<25} {coverage_sipp_only.get('sipp_panel', float('nan')):<20.4f} {'N/A':<20}")
    print(f"{'CPS-only':<25} {'N/A':<20} {coverage_cps_only.get('cps_crosssection', float('nan')):<20.4f}")
    print("\n(Lower coverage = better fit to holdout data)")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    sipp_improvement = (coverage_sipp_only.get('sipp_panel', 0) - coverage.get('sipp_panel', 0)) / coverage_sipp_only.get('sipp_panel', 1) * 100
    print(f"\nSIPP panel: Fused model improves over SIPP-only by {sipp_improvement:.1f}%")
    print("  - Fusion learns shared wage_income patterns from both surveys")
    print("  - CPS data provides additional context for income distributions")

    cps_penalty = (coverage.get('cps_crosssection', 0) - coverage_cps_only.get('cps_crosssection', 0)) / coverage_cps_only.get('cps_crosssection', 1) * 100
    print(f"\nCPS cross-section: Fused model is {cps_penalty:.1f}% worse than CPS-only")
    print("  - SIPP's monthly panel structure adds complexity")
    print("  - CPS-specific income components (interest, dividends) less accurate in fusion")

    return synthesizer, synth_df, coverage


if __name__ == "__main__":
    synthesizer, synth_df, coverage = run_fusion_experiment()
