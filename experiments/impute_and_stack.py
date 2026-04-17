"""Impute-then-stack approach for multi-source fusion using existing microplex Synthesizer.

Strategy:
1. Identify shared vs survey-specific variables
2. Train Synthesizer (with ZI-QDNN) on each survey to predict its unique vars from shared vars
3. Use trained models to impute missing vars onto the other survey
4. Stack completed records into unified dataset
5. Train panel synthesizer on unified data
6. Evaluate coverage on holdouts from each survey
"""

import sys

sys.stdout.reconfigure(line_buffering=True)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from experiments.sipp_initial_state_model import FullSynthesizer
from experiments.sipp_inspect_holdouts import prepare_sipp_panel

# Import synthesizer components directly to avoid polars dependency
from microplex.synthesizer import Synthesizer
from pipelines.data_loaders import load_sipp


class ConditionalImputer:
    """Imputes missing variables using microplex Synthesizer (ZI-QDNN under the hood)."""

    def __init__(self, cond_cols: list[str], target_cols: list[str]):
        self.cond_cols = cond_cols
        self.target_cols = target_cols
        self.synthesizer = None

    def fit(self, df: pd.DataFrame, epochs: int = 100, verbose: bool = True):
        """Train imputation model on complete cases."""
        print(f"  Training imputer: {self.cond_cols} → {self.target_cols}")

        self.synthesizer = Synthesizer(
            target_vars=self.target_cols,
            condition_vars=self.cond_cols,
            zero_inflated=True,
            log_transform=True,
            n_layers=4,
            hidden_dim=64,
        )

        self.synthesizer.fit(
            df,
            weight_col=None,
            epochs=epochs,
            batch_size=256,
            verbose=verbose,
        )

        return self

    def impute(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """Impute target columns given conditioning columns."""
        # Generate synthetic values conditioned on the condition columns
        result = self.synthesizer.generate(df, seed=seed)

        # Copy original dataframe and add imputed values
        df_out = df.copy()
        for col in self.target_cols:
            df_out[col] = result[col].values

        return df_out


def load_cps_asec_data() -> pd.DataFrame:
    """Load CPS ASEC data from local parquet files."""
    try:
        from microplex.data import load_cps_asec
        _, persons = load_cps_asec()

        # Map to expected columns
        df = pd.DataFrame({
            'person_id': np.arange(len(persons)),
            'age': persons['age'].values,
            'employment_income': persons['employment_income'].values if 'employment_income' in persons.columns else persons['income'].values,
            'weight': persons['weight'].values if 'weight' in persons.columns else 1.0,
            'period': 0,  # Cross-sectional
        })

        # Filter to working-age adults with valid data
        df = df[(df['age'] >= 18) & (df['age'] <= 80)]
        df = df[df['employment_income'] >= 0]

        print(f"  Loaded {len(df)} CPS persons")
        return df
    except Exception as e:
        print(f"Error loading CPS: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_coverage(
    holdout_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_cols: list[str],
    n_periods: int = 6,
) -> float:
    """Compute mean nearest-neighbor distance from holdout to synthetic."""

    def to_matrix(df, n_periods):
        persons = sorted(df['person_id'].unique())

        rows = []
        for pid in persons:
            person_data = df[df['person_id'] == pid].sort_values('period')
            if len(person_data) >= n_periods:
                rows.append(person_data[feature_cols].values[:n_periods].flatten())

        return np.array(rows) if rows else np.zeros((0, len(feature_cols) * n_periods))

    holdout_mat = to_matrix(holdout_df, n_periods)
    synth_mat = to_matrix(synth_df, n_periods)

    if len(holdout_mat) == 0 or len(synth_mat) == 0:
        return float('inf')

    scaler = StandardScaler().fit(synth_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(holdout_scaled)

    return float(np.mean(distances))


def main():
    print("=" * 70)
    print("IMPUTE-THEN-STACK FUSION EXPERIMENT (using microplex Synthesizer)")
    print("=" * 70)

    # Load SIPP
    print("\n1. Loading SIPP...")
    sipp_raw = load_sipp(sample_frac=1.0)
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    sipp_specific = ['job1_income', 'job2_income', 'job3_income', 'tip_income']
    shared_cols = ['age', 'total_income']

    # Filter to complete 6-period panels
    periods_per_person = sipp.groupby('person_id')['period'].nunique()
    complete_persons = periods_per_person[periods_per_person >= 6].index
    sipp = sipp[sipp['person_id'].isin(complete_persons)]
    sipp = sipp.sort_values(['person_id', 'period']).groupby('person_id').head(6)

    print(f"  SIPP: {sipp['person_id'].nunique()} complete panels")

    # Load CPS
    print("\n2. Loading CPS ASEC...")
    cps = load_cps_asec_data()
    if cps is None:
        print("Failed to load CPS")
        return

    # Scale CPS annual income to monthly to match SIPP
    cps['employment_income'] = cps['employment_income'] / 12.0
    cps['total_income'] = cps['employment_income']  # Rename to match
    print(f"  CPS: {len(cps)} records (scaled to monthly)")

    # Split both into train/holdout (80/20)
    np.random.seed(42)

    sipp_persons = sipp['person_id'].unique()
    np.random.shuffle(sipp_persons)
    n_sipp_train = int(len(sipp_persons) * 0.8)
    sipp_train = sipp[sipp['person_id'].isin(sipp_persons[:n_sipp_train])]
    sipp_holdout = sipp[sipp['person_id'].isin(sipp_persons[n_sipp_train:])]

    cps_persons = cps['person_id'].unique()
    np.random.shuffle(cps_persons)
    n_cps_train = int(len(cps_persons) * 0.8)
    cps_train = cps[cps['person_id'].isin(cps_persons[:n_cps_train])]
    cps_holdout = cps[cps['person_id'].isin(cps_persons[n_cps_train:])]

    print(f"\n  SIPP train: {sipp_train['person_id'].nunique()}, holdout: {sipp_holdout['person_id'].nunique()}")
    print(f"  CPS train: {cps_train['person_id'].nunique()}, holdout: {cps_holdout['person_id'].nunique()}")

    # Train imputer: predict SIPP-specific vars from shared vars
    print("\n3. Training ZI-QDNN imputer (shared → SIPP-specific)...")

    # Get first period for each person (not period==0, since SIPP periods are 1-indexed)
    min_periods = sipp_train.groupby('person_id')['period'].min().reset_index()
    min_periods.columns = ['person_id', 'min_period']
    sipp_train_merged = sipp_train.merge(min_periods, on='person_id')
    sipp_t0 = sipp_train_merged[sipp_train_merged['period'] == sipp_train_merged['min_period']].copy()
    print(f"  SIPP first-period records: {len(sipp_t0)}")

    imputer = ConditionalImputer(
        cond_cols=shared_cols,
        target_cols=sipp_specific,
    )
    imputer.fit(sipp_t0, epochs=100, verbose=True)

    # Impute SIPP-specific vars onto CPS
    print("\n4. Imputing SIPP-specific vars onto CPS...")
    cps_imputed = imputer.impute(cps_train, seed=42)
    cps_imputed['source'] = 'cps'

    print("  Sample imputed CPS record:")
    sample = cps_imputed.iloc[0]
    print(f"    age={sample['age']:.0f}, total_income={sample['total_income']:.0f}")
    print(f"    job1={sample['job1_income']:.0f}, job2={sample['job2_income']:.0f}, job3={sample['job3_income']:.0f}, tip={sample['tip_income']:.0f}")

    # Create unified dataset
    print("\n5. Creating unified dataset...")

    # SIPP already complete - remap to integer person_ids
    sipp_train_unified = sipp_train.copy()
    sipp_train_unified['source'] = 'sipp'
    # Map string person_ids to integers
    sipp_person_map = {pid: i for i, pid in enumerate(sipp_train_unified['person_id'].unique())}
    sipp_train_unified['person_id'] = sipp_train_unified['person_id'].map(sipp_person_map)
    max_sipp_pid = sipp_train_unified['person_id'].max()

    # For CPS, replicate to 6 periods (same values - will learn transitions from combined data)
    cps_expanded = []
    for _, row in cps_imputed.iterrows():
        for t in range(6):
            new_row = row.to_dict()
            new_row['period'] = t
            cps_expanded.append(new_row)
    cps_expanded = pd.DataFrame(cps_expanded)
    cps_expanded['person_id'] = cps_expanded['person_id'] + max_sipp_pid + 1

    unified_cols = ['person_id', 'period', 'age', 'total_income'] + sipp_specific + ['source']
    sipp_final = sipp_train_unified[unified_cols].copy()
    cps_final = cps_expanded[unified_cols].copy()

    unified = pd.concat([sipp_final, cps_final], ignore_index=True)

    print(f"  Unified: {unified['person_id'].nunique()} persons")
    print(f"    SIPP: {sipp_final['person_id'].nunique()}")
    print(f"    CPS: {cps_final['person_id'].nunique()}")

    # Train synthesizer on unified data
    print("\n6. Training FullSynthesizer on unified data...")
    synth = FullSynthesizer(n_features=len(feature_cols))
    synth.fit(unified, feature_cols, epochs=100)

    # Generate synthetics
    n_synth = unified['person_id'].nunique()
    print(f"\n7. Generating {n_synth} synthetic trajectories...")
    synth_df = synth.generate(n_synth, n_periods=6, seed=42)

    # Evaluate coverage on each holdout
    print("\n8. Evaluating coverage...")

    # SIPP panel holdout - remap to integer person_ids
    sipp_holdout_int = sipp_holdout.copy()
    holdout_person_map = {pid: i for i, pid in enumerate(sipp_holdout_int['person_id'].unique())}
    sipp_holdout_int['person_id'] = sipp_holdout_int['person_id'].map(holdout_person_map)
    sipp_coverage = compute_coverage(sipp_holdout_int, synth_df, feature_cols)
    print(f"  SIPP panel holdout: {sipp_coverage:.4f}")

    # CPS cross-section holdout (expand to fake panel for comparison)
    cps_holdout_expanded = []
    for _, row in cps_holdout.iterrows():
        for t in range(6):
            new_row = {
                'person_id': row['person_id'],
                'period': t,
                'age': row['age'],
                'total_income': row['total_income'],
                'job1_income': 0,
                'job2_income': 0,
                'job3_income': 0,
                'tip_income': 0,
            }
            cps_holdout_expanded.append(new_row)
    cps_holdout_df = pd.DataFrame(cps_holdout_expanded)

    # For CPS, compare on shared vars only
    cps_coverage = compute_coverage(cps_holdout_df, synth_df, shared_cols)
    print(f"  CPS cross-section holdout (shared vars only): {cps_coverage:.4f}")

    # Compare to single-source baselines
    print("\n9. Training single-source baselines...")

    # SIPP-only - use the already-remapped holdout
    sipp_train_int = sipp_train.copy()
    train_person_map = {pid: i for i, pid in enumerate(sipp_train_int['person_id'].unique())}
    sipp_train_int['person_id'] = sipp_train_int['person_id'].map(train_person_map)
    synth_sipp = FullSynthesizer(n_features=len(feature_cols))
    synth_sipp.fit(sipp_train_int, feature_cols, epochs=100)
    synth_sipp_df = synth_sipp.generate(sipp_train_int['person_id'].nunique(), n_periods=6, seed=42)
    sipp_only_coverage = compute_coverage(sipp_holdout_int, synth_sipp_df, feature_cols)
    print(f"  SIPP-only baseline: {sipp_only_coverage:.4f}")

    # CPS-only (expand to fake panel)
    cps_train_expanded = []
    for _, row in cps_train.iterrows():
        for t in range(6):
            cps_train_expanded.append({
                'person_id': row['person_id'],
                'period': t,
                'age': row['age'],
                'total_income': row['total_income'],
                'job1_income': 0,
                'job2_income': 0,
                'job3_income': 0,
                'tip_income': 0,
            })
    cps_train_df = pd.DataFrame(cps_train_expanded)

    synth_cps = FullSynthesizer(n_features=len(feature_cols))
    synth_cps.fit(cps_train_df, feature_cols, epochs=100)
    synth_cps_df = synth_cps.generate(cps_train['person_id'].nunique(), n_periods=6, seed=42)
    cps_only_coverage = compute_coverage(cps_holdout_df, synth_cps_df, shared_cols)
    print(f"  CPS-only baseline (shared vars): {cps_only_coverage:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Impute-then-Stack vs Single-Source")
    print("=" * 70)

    sipp_delta = sipp_coverage - sipp_only_coverage
    cps_delta = cps_coverage - cps_only_coverage

    print("\nSIPP panel holdout (all vars):")
    print(f"  Impute-stack: {sipp_coverage:.4f}")
    print(f"  SIPP-only:    {sipp_only_coverage:.4f}")
    print(f"  Δ: {sipp_delta:+.4f} ({'worse' if sipp_delta > 0 else 'BETTER'})")

    print("\nCPS cross-section holdout (shared vars):")
    print(f"  Impute-stack: {cps_coverage:.4f}")
    print(f"  CPS-only:     {cps_only_coverage:.4f}")
    print(f"  Δ: {cps_delta:+.4f} ({'worse' if cps_delta > 0 else 'BETTER'})")

    # Save results
    results = pd.DataFrame([
        {'model': 'impute_stack', 'holdout': 'sipp_panel', 'coverage': sipp_coverage},
        {'model': 'impute_stack', 'holdout': 'cps_crosssection', 'coverage': cps_coverage},
        {'model': 'sipp_only', 'holdout': 'sipp_panel', 'coverage': sipp_only_coverage},
        {'model': 'cps_only', 'holdout': 'cps_crosssection', 'coverage': cps_only_coverage},
    ])
    results_path = Path(__file__).parent / "impute_stack_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Save the unified model
    synth.save(Path(__file__).parent / "impute_stack_synthesizer.pt")
    print("Model saved to experiments/impute_stack_synthesizer.pt")


if __name__ == "__main__":
    main()
