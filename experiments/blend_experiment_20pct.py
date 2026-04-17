"""Blending experiment with 20% training data: bootstrap (real) + synthetic records.

Same as blend_experiment.py but with 20/80 train/holdout split instead of 80/20.
This tests how well the synthesizer performs with limited training data.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_loaders import load_sipp
from experiments.sipp_inspect_holdouts import prepare_sipp_panel
from experiments.sipp_initial_state_model import FullSynthesizer, compute_coverage


def generate_bootstrap_pool(
    synth: FullSynthesizer,
    train_df: pd.DataFrame,
    feature_cols: list,
    n_bootstrap: int,
    n_synth: int,
    n_periods: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate pool with mix of bootstrap (real initial states) and synthetic records.

    Args:
        synth: Trained synthesizer model
        train_df: Training data for bootstrapping initial states
        feature_cols: Feature columns
        n_bootstrap: Number of records to bootstrap from real data
        n_synth: Number of purely synthetic records
        n_periods: Number of time periods per trajectory
        seed: Random seed

    Returns:
        DataFrame with blended pool
    """
    np.random.seed(seed)

    records = []
    pid = 0

    # Bootstrap records: real initial states + learned transitions
    if n_bootstrap > 0:
        init_states = [
            train_df[train_df['person_id'] == p].sort_values('period')[feature_cols].iloc[0].values
            for p in train_df['person_id'].unique()
        ]
        init_states = np.array(init_states)

        bootstrap_idx = np.random.choice(len(init_states), n_bootstrap, replace=True)

        for i in bootstrap_idx:
            state = init_states[i].copy()
            for t in range(n_periods):
                state = np.clip(np.nan_to_num(state, 0), 0, 1e10)
                records.append({
                    'person_id': pid,
                    'period': t,
                    'source': 'bootstrap',
                    **{col: float(state[j]) for j, col in enumerate(feature_cols)}
                })
                if t < n_periods - 1:
                    state = synth.transition_model.sample(state)
            pid += 1

    # Synthetic records: learned initial states + learned transitions
    if n_synth > 0:
        synth_df = synth.generate(n_synth, n_periods, seed=seed + 1)
        synth_df['source'] = 'synthetic'
        synth_df['person_id'] = synth_df['person_id'] + pid
        records.extend(synth_df.to_dict('records'))

    return pd.DataFrame(records)


def compute_trajectory_coverage(
    holdout_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    feature_cols: list,
    zero_cols: list,
    scaler: StandardScaler = None,
) -> tuple:
    """Compute trajectory coverage metrics.

    Args:
        holdout_df: Holdout data
        pool_df: Pool of bootstrap + synthetic records
        feature_cols: Feature columns
        zero_cols: Columns to add zero indicators for
        scaler: Pre-fitted scaler (optional)

    Returns:
        (mean_distance, median_distance, p90_distance)
    """
    eval_cols = feature_cols + [f'{c}_nz' for c in zero_cols]

    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nz'] = (df[col] > 0).astype(float)
        return df

    def to_matrix(df):
        return np.array([
            df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
            for pid in sorted(df['person_id'].unique())
        ])

    holdout_mat = to_matrix(augment(holdout_df))
    pool_mat = to_matrix(augment(pool_df))

    if scaler is None:
        scaler = StandardScaler().fit(pool_mat)

    holdout_scaled = scaler.transform(holdout_mat)
    pool_scaled = scaler.transform(pool_mat)

    nn_model = NearestNeighbors(n_neighbors=1).fit(pool_scaled)
    distances, _ = nn_model.kneighbors(holdout_scaled)
    distances = distances.flatten()

    return (
        float(np.mean(distances)),
        float(np.median(distances)),
        float(np.percentile(distances, 90)),
    )


def main(sample_frac: float = 1.0, n_periods: int = 6, train_frac: float = 0.2):
    print("=" * 70)
    print("BLENDING EXPERIMENT: Bootstrap vs Synthetic (20% Training Data)")
    print("=" * 70)
    print(f"\nUsing {train_frac:.0%} for training, {1-train_frac:.0%} for holdout")

    # Load model
    model_path = Path(__file__).parent / "sipp_synthesizer.pt"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run sipp_initial_state_model.py first to train the model.")
        return

    synth = FullSynthesizer.load(model_path)

    # Load and prepare data
    print("\nLoading SIPP...")
    sipp_raw = load_sipp(sample_frac=sample_frac)
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    zero_cols = ['total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']

    # Split into train/holdout with 20/80 split
    persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(persons)
    n_train = int(len(persons) * train_frac)
    train_persons, holdout_persons = persons[:n_train], persons[n_train:]

    train_df = sipp[sipp['person_id'].isin(train_persons)]
    holdout_df = sipp[sipp['person_id'].isin(holdout_persons)]

    # Filter to complete panels
    def filter_complete(df, n_periods):
        periods_per_person = df.groupby('person_id')['period'].nunique()
        complete = periods_per_person[periods_per_person >= n_periods].index
        df = df[df['person_id'].isin(complete)]
        df = df.sort_values(['person_id', 'period']).groupby('person_id').head(n_periods)
        return df

    train_df = filter_complete(train_df, n_periods)
    holdout_df = filter_complete(holdout_df, n_periods)

    n_train_persons = train_df['person_id'].nunique()
    n_holdout_persons = holdout_df['person_id'].nunique()

    print(f"Train: {n_train_persons} persons ({train_frac:.0%})")
    print(f"Holdout: {n_holdout_persons} persons ({1-train_frac:.0%})")

    # Baseline pool size = training size
    baseline_n = n_train_persons

    # Create scaler from training data
    def to_traj_matrix(df):
        eval_cols = feature_cols + [f'{c}_nz' for c in zero_cols]
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nz'] = (df[col] > 0).astype(float)
        return np.array([
            df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
            for pid in sorted(df['person_id'].unique())
        ])

    train_traj = to_traj_matrix(train_df)
    scaler = StandardScaler().fit(train_traj)

    results = []

    # =========================================================================
    # EXPERIMENT 1: Different blend ratios (same total n as training)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Blend Ratios (Total n = Training n)")
    print("=" * 70)

    blend_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ratio in blend_ratios:
        n_bootstrap = int(baseline_n * ratio)
        n_synth = baseline_n - n_bootstrap

        print(f"\n  Ratio {ratio:.0%} bootstrap: {n_bootstrap} bootstrap + {n_synth} synthetic = {baseline_n} total")

        pool_df = generate_bootstrap_pool(
            synth, train_df, feature_cols,
            n_bootstrap=n_bootstrap,
            n_synth=n_synth,
            n_periods=n_periods,
            seed=42,
        )

        mean_dist, median_dist, p90_dist = compute_trajectory_coverage(
            holdout_df, pool_df, feature_cols, zero_cols, scaler
        )

        print(f"    Mean distance: {mean_dist:.4f}")
        print(f"    Median distance: {median_dist:.4f}")
        print(f"    P90 distance: {p90_dist:.4f}")

        results.append({
            'experiment': 'blend_ratio',
            'train_frac': train_frac,
            'bootstrap_ratio': ratio,
            'n_bootstrap': n_bootstrap,
            'n_synth': n_synth,
            'total_n': n_bootstrap + n_synth,
            'mean_distance': mean_dist,
            'median_distance': median_dist,
            'p90_distance': p90_dist,
        })

    # =========================================================================
    # EXPERIMENT 2: Scaling up (100% bootstrap + extra synthetics)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Scaling Up (100% bootstrap + extra synthetics)")
    print("=" * 70)

    extra_synth_ratios = [0.0, 0.5, 1.0, 2.0, 4.0]

    for extra_ratio in extra_synth_ratios:
        n_bootstrap = baseline_n
        n_extra_synth = int(baseline_n * extra_ratio)
        total_n = n_bootstrap + n_extra_synth

        print(f"\n  Extra {extra_ratio:.0%}: {n_bootstrap} bootstrap + {n_extra_synth} synthetic = {total_n} total")

        pool_df = generate_bootstrap_pool(
            synth, train_df, feature_cols,
            n_bootstrap=n_bootstrap,
            n_synth=n_extra_synth,
            n_periods=n_periods,
            seed=42,
        )

        mean_dist, median_dist, p90_dist = compute_trajectory_coverage(
            holdout_df, pool_df, feature_cols, zero_cols, scaler
        )

        print(f"    Mean distance: {mean_dist:.4f}")
        print(f"    Median distance: {median_dist:.4f}")
        print(f"    P90 distance: {p90_dist:.4f}")

        results.append({
            'experiment': 'scale_up',
            'train_frac': train_frac,
            'extra_synth_ratio': extra_ratio,
            'n_bootstrap': n_bootstrap,
            'n_synth': n_extra_synth,
            'total_n': total_n,
            'mean_distance': mean_dist,
            'median_distance': median_dist,
            'p90_distance': p90_dist,
        })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    print("\n--- Experiment 1: Blend Ratios ---")
    exp1 = results_df[results_df['experiment'] == 'blend_ratio'].sort_values('bootstrap_ratio')
    print(f"{'Bootstrap %':<15} {'Total N':<10} {'Mean Dist':<12} {'Median Dist':<12} {'P90 Dist':<12}")
    print("-" * 60)
    for _, row in exp1.iterrows():
        print(f"{row['bootstrap_ratio']:.0%}            {row['total_n']:<10} {row['mean_distance']:<12.4f} {row['median_distance']:<12.4f} {row['p90_distance']:<12.4f}")

    best_blend = exp1.loc[exp1['mean_distance'].idxmin()]
    print(f"\nBest blend ratio: {best_blend['bootstrap_ratio']:.0%} (mean dist = {best_blend['mean_distance']:.4f})")

    print("\n--- Experiment 2: Scaling Up ---")
    exp2 = results_df[results_df['experiment'] == 'scale_up'].sort_values('extra_synth_ratio')
    print(f"{'Extra Synth %':<15} {'Total N':<10} {'Mean Dist':<12} {'Median Dist':<12} {'P90 Dist':<12}")
    print("-" * 60)
    for _, row in exp2.iterrows():
        print(f"{row['extra_synth_ratio']:.0%}            {row['total_n']:<10} {row['mean_distance']:<12.4f} {row['median_distance']:<12.4f} {row['p90_distance']:<12.4f}")

    best_scale = exp2.loc[exp2['mean_distance'].idxmin()]
    print(f"\nBest scaling: +{best_scale['extra_synth_ratio']:.0%} synthetics (mean dist = {best_scale['mean_distance']:.4f})")

    # Save results
    results_path = Path(__file__).parent / "blend_experiment_20pct_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    baseline_dist = exp1[exp1['bootstrap_ratio'] == 1.0]['mean_distance'].values[0]
    best_blend_dist = best_blend['mean_distance']
    best_scale_dist = best_scale['mean_distance']

    print(f"\n1. Baseline (100% bootstrap): mean distance = {baseline_dist:.4f}")
    print(f"2. Best blend ratio ({best_blend['bootstrap_ratio']:.0%}): mean distance = {best_blend_dist:.4f}")
    print(f"   Improvement: {(baseline_dist - best_blend_dist) / baseline_dist * 100:.1f}%")
    print(f"3. Best scaling (+{best_scale['extra_synth_ratio']:.0%}): mean distance = {best_scale_dist:.4f}")
    print(f"   Improvement: {(baseline_dist - best_scale_dist) / baseline_dist * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blending experiment with 20% training data")
    parser.add_argument("--sample-frac", type=float, default=1.0,
                        help="Fraction of SIPP data to use")
    parser.add_argument("--n-periods", type=int, default=6,
                        help="Number of time periods")
    parser.add_argument("--train-frac", type=float, default=0.2,
                        help="Fraction of data for training (default: 0.2)")
    args = parser.parse_args()

    main(sample_frac=args.sample_frac, n_periods=args.n_periods, train_frac=args.train_frac)
