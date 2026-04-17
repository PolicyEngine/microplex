"""Inspect the hardest holdouts - why can't we match them?"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.models.trajectory_vae import TrajectoryVAE


def generate_panel(n_persons: int = 500, T: int = 24, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []
    for pid in range(n_persons):
        age = np.random.randint(25, 60)
        income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000))
        wealth = np.random.lognormal(10, 1.5)
        for t in range(T):
            records.append({
                'person_id': pid, 'period': t,
                'age': age + t / 12,
                'income': max(0, income * (1 + np.random.normal(0.02/12, 0.1))),
                'net_worth': wealth * (1 + np.random.normal(0.05/12, 0.02)),
            })
            income, wealth = records[-1]['income'], records[-1]['net_worth']
    return pd.DataFrame(records)


def get_person_summary(df: pd.DataFrame, pid: int) -> dict:
    """Get summary stats for a person's trajectory."""
    person = df[df['person_id'] == pid].sort_values('period')
    return {
        'person_id': pid,
        'age_start': person['age'].iloc[0],
        'age_end': person['age'].iloc[-1],
        'income_mean': person['income'].mean(),
        'income_std': person['income'].std(),
        'wealth_start': person['net_worth'].iloc[0],
        'wealth_end': person['net_worth'].iloc[-1],
        'wealth_growth': person['net_worth'].iloc[-1] / person['net_worth'].iloc[0] - 1,
    }


def main():
    print("=" * 70)
    print("INSPECTING HARD HOLDOUTS")
    print("=" * 70)

    df = generate_panel(n_persons=500, T=24, seed=42)
    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    # Train VAE
    print("\nTraining VAE...")
    vae = TrajectoryVAE(n_features=3, latent_dim=16, hidden_dim=64, n_layers=2)
    vae.fit(train_df, epochs=50, verbose=False)

    # Generate synthetic
    n_synth = 500
    synth = vae.generate(n=n_synth, T=24, seed=123)
    synth_df = synth.persons

    # Get distances in raw feature space
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    feature_cols = ['age', 'income', 'net_worth']

    def to_matrix(df):
        matrices = []
        for pid in sorted(df['person_id'].unique()):
            person = df[df['person_id'] == pid].sort_values('period')
            matrices.append(person[feature_cols].values.flatten())
        return np.array(matrices)

    train_mat = to_matrix(train_df)
    holdout_mat = to_matrix(holdout_df)
    synth_mat = to_matrix(synth_df)

    scaler = StandardScaler().fit(train_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, indices = nn.kneighbors(holdout_scaled)

    holdout_pids = sorted(holdout_df['person_id'].unique())

    # Find hardest holdouts
    hard_idx = np.argsort(distances[:, 0])[-5:][::-1]  # Top 5 hardest

    print(f"\n{'='*70}")
    print("TOP 5 HARDEST HOLDOUTS (largest distance to nearest synthetic)")
    print(f"{'='*70}")

    for rank, idx in enumerate(hard_idx):
        holdout_pid = holdout_pids[idx]
        nearest_synth_idx = indices[idx, 0]
        nearest_synth_pid = sorted(synth_df['person_id'].unique())[nearest_synth_idx]
        dist = distances[idx, 0]

        holdout_summary = get_person_summary(holdout_df, holdout_pid)
        synth_summary = get_person_summary(synth_df, nearest_synth_pid)

        print(f"\n#{rank+1} Distance: {dist:.2f}")
        print(f"  HOLDOUT (person {holdout_pid}):")
        print(f"    Age: {holdout_summary['age_start']:.0f} → {holdout_summary['age_end']:.0f}")
        print(f"    Income: ${holdout_summary['income_mean']:,.0f} (std ${holdout_summary['income_std']:,.0f})")
        print(f"    Wealth: ${holdout_summary['wealth_start']:,.0f} → ${holdout_summary['wealth_end']:,.0f} "
              f"({holdout_summary['wealth_growth']:+.0%})")

        print(f"  NEAREST SYNTHETIC (person {nearest_synth_pid}):")
        print(f"    Age: {synth_summary['age_start']:.0f} → {synth_summary['age_end']:.0f}")
        print(f"    Income: ${synth_summary['income_mean']:,.0f} (std ${synth_summary['income_std']:,.0f})")
        print(f"    Wealth: ${synth_summary['wealth_start']:,.0f} → ${synth_summary['wealth_end']:,.0f} "
              f"({synth_summary['wealth_growth']:+.0%})")

        # What's the gap?
        print("  GAP:")
        print(f"    Age: {abs(holdout_summary['age_start'] - synth_summary['age_start']):.0f} years")
        print(f"    Income: ${abs(holdout_summary['income_mean'] - synth_summary['income_mean']):,.0f}")
        print(f"    Wealth: ${abs(holdout_summary['wealth_start'] - synth_summary['wealth_start']):,.0f}")

    # Distribution analysis
    print(f"\n{'='*70}")
    print("DISTRIBUTION ANALYSIS: Where do hard holdouts fall?")
    print(f"{'='*70}")

    holdout_summaries = [get_person_summary(holdout_df, pid) for pid in holdout_pids]
    holdout_stats = pd.DataFrame(holdout_summaries)

    train_summaries = [get_person_summary(train_df, pid) for pid in sorted(train_df['person_id'].unique())]
    train_stats = pd.DataFrame(train_summaries)

    print("\nTraining set ranges:")
    for col in ['age_start', 'income_mean', 'wealth_start']:
        print(f"  {col}: {train_stats[col].min():.0f} - {train_stats[col].max():.0f} "
              f"(mean {train_stats[col].mean():.0f})")

    print("\nHard holdouts:")
    hard_pids = [holdout_pids[i] for i in hard_idx]
    hard_stats = holdout_stats[holdout_stats['person_id'].isin(hard_pids)]
    for col in ['age_start', 'income_mean', 'wealth_start']:
        values = hard_stats[col].values
        print(f"  {col}: {values}")

    # Check if hard holdouts are outliers
    print("\nAre hard holdouts outliers?")
    for col in ['income_mean', 'wealth_start']:
        train_p99 = train_stats[col].quantile(0.99)
        train_p01 = train_stats[col].quantile(0.01)
        for i, pid in enumerate(hard_pids):
            val = holdout_stats[holdout_stats['person_id'] == pid][col].values[0]
            if val > train_p99:
                print(f"  Person {pid}: {col}=${val:,.0f} > p99=${train_p99:,.0f} (OUTLIER)")
            elif val < train_p01:
                print(f"  Person {pid}: {col}=${val:,.0f} < p01=${train_p01:,.0f} (OUTLIER)")


if __name__ == "__main__":
    main()
