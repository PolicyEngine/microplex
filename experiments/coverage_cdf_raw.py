"""Coverage CDF in RAW feature space - no embeddings.

Measures actual similarity: how close is each holdout to nearest synthetic
in (age, income, net_worth) space?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.models.trajectory_transformer import TrajectoryTransformer
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


def to_trajectory_matrix(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Convert panel to (n_persons, T * n_features) matrix for NN."""
    persons = df.groupby('person_id')
    matrices = []
    for pid, group in persons:
        group = group.sort_values('period')
        traj = group[feature_cols].values.flatten()  # Flatten T x F into single vector
        matrices.append(traj)
    return np.array(matrices)


def get_nn_distances_raw(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                         train_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Distance from each real trajectory to nearest synthetic in raw space.

    Scaler is fit on TRAINING data only.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    train_mat = to_trajectory_matrix(train_df, feature_cols)
    real_mat = to_trajectory_matrix(real_df, feature_cols)
    synth_mat = to_trajectory_matrix(synth_df, feature_cols)

    scaler = StandardScaler()
    scaler.fit(train_mat)  # Fit on training only
    real_scaled = scaler.transform(real_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(real_scaled)
    return distances[:, 0]


def print_cdf(distances: np.ndarray, name: str):
    sorted_d = np.sort(distances)
    n = len(sorted_d)
    print(f"\n{name}")
    print(f"  {'Percentile':>10} {'Distance':>10}")
    print(f"  {'-'*10} {'-'*10}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = int(n * p / 100)
        d = sorted_d[min(idx, n-1)]
        print(f"  {p:>9}% {d:>10.3f}")
    print(f"  {'max':>10} {sorted_d[-1]:>10.3f}")


def main():
    print("=" * 60)
    print("COVERAGE CDF in RAW FEATURE SPACE")
    print("Distance = ||real_trajectory - synth_trajectory||")
    print("=" * 60)

    df = generate_panel(n_persons=500, T=24, seed=42)
    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    feature_cols = ['age', 'income', 'net_worth']
    print(f"\nFeatures: {feature_cols}")
    print("Trajectory length: 24 periods")
    print(f"Vector dimension: {24 * 3} = 72")

    # Train models
    print("\nTraining VAE...")
    vae = TrajectoryVAE(n_features=3, latent_dim=16, hidden_dim=64, n_layers=2)
    vae.fit(train_df, epochs=50, verbose=False)

    print("Training Transformer...")
    trans = TrajectoryTransformer(n_features=3, hidden_dim=64, n_heads=4, n_layers=2)
    trans.fit(train_df, epochs=50, verbose=False)

    for n_synth in [100, 500]:
        print(f"\n{'='*60}")
        print(f"Synthetic population: {n_synth}")
        print(f"{'='*60}")

        # VAE
        vae_synth = vae.generate(n=n_synth, T=24, seed=123)
        vae_dist = get_nn_distances_raw(holdout_df, vae_synth.persons, feature_cols)
        print_cdf(vae_dist, f"VAE (n={n_synth})")

        # Transformer
        trans_synth = trans.generate(n=n_synth, T=24, seed=123)
        trans_dist = get_nn_distances_raw(holdout_df, trans_synth.persons, feature_cols)
        print_cdf(trans_dist, f"Transformer (n={n_synth})")

        print("\n  Median distance (lower = better):")
        print(f"    VAE:         {np.median(vae_dist):.3f}")
        print(f"    Transformer: {np.median(trans_dist):.3f}")


if __name__ == "__main__":
    main()
