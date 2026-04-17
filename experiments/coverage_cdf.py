"""Coverage CDF: Distance from each holdout to nearest synthetic.

No thresholds. Just the raw distribution of how close we get.
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


def get_nn_distances(real_emb: np.ndarray, synth_emb: np.ndarray) -> np.ndarray:
    """Distance from each real to its nearest synthetic."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_emb)
    synth_scaled = scaler.transform(synth_emb)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(real_scaled)
    return distances[:, 0]


def print_cdf(distances: np.ndarray, name: str):
    """Print CDF as ASCII."""
    sorted_d = np.sort(distances)
    n = len(sorted_d)

    print(f"\n{name}")
    print("  Distance to nearest synthetic (lower = better)")
    print(f"  {'Percentile':>10} {'Distance':>10}")
    print(f"  {'-'*10} {'-'*10}")

    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = int(n * p / 100)
        d = sorted_d[min(idx, n-1)]
        bar = "█" * max(1, int(40 * (1 - d / sorted_d[-1])))
        print(f"  {p:>9}% {d:>10.3f} {bar}")

    print(f"  {'max':>10} {sorted_d[-1]:>10.3f}")


def main():
    print("=" * 60)
    print("COVERAGE CDF: Distance to Nearest Synthetic")
    print("=" * 60)

    df = generate_panel(n_persons=500, T=24, seed=42)
    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    print("\nTrain: 400, Holdout: 100")

    # Train models
    print("\nTraining VAE...")
    vae = TrajectoryVAE(n_features=3, latent_dim=16, hidden_dim=64, n_layers=2)
    vae.fit(train_df, epochs=50, verbose=False)

    print("Training Transformer...")
    trans = TrajectoryTransformer(n_features=3, hidden_dim=64, n_heads=4, n_layers=2)
    trans.fit(train_df, epochs=50, verbose=False)

    # Compare at different sample sizes
    for n_synth in [100, 500]:
        print(f"\n{'='*60}")
        print(f"Synthetic population size: {n_synth}")
        print(f"{'='*60}")

        # VAE
        vae_synth = vae.generate(n=n_synth, T=24, seed=123)
        real_emb = vae.encode(holdout_df, deterministic=True)
        synth_emb = vae.encode(vae_synth.persons, deterministic=True)
        vae_distances = get_nn_distances(real_emb, synth_emb)
        print_cdf(vae_distances, f"VAE (n={n_synth})")

        # Transformer
        trans_synth = trans.generate(n=n_synth, T=24, seed=123)
        real_emb_t = trans.encode(holdout_df)
        synth_emb_t = trans.encode(trans_synth.persons)
        trans_distances = get_nn_distances(real_emb_t, synth_emb_t)
        print_cdf(trans_distances, f"Transformer (n={n_synth})")

        # Direct comparison
        print("\n  Head-to-head (median distance):")
        print(f"    VAE:         {np.median(vae_distances):.3f}")
        print(f"    Transformer: {np.median(trans_distances):.3f}")
        better = "VAE" if np.median(vae_distances) < np.median(trans_distances) else "Transformer"
        print(f"    Winner: {better}")


if __name__ == "__main__":
    main()
