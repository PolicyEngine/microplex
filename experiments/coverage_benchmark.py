"""Coverage benchmark: The ONLY metric that matters.

For each holdout record, is there a similar synthetic record?
That's it. That's the whole game.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.models.trajectory_transformer import TrajectoryTransformer
from microplex.models.trajectory_vae import TrajectoryVAE


def generate_panel(n_persons: int = 500, T: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generate realistic panel data."""
    np.random.seed(seed)
    records = []
    for pid in range(n_persons):
        age = np.random.randint(25, 60)
        income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000))
        wealth = np.random.lognormal(10, 1.5)

        for t in range(T):
            records.append({
                'person_id': pid,
                'period': t,
                'age': age + t / 12,
                'income': max(0, income * (1 + np.random.normal(0.02/12, 0.1))),
                'net_worth': wealth * (1 + np.random.normal(0.05/12, 0.02)),
            })
            income = records[-1]['income']
            wealth = records[-1]['net_worth']
    return pd.DataFrame(records)


def coverage_at_k(real_emb: np.ndarray, synth_emb: np.ndarray, k: int = 1) -> float:
    """What fraction of real records have a synthetic neighbor within k-NN distance?"""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_emb)
    synth_scaled = scaler.transform(synth_emb)

    # For each real, find distance to nearest synthetic
    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(real_scaled)

    # Coverage threshold: distance to k-th nearest real neighbor
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real_scaled)
    real_distances, _ = nn_real.kneighbors(real_scaled)
    threshold = np.median(real_distances[:, k])  # Use median k-NN distance as threshold

    covered = (distances[:, 0] <= threshold).mean()
    return covered


def main():
    print("=" * 60)
    print("COVERAGE BENCHMARK")
    print("Goal: For each holdout, is there a similar synthetic?")
    print("=" * 60)
    print()

    # Generate data
    df = generate_panel(n_persons=500, T=24, seed=42)

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_persons = persons[:400]
    holdout_persons = persons[400:]

    train_df = df[df['person_id'].isin(train_persons)]
    holdout_df = df[df['person_id'].isin(holdout_persons)]

    print(f"Train: {len(train_persons)} individuals")
    print(f"Holdout: {len(holdout_persons)} individuals")
    print()

    results = {}

    # VAE
    print("Training VAE...")
    vae = TrajectoryVAE(n_features=3, latent_dim=16, hidden_dim=64, n_layers=2)
    vae.fit(train_df, epochs=50, verbose=False)

    # Generate MORE synthetic than holdout to maximize coverage
    for n_synth in [100, 200, 500, 1000]:
        synth = vae.generate(n=n_synth, T=24, seed=123)

        real_emb = vae.encode(holdout_df, deterministic=True)
        synth_emb = vae.encode(synth.persons, deterministic=True)

        cov = coverage_at_k(real_emb, synth_emb, k=5)
        results[f'VAE (n={n_synth})'] = cov
        print(f"  VAE n={n_synth}: {cov:.1%} holdout coverage")

    print()

    # Transformer
    print("Training Transformer...")
    trans = TrajectoryTransformer(n_features=3, hidden_dim=64, n_heads=4, n_layers=2)
    trans.fit(train_df, epochs=50, verbose=False)

    for n_synth in [100, 200, 500, 1000]:
        synth = trans.generate(n=n_synth, T=24, seed=123)

        real_emb = trans.encode(holdout_df)
        synth_emb = trans.encode(synth.persons)

        cov = coverage_at_k(real_emb, synth_emb, k=5)
        results[f'Transformer (n={n_synth})'] = cov
        print(f"  Transformer n={n_synth}: {cov:.1%} holdout coverage")

    print()
    print("=" * 60)
    print("RESULTS: Holdout Coverage")
    print("=" * 60)
    for name, cov in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(cov * 40)
        print(f"  {name:25s} {cov:6.1%} {bar}")


if __name__ == "__main__":
    main()
