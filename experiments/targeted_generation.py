"""Targeted generation: Generate synthetics to match specific holdouts.

Instead of random generation, condition on the holdout's initial state.
"""

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


def main():
    print("=" * 70)
    print("TARGETED GENERATION: Condition on holdout's initial state")
    print("=" * 70)

    df = generate_panel(n_persons=500, T=24, seed=42)
    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    print("\nTraining VAE...")
    vae = TrajectoryVAE(n_features=3, latent_dim=16, hidden_dim=64, n_layers=2)
    vae.fit(train_df, epochs=50, verbose=False)

    # Hard holdouts from previous analysis
    hard_pids = [157, 258, 127, 240, 381]

    print(f"\n{'='*70}")
    print("APPROACH 1: Random generation (baseline)")
    print(f"{'='*70}")

    synth_random = vae.generate(n=500, T=24, seed=123)
    synth_df = synth_random.persons

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    feature_cols = ['age', 'income', 'net_worth']

    def to_matrix(df, pids=None):
        if pids is None:
            pids = sorted(df['person_id'].unique())
        matrices = []
        for pid in pids:
            person = df[df['person_id'] == pid].sort_values('period')
            matrices.append(person[feature_cols].values.flatten())
        return np.array(matrices)

    train_mat = to_matrix(train_df)
    scaler = StandardScaler().fit(train_mat)

    for pid in hard_pids:
        holdout_vec = to_matrix(holdout_df, [pid])
        synth_mat = to_matrix(synth_df)

        holdout_scaled = scaler.transform(holdout_vec)
        synth_scaled = scaler.transform(synth_mat)

        nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
        dist, _ = nn.kneighbors(holdout_scaled)
        print(f"  Person {pid}: distance = {dist[0,0]:.2f}")

    print(f"\n{'='*70}")
    print("APPROACH 2: Targeted - impute from holdout's t=0 state")
    print(f"{'='*70}")

    for pid in hard_pids:
        # Get holdout's initial state
        holdout_person = holdout_df[holdout_df['person_id'] == pid].sort_values('period')
        initial = holdout_person[holdout_person['period'] == 0][feature_cols].copy()

        # Generate multiple trajectories from this initial state
        n_samples = 50
        for s in range(n_samples):
            vae.impute(initial, n_samples=1)
            # This gives us imputed values, but we want full trajectories
            # Let's use reconstruct instead
            pass

        # Actually, let's encode the holdout and sample nearby in latent space
        holdout_emb = vae.encode(holdout_person.assign(person_id=0), deterministic=True)

        # Sample around this embedding
        float('inf')
        for s in range(100):
            # Perturb embedding
            noise = np.random.randn(*holdout_emb.shape) * 0.5
            holdout_emb + noise

            # Decode (this requires implementing decode in VAE)
            # For now, let's just search the existing synthetic pool

        # Simpler: find synthetics in same region of latent space
        holdout_emb = vae.encode(holdout_df[holdout_df['person_id'] == pid], deterministic=True)
        synth_embs = vae.encode(synth_df, deterministic=True)

        # Find closest in embedding space
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(holdout_emb, synth_embs)[0]
        top_synth_idx = np.argsort(sims)[-5:][::-1]

        # Get raw distances for these
        holdout_vec = to_matrix(holdout_df, [pid])
        holdout_scaled = scaler.transform(holdout_vec)

        best_raw_dist = float('inf')
        for synth_idx in top_synth_idx:
            synth_pid = sorted(synth_df['person_id'].unique())[synth_idx]
            synth_vec = to_matrix(synth_df, [synth_pid])
            synth_scaled_vec = scaler.transform(synth_vec)
            raw_dist = np.linalg.norm(holdout_scaled - synth_scaled_vec)
            if raw_dist < best_raw_dist:
                best_raw_dist = raw_dist

        print(f"  Person {pid}: best match in latent space = {best_raw_dist:.2f}")

    print(f"\n{'='*70}")
    print("APPROACH 3: Oversample - generate MORE synthetics")
    print(f"{'='*70}")

    for n_synth in [500, 2000, 5000]:
        synth = vae.generate(n=n_synth, T=24, seed=123)
        synth_mat = to_matrix(synth.persons)
        synth_scaled = scaler.transform(synth_mat)
        nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)

        print(f"\n  n={n_synth}:")
        for pid in hard_pids:
            holdout_vec = to_matrix(holdout_df, [pid])
            holdout_scaled = scaler.transform(holdout_vec)
            dist, _ = nn.kneighbors(holdout_scaled)
            print(f"    Person {pid}: {dist[0,0]:.2f}")


if __name__ == "__main__":
    main()
