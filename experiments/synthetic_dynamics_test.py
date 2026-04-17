"""Test VAE and Transformer on synthetic data with known dynamics.

Since SIPP data lacks true temporal variation (age/wealth constant within person),
we create synthetic data with explicit dynamics to verify models capture them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.eval.coverage import compute_prdc
from microplex.models.trajectory_transformer import TrajectoryTransformer
from microplex.models.trajectory_vae import TrajectoryVAE


def generate_dynamic_panel(n_persons: int = 500, T: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generate panel data with known dynamics.

    Dynamics:
    - Age: increases by 1/12 per month
    - Income: AR(1) with growth trend, correlated with age
    - Wealth: accumulates from savings (income - consumption)
    """
    np.random.seed(seed)

    records = []
    for pid in range(n_persons):
        # Initial conditions
        age_start = np.random.randint(25, 60)
        income_start = 30000 + 1000 * (age_start - 25) + np.random.normal(0, 10000)
        income_start = max(0, income_start)
        wealth_start = np.random.lognormal(10, 1.5)

        age = age_start
        income = income_start
        wealth = wealth_start

        for t in range(T):
            records.append({
                'person_id': pid,
                'period': t,
                'age': age,
                'income': income,
                'net_worth': wealth,
            })

            # Dynamics
            age += 1 / 12

            # Income: AR(1) with growth
            income_growth = 0.02 / 12  # 2% annual growth
            income_persistence = 0.9
            income_shock = np.random.normal(0, income * 0.1)
            income = income * income_persistence + income * (1 - income_persistence) * (1 + income_growth) + income_shock
            income = max(0, income)

            # Wealth: accumulates with savings rate
            savings_rate = 0.1 + 0.005 * (age - 25)  # Savings rate increases with age
            savings = income / 12 * savings_rate
            wealth_return = 0.05 / 12  # 5% annual return
            wealth = wealth * (1 + wealth_return) + savings + np.random.normal(0, wealth * 0.02)

    return pd.DataFrame(records)


def compute_autocorrelation(df: pd.DataFrame, col: str, max_persons: int = 100) -> float:
    """Compute average lag-1 autocorrelation."""
    autocorrs = []
    for pid in df['person_id'].unique()[:max_persons]:
        person = df[df['person_id'] == pid].sort_values('period')
        if len(person) > 1:
            vals = person[col].values
            if np.std(vals) > 1e-6:
                autocorr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
    return np.mean(autocorrs) if autocorrs else 0.0


def main():
    print("=" * 70)
    print("microplex: Synthetic Dynamics Test (VAE vs Transformer)")
    print("=" * 70)
    print()

    # 1. Generate data with known dynamics
    print("1. Generating synthetic panel with dynamics...")
    full_df = generate_dynamic_panel(n_persons=500, T=24, seed=42)
    print(f"   Generated {full_df['person_id'].nunique()} individuals × {full_df['period'].nunique()} periods")
    print()

    print("   Data statistics:")
    print(full_df[['age', 'income', 'net_worth']].describe().round(2))
    print()

    print("   True data autocorrelations:")
    for col in ['age', 'income', 'net_worth']:
        ac = compute_autocorrelation(full_df, col)
        print(f"     {col}: {ac:.3f}")
    print()

    # Split train/test
    all_persons = full_df['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_persons)
    train_persons = all_persons[:400]
    test_persons = all_persons[400:]

    train_df = full_df[full_df['person_id'].isin(train_persons)].copy()
    test_df = full_df[full_df['person_id'].isin(test_persons)].copy()

    print(f"   Train: {len(train_persons)} individuals")
    print(f"   Test: {len(test_persons)} individuals")
    print()

    # 2. Train VAE
    print("2. Training TrajectoryVAE...")
    vae = TrajectoryVAE(
        n_features=3,
        latent_dim=16,
        hidden_dim=64,
        n_layers=2,
    )
    vae.fit(train_df, epochs=50, verbose=True)
    print()

    # 3. Generate and evaluate VAE
    print("3. Evaluating VAE...")
    vae_synth = vae.generate(n=100, T=24, seed=123)
    vae_df = vae_synth.persons

    print("   VAE synthetic autocorrelations:")
    for col in ['age', 'income', 'net_worth']:
        ac = compute_autocorrelation(vae_df, col)
        print(f"     {col}: {ac:.3f}")

    # PRDC with embeddings
    real_emb = vae.encode(test_df, deterministic=True)
    synth_emb = vae.encode(vae_df, deterministic=True)
    prdc = compute_prdc(real_emb, synth_emb, k=5)
    print(f"   VAE PRDC (embeddings): precision={prdc.precision:.3f}, coverage={prdc.coverage:.3f}")
    print()

    # 4. Train Transformer
    print("4. Training TrajectoryTransformer...")
    transformer = TrajectoryTransformer(
        n_features=3,
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
    )
    transformer.fit(train_df, epochs=50, verbose=True)
    print()

    # 5. Generate and evaluate Transformer
    print("5. Evaluating Transformer...")
    trans_synth = transformer.generate(n=100, T=24, seed=123)
    trans_df = trans_synth.persons

    print("   Transformer synthetic autocorrelations:")
    for col in ['age', 'income', 'net_worth']:
        ac = compute_autocorrelation(trans_df, col)
        print(f"     {col}: {ac:.3f}")

    # PRDC with embeddings
    real_emb_t = transformer.encode(test_df)
    synth_emb_t = transformer.encode(trans_df)
    prdc_t = compute_prdc(real_emb_t, synth_emb_t, k=5)
    print(f"   Transformer PRDC (embeddings): precision={prdc_t.precision:.3f}, coverage={prdc_t.coverage:.3f}")
    print()

    # 6. Sample trajectories
    print("6. Sample trajectories...")
    print("\n   Real:")
    for pid in test_df['person_id'].unique()[:3]:
        person = test_df[test_df['person_id'] == pid].sort_values('period')
        print(f"     Person {pid}: net_worth ${person['net_worth'].iloc[0]:,.0f} → ${person['net_worth'].iloc[-1]:,.0f}")

    print("\n   VAE Synthetic:")
    for pid in vae_df['person_id'].unique()[:3]:
        person = vae_df[vae_df['person_id'] == pid].sort_values('period')
        print(f"     Person {pid}: net_worth ${person['net_worth'].iloc[0]:,.0f} → ${person['net_worth'].iloc[-1]:,.0f}")

    print("\n   Transformer Synthetic:")
    for pid in trans_df['person_id'].unique()[:3]:
        person = trans_df[trans_df['person_id'] == pid].sort_values('period')
        print(f"     Person {pid}: net_worth ${person['net_worth'].iloc[0]:,.0f} → ${person['net_worth'].iloc[-1]:,.0f}")
    print()

    print("=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
