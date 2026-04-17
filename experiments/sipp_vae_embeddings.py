"""SIPP integration with learned embeddings from TrajectoryVAE.

This demonstrates microplex's core capability:
1. Load real SIPP panel data
2. Train TrajectoryVAE to learn trajectory embeddings
3. Generate synthetic trajectories
4. Compute PRDC using learned embeddings (not raw features)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.eval.coverage import compute_prdc
from microplex.models.trajectory_vae import TrajectoryVAE


def load_sipp_panel(data_path: Path, max_individuals: int = 1000) -> pd.DataFrame:
    """Load actual SIPP 2022 panel data."""
    if not data_path.exists():
        raise FileNotFoundError(f"SIPP data not found at {data_path}")

    print("Loading SIPP 2022 data...")
    key_cols = ['SSUID', 'PNUM', 'MONTHCODE', 'SWAVE',
                'TAGE', 'ESEX', 'TPEARN', 'TNETWORTH']

    df = pd.read_csv(data_path, sep='|', usecols=key_cols, low_memory=False)

    # Create unique person ID
    df['person_id'] = df['SSUID'].astype(str) + '_' + df['PNUM'].astype(str)

    # Create time index (wave * 12 + month)
    df['period'] = (df['SWAVE'] - 1) * 12 + df['MONTHCODE'] - 1  # 0-indexed

    # Rename for consistency
    df = df.rename(columns={
        'TAGE': 'age',
        'TPEARN': 'income',
        'TNETWORTH': 'net_worth',
    })

    # Annualize income
    df['income'] = df['income'] * 12

    # Sample individuals if needed
    unique_persons = df['person_id'].unique()
    if len(unique_persons) > max_individuals:
        np.random.seed(42)
        sampled = np.random.choice(unique_persons, max_individuals, replace=False)
        df = df[df['person_id'].isin(sampled)]

    # Clean data - remove extreme outliers
    df = df.dropna(subset=['age', 'income', 'net_worth'])
    df = df[df['income'] >= 0]
    df = df[df['age'] >= 18]
    df = df[df['age'] <= 85]

    # Keep key columns
    df = df[['person_id', 'period', 'age', 'income', 'net_worth']]
    df = df.sort_values(['person_id', 'period']).reset_index(drop=True)

    # Re-encode person_id as integers
    person_map = {p: i for i, p in enumerate(df['person_id'].unique())}
    df['person_id'] = df['person_id'].map(person_map)

    print(f"Loaded {len(df['person_id'].unique())} individuals, {len(df)} person-months")
    print(f"Periods: {df['period'].min()} to {df['period'].max()}")

    return df


def main():
    print("=" * 60)
    print("microplex: SIPP VAE Embeddings Integration Test")
    print("=" * 60)
    print()

    # 1. Load data
    data_path = Path(__file__).parent.parent.parent / "popdgp" / "data" / "pu2022.csv"
    sipp_df = load_sipp_panel(data_path, max_individuals=500)

    print()
    print("Data statistics:")
    print(sipp_df[['age', 'income', 'net_worth']].describe().round(2))
    print()

    # Hold out some individuals for testing
    all_persons = sipp_df['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_persons)
    train_persons = all_persons[:int(len(all_persons) * 0.8)]
    test_persons = all_persons[int(len(all_persons) * 0.8):]

    train_df = sipp_df[sipp_df['person_id'].isin(train_persons)].copy()
    test_df = sipp_df[sipp_df['person_id'].isin(test_persons)].copy()

    print(f"Train: {len(train_persons)} individuals, {len(train_df)} rows")
    print(f"Test: {len(test_persons)} individuals, {len(test_df)} rows")
    print()

    # 2. Train VAE
    print("Training TrajectoryVAE...")
    vae = TrajectoryVAE(
        n_features=3,  # age, income, net_worth
        latent_dim=16,
        hidden_dim=64,
        n_layers=2,
    )

    vae.fit(train_df, epochs=20, verbose=True)
    print()

    # 3. Generate synthetic trajectories
    print("Generating synthetic trajectories...")
    n_periods = train_df['period'].nunique()
    synthetic = vae.generate(n=len(test_persons), T=n_periods, seed=123)
    print(f"Generated {synthetic.n_persons} individuals × {synthetic.n_periods} periods")
    print()

    # 4. Compute coverage using learned embeddings
    print("Computing PRDC with learned embeddings...")

    # Get embeddings
    real_embeddings = vae.encode(test_df, deterministic=True)
    synth_embeddings = vae.encode(synthetic.persons, deterministic=True)

    print(f"Real embeddings shape: {real_embeddings.shape}")
    print(f"Synth embeddings shape: {synth_embeddings.shape}")

    prdc = compute_prdc(real_embeddings, synth_embeddings, k=5)
    print()
    print("PRDC metrics (learned embeddings):")
    print(f"  Precision: {prdc.precision:.3f}")
    print(f"  Recall:    {prdc.recall:.3f}")
    print(f"  Density:   {prdc.density:.3f}")
    print(f"  Coverage:  {prdc.coverage:.3f}")
    print()

    # 5. Compare with raw feature space
    print("Comparing with raw feature space PRDC...")
    from sklearn.preprocessing import StandardScaler

    # Get cross-sectional snapshots (first period)
    test_first = test_df[test_df['period'] == test_df['period'].min()][['age', 'income', 'net_worth']].values
    synth_first = synthetic.persons[synthetic.persons['period'] == 0][['age', 'income', 'net_worth']].values

    scaler = StandardScaler()
    test_scaled = scaler.fit_transform(test_first)
    synth_scaled = scaler.transform(synth_first)

    prdc_raw = compute_prdc(test_scaled, synth_scaled, k=5)
    print("PRDC metrics (raw features):")
    print(f"  Precision: {prdc_raw.precision:.3f}")
    print(f"  Recall:    {prdc_raw.recall:.3f}")
    print(f"  Density:   {prdc_raw.density:.3f}")
    print(f"  Coverage:  {prdc_raw.coverage:.3f}")
    print()

    # 6. Analyze temporal dynamics
    print("Analyzing temporal dynamics...")

    # Check autocorrelation in synthetic data
    synth_df = synthetic.persons
    autocorrs = []
    for pid in synth_df['person_id'].unique()[:50]:
        person = synth_df[synth_df['person_id'] == pid].sort_values('period')
        if len(person) > 1:
            wealth = person['net_worth'].values
            if np.std(wealth) > 0:
                autocorr = np.corrcoef(wealth[:-1], wealth[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)

    if autocorrs:
        print(f"Synthetic wealth autocorrelation: {np.mean(autocorrs):.3f} (n={len(autocorrs)})")

    # Compare with real data
    real_autocorrs = []
    for pid in test_df['person_id'].unique()[:50]:
        person = test_df[test_df['person_id'] == pid].sort_values('period')
        if len(person) > 1:
            wealth = person['net_worth'].values
            if np.std(wealth) > 0:
                autocorr = np.corrcoef(wealth[:-1], wealth[1:])[0, 1]
                if not np.isnan(autocorr):
                    real_autocorrs.append(autocorr)

    if real_autocorrs:
        print(f"Real wealth autocorrelation:      {np.mean(real_autocorrs):.3f} (n={len(real_autocorrs)})")
    print()

    # 7. Show sample trajectories
    print("Sample synthetic trajectories:")
    for pid in synth_df['person_id'].unique()[:3]:
        person = synth_df[synth_df['person_id'] == pid].sort_values('period')
        wealth_start = person['net_worth'].iloc[0]
        wealth_end = person['net_worth'].iloc[-1]
        income_mean = person['income'].mean()
        print(f"  Person {pid}: net_worth ${wealth_start:,.0f} → ${wealth_end:,.0f}, "
              f"mean income ${income_mean:,.0f}")
    print()

    print("=" * 60)
    print("Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
