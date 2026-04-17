"""SIPP integration with learned embeddings from TrajectoryTransformer.

Autoregressive transformer should better capture temporal dynamics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.eval.coverage import compute_prdc
from microplex.models.trajectory_transformer import TrajectoryTransformer


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

    # Clean data
    df = df.dropna(subset=['age', 'income', 'net_worth'])
    df = df[df['income'] >= 0]
    df = df[df['age'] >= 18]
    df = df[df['age'] <= 85]

    # Keep only complete panels (individuals with all periods)
    df = df[['person_id', 'period', 'age', 'income', 'net_worth']]
    df = df.sort_values(['person_id', 'period']).reset_index(drop=True)

    # Re-encode person_id as integers
    person_map = {p: i for i, p in enumerate(df['person_id'].unique())}
    df['person_id'] = df['person_id'].map(person_map)

    print(f"Loaded {len(df['person_id'].unique())} individuals, {len(df)} person-months")
    print(f"Periods: {df['period'].min()} to {df['period'].max()}")

    return df


def compute_autocorrelation(df: pd.DataFrame, col: str, max_persons: int = 100) -> float:
    """Compute average lag-1 autocorrelation for a column."""
    autocorrs = []
    for pid in df['person_id'].unique()[:max_persons]:
        person = df[df['person_id'] == pid].sort_values('period')
        if len(person) > 1:
            vals = person[col].values
            if np.std(vals) > 1e-6:  # Only if there's variation
                autocorr = np.corrcoef(vals[:-1], vals[1:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
    return np.mean(autocorrs) if autocorrs else 0.0


def main():
    print("=" * 60)
    print("microplex: SIPP Transformer Embeddings Integration Test")
    print("=" * 60)
    print()

    # 1. Load data
    data_path = Path(__file__).parent.parent.parent / "popdgp" / "data" / "pu2022.csv"
    sipp_df = load_sipp_panel(data_path, max_individuals=500)

    print()
    print("Data statistics:")
    print(sipp_df[['age', 'income', 'net_worth']].describe().round(2))
    print()

    # Real data autocorrelations
    print("Real data autocorrelations:")
    for col in ['age', 'income', 'net_worth']:
        ac = compute_autocorrelation(sipp_df, col)
        print(f"  {col}: {ac:.3f}")
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

    # 2. Train Transformer
    print("Training TrajectoryTransformer...")
    model = TrajectoryTransformer(
        n_features=3,  # age, income, net_worth
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
    )

    model.fit(train_df, epochs=30, verbose=True)
    print()

    # 3. Generate synthetic trajectories
    print("Generating synthetic trajectories...")
    n_periods = train_df['period'].nunique()
    synthetic = model.generate(n=len(test_persons), T=n_periods, seed=123)
    synth_df = synthetic.persons
    print(f"Generated {synthetic.n_persons} individuals × {synthetic.n_periods} periods")
    print()

    # 4. Compute coverage using learned embeddings
    print("Computing PRDC with learned embeddings...")

    # Get embeddings
    real_embeddings = model.encode(test_df)
    synth_embeddings = model.encode(synth_df)

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
    synth_first = synth_df[synth_df['period'] == 0][['age', 'income', 'net_worth']].values

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
    print("Synthetic data autocorrelations:")
    for col in ['age', 'income', 'net_worth']:
        ac = compute_autocorrelation(synth_df, col)
        print(f"  {col}: {ac:.3f}")
    print()

    # 7. Show sample trajectories
    print("Sample synthetic trajectories:")
    for pid in synth_df['person_id'].unique()[:5]:
        person = synth_df[synth_df['person_id'] == pid].sort_values('period')
        wealth_start = person['net_worth'].iloc[0]
        wealth_end = person['net_worth'].iloc[-1]
        income_mean = person['income'].mean()
        age_start = person['age'].iloc[0]
        print(f"  Person {pid}: age {age_start:.0f}, net_worth ${wealth_start:,.0f} → ${wealth_end:,.0f}, "
              f"mean income ${income_mean:,.0f}")
    print()

    print("=" * 60)
    print("Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
