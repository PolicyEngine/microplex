"""Coverage with zero-inflation indicators.

Instead of modeling ZI separately, augment the feature space with binary
"is_nonzero" indicators. This keeps everything unified under coverage.

For each feature that can be zero (like dividend_income):
  - Original: [dividend_income]
  - Augmented: [dividend_income, has_dividend_income]

The coverage metric then captures whether synthetics match the zero/nonzero
pattern of holdouts.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex.models.trajectory_vae import TrajectoryVAE


def generate_panel_with_zeros(n_persons: int = 500, T: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generate panel with realistic zero-inflation patterns."""
    np.random.seed(seed)
    records = []
    for pid in range(n_persons):
        age = np.random.randint(25, 60)
        # Base income with unemployment risk
        employed = np.random.random() > 0.15  # 15% unemployed
        income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000)) if employed else 0

        # Wealth - can be zero for young/poor
        has_wealth = np.random.random() > 0.3  # 30% have no wealth
        wealth = np.random.lognormal(10, 1.5) if has_wealth else 0

        # Dividend income - most have zero
        has_dividends = np.random.random() > 0.7  # Only 30% have dividends
        dividend_base = np.random.lognormal(7, 1) if has_dividends else 0

        for t in range(T):
            # Employment can change
            if np.random.random() < 0.02:
                employed = not employed
            if employed and income == 0:
                income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000))
            elif not employed:
                income = 0
            else:
                income = max(0, income * (1 + np.random.normal(0.02/12, 0.1)))

            # Wealth evolution
            if wealth > 0:
                wealth = wealth * (1 + np.random.normal(0.05/12, 0.02))

            # Dividend income
            dividend = dividend_base * (1 + np.random.normal(0, 0.1)) if dividend_base > 0 else 0

            records.append({
                'person_id': pid, 'period': t,
                'age': age + t / 12,
                'income': income,
                'net_worth': wealth,
                'dividend_income': max(0, dividend),
            })
    return pd.DataFrame(records)


def augment_with_zero_indicators(df: pd.DataFrame, zero_cols: list) -> pd.DataFrame:
    """Add binary is_nonzero columns for specified features."""
    df = df.copy()
    for col in zero_cols:
        df[f'{col}_nonzero'] = (df[col] > 0).astype(float)
    return df


def to_matrix(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Convert panel to (n_persons, T * n_features) matrix."""
    matrices = []
    for pid in sorted(df['person_id'].unique()):
        person = df[df['person_id'] == pid].sort_values('period')
        matrices.append(person[feature_cols].values.flatten())
    return np.array(matrices)


def get_coverage_distances(holdout_df, synth_df, train_df, feature_cols):
    """Get NN distances from holdouts to synthetics."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    train_mat = to_matrix(train_df, feature_cols)
    holdout_mat = to_matrix(holdout_df, feature_cols)
    synth_mat = to_matrix(synth_df, feature_cols)

    scaler = StandardScaler().fit(train_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(holdout_scaled)
    return distances[:, 0]


class QuantileDNN(nn.Module):
    """Quantile regression DNN for comparison."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_quantiles: int = 19):
        super().__init__()
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * n_quantiles),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.output_dim, self.n_quantiles)

    def loss(self, x, target):
        q_pred = self.forward(x)
        errors = target.unsqueeze(-1) - q_pred
        ql = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        ).mean()
        return ql

    def sample(self, x):
        with torch.no_grad():
            q = self.forward(x)
            q_sorted = torch.sort(q, dim=-1)[0]
            u = torch.rand(x.shape[0], self.output_dim, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            samples = q_sorted.gather(-1, idx).squeeze(-1)
            return samples


def train_qdnn_panel(train_df, feature_cols, epochs=100):
    """Train QDNN for panel synthesis."""
    n_features = len(feature_cols)
    train_df.groupby('person_id').size().iloc[0]

    # Prepare training data: predict t+1 from t
    X_list, Y_list = [], []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[feature_cols].values
        for t in range(len(values) - 1):
            X_list.append(values[t])
            Y_list.append(values[t + 1])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    # Normalize
    X_mean, X_std = X.mean(0), X.std(0) + 1e-6
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-6
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    model = QuantileDNN(n_features, n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(X_norm, Y_norm)
        loss.backward()
        optimizer.step()

    return model, X_mean, X_std, Y_mean, Y_std


def generate_qdnn_panel(model, train_df, feature_cols, n_synth, T,
                        X_mean, X_std, Y_mean, Y_std, seed=42):
    """Generate synthetic panel using trained QDNN."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sample initial states from training data
    init_states = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        init_states.append(person[feature_cols].iloc[0].values)
    init_states = np.array(init_states)

    records = []
    for pid in range(n_synth):
        # Sample random initial state
        init_idx = np.random.randint(len(init_states))
        state = init_states[init_idx].copy()

        for t in range(T):
            records.append({
                'person_id': pid,
                'period': t,
                **{col: state[i] for i, col in enumerate(feature_cols)}
            })

            if t < T - 1:
                # Predict next state
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                x_norm = (x - X_mean) / X_std
                next_norm = model.sample(x_norm)
                next_state = (next_norm * Y_std + Y_mean).numpy()[0]
                state = np.maximum(0, next_state)  # Enforce non-negative

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("COVERAGE WITH ZERO INDICATORS")
    print("=" * 70)

    # Generate data with realistic zero patterns
    df = generate_panel_with_zeros(n_persons=500, T=12, seed=42)

    # Check zero rates
    print("\nZero rates in original data:")
    for col in ['income', 'net_worth', 'dividend_income']:
        zero_rate = (df[col] == 0).mean()
        print(f"  {col}: {zero_rate:.1%} zeros")

    # Split
    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    # Define feature sets
    base_features = ['age', 'income', 'net_worth', 'dividend_income']
    zero_cols = ['income', 'net_worth', 'dividend_income']

    # Augment with zero indicators
    train_aug = augment_with_zero_indicators(train_df, zero_cols)
    holdout_aug = augment_with_zero_indicators(holdout_df, zero_cols)

    augmented_features = base_features + [f'{c}_nonzero' for c in zero_cols]

    print(f"\nBase features: {base_features}")
    print(f"Augmented features: {augmented_features}")

    # Train VAE on base features (doesn't know about zeros)
    print("\n" + "=" * 70)
    print("Training VAE (base features only)...")
    print("=" * 70)
    vae = TrajectoryVAE(n_features=4, latent_dim=16, hidden_dim=64, n_layers=2)
    vae.fit(train_df, epochs=50, verbose=False)

    # Train QDNN on augmented features
    print("\nTraining QDNN (augmented features)...")
    qdnn, X_mean, X_std, Y_mean, Y_std = train_qdnn_panel(
        train_aug, augmented_features, epochs=100
    )

    # Generate synthetics
    n_synth = 500
    print(f"\nGenerating {n_synth} synthetics...")

    vae_synth = vae.generate(n=n_synth, T=12, seed=123)
    vae_synth_df = vae_synth.persons

    qdnn_synth_df = generate_qdnn_panel(
        qdnn, train_aug, augmented_features, n_synth, 12,
        X_mean, X_std, Y_mean, Y_std, seed=123
    )

    # Binarize the indicator columns in QDNN output
    for col in zero_cols:
        qdnn_synth_df[f'{col}_nonzero'] = (qdnn_synth_df[f'{col}_nonzero'] > 0.5).astype(float)
        # Also zero out the value if indicator says zero
        qdnn_synth_df.loc[qdnn_synth_df[f'{col}_nonzero'] == 0, col] = 0

    # Check zero rates in synthetics
    print("\nZero rates in synthetics:")
    print("  VAE:")
    for col in zero_cols:
        zero_rate = (vae_synth_df[col] <= 0.01).mean()  # VAE won't produce exact zeros
        print(f"    {col}: {zero_rate:.1%} near-zero")

    print("  QDNN (with indicators):")
    for col in zero_cols:
        zero_rate = (qdnn_synth_df[col] == 0).mean()
        print(f"    {col}: {zero_rate:.1%} zeros")

    # Augment VAE output for fair comparison
    vae_synth_aug = augment_with_zero_indicators(vae_synth_df, zero_cols)

    # Compare coverage with and without zero indicators
    print("\n" + "=" * 70)
    print("COVERAGE COMPARISON")
    print("=" * 70)

    print("\n1. Base features only (age, income, wealth, dividends):")
    vae_dist_base = get_coverage_distances(holdout_df, vae_synth_df, train_df, base_features)
    qdnn_dist_base = get_coverage_distances(holdout_df, qdnn_synth_df, train_df, base_features)
    print(f"   VAE:  median={np.median(vae_dist_base):.2f}, p90={np.percentile(vae_dist_base, 90):.2f}")
    print(f"   QDNN: median={np.median(qdnn_dist_base):.2f}, p90={np.percentile(qdnn_dist_base, 90):.2f}")

    print("\n2. With zero indicators (base + has_income, has_wealth, has_dividends):")
    vae_dist_aug = get_coverage_distances(holdout_aug, vae_synth_aug, train_aug, augmented_features)
    qdnn_dist_aug = get_coverage_distances(holdout_aug, qdnn_synth_df, train_aug, augmented_features)
    print(f"   VAE:  median={np.median(vae_dist_aug):.2f}, p90={np.percentile(vae_dist_aug, 90):.2f}")
    print(f"   QDNN: median={np.median(qdnn_dist_aug):.2f}, p90={np.percentile(qdnn_dist_aug, 90):.2f}")

    # Analyze: for holdouts with zeros, how well do synthetics match?
    print("\n" + "=" * 70)
    print("ZERO-MATCHING ANALYSIS")
    print("=" * 70)

    holdout_pids = sorted(holdout_df['person_id'].unique())

    for col in zero_cols:
        # Find holdouts where this person has zeros for all periods
        always_zero = []
        never_zero = []
        for pid in holdout_pids:
            person = holdout_df[holdout_df['person_id'] == pid]
            if (person[col] == 0).all():
                always_zero.append(pid)
            elif (person[col] > 0).all():
                never_zero.append(pid)

        print(f"\n{col}:")
        print(f"  Holdouts always zero: {len(always_zero)}")
        print(f"  Holdouts never zero: {len(never_zero)}")

        if len(always_zero) > 0:
            # For these holdouts, check if nearest synthetic also has zeros
            holdout_zeros = holdout_aug[holdout_aug['person_id'].isin(always_zero)]

            # Find nearest synthetics
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            train_mat = to_matrix(train_aug, augmented_features)
            holdout_mat = to_matrix(holdout_zeros, augmented_features)

            scaler = StandardScaler().fit(train_mat)
            holdout_scaled = scaler.transform(holdout_mat)

            # Check VAE
            vae_mat = to_matrix(vae_synth_aug, augmented_features)
            vae_scaled = scaler.transform(vae_mat)
            nn = NearestNeighbors(n_neighbors=1).fit(vae_scaled)
            _, vae_idx = nn.kneighbors(holdout_scaled)

            vae_synth_pids = sorted(vae_synth_aug['person_id'].unique())
            vae_match_zero = 0
            for i, holdout_pid in enumerate(always_zero):
                synth_pid = vae_synth_pids[vae_idx[i, 0]]
                synth_person = vae_synth_aug[vae_synth_aug['person_id'] == synth_pid]
                if (synth_person[col] <= 0.01).all():  # Near-zero for VAE
                    vae_match_zero += 1
            print(f"  VAE nearest also zero: {vae_match_zero}/{len(always_zero)} ({100*vae_match_zero/len(always_zero):.0f}%)")

            # Check QDNN
            qdnn_mat = to_matrix(qdnn_synth_df, augmented_features)
            qdnn_scaled = scaler.transform(qdnn_mat)
            nn = NearestNeighbors(n_neighbors=1).fit(qdnn_scaled)
            _, qdnn_idx = nn.kneighbors(holdout_scaled)

            qdnn_synth_pids = sorted(qdnn_synth_df['person_id'].unique())
            qdnn_match_zero = 0
            for i, holdout_pid in enumerate(always_zero):
                synth_pid = qdnn_synth_pids[qdnn_idx[i, 0]]
                synth_person = qdnn_synth_df[qdnn_synth_df['person_id'] == synth_pid]
                if (synth_person[col] == 0).all():
                    qdnn_match_zero += 1
            print(f"  QDNN nearest also zero: {qdnn_match_zero}/{len(always_zero)} ({100*qdnn_match_zero/len(always_zero):.0f}%)")


if __name__ == "__main__":
    main()
