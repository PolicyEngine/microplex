"""Compare ZI-QDNN vs standard QDNN using coverage with zero indicators.

Evaluation metric: Coverage distance INCLUDING binary is_nonzero indicators
- This captures whether synthetics match holdouts' zero patterns

ZI component: Simple logistic regression head predicting P(zero | features)
Not a complex ML model - just a learned conditional probability.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_panel_with_zeros(n_persons: int = 500, T: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generate panel with realistic zero-inflation patterns."""
    np.random.seed(seed)
    records = []
    for pid in range(n_persons):
        age = np.random.randint(25, 60)

        # Employment (affects income)
        employed = np.random.random() > 0.15
        income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000)) if employed else 0

        # Wealth - can be zero
        has_wealth = np.random.random() > 0.3
        wealth = np.random.lognormal(10, 1.5) if has_wealth else 0

        # Dividend income - most have zero
        has_dividends = np.random.random() > 0.7
        dividend_base = np.random.lognormal(7, 1) if has_dividends else 0

        for t in range(T):
            if np.random.random() < 0.02:
                employed = not employed
            if employed and income == 0:
                income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000))
            elif not employed:
                income = 0
            else:
                income = max(0, income * (1 + np.random.normal(0.02/12, 0.1)))

            if wealth > 0:
                wealth = wealth * (1 + np.random.normal(0.05/12, 0.02))

            dividend = dividend_base * (1 + np.random.normal(0, 0.1)) if dividend_base > 0 else 0

            records.append({
                'person_id': pid, 'period': t,
                'age': age + t / 12,
                'income': income,
                'net_worth': wealth,
                'dividend_income': max(0, dividend),
            })
    return pd.DataFrame(records)


class StandardQDNN(nn.Module):
    """Standard Quantile DNN - no explicit zero-inflation handling."""

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
            return torch.clamp(samples, min=0)


class ZeroInflatedQDNN(nn.Module):
    """Quantile DNN with explicit zero-inflation head.

    Architecture:
      - Shared hidden layers
      - Zero head: P(zero | x) via logistic regression on hidden features
      - Quantile head: Conditional quantiles for non-zero values

    The zero head is just: Linear(hidden_dim, output_dim) → Sigmoid
    This is equivalent to logistic regression on the learned features.

    IMPORTANT: This model works on ORIGINAL scale (not normalized) so the
    zero head can properly detect zeros. The quantile head needs to output
    larger values, handled via log-space output.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_quantiles: int = 19):
        super().__init__()
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles

        # Shared feature extraction (deeper for original scale)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Zero-inflation head: Simple logistic regression on hidden features
        # This learns P(value = 0 | features) for each output dimension
        self.zero_head = nn.Linear(hidden_dim, output_dim)

        # Quantile head for non-zero values (outputs log-scale)
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * n_quantiles),
        )

    def forward(self, x):
        h = self.shared(x)
        zero_logits = self.zero_head(h)  # (batch, output_dim)
        q = self.quantile_head(h).view(-1, self.output_dim, self.n_quantiles)
        return zero_logits, q

    def loss(self, x, target):
        zero_logits, q_pred = self.forward(x)

        # Binary cross-entropy for zero classification
        is_zero = (target == 0).float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='mean'
        )

        # Quantile loss only on non-zero values (in log space for stability)
        non_zero_mask = (target > 0)
        if non_zero_mask.any():
            # Work in log space for numerical stability
            target_log = torch.log1p(target)
            ql_total = 0
            count = 0
            for j in range(self.output_dim):
                mask_j = non_zero_mask[:, j]
                if mask_j.sum() > 0:
                    target_j = target_log[mask_j, j].unsqueeze(-1)
                    pred_j = q_pred[mask_j, j, :]  # Already in log space
                    errors = target_j - pred_j
                    ql = torch.max(
                        (self.quantiles - 1) * errors,
                        self.quantiles * errors
                    ).mean()
                    ql_total += ql
                    count += 1
            ql_loss = ql_total / max(count, 1)
        else:
            ql_loss = 0

        return bce_loss + ql_loss

    def sample(self, x):
        with torch.no_grad():
            zero_logits, q = self.forward(x)
            p_zero = torch.sigmoid(zero_logits)

            # q is in log space, clamp to avoid overflow, then convert back
            q_clamped = torch.clamp(q, max=20)  # exp(20) ≈ 500M, reasonable max
            q_exp = torch.expm1(q_clamped)
            q_sorted = torch.sort(q_exp, dim=-1)[0]
            q_sorted = torch.clamp(q_sorted, min=0, max=1e10)
            u = torch.rand(x.shape[0], self.output_dim, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            samples = q_sorted.gather(-1, idx).squeeze(-1)

            # Zero out based on zero probability
            is_zero = torch.rand_like(p_zero) < p_zero
            samples = torch.where(is_zero, torch.zeros_like(samples), samples)

            return samples


def train_model(model, train_df, feature_cols, epochs=100, is_zi=False):
    """Train a QDNN model on panel data.

    For ZI models, we train on original scale (not normalized) so the
    zero-inflation head can properly learn P(zero | features).
    """
    # Prepare transitions: predict t+1 from t
    X_list, Y_list = [], []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[feature_cols].values
        for t in range(len(values) - 1):
            X_list.append(values[t])
            Y_list.append(values[t + 1])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    # Normalize inputs
    X_mean, X_std = X.mean(0), X.std(0) + 1e-6
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-6
    X_norm = (X - X_mean) / X_std

    if is_zi:
        # For ZI model: normalize X but keep Y on original scale
        # This lets the zero head learn from actual zeros
        Y_train = Y
        Y_mean_ret = torch.zeros_like(Y_mean)
        Y_std_ret = torch.ones_like(Y_std)
    else:
        Y_train = (Y - Y_mean) / Y_std
        Y_mean_ret = Y_mean
        Y_std_ret = Y_std

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(X_norm, Y_train)
        loss.backward()
        optimizer.step()

    return X_mean, X_std, Y_mean_ret, Y_std_ret


def generate_panel(model, train_df, feature_cols, n_synth, T,
                   X_mean, X_std, Y_mean, Y_std, seed=42):
    """Generate synthetic panel using trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get initial states from training data
    init_states = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        init_states.append(person[feature_cols].iloc[0].values)
    init_states = np.array(init_states)

    records = []
    for pid in range(n_synth):
        init_idx = np.random.randint(len(init_states))
        state = init_states[init_idx].copy()

        for t in range(T):
            # Clamp state to avoid inf/nan
            state = np.clip(state, 0, 1e10)
            state = np.nan_to_num(state, nan=0, posinf=1e10, neginf=0)

            records.append({
                'person_id': pid,
                'period': t,
                **{col: float(state[i]) for i, col in enumerate(feature_cols)}
            })

            if t < T - 1:
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                x_norm = (x - X_mean) / X_std
                next_norm = model.sample(x_norm)
                next_state = (next_norm * Y_std + Y_mean).numpy()[0]
                state = np.maximum(0, next_state)
                state = np.clip(state, 0, 1e10)
                state = np.nan_to_num(state, nan=0, posinf=1e10, neginf=0)

    return pd.DataFrame(records)


def compute_coverage_with_indicators(holdout_df, synth_df, train_df, base_cols, zero_cols):
    """Compute coverage distance INCLUDING zero indicators.

    The evaluation metric includes binary is_nonzero columns for each
    zero-inflated feature. This captures whether synthetics match
    holdouts' zero/nonzero patterns.
    """
    # Augment all dataframes with zero indicators
    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nonzero'] = (df[col] > 0).astype(float)
        return df

    train_aug = augment(train_df)
    holdout_aug = augment(holdout_df)
    synth_aug = augment(synth_df)

    # Full feature set for evaluation
    eval_cols = base_cols + [f'{c}_nonzero' for c in zero_cols]

    # Convert to matrices
    def to_matrix(df):
        matrices = []
        for pid in sorted(df['person_id'].unique()):
            person = df[df['person_id'] == pid].sort_values('period')
            matrices.append(person[eval_cols].values.flatten())
        return np.array(matrices)

    train_mat = to_matrix(train_aug)
    holdout_mat = to_matrix(holdout_aug)
    synth_mat = to_matrix(synth_aug)

    # Fit scaler on training data
    scaler = StandardScaler().fit(train_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    # Nearest neighbor distances
    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(holdout_scaled)

    return distances[:, 0]


def main():
    print("=" * 70)
    print("ZI-QDNN vs STANDARD QDNN COMPARISON")
    print("Evaluation: Coverage with zero indicators")
    print("=" * 70)

    # Generate data
    df = generate_panel_with_zeros(n_persons=500, T=12, seed=42)

    # Check actual zero rates
    base_cols = ['age', 'income', 'net_worth', 'dividend_income']
    zero_cols = ['income', 'net_worth', 'dividend_income']

    print("\nTrue zero rates:")
    for col in zero_cols:
        zero_rate = (df[col] == 0).mean()
        print(f"  {col}: {zero_rate:.1%}")

    # Split data
    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    # Train both models
    print("\n" + "=" * 70)
    print("Training models...")
    print("=" * 70)

    n_features = len(base_cols)

    print("\n1. Standard QDNN (no ZI component)")
    std_model = StandardQDNN(n_features, n_features)
    std_stats = train_model(std_model, train_df, base_cols, epochs=100, is_zi=False)

    print("2. ZI-QDNN (with logistic zero head)")
    zi_model = ZeroInflatedQDNN(n_features, n_features)
    zi_stats = train_model(zi_model, train_df, base_cols, epochs=100, is_zi=True)

    # Generate synthetics
    n_synth = 500
    print(f"\nGenerating {n_synth} synthetics from each model...")

    std_synth = generate_panel(std_model, train_df, base_cols, n_synth, 12, *std_stats, seed=123)
    zi_synth = generate_panel(zi_model, train_df, base_cols, n_synth, 12, *zi_stats, seed=123)

    # Check zero rates in synthetics
    print("\n" + "=" * 70)
    print("Zero rates in synthetic data:")
    print("=" * 70)

    print("\nStandard QDNN:")
    for col in zero_cols:
        exact_zero = (std_synth[col] == 0).mean()
        near_zero = (std_synth[col] < 0.1 * df[col].std()).mean()
        print(f"  {col}: {exact_zero:.1%} exact zeros, {near_zero:.1%} near-zero")

    print("\nZI-QDNN:")
    for col in zero_cols:
        exact_zero = (zi_synth[col] == 0).mean()
        near_zero = (zi_synth[col] < 0.1 * df[col].std()).mean()
        print(f"  {col}: {exact_zero:.1%} exact zeros, {near_zero:.1%} near-zero")

    # Compute coverage WITH zero indicators
    print("\n" + "=" * 70)
    print("COVERAGE (with zero indicators in distance metric)")
    print("=" * 70)

    std_dist = compute_coverage_with_indicators(holdout_df, std_synth, train_df, base_cols, zero_cols)
    zi_dist = compute_coverage_with_indicators(holdout_df, zi_synth, train_df, base_cols, zero_cols)

    print("\nStandard QDNN:")
    print(f"  median: {np.median(std_dist):.2f}")
    print(f"  p90:    {np.percentile(std_dist, 90):.2f}")
    print(f"  max:    {np.max(std_dist):.2f}")

    print("\nZI-QDNN:")
    print(f"  median: {np.median(zi_dist):.2f}")
    print(f"  p90:    {np.percentile(zi_dist, 90):.2f}")
    print(f"  max:    {np.max(zi_dist):.2f}")

    # Improvement
    improvement = (np.median(std_dist) - np.median(zi_dist)) / np.median(std_dist) * 100
    print(f"\nZI-QDNN improvement: {improvement:+.1f}% median distance")

    # Breakdown by holdout type
    print("\n" + "=" * 70)
    print("BREAKDOWN: Holdouts with/without zeros")
    print("=" * 70)

    holdout_pids = sorted(holdout_df['person_id'].unique())

    # Classify holdouts
    has_zeros = []
    no_zeros = []
    for pid in holdout_pids:
        person = holdout_df[holdout_df['person_id'] == pid]
        if any((person[col] == 0).any() for col in zero_cols):
            has_zeros.append(pid)
        else:
            no_zeros.append(pid)

    print(f"\nHoldouts with zeros: {len(has_zeros)}")
    print(f"Holdouts without zeros: {len(no_zeros)}")

    # Get indices
    has_zeros_idx = [holdout_pids.index(pid) for pid in has_zeros]
    no_zeros_idx = [holdout_pids.index(pid) for pid in no_zeros]

    print("\nHoldouts WITH zeros:")
    print(f"  Standard QDNN: median={np.median(std_dist[has_zeros_idx]):.2f}")
    print(f"  ZI-QDNN:       median={np.median(zi_dist[has_zeros_idx]):.2f}")

    print("\nHoldouts WITHOUT zeros:")
    print(f"  Standard QDNN: median={np.median(std_dist[no_zeros_idx]):.2f}")
    print(f"  ZI-QDNN:       median={np.median(zi_dist[no_zeros_idx]):.2f}")


if __name__ == "__main__":
    main()
