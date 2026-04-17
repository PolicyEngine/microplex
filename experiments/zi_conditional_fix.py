"""Fix ZI head by adding explicit zero indicators to input.

Problem: ZI head predicts marginal P(zero)=70% regardless of current wealth.
Fix: Add binary is_nonzero indicators to input so ZI can learn transitions.

Hypothesis: P(zero at t+1 | nonzero at t) << P(zero at t+1 | zero at t)
The model needs to see the current zero/nonzero status explicitly.
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


def generate_rich_panel(n_persons: int = 500, T: int = 12, n_income_types: int = 20,
                        n_asset_types: int = 20, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    records = []

    for pid in range(n_persons):
        age = np.random.randint(25, 65)
        income_probs = np.random.beta(2, 5, n_income_types)
        income_bases = np.random.lognormal(10, 1, n_income_types)
        income_active = np.random.random(n_income_types) < income_probs
        asset_probs = np.random.beta(2, 5, n_asset_types)
        asset_bases = np.random.lognormal(11, 1.5, n_asset_types)
        asset_active = np.random.random(n_asset_types) < asset_probs

        for t in range(T):
            row = {'person_id': pid, 'period': t, 'age': age + t / 12}
            for i in range(n_income_types):
                if income_active[i]:
                    val = income_bases[i] * (1 + np.random.normal(0.02/12, 0.1))
                    income_bases[i] = val
                else:
                    val = 0
                if np.random.random() < 0.01:
                    income_active[i] = not income_active[i]
                    if income_active[i]:
                        income_bases[i] = np.random.lognormal(10, 1)
                row[f'income_{i}'] = max(0, val)
            for i in range(n_asset_types):
                if asset_active[i]:
                    val = asset_bases[i] * (1 + np.random.normal(0.05/12, 0.02))
                    asset_bases[i] = val
                else:
                    val = 0
                row[f'asset_{i}'] = max(0, val)
            records.append(row)
    return pd.DataFrame(records)


class ConditionalZIModel(nn.Module):
    """ZI model with explicit zero indicators in input.

    Input: [continuous_features, is_nonzero_indicators]
    This lets the ZI head learn transition probabilities:
    - P(zero | was_zero) - staying zero
    - P(zero | was_nonzero) - becoming zero (should be low for stable features)
    """

    def __init__(self, n_features: int, n_zero_cols: int, hidden_dim: int = 256, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.n_zero_cols = n_zero_cols
        self.input_dim = n_features + n_zero_cols  # Add indicators
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.zero_head = nn.Linear(hidden_dim, n_features)
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

    def forward(self, x):
        h = self.shared(x)
        return h, self.zero_head(h), self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)

    def loss(self, x, target):
        h, zero_logits, q_pred = self.forward(x)
        bce = nn.functional.binary_cross_entropy_with_logits(zero_logits, (target == 0).float())
        target_log = torch.log1p(torch.clamp(target, min=0))
        errors = target_log.unsqueeze(-1) - q_pred
        ql = torch.max((self.quantiles - 1) * errors, self.quantiles * errors).mean()
        return bce + ql

    def get_latent(self, x):
        with torch.no_grad():
            return self.shared(x)

    def sample_values(self, h):
        with torch.no_grad():
            q = self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)
            q_exp = torch.expm1(torch.clamp(q, max=20))
            q_sorted = torch.clamp(torch.sort(q_exp, dim=-1)[0], min=0, max=1e10)
            idx = (torch.rand(h.shape[0], self.n_features, 1) * (self.n_quantiles - 1)).long()
            return q_sorted.gather(-1, idx).squeeze(-1)


class OriginalZIModel(nn.Module):
    """Original ZI model without zero indicators (baseline)."""

    def __init__(self, n_features: int, hidden_dim: int = 256, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.zero_head = nn.Linear(hidden_dim, n_features)
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

    def forward(self, x):
        h = self.shared(x)
        return h, self.zero_head(h), self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)

    def loss(self, x, target):
        h, zero_logits, q_pred = self.forward(x)
        bce = nn.functional.binary_cross_entropy_with_logits(zero_logits, (target == 0).float())
        target_log = torch.log1p(torch.clamp(target, min=0))
        errors = target_log.unsqueeze(-1) - q_pred
        ql = torch.max((self.quantiles - 1) * errors, self.quantiles * errors).mean()
        return bce + ql

    def get_latent(self, x):
        with torch.no_grad():
            return self.shared(x)

    def sample_values(self, h):
        with torch.no_grad():
            q = self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)
            q_exp = torch.expm1(torch.clamp(q, max=20))
            q_sorted = torch.clamp(torch.sort(q_exp, dim=-1)[0], min=0, max=1e10)
            idx = (torch.rand(h.shape[0], self.n_features, 1) * (self.n_quantiles - 1)).long()
            return q_sorted.gather(-1, idx).squeeze(-1)


class ConditionalModel:
    """Model with zero indicators in input."""

    def __init__(self, n_features, zero_col_indices):
        self.n_features = n_features
        self.zero_col_indices = zero_col_indices
        self.n_zero_cols = len(zero_col_indices)
        self.feature_model = ConditionalZIModel(n_features, self.n_zero_cols)

    def _add_indicators(self, X):
        """Add binary is_nonzero indicators for zero-inflated columns."""
        indicators = (X[:, self.zero_col_indices] > 0).astype(float)
        return np.hstack([X, indicators])

    def fit(self, train_df, feature_cols, epochs=100):
        X_list, Y_list = [], []
        for pid in train_df['person_id'].unique():
            person = train_df[train_df['person_id'] == pid].sort_values('period')
            values = person[feature_cols].values
            for t in range(len(values) - 1):
                X_list.append(values[t])
                Y_list.append(values[t + 1])
        X, Y = np.array(X_list), np.array(Y_list)

        self.X_mean, self.X_std = X.mean(0), X.std(0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std

        # Add indicators AFTER normalization (indicators are 0/1, don't normalize)
        X_aug = self._add_indicators(X)  # Use raw X for indicators
        X_aug_norm = np.hstack([X_norm, X_aug[:, -self.n_zero_cols:]])

        X_t = torch.tensor(X_aug_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)

        opt = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            self.feature_model.loss(X_t, Y_t).backward()
            opt.step()

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        indicators = (x_raw[self.zero_col_indices] > 0).astype(float)
        x_aug = np.concatenate([x_norm, indicators])
        x_t = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0)

        h = self.feature_model.get_latent(x_t)
        values = np.clip(self.feature_model.sample_values(h).numpy()[0], 0, 1e10)

        with torch.no_grad():
            _, zl, _ = self.feature_model(x_t)
            p_zero = torch.sigmoid(zl).numpy()[0]
        for j in range(self.n_features):
            if np.random.random() < p_zero[j]:
                values[j] = 0
        return values

    def get_p_zero(self, x_raw):
        """Get P(zero) predictions for debugging."""
        x_norm = (x_raw - self.X_mean) / self.X_std
        indicators = (x_raw[self.zero_col_indices] > 0).astype(float)
        x_aug = np.concatenate([x_norm, indicators])
        x_t = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, zl, _ = self.feature_model(x_t)
            return torch.sigmoid(zl).numpy()[0]


class OriginalModel:
    """Original model without indicators (baseline)."""

    def __init__(self, n_features):
        self.n_features = n_features
        self.feature_model = OriginalZIModel(n_features)

    def fit(self, train_df, feature_cols, epochs=100):
        X_list, Y_list = [], []
        for pid in train_df['person_id'].unique():
            person = train_df[train_df['person_id'] == pid].sort_values('period')
            values = person[feature_cols].values
            for t in range(len(values) - 1):
                X_list.append(values[t])
                Y_list.append(values[t + 1])
        X, Y = np.array(X_list), np.array(Y_list)

        self.X_mean, self.X_std = X.mean(0), X.std(0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)

        opt = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            self.feature_model.loss(X_t, Y_t).backward()
            opt.step()

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

        h = self.feature_model.get_latent(x_t)
        values = np.clip(self.feature_model.sample_values(h).numpy()[0], 0, 1e10)

        with torch.no_grad():
            _, zl, _ = self.feature_model(x_t)
            p_zero = torch.sigmoid(zl).numpy()[0]
        for j in range(self.n_features):
            if np.random.random() < p_zero[j]:
                values[j] = 0
        return values

    def get_p_zero(self, x_raw):
        """Get P(zero) predictions for debugging."""
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, zl, _ = self.feature_model(x_t)
            return torch.sigmoid(zl).numpy()[0]


def generate_synth(model, train_df, feature_cols, n_synth, T, seed=42):
    np.random.seed(seed)
    init_states = [train_df[train_df['person_id'] == pid].sort_values('period')[feature_cols].iloc[0].values
                   for pid in train_df['person_id'].unique()]
    records = []
    for pid in range(n_synth):
        state = init_states[np.random.randint(len(init_states))].copy()
        for t in range(T):
            state = np.clip(np.nan_to_num(state, 0), 0, 1e10)
            records.append({'person_id': pid, 'period': t,
                           **{col: float(state[i]) for i, col in enumerate(feature_cols)}})
            if t < T - 1:
                state = np.clip(model.sample(state), 0, 1e10)
    return pd.DataFrame(records)


def compute_coverage(holdout_df, synth_df, train_df, feature_cols, zero_cols):
    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nz'] = (df[col] > 0).astype(float)
        return df

    eval_cols = feature_cols + [f'{c}_nz' for c in zero_cols]

    def to_matrix(df):
        return np.array([df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
                        for pid in sorted(df['person_id'].unique())])

    train_mat = to_matrix(augment(train_df))
    holdout_mat = to_matrix(augment(holdout_df))
    synth_mat = to_matrix(augment(synth_df))

    scaler = StandardScaler().fit(train_mat)
    nn_model = NearestNeighbors(n_neighbors=1).fit(scaler.transform(synth_mat))
    distances, _ = nn_model.kneighbors(scaler.transform(holdout_mat))
    return distances[:, 0]


def main():
    print("=" * 70)
    print("CONDITIONAL ZI FIX: Add zero indicators to input")
    print("=" * 70)

    n_income, n_asset = 20, 20
    df = generate_rich_panel(n_persons=500, T=12, n_income_types=n_income, n_asset_types=n_asset, seed=42)

    feature_cols = ['age'] + [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    zero_cols = [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    n_features = len(feature_cols)

    # Get indices of zero-inflated columns (all except age)
    zero_col_indices = list(range(1, n_features))

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    # First, show the transition rates in training data
    print("\n" + "=" * 70)
    print("TRANSITION RATES IN TRAINING DATA")
    print("=" * 70)

    # Build transition pairs
    transitions = {'stay_zero': 0, 'become_nonzero': 0, 'stay_nonzero': 0, 'become_zero': 0}
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[zero_cols].values
        for t in range(len(values) - 1):
            for j in range(len(zero_cols)):
                was_zero = values[t, j] == 0
                is_zero = values[t+1, j] == 0
                if was_zero and is_zero:
                    transitions['stay_zero'] += 1
                elif was_zero and not is_zero:
                    transitions['become_nonzero'] += 1
                elif not was_zero and not is_zero:
                    transitions['stay_nonzero'] += 1
                else:
                    transitions['become_zero'] += 1

    total_from_zero = transitions['stay_zero'] + transitions['become_nonzero']
    total_from_nonzero = transitions['stay_nonzero'] + transitions['become_zero']

    print(f"From ZERO:    stay_zero={transitions['stay_zero']/total_from_zero:.1%}, become_nonzero={transitions['become_nonzero']/total_from_zero:.1%}")
    print(f"From NONZERO: stay_nonzero={transitions['stay_nonzero']/total_from_nonzero:.1%}, become_zero={transitions['become_zero']/total_from_nonzero:.1%}")
    print(f"\nKey insight: P(zero|was_nonzero) = {transitions['become_zero']/total_from_nonzero:.1%}")
    print(f"             P(zero|was_zero) = {transitions['stay_zero']/total_from_zero:.1%}")
    print(f"             Ratio: {(transitions['stay_zero']/total_from_zero) / (transitions['become_zero']/total_from_nonzero):.1f}x")

    # Train both models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    n_synth = 5000
    n_runs = 3

    results = {}

    for name, Model, kwargs in [
        ("Original (no indicators)", OriginalModel, {'n_features': n_features}),
        ("Conditional (with indicators)", ConditionalModel, {'n_features': n_features, 'zero_col_indices': zero_col_indices}),
    ]:
        print(f"\n{name}:")
        medians = []

        for run in range(n_runs):
            model = Model(**kwargs)
            model.fit(train_df, feature_cols, epochs=100)
            synth_df = generate_synth(model, train_df, feature_cols, n_synth, 12, seed=123 + run)
            dist = compute_coverage(holdout_df, synth_df, train_df, feature_cols, zero_cols)
            medians.append(np.median(dist))
            print(f"  Run {run+1}: {np.median(dist):.2f}")

        results[name] = {'median': np.mean(medians), 'std': np.std(medians)}
        print(f"  → Mean: {np.mean(medians):.2f} ± {np.std(medians):.2f}")

    # Test P(zero) predictions on rich person
    print("\n" + "=" * 70)
    print("P(ZERO) PREDICTIONS FOR RICH PERSON")
    print("=" * 70)

    # Find richest person in holdout
    holdout_totals = []
    for pid in holdout_df['person_id'].unique():
        person = holdout_df[holdout_df['person_id'] == pid]
        total = person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()
        holdout_totals.append((pid, total))
    richest_pid = max(holdout_totals, key=lambda x: x[1])[0]
    richest_person = holdout_df[holdout_df['person_id'] == richest_pid].sort_values('period')
    rich_state = richest_person[feature_cols].iloc[0].values

    print(f"Rich person total assets: ${richest_person[[f'asset_{i}' for i in range(n_asset)]].sum().sum():,.0f}")

    # Train fresh models and check P(zero)
    orig_model = OriginalModel(n_features)
    orig_model.fit(train_df, feature_cols, epochs=100)

    cond_model = ConditionalModel(n_features, zero_col_indices)
    cond_model.fit(train_df, feature_cols, epochs=100)

    orig_pz = orig_model.get_p_zero(rich_state)
    cond_pz = cond_model.get_p_zero(rich_state)

    # Asset columns are indices 21-40 (after age and 20 income cols)
    asset_start = 1 + n_income

    print("\nOriginal model P(zero) for assets:")
    print(f"  Mean: {orig_pz[asset_start:].mean():.3f}")
    print(f"  Range: {orig_pz[asset_start:].min():.3f} - {orig_pz[asset_start:].max():.3f}")
    print(f"  Assets with P(zero) > 0.5: {(orig_pz[asset_start:] > 0.5).sum()}/{n_asset}")

    print("\nConditional model P(zero) for assets:")
    print(f"  Mean: {cond_pz[asset_start:].mean():.3f}")
    print(f"  Range: {cond_pz[asset_start:].min():.3f} - {cond_pz[asset_start:].max():.3f}")
    print(f"  Assets with P(zero) > 0.5: {(cond_pz[asset_start:] > 0.5).sum()}/{n_asset}")

    # Count how many assets are actually non-zero for this person
    n_nonzero_assets = (rich_state[asset_start:] > 0).sum()
    print(f"\nActual non-zero assets: {n_nonzero_assets}/{n_asset}")
    print(f"Expected P(zero|nonzero) from training: {transitions['become_zero']/total_from_nonzero:.3f}")

    # Test generation from rich seed
    print("\n" + "=" * 70)
    print("GENERATION FROM RICH SEED")
    print("=" * 70)

    np.random.seed(42)

    print("\nOriginal model (5 samples):")
    for i in range(5):
        next_state = orig_model.sample(rich_state)
        next_total = next_state[asset_start:].sum()
        print(f"  ${rich_state[asset_start:].sum():,.0f} → ${next_total:,.0f} (ratio: {next_total/rich_state[asset_start:].sum():.2f})")

    print("\nConditional model (5 samples):")
    for i in range(5):
        next_state = cond_model.sample(rich_state)
        next_total = next_state[asset_start:].sum()
        print(f"  ${rich_state[asset_start:].sum():,.0f} → ${next_total:,.0f} (ratio: {next_total/rich_state[asset_start:].sum():.2f})")

    # True next state
    true_next = richest_person[feature_cols].iloc[1].values
    print(f"\nTrue trajectory: ${rich_state[asset_start:].sum():,.0f} → ${true_next[asset_start:].sum():,.0f} (ratio: {true_next[asset_start:].sum()/rich_state[asset_start:].sum():.2f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"{name}: {r['median']:.2f} ± {r['std']:.2f}")


if __name__ == "__main__":
    main()
