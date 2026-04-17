"""ZI model with masked quantile loss.

Key insight: The quantile head should ONLY be trained on non-zero samples.
The ZI head decides zero/nonzero, the quantile head predicts non-zero values.

Previous problem: 68% zeros → quantile head learns to predict ~0 for everything.
Fix: Mask quantile loss to only include non-zero targets.
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


class MaskedQuantileModel(nn.Module):
    """ZI model with masked quantile loss."""

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

        # BCE loss for zero classification (on ALL samples)
        is_zero = (target == 0).float()
        bce = nn.functional.binary_cross_entropy_with_logits(zero_logits, is_zero)

        # Quantile loss ONLY on non-zero targets
        nonzero_mask = (target > 0)  # (batch, n_features)

        if nonzero_mask.any():
            target_log = torch.log1p(torch.clamp(target, min=0))
            errors = target_log.unsqueeze(-1) - q_pred  # (batch, n_features, n_quantiles)

            # Mask: only compute loss where target is non-zero
            mask = nonzero_mask.unsqueeze(-1).float()  # (batch, n_features, 1)

            ql_raw = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql_masked = (ql_raw * mask).sum() / mask.sum() / self.n_quantiles
        else:
            ql_masked = torch.tensor(0.0)

        return bce + ql_masked

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


class OriginalModel(nn.Module):
    """Original model with unmasked quantile loss."""

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


class Model:
    """Wrapper for training and sampling."""

    def __init__(self, n_features, use_masked_loss=False):
        self.n_features = n_features
        if use_masked_loss:
            self.feature_model = MaskedQuantileModel(n_features)
        else:
            self.feature_model = OriginalModel(n_features)

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
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, zl, _ = self.feature_model(x_t)
            return torch.sigmoid(zl).numpy()[0]

    def get_quantile_preds(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, _, q = self.feature_model(x_t)
            q_exp = torch.expm1(torch.clamp(q, max=20))
            return torch.clamp(q_exp, min=0, max=1e10).numpy()[0]


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
    print("MASKED QUANTILE LOSS FIX")
    print("=" * 70)

    n_income, n_asset = 20, 20
    df = generate_rich_panel(n_persons=500, T=12, n_income_types=n_income, n_asset_types=n_asset, seed=42)

    feature_cols = ['age'] + [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    zero_cols = [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    n_features = len(feature_cols)
    asset_start = 1 + n_income

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    # Find richest holdout person
    holdout_totals = []
    for pid in holdout_df['person_id'].unique():
        person = holdout_df[holdout_df['person_id'] == pid]
        total = person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()
        holdout_totals.append((pid, total))
    richest_pid = max(holdout_totals, key=lambda x: x[1])[0]
    richest_person = holdout_df[holdout_df['person_id'] == richest_pid].sort_values('period')
    rich_state = richest_person[feature_cols].iloc[0].values
    true_next = richest_person[feature_cols].iloc[1].values

    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    n_synth = 5000
    n_runs = 3

    results = {}

    for name, use_masked in [("Original", False), ("Masked", True)]:
        print(f"\n{name}:")
        medians = []

        for run in range(n_runs):
            model = Model(n_features, use_masked_loss=use_masked)
            model.fit(train_df, feature_cols, epochs=100)
            synth_df = generate_synth(model, train_df, feature_cols, n_synth, 12, seed=123 + run)
            dist = compute_coverage(holdout_df, synth_df, train_df, feature_cols, zero_cols)
            medians.append(np.median(dist))
            print(f"  Run {run+1}: {np.median(dist):.2f}")

        results[name] = {'median': np.mean(medians), 'std': np.std(medians)}
        print(f"  → Mean: {np.mean(medians):.2f} ± {np.std(medians):.2f}")

    # Debug: Check quantile predictions
    print("\n" + "=" * 70)
    print("QUANTILE PREDICTIONS ON RICH PERSON")
    print("=" * 70)

    orig_model = Model(n_features, use_masked_loss=False)
    orig_model.fit(train_df, feature_cols, epochs=100)

    masked_model = Model(n_features, use_masked_loss=True)
    masked_model.fit(train_df, feature_cols, epochs=100)

    orig_q = orig_model.get_quantile_preds(rich_state)
    masked_q = masked_model.get_quantile_preds(rich_state)

    print(f"\nRich person assets: ${rich_state[asset_start:].sum():,.0f}")
    print(f"True next: ${true_next[asset_start:].sum():,.0f}")

    # Show predictions for non-zero assets
    print("\nNon-zero assets:")
    for i in range(n_asset):
        col_idx = asset_start + i
        if rich_state[col_idx] > 0:
            print(f"  asset_{i}: input=${rich_state[col_idx]:,.0f}, true=${true_next[col_idx]:,.0f}")
            print(f"    Original q50: ${orig_q[col_idx, 9]:,.0f}")
            print(f"    Masked q50: ${masked_q[col_idx, 9]:,.0f}")

    # Test generation
    print("\n" + "=" * 70)
    print("GENERATION FROM RICH SEED")
    print("=" * 70)

    np.random.seed(42)

    print("\nOriginal model (5 samples):")
    for i in range(5):
        next_state = orig_model.sample(rich_state)
        next_total = next_state[asset_start:].sum()
        ratio = next_total / max(rich_state[asset_start:].sum(), 1)
        print(f"  ${rich_state[asset_start:].sum():,.0f} → ${next_total:,.0f} (ratio: {ratio:.2f})")

    print("\nMasked model (5 samples):")
    for i in range(5):
        next_state = masked_model.sample(rich_state)
        next_total = next_state[asset_start:].sum()
        ratio = next_total / max(rich_state[asset_start:].sum(), 1)
        print(f"  ${rich_state[asset_start:].sum():,.0f} → ${next_total:,.0f} (ratio: {ratio:.2f})")

    print(f"\nTrue: ${rich_state[asset_start:].sum():,.0f} → ${true_next[asset_start:].sum():,.0f} "
          f"(ratio: {true_next[asset_start:].sum() / max(rich_state[asset_start:].sum(), 1):.2f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"{name}: {r['median']:.2f} ± {r['std']:.2f}")
    improvement = (results['Original']['median'] - results['Masked']['median']) / results['Original']['median'] * 100
    print(f"\nImprovement: {improvement:.1f}%")


if __name__ == "__main__":
    main()
