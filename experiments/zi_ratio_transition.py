"""Ratio-based quantiles + transition-based ZI.

Combines two fixes:
1. Ratio quantiles: predict log(next/current) instead of absolute values
2. Transition ZI: separate heads for P(become_zero|nonzero) vs P(stay_zero|zero)

This should fix both:
- Wealth explosion (via ratio predictions)
- Wealth collapse (via proper transition probabilities)
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


class RatioTransitionModel(nn.Module):
    """Combines ratio quantiles with transition-based ZI."""

    def __init__(self, n_features: int, hidden_dim: int = 256, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        # Add zero indicators to input
        self.input_dim = n_features + n_features  # features + is_nonzero indicators

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Transition heads
        self.become_zero_head = nn.Linear(hidden_dim, n_features)  # P(zero | was_nonzero)
        self.stay_zero_head = nn.Linear(hidden_dim, n_features)    # P(zero | was_zero)

        # Ratio quantile head: predicts log(y/x) for non-zeros
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

        # Initialization head: for when x=0 but y>0
        self.init_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

    def forward(self, x_aug):
        h = self.shared(x_aug)
        become_zero = self.become_zero_head(h)
        stay_zero = self.stay_zero_head(h)
        ratio_q = self.ratio_head(h).view(-1, self.n_features, self.n_quantiles)
        init_q = self.init_head(h).view(-1, self.n_features, self.n_quantiles)
        return h, become_zero, stay_zero, ratio_q, init_q

    def loss(self, x_raw, x_aug, target):
        h, become_zero_logits, stay_zero_logits, ratio_q, init_q = self.forward(x_aug)

        # Transition BCE loss
        is_zero_output = (target == 0).float()
        is_nonzero_input = (x_raw > 0).float()

        become_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            become_zero_logits, is_zero_output, reduction='none'
        )
        stay_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            stay_zero_logits, is_zero_output, reduction='none'
        )

        # Weight by which transition applies
        bce = (is_nonzero_input * become_zero_loss + (1 - is_nonzero_input) * stay_zero_loss).mean()

        # Ratio quantile loss: only for (x>0, y>0) pairs
        x_nonzero = (x_raw > 0)
        y_nonzero = (target > 0)
        ratio_mask = x_nonzero & y_nonzero

        if ratio_mask.any():
            log_ratio = torch.log(target + 1e-8) - torch.log(x_raw + 1e-8)
            errors = log_ratio.unsqueeze(-1) - ratio_q
            mask = ratio_mask.unsqueeze(-1).float()
            ql_ratio = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql_ratio = (ql_ratio * mask).sum() / (mask.sum() + 1e-8) / self.n_quantiles
        else:
            ql_ratio = torch.tensor(0.0)

        # Init quantile loss: only for (x=0, y>0) pairs
        init_mask = (~x_nonzero) & y_nonzero

        if init_mask.any():
            log_y = torch.log1p(target)
            errors = log_y.unsqueeze(-1) - init_q
            mask = init_mask.unsqueeze(-1).float()
            ql_init = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql_init = (ql_init * mask).sum() / (mask.sum() + 1e-8) / self.n_quantiles
        else:
            ql_init = torch.tensor(0.0)

        return bce + ql_ratio + ql_init


class CombinedModel:
    """Wrapper for training and sampling."""

    def __init__(self, n_features):
        self.n_features = n_features
        self.feature_model = RatioTransitionModel(n_features)

    def _augment(self, X):
        """Add is_nonzero indicators."""
        indicators = (X > 0).astype(float)
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

        # Augment with indicators (don't normalize indicators)
        X_aug = np.hstack([X_norm, (X > 0).astype(float)])

        X_raw_t = torch.tensor(X, dtype=torch.float32)
        X_aug_t = torch.tensor(X_aug, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)

        opt = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            self.feature_model.loss(X_raw_t, X_aug_t, Y_t).backward()
            opt.step()

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        indicators = (x_raw > 0).astype(float)
        x_aug = np.concatenate([x_norm, indicators])

        torch.tensor(x_raw, dtype=torch.float32).unsqueeze(0)
        x_aug_t = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, become_zero_logits, stay_zero_logits, ratio_q, init_q = self.feature_model.forward(x_aug_t)

            # Get P(zero) based on current state
            p_become_zero = torch.sigmoid(become_zero_logits).numpy()[0]
            p_stay_zero = torch.sigmoid(stay_zero_logits).numpy()[0]

            # Sample from ratio/init distributions
            n_quantiles = ratio_q.shape[-1]
            idx = (torch.rand(1, self.n_features, 1) * (n_quantiles - 1)).long()

            ratio_samples = ratio_q.gather(-1, idx).squeeze(-1)
            ratios = torch.exp(ratio_samples).numpy()[0]

            init_samples = init_q.gather(-1, idx).squeeze(-1)
            init_vals = torch.expm1(torch.clamp(init_samples, max=20)).numpy()[0]

        values = np.zeros(self.n_features)

        for j in range(self.n_features):
            if x_raw[j] > 0:
                # Was non-zero
                if np.random.random() < p_become_zero[j]:
                    values[j] = 0
                else:
                    values[j] = x_raw[j] * ratios[j]
            else:
                # Was zero
                if np.random.random() < p_stay_zero[j]:
                    values[j] = 0
                else:
                    values[j] = max(0, init_vals[j])

        return np.clip(values, 0, 1e10)

    def get_transition_probs(self, x_raw):
        """Get transition probabilities for debugging."""
        x_norm = (x_raw - self.X_mean) / self.X_std
        indicators = (x_raw > 0).astype(float)
        x_aug = np.concatenate([x_norm, indicators])
        x_aug_t = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, become_zero, stay_zero, _, _ = self.feature_model.forward(x_aug_t)
            return (torch.sigmoid(become_zero).numpy()[0],
                    torch.sigmoid(stay_zero).numpy()[0])


class OriginalModel:
    """Original model for comparison."""

    def __init__(self, n_features):
        self.n_features = n_features
        self.shared = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.zero_head = nn.Linear(256, n_features)
        self.quantile_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_features * 19),
        )
        self.quantiles = torch.linspace(0.05, 0.95, 19)

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

        params = list(self.shared.parameters()) + list(self.zero_head.parameters()) + list(self.quantile_head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        for _ in range(epochs):
            opt.zero_grad()
            h = self.shared(X_t)
            zl = self.zero_head(h)
            q = self.quantile_head(h).view(-1, self.n_features, 19)

            bce = nn.functional.binary_cross_entropy_with_logits(zl, (Y_t == 0).float())
            target_log = torch.log1p(torch.clamp(Y_t, min=0))
            errors = target_log.unsqueeze(-1) - q
            ql = torch.max((self.quantiles - 1) * errors, self.quantiles * errors).mean()
            (bce + ql).backward()
            opt.step()

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            h = self.shared(x_t)
            q = self.quantile_head(h).view(-1, self.n_features, 19)
            q_exp = torch.expm1(torch.clamp(q, max=20))
            q_sorted = torch.clamp(torch.sort(q_exp, dim=-1)[0], min=0, max=1e10)
            idx = (torch.rand(1, self.n_features, 1) * 18).long()
            values = q_sorted.gather(-1, idx).squeeze(-1).numpy()[0]

            zl = self.zero_head(h)
            p_zero = torch.sigmoid(zl).numpy()[0]

        values = np.clip(values, 0, 1e10)
        for j in range(self.n_features):
            if np.random.random() < p_zero[j]:
                values[j] = 0
        return values


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
    print("RATIO QUANTILES + TRANSITION ZI")
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

    # Compute true transition rates
    print("\n" + "=" * 70)
    print("TRUE TRANSITION RATES")
    print("=" * 70)

    X_list, Y_list = [], []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[feature_cols].values
        for t in range(len(values) - 1):
            X_list.append(values[t])
            Y_list.append(values[t + 1])
    X, Y = np.array(X_list), np.array(Y_list)

    # Count transitions for all zero-inflated features
    become_zero = ((X[:, 1:] > 0) & (Y[:, 1:] == 0)).sum()
    stay_nonzero = ((X[:, 1:] > 0) & (Y[:, 1:] > 0)).sum()
    stay_zero = ((X[:, 1:] == 0) & (Y[:, 1:] == 0)).sum()
    become_nonzero = ((X[:, 1:] == 0) & (Y[:, 1:] > 0)).sum()

    print(f"P(become_zero | nonzero) = {become_zero / (become_zero + stay_nonzero):.4f}")
    print(f"P(stay_zero | zero) = {stay_zero / (stay_zero + become_nonzero):.4f}")

    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    n_synth = 5000
    n_runs = 3

    results = {}

    for name, Model in [("Original", OriginalModel), ("Combined", CombinedModel)]:
        print(f"\n{name}:")
        medians = []

        for run in range(n_runs):
            model = Model(n_features)
            model.fit(train_df, feature_cols, epochs=100)
            synth_df = generate_synth(model, train_df, feature_cols, n_synth, 12, seed=123 + run)
            dist = compute_coverage(holdout_df, synth_df, train_df, feature_cols, zero_cols)
            medians.append(np.median(dist))
            print(f"  Run {run+1}: {np.median(dist):.2f}")

        results[name] = {'median': np.mean(medians), 'std': np.std(medians)}
        print(f"  → Mean: {np.mean(medians):.2f} ± {np.std(medians):.2f}")

    # Debug: Check transition probs
    print("\n" + "=" * 70)
    print("TRANSITION PROBS ON RICH PERSON")
    print("=" * 70)

    combined_model = CombinedModel(n_features)
    combined_model.fit(train_df, feature_cols, epochs=100)

    p_become, p_stay = combined_model.get_transition_probs(rich_state)

    print(f"\nRich person assets: ${rich_state[asset_start:].sum():,.0f}")

    print("\nNon-zero assets - P(become_zero) (should be ~0.005):")
    for i in range(n_asset):
        col_idx = asset_start + i
        if rich_state[col_idx] > 0:
            print(f"  asset_{i}: P(become_zero)={p_become[col_idx]:.4f}")

    # Test generation
    print("\n" + "=" * 70)
    print("GENERATION FROM RICH SEED")
    print("=" * 70)

    np.random.seed(42)

    print("\nCombined model (10 samples):")
    for i in range(10):
        next_state = combined_model.sample(rich_state)
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

    if results['Combined']['median'] < results['Original']['median']:
        improvement = (results['Original']['median'] - results['Combined']['median']) / results['Original']['median'] * 100
        print(f"\n✓ Combined model improves by {improvement:.1f}%")
    else:
        degradation = (results['Combined']['median'] - results['Original']['median']) / results['Original']['median'] * 100
        print(f"\n✗ Combined model degrades by {degradation:.1f}%")


if __name__ == "__main__":
    main()
