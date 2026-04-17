"""Debug quantile head predictions.

The ZI head seems to work (P(zero|nonzero)=0.088), but quantile predictions
are exploding: $6M → $485M-$970M. Let's see what's happening.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


class Model(nn.Module):
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


def main():
    print("=" * 70)
    print("DEBUG QUANTILE HEAD")
    print("=" * 70)

    n_income, n_asset = 20, 20
    df = generate_rich_panel(n_persons=500, T=12, n_income_types=n_income, n_asset_types=n_asset, seed=42)

    feature_cols = ['age'] + [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    n_features = len(feature_cols)
    asset_start = 1 + n_income

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    # Build training data
    X_list, Y_list = [], []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[feature_cols].values
        for t in range(len(values) - 1):
            X_list.append(values[t])
            Y_list.append(values[t + 1])
    X, Y = np.array(X_list), np.array(Y_list)

    X_mean, X_std = X.mean(0), X.std(0) + 1e-6
    X_norm = (X - X_mean) / X_std

    # Check training data distribution
    print("\n" + "=" * 70)
    print("TRAINING DATA DISTRIBUTION")
    print("=" * 70)

    for i in range(n_asset):
        col_idx = asset_start + i
        nonzero_mask = Y[:, col_idx] > 0
        if nonzero_mask.sum() > 0:
            nonzero_vals = Y[:, col_idx][nonzero_mask]
            print(f"asset_{i}: n={nonzero_mask.sum()}, "
                  f"min=${nonzero_vals.min():,.0f}, "
                  f"median=${np.median(nonzero_vals):,.0f}, "
                  f"max=${nonzero_vals.max():,.0f}")

    # Find the richest training sample
    rich_idx = np.argmax(Y[:, asset_start:].sum(axis=1))
    print(f"\nRichest training sample: ${Y[rich_idx, asset_start:].sum():,.0f}")
    print(f"  Previous state: ${X[rich_idx, asset_start:].sum():,.0f}")

    # Train model
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)

    model = Model(n_features)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        opt.zero_grad()
        loss = model.loss(X_t, Y_t)
        loss.backward()
        opt.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")

    # Check quantile predictions on training data
    print("\n" + "=" * 70)
    print("QUANTILE PREDICTIONS ON TRAINING DATA")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        h, _, q_pred = model.forward(X_t)
        # q_pred is in log1p space
        q_pred_exp = torch.expm1(torch.clamp(q_pred, max=20))
        q_pred_exp = torch.clamp(q_pred_exp, min=0, max=1e10)

    # For a rich training sample
    print(f"\nRich training sample (idx={rich_idx}):")
    print(f"  Input assets: ${X[rich_idx, asset_start:].sum():,.0f}")
    print(f"  True output assets: ${Y[rich_idx, asset_start:].sum():,.0f}")

    # Sum predicted quantiles across asset columns
    q_median_idx = 9  # Middle of 19 quantiles
    pred_median = q_pred_exp[rich_idx, asset_start:, q_median_idx].sum().item()
    pred_low = q_pred_exp[rich_idx, asset_start:, 0].sum().item()
    pred_high = q_pred_exp[rich_idx, asset_start:, -1].sum().item()
    print(f"  Predicted quantiles (sum): q5=${pred_low:,.0f}, q50=${pred_median:,.0f}, q95=${pred_high:,.0f}")

    # Check individual asset with highest value
    rich_asset_idx = np.argmax(Y[rich_idx, asset_start:]) + asset_start
    true_val = Y[rich_idx, rich_asset_idx]
    pred_q = q_pred_exp[rich_idx, rich_asset_idx, :].numpy()
    print(f"\nHighest-value asset (idx={rich_asset_idx}):")
    print(f"  True value: ${true_val:,.0f}")
    print(f"  Predicted quantiles: q5=${pred_q[0]:,.0f}, q50=${pred_q[9]:,.0f}, q95=${pred_q[18]:,.0f}")

    # Check a random median wealth sample
    median_idx = np.argsort(Y[:, asset_start:].sum(axis=1))[len(Y)//2]
    print(f"\nMedian training sample (idx={median_idx}):")
    print(f"  Input assets: ${X[median_idx, asset_start:].sum():,.0f}")
    print(f"  True output assets: ${Y[median_idx, asset_start:].sum():,.0f}")
    pred_median = q_pred_exp[median_idx, asset_start:, q_median_idx].sum().item()
    print(f"  Predicted q50: ${pred_median:,.0f}")

    # Now check on holdout rich person
    print("\n" + "=" * 70)
    print("QUANTILE PREDICTIONS ON HOLDOUT RICH PERSON")
    print("=" * 70)

    holdout_totals = []
    for pid in holdout_df['person_id'].unique():
        person = holdout_df[holdout_df['person_id'] == pid]
        total = person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()
        holdout_totals.append((pid, total))
    richest_pid = max(holdout_totals, key=lambda x: x[1])[0]
    richest_person = holdout_df[holdout_df['person_id'] == richest_pid].sort_values('period')
    rich_state = richest_person[feature_cols].iloc[0].values
    true_next = richest_person[feature_cols].iloc[1].values

    rich_norm = (rich_state - X_mean) / X_std
    rich_t = torch.tensor(rich_norm, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        h, _, q_pred = model.forward(rich_t)
        q_pred_exp = torch.expm1(torch.clamp(q_pred, max=20))
        q_pred_exp = torch.clamp(q_pred_exp, min=0, max=1e10)

    print("Rich holdout person:")
    print(f"  Input assets: ${rich_state[asset_start:].sum():,.0f}")
    print(f"  True output assets: ${true_next[asset_start:].sum():,.0f}")

    pred_median = q_pred_exp[0, asset_start:, 9].sum().item()
    pred_low = q_pred_exp[0, asset_start:, 0].sum().item()
    pred_high = q_pred_exp[0, asset_start:, -1].sum().item()
    print(f"  Predicted quantiles: q5=${pred_low:,.0f}, q50=${pred_median:,.0f}, q95=${pred_high:,.0f}")

    # Check individual non-zero assets
    print("\nIndividual non-zero assets:")
    for i in range(n_asset):
        col_idx = asset_start + i
        if rich_state[col_idx] > 0:
            true_val = true_next[col_idx]
            pred_q = q_pred_exp[0, col_idx, :].numpy()
            print(f"  asset_{i}: input=${rich_state[col_idx]:,.0f}, true_next=${true_val:,.0f}, "
                  f"pred q50=${pred_q[9]:,.0f}")

    # Check zero assets
    print("\nSample zero assets (should predict ~0):")
    zero_count = 0
    for i in range(n_asset):
        col_idx = asset_start + i
        if rich_state[col_idx] == 0 and zero_count < 5:
            true_val = true_next[col_idx]
            pred_q = q_pred_exp[0, col_idx, :].numpy()
            print(f"  asset_{i}: input=$0, true_next=${true_val:,.0f}, pred q50=${pred_q[9]:,.0f}")
            zero_count += 1

    # The issue: quantile head predicts marginal distribution, not conditional
    print("\n" + "=" * 70)
    print("DIAGNOSIS: Marginal vs Conditional Quantiles")
    print("=" * 70)

    # For each asset, what's the marginal q50 in training data?
    print("\nAsset-wise comparison (non-zero only):")
    for i in range(5):  # First 5 assets
        col_idx = asset_start + i
        nonzero_mask = Y[:, col_idx] > 0
        if nonzero_mask.sum() > 10:
            marginal_q50 = np.median(Y[:, col_idx][nonzero_mask])

            # What does model predict for average non-zero input?
            avg_nonzero_input = X[nonzero_mask].mean(axis=0)
            avg_norm = (avg_nonzero_input - X_mean) / X_std
            avg_t = torch.tensor(avg_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, q = model.forward(avg_t)
                q_exp = torch.expm1(torch.clamp(q, max=20))
                pred_q50 = q_exp[0, col_idx, 9].item()

            print(f"  asset_{i}: marginal_q50=${marginal_q50:,.0f}, pred_q50=${pred_q50:,.0f}")


if __name__ == "__main__":
    main()
