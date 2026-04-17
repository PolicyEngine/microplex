"""Debug raw quantile predictions (before expm1)."""

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
    print("DEBUG RAW QUANTILE PREDICTIONS")
    print("=" * 70)

    n_income, n_asset = 20, 20
    df = generate_rich_panel(n_persons=500, T=12, n_income_types=n_income, n_asset_types=n_asset, seed=42)

    feature_cols = ['age'] + [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    n_features = len(feature_cols)
    asset_start = 1 + n_income

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]

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

    # What are the target log1p values?
    Y_log = np.log1p(Y)
    print("\n" + "=" * 70)
    print("TARGET LOG1P DISTRIBUTION")
    print("=" * 70)

    for i in range(3):  # First 3 assets
        col_idx = asset_start + i
        nonzero_mask = Y[:, col_idx] > 0
        if nonzero_mask.sum() > 10:
            log_vals = Y_log[:, col_idx][nonzero_mask]
            print(f"asset_{i}: log1p min={log_vals.min():.2f}, median={np.median(log_vals):.2f}, max={log_vals.max():.2f}")
            print(f"         (raw: ${np.expm1(log_vals.min()):,.0f} to ${np.expm1(log_vals.max()):,.0f})")

    # Train model
    model = Model(n_features)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        opt.zero_grad()
        loss = model.loss(X_t, Y_t)
        loss.backward()
        opt.step()

    # Check raw quantile predictions
    print("\n" + "=" * 70)
    print("RAW QUANTILE PREDICTIONS (LOG1P SPACE)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        h, _, q_pred = model.forward(X_t)
        # q_pred is raw log1p predictions
        q_raw = q_pred.numpy()

    # For the richest training sample
    rich_idx = np.argmax(Y[:, asset_start:].sum(axis=1))
    print(f"\nRich training sample (idx={rich_idx}):")

    for i in range(3):  # First 3 assets
        col_idx = asset_start + i
        true_log = Y_log[rich_idx, col_idx]
        pred_raw = q_raw[rich_idx, col_idx, :]
        print(f"\n  asset_{i}:")
        print(f"    True log1p: {true_log:.2f} (raw: ${Y[rich_idx, col_idx]:,.0f})")
        print(f"    Pred q5: {pred_raw[0]:.2f}, q50: {pred_raw[9]:.2f}, q95: {pred_raw[18]:.2f}")
        print(f"    Pred range (raw): ${max(0, np.expm1(pred_raw[0])):,.0f} to ${np.expm1(min(20, pred_raw[18])):,.0f}")

    # Check what happens to the quantile loss
    print("\n" + "=" * 70)
    print("QUANTILE LOSS DECOMPOSITION")
    print("=" * 70)

    with torch.no_grad():
        _, _, q_pred = model.forward(X_t)
        target_log = torch.log1p(torch.clamp(Y_t, min=0))
        errors = target_log.unsqueeze(-1) - q_pred

        # Compute quantile loss per feature
        quantiles = torch.linspace(0.05, 0.95, 19)
        ql_per_feature = torch.max((quantiles - 1) * errors, quantiles * errors).mean(dim=(0, 2))

        print("\nQuantile loss by feature (top 10):")
        for idx in torch.argsort(ql_per_feature, descending=True)[:10]:
            if idx == 0:
                name = "age"
            elif idx < 1 + n_income:
                name = f"income_{idx - 1}"
            else:
                name = f"asset_{idx - 1 - n_income}"
            print(f"  {name}: {ql_per_feature[idx].item():.4f}")

    # The problem: quantile head sees all zeros and predicts low
    print("\n" + "=" * 70)
    print("PROBLEM: QUANTILE LOSS ON ZEROS VS NON-ZEROS")
    print("=" * 70)

    with torch.no_grad():
        Y_arr = Y_t.numpy()
        zero_mask = (Y_arr == 0)
        nonzero_mask = (Y_arr > 0)

        # Compute loss separately
        errors_np = errors.numpy()
        quantiles_np = quantiles.numpy()

        ql_on_zeros = []
        ql_on_nonzeros = []

        for j in range(n_features):
            z_mask = zero_mask[:, j]
            nz_mask = nonzero_mask[:, j]

            if z_mask.sum() > 0:
                err_z = errors_np[z_mask, j, :]
                ql_z = np.maximum((quantiles_np - 1) * err_z, quantiles_np * err_z).mean()
                ql_on_zeros.append(ql_z)

            if nz_mask.sum() > 0:
                err_nz = errors_np[nz_mask, j, :]
                ql_nz = np.maximum((quantiles_np - 1) * err_nz, quantiles_np * err_nz).mean()
                ql_on_nonzeros.append(ql_nz)

        print(f"Avg quantile loss on ZERO targets: {np.mean(ql_on_zeros):.4f}")
        print(f"Avg quantile loss on NON-ZERO targets: {np.mean(ql_on_nonzeros):.4f}")
        print(f"\nZero samples: {zero_mask.sum()}")
        print(f"Non-zero samples: {nonzero_mask.sum()}")
        print(f"Zero fraction: {zero_mask.sum() / (zero_mask.sum() + nonzero_mask.sum()):.1%}")

    # What should the model predict for zeros?
    print("\n" + "=" * 70)
    print("OPTIMAL PREDICTION FOR ZEROS")
    print("=" * 70)

    # If target is 0, log1p(0) = 0
    # Quantile loss for predicting q at target=0 is:
    # max((tau-1)*(-q), tau*(-q)) = max((1-tau)*q, -tau*q)
    # If q > 0: loss = (1-tau)*q (positive since we're above true)
    # If q < 0: loss = -tau*q (positive since we're below true)
    # Minimum is at q = 0

    print("For target=0 (log1p=0), optimal prediction is q=0 for all quantiles")
    print("This means the model should predict expm1(0)=0 for zero-valued targets")
    print("\nBUT: The quantile loss averages across all features equally")
    print("With 70% zeros, the model learns to predict ~0 for everything!")


if __name__ == "__main__":
    main()
