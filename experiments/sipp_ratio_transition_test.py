"""Test ratio+transition model on real SIPP panel data."""

import sys

# Unbuffer output
sys.stdout.reconfigure(line_buffering=True)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_loaders import load_sipp


def prepare_sipp_panel(sipp_df: pd.DataFrame) -> pd.DataFrame:
    """Convert SIPP to panel format with person_id and period."""
    df = sipp_df.copy()

    # Create person_id from household + person number
    df['person_id'] = df['SSUID'].astype(str) + '_' + df['PNUM'].astype(str)

    # Create period from wave and month
    df['period'] = (df['SWAVE'] - 1) * 12 + df['MONTHCODE']

    # Extract features
    # Age
    df['age'] = df['TAGE'].fillna(0)

    # Total income (many zeros for unemployed/retired)
    df['total_income'] = df['TPTOTINC'].fillna(0).clip(lower=0)

    # Job 1-3 earnings (highly sparse - most people don't have multiple jobs)
    for i in range(1, 4):
        col = f'TJB{i}_MSUM'
        if col in df.columns:
            df[f'job{i}_income'] = df[col].fillna(0).clip(lower=0)
        else:
            df[f'job{i}_income'] = 0

    # Tips income (derived, very sparse)
    df['tip_income'] = df['tip_income'].fillna(0).clip(lower=0) if 'tip_income' in df.columns else 0

    # Filter to people with at least 6 periods of data
    periods_per_person = df.groupby('person_id')['period'].nunique()
    valid_persons = periods_per_person[periods_per_person >= 6].index
    df = df[df['person_id'].isin(valid_persons)]

    return df


class RatioTransitionModel(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 128, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles
        self.input_dim = n_features + n_features

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.become_zero_head = nn.Linear(hidden_dim, n_features)
        self.stay_zero_head = nn.Linear(hidden_dim, n_features)
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )
        self.init_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

    def forward(self, x_aug):
        h = self.shared(x_aug)
        return (h, self.become_zero_head(h), self.stay_zero_head(h),
                self.ratio_head(h).view(-1, self.n_features, self.n_quantiles),
                self.init_head(h).view(-1, self.n_features, self.n_quantiles))

    def loss(self, x_raw, x_aug, target):
        h, become_zero_logits, stay_zero_logits, ratio_q, init_q = self.forward(x_aug)

        is_zero_output = (target == 0).float()
        is_nonzero_input = (x_raw > 0).float()

        become_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            become_zero_logits, is_zero_output, reduction='none')
        stay_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            stay_zero_logits, is_zero_output, reduction='none')
        bce = (is_nonzero_input * become_zero_loss + (1 - is_nonzero_input) * stay_zero_loss).mean()

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
    def __init__(self, n_features):
        self.n_features = n_features
        self.feature_model = RatioTransitionModel(n_features)

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

        x_aug_t = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, become_zero_logits, stay_zero_logits, ratio_q, init_q = self.feature_model.forward(x_aug_t)
            p_become_zero = torch.sigmoid(become_zero_logits).numpy()[0]
            p_stay_zero = torch.sigmoid(stay_zero_logits).numpy()[0]

            n_quantiles = ratio_q.shape[-1]
            idx = (torch.rand(1, self.n_features, 1) * (n_quantiles - 1)).long()
            ratio_samples = ratio_q.gather(-1, idx).squeeze(-1)
            ratios = torch.exp(ratio_samples).numpy()[0]
            init_samples = init_q.gather(-1, idx).squeeze(-1)
            init_vals = torch.expm1(torch.clamp(init_samples, max=20)).numpy()[0]

        values = np.zeros(self.n_features)
        for j in range(self.n_features):
            if x_raw[j] > 0:
                if np.random.random() < p_become_zero[j]:
                    values[j] = 0
                else:
                    values[j] = x_raw[j] * ratios[j]
            else:
                if np.random.random() < p_stay_zero[j]:
                    values[j] = 0
                else:
                    values[j] = max(0, init_vals[j])
        return np.clip(values, 0, 1e10)


class OriginalModel:
    def __init__(self, n_features):
        self.n_features = n_features
        self.shared = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.zero_head = nn.Linear(128, n_features)
        self.quantile_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_features * 19),
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


def compute_coverage(holdout_df, synth_df, train_df, base_cols, zero_cols):
    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nonzero'] = (df[col] > 0).astype(float)
        return df

    eval_cols = base_cols + [f'{c}_nonzero' for c in zero_cols]

    def to_matrix(df):
        return np.array([df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
                        for pid in sorted(df['person_id'].unique())])

    train_mat = to_matrix(augment(train_df))
    holdout_mat = to_matrix(augment(holdout_df))
    synth_mat = to_matrix(augment(synth_df))

    scaler = StandardScaler().fit(train_mat)
    nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(synth_mat))
    distances, _ = nn.kneighbors(scaler.transform(holdout_mat))
    return distances[:, 0]


def main(sample_frac: float = 0.5, n_periods: int = 6, n_synth: int = 2000, n_runs: int = 3):
    """
    Args:
        sample_frac: Fraction of SIPP to load (0.0-1.0)
        n_periods: Number of periods per person to require
        n_synth: Number of synthetic records to generate
        n_runs: Number of runs for each model
    """
    print("=" * 70)
    print("RATIO+TRANSITION MODEL ON REAL SIPP DATA")
    print("=" * 70)

    print(f"\nLoading SIPP ({sample_frac*100:.0f}% sample)...")
    sipp_raw = load_sipp(sample_frac=sample_frac)

    # Prepare panel
    print("\nPreparing panel structure...")
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    zero_cols = ['total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    n_features = len(feature_cols)

    print("\nPanel stats:")
    print(f"  Records: {len(sipp):,}")
    print(f"  Unique persons: {sipp['person_id'].nunique():,}")
    print(f"  Periods per person: {sipp.groupby('person_id')['period'].nunique().median():.0f} (median)")

    print("\nZero rates:")
    for col in zero_cols:
        print(f"  {col}: {(sipp[col] == 0).mean():.1%}")

    # Train/holdout split by person
    persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(persons)
    n_train = int(len(persons) * 0.8)
    train_persons = persons[:n_train]
    holdout_persons = persons[n_train:]

    train_df = sipp[sipp['person_id'].isin(train_persons)]
    holdout_df = sipp[sipp['person_id'].isin(holdout_persons)]

    print(f"\nTrain: {len(train_persons):,} persons, {len(train_df):,} records")
    print(f"Holdout: {len(holdout_persons):,} persons, {len(holdout_df):,} records")

    # Standardize period length - keep only people with exactly 12 periods
    def filter_complete_panels(df, n_periods=12):
        periods_per_person = df.groupby('person_id')['period'].nunique()
        complete_persons = periods_per_person[periods_per_person >= n_periods].index
        df = df[df['person_id'].isin(complete_persons)]
        # Keep first n_periods for each person
        df = df.sort_values(['person_id', 'period'])
        df = df.groupby('person_id').head(n_periods)
        return df

    train_df = filter_complete_panels(train_df, n_periods)
    holdout_df = filter_complete_panels(holdout_df, n_periods)

    print(f"\nAfter filtering to {n_periods} periods:")
    print(f"  Train: {train_df['person_id'].nunique():,} persons")
    print(f"  Holdout: {holdout_df['person_id'].nunique():,} persons")

    n_synth = min(n_synth, holdout_df['person_id'].nunique() * 2)

    results = {}

    for name, Model in [("Original ZI-QDNN", OriginalModel), ("Ratio+Transition", CombinedModel)]:
        print(f"\n{name}:")
        medians = []

        for run in range(n_runs):
            model = Model(n_features)
            model.fit(train_df, feature_cols, epochs=100)
            synth_df = generate_synth(model, train_df, feature_cols, n_synth, n_periods, seed=123 + run)
            dist = compute_coverage(holdout_df, synth_df, train_df, feature_cols, zero_cols)
            medians.append(np.median(dist))
            print(f"  Run {run+1}: {np.median(dist):.2f}")

        results[name] = {'median': np.mean(medians), 'std': np.std(medians)}
        print(f"  → Mean: {np.mean(medians):.2f} ± {np.std(medians):.2f}")

    print("\n" + "=" * 70)
    print(f"SUMMARY (REAL SIPP, {n_features} features, n={n_synth}, {n_runs} runs)")
    print("=" * 70)
    for name, r in results.items():
        print(f"{name}: {r['median']:.2f} ± {r['std']:.2f}")

    if results['Ratio+Transition']['median'] < results['Original ZI-QDNN']['median']:
        improvement = (results['Original ZI-QDNN']['median'] - results['Ratio+Transition']['median']) / results['Original ZI-QDNN']['median'] * 100
        print(f"\n✓ Ratio+Transition improves by {improvement:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test ratio+transition model on SIPP")
    parser.add_argument("--sample-frac", type=float, default=0.5, help="Fraction of SIPP to load")
    parser.add_argument("--n-periods", type=int, default=6, help="Periods per person")
    parser.add_argument("--n-synth", type=int, default=2000, help="Synthetic records to generate")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()
    main(sample_frac=args.sample_frac, n_periods=args.n_periods, n_synth=args.n_synth, n_runs=args.n_runs)
