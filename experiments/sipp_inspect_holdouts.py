"""Inspect hard holdouts on real SIPP data."""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_loaders import load_sipp


def prepare_sipp_panel(sipp_df: pd.DataFrame) -> pd.DataFrame:
    df = sipp_df.copy()
    df['person_id'] = df['SSUID'].astype(str) + '_' + df['PNUM'].astype(str)
    df['period'] = (df['SWAVE'] - 1) * 12 + df['MONTHCODE']
    df['age'] = df['TAGE'].fillna(0)
    df['total_income'] = df['TPTOTINC'].fillna(0).clip(lower=0)
    for i in range(1, 4):
        col = f'TJB{i}_MSUM'
        if col in df.columns:
            df[f'job{i}_income'] = df[col].fillna(0).clip(lower=0)
        else:
            df[f'job{i}_income'] = 0
    df['tip_income'] = df['tip_income'].fillna(0).clip(lower=0) if 'tip_income' in df.columns else 0
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
        become_zero_loss = nn.functional.binary_cross_entropy_with_logits(become_zero_logits, is_zero_output, reduction='none')
        stay_zero_loss = nn.functional.binary_cross_entropy_with_logits(stay_zero_logits, is_zero_output, reduction='none')
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


def main():
    print("=" * 70)
    print("INSPECTING HARD HOLDOUTS ON REAL SIPP")
    print("=" * 70)

    print("\nLoading SIPP...")
    sipp_raw = load_sipp(sample_frac=0.5)
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    zero_cols = ['total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    n_features = len(feature_cols)
    n_periods = 6

    # Split
    persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(persons)
    n_train = int(len(persons) * 0.8)
    train_persons = persons[:n_train]
    holdout_persons = persons[n_train:]

    train_df = sipp[sipp['person_id'].isin(train_persons)]
    holdout_df = sipp[sipp['person_id'].isin(holdout_persons)]

    # Filter to complete panels
    def filter_complete(df, n_periods):
        periods_per_person = df.groupby('person_id')['period'].nunique()
        complete = periods_per_person[periods_per_person >= n_periods].index
        df = df[df['person_id'].isin(complete)]
        df = df.sort_values(['person_id', 'period']).groupby('person_id').head(n_periods)
        return df

    train_df = filter_complete(train_df, n_periods)
    holdout_df = filter_complete(holdout_df, n_periods)

    print(f"Train: {train_df['person_id'].nunique()} persons")
    print(f"Holdout: {holdout_df['person_id'].nunique()} persons")

    # Train model
    print("\nTraining ratio+transition model...")
    model = CombinedModel(n_features)
    model.fit(train_df, feature_cols, epochs=100)

    # Generate synthetics
    print("Generating synthetics...")
    n_synth = 2000
    synth_df = generate_synth(model, train_df, feature_cols, n_synth, n_periods, seed=42)

    # Compute coverage
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
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn_model = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, indices = nn_model.kneighbors(holdout_scaled)

    holdout_pids = sorted(holdout_df['person_id'].unique())
    synth_pids = sorted(synth_df['person_id'].unique())

    # Find hardest holdouts
    print("\n" + "=" * 70)
    print("TOP 5 HARDEST HOLDOUTS")
    print("=" * 70)

    hard_idx = np.argsort(distances[:, 0])[-5:][::-1]

    for rank, idx in enumerate(hard_idx):
        holdout_pid = holdout_pids[idx]
        nearest_synth_idx = indices[idx, 0]
        nearest_synth_pid = synth_pids[nearest_synth_idx]
        dist = distances[idx, 0]

        holdout_person = holdout_df[holdout_df['person_id'] == holdout_pid].sort_values('period')
        synth_person = synth_df[synth_df['person_id'] == nearest_synth_pid].sort_values('period')

        print(f"\n#{rank+1} Distance: {dist:.3f}")
        print(f"  HOLDOUT (person {holdout_pid}):")
        print(f"    Age: {holdout_person['age'].iloc[0]:.0f}")
        print(f"    Total income trajectory: {list(holdout_person['total_income'].round(0).astype(int))}")
        print(f"    Job1 income trajectory: {list(holdout_person['job1_income'].round(0).astype(int))}")

        print(f"  NEAREST SYNTHETIC (person {nearest_synth_pid}):")
        print(f"    Age: {synth_person['age'].iloc[0]:.0f}")
        print(f"    Total income trajectory: {list(synth_person['total_income'].round(0).astype(int))}")
        print(f"    Job1 income trajectory: {list(synth_person['job1_income'].round(0).astype(int))}")

    # Also show easiest for comparison
    print("\n" + "=" * 70)
    print("TOP 5 EASIEST HOLDOUTS (for comparison)")
    print("=" * 70)

    easy_idx = np.argsort(distances[:, 0])[:5]

    for rank, idx in enumerate(easy_idx):
        holdout_pid = holdout_pids[idx]
        dist = distances[idx, 0]
        holdout_person = holdout_df[holdout_df['person_id'] == holdout_pid].sort_values('period')

        print(f"\n#{rank+1} Distance: {dist:.3f}")
        print(f"  Age: {holdout_person['age'].iloc[0]:.0f}")
        print(f"  Total income: {list(holdout_person['total_income'].round(0).astype(int))}")

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Median distance: {np.median(distances):.3f}")
    print(f"Mean distance: {np.mean(distances):.3f}")
    print(f"Hard holdout distance: {distances[hard_idx[0], 0]:.3f}")
    print(f"Easy holdout distance: {distances[easy_idx[0], 0]:.3f}")


if __name__ == "__main__":
    main()
