"""Inspect hardest holdouts at scale (41 features)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
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


class JointFeatureExtractor(nn.Module):
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
    def __init__(self, n_features, zi_classifier=None):
        self.n_features = n_features
        self.feature_model = JointFeatureExtractor(n_features)
        self.zi_classifier = zi_classifier
        self.zi_models = None

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
        X_t, Y_t = torch.tensor(X_norm, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

        opt = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            self.feature_model.loss(X_t, Y_t).backward()
            opt.step()

        H = self.feature_model.get_latent(X_t).numpy()
        if self.zi_classifier:
            self.zi_models = []
            for j in range(self.n_features):
                is_zero = (Y[:, j] == 0).astype(int)
                if 0 < is_zero.sum() < len(is_zero):
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(H, is_zero)
                    self.zi_models.append(clf)
                else:
                    self.zi_models.append(None)
        else:
            self.zi_models = 'neural'

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        h = self.feature_model.get_latent(x_t)
        values = np.clip(self.feature_model.sample_values(h).numpy()[0], 0, 1e10)

        if self.zi_models == 'neural':
            with torch.no_grad():
                _, zl, _ = self.feature_model(x_t)
                p_zero = torch.sigmoid(zl).numpy()[0]
            for j in range(self.n_features):
                if np.random.random() < p_zero[j]:
                    values[j] = 0
        else:
            h_np = h.numpy()
            for j, clf in enumerate(self.zi_models):
                if clf and np.random.random() < clf.predict_proba(h_np)[0, 1]:
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


def main():
    print("=" * 70)
    print("INSPECTING HARD HOLDOUTS (41 features)")
    print("=" * 70)

    n_income, n_asset = 20, 20
    df = generate_rich_panel(n_persons=500, T=12, n_income_types=n_income, n_asset_types=n_asset, seed=42)

    feature_cols = ['age'] + [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    zero_cols = [f'income_{i}' for i in range(n_income)] + [f'asset_{i}' for i in range(n_asset)]
    n_features = len(feature_cols)

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    print(f"\nTraining model ({n_features} features)...")
    model = Model(n_features, zi_classifier=LogisticRegression(max_iter=1000))
    model.fit(train_df, feature_cols, epochs=100)

    print("Generating 5000 synthetics...")
    synth_df = generate_synth(model, train_df, feature_cols, 5000, 12, seed=123)

    # Compute distances with zero indicators
    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nz'] = (df[col] > 0).astype(float)
        return df

    eval_cols = feature_cols + [f'{c}_nz' for c in zero_cols]

    def to_matrix(df):
        return np.array([df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
                        for pid in sorted(df['person_id'].unique())])

    train_aug = augment(train_df)
    holdout_aug = augment(holdout_df)
    synth_aug = augment(synth_df)

    train_mat = to_matrix(train_aug)
    holdout_mat = to_matrix(holdout_aug)
    synth_mat = to_matrix(synth_aug)

    scaler = StandardScaler().fit(train_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn_model = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, indices = nn_model.kneighbors(holdout_scaled)

    holdout_pids = sorted(holdout_df['person_id'].unique())
    synth_pids = sorted(synth_df['person_id'].unique())

    # Find hardest holdouts
    hard_idx = np.argsort(distances[:, 0])[-10:][::-1]

    print(f"\n{'=' * 70}")
    print("TOP 10 HARDEST HOLDOUTS")
    print(f"{'=' * 70}")

    for rank, idx in enumerate(hard_idx):
        holdout_pid = holdout_pids[idx]
        nearest_synth_idx = indices[idx, 0]
        nearest_synth_pid = synth_pids[nearest_synth_idx]
        dist = distances[idx, 0]

        holdout_person = holdout_df[holdout_df['person_id'] == holdout_pid]
        synth_person = synth_df[synth_df['person_id'] == nearest_synth_pid]

        # Summary stats
        h_age = holdout_person['age'].mean()
        s_age = synth_person['age'].mean()

        h_nonzero_income = sum((holdout_person[f'income_{i}'] > 0).any() for i in range(n_income))
        s_nonzero_income = sum((synth_person[f'income_{i}'] > 0).any() for i in range(n_income))

        h_nonzero_asset = sum((holdout_person[f'asset_{i}'] > 0).any() for i in range(n_asset))
        s_nonzero_asset = sum((synth_person[f'asset_{i}'] > 0).any() for i in range(n_asset))

        h_total_income = holdout_person[[f'income_{i}' for i in range(n_income)]].sum().sum()
        s_total_income = synth_person[[f'income_{i}' for i in range(n_income)]].sum().sum()

        h_total_asset = holdout_person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()
        s_total_asset = synth_person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()

        print(f"\n#{rank+1} Distance: {dist:.2f} (person {holdout_pid})")
        print(f"  HOLDOUT: age={h_age:.0f}, {h_nonzero_income} income sources, {h_nonzero_asset} asset types")
        print(f"           total_income=${h_total_income:,.0f}, total_assets=${h_total_asset:,.0f}")
        print(f"  SYNTH:   age={s_age:.0f}, {s_nonzero_income} income sources, {s_nonzero_asset} asset types")
        print(f"           total_income=${s_total_income:,.0f}, total_assets=${s_total_asset:,.0f}")

        # What's the gap?
        print(f"  GAP: income sources={abs(h_nonzero_income - s_nonzero_income)}, "
              f"asset types={abs(h_nonzero_asset - s_nonzero_asset)}")

    # Analyze patterns in hard vs easy holdouts
    print(f"\n{'=' * 70}")
    print("PATTERN ANALYSIS: What makes holdouts hard?")
    print(f"{'=' * 70}")

    easy_idx = np.argsort(distances[:, 0])[:10]

    def get_stats(idx_list):
        stats = []
        for idx in idx_list:
            pid = holdout_pids[idx]
            person = holdout_df[holdout_df['person_id'] == pid]
            n_income_active = sum((person[f'income_{i}'] > 0).any() for i in range(n_income))
            n_asset_active = sum((person[f'asset_{i}'] > 0).any() for i in range(n_asset))
            total_income = person[[f'income_{i}' for i in range(n_income)]].sum().sum()
            total_asset = person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()
            stats.append({
                'n_income': n_income_active,
                'n_asset': n_asset_active,
                'total_income': total_income,
                'total_asset': total_asset,
                'dist': distances[idx, 0],
            })
        return pd.DataFrame(stats)

    hard_stats = get_stats(hard_idx)
    easy_stats = get_stats(easy_idx)

    print("\nHARD holdouts (top 10):")
    print(f"  Avg income sources: {hard_stats['n_income'].mean():.1f}")
    print(f"  Avg asset types: {hard_stats['n_asset'].mean():.1f}")
    print(f"  Avg total income: ${hard_stats['total_income'].mean():,.0f}")
    print(f"  Avg total assets: ${hard_stats['total_asset'].mean():,.0f}")

    print("\nEASY holdouts (bottom 10):")
    print(f"  Avg income sources: {easy_stats['n_income'].mean():.1f}")
    print(f"  Avg asset types: {easy_stats['n_asset'].mean():.1f}")
    print(f"  Avg total income: ${easy_stats['total_income'].mean():,.0f}")
    print(f"  Avg total assets: ${easy_stats['total_asset'].mean():,.0f}")

    # Check training set distribution
    train_stats = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid]
        n_inc = sum((person[f'income_{i}'] > 0).any() for i in range(n_income))
        n_ast = sum((person[f'asset_{i}'] > 0).any() for i in range(n_asset))
        train_stats.append({'n_income': n_inc, 'n_asset': n_ast})
    train_stats = pd.DataFrame(train_stats)

    print("\nTRAINING SET distribution:")
    print(f"  Income sources: {train_stats['n_income'].min()}-{train_stats['n_income'].max()} "
          f"(mean {train_stats['n_income'].mean():.1f})")
    print(f"  Asset types: {train_stats['n_asset'].min()}-{train_stats['n_asset'].max()} "
          f"(mean {train_stats['n_asset'].mean():.1f})")


if __name__ == "__main__":
    main()
