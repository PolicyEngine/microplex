"""Scale test: More features, more synthetics.

Test ZI classifiers at realistic scale:
- 20-50 features (income types, assets, demographics)
- n=5000 synthetics
- Compare runtime and coverage
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def generate_rich_panel(n_persons: int = 500, T: int = 12, n_income_types: int = 5,
                        n_asset_types: int = 5, seed: int = 42) -> pd.DataFrame:
    """Generate panel with many zero-inflated features.

    Features:
    - age (continuous)
    - n_income_types income sources (many zeros)
    - n_asset_types asset types (many zeros)
    - Total: 1 + n_income_types + n_asset_types features
    """
    np.random.seed(seed)
    records = []

    for pid in range(n_persons):
        age = np.random.randint(25, 65)

        # Income sources - each has probability of being active
        income_probs = np.random.beta(2, 5, n_income_types)  # Most are low prob
        income_bases = np.random.lognormal(10, 1, n_income_types)
        income_active = np.random.random(n_income_types) < income_probs

        # Asset types - similar pattern
        asset_probs = np.random.beta(2, 5, n_asset_types)
        asset_bases = np.random.lognormal(11, 1.5, n_asset_types)
        asset_active = np.random.random(n_asset_types) < asset_probs

        for t in range(T):
            row = {
                'person_id': pid,
                'period': t,
                'age': age + t / 12,
            }

            # Income sources
            for i in range(n_income_types):
                if income_active[i]:
                    val = income_bases[i] * (1 + np.random.normal(0.02/12, 0.1))
                    income_bases[i] = val
                else:
                    val = 0
                # Small chance of status change
                if np.random.random() < 0.01:
                    income_active[i] = not income_active[i]
                    if income_active[i]:
                        income_bases[i] = np.random.lognormal(10, 1)
                row[f'income_{i}'] = max(0, val)

            # Asset types
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
    """Feature extractor for high-dimensional data."""

    def __init__(self, n_features: int, hidden_dim: int = 256, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        # Deeper for more features
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.zero_head = nn.Linear(hidden_dim, n_features)

        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

    def forward(self, x):
        h = self.shared(x)
        zero_logits = self.zero_head(h)
        q = self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)
        return h, zero_logits, q

    def loss(self, x, target):
        h, zero_logits, q_pred = self.forward(x)

        is_zero = (target == 0).float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='mean'
        )

        target_log = torch.log1p(torch.clamp(target, min=0))
        errors = target_log.unsqueeze(-1) - q_pred
        ql_loss = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        ).mean()

        return bce_loss + ql_loss

    def get_latent(self, x):
        with torch.no_grad():
            return self.shared(x)

    def sample_values(self, h):
        with torch.no_grad():
            q = self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)
            q_clamped = torch.clamp(q, max=20)
            q_exp = torch.expm1(q_clamped)
            q_sorted = torch.sort(q_exp, dim=-1)[0]
            q_sorted = torch.clamp(q_sorted, min=0, max=1e10)
            u = torch.rand(h.shape[0], self.n_features, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            return q_sorted.gather(-1, idx).squeeze(-1)


class FairZIModel:
    def __init__(self, n_features: int, hidden_dim: int = 256, zi_classifier=None):
        self.n_features = n_features
        self.feature_model = JointFeatureExtractor(n_features, hidden_dim)
        self.zi_classifier = zi_classifier
        self.zi_models = None
        self.X_mean = None
        self.X_std = None

    def fit(self, train_df, feature_cols, epochs=100):
        X_list, Y_list = [], []
        for pid in train_df['person_id'].unique():
            person = train_df[train_df['person_id'] == pid].sort_values('period')
            values = person[feature_cols].values
            for t in range(len(values) - 1):
                X_list.append(values[t])
                Y_list.append(values[t + 1])

        X = np.array(X_list)
        Y = np.array(Y_list)

        self.X_mean = X.mean(0)
        self.X_std = X.std(0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std

        X_torch = torch.tensor(X_norm, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.feature_model.loss(X_torch, Y_torch)
            loss.backward()
            optimizer.step()

        H = self.feature_model.get_latent(X_torch).numpy()

        if self.zi_classifier is not None:
            self.zi_models = []
            for j in range(self.n_features):
                is_zero = (Y[:, j] == 0).astype(int)
                if 0 < is_zero.sum() < len(is_zero):
                    clf = self._clone_classifier()
                    clf.fit(H, is_zero)
                    self.zi_models.append(clf)
                else:
                    self.zi_models.append(None)
        else:
            self.zi_models = 'neural'

    def _clone_classifier(self):
        clf = self.zi_classifier
        if isinstance(clf, LogisticRegression):
            return LogisticRegression(max_iter=1000)
        elif isinstance(clf, GradientBoostingClassifier):
            return GradientBoostingClassifier(
                n_estimators=clf.n_estimators,
                max_depth=clf.max_depth,
                learning_rate=clf.learning_rate
            )
        elif isinstance(clf, RandomForestClassifier):
            return RandomForestClassifier(
                n_estimators=clf.n_estimators,
                max_depth=clf.max_depth,
                n_jobs=-1  # Parallel for speed
            )
        raise ValueError(f"Unknown: {type(clf)}")

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

        h = self.feature_model.get_latent(x_torch)
        values = self.feature_model.sample_values(h).numpy()[0]
        values = np.clip(values, 0, 1e10)

        if self.zi_models == 'neural':
            with torch.no_grad():
                _, zero_logits, _ = self.feature_model(x_torch)
                p_zero = torch.sigmoid(zero_logits).numpy()[0]
            for j in range(self.n_features):
                if np.random.random() < p_zero[j]:
                    values[j] = 0
        else:
            h_np = h.numpy()
            for j, clf in enumerate(self.zi_models):
                if clf is not None:
                    p_zero = clf.predict_proba(h_np)[0, 1]
                    if np.random.random() < p_zero:
                        values[j] = 0

        return values


def generate_panel(model, train_df, feature_cols, n_synth, T, seed=42):
    np.random.seed(seed)
    init_states = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        init_states.append(person[feature_cols].iloc[0].values)
    init_states = np.array(init_states)

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


def main():
    print("=" * 70)
    print("SCALE TEST: More features, more synthetics")
    print("=" * 70)

    # Test configurations
    configs = [
        {'n_income': 5, 'n_asset': 5, 'n_synth': 2000},   # 11 features
        {'n_income': 10, 'n_asset': 10, 'n_synth': 2000}, # 21 features
        {'n_income': 20, 'n_asset': 20, 'n_synth': 5000}, # 41 features
    ]

    for config in configs:
        n_income = config['n_income']
        n_asset = config['n_asset']
        n_synth = config['n_synth']
        n_features = 1 + n_income + n_asset

        print(f"\n{'=' * 70}")
        print(f"CONFIG: {n_features} features, n={n_synth}")
        print("=" * 70)

        df = generate_rich_panel(n_persons=500, T=12,
                                 n_income_types=n_income,
                                 n_asset_types=n_asset, seed=42)

        feature_cols = ['age'] + [f'income_{i}' for i in range(n_income)] + \
                       [f'asset_{i}' for i in range(n_asset)]
        zero_cols = [f'income_{i}' for i in range(n_income)] + \
                    [f'asset_{i}' for i in range(n_asset)]

        # Check zero rates
        avg_zero_rate = np.mean([(df[col] == 0).mean() for col in zero_cols])
        print(f"Average zero rate: {avg_zero_rate:.1%}")

        persons = df['person_id'].unique()
        np.random.shuffle(persons)
        train_df = df[df['person_id'].isin(persons[:400])]
        holdout_df = df[df['person_id'].isin(persons[400:])]

        classifiers = {
            'Neural ZI': None,
            'Logistic': LogisticRegression(max_iter=1000),
            'GB': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1),
            # RF skipped - too slow at scale
        }

        results = []

        for name, clf in classifiers.items():
            print(f"\n  {name}:")

            t0 = time.time()
            model = FairZIModel(n_features, hidden_dim=256, zi_classifier=clf)
            model.fit(train_df, feature_cols, epochs=100)
            train_time = time.time() - t0

            t0 = time.time()
            synth_df = generate_panel(model, train_df, feature_cols, n_synth, 12, seed=123)
            gen_time = time.time() - t0

            dist = compute_coverage(holdout_df, synth_df, train_df, feature_cols, zero_cols)
            median = np.median(dist)

            print(f"    median={median:.2f}, train={train_time:.1f}s, gen={gen_time:.1f}s")

            results.append({
                'config': f"{n_features}f/n={n_synth}",
                'classifier': name,
                'median': median,
                'train_time': train_time,
                'gen_time': gen_time,
            })

        # Summary for this config
        print(f"\n  Summary ({n_features} features):")
        for r in results[-3:]:
            total_time = r['train_time'] + r['gen_time']
            print(f"    {r['classifier']:<10}: {r['median']:.2f} ({total_time:.0f}s total)")


if __name__ == "__main__":
    main()
