"""ZI classifiers on LATENT space (hidden layer features).

Compare:
1. Logistic regression on raw inputs (current sklearn approach)
2. Logistic regression on latent space (current neural ZI head)
3. ML classifiers (GB, RF) on latent space

The hypothesis: ML classifiers on latent space may outperform
both raw-input classifiers AND simple logistic on latent.
"""

import sys
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


def generate_panel_with_zeros(n_persons: int = 500, T: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generate panel with realistic zero-inflation patterns."""
    np.random.seed(seed)
    records = []
    for pid in range(n_persons):
        age = np.random.randint(25, 60)
        employed = np.random.random() > 0.15
        income = max(0, 30000 + 1000 * (age - 25) + np.random.normal(0, 15000)) if employed else 0
        has_wealth = np.random.random() > 0.3
        wealth = np.random.lognormal(10, 1.5) if has_wealth else 0
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


class FeatureExtractor(nn.Module):
    """Neural network for extracting latent features."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class QuantileHead(nn.Module):
    """Quantile prediction head."""

    def __init__(self, hidden_dim: int, output_dim: int, n_quantiles: int = 19):
        super().__init__()
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles
        self.head = nn.Linear(hidden_dim, output_dim * n_quantiles)

    def forward(self, h):
        return self.head(h).view(-1, self.output_dim, self.n_quantiles)

    def loss(self, h, target):
        q_pred = self.forward(h)
        target_log = torch.log1p(torch.clamp(target, min=0))
        errors = target_log.unsqueeze(-1) - q_pred
        return torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        ).mean()

    def sample(self, h):
        with torch.no_grad():
            q = self.forward(h)
            q_clamped = torch.clamp(q, max=20)
            q_exp = torch.expm1(q_clamped)
            q_sorted = torch.sort(q_exp, dim=-1)[0]
            q_sorted = torch.clamp(q_sorted, min=0, max=1e10)
            u = torch.rand(h.shape[0], self.output_dim, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            return q_sorted.gather(-1, idx).squeeze(-1)


class LatentZIModel:
    """Model with ZI classifier on latent space.

    Architecture:
    1. Feature extractor (shared DNN) produces latent representation
    2. ZI classifier (sklearn) predicts P(zero) from latent features
    3. Quantile head predicts conditional distribution of non-zeros
    """

    def __init__(self, n_features: int, hidden_dim: int = 128, zi_classifier=None):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.feature_extractor = FeatureExtractor(n_features, hidden_dim)
        self.quantile_head = QuantileHead(hidden_dim, n_features)
        self.zi_classifier = zi_classifier  # sklearn classifier or None
        self.zi_models = None
        self.X_mean = None
        self.X_std = None

    def fit(self, train_df, feature_cols, epochs=100):
        """Train feature extractor, then ZI classifier on latent space."""
        # Prepare transitions
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

        # Step 1: Train feature extractor + quantile head jointly
        params = list(self.feature_extractor.parameters()) + list(self.quantile_head.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        for epoch in range(epochs):
            optimizer.zero_grad()
            h = self.feature_extractor(X_torch)
            loss = self.quantile_head.loss(h, Y_torch)
            loss.backward()
            optimizer.step()

        # Step 2: Extract latent features for ZI classifier training
        with torch.no_grad():
            H = self.feature_extractor(X_torch).numpy()

        # Step 3: Train ZI classifier on latent features
        if self.zi_classifier is not None:
            self.zi_models = []
            for j in range(self.n_features):
                is_zero = (Y[:, j] == 0).astype(int)
                if is_zero.sum() > 0 and is_zero.sum() < len(is_zero):
                    clf = self._clone_classifier()
                    clf.fit(H, is_zero)
                    self.zi_models.append(clf)
                else:
                    self.zi_models.append(None)
        else:
            self.zi_models = [None] * self.n_features

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
                max_depth=clf.max_depth
            )
        else:
            raise ValueError(f"Unknown classifier: {type(clf)}")

    def sample(self, x_raw):
        """Generate sample from model."""
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            h = self.feature_extractor(x_torch)
            values = self.quantile_head.sample(h).numpy()[0]
            values = np.clip(values, 0, 1e10)
            h_np = h.numpy()

        # Apply ZI classification on latent features
        for j, clf in enumerate(self.zi_models):
            if clf is not None:
                p_zero = clf.predict_proba(h_np)[0, 1]
                if np.random.random() < p_zero:
                    values[j] = 0

        return values


class JointNeuralZI(nn.Module):
    """Jointly trained neural ZI (baseline from previous experiment)."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Simple logistic regression on latent for ZI
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
        return zero_logits, q

    def loss(self, x, target):
        zero_logits, q_pred = self.forward(x)

        is_zero = (target == 0).float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='mean'
        )

        non_zero_mask = (target > 0)
        if non_zero_mask.any():
            target_log = torch.log1p(target)
            ql_total = 0
            count = 0
            for j in range(self.n_features):
                mask_j = non_zero_mask[:, j]
                if mask_j.sum() > 0:
                    target_j = target_log[mask_j, j].unsqueeze(-1)
                    pred_j = q_pred[mask_j, j, :]
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

            q_clamped = torch.clamp(q, max=20)
            q_exp = torch.expm1(q_clamped)
            q_sorted = torch.sort(q_exp, dim=-1)[0]
            q_sorted = torch.clamp(q_sorted, min=0, max=1e10)
            u = torch.rand(x.shape[0], self.n_features, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            samples = q_sorted.gather(-1, idx).squeeze(-1)

            is_zero = torch.rand_like(p_zero) < p_zero
            samples = torch.where(is_zero, torch.zeros_like(samples), samples)

            return samples


class JointNeuralZIWrapper:
    """Wrapper to match the interface."""

    def __init__(self, n_features, hidden_dim=128):
        self.model = JointNeuralZI(n_features, hidden_dim)
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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.model.loss(X_torch, Y_torch)
            loss.backward()
            optimizer.step()

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        samples = self.model.sample(x_torch).numpy()[0]
        return np.clip(samples, 0, 1e10)


def generate_panel(model, train_df, feature_cols, n_synth, T, seed=42):
    """Generate synthetic panel."""
    np.random.seed(seed)

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
            state = np.clip(state, 0, 1e10)
            state = np.nan_to_num(state, nan=0, posinf=1e10, neginf=0)

            records.append({
                'person_id': pid,
                'period': t,
                **{col: float(state[i]) for i, col in enumerate(feature_cols)}
            })

            if t < T - 1:
                state = model.sample(state)
                state = np.clip(state, 0, 1e10)

    return pd.DataFrame(records)


def compute_coverage_with_indicators(holdout_df, synth_df, train_df, base_cols, zero_cols):
    """Compute coverage distance INCLUDING zero indicators."""
    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nonzero'] = (df[col] > 0).astype(float)
        return df

    train_aug = augment(train_df)
    holdout_aug = augment(holdout_df)
    synth_aug = augment(synth_df)

    eval_cols = base_cols + [f'{c}_nonzero' for c in zero_cols]

    def to_matrix(df):
        matrices = []
        for pid in sorted(df['person_id'].unique()):
            person = df[df['person_id'] == pid].sort_values('period')
            matrices.append(person[eval_cols].values.flatten())
        return np.array(matrices)

    train_mat = to_matrix(train_aug)
    holdout_mat = to_matrix(holdout_aug)
    synth_mat = to_matrix(synth_aug)

    scaler = StandardScaler().fit(train_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn.kneighbors(holdout_scaled)

    return distances[:, 0]


def main():
    print("=" * 70)
    print("ZI CLASSIFIERS ON LATENT SPACE")
    print("=" * 70)

    df = generate_panel_with_zeros(n_persons=500, T=12, seed=42)

    base_cols = ['age', 'income', 'net_worth', 'dividend_income']
    zero_cols = ['income', 'net_worth', 'dividend_income']

    print("\nTrue zero rates:")
    for col in zero_cols:
        print(f"  {col}: {(df[col] == 0).mean():.1%}")

    persons = df['person_id'].unique()
    np.random.shuffle(persons)
    train_df = df[df['person_id'].isin(persons[:400])]
    holdout_df = df[df['person_id'].isin(persons[400:])]

    n_features = len(base_cols)
    n_synth = 500

    models = {
        'No ZI': LatentZIModel(n_features, zi_classifier=None),
        'Joint Neural ZI (logistic on latent)': JointNeuralZIWrapper(n_features),
        'Latent + Logistic Regression': LatentZIModel(
            n_features, zi_classifier=LogisticRegression(max_iter=1000)
        ),
        'Latent + Gradient Boosting': LatentZIModel(
            n_features, zi_classifier=GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1
            )
        ),
        'Latent + Random Forest': LatentZIModel(
            n_features, zi_classifier=RandomForestClassifier(
                n_estimators=100, max_depth=5
            )
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {name}")
        print("=" * 70)

        model.fit(train_df, base_cols, epochs=100)

        print(f"Generating {n_synth} synthetics...")
        synth_df = generate_panel(model, train_df, base_cols, n_synth, 12, seed=123)

        print("\nZero rates:")
        zero_rates = {}
        for col in zero_cols:
            rate = (synth_df[col] == 0).mean()
            zero_rates[col] = rate
            print(f"  {col}: {rate:.1%}")

        dist = compute_coverage_with_indicators(holdout_df, synth_df, train_df, base_cols, zero_cols)

        print("\nCoverage (with zero indicators):")
        print(f"  median: {np.median(dist):.2f}")
        print(f"  p90:    {np.percentile(dist, 90):.2f}")

        results.append({
            'model': name,
            'median_dist': np.median(dist),
            'p90_dist': np.percentile(dist, 90),
            **{f'{c}_zeros': zero_rates[c] for c in zero_cols}
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<40} {'Median':>8} {'P90':>8} {'Income':>8} {'Wealth':>8} {'Divid':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<40} {r['median_dist']:>8.2f} {r['p90_dist']:>8.2f} "
              f"{r['income_zeros']:>7.1%} {r['net_worth_zeros']:>7.1%} {r['dividend_income_zeros']:>7.1%}")

    print("\nTrue zero rates: income=23.8%, wealth=32.6%, dividends=71.8%")

    best = min(results, key=lambda x: x['median_dist'])
    print(f"\n✓ Best: {best['model']} (median={best['median_dist']:.2f})")


if __name__ == "__main__":
    main()
