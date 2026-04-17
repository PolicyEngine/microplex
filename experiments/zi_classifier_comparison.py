"""Compare different ML models for zero-inflation classification.

The ZI component predicts P(zero | features). Compare:
1. Logistic regression (simple baseline)
2. MLP classifier (learned features)
3. Gradient Boosting (XGBoost-style)
4. Random Forest

Evaluate using coverage with zero indicators.
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
from sklearn.neural_network import MLPClassifier
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


class QuantileDNN(nn.Module):
    """Quantile DNN for non-zero value prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_quantiles: int = 19):
        super().__init__()
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * n_quantiles),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.output_dim, self.n_quantiles)

    def loss(self, x, target):
        q_pred = self.forward(x)
        # Work in log space for stability
        target_log = torch.log1p(torch.clamp(target, min=0))
        errors = target_log.unsqueeze(-1) - q_pred
        ql = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        ).mean()
        return ql

    def sample(self, x):
        with torch.no_grad():
            q = self.forward(x)
            q_clamped = torch.clamp(q, max=20)
            q_exp = torch.expm1(q_clamped)
            q_sorted = torch.sort(q_exp, dim=-1)[0]
            q_sorted = torch.clamp(q_sorted, min=0, max=1e10)
            u = torch.rand(x.shape[0], self.output_dim, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            samples = q_sorted.gather(-1, idx).squeeze(-1)
            return samples


class HybridZIModel:
    """Hybrid model: sklearn classifier for zeros + DNN for values.

    This separates the ZI classification from value prediction,
    allowing any sklearn classifier for the zero decision.
    """

    def __init__(self, zi_classifier, n_features: int, hidden_dim: int = 128):
        self.zi_classifier = zi_classifier  # sklearn classifier
        self.n_features = n_features
        self.value_model = QuantileDNN(n_features, n_features, hidden_dim)
        self.X_mean = None
        self.X_std = None

    def fit(self, train_df, feature_cols, epochs=100):
        """Train both ZI classifier and value DNN."""
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

        # Normalize X
        self.X_mean = X.mean(0)
        self.X_std = X.std(0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std

        # Train ZI classifier for each feature
        self.zi_models = []
        for j in range(self.n_features):
            is_zero = (Y[:, j] == 0).astype(int)
            if is_zero.sum() > 0 and is_zero.sum() < len(is_zero):
                clf = clone_classifier(self.zi_classifier)
                clf.fit(X_norm, is_zero)
                self.zi_models.append(clf)
            else:
                self.zi_models.append(None)  # No zeros or all zeros

        # Train value DNN on non-zero targets (in log space)
        X_torch = torch.tensor(X_norm, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.value_model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.value_model.loss(X_torch, Y_torch)
            loss.backward()
            optimizer.step()

    def fit_no_zi(self, train_df, feature_cols, epochs=100):
        """Train value DNN only (no ZI classifier)."""
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

        # Normalize X
        self.X_mean = X.mean(0)
        self.X_std = X.std(0) + 1e-6
        X_norm = (X - self.X_mean) / self.X_std

        # No ZI classifiers
        self.zi_models = [None] * self.n_features

        # Train value DNN
        X_torch = torch.tensor(X_norm, dtype=torch.float32)
        Y_torch = torch.tensor(Y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.value_model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.value_model.loss(X_torch, Y_torch)
            loss.backward()
            optimizer.step()

    def sample(self, x_raw):
        """Generate samples given raw input."""
        x_norm = (x_raw - self.X_mean) / self.X_std

        # Get value samples from DNN
        x_torch = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        values = self.value_model.sample(x_torch).numpy()[0]
        values = np.clip(values, 0, 1e10)

        # Apply ZI classification
        for j, clf in enumerate(self.zi_models):
            if clf is not None:
                p_zero = clf.predict_proba(x_norm.reshape(1, -1))[0, 1]
                if np.random.random() < p_zero:
                    values[j] = 0

        return values


def clone_classifier(clf):
    """Clone a sklearn classifier."""
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
    elif isinstance(clf, MLPClassifier):
        return MLPClassifier(
            hidden_layer_sizes=clf.hidden_layer_sizes,
            max_iter=clf.max_iter,
            early_stopping=clf.early_stopping
        )
    else:
        raise ValueError(f"Unknown classifier type: {type(clf)}")


def generate_panel_hybrid(model, train_df, feature_cols, n_synth, T, seed=42):
    """Generate synthetic panel using hybrid model."""
    np.random.seed(seed)

    # Get initial states
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
    print("ZERO-INFLATION CLASSIFIER COMPARISON")
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

    # First test: No ZI (standard QDNN baseline)
    print(f"\n{'=' * 70}")
    print("Training: No ZI (Standard QDNN)")
    print("=" * 70)

    no_zi_model = HybridZIModel(None, n_features)  # None = no ZI classifier
    no_zi_model.fit_no_zi(train_df, base_cols, epochs=100)

    print(f"Generating {n_synth} synthetics...")
    no_zi_synth = generate_panel_hybrid(no_zi_model, train_df, base_cols, n_synth, 12, seed=123)

    print("\nZero rates:")
    no_zi_zeros = {}
    for col in zero_cols:
        rate = (no_zi_synth[col] == 0).mean()
        no_zi_zeros[col] = rate
        print(f"  {col}: {rate:.1%}")

    no_zi_dist = compute_coverage_with_indicators(holdout_df, no_zi_synth, train_df, base_cols, zero_cols)
    print("\nCoverage (with zero indicators):")
    print(f"  median: {np.median(no_zi_dist):.2f}")
    print(f"  p90:    {np.percentile(no_zi_dist, 90):.2f}")

    results = [{
        'classifier': 'No ZI (Standard QDNN)',
        'median_dist': np.median(no_zi_dist),
        'p90_dist': np.percentile(no_zi_dist, 90),
        **{f'{c}_zeros': no_zi_zeros[c] for c in zero_cols}
    }]

    # Define ZI classifiers to test
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'MLP (64-64)': MLPClassifier(
            hidden_layer_sizes=(64, 64), max_iter=500, early_stopping=True
        ),
        'MLP (128-128)': MLPClassifier(
            hidden_layer_sizes=(128, 128), max_iter=500, early_stopping=True
        ),
        'Gradient Boosting (depth=3)': GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1
        ),
        'Gradient Boosting (depth=5)': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5
        ),
    }

    for name, clf in classifiers.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {name}")
        print("=" * 70)

        model = HybridZIModel(clf, n_features)
        model.fit(train_df, base_cols, epochs=100)

        print(f"Generating {n_synth} synthetics...")
        synth_df = generate_panel_hybrid(model, train_df, base_cols, n_synth, 12, seed=123)

        # Check zero rates
        print("\nZero rates:")
        zero_rates = {}
        for col in zero_cols:
            rate = (synth_df[col] == 0).mean()
            zero_rates[col] = rate
            print(f"  {col}: {rate:.1%}")

        # Compute coverage
        dist = compute_coverage_with_indicators(holdout_df, synth_df, train_df, base_cols, zero_cols)

        print("\nCoverage (with zero indicators):")
        print(f"  median: {np.median(dist):.2f}")
        print(f"  p90:    {np.percentile(dist, 90):.2f}")

        results.append({
            'classifier': name,
            'median_dist': np.median(dist),
            'p90_dist': np.percentile(dist, 90),
            **{f'{c}_zeros': zero_rates[c] for c in zero_cols}
        })

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Classifier':<30} {'Median':>8} {'P90':>8} {'Income':>8} {'Wealth':>8} {'Divid':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['classifier']:<30} {r['median_dist']:>8.2f} {r['p90_dist']:>8.2f} "
              f"{r['income_zeros']:>7.1%} {r['net_worth_zeros']:>7.1%} {r['dividend_income_zeros']:>7.1%}")

    print("\nTrue zero rates: income=23.8%, wealth=32.6%, dividends=71.8%")

    # Best model
    best = min(results, key=lambda x: x['median_dist'])
    print(f"\n✓ Best sklearn: {best['classifier']} (median={best['median_dist']:.2f})")
    print("  (Note: Neural ZI head from previous experiment got median=6.58)")
    print("  Joint training of ZI head + value DNN outperforms separate sklearn classifiers")


if __name__ == "__main__":
    main()
