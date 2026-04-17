"""Fair comparison: Train features for BOTH tasks, then compare classifiers.

Previous experiment was unfair:
- Joint neural: features optimized for zero + values
- Latent + RF: features optimized for values only, then RF added

Fair comparison:
1. Train feature extractor with BOTH losses (BCE + quantile)
2. Freeze features
3. Compare logistic vs RF vs GB on those same features
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


def generate_panel_with_zeros(n_persons: int = 500, T: int = 12, seed: int = 42) -> pd.DataFrame:
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


class JointFeatureExtractor(nn.Module):
    """Feature extractor trained with BOTH zero and quantile losses."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        # 3 hidden layers (same as 6.61 baseline)
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Zero head (for joint training)
        self.zero_head = nn.Linear(hidden_dim, n_features)

        # Quantile head (deeper, matches 6.61 baseline)
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

        # BCE loss for zero classification
        is_zero = (target == 0).float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='mean'
        )

        # Quantile loss (log space)
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
        """Sample from quantile head given latent features."""
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
    """Model with swappable ZI classifier on jointly-trained features."""

    def __init__(self, n_features: int, hidden_dim: int = 128, zi_classifier=None):
        self.n_features = n_features
        self.feature_model = JointFeatureExtractor(n_features, hidden_dim)
        self.zi_classifier = zi_classifier
        self.zi_models = None
        self.X_mean = None
        self.X_std = None

    def fit(self, train_df, feature_cols, epochs=100):
        # Prepare data
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

        # Train with BOTH losses
        optimizer = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.feature_model.loss(X_torch, Y_torch)
            loss.backward()
            optimizer.step()

        # Extract jointly-trained latent features
        H = self.feature_model.get_latent(X_torch).numpy()

        # Train ZI classifier on these features
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
            # Use the neural zero head
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
                max_depth=clf.max_depth
            )
        raise ValueError(f"Unknown: {type(clf)}")

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_torch = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

        h = self.feature_model.get_latent(x_torch)
        values = self.feature_model.sample_values(h).numpy()[0]
        values = np.clip(values, 0, 1e10)

        if self.zi_models == 'neural':
            # Use neural zero head
            with torch.no_grad():
                _, zero_logits, _ = self.feature_model(x_torch)
                p_zero = torch.sigmoid(zero_logits).numpy()[0]
            for j in range(self.n_features):
                if np.random.random() < p_zero[j]:
                    values[j] = 0
        else:
            # Use sklearn classifier
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
    print("FAIR COMPARISON: Same jointly-trained features, different ZI classifiers")
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
    n_synth = 2000

    # All use the SAME jointly-trained features
    classifiers = {
        'Neural ZI head (logistic, joint grad)': None,  # Uses built-in zero head
        'Logistic Regression (frozen features)': LogisticRegression(max_iter=1000),
        'Gradient Boosting (frozen features)': GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1
        ),
        'Random Forest (frozen features)': RandomForestClassifier(
            n_estimators=100, max_depth=5
        ),
    }

    results = []
    n_runs = 3  # Fewer runs since n=2000 is slower

    for name, clf in classifiers.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {name} ({n_runs} runs)")
        print("=" * 70)

        medians = []
        train_times = []
        gen_times = []

        for run in range(n_runs):
            t0 = time.time()
            model = FairZIModel(n_features, zi_classifier=clf)
            model.fit(train_df, base_cols, epochs=100)
            train_times.append(time.time() - t0)

            t0 = time.time()
            synth_df = generate_panel(model, train_df, base_cols, n_synth, 12, seed=123 + run)
            gen_times.append(time.time() - t0)

            dist = compute_coverage(holdout_df, synth_df, train_df, base_cols, zero_cols)
            medians.append(np.median(dist))
            print(f"  Run {run+1}: median={np.median(dist):.2f}")

        avg_median = np.mean(medians)
        std_median = np.std(medians)
        avg_train = np.mean(train_times)
        avg_gen = np.mean(gen_times)

        print(f"  → Mean: {avg_median:.2f} ± {std_median:.2f}")

        results.append({
            'classifier': name,
            'median': avg_median,
            'std': std_median,
            'train_time': avg_train,
            'gen_time': avg_gen,
        })

    print("\n" + "=" * 70)
    print(f"SUMMARY ({n_runs} runs each)")
    print("=" * 70)

    print(f"\n{'Classifier':<40} {'Median':>12} {'Train':>7} {'Gen':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r['classifier']:<40} {r['median']:>5.2f} ± {r['std']:<5.2f} "
              f"{r['train_time']:>6.1f}s {r['gen_time']:>6.1f}s")

    print("\nTrue zero rates: income=23.8%, wealth=32.6%, dividends=71.8%")

    best = min(results, key=lambda x: x['median'])
    print(f"\n✓ Best: {best['classifier']} (median={best['median']:.2f})")


if __name__ == "__main__":
    main()
