"""Transition-based ZI model.

Key insight: The zero/nonzero transition is a 2x2 Markov chain per feature:
- P(stay_zero | zero) ≈ 99.5%
- P(become_zero | nonzero) ≈ 0.5%

Instead of predicting P(zero) directly, predict transition probabilities
and apply the correct one based on current state.
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


class TransitionZIModel(nn.Module):
    """ZI model with separate transition heads.

    Instead of P(zero), predicts:
    - P(become_zero | was_nonzero) - should be ~0.5%
    - P(stay_zero | was_zero) - should be ~99.5%

    Then applies the correct transition based on current state.
    """

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

        # Two transition heads:
        # - become_zero_head: P(zero at t+1 | nonzero at t)
        # - stay_zero_head: P(zero at t+1 | zero at t)
        self.become_zero_head = nn.Linear(hidden_dim, n_features)
        self.stay_zero_head = nn.Linear(hidden_dim, n_features)

        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_quantiles),
        )

    def forward(self, x, is_nonzero=None):
        h = self.shared(x)
        become_zero_logits = self.become_zero_head(h)
        stay_zero_logits = self.stay_zero_head(h)
        q = self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)
        return h, become_zero_logits, stay_zero_logits, q

    def loss(self, x, target, is_nonzero_input):
        """Compute loss with correct transition labels.

        is_nonzero_input: whether input was nonzero (batch, n_features)
        """
        h, become_zero_logits, stay_zero_logits, q_pred = self.forward(x)

        # Target: is output zero?
        is_zero_output = (target == 0).float()

        # For samples where input was nonzero, use become_zero_head
        # For samples where input was zero, use stay_zero_head
        is_nonzero_input = is_nonzero_input.float()

        # Compute losses separately
        become_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            become_zero_logits, is_zero_output, reduction='none'
        )
        stay_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            stay_zero_logits, is_zero_output, reduction='none'
        )

        # Weight by which transition applies
        bce_loss = (is_nonzero_input * become_zero_loss + (1 - is_nonzero_input) * stay_zero_loss).mean()

        # Quantile loss
        target_log = torch.log1p(torch.clamp(target, min=0))
        errors = target_log.unsqueeze(-1) - q_pred
        ql_loss = torch.max((self.quantiles - 1) * errors, self.quantiles * errors).mean()

        return bce_loss + ql_loss

    def get_latent(self, x):
        with torch.no_grad():
            return self.shared(x)

    def get_p_zero(self, h, is_nonzero):
        """Get P(zero) using correct transition head."""
        with torch.no_grad():
            become_zero_logits = self.become_zero_head(h)
            stay_zero_logits = self.stay_zero_head(h)

            p_become_zero = torch.sigmoid(become_zero_logits)
            p_stay_zero = torch.sigmoid(stay_zero_logits)

            # Select correct probability based on current state
            is_nonzero = is_nonzero.float()
            p_zero = is_nonzero * p_become_zero + (1 - is_nonzero) * p_stay_zero
            return p_zero

    def sample_values(self, h):
        with torch.no_grad():
            q = self.quantile_head(h).view(-1, self.n_features, self.n_quantiles)
            q_exp = torch.expm1(torch.clamp(q, max=20))
            q_sorted = torch.clamp(torch.sort(q_exp, dim=-1)[0], min=0, max=1e10)
            idx = (torch.rand(h.shape[0], self.n_features, 1) * (self.n_quantiles - 1)).long()
            return q_sorted.gather(-1, idx).squeeze(-1)


class TransitionModel:
    """Model with separate transition heads."""

    def __init__(self, n_features):
        self.n_features = n_features
        self.feature_model = TransitionZIModel(n_features)

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
        is_nonzero_t = torch.tensor((X > 0), dtype=torch.float32)

        opt = torch.optim.Adam(self.feature_model.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad()
            self.feature_model.loss(X_t, Y_t, is_nonzero_t).backward()
            opt.step()

    def sample(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        is_nonzero = torch.tensor((x_raw > 0), dtype=torch.float32).unsqueeze(0)

        h = self.feature_model.get_latent(x_t)
        values = np.clip(self.feature_model.sample_values(h).numpy()[0], 0, 1e10)

        p_zero = self.feature_model.get_p_zero(h, is_nonzero).numpy()[0]
        for j in range(self.n_features):
            if np.random.random() < p_zero[j]:
                values[j] = 0
        return values

    def get_transition_probs(self, x_raw):
        """Get transition probabilities for debugging."""
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        is_nonzero = torch.tensor((x_raw > 0), dtype=torch.float32).unsqueeze(0)

        h = self.feature_model.get_latent(x_t)
        with torch.no_grad():
            p_become = torch.sigmoid(self.feature_model.become_zero_head(h)).numpy()[0]
            p_stay = torch.sigmoid(self.feature_model.stay_zero_head(h)).numpy()[0]
        return p_become, p_stay, is_nonzero.numpy()[0]


class OriginalModel:
    """Original model with single P(zero) head."""

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

    def get_p_zero(self, x_raw):
        x_norm = (x_raw - self.X_mean) / self.X_std
        x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            h = self.shared(x_t)
            return torch.sigmoid(self.zero_head(h)).numpy()[0]


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
    print("TRANSITION-BASED ZI MODEL")
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

    # Compute true transition rates
    transitions = {'stay_zero': 0, 'become_nonzero': 0, 'stay_nonzero': 0, 'become_zero': 0}
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[zero_cols].values
        for t in range(len(values) - 1):
            for j in range(len(zero_cols)):
                was_zero = values[t, j] == 0
                is_zero = values[t+1, j] == 0
                if was_zero and is_zero:
                    transitions['stay_zero'] += 1
                elif was_zero and not is_zero:
                    transitions['become_nonzero'] += 1
                elif not was_zero and not is_zero:
                    transitions['stay_nonzero'] += 1
                else:
                    transitions['become_zero'] += 1

    total_from_zero = transitions['stay_zero'] + transitions['become_nonzero']
    total_from_nonzero = transitions['stay_nonzero'] + transitions['become_zero']

    print("\nTrue transition rates:")
    print(f"  P(stay_zero | zero) = {transitions['stay_zero']/total_from_zero:.3f}")
    print(f"  P(become_zero | nonzero) = {transitions['become_zero']/total_from_nonzero:.3f}")

    # Train models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    n_synth = 5000
    n_runs = 3

    results = {}

    for name, Model in [("Original", OriginalModel), ("Transition", TransitionModel)]:
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

    # Test on rich person
    print("\n" + "=" * 70)
    print("TESTING ON RICH PERSON")
    print("=" * 70)

    # Find richest person
    holdout_totals = []
    for pid in holdout_df['person_id'].unique():
        person = holdout_df[holdout_df['person_id'] == pid]
        total = person[[f'asset_{i}' for i in range(n_asset)]].sum().sum()
        holdout_totals.append((pid, total))
    richest_pid = max(holdout_totals, key=lambda x: x[1])[0]
    richest_person = holdout_df[holdout_df['person_id'] == richest_pid].sort_values('period')
    rich_state = richest_person[feature_cols].iloc[0].values

    print(f"Rich person total assets: ${rich_state[asset_start:].sum():,.0f}")
    print(f"Non-zero assets: {(rich_state[asset_start:] > 0).sum()}/{n_asset}")

    # Train fresh models
    orig_model = OriginalModel(n_features)
    orig_model.fit(train_df, feature_cols, epochs=100)

    trans_model = TransitionModel(n_features)
    trans_model.fit(train_df, feature_cols, epochs=100)

    # Check P(zero) for original
    orig_pz = orig_model.get_p_zero(rich_state)
    print("\nOriginal model P(zero) for assets:")
    print(f"  Mean for non-zero assets: {orig_pz[asset_start:][rich_state[asset_start:] > 0].mean():.3f}")
    print(f"  Mean for zero assets: {orig_pz[asset_start:][rich_state[asset_start:] == 0].mean():.3f}")

    # Check transition probs
    p_become, p_stay, is_nz = trans_model.get_transition_probs(rich_state)
    print("\nTransition model:")
    print(f"  P(become_zero | nonzero) for assets: {p_become[asset_start:][rich_state[asset_start:] > 0].mean():.3f}")
    print(f"  P(stay_zero | zero) for assets: {p_stay[asset_start:][rich_state[asset_start:] == 0].mean():.3f}")
    print("  Expected: become_zero≈0.005, stay_zero≈0.995")

    # Generate from rich seed
    print("\n" + "=" * 70)
    print("GENERATION FROM RICH SEED")
    print("=" * 70)

    np.random.seed(42)

    print("\nOriginal model (5 samples):")
    for i in range(5):
        next_state = orig_model.sample(rich_state)
        next_total = next_state[asset_start:].sum()
        orig_ratio = next_total/max(rich_state[asset_start:].sum(), 1)
        print(f"  ${rich_state[asset_start:].sum():,.0f} → ${next_total:,.0f} (ratio: {orig_ratio:.2f})")

    print("\nTransition model (5 samples):")
    for i in range(5):
        next_state = trans_model.sample(rich_state)
        next_total = next_state[asset_start:].sum()
        trans_ratio = next_total/max(rich_state[asset_start:].sum(), 1)
        print(f"  ${rich_state[asset_start:].sum():,.0f} → ${next_total:,.0f} (ratio: {trans_ratio:.2f})")

    true_next = richest_person[feature_cols].iloc[1].values
    print(f"\nTrue: ${rich_state[asset_start:].sum():,.0f} → ${true_next[asset_start:].sum():,.0f} (ratio: {true_next[asset_start:].sum()/max(rich_state[asset_start:].sum(), 1):.2f})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"{name}: {r['median']:.2f} ± {r['std']:.2f}")


if __name__ == "__main__":
    main()
