"""Compare QDNN vs RNN-based models for time series panel synthesis.

Compares:
1. ZI-QDNN (Zero-Inflated Quantile DNN) - current approach
2. GRU-based transition model - recurrent alternative

Metrics:
- Trajectory coverage (median NN distance to holdout)
- Zero-rate accuracy (does model generate realistic zero patterns?)
- Training time
- Generation time

Uses real SIPP panel data.
"""

import sys

sys.stdout.reconfigure(line_buffering=True)

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.sipp_inspect_holdouts import prepare_sipp_panel
from pipelines.data_loaders import load_sipp

# =============================================================================
# Model 1: ZI-QDNN (Zero-Inflated Quantile DNN)
# =============================================================================

class ZeroInflatedQDNN(nn.Module):
    """Quantile DNN with explicit zero-inflation head.

    Architecture:
      - Shared hidden layers
      - Zero head: P(zero | x) via logistic regression on hidden features
      - Quantile head: Conditional quantiles for non-zero values (log space)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_quantiles: int = 19):
        super().__init__()
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Zero-inflation head
        self.zero_head = nn.Linear(hidden_dim, output_dim)

        # Quantile head for non-zero values (outputs log-scale)
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * n_quantiles),
        )

    def forward(self, x):
        h = self.shared(x)
        zero_logits = self.zero_head(h)
        q = self.quantile_head(h).view(-1, self.output_dim, self.n_quantiles)
        return zero_logits, q

    def loss(self, x, target):
        zero_logits, q_pred = self.forward(x)

        # Binary cross-entropy for zero classification
        is_zero = (target == 0).float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='mean'
        )

        # Quantile loss only on non-zero values (log space)
        non_zero_mask = (target > 0)
        if non_zero_mask.any():
            target_log = torch.log1p(target)
            ql_total = 0
            count = 0
            for j in range(self.output_dim):
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

            # Convert from log space
            q_clamped = torch.clamp(q, max=20)
            q_exp = torch.expm1(q_clamped)
            q_sorted = torch.sort(q_exp, dim=-1)[0]
            q_sorted = torch.clamp(q_sorted, min=0, max=1e10)

            # Sample from quantiles
            u = torch.rand(x.shape[0], self.output_dim, 1)
            idx = (u * (self.n_quantiles - 1)).long()
            samples = q_sorted.gather(-1, idx).squeeze(-1)

            # Zero out based on zero probability
            is_zero = torch.rand_like(p_zero) < p_zero
            samples = torch.where(is_zero, torch.zeros_like(samples), samples)

            return samples


# =============================================================================
# Model 2: GRU-based Transition Model
# =============================================================================

class GRUTransitionModel(nn.Module):
    """GRU-based transition model for panel synthesis.

    Uses GRU to encode history, then predicts next state with:
    - Zero head: P(zero | hidden)
    - Value head: Gaussian mixture for non-zero values

    Key difference from QDNN: Uses recurrent structure to model
    temporal dependencies, rather than feedforward with single-step input.
    """

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # GRU encoder
        self.gru = nn.GRU(
            input_size=n_features + n_features,  # features + zero indicators
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0,
        )

        # Output heads
        self.zero_head = nn.Linear(hidden_dim, n_features)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.logstd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x_seq, indicators_seq):
        """Forward pass.

        Args:
            x_seq: (batch, seq_len, n_features) - normalized features
            indicators_seq: (batch, seq_len, n_features) - zero indicators

        Returns:
            zero_logits: (batch, n_features)
            means: (batch, n_features) - in log space
            log_stds: (batch, n_features)
        """
        # Concatenate features and indicators
        combined = torch.cat([x_seq, indicators_seq], dim=-1)

        # GRU forward
        _, h_n = self.gru(combined)  # h_n: (n_layers, batch, hidden)
        h_last = h_n[-1]  # (batch, hidden)

        # Output heads
        zero_logits = self.zero_head(h_last)
        means = self.mean_head(h_last)  # Log-space means
        log_stds = self.logstd_head(h_last).clamp(-5, 2)

        return zero_logits, means, log_stds

    def loss(self, x_seq, indicators_seq, target):
        """Compute loss.

        Args:
            x_seq: Input sequences
            indicators_seq: Zero indicators
            target: Target values (original scale)
        """
        zero_logits, means, log_stds = self.forward(x_seq, indicators_seq)

        # BCE for zero classification
        is_zero = (target == 0).float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='mean'
        )

        # Gaussian NLL for non-zero values (log space)
        non_zero_mask = (target > 0)
        if non_zero_mask.any():
            target_log = torch.log1p(target)
            stds = torch.exp(log_stds)

            # Compute NLL only for non-zero elements
            nll = 0.5 * (((target_log - means) / stds) ** 2 + 2 * log_stds)
            nll_loss = (nll * non_zero_mask.float()).sum() / non_zero_mask.sum()
        else:
            nll_loss = 0

        return bce_loss + nll_loss

    def sample(self, x_seq, indicators_seq):
        """Sample next state given history."""
        with torch.no_grad():
            zero_logits, means, log_stds = self.forward(x_seq, indicators_seq)
            p_zero = torch.sigmoid(zero_logits)
            stds = torch.exp(log_stds)

            # Sample from Gaussian in log space
            log_samples = means + stds * torch.randn_like(means)
            samples = torch.expm1(torch.clamp(log_samples, max=20))
            samples = torch.clamp(samples, min=0, max=1e10)

            # Apply zero mask
            is_zero = torch.rand_like(p_zero) < p_zero
            samples = torch.where(is_zero, torch.zeros_like(samples), samples)

            return samples


class GRUTransitionWrapper:
    """Wrapper for GRU model with training and generation utilities."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 2,
                 context_length: int = 3):
        self.n_features = n_features
        self.context_length = context_length
        self.model = GRUTransitionModel(n_features, hidden_dim, n_layers)
        self.X_mean = None
        self.X_std = None

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str],
            epochs: int = 100, lr: float = 1e-3):
        """Train GRU model on panel data."""
        # Prepare sequences: for each person, create (history, target) pairs
        X_seqs, indicators_seqs, targets = [], [], []

        for pid in train_df['person_id'].unique():
            person = train_df[train_df['person_id'] == pid].sort_values('period')
            values = person[feature_cols].values

            # Create sequences with context window
            for t in range(self.context_length, len(values)):
                # History window
                seq = values[t - self.context_length:t]
                X_seqs.append(seq)
                indicators_seqs.append((seq > 0).astype(float))
                targets.append(values[t])

        X_seqs = np.array(X_seqs)  # (n_samples, context_length, n_features)
        indicators_seqs = np.array(indicators_seqs)
        targets = np.array(targets)  # (n_samples, n_features)

        # Normalize
        self.X_mean = X_seqs.mean(axis=(0, 1))
        self.X_std = X_seqs.std(axis=(0, 1)) + 1e-6

        X_norm = (X_seqs - self.X_mean) / self.X_std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        ind_t = torch.tensor(indicators_seqs, dtype=torch.float32)
        Y_t = torch.tensor(targets, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.model.loss(X_t, ind_t, Y_t)
            loss.backward()
            optimizer.step()

    def sample(self, history: np.ndarray) -> np.ndarray:
        """Sample next state given history.

        Args:
            history: (context_length, n_features) array
        """
        # Normalize
        history_norm = (history - self.X_mean) / self.X_std
        indicators = (history > 0).astype(float)

        x_t = torch.tensor(history_norm, dtype=torch.float32).unsqueeze(0)
        ind_t = torch.tensor(indicators, dtype=torch.float32).unsqueeze(0)

        next_state = self.model.sample(x_t, ind_t).numpy()[0]
        return np.clip(next_state, 0, 1e10)


# =============================================================================
# Training and Generation Functions
# =============================================================================

def train_qdnn(train_df: pd.DataFrame, feature_cols: list[str],
               epochs: int = 100) -> tuple[ZeroInflatedQDNN, np.ndarray, np.ndarray]:
    """Train ZI-QDNN model."""
    # Prepare transitions
    X_list, Y_list = [], []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        values = person[feature_cols].values
        for t in range(len(values) - 1):
            X_list.append(values[t])
            Y_list.append(values[t + 1])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    # Normalize inputs only (Y stays original for zero detection)
    X_mean = X.mean(0).numpy()
    X_std = (X.std(0) + 1e-6).numpy()
    X_norm = (X - torch.tensor(X_mean)) / torch.tensor(X_std)

    n_features = len(feature_cols)
    model = ZeroInflatedQDNN(n_features, n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(X_norm, Y)
        loss.backward()
        optimizer.step()

    return model, X_mean, X_std


def generate_qdnn(model: ZeroInflatedQDNN, train_df: pd.DataFrame,
                  feature_cols: list[str], n_synth: int, n_periods: int,
                  X_mean: np.ndarray, X_std: np.ndarray,
                  seed: int = 42) -> pd.DataFrame:
    """Generate synthetic panel using QDNN."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get initial states from training
    init_states = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        init_states.append(person[feature_cols].iloc[0].values)
    init_states = np.array(init_states)

    records = []
    for pid in range(n_synth):
        idx = np.random.randint(len(init_states))
        state = init_states[idx].copy()

        for t in range(n_periods):
            state = np.clip(np.nan_to_num(state, 0), 0, 1e10)
            records.append({
                'person_id': pid, 'period': t,
                **{col: float(state[i]) for i, col in enumerate(feature_cols)}
            })

            if t < n_periods - 1:
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                x_norm = (x - torch.tensor(X_mean)) / torch.tensor(X_std)
                next_state = model.sample(x_norm).numpy()[0]
                state = np.clip(next_state, 0, 1e10)

    return pd.DataFrame(records)


def generate_gru(model: GRUTransitionWrapper, train_df: pd.DataFrame,
                 feature_cols: list[str], n_synth: int, n_periods: int,
                 seed: int = 42) -> pd.DataFrame:
    """Generate synthetic panel using GRU."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    context_length = model.context_length

    # Get initial sequences (first context_length periods)
    init_seqs = []
    for pid in train_df['person_id'].unique():
        person = train_df[train_df['person_id'] == pid].sort_values('period')
        if len(person) >= context_length:
            init_seqs.append(person[feature_cols].iloc[:context_length].values)

    records = []
    for pid in range(n_synth):
        idx = np.random.randint(len(init_seqs))
        history = init_seqs[idx].copy()

        # Record initial states
        for t in range(context_length):
            records.append({
                'person_id': pid, 'period': t,
                **{col: float(history[t, i]) for i, col in enumerate(feature_cols)}
            })

        # Generate remaining periods
        for t in range(context_length, n_periods):
            next_state = model.sample(history)
            next_state = np.clip(np.nan_to_num(next_state, 0), 0, 1e10)

            records.append({
                'person_id': pid, 'period': t,
                **{col: float(next_state[i]) for i, col in enumerate(feature_cols)}
            })

            # Update history (sliding window)
            history = np.vstack([history[1:], next_state])

    return pd.DataFrame(records)


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_coverage(holdout_df: pd.DataFrame, synth_df: pd.DataFrame,
                     train_df: pd.DataFrame, feature_cols: list[str],
                     zero_cols: list[str]) -> dict[str, float]:
    """Compute coverage metrics including zero indicators.

    Returns:
        Dict with median, p90, max distances
    """
    def augment(df):
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nz'] = (df[col] > 0).astype(float)
        return df

    eval_cols = feature_cols + [f'{c}_nz' for c in zero_cols]

    def to_matrix(df):
        return np.array([
            df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
            for pid in sorted(df['person_id'].unique())
        ])

    train_mat = to_matrix(augment(train_df))
    holdout_mat = to_matrix(augment(holdout_df))
    synth_mat = to_matrix(augment(synth_df))

    scaler = StandardScaler().fit(train_mat)
    holdout_scaled = scaler.transform(holdout_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn_model = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn_model.kneighbors(holdout_scaled)

    return {
        'median': float(np.median(distances)),
        'p90': float(np.percentile(distances, 90)),
        'max': float(np.max(distances)),
        'mean': float(np.mean(distances)),
    }


def compute_zero_rate_accuracy(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                               zero_cols: list[str]) -> dict[str, float]:
    """Compare zero rates between real and synthetic data."""
    results = {}
    for col in zero_cols:
        real_rate = (real_df[col] == 0).mean()
        synth_rate = (synth_df[col] == 0).mean()
        results[f'{col}_real'] = real_rate
        results[f'{col}_synth'] = synth_rate
        results[f'{col}_abs_error'] = abs(real_rate - synth_rate)

    results['mean_abs_error'] = np.mean([results[f'{c}_abs_error'] for c in zero_cols])
    return results


# =============================================================================
# Main Comparison
# =============================================================================

def main(sample_frac: float = 0.5, n_periods: int = 6, n_synth: int = 2000,
         epochs: int = 100):
    print("=" * 70)
    print("QDNN vs RNN COMPARISON FOR PANEL SYNTHESIS")
    print("=" * 70)

    # Load data
    print("\nLoading SIPP data...")
    sipp_raw = load_sipp(sample_frac=sample_frac)
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income',
                    'job3_income', 'tip_income']
    zero_cols = ['total_income', 'job1_income', 'job2_income',
                 'job3_income', 'tip_income']
    n_features = len(feature_cols)

    # Split data
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

    # Print actual zero rates
    print("\nActual zero rates in training data:")
    for col in zero_cols:
        rate = (train_df[col] == 0).mean()
        print(f"  {col}: {rate:.1%}")

    results = {}

    # ==========================================================================
    # Train and evaluate ZI-QDNN
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ZI-QDNN (Zero-Inflated Quantile DNN)")
    print("=" * 70)

    start_time = time.time()
    qdnn_model, X_mean, X_std = train_qdnn(train_df, feature_cols, epochs=epochs)
    qdnn_train_time = time.time() - start_time
    print(f"Training time: {qdnn_train_time:.2f}s")

    start_time = time.time()
    qdnn_synth = generate_qdnn(qdnn_model, train_df, feature_cols, n_synth,
                               n_periods, X_mean, X_std, seed=42)
    qdnn_gen_time = time.time() - start_time
    print(f"Generation time ({n_synth} trajectories): {qdnn_gen_time:.2f}s")

    qdnn_coverage = compute_coverage(holdout_df, qdnn_synth, train_df,
                                     feature_cols, zero_cols)
    qdnn_zero_accuracy = compute_zero_rate_accuracy(holdout_df, qdnn_synth, zero_cols)

    print("\nCoverage (with zero indicators):")
    print(f"  Median distance: {qdnn_coverage['median']:.3f}")
    print(f"  P90 distance: {qdnn_coverage['p90']:.3f}")
    print(f"  Max distance: {qdnn_coverage['max']:.3f}")

    print("\nZero rate accuracy:")
    for col in zero_cols:
        print(f"  {col}: real={qdnn_zero_accuracy[f'{col}_real']:.1%}, "
              f"synth={qdnn_zero_accuracy[f'{col}_synth']:.1%}, "
              f"error={qdnn_zero_accuracy[f'{col}_abs_error']:.1%}")

    results['qdnn'] = {
        'train_time': qdnn_train_time,
        'gen_time': qdnn_gen_time,
        'coverage': qdnn_coverage,
        'zero_accuracy': qdnn_zero_accuracy,
    }

    # ==========================================================================
    # Train and evaluate GRU
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: GRU-based Transition Model")
    print("=" * 70)

    context_length = 3  # Look at last 3 periods

    start_time = time.time()
    gru_model = GRUTransitionWrapper(n_features, hidden_dim=128, n_layers=2,
                                     context_length=context_length)
    gru_model.fit(train_df, feature_cols, epochs=epochs)
    gru_train_time = time.time() - start_time
    print(f"Training time: {gru_train_time:.2f}s")

    start_time = time.time()
    gru_synth = generate_gru(gru_model, train_df, feature_cols, n_synth,
                             n_periods, seed=42)
    gru_gen_time = time.time() - start_time
    print(f"Generation time ({n_synth} trajectories): {gru_gen_time:.2f}s")

    gru_coverage = compute_coverage(holdout_df, gru_synth, train_df,
                                    feature_cols, zero_cols)
    gru_zero_accuracy = compute_zero_rate_accuracy(holdout_df, gru_synth, zero_cols)

    print("\nCoverage (with zero indicators):")
    print(f"  Median distance: {gru_coverage['median']:.3f}")
    print(f"  P90 distance: {gru_coverage['p90']:.3f}")
    print(f"  Max distance: {gru_coverage['max']:.3f}")

    print("\nZero rate accuracy:")
    for col in zero_cols:
        print(f"  {col}: real={gru_zero_accuracy[f'{col}_real']:.1%}, "
              f"synth={gru_zero_accuracy[f'{col}_synth']:.1%}, "
              f"error={gru_zero_accuracy[f'{col}_abs_error']:.1%}")

    results['gru'] = {
        'train_time': gru_train_time,
        'gen_time': gru_gen_time,
        'coverage': gru_coverage,
        'zero_accuracy': gru_zero_accuracy,
    }

    # ==========================================================================
    # Summary comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'ZI-QDNN':<15} {'GRU':<15} {'Winner':<10}")
    print("-" * 75)

    # Coverage (lower is better)
    qdnn_cov = results['qdnn']['coverage']['median']
    gru_cov = results['gru']['coverage']['median']
    winner = 'QDNN' if qdnn_cov < gru_cov else 'GRU'
    print(f"{'Coverage (median distance)':<35} {qdnn_cov:<15.3f} {gru_cov:<15.3f} {winner:<10}")

    qdnn_cov90 = results['qdnn']['coverage']['p90']
    gru_cov90 = results['gru']['coverage']['p90']
    winner = 'QDNN' if qdnn_cov90 < gru_cov90 else 'GRU'
    print(f"{'Coverage (P90 distance)':<35} {qdnn_cov90:<15.3f} {gru_cov90:<15.3f} {winner:<10}")

    # Zero rate accuracy (lower error is better)
    qdnn_zero = results['qdnn']['zero_accuracy']['mean_abs_error']
    gru_zero = results['gru']['zero_accuracy']['mean_abs_error']
    winner = 'QDNN' if qdnn_zero < gru_zero else 'GRU'
    print(f"{'Zero rate mean abs error':<35} {qdnn_zero:<15.3%} {gru_zero:<15.3%} {winner:<10}")

    # Training time (lower is better)
    qdnn_train = results['qdnn']['train_time']
    gru_train = results['gru']['train_time']
    winner = 'QDNN' if qdnn_train < gru_train else 'GRU'
    print(f"{'Training time (s)':<35} {qdnn_train:<15.2f} {gru_train:<15.2f} {winner:<10}")

    # Generation time (lower is better)
    qdnn_gen = results['qdnn']['gen_time']
    gru_gen = results['gru']['gen_time']
    winner = 'QDNN' if qdnn_gen < gru_gen else 'GRU'
    print(f"{'Generation time (s)':<35} {qdnn_gen:<15.2f} {gru_gen:<15.2f} {winner:<10}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    coverage_better = 'QDNN' if qdnn_cov < gru_cov else 'GRU'
    zero_better = 'QDNN' if qdnn_zero < gru_zero else 'GRU'

    print(f"\n1. Coverage: {coverage_better} achieves better trajectory coverage")
    print(f"   - Improvement: {abs(qdnn_cov - gru_cov) / max(qdnn_cov, gru_cov) * 100:.1f}%")

    print(f"\n2. Zero patterns: {zero_better} better captures zero-inflation")
    print(f"   - Mean error difference: {abs(qdnn_zero - gru_zero):.1%}")

    print("\n3. Computational efficiency:")
    print(f"   - QDNN is {gru_train / qdnn_train:.1f}x faster to train")
    print(f"   - QDNN is {gru_gen / qdnn_gen:.1f}x faster to generate")

    # Save results
    results_df = pd.DataFrame({
        'model': ['ZI-QDNN', 'GRU'],
        'coverage_median': [qdnn_cov, gru_cov],
        'coverage_p90': [qdnn_cov90, gru_cov90],
        'zero_rate_mae': [qdnn_zero, gru_zero],
        'train_time_s': [qdnn_train, gru_train],
        'gen_time_s': [qdnn_gen, gru_gen],
    })

    results_path = Path(__file__).parent / 'qdnn_vs_rnn_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare QDNN vs RNN for panel synthesis")
    parser.add_argument("--sample-frac", type=float, default=0.5,
                        help="Fraction of SIPP data to use")
    parser.add_argument("--n-periods", type=int, default=6,
                        help="Number of periods per trajectory")
    parser.add_argument("--n-synth", type=int, default=2000,
                        help="Number of synthetic trajectories to generate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    args = parser.parse_args()

    main(sample_frac=args.sample_frac, n_periods=args.n_periods,
         n_synth=args.n_synth, epochs=args.epochs)
