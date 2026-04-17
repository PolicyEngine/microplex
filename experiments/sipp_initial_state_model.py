"""Initial state model + cross-section coverage evaluation."""

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
from experiments.sipp_inspect_holdouts import (
    prepare_sipp_panel, CombinedModel, RatioTransitionModel
)


class InitialStateModel(nn.Module):
    """Generate synthetic period-0 states using quantile regression on noise."""

    def __init__(self, n_features: int, hidden_dim: int = 128, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.n_quantiles = n_quantiles
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)

        # Input is noise, output is quantiles for each feature
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Separate heads for P(zero) and value quantiles (log-scale)
        self.zero_head = nn.Linear(hidden_dim, n_features)
        self.value_head = nn.Linear(hidden_dim, n_features * n_quantiles)

    def forward(self, noise):
        h = self.network(noise)
        zero_logits = self.zero_head(h)
        value_quantiles = self.value_head(h).view(-1, self.n_features, self.n_quantiles)
        return zero_logits, value_quantiles

    def loss(self, noise, targets):
        zero_logits, value_q = self.forward(noise)

        # BCE for zero/nonzero
        is_zero = (targets == 0).float()
        bce = nn.functional.binary_cross_entropy_with_logits(zero_logits, is_zero)

        # Quantile loss for nonzero values (log scale)
        nonzero_mask = targets > 0
        if nonzero_mask.any():
            log_targets = torch.log1p(targets)
            errors = log_targets.unsqueeze(-1) - value_q
            mask = nonzero_mask.unsqueeze(-1).float()
            ql = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql = (ql * mask).sum() / (mask.sum() + 1e-8) / self.n_quantiles
        else:
            ql = torch.tensor(0.0)

        return bce + ql


class FullSynthesizer:
    """Combines initial state model + transition model."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.initial_model = InitialStateModel(n_features)
        self.transition_model = CombinedModel(n_features)

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'n_features': self.n_features,
            'initial_model': self.initial_model.state_dict(),
            'transition_model': self.transition_model.feature_model.state_dict(),
            'X_mean': self.transition_model.X_mean,
            'X_std': self.transition_model.X_std,
        }, path)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> 'FullSynthesizer':
        """Load model from disk."""
        checkpoint = torch.load(path, weights_only=False)
        synth = cls(checkpoint['n_features'])
        synth.initial_model.load_state_dict(checkpoint['initial_model'])
        synth.transition_model.feature_model.load_state_dict(checkpoint['transition_model'])
        synth.transition_model.X_mean = checkpoint['X_mean']
        synth.transition_model.X_std = checkpoint['X_std']
        print(f"Loaded model from {path}")
        return synth

    def fit(self, train_df: pd.DataFrame, feature_cols: list, epochs: int = 100):
        """Train both models on training data."""
        # Extract initial states (period 0 for each person)
        initial_states = []
        for pid in train_df['person_id'].unique():
            person = train_df[train_df['person_id'] == pid].sort_values('period')
            initial_states.append(person[feature_cols].iloc[0].values)
        X0 = np.array(initial_states)

        print(f"  Training initial state model on {len(X0)} samples...")

        # Train initial state model
        X0_t = torch.tensor(X0, dtype=torch.float32)
        noise = torch.randn_like(X0_t)

        opt = torch.optim.Adam(self.initial_model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            opt.zero_grad()
            # Resample noise each epoch for variety
            noise = torch.randn_like(X0_t)
            loss = self.initial_model.loss(noise, X0_t)
            loss.backward()
            opt.step()

        print(f"  Training transition model...")
        self.transition_model.fit(train_df, feature_cols, epochs=epochs)

    def sample_initial(self, n: int) -> np.ndarray:
        """Generate n synthetic initial states."""
        noise = torch.randn(n, self.n_features)
        with torch.no_grad():
            zero_logits, value_q = self.initial_model.forward(noise)
            p_zero = torch.sigmoid(zero_logits).numpy()

            # Sample quantile index for each feature
            n_quantiles = value_q.shape[-1]
            idx = (torch.rand(n, self.n_features, 1) * (n_quantiles - 1)).long()
            sampled_log = value_q.gather(-1, idx).squeeze(-1)
            sampled_values = torch.expm1(torch.clamp(sampled_log, max=20)).numpy()

        # Apply zero mask
        states = np.zeros((n, self.n_features))
        for i in range(n):
            for j in range(self.n_features):
                if np.random.random() < p_zero[i, j]:
                    states[i, j] = 0
                else:
                    states[i, j] = max(0, sampled_values[i, j])

        return states

    def generate(self, n_synth: int, n_periods: int, seed: int = 42) -> pd.DataFrame:
        """Generate full synthetic panel."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate synthetic initial states
        initial_states = self.sample_initial(n_synth)

        feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
        records = []

        for pid in range(n_synth):
            state = initial_states[pid].copy()
            for t in range(n_periods):
                state = np.clip(np.nan_to_num(state, 0), 0, 1e10)
                records.append({
                    'person_id': pid,
                    'period': t,
                    **{col: float(state[i]) for i, col in enumerate(feature_cols)}
                })
                if t < n_periods - 1:
                    state = self.transition_model.sample(state)

        return pd.DataFrame(records)


def compute_coverage(real_mat: np.ndarray, synth_mat: np.ndarray, scaler=None) -> float:
    """Compute mean NN distance from real to synthetic."""
    if scaler is None:
        scaler = StandardScaler().fit(real_mat)
    real_scaled = scaler.transform(real_mat)
    synth_scaled = scaler.transform(synth_mat)

    nn_model = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    distances, _ = nn_model.kneighbors(real_scaled)
    return float(np.mean(distances))


def main(sample_frac: float = 0.5, n_periods: int = 6, n_synth: int = 2000,
         model_path: str = None, save_model: bool = True):
    print("=" * 70)
    print("INITIAL STATE MODEL + CROSS-SECTION COVERAGE")
    print("=" * 70)

    model_path = model_path or Path(__file__).parent / "sipp_synthesizer.pt"

    print("\nLoading SIPP...")
    sipp_raw = load_sipp(sample_frac=sample_frac)
    sipp = prepare_sipp_panel(sipp_raw)

    feature_cols = ['age', 'total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    zero_cols = ['total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']

    # Split persons
    persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(persons)
    n_train = int(len(persons) * 0.8)
    train_persons, holdout_persons = persons[:n_train], persons[n_train:]

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

    # Extract initial states for coverage eval
    def get_initial_states(df):
        states = []
        for pid in df['person_id'].unique():
            person = df[df['person_id'] == pid].sort_values('period')
            states.append(person[feature_cols].iloc[0].values)
        return np.array(states)

    train_initial = get_initial_states(train_df)
    holdout_initial = get_initial_states(holdout_df)

    # Augment with zero indicators for coverage
    def augment(mat, zero_cols_idx):
        indicators = (mat[:, zero_cols_idx] > 0).astype(float)
        return np.hstack([mat, indicators])

    zero_idx = [feature_cols.index(c) for c in zero_cols]

    print("\n" + "=" * 70)
    print("BASELINE: Bootstrap Initial States (current approach)")
    print("=" * 70)

    # Current approach: bootstrap from training
    bootstrap_initial = train_initial[np.random.choice(len(train_initial), n_synth, replace=True)]

    train_aug = augment(train_initial, zero_idx)
    holdout_aug = augment(holdout_initial, zero_idx)
    bootstrap_aug = augment(bootstrap_initial, zero_idx)

    scaler = StandardScaler().fit(train_aug)
    bootstrap_coverage = compute_coverage(holdout_aug, bootstrap_aug, scaler)
    print(f"Bootstrap cross-section coverage: {bootstrap_coverage:.3f}")

    print("\n" + "=" * 70)
    print("NEW: Learned Initial State Model")
    print("=" * 70)

    # Load or train synthesizer
    if Path(model_path).exists():
        synthesizer = FullSynthesizer.load(model_path)
    else:
        print("\nTraining full synthesizer...")
        synthesizer = FullSynthesizer(len(feature_cols))
        synthesizer.fit(train_df, feature_cols, epochs=100)
        if save_model:
            synthesizer.save(model_path)

    # Generate synthetic initial states
    print(f"\nGenerating {n_synth} synthetic initial states...")
    synth_initial = synthesizer.sample_initial(n_synth)
    synth_initial_aug = augment(synth_initial, zero_idx)

    learned_coverage = compute_coverage(holdout_aug, synth_initial_aug, scaler)
    print(f"Learned model cross-section coverage: {learned_coverage:.3f}")

    print("\n" + "=" * 70)
    print("FULL TRAJECTORY COVERAGE")
    print("=" * 70)

    # Generate full trajectories with learned initial states
    print(f"\nGenerating {n_synth} full trajectories...")
    synth_df = synthesizer.generate(n_synth, n_periods, seed=42)

    # Also generate with bootstrap for comparison
    from experiments.sipp_inspect_holdouts import generate_synth
    bootstrap_df = generate_synth(synthesizer.transition_model, train_df, feature_cols, n_synth, n_periods, seed=42)

    # Full trajectory coverage
    def to_trajectory_matrix(df):
        eval_cols = feature_cols + [f'{c}_nz' for c in zero_cols]
        df = df.copy()
        for col in zero_cols:
            df[f'{col}_nz'] = (df[col] > 0).astype(float)
        return np.array([
            df[df['person_id'] == pid].sort_values('period')[eval_cols].values.flatten()
            for pid in sorted(df['person_id'].unique())
        ])

    holdout_traj = to_trajectory_matrix(holdout_df)
    synth_traj = to_trajectory_matrix(synth_df)
    bootstrap_traj = to_trajectory_matrix(bootstrap_df)

    traj_scaler = StandardScaler().fit(to_trajectory_matrix(train_df))

    learned_traj_coverage = compute_coverage(holdout_traj, synth_traj, traj_scaler)
    bootstrap_traj_coverage = compute_coverage(holdout_traj, bootstrap_traj, traj_scaler)

    print(f"Bootstrap trajectory coverage: {bootstrap_traj_coverage:.3f}")
    print(f"Learned trajectory coverage: {learned_traj_coverage:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Bootstrap':<12} {'Learned':<12}")
    print("-" * 60)
    print(f"{'Cross-section coverage':<35} {bootstrap_coverage:<12.3f} {learned_coverage:<12.3f}")
    print(f"{'Full trajectory coverage':<35} {bootstrap_traj_coverage:<12.3f} {learned_traj_coverage:<12.3f}")

    improvement_cs = (bootstrap_coverage - learned_coverage) / bootstrap_coverage * 100
    improvement_traj = (bootstrap_traj_coverage - learned_traj_coverage) / bootstrap_traj_coverage * 100
    print(f"\nCross-section improvement: {improvement_cs:+.1f}%")
    print(f"Trajectory improvement: {improvement_traj:+.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Initial state model evaluation")
    parser.add_argument("--sample-frac", type=float, default=0.5)
    parser.add_argument("--n-periods", type=int, default=6)
    parser.add_argument("--n-synth", type=int, default=2000)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--no-save", action="store_true", help="Don't save trained model")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if model exists")
    args = parser.parse_args()

    model_path = args.model_path
    if args.retrain and model_path and Path(model_path).exists():
        Path(model_path).unlink()

    main(sample_frac=args.sample_frac, n_periods=args.n_periods, n_synth=args.n_synth,
         model_path=model_path, save_model=not args.no_save)
