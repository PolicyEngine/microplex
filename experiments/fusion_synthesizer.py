"""Multi-source fusion synthesizer prototype.

Trains on multiple surveys (SIPP, CPS, SCF) with different variables.
Uses masking to handle missing variables: predict observed vars,
condition on is_observed indicators.

Key idea: Each survey contributes different variables. The fusion model
learns correlations across all variables from whichever survey provides them.

Architecture:
1. FusedSynthesizer class:
   - Takes a list of all possible variables across surveys
   - fit() accepts dict of dataframes: {'sipp': df, 'cps': df, ...}
   - Each df has its own columns; missing vars treated as NaN
   - Uses masked loss to train only on observed variables
   - generate() produces records with all variables

2. MaskedInitialStateModel: Generates initial states with quantile regression
   - Handles sparse observations via observation mask
   - Predicts P(zero) + value quantiles for each variable

3. MaskedTransitionModel: Models state transitions
   - Predicts zero transitions + value ratios/initializations
   - Masked loss for multi-source training

4. evaluate_coverage(): Measures coverage separately for each survey's
   holdout using only that survey's variables

Usage:
    synthesizer = FusedSynthesizer(['age', 'income', 'job1', 'job2'])
    synthesizer.fit({
        'cps': df_cps[['person_id', 'period', 'age', 'income']],
        'sipp': df_sipp[['person_id', 'period', 'age', 'job1', 'job2']],
    })
    synth_df = synthesizer.generate(n_synth=1000, n_periods=6)
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipelines.data_loaders import load_sipp
from experiments.sipp_inspect_holdouts import prepare_sipp_panel


class MaskedInitialStateModel(nn.Module):
    """Generate synthetic initial states with masked training for sparse observations.

    Handles variables that are only observed in some surveys by using a mask
    to compute loss only on observed values.

    Uses direct value prediction (not log-scale) to avoid numerical issues.
    """

    def __init__(self, n_features: int, hidden_dim: int = 256, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.n_quantiles = n_quantiles
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)

        # Input: noise + observation mask (which variables are observed)
        # This conditions generation on what we know
        self.input_dim = n_features + n_features  # noise + is_observed mask

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Separate heads for P(zero) and value quantiles
        self.zero_head = nn.Linear(hidden_dim, n_features)
        self.value_head = nn.Linear(hidden_dim, n_features * n_quantiles)

        # Store normalization stats for value prediction
        self.register_buffer('value_mean', torch.zeros(n_features))
        self.register_buffer('value_std', torch.ones(n_features))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization stats for value prediction."""
        self.value_mean = torch.tensor(mean, dtype=torch.float32)
        self.value_std = torch.tensor(std, dtype=torch.float32)

    def forward(self, noise, obs_mask):
        """
        Args:
            noise: (batch, n_features) random noise
            obs_mask: (batch, n_features) binary mask of observed variables
        """
        x = torch.cat([noise, obs_mask], dim=-1)
        h = self.network(x)
        zero_logits = self.zero_head(h)
        # Output quantiles in normalized space
        value_quantiles = self.value_head(h).view(-1, self.n_features, self.n_quantiles)
        return zero_logits, value_quantiles

    def loss(self, noise, obs_mask, targets, targets_normalized):
        """Masked loss: only compute on observed variables.

        Args:
            noise: (batch, n_features) random noise
            obs_mask: (batch, n_features) binary mask, 1 where variable is observed
            targets: (batch, n_features) target values (raw), may contain NaN
            targets_normalized: (batch, n_features) normalized target values
        """
        zero_logits, value_q = self.forward(noise, obs_mask)

        # Replace NaN with 0 for computation (masked out anyway)
        targets_clean = torch.nan_to_num(targets, nan=0.0)
        targets_norm_clean = torch.nan_to_num(targets_normalized, nan=0.0)

        # BCE for zero/nonzero, masked
        is_zero = (targets_clean == 0).float()
        bce_raw = nn.functional.binary_cross_entropy_with_logits(
            zero_logits, is_zero, reduction='none'
        )
        # Only count loss where observed
        bce = (bce_raw * obs_mask).sum() / (obs_mask.sum() + 1e-8)

        # Quantile loss for nonzero values (normalized space), masked
        nonzero_mask = (targets_clean > 0) & (obs_mask > 0)
        if nonzero_mask.any():
            errors = targets_norm_clean.unsqueeze(-1) - value_q
            mask = nonzero_mask.unsqueeze(-1).float()
            ql = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql = (ql * mask).sum() / (mask.sum() + 1e-8) / self.n_quantiles
        else:
            ql = torch.tensor(0.0)

        return bce + ql


class MaskedTransitionModel(nn.Module):
    """Transition model with masking for sparse observations."""

    def __init__(self, n_features: int, hidden_dim: int = 256, n_quantiles: int = 19):
        super().__init__()
        self.n_features = n_features
        self.quantiles = torch.linspace(0.05, 0.95, n_quantiles)
        self.n_quantiles = n_quantiles

        # Input: current values (normalized) + is_nonzero indicators + is_observed mask
        self.input_dim = n_features + n_features + n_features

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        # Zero/nonzero transition heads
        self.become_zero_head = nn.Linear(hidden_dim, n_features)
        self.stay_zero_head = nn.Linear(hidden_dim, n_features)

        # Value heads
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
        return (
            h,
            self.become_zero_head(h),
            self.stay_zero_head(h),
            self.ratio_head(h).view(-1, self.n_features, self.n_quantiles),
            self.init_head(h).view(-1, self.n_features, self.n_quantiles),
        )

    def loss(self, x_raw, x_aug, target, obs_mask):
        """Masked transition loss."""
        h, become_zero_logits, stay_zero_logits, ratio_q, init_q = self.forward(x_aug)

        # Clean targets
        target_clean = torch.nan_to_num(target, nan=0.0)
        x_raw_clean = torch.nan_to_num(x_raw, nan=0.0)

        is_zero_output = (target_clean == 0).float()
        is_nonzero_input = (x_raw_clean > 0).float()

        # BCE loss for zero transitions, masked
        become_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            become_zero_logits, is_zero_output, reduction='none'
        )
        stay_zero_loss = nn.functional.binary_cross_entropy_with_logits(
            stay_zero_logits, is_zero_output, reduction='none'
        )

        bce_raw = is_nonzero_input * become_zero_loss + (1 - is_nonzero_input) * stay_zero_loss
        bce = (bce_raw * obs_mask).sum() / (obs_mask.sum() + 1e-8)

        # Ratio quantile loss for nonzero -> nonzero, masked
        x_nonzero = (x_raw_clean > 0)
        y_nonzero = (target_clean > 0)
        ratio_mask = x_nonzero & y_nonzero & (obs_mask > 0)

        if ratio_mask.any():
            log_ratio = torch.log(target_clean + 1e-8) - torch.log(x_raw_clean + 1e-8)
            errors = log_ratio.unsqueeze(-1) - ratio_q
            mask = ratio_mask.unsqueeze(-1).float()
            ql_ratio = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql_ratio = (ql_ratio * mask).sum() / (mask.sum() + 1e-8) / self.n_quantiles
        else:
            ql_ratio = torch.tensor(0.0)

        # Init quantile loss for zero -> nonzero, masked
        init_mask = (~x_nonzero) & y_nonzero & (obs_mask > 0)
        if init_mask.any():
            log_y = torch.log1p(target_clean)
            errors = log_y.unsqueeze(-1) - init_q
            mask = init_mask.unsqueeze(-1).float()
            ql_init = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
            ql_init = (ql_init * mask).sum() / (mask.sum() + 1e-8) / self.n_quantiles
        else:
            ql_init = torch.tensor(0.0)

        return bce + ql_ratio + ql_init


class FusedSynthesizer:
    """Multi-source fusion synthesizer.

    Trains on multiple surveys with different variables using masked loss.
    Each survey contributes its observed variables; missing vars are masked.
    """

    def __init__(self, all_variables: List[str], hidden_dim: int = 256):
        """
        Args:
            all_variables: List of all possible variable names across all surveys.
        """
        self.all_variables = all_variables
        self.n_features = len(all_variables)
        self.hidden_dim = hidden_dim

        self.initial_model = MaskedInitialStateModel(self.n_features, hidden_dim)
        self.transition_model = MaskedTransitionModel(self.n_features, hidden_dim)

        # Normalization stats (computed during fit)
        self.X_mean = None
        self.X_std = None

    def fit(
        self,
        surveys: Dict[str, pd.DataFrame],
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        """Train on multiple surveys with different variables.

        Args:
            surveys: Dict mapping survey name to DataFrame.
                     Each DataFrame has columns for its observed variables.
                     Missing variables in a survey are treated as unobserved.
        """
        if verbose:
            print("=" * 70)
            print("FUSION TRAINING")
            print("=" * 70)
            print(f"All variables ({self.n_features}): {self.all_variables}")

        # Extract initial states and transitions from all surveys
        all_initial_states = []
        all_initial_masks = []
        all_X = []
        all_Y = []
        all_masks = []

        for survey_name, df in surveys.items():
            survey_vars = [v for v in self.all_variables if v in df.columns]
            if verbose:
                print(f"\n{survey_name}: {len(df)} records, vars: {survey_vars}")

            # Create observation mask for this survey
            mask = np.array([1.0 if v in df.columns else 0.0 for v in self.all_variables])

            # Extract values for all variables (NaN for missing)
            values = np.zeros((len(df), self.n_features))
            for i, var in enumerate(self.all_variables):
                if var in df.columns:
                    values[:, i] = df[var].fillna(0).values
                else:
                    values[:, i] = np.nan

            # Check if this is panel data
            if 'person_id' in df.columns and 'period' in df.columns:
                # Panel data: extract initial states and transitions
                for pid in df['person_id'].unique():
                    person = df[df['person_id'] == pid].sort_values('period')
                    if len(person) < 1:
                        continue

                    # Initial state
                    idx = person.index[0]
                    init_vals = np.zeros(self.n_features)
                    for i, var in enumerate(self.all_variables):
                        if var in df.columns:
                            init_vals[i] = person[var].iloc[0] if not pd.isna(person[var].iloc[0]) else 0
                        else:
                            init_vals[i] = np.nan
                    all_initial_states.append(init_vals)
                    all_initial_masks.append(mask.copy())

                    # Transitions
                    for t in range(len(person) - 1):
                        x_vals = np.zeros(self.n_features)
                        y_vals = np.zeros(self.n_features)
                        for i, var in enumerate(self.all_variables):
                            if var in df.columns:
                                x_vals[i] = person[var].iloc[t] if not pd.isna(person[var].iloc[t]) else 0
                                y_vals[i] = person[var].iloc[t + 1] if not pd.isna(person[var].iloc[t + 1]) else 0
                            else:
                                x_vals[i] = np.nan
                                y_vals[i] = np.nan
                        all_X.append(x_vals)
                        all_Y.append(y_vals)
                        all_masks.append(mask.copy())
            else:
                # Cross-sectional data: treat each record as initial state only
                for idx in df.index:
                    init_vals = values[df.index.get_loc(idx)]
                    all_initial_states.append(init_vals)
                    all_initial_masks.append(mask.copy())

        # Convert to arrays
        X0 = np.array(all_initial_states) if all_initial_states else np.zeros((0, self.n_features))
        X0_mask = np.array(all_initial_masks) if all_initial_masks else np.zeros((0, self.n_features))

        if verbose:
            print(f"\nTotal initial states: {len(X0)}")

        # Compute normalization stats from observed values
        X0_clean = np.nan_to_num(X0, nan=0.0)
        self.X_mean = np.nanmean(X0, axis=0)
        self.X_std = np.nanstd(X0, axis=0) + 1e-6
        self.X_mean = np.nan_to_num(self.X_mean, nan=0.0)
        self.X_std = np.nan_to_num(self.X_std, nan=1.0)

        # Train initial state model
        if verbose:
            print("\nTraining initial state model...")
            # Debug: show what we're training on
            for i, var in enumerate(self.all_variables):
                obs_count = (X0_mask[:, i] > 0).sum()
                if obs_count > 0:
                    vals = X0[X0_mask[:, i] > 0, i]
                    vals_clean = vals[~np.isnan(vals)]
                    if len(vals_clean) > 0:
                        print(f"    {var}: {obs_count} obs, mean={np.mean(vals_clean):.1f}, median={np.median(vals_clean):.1f}, nonzero={np.mean(vals_clean > 0)*100:.1f}%")

        # Compute per-variable normalization stats (for nonzero values only)
        var_mean = np.zeros(self.n_features)
        var_std = np.ones(self.n_features)
        for i in range(self.n_features):
            vals = X0[X0_mask[:, i] > 0, i]
            vals = vals[~np.isnan(vals)]
            vals_nz = vals[vals > 0]
            if len(vals_nz) > 0:
                var_mean[i] = np.mean(vals_nz)
                var_std[i] = np.std(vals_nz) + 1e-6

        self.initial_model.set_normalization(var_mean, var_std)

        # Normalize X0 for training
        X0_clean = np.nan_to_num(X0, nan=0.0)
        X0_normalized = np.zeros_like(X0_clean)
        for i in range(self.n_features):
            X0_normalized[:, i] = (X0_clean[:, i] - var_mean[i]) / var_std[i]

        X0_t = torch.tensor(X0, dtype=torch.float32)
        X0_norm_t = torch.tensor(X0_normalized, dtype=torch.float32)
        X0_mask_t = torch.tensor(X0_mask, dtype=torch.float32)

        opt = torch.optim.Adam(self.initial_model.parameters(), lr=lr)
        for epoch in range(epochs):
            opt.zero_grad()
            noise = torch.randn_like(X0_t)
            loss = self.initial_model.loss(noise, X0_mask_t, X0_t, X0_norm_t)
            loss.backward()
            opt.step()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

        # Train transition model if we have transitions
        if len(all_X) > 0:
            if verbose:
                print(f"\nTraining transition model on {len(all_X)} transitions...")

            X = np.array(all_X)
            Y = np.array(all_Y)
            masks = np.array(all_masks)

            # Normalize X
            X_clean = np.nan_to_num(X, nan=0.0)
            X_norm = (X_clean - self.X_mean) / self.X_std
            X_indicators = (X_clean > 0).astype(float)
            X_aug = np.hstack([X_norm, X_indicators, masks])

            X_raw_t = torch.tensor(X, dtype=torch.float32)
            X_aug_t = torch.tensor(X_aug, dtype=torch.float32)
            Y_t = torch.tensor(Y, dtype=torch.float32)
            masks_t = torch.tensor(masks, dtype=torch.float32)

            opt = torch.optim.Adam(self.transition_model.parameters(), lr=lr)
            for epoch in range(epochs):
                opt.zero_grad()
                loss = self.transition_model.loss(X_raw_t, X_aug_t, Y_t, masks_t)
                loss.backward()
                opt.step()

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

        if verbose:
            print("\nTraining complete!")

    def sample_initial(self, n: int, observed_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate n synthetic initial states.

        Args:
            n: Number of samples to generate
            observed_mask: If provided, (n_features,) mask indicating which vars to condition on.
                          If None, assume all variables should be generated.
        """
        if observed_mask is None:
            observed_mask = np.ones(self.n_features)

        obs_mask_t = torch.tensor(observed_mask, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        noise = torch.randn(n, self.n_features)

        with torch.no_grad():
            zero_logits, value_q = self.initial_model.forward(noise, obs_mask_t)
            p_zero = torch.sigmoid(zero_logits).numpy()

            # Sample quantile index for each feature - sample across full range
            n_quantiles = value_q.shape[-1]
            idx = (torch.rand(n, self.n_features, 1) * n_quantiles).long().clamp(0, n_quantiles - 1)
            sampled_normalized = value_q.gather(-1, idx).squeeze(-1)

            # Denormalize: convert from normalized space back to original scale
            value_mean = self.initial_model.value_mean.numpy()
            value_std = self.initial_model.value_std.numpy()
            sampled_values = sampled_normalized.numpy() * value_std + value_mean

        # Apply zero mask
        states = np.zeros((n, self.n_features))
        for i in range(n):
            for j in range(self.n_features):
                if np.random.random() < p_zero[i, j]:
                    states[i, j] = 0
                else:
                    states[i, j] = max(0, sampled_values[i, j])

        return states

    def sample_transition(self, x_raw: np.ndarray) -> np.ndarray:
        """Sample next state given current state."""
        x_clean = np.nan_to_num(x_raw, nan=0.0)
        x_norm = (x_clean - self.X_mean) / self.X_std
        indicators = (x_clean > 0).astype(float)
        obs_mask = np.ones(self.n_features)  # During generation, assume all observed
        x_aug = np.concatenate([x_norm, indicators, obs_mask])

        x_aug_t = torch.tensor(x_aug, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, become_zero_logits, stay_zero_logits, ratio_q, init_q = self.transition_model.forward(x_aug_t)
            p_become_zero = torch.sigmoid(become_zero_logits).numpy()[0]
            p_stay_zero = torch.sigmoid(stay_zero_logits).numpy()[0]

            n_quantiles = ratio_q.shape[-1]
            # Sample quantile index
            idx = (torch.rand(1, self.n_features, 1) * n_quantiles).long().clamp(0, n_quantiles - 1)
            ratio_samples = ratio_q.gather(-1, idx).squeeze(-1)
            # Ratios are in log space, clamp to reasonable range
            ratios = torch.exp(torch.clamp(ratio_samples, min=-1.0, max=1.0)).numpy()[0]

            # Init values use same normalization as initial model
            init_samples = init_q.gather(-1, idx).squeeze(-1).squeeze(0)  # Shape: (n_features,)
            value_mean = self.initial_model.value_mean.numpy()
            value_std = self.initial_model.value_std.numpy()
            init_vals = init_samples.numpy() * value_std + value_mean

        values = np.zeros(self.n_features)
        for j in range(self.n_features):
            if x_clean[j] > 0:
                if np.random.random() < p_become_zero[j]:
                    values[j] = 0
                else:
                    values[j] = x_clean[j] * ratios[j]
            else:
                if np.random.random() < p_stay_zero[j]:
                    values[j] = 0
                else:
                    values[j] = max(0, init_vals[j])

        return np.clip(values, 0, 1e8)

    def generate(self, n_synth: int, n_periods: int, seed: int = 42) -> pd.DataFrame:
        """Generate full synthetic panel with all variables."""
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate synthetic initial states
        initial_states = self.sample_initial(n_synth)

        records = []
        for pid in range(n_synth):
            state = initial_states[pid].copy()
            for t in range(n_periods):
                state = np.clip(np.nan_to_num(state, 0), 0, 1e10)
                record = {
                    'person_id': pid,
                    'period': t,
                }
                for i, var in enumerate(self.all_variables):
                    record[var] = float(state[i])
                records.append(record)

                if t < n_periods - 1:
                    state = self.sample_transition(state)

        return pd.DataFrame(records)

    def save(self, path: str):
        """Save model to disk."""
        torch.save({
            'all_variables': self.all_variables,
            'n_features': self.n_features,
            'hidden_dim': self.hidden_dim,
            'initial_model': self.initial_model.state_dict(),
            'transition_model': self.transition_model.state_dict(),
            'X_mean': self.X_mean,
            'X_std': self.X_std,
        }, path)
        print(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> 'FusedSynthesizer':
        """Load model from disk."""
        checkpoint = torch.load(path, weights_only=False)
        synth = cls(checkpoint['all_variables'], checkpoint['hidden_dim'])
        synth.initial_model.load_state_dict(checkpoint['initial_model'])
        synth.transition_model.load_state_dict(checkpoint['transition_model'])
        synth.X_mean = checkpoint['X_mean']
        synth.X_std = checkpoint['X_std']
        print(f"Loaded model from {path}")
        return synth


def evaluate_coverage(
    synth_df: pd.DataFrame,
    holdout_dfs: Dict[str, pd.DataFrame],
    all_variables: List[str],
    n_periods: int = 6,
    include_zero_indicators: bool = True,
) -> Dict[str, float]:
    """Evaluate coverage separately for each survey's holdout.

    Uses only the variables that each survey observes.
    Optionally includes zero/nonzero indicators for sparse variables.
    """
    results = {}

    # Variables that should have zero indicators (income-like variables)
    zero_indicator_vars = ['total_income', 'job1_income', 'job2_income', 'job3_income', 'tip_income']

    for survey_name, holdout_df in holdout_dfs.items():
        # Get variables observed in this survey
        survey_vars = [v for v in all_variables if v in holdout_df.columns]
        if not survey_vars:
            print(f"  {survey_name}: No overlapping variables, skipping")
            continue

        # Check if panel or cross-sectional
        is_panel = 'person_id' in holdout_df.columns and 'period' in holdout_df.columns

        # Add zero indicators
        def augment_with_indicators(df, vars):
            df = df.copy()
            for var in vars:
                if var in zero_indicator_vars and var in df.columns:
                    df[f'{var}_nz'] = (df[var] > 0).astype(float)
            return df

        if include_zero_indicators:
            holdout_df = augment_with_indicators(holdout_df, survey_vars)
            synth_df_aug = augment_with_indicators(synth_df, survey_vars)
            eval_vars = survey_vars + [f'{v}_nz' for v in survey_vars if v in zero_indicator_vars]
        else:
            synth_df_aug = synth_df
            eval_vars = survey_vars

        if is_panel:
            # Full trajectory coverage
            def to_trajectory_matrix(df, vars):
                persons = sorted(df['person_id'].unique())
                # Pre-compute existing vars based on df columns
                existing_vars = [v for v in vars if v in df.columns]
                if not existing_vars:
                    return np.zeros((0, 0))
                result = []
                for pid in persons:
                    person = df[df['person_id'] == pid].sort_values('period')
                    if len(person) < n_periods:
                        continue
                    person = person.head(n_periods)
                    traj = person[existing_vars].values.flatten()
                    result.append(traj)
                return np.array(result) if result else np.zeros((0, len(existing_vars) * n_periods))

            holdout_mat = to_trajectory_matrix(holdout_df, eval_vars)
            synth_mat = to_trajectory_matrix(synth_df_aug, eval_vars)
        else:
            # Cross-sectional coverage
            existing_vars = [v for v in eval_vars if v in holdout_df.columns]
            holdout_mat = holdout_df[existing_vars].values
            synth_mat = synth_df_aug[synth_df_aug['period'] == 0][existing_vars].values

        if len(holdout_mat) == 0 or len(synth_mat) == 0:
            print(f"  {survey_name}: Empty matrices, skipping")
            continue

        # Compute coverage
        scaler = StandardScaler().fit(holdout_mat)
        holdout_scaled = scaler.transform(holdout_mat)
        synth_scaled = scaler.transform(synth_mat)

        nn_model = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
        distances, _ = nn_model.kneighbors(holdout_scaled)
        coverage = float(np.mean(distances))

        results[survey_name] = coverage
        n_eval_vars = len([v for v in eval_vars if v in holdout_df.columns or v.replace('_nz', '') in holdout_df.columns])
        print(f"  {survey_name}: coverage = {coverage:.4f} ({n_eval_vars} vars+indicators, {len(holdout_mat)} holdout records)")

    return results


def simulate_multi_source():
    """Simulate multi-source fusion using SIPP with different column subsets.

    This demonstrates the fusion concept by pretending we have two surveys:
    - 'survey_income': Has income variables but not job detail
    - 'survey_jobs': Has job detail but not total income
    """
    print("=" * 70)
    print("MULTI-SOURCE FUSION SYNTHESIZER PROTOTYPE")
    print("=" * 70)
    print("\nSimulating multi-source with SIPP column subsets...")

    # Load SIPP
    print("\nLoading SIPP data...")
    sipp_raw = load_sipp(sample_frac=0.3)
    sipp = prepare_sipp_panel(sipp_raw)

    # Define all variables we want to model
    all_variables = [
        'age',
        'total_income',
        'job1_income',
        'job2_income',
        'job3_income',
        'tip_income',
    ]

    # Split into train/holdout by person
    persons = sipp['person_id'].unique()
    np.random.seed(42)
    np.random.shuffle(persons)
    n_train = int(len(persons) * 0.8)
    train_persons = persons[:n_train]
    holdout_persons = persons[n_train:]

    # Filter to complete panels
    def filter_complete(df, n_periods):
        periods_per_person = df.groupby('person_id')['period'].nunique()
        complete = periods_per_person[periods_per_person >= n_periods].index
        df = df[df['person_id'].isin(complete)]
        df = df.sort_values(['person_id', 'period']).groupby('person_id').head(n_periods)
        return df

    n_periods = 6
    train_full = filter_complete(sipp[sipp['person_id'].isin(train_persons)], n_periods)
    holdout_full = filter_complete(sipp[sipp['person_id'].isin(holdout_persons)], n_periods)

    print(f"\nFull train set: {train_full['person_id'].nunique()} persons")
    print(f"Full holdout set: {holdout_full['person_id'].nunique()} persons")

    # Simulate two surveys with different variables
    # Survey A: demographics + total income (like CPS)
    survey_a_vars = ['age', 'total_income']
    train_a = train_full[['person_id', 'period'] + survey_a_vars].copy()

    # Survey B: job-level detail (like SIPP job history)
    survey_b_vars = ['age', 'job1_income', 'job2_income', 'job3_income', 'tip_income']
    train_b = train_full[['person_id', 'period'] + survey_b_vars].copy()

    print(f"\nSimulated surveys:")
    print(f"  Survey A (income-focused): {survey_a_vars}")
    print(f"  Survey B (job-focused): {survey_b_vars}")

    # Create fusion synthesizer
    print("\n" + "=" * 70)
    print("TRAINING FUSED SYNTHESIZER")
    print("=" * 70)

    synthesizer = FusedSynthesizer(all_variables, hidden_dim=256)
    synthesizer.fit(
        surveys={
            'survey_a': train_a,
            'survey_b': train_b,
        },
        epochs=100,
        verbose=True,
    )

    # Generate synthetic data
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 70)

    n_synth = 1000
    synth_df = synthesizer.generate(n_synth, n_periods, seed=42)
    print(f"\nGenerated {n_synth} synthetic trajectories")
    print(f"Variables: {list(synth_df.columns)}")

    # Show sample
    print("\nSample synthetic record (period 0):")
    sample = synth_df[synth_df['person_id'] == 0].head(1)[all_variables]
    print(sample.to_string())

    # Evaluate coverage separately for each survey's variables
    print("\n" + "=" * 70)
    print("EVALUATING COVERAGE PER SURVEY")
    print("=" * 70)

    # Holdouts: use full holdout but evaluate only on each survey's vars
    holdout_a = holdout_full[['person_id', 'period'] + survey_a_vars].copy()
    holdout_b = holdout_full[['person_id', 'period'] + survey_b_vars].copy()

    coverage = evaluate_coverage(
        synth_df,
        {
            'survey_a': holdout_a,
            'survey_b': holdout_b,
            'full': holdout_full[['person_id', 'period'] + all_variables],
        },
        all_variables,
        n_periods=n_periods,
    )

    # Compare to single-source baseline
    print("\n" + "=" * 70)
    print("BASELINE: SINGLE-SOURCE TRAINING")
    print("=" * 70)

    print("\nTraining on Survey A only (income-focused)...")
    synth_a_only = FusedSynthesizer(survey_a_vars, hidden_dim=256)
    synth_a_only.fit({'survey_a': train_a}, epochs=100, verbose=False)
    synth_df_a = synth_a_only.generate(n_synth, n_periods, seed=42)

    print("Training on Survey B only (job-focused)...")
    synth_b_only = FusedSynthesizer(survey_b_vars, hidden_dim=256)
    synth_b_only.fit({'survey_b': train_b}, epochs=100, verbose=False)
    synth_df_b = synth_b_only.generate(n_synth, n_periods, seed=42)

    # Evaluate single-source baselines
    coverage_a_only = evaluate_coverage(
        synth_df_a,
        {'survey_a': holdout_a},
        survey_a_vars,
        n_periods=n_periods,
    )

    coverage_b_only = evaluate_coverage(
        synth_df_b,
        {'survey_b': holdout_b},
        survey_b_vars,
        n_periods=n_periods,
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Survey A Coverage':<20} {'Survey B Coverage':<20}")
    print("-" * 70)
    print(f"{'Fused (A + B)':<30} {coverage.get('survey_a', float('nan')):<20.4f} {coverage.get('survey_b', float('nan')):<20.4f}")
    print(f"{'Single-source A only':<30} {coverage_a_only.get('survey_a', float('nan')):<20.4f} {'N/A':<20}")
    print(f"{'Single-source B only':<30} {'N/A':<20} {coverage_b_only.get('survey_b', float('nan')):<20.4f}")

    print(f"\nFull-variable coverage (fusion only): {coverage.get('full', float('nan')):.4f}")
    print("\n(Lower coverage = better; measures mean NN distance)")

    # Diagnostics: Check marginal distributions
    print("\n" + "=" * 70)
    print("DIAGNOSTICS: MARGINAL DISTRIBUTIONS")
    print("=" * 70)

    print("\nZero rates (% of records with value = 0):")
    print(f"{'Variable':<20} {'Holdout':<15} {'Synth (fused)':<15} {'Synth A-only':<15} {'Synth B-only':<15}")
    print("-" * 80)

    for var in all_variables:
        holdout_zero_rate = (holdout_full[var] == 0).mean() * 100
        synth_zero_rate = (synth_df[var] == 0).mean() * 100

        if var in survey_a_vars:
            synth_a_zero = (synth_df_a[var] == 0).mean() * 100
            synth_a_str = f"{synth_a_zero:.1f}%"
        else:
            synth_a_str = "N/A"

        if var in survey_b_vars:
            synth_b_zero = (synth_df_b[var] == 0).mean() * 100
            synth_b_str = f"{synth_b_zero:.1f}%"
        else:
            synth_b_str = "N/A"

        print(f"{var:<20} {holdout_zero_rate:<15.1f}% {synth_zero_rate:<15.1f}% {synth_a_str:<15} {synth_b_str:<15}")

    print("\nMean values (among nonzero):")
    print(f"{'Variable':<20} {'Holdout':<15} {'Synth (fused)':<15}")
    print("-" * 50)

    for var in all_variables:
        holdout_nz = holdout_full[holdout_full[var] > 0][var]
        synth_nz = synth_df[synth_df[var] > 0][var]

        if len(holdout_nz) > 0 and len(synth_nz) > 0:
            print(f"{var:<20} {holdout_nz.mean():<15,.0f} {synth_nz.mean():<15,.0f}")

    # Save model
    model_path = Path(__file__).parent / "fusion_synthesizer.pt"
    synthesizer.save(str(model_path))

    return synthesizer, synth_df, coverage


if __name__ == "__main__":
    synthesizer, synth_df, coverage = simulate_multi_source()
