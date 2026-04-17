"""
Autoregressive Transformer for trajectory/panel data synthesis.

Predicts next time step given previous steps, enabling:
- Realistic temporal dynamics
- Conditional generation from initial state
- Uncertainty via sampling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseSynthesisModel, ImputationResult, SyntheticPopulation


@dataclass
class TransformerConfig:
    """Configuration for TrajectoryTransformer."""
    n_features: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64


class CausalTransformerBlock(nn.Module):
    """Transformer block with causal (autoregressive) masking."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class TrajectoryTransformerModel(nn.Module):
    """Autoregressive Transformer for trajectory prediction."""

    def __init__(self, config: TransformerConfig, max_T: int = 48):
        super().__init__()
        self.config = config
        self.max_T = max_T

        # Input embedding
        self.input_proj = nn.Linear(config.n_features, config.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_T, config.hidden_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            CausalTransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # Output heads
        self.output_mu = nn.Linear(config.hidden_dim, config.n_features)
        self.output_logvar = nn.Linear(config.hidden_dim, config.n_features)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_T, max_T), diagonal=1).bool()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, T, n_features) input trajectories
            mask: (batch, T, n_features) True = observed

        Returns:
            mu: (batch, T, n_features) predicted means (shifted right)
            logvar: (batch, T, n_features) predicted log variances
        """
        batch, T, D = x.shape

        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)

        # Input projection + positional encoding
        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :T, :]

        # Get causal mask for this sequence length
        causal_mask = self.causal_mask[:T, :T]

        # Key padding mask for missing data
        if mask is not None:
            key_padding_mask = ~mask.any(dim=-1)  # (batch, T)
        else:
            key_padding_mask = None

        # Transformer layers
        for layer in self.layers:
            h = layer(h, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        # Output distribution
        mu = self.output_mu(h)
        logvar = self.output_logvar(h)

        return mu, logvar

    def get_embeddings(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Extract trajectory embeddings (pooled over time)."""
        batch, T, D = x.shape

        x = torch.nan_to_num(x, nan=0.0)

        h = self.input_proj(x)
        h = h + self.pos_encoding[:, :T, :]

        causal_mask = self.causal_mask[:T, :T]

        if mask is not None:
            key_padding_mask = ~mask.any(dim=-1)
        else:
            key_padding_mask = None

        for layer in self.layers:
            h = layer(h, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        # Pool over time (use last position or mean)
        embeddings = h.mean(dim=1)  # (batch, hidden_dim)

        return embeddings


class TrajectoryTransformer(BaseSynthesisModel):
    """
    Autoregressive Transformer for trajectory synthesis.

    Predicts p(x_t | x_{<t}) for each time step, enabling:
    - Realistic temporal dynamics
    - Conditional generation from initial state
    - Proper uncertainty quantification
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
    ):
        self.config = TransformerConfig(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        self.hidden_dim = hidden_dim
        self.n_features = n_features

        self.model = TrajectoryTransformerModel(self.config)

        self.is_fitted = False
        self.feature_columns: list[str] = []
        self.id_col = "person_id"
        self.time_col = "period"

        # Normalization
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_trajectories(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Convert DataFrame to trajectory tensor."""
        id_col = self.id_col if self.id_col in df.columns else None
        time_col = self.time_col if self.time_col in df.columns else None

        exclude = {id_col, time_col} - {None}
        feature_cols = [c for c in df.columns if c not in exclude]

        if not self.feature_columns:
            self.feature_columns = feature_cols

        if id_col:
            persons = df[id_col].unique()
            n_persons = len(persons)
        else:
            n_persons = len(df)
            persons = np.arange(n_persons)

        if time_col:
            times = sorted(df[time_col].unique())
            T = len(times)
        else:
            T = 1
            times = [0]

        n_features = len(feature_cols)
        data = np.full((n_persons, T, n_features), np.nan)
        mask = np.zeros((n_persons, T, n_features), dtype=bool)

        if id_col and time_col:
            person_idx = {p: i for i, p in enumerate(persons)}
            time_idx = {t: i for i, t in enumerate(times)}

            for _, row in df.iterrows():
                pi = person_idx[row[id_col]]
                ti = time_idx[row[time_col]]
                for fi, fc in enumerate(feature_cols):
                    val = row[fc]
                    if pd.notna(val):
                        data[pi, ti, fi] = val
                        mask[pi, ti, fi] = True
        else:
            for i, (_, row) in enumerate(df.iterrows()):
                for fi, fc in enumerate(feature_cols):
                    val = row[fc]
                    if pd.notna(val):
                        data[i, 0, fi] = val
                        mask[i, 0, fi] = True

        return data, mask, feature_cols

    def _normalize(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalize data."""
        if self.feature_means is None:
            self.feature_means = np.nanmean(data, axis=(0, 1))
            self.feature_stds = np.nanstd(data, axis=(0, 1))
            self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)

        normalized = (data - self.feature_means) / self.feature_stds
        return np.nan_to_num(normalized, nan=0.0)

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        return data * self.feature_stds + self.feature_means

    def fit(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame | None = None,
        epochs: int = 100,
        verbose: bool = True,
        **kwargs,
    ) -> Self:
        """
        Fit transformer on trajectory data.

        Uses teacher forcing: predict x_t from x_{<t}.
        """
        traj_data, traj_mask, _ = self._to_trajectories(data)
        traj_normalized = self._normalize(traj_data, traj_mask)

        X = torch.tensor(traj_normalized, dtype=torch.float32)
        M = torch.tensor(traj_mask, dtype=torch.bool)

        dataset = TensorDataset(X, M)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self.model.to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0

            for batch_x, batch_m in loader:
                batch_x = batch_x.to(self.device)
                batch_m = batch_m.to(self.device)

                optimizer.zero_grad()

                # Teacher forcing: input is x_{0:T-1}, target is x_{1:T}
                input_x = batch_x[:, :-1, :]
                input_m = batch_m[:, :-1, :]
                target_x = batch_x[:, 1:, :]
                target_m = batch_m[:, 1:, :]

                # Forward pass
                mu, logvar = self.model(input_x, input_m)

                # Clamp logvar to prevent numerical issues
                logvar = torch.clamp(logvar, min=-10, max=10)

                # Loss on observed targets
                var = torch.exp(logvar)
                loss = 0.5 * (logvar + (target_x - mu) ** 2 / var)
                loss = (loss * target_m.float()).sum() / target_m.sum().clamp(min=1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}")

        self.is_fitted = True
        self.model.eval()

        return self

    def generate(
        self,
        n: int,
        T: int = 12,
        seed: int | None = None,
        **kwargs,
    ) -> SyntheticPopulation:
        """
        Generate synthetic trajectories autoregressively.

        Samples initial state from learned distribution, then
        predicts each subsequent step.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.model.eval()

        # Initialize from prior (standard normal in normalized space)
        trajectories = torch.zeros(n, T, self.n_features, device=self.device)
        trajectories[:, 0, :] = torch.randn(n, self.n_features, device=self.device)

        with torch.no_grad():
            for t in range(1, T):
                # Get predictions from history
                input_x = trajectories[:, :t, :]
                mu, logvar = self.model(input_x, mask=None)

                # Sample next step from predicted distribution
                # Clamp logvar to prevent numerical overflow
                logvar_clamped = torch.clamp(logvar[:, -1, :], min=-10, max=10)
                std = torch.exp(0.5 * logvar_clamped)
                next_step = mu[:, -1, :] + std * torch.randn_like(std)
                # Clamp next step to prevent extreme values
                next_step = torch.clamp(next_step, min=-10, max=10)

                trajectories[:, t, :] = next_step

        # Denormalize
        traj_np = trajectories.cpu().numpy()
        traj_denorm = self._denormalize(traj_np)

        # Convert to DataFrame
        records = []
        for pid in range(n):
            for t in range(T):
                record = {"person_id": pid, "period": t}
                for fi, fc in enumerate(self.feature_columns):
                    record[fc] = traj_denorm[pid, t, fi]
                records.append(record)

        return SyntheticPopulation(persons=pd.DataFrame(records))

    def generate_from_initial(
        self,
        initial: pd.DataFrame,
        T: int = 12,
        seed: int | None = None,
    ) -> SyntheticPopulation:
        """
        Generate trajectories conditioned on initial state.

        Args:
            initial: DataFrame with initial values for each person
            T: Number of time steps to generate
            seed: Random seed

        Returns:
            SyntheticPopulation starting from initial conditions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        n = len(initial)

        # Normalize initial values
        initial_arr = initial[self.feature_columns].values
        initial_norm = (initial_arr - self.feature_means) / self.feature_stds
        initial_norm = np.nan_to_num(initial_norm, nan=0.0)

        trajectories = torch.zeros(n, T, self.n_features, device=self.device)
        trajectories[:, 0, :] = torch.tensor(initial_norm, dtype=torch.float32, device=self.device)

        self.model.eval()

        with torch.no_grad():
            for t in range(1, T):
                input_x = trajectories[:, :t, :]
                mu, logvar = self.model(input_x, mask=None)

                # Clamp logvar to prevent numerical overflow
                logvar_clamped = torch.clamp(logvar[:, -1, :], min=-10, max=10)
                std = torch.exp(0.5 * logvar_clamped)
                next_step = mu[:, -1, :] + std * torch.randn_like(std)
                # Clamp next step to prevent extreme values
                next_step = torch.clamp(next_step, min=-10, max=10)

                trajectories[:, t, :] = next_step

        # Denormalize
        traj_np = trajectories.cpu().numpy()
        traj_denorm = self._denormalize(traj_np)

        # Convert to DataFrame
        records = []
        for pid in range(n):
            for t in range(T):
                record = {"person_id": pid, "period": t}
                for fi, fc in enumerate(self.feature_columns):
                    record[fc] = traj_denorm[pid, t, fi]
                records.append(record)

        return SyntheticPopulation(persons=pd.DataFrame(records))

    def encode(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Extract trajectory embeddings.

        Args:
            data: Panel DataFrame

        Returns:
            (n_persons, hidden_dim) embeddings
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before encoding")

        traj_data, traj_mask, _ = self._to_trajectories(data)
        traj_normalized = self._normalize(traj_data, traj_mask)

        X = torch.tensor(traj_normalized, dtype=torch.float32).to(self.device)
        M = torch.tensor(traj_mask, dtype=torch.bool).to(self.device)

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(X, M)

        return embeddings.cpu().numpy()

    def impute(
        self,
        partial_obs: pd.DataFrame,
        n_samples: int = 100,
        **kwargs,
    ) -> ImputationResult:
        """Conditional generation given partial observations."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before imputation")

        # For now, treat partial observation as initial state
        # and generate forward
        n_rows = len(partial_obs)

        all_samples = []

        for s in range(n_samples):
            pop = self.generate_from_initial(partial_obs, T=1, seed=s)

            for pid in range(n_rows):
                record = {"_input_row_id": pid}
                person = pop.persons[pop.persons["person_id"] == pid].iloc[0]
                for fc in self.feature_columns:
                    record[fc] = person[fc]
                all_samples.append(record)

        samples_df = pd.DataFrame(all_samples)
        input_mask = ~partial_obs[self.feature_columns].isna()

        return ImputationResult(
            samples=samples_df,
            input_mask=input_mask,
            n_samples=n_samples,
        )

    def log_prob(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Compute log probability under the autoregressive model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")

        traj_data, traj_mask, _ = self._to_trajectories(data)
        traj_normalized = self._normalize(traj_data, traj_mask)

        X = torch.tensor(traj_normalized, dtype=torch.float32).to(self.device)
        M = torch.tensor(traj_mask, dtype=torch.bool).to(self.device)

        self.model.eval()
        with torch.no_grad():
            input_x = X[:, :-1, :]
            input_m = M[:, :-1, :]
            target_x = X[:, 1:, :]
            target_m = M[:, 1:, :]

            mu, logvar = self.model(input_x, input_m)

            # Log prob per sample
            var = torch.exp(logvar)
            log_prob = -0.5 * (
                logvar + (target_x - mu) ** 2 / var + np.log(2 * np.pi)
            )
            log_prob = (log_prob * target_m.float()).sum(dim=(1, 2))

        log_prob_np = log_prob.cpu().numpy()

        # Expand to per-row
        n_persons = len(log_prob_np)
        T = traj_data.shape[1]
        n_rows = len(data)

        if n_rows == n_persons * T:
            return np.repeat(log_prob_np / T, T)
        else:
            return log_prob_np
