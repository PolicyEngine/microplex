"""
Trajectory VAE for panel data synthesis.

Encodes person trajectories (T time periods, D features) into latent space,
with support for:
- Missing data (multi-survey fusion)
- Mixed types (continuous + categorical)
- Uncertainty quantification via sampling
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
class VAEConfig:
    """Configuration for TrajectoryVAE."""
    n_features: int
    latent_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    beta: float = 1.0  # KL weight
    learning_rate: float = 1e-3
    batch_size: int = 64


class TrajectoryEncoder(nn.Module):
    """Encode trajectory (T, D) -> latent (latent_dim)."""

    def __init__(self, config: VAEConfig, max_T: int = 48):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.n_features, config.hidden_dim)

        # Positional encoding for time
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_T, config.hidden_dim) * 0.02
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Pool over time -> latent
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Latent projections (mean and log variance)
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, T, n_features) trajectory data
            mask: (batch, T, n_features) True = observed, False = missing

        Returns:
            mu: (batch, latent_dim) posterior mean
            logvar: (batch, latent_dim) posterior log variance
        """
        batch, T, D = x.shape

        # Handle missing values: replace NaN with 0, will be masked
        x = torch.nan_to_num(x, nan=0.0)

        # Input projection + positional encoding
        h = self.input_proj(x)  # (batch, T, hidden)
        h = h + self.pos_encoding[:, :T, :]

        # Create attention mask if needed (mask out missing observations)
        if mask is not None:
            # Any feature missing = mask that timestep
            src_key_padding_mask = ~mask.any(dim=-1)  # (batch, T)
        else:
            src_key_padding_mask = None

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)

        # Pool over time
        h = h.transpose(1, 2)  # (batch, hidden, T)
        h = self.pool(h).squeeze(-1)  # (batch, hidden)

        # Latent distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """Decode latent (latent_dim) -> trajectory (T, D)."""

    def __init__(self, config: VAEConfig, max_T: int = 48):
        super().__init__()
        self.config = config
        self.max_T = max_T

        # Latent -> hidden
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_T, config.hidden_dim) * 0.02
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.n_features)

        # Output distribution parameters (for uncertainty)
        self.output_logvar = nn.Linear(config.hidden_dim, config.n_features)

    def forward(
        self,
        z: torch.Tensor,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, latent_dim) latent codes
            T: number of time steps to generate

        Returns:
            mu: (batch, T, n_features) output means
            logvar: (batch, T, n_features) output log variances
        """
        batch = z.shape[0]

        # Project latent to hidden
        memory = self.latent_proj(z).unsqueeze(1)  # (batch, 1, hidden)

        # Create time queries
        queries = self.pos_encoding[:, :T, :].expand(batch, -1, -1)  # (batch, T, hidden)

        # Decode
        h = self.transformer(queries, memory)  # (batch, T, hidden)

        # Output distribution
        mu = self.output_proj(h)
        logvar = self.output_logvar(h)

        return mu, logvar


class TrajectoryVAE(BaseSynthesisModel):
    """
    Variational Autoencoder for trajectory/panel data synthesis.

    Learns latent representation of individual trajectories,
    enabling:
    - Unconditional generation (sample from prior)
    - Conditional generation / imputation (encode partial, decode full)
    - Embedding extraction for coverage evaluation
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        beta: float = 1.0,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
    ):
        self.config = VAEConfig(
            n_features=n_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            beta=beta,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        self.latent_dim = latent_dim
        self.n_features = n_features

        self.encoder = TrajectoryEncoder(self.config)
        self.decoder = TrajectoryDecoder(self.config)

        self.is_fitted = False
        self.feature_columns: list[str] = []
        self.id_col = "person_id"
        self.time_col = "period"

        # Normalization stats
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_trajectories(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Convert DataFrame to trajectory tensor.

        Args:
            df: Panel data with person_id, period, and feature columns

        Returns:
            data: (n_persons, T, n_features) array
            mask: (n_persons, T, n_features) boolean array (True = observed)
            feature_cols: list of feature column names
        """
        # Identify columns
        id_col = self.id_col if self.id_col in df.columns else None
        time_col = self.time_col if self.time_col in df.columns else None

        exclude = {id_col, time_col} - {None}
        feature_cols = [c for c in df.columns if c not in exclude]

        if not self.feature_columns:
            self.feature_columns = feature_cols

        # Get unique persons and times
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

        # Build tensor
        n_features = len(feature_cols)
        data = np.full((n_persons, T, n_features), np.nan)
        mask = np.zeros((n_persons, T, n_features), dtype=bool)

        if id_col and time_col:
            # Panel data
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
            # Cross-sectional
            for i, (_, row) in enumerate(df.iterrows()):
                for fi, fc in enumerate(feature_cols):
                    val = row[fc]
                    if pd.notna(val):
                        data[i, 0, fi] = val
                        mask[i, 0, fi] = True

        return data, mask, feature_cols

    def _normalize(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Normalize data using training statistics."""
        if self.feature_means is None:
            # Compute from observed values only
            self.feature_means = np.nanmean(data, axis=(0, 1))
            self.feature_stds = np.nanstd(data, axis=(0, 1))
            self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds)

        normalized = (data - self.feature_means) / self.feature_stds
        return np.nan_to_num(normalized, nan=0.0)

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        return data * self.feature_stds + self.feature_means

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def fit(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame | None = None,
        epochs: int = 100,
        verbose: bool = True,
        **kwargs,
    ) -> Self:
        """
        Fit VAE on trajectory data.

        Args:
            data: Panel DataFrame with person_id, period, and features
            mask: Optional mask DataFrame (True = observed)
            epochs: Number of training epochs
            verbose: Print training progress

        Returns:
            self
        """
        # Convert to trajectories
        traj_data, traj_mask, _ = self._to_trajectories(data)

        # Normalize
        traj_normalized = self._normalize(traj_data, traj_mask)

        # Convert to tensors
        X = torch.tensor(traj_normalized, dtype=torch.float32)
        M = torch.tensor(traj_mask, dtype=torch.bool)

        # Create dataloader
        dataset = TensorDataset(X, M)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Move to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.config.learning_rate,
        )

        # Training loop
        self.encoder.train()
        self.decoder.train()

        T = X.shape[1]

        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kl = 0

            for batch_x, batch_m in loader:
                batch_x = batch_x.to(self.device)
                batch_m = batch_m.to(self.device)

                optimizer.zero_grad()

                # Encode
                mu, logvar = self.encoder(batch_x, batch_m)

                # Reparameterize
                z = self._reparameterize(mu, logvar)

                # Decode
                recon_mu, recon_logvar = self.decoder(z, T)

                # Reconstruction loss (masked)
                recon_var = torch.exp(recon_logvar)
                recon_loss = 0.5 * (
                    recon_logvar + (batch_x - recon_mu) ** 2 / recon_var
                )
                recon_loss = (recon_loss * batch_m.float()).sum() / batch_m.sum()

                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # Total loss
                loss = recon_loss + self.config.beta * kl_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss={total_loss/len(loader):.4f} "
                    f"Recon={total_recon/len(loader):.4f} "
                    f"KL={total_kl/len(loader):.4f}"
                )

        self.is_fitted = True
        self.encoder.eval()
        self.decoder.eval()

        return self

    def encode(
        self,
        data: pd.DataFrame,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Encode trajectories to latent space.

        Args:
            data: Panel DataFrame or SyntheticPopulation.persons
            deterministic: If True, return posterior mean; else sample

        Returns:
            (n_persons, latent_dim) array of latent codes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before encoding")

        if isinstance(data, pd.DataFrame):
            traj_data, traj_mask, _ = self._to_trajectories(data)
        else:
            raise TypeError(f"Expected DataFrame, got {type(data)}")

        traj_normalized = self._normalize(traj_data, traj_mask)

        X = torch.tensor(traj_normalized, dtype=torch.float32).to(self.device)
        M = torch.tensor(traj_mask, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            mu, logvar = self.encoder(X, M)

            if deterministic:
                z = mu
            else:
                z = self._reparameterize(mu, logvar)

        return z.cpu().numpy()

    def generate(
        self,
        n: int,
        T: int = 12,
        seed: int | None = None,
        **kwargs,
    ) -> SyntheticPopulation:
        """
        Generate synthetic trajectories by sampling from prior.

        Args:
            n: Number of individuals to generate
            T: Number of time periods
            seed: Random seed

        Returns:
            SyntheticPopulation with generated trajectories
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Sample from prior
        z = torch.randn(n, self.latent_dim, device=self.device)

        with torch.no_grad():
            # Decode
            recon_mu, recon_logvar = self.decoder(z, T)

            # Sample from output distribution
            recon_std = torch.exp(0.5 * recon_logvar)
            samples = recon_mu + recon_std * torch.randn_like(recon_mu)

        # Denormalize
        samples_np = samples.cpu().numpy()
        samples_denorm = self._denormalize(samples_np)

        # Convert to DataFrame
        records = []
        for pid in range(n):
            for t in range(T):
                record = {
                    "person_id": pid,
                    "period": t,
                }
                for fi, fc in enumerate(self.feature_columns):
                    record[fc] = samples_denorm[pid, t, fi]
                records.append(record)

        persons_df = pd.DataFrame(records)

        return SyntheticPopulation(persons=persons_df)

    def reconstruct(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct trajectories (encode then decode).

        Args:
            data: Panel DataFrame

        Returns:
            Reconstructed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before reconstruction")

        traj_data, traj_mask, _ = self._to_trajectories(data)
        traj_normalized = self._normalize(traj_data, traj_mask)
        T = traj_data.shape[1]

        X = torch.tensor(traj_normalized, dtype=torch.float32).to(self.device)
        M = torch.tensor(traj_mask, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            mu, logvar = self.encoder(X, M)
            z = mu  # Use mean for reconstruction
            recon_mu, _ = self.decoder(z, T)

        recon_np = recon_mu.cpu().numpy()
        recon_denorm = self._denormalize(recon_np)

        # Convert to DataFrame
        n_persons = recon_denorm.shape[0]
        records = []
        for pid in range(n_persons):
            for t in range(T):
                record = {
                    "person_id": pid,
                    "period": t,
                }
                for fi, fc in enumerate(self.feature_columns):
                    record[fc] = recon_denorm[pid, t, fi]
                records.append(record)

        return pd.DataFrame(records)

    def impute(
        self,
        partial_obs: pd.DataFrame,
        n_samples: int = 100,
        **kwargs,
    ) -> ImputationResult:
        """
        Conditional generation: given partial observations, sample the rest.

        Args:
            partial_obs: DataFrame with some columns filled, others NaN
            n_samples: Number of samples per input row

        Returns:
            ImputationResult with samples
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before imputation")

        n_rows = len(partial_obs)

        # Add dummy IDs if needed
        if self.id_col not in partial_obs.columns:
            partial_obs = partial_obs.copy()
            partial_obs[self.id_col] = range(n_rows)
        if self.time_col not in partial_obs.columns:
            partial_obs = partial_obs.copy()
            partial_obs[self.time_col] = 0

        traj_data, traj_mask, _ = self._to_trajectories(partial_obs)
        traj_normalized = self._normalize(traj_data, traj_mask)
        T = traj_data.shape[1]

        X = torch.tensor(traj_normalized, dtype=torch.float32).to(self.device)
        M = torch.tensor(traj_mask, dtype=torch.bool).to(self.device)

        all_samples = []

        with torch.no_grad():
            for s in range(n_samples):
                # Encode with stochastic sampling
                mu, logvar = self.encoder(X, M)
                z = self._reparameterize(mu, logvar)

                # Decode
                recon_mu, recon_logvar = self.decoder(z, T)
                recon_std = torch.exp(0.5 * recon_logvar)
                sample = recon_mu + recon_std * torch.randn_like(recon_mu)

                sample_np = sample.cpu().numpy()
                sample_denorm = self._denormalize(sample_np)

                # Convert to records
                for pid in range(n_rows):
                    record = {"_input_row_id": pid}
                    for fi, fc in enumerate(self.feature_columns):
                        record[fc] = sample_denorm[pid, 0, fi]  # T=0 for cross-section
                    all_samples.append(record)

        samples_df = pd.DataFrame(all_samples)

        # Input mask (which features were observed)
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
        """
        Compute approximate log probability (ELBO) for data.

        Args:
            data: Panel DataFrame
            mask: Optional mask

        Returns:
            Array of log probabilities per row
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing log_prob")

        traj_data, traj_mask, _ = self._to_trajectories(data)
        traj_normalized = self._normalize(traj_data, traj_mask)
        T = traj_data.shape[1]

        X = torch.tensor(traj_normalized, dtype=torch.float32).to(self.device)
        M = torch.tensor(traj_mask, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            mu, logvar = self.encoder(X, M)
            z = self._reparameterize(mu, logvar)
            recon_mu, recon_logvar = self.decoder(z, T)

            # Reconstruction log prob
            recon_var = torch.exp(recon_logvar)
            log_prob = -0.5 * (
                recon_logvar + (X - recon_mu) ** 2 / recon_var + np.log(2 * np.pi)
            )
            log_prob = (log_prob * M.float()).sum(dim=(1, 2))

            # KL per sample
            kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)

            elbo = log_prob - kl

        # Expand to per-row (accounting for T observations per person)
        elbo_per_person = elbo.cpu().numpy()

        # Map back to original DataFrame rows
        n_persons = len(elbo_per_person)
        n_rows = len(data)

        if n_rows == n_persons * T:
            # Panel data: repeat ELBO for each time step
            return np.repeat(elbo_per_person / T, T)
        else:
            return elbo_per_person
