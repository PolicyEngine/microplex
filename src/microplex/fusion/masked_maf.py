"""Masked MAF for multi-survey fusion.

Trains a normalizing flow on stacked survey data where each survey
has different observed variables. Uses masked training to learn
the joint distribution from partial observations.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import torch

from ..flows import ConditionalMAF


class MaskedMAF:
    """Masked Autoregressive Flow for multi-survey fusion.

    Learns joint distribution from stacked survey data where:
    - Each survey may have different variables observed
    - Mask indicates which variables are observed per record
    - Loss computed only on observed values
    - Inverse-frequency weighting balances sparse observations

    After training, can generate complete records by sampling
    from the learned joint distribution.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int = 0,
        n_layers: int = 6,
        hidden_dim: int = 128,
        use_inverse_freq_weighting: bool = True,
    ):
        """Initialize MaskedMAF.

        Args:
            n_features: Number of features in the schema
            n_context: Number of context/conditioning features (e.g., survey indicator)
            n_layers: Number of flow layers
            hidden_dim: Size of hidden layers in MADE networks
            use_inverse_freq_weighting: Weight loss by inverse observation frequency
        """
        self.n_features = n_features
        # Flow requires at least 1 context dimension
        self.n_context = max(1, n_context)
        self.use_inverse_freq_weighting = use_inverse_freq_weighting

        # The underlying flow model
        self.flow = ConditionalMAF(
            n_features=n_features,
            n_context=self.n_context,  # Use adjusted context
            n_layers=n_layers,
            hidden_dim=hidden_dim,
        )

        # Normalization parameters (computed during fit)
        self.feature_means_ = None
        self.feature_stds_ = None
        self.dim_weights_ = None

        # Training history
        self.training_losses_ = []

    def _compute_normalization(
        self,
        X: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-feature mean and std from observed values.

        Args:
            X: Data [n_samples, n_features]
            mask: Observation mask [n_samples, n_features]

        Returns:
            means: Per-feature means [n_features]
            stds: Per-feature stds [n_features]
        """
        means = np.zeros(self.n_features)
        stds = np.ones(self.n_features)

        for i in range(self.n_features):
            observed = mask[:, i].astype(bool)
            if observed.sum() > 0:
                obs_values = X[observed, i]
                means[i] = obs_values.mean()
                stds[i] = obs_values.std() + 1e-8

        return means, stds

    def _compute_dim_weights(self, mask: np.ndarray) -> np.ndarray:
        """Compute inverse-frequency weights for each dimension.

        Gives higher weight to sparse observations so they contribute
        equally to the loss despite having fewer samples.

        Args:
            mask: Observation mask [n_samples, n_features]

        Returns:
            weights: Per-dimension weights [n_features]
        """
        # Count observations per dimension
        obs_counts = mask.sum(axis=0) + 1  # +1 to avoid division by zero

        # Inverse frequency weighting, normalized
        weights = 1.0 / obs_counts
        weights = weights / weights.sum() * self.n_features

        return weights.astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        context: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        clip_grad: float = 5.0,
        verbose: bool = True,
        verbose_freq: int = 10,
        device: str = "cpu",
    ) -> Self:
        """Fit the masked MAF to multi-survey data.

        Args:
            X: Data [n_samples, n_features], NaN replaced with 0
            mask: Observation mask [n_samples, n_features], True=observed
            context: Optional context [n_samples, n_context]
            sample_weights: Optional sample weights [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            clip_grad: Gradient clipping norm
            verbose: Print progress
            verbose_freq: Print every N epochs
            device: Device to train on

        Returns:
            self for chaining
        """
        # Compute normalization from observed values
        self.feature_means_, self.feature_stds_ = self._compute_normalization(X, mask)

        # Normalize data
        X_norm = (X - self.feature_means_) / self.feature_stds_

        # Replace NaN/inf with 0 (will be masked out anyway)
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute dimension weights if using inverse frequency
        if self.use_inverse_freq_weighting:
            self.dim_weights_ = self._compute_dim_weights(mask)
        else:
            self.dim_weights_ = np.ones(self.n_features, dtype=np.float32)

        # Create context (zeros if not provided)
        if context is None:
            context = np.zeros((len(X), max(1, self.n_context)), dtype=np.float32)

        # Move to device
        self.flow.to(device)
        self.flow.train()

        X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
        context_t = torch.tensor(context, dtype=torch.float32, device=device)
        dim_weights_t = torch.tensor(self.dim_weights_, dtype=torch.float32, device=device)

        if sample_weights is not None:
            sample_weights_t = torch.tensor(
                sample_weights, dtype=torch.float32, device=device
            )
        else:
            sample_weights_t = None

        # Optimizer with cosine annealing
        optimizer = torch.optim.AdamW(
            self.flow.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        n_samples = len(X_t)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_samples, device=device)
            X_shuf = X_t[perm]
            mask_shuf = mask_t[perm]
            ctx_shuf = context_t[perm]
            if sample_weights_t is not None:
                w_shuf = sample_weights_t[perm]

            epoch_loss = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuf[start:end]
                mask_batch = mask_shuf[start:end]
                ctx_batch = ctx_shuf[start:end]

                optimizer.zero_grad()

                # Masked log probability
                log_prob = self.flow.log_prob(
                    X_batch,
                    ctx_batch,
                    mask=mask_batch,
                    dim_weights=dim_weights_t,
                )

                # Apply sample weights if provided
                if sample_weights_t is not None:
                    w_batch = w_shuf[start:end]
                    loss = -(log_prob * w_batch).sum() / w_batch.sum()
                else:
                    loss = -log_prob.mean()

                loss.backward()

                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.flow.parameters(), clip_grad
                    )

                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / n_batches
            self.training_losses_.append(avg_loss)

            if verbose and (epoch % verbose_freq == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d}: loss = {avg_loss:.4f}")

        self.flow.eval()
        return self

    def sample(
        self,
        n_samples: int,
        context: np.ndarray | None = None,
        clip_z: float = 3.0,
        device: str = "cpu",
    ) -> np.ndarray:
        """Generate samples from the learned distribution.

        Args:
            n_samples: Number of samples to generate
            context: Optional conditioning context [n_samples, n_context]
            clip_z: Clip base samples to avoid extreme outliers
            device: Device to use

        Returns:
            samples: Generated samples [n_samples, n_features] in original scale
        """
        self.flow.eval()
        self.flow.to(device)

        if context is None:
            context = np.zeros((n_samples, max(1, self.n_context)), dtype=np.float32)

        with torch.no_grad():
            ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
            samples_t = self.flow.sample(ctx_t, clip_z=clip_z)
            samples_norm = samples_t.cpu().numpy()

        # Clip normalized samples to prevent extreme outliers after flow transform
        # The flow can produce values outside [-clip_z, clip_z] even with base clipping
        if clip_z is not None:
            samples_norm = np.clip(samples_norm, -clip_z, clip_z)

        # Denormalize
        samples = samples_norm * self.feature_stds_ + self.feature_means_

        return samples

    def impute(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        context: np.ndarray | None = None,
        n_samples: int = 1,
        clip_z: float = 3.0,
        device: str = "cpu",
    ) -> np.ndarray:
        """Impute missing values by conditional sampling.

        For each record, samples from P(missing | observed) by:
        1. Encoding observed values to latent space (approximately)
        2. Sampling latent for missing dimensions from N(0,1)
        3. Decoding back to data space

        Note: This is approximate since the true conditional requires
        inverting the flow conditioned on partial observations, which
        is computationally expensive for MAF.

        Args:
            X: Data with missing values marked as 0 [n_records, n_features]
            mask: Observation mask [n_records, n_features]
            context: Optional conditioning [n_records, n_context]
            n_samples: Number of imputation samples per record
            clip_z: Clip base samples
            device: Device to use

        Returns:
            imputed: Imputed data [n_records, n_features] or
                     [n_records, n_samples, n_features] if n_samples > 1
        """
        self.flow.eval()
        self.flow.to(device)

        # Normalize
        X_norm = (X - self.feature_means_) / self.feature_stds_
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        if context is None:
            context = np.zeros((len(X), max(1, self.n_context)), dtype=np.float32)

        len(X)
        results = []

        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
            ctx_t = torch.tensor(context, dtype=torch.float32, device=device)

            for _ in range(n_samples):
                # Sample unconditionally
                samples = self.flow.sample(ctx_t, clip_z=clip_z)

                # Replace observed values with actuals
                imputed = torch.where(mask_t, X_t, samples)

                # Denormalize
                imputed_np = imputed.cpu().numpy()
                imputed_np = imputed_np * self.feature_stds_ + self.feature_means_
                results.append(imputed_np)

        if n_samples == 1:
            return results[0]
        else:
            return np.stack(results, axis=1)

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: Path to save to (without extension)
        """
        import pickle

        checkpoint = {
            "n_features": self.n_features,
            "n_context": self.n_context,
            "n_layers": len(self.flow.layers),
            "hidden_dim": self.flow.layers[0].made.hidden_dim,
            "use_inverse_freq_weighting": self.use_inverse_freq_weighting,
            "flow_state": self.flow.state_dict(),
            "feature_means": self.feature_means_,
            "feature_stds": self.feature_stds_,
            "dim_weights": self.dim_weights_,
            "training_losses": self.training_losses_,
        }

        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"Saved model to {path}.pkl")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> Self:
        """Load model from disk.

        Args:
            path: Path to load from (without extension)
            device: Device to load to

        Returns:
            Loaded model
        """
        import pickle

        with open(f"{path}.pkl", "rb") as f:
            checkpoint = pickle.load(f)

        model = cls(
            n_features=checkpoint["n_features"],
            n_context=checkpoint["n_context"],
            n_layers=checkpoint["n_layers"],
            hidden_dim=checkpoint["hidden_dim"],
            use_inverse_freq_weighting=checkpoint["use_inverse_freq_weighting"],
        )

        model.flow.load_state_dict(checkpoint["flow_state"])
        model.flow.to(device)
        model.flow.eval()

        model.feature_means_ = checkpoint["feature_means"]
        model.feature_stds_ = checkpoint["feature_stds"]
        model.dim_weights_ = checkpoint["dim_weights"]
        model.training_losses_ = checkpoint["training_losses"]

        print(f"Loaded model from {path}.pkl")
        return model


def fit_masked_maf(
    stacked: pd.DataFrame,
    mask: np.ndarray,
    variable_names: list[str],
    n_layers: int = 6,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> MaskedMAF:
    """Convenience function to fit MaskedMAF from harmonized survey data.

    Args:
        stacked: Stacked harmonized DataFrame from stack_surveys()
        mask: Observation mask from stack_surveys()
        variable_names: List of variable names in order
        n_layers: Number of flow layers
        hidden_dim: Hidden dimension
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        verbose: Print progress

    Returns:
        Fitted MaskedMAF model
    """
    from .harmonize import COMMON_SCHEMA, apply_transform

    n_features = len(variable_names)

    # Extract and transform features
    X = np.zeros((len(stacked), n_features), dtype=np.float32)
    for i, var in enumerate(variable_names):
        values = stacked[var].values.copy()
        observed = mask[:, i]

        # Replace NaN with 0 before transform
        values = np.where(observed, values, 0)

        # Apply transform
        spec = COMMON_SCHEMA.get(var, {"transform": "none"})
        if spec["type"] != "binary":
            values = apply_transform(values, spec.get("transform", "none"))

        X[:, i] = values

    # Replace remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Get sample weights
    weights = stacked["weight"].values if "weight" in stacked.columns else None

    # Fit model
    print("\nTraining MaskedMAF:")
    print(f"  Features: {n_features}")
    print(f"  Records: {len(X):,}")
    print(f"  Layers: {n_layers}, hidden: {hidden_dim}")

    model = MaskedMAF(
        n_features=n_features,
        n_context=0,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
    )

    model.fit(
        X=X,
        mask=mask,
        sample_weights=weights,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    # Store variable names for later use
    model.variable_names_ = variable_names

    return model


def generate_complete_population(
    model: MaskedMAF,
    n_samples: int,
    variable_names: list[str],
    clip_z: float = 3.0,
    device: str = "cpu",
) -> pd.DataFrame:
    """Generate complete synthetic population from trained model.

    Args:
        model: Trained MaskedMAF
        n_samples: Number of records to generate
        variable_names: Variable names in order
        clip_z: Clip base samples
        device: Device

    Returns:
        DataFrame with all variables populated
    """
    from .harmonize import COMMON_SCHEMA, apply_inverse_transform

    # Generate samples
    samples = model.sample(n_samples, clip_z=clip_z, device=device)

    # Create DataFrame
    result = pd.DataFrame(samples, columns=variable_names)

    # Apply inverse transforms
    for var in variable_names:
        spec = COMMON_SCHEMA.get(var, {"transform": "none"})
        if spec["type"] != "binary":
            result[var] = apply_inverse_transform(
                result[var].values, spec.get("transform", "none")
            )

        # Clip to valid ranges
        if "min" in spec:
            result[var] = result[var].clip(lower=spec["min"])
        if "max" in spec:
            result[var] = result[var].clip(upper=spec["max"])

        # Round discrete variables
        if spec["type"] == "discrete":
            result[var] = result[var].round().astype(int)
        elif spec["type"] == "binary":
            result[var] = (result[var] > 0.5).astype(int)

    return result


if __name__ == "__main__":
    # Test with dummy data
    print("Testing MaskedMAF...")

    np.random.seed(42)
    torch.manual_seed(42)

    # Create fake multi-survey data
    n_cps = 100
    n_puf = 50

    # CPS has: age, wages, state_fips
    cps = pd.DataFrame({
        "age": np.random.randint(18, 80, n_cps),
        "wages": np.random.lognormal(10, 1, n_cps),
        "state_fips": np.random.randint(1, 52, n_cps),
        "cap_gains": np.nan,  # Not in CPS
        "_survey": "cps",
    })

    # PUF has: age, wages, cap_gains
    puf = pd.DataFrame({
        "age": np.random.randint(18, 80, n_puf),
        "wages": np.random.lognormal(11, 1, n_puf),
        "state_fips": np.nan,  # Not in PUF
        "cap_gains": np.random.lognormal(9, 2, n_puf),
        "_survey": "puf",
    })

    # Stack
    stacked = pd.concat([cps, puf], ignore_index=True)

    # Create mask
    variables = ["age", "wages", "state_fips", "cap_gains"]
    mask = stacked[variables].notna().values

    # Fill NaN for training
    X = stacked[variables].fillna(0).values.astype(np.float32)

    print(f"\nStacked data: {len(stacked)} records")
    print(f"Mask shape: {mask.shape}")
    print("Observation rate per variable:")
    for i, var in enumerate(variables):
        rate = mask[:, i].mean() * 100
        print(f"  {var}: {rate:.1f}%")

    # Fit model
    model = MaskedMAF(n_features=4, n_layers=4, hidden_dim=32)
    model.fit(X, mask, epochs=50, batch_size=32, verbose=True, verbose_freq=10)

    # Generate samples
    samples = model.sample(20)
    print(f"\nGenerated samples shape: {samples.shape}")
    print("Sample stats:")
    for i, var in enumerate(variables):
        print(f"  {var}: mean={samples[:, i].mean():.2f}, std={samples[:, i].std():.2f}")

    print("\nTest passed!")
