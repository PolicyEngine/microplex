"""
Normalizing flow models for conditional generation.

Implements Conditional Masked Autoregressive Flow (MAF) for learning
the joint distribution of tax variables conditioned on demographics.
"""


from __future__ import annotations

from typing import Self

import numpy as np
import torch
import torch.nn as nn


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).

    Implements autoregressive property: output[i] only depends on input[:i].
    Used as the conditioner network in MAF.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        hidden_dim: int,
        n_hidden: int = 2,
    ):
        """
        Initialize MADE network.

        Args:
            n_features: Number of input/output features
            n_context: Number of context/conditioning features
            hidden_dim: Size of hidden layers
            n_hidden: Number of hidden layers
        """
        super().__init__()
        self.n_features = n_features
        self.n_context = n_context
        self.hidden_dim = hidden_dim

        # Create masks for autoregressive property
        self._create_masks(n_hidden)

        # Input layer: takes concatenated [x, context]
        self.input_layer = nn.Linear(n_features + n_context, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)
        ])

        # Output layer: predicts mu and log_scale for each feature
        self.output_layer = nn.Linear(hidden_dim, n_features * 2)

        self.activation = nn.ReLU()

    def _create_masks(self, n_hidden: int):
        """Create masks enforcing autoregressive property."""
        # Assign each hidden unit to a degree (which inputs it can see)
        rng = np.random.RandomState(42)

        # Input degrees: features have degrees 0 to n_features-1
        # Context features can connect to all (degree -1)
        input_degrees = np.concatenate([
            np.arange(self.n_features),
            np.full(self.n_context, -1)  # Context connects to all
        ])

        # Hidden degrees: uniform random assignment
        hidden_degrees = []
        for _ in range(n_hidden + 1):  # +1 for output layer
            if self.n_features > 1:
                degrees = rng.randint(0, self.n_features - 1, self.hidden_dim)
            else:
                # For single feature, all hidden units have degree 0
                degrees = np.zeros(self.hidden_dim, dtype=np.int64)
            hidden_degrees.append(degrees)

        # Output degrees: each output i needs degree < i
        output_degrees = np.arange(self.n_features)

        # Create masks
        # Input -> hidden1: hidden can see input if hidden_degree >= input_degree
        self.register_buffer(
            "input_mask",
            torch.tensor(
                hidden_degrees[0][:, None] >= input_degrees[None, :],
                dtype=torch.float32,
            ),
        )

        # Hidden -> hidden masks
        self.hidden_masks = []
        for i in range(n_hidden):
            mask = torch.tensor(
                hidden_degrees[i + 1][:, None] >= hidden_degrees[i][None, :],
                dtype=torch.float32,
            )
            self.register_buffer(f"hidden_mask_{i}", mask)
            self.hidden_masks.append(mask)

        # Hidden -> output: output i can see hidden if output_degree > hidden_degree
        # (strictly greater for autoregressive: output[i] depends on input[:i])
        self.register_buffer(
            "output_mask",
            torch.tensor(
                output_degrees[:, None] > hidden_degrees[-1][None, :],
                dtype=torch.float32,
            ),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MADE.

        Args:
            x: Input features [batch, n_features]
            context: Context features [batch, n_context]

        Returns:
            mu: Mean parameters [batch, n_features]
            log_scale: Log scale parameters [batch, n_features]
        """
        # Concatenate input and context
        h = torch.cat([x, context], dim=-1)

        # Masked input layer
        h = self.activation(
            nn.functional.linear(h, self.input_mask * self.input_layer.weight,
                                 self.input_layer.bias)
        )

        # Masked hidden layers
        for i, layer in enumerate(self.hidden_layers):
            mask = getattr(self, f"hidden_mask_{i}")
            h = self.activation(
                nn.functional.linear(h, mask * layer.weight, layer.bias)
            )

        # Masked output layer
        out = nn.functional.linear(
            h, self.output_mask.repeat(2, 1) * self.output_layer.weight,
            self.output_layer.bias
        )

        # Split into mu and log_scale
        mu, log_scale = out.chunk(2, dim=-1)

        # Clamp log_scale for stability
        log_scale = torch.clamp(log_scale, min=-5, max=3)

        return mu, log_scale


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer using MADE as the conditioner.

    Transform: z = (x - mu(x, context)) / exp(log_scale(x, context))
    This is invertible and the Jacobian is easy to compute.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        hidden_dim: int,
    ):
        """
        Initialize affine coupling layer.

        Args:
            n_features: Number of features to transform
            n_context: Number of context/conditioning features
            hidden_dim: Size of hidden layers in MADE
        """
        super().__init__()
        self.n_features = n_features
        self.made = MADE(n_features, n_context, hidden_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: x -> z.

        Args:
            x: Input [batch, n_features]
            context: Context [batch, n_context]

        Returns:
            z: Transformed output
            log_det: Log determinant of Jacobian
        """
        mu, log_scale = self.made(x, context)

        # Affine transform
        z = (x - mu) * torch.exp(-log_scale)

        # Log det Jacobian = -sum(log_scale)
        log_det = -log_scale.sum(dim=-1)

        return z, log_det

    def inverse(
        self, z: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse transformation: z -> x.

        Must be done autoregressively since mu, log_scale depend on x.

        Args:
            z: Latent space input
            context: Context features

        Returns:
            x: Reconstructed input
        """
        z.shape[0]
        x = torch.zeros_like(z)

        for i in range(self.n_features):
            mu, log_scale = self.made(x, context)
            x[:, i] = z[:, i] * torch.exp(log_scale[:, i]) + mu[:, i]

        return x


class ConditionalMAF(nn.Module):
    """
    Conditional Masked Autoregressive Flow.

    Stacks multiple affine coupling layers with permutations between them
    to model complex distributions.
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        n_layers: int = 4,
        hidden_dim: int = 64,
    ):
        """
        Initialize conditional MAF.

        Args:
            n_features: Number of features to model
            n_context: Number of context/conditioning features
            n_layers: Number of flow layers
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        self.n_features = n_features
        self.n_context = n_context

        # Stack of affine coupling layers
        self.layers = nn.ModuleList([
            AffineCouplingLayer(n_features, n_context, hidden_dim)
            for _ in range(n_layers)
        ])

        # Permutations between layers (reverse order alternating)
        self.permutations = []
        for i in range(n_layers):
            if i % 2 == 0:
                perm = torch.arange(n_features)
            else:
                perm = torch.arange(n_features - 1, -1, -1)
            self.register_buffer(f"perm_{i}", perm)
            self.permutations.append(perm)

        # Base distribution (standard normal)
        self.register_buffer(
            "base_mean", torch.zeros(n_features)
        )
        self.register_buffer(
            "base_std", torch.ones(n_features)
        )

    def log_prob(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor = None,
        dim_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute log probability of x given context, with optional masking.

        When mask is provided, only computes loss on observed (mask=1) dimensions.
        This enables training on multi-survey data with missing values.

        Args:
            x: Data [batch, n_features]
            context: Context [batch, n_context]
            mask: Optional observation mask [batch, n_features], 1=observed, 0=missing
            dim_weights: Optional per-dimension weights [n_features] for balancing
                         sparse observations (inverse frequency weighting)

        Returns:
            Log probability [batch]
        """
        z = x

        # Track per-dimension log_det contributions
        per_dim_log_det = torch.zeros_like(x)

        for i, layer in enumerate(self.layers):
            # Apply permutation
            perm = getattr(self, f"perm_{i}")
            z = z[:, perm]
            if mask is not None:
                # Track which original dimensions are where after permutation
                mask[:, perm]

            # Apply affine coupling - get per-dimension log_det
            mu, log_scale = layer.made(z, context)
            z_new = (z - mu) * torch.exp(-log_scale)

            # Accumulate per-dimension log_det (before permutation reversal)
            inv_perm = torch.argsort(perm)
            per_dim_log_det = per_dim_log_det + (-log_scale)[:, inv_perm]

            z = z_new

        # Per-dimension base log prob: -0.5 * (log(2*pi) + z^2)
        per_dim_base = -0.5 * (np.log(2 * np.pi) + z ** 2)

        # If mask provided, only sum over observed dimensions
        if mask is not None:
            # Mask is in original dimension order, z is in final permuted order
            # Need to apply final permutation to mask
            final_perm = getattr(self, f"perm_{len(self.layers)-1}")
            z_mask = mask[:, final_perm]

            # Apply dimension weights if provided (for balancing sparse observations)
            if dim_weights is not None:
                weighted_mask = mask * dim_weights.unsqueeze(0)
                weighted_z_mask = z_mask * dim_weights[final_perm].unsqueeze(0)
            else:
                weighted_mask = mask
                weighted_z_mask = z_mask

            # Masked base log prob (weighted)
            base_log_prob = (per_dim_base * weighted_z_mask).sum(dim=-1)

            # Masked log det - sum only observed dimensions (weighted)
            log_det = (per_dim_log_det * weighted_mask).sum(dim=-1)

            return base_log_prob + log_det
        else:
            # Standard: sum over all dimensions (with optional weights)
            if dim_weights is not None:
                base_log_prob = (per_dim_base * dim_weights.unsqueeze(0)).sum(dim=-1)
                log_det = (per_dim_log_det * dim_weights.unsqueeze(0)).sum(dim=-1)
            else:
                base_log_prob = per_dim_base.sum(dim=-1)
                log_det = per_dim_log_det.sum(dim=-1)
            return base_log_prob + log_det

    def sample(
        self,
        context: torch.Tensor,
        clip_z: float = None,
    ) -> torch.Tensor:
        """
        Sample from the flow given context.

        Args:
            context: Context [batch, n_context]
            clip_z: If provided, clip base samples to [-clip_z, clip_z]

        Returns:
            Samples [batch, n_features]
        """
        batch_size = context.shape[0]

        # Sample from base distribution
        z = torch.randn(batch_size, self.n_features, device=context.device)

        # Clip base samples to avoid extreme outliers
        if clip_z is not None:
            z = torch.clamp(z, min=-clip_z, max=clip_z)

        # Inverse transform through layers (in reverse order)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]

            # Inverse affine coupling
            z = layer.inverse(z, context)

            # Inverse permutation
            perm = getattr(self, f"perm_{i}")
            inv_perm = torch.argsort(perm)
            z = z[:, inv_perm]

        return z

    def fit(
        self,
        X: np.ndarray,
        context: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        verbose: bool = True,
        verbose_freq: int = 10,
        clip_grad: float = 5.0,
        device: str = "cpu",
    ) -> Self:
        """
        Train the flow on data.

        Args:
            X: Training data [n_samples, n_features]
            context: Conditioning data [n_samples, n_context]
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            verbose: Print progress
            verbose_freq: Print every N epochs
            clip_grad: Gradient clipping norm
            device: Device to train on

        Returns:
            self for chaining
        """
        self.to(device)
        self.train()

        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        C_t = torch.tensor(context, dtype=torch.float32, device=device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        n_samples = len(X_t)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Store training stats
        self.training_losses_ = []

        for epoch in range(epochs):
            # Shuffle data
            perm = torch.randperm(n_samples, device=device)
            X_shuffled = X_t[perm]
            C_shuffled = C_t[perm]

            epoch_loss = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                C_batch = C_shuffled[start:end]

                optimizer.zero_grad()

                # Negative log likelihood
                log_prob = self.log_prob(X_batch, C_batch)
                loss = -log_prob.mean()

                loss.backward()

                # Gradient clipping
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / n_batches
            self.training_losses_.append(avg_loss)

            if verbose and (epoch % verbose_freq == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch:4d}: loss = {avg_loss:.4f}")

        self.eval()
        return self

    def generate(
        self,
        context: np.ndarray,
        clip_z: float = 3.0,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Generate samples given context (numpy interface).

        Args:
            context: Conditioning data [n_samples, n_context]
            clip_z: Clip base distribution samples to avoid outliers
            device: Device to use

        Returns:
            Generated samples [n_samples, n_features]
        """
        self.eval()
        self.to(device)

        with torch.no_grad():
            C_t = torch.tensor(context, dtype=torch.float32, device=device)
            samples = self.sample(C_t, clip_z=clip_z)

        return samples.cpu().numpy()
