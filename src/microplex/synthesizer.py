"""
Synthesizer: Main class for conditional microdata synthesis.

Uses normalizing flows to learn the joint distribution of target
variables conditioned on context variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .discrete import BinaryModel, DiscreteModelCollection
from .flows import ConditionalMAF
from .transforms import MultiVariableTransformer


@dataclass
class SynthesizerConfig:
    """Configuration for Synthesizer."""

    target_vars: list[str]
    condition_vars: list[str]
    discrete_vars: list[str] | None = None

    # Model architecture
    n_layers: int = 6
    hidden_dim: int = 64

    # Variable handling
    zero_inflated: bool = True
    log_transform: bool = True

    # Variance regularization
    variance_regularization: float = 0.1
    sample_clipping: float = 3.0  # Clip z-samples beyond this std (3.0 works well for log-normal)


class Synthesizer:
    """
    Conditional microdata synthesizer using normalizing flows.

    Learns P(target_vars | condition_vars) from training data,
    then generates synthetic target variables for new observations.

    Key features:
    - Handles zero-inflated variables (common in economic data)
    - Preserves joint correlations between target variables
    - Supports sample weights for survey data
    - Reproducible generation with seed parameter

    Example:
        >>> synth = Synthesizer(
        ...     target_vars=["income", "expenditure"],
        ...     condition_vars=["age", "education", "region"],
        ... )
        >>> synth.fit(training_data, weight_col="weight")
        >>> synthetic = synth.generate(new_demographics)
    """

    def __init__(
        self,
        target_vars: list[str],
        condition_vars: list[str],
        discrete_vars: list[str] | None = None,
        n_layers: int = 6,
        hidden_dim: int = 64,
        zero_inflated: bool = True,
        log_transform: bool = True,
        variance_regularization: float = 0.1,
        sample_clipping: float = 3.0,
    ):
        """
        Initialize synthesizer.

        Args:
            target_vars: Variables to synthesize (continuous)
            condition_vars: Variables to condition on (preserved in output)
            discrete_vars: Additional discrete target variables (optional)
            n_layers: Number of layers in normalizing flow
            hidden_dim: Hidden layer size
            zero_inflated: Whether target vars have many zeros
            log_transform: Whether to log-transform positive values
            variance_regularization: Weight for variance matching loss (0.0 to disable)
            sample_clipping: Clip z-samples beyond this many std devs (None to disable)
        """
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.discrete_vars = discrete_vars or []
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.zero_inflated = zero_inflated
        self.log_transform = log_transform
        self.variance_regularization = variance_regularization
        self.sample_clipping = sample_clipping

        # Will be set during fit
        self.transformer_: MultiVariableTransformer | None = None
        self.flow_model_: ConditionalMAF | None = None
        self.zero_indicators_: nn.ModuleDict | None = None
        self.discrete_model_: DiscreteModelCollection | None = None
        self.is_fitted_: bool = False
        self.training_history_: list[float] = []
        self._actual_n_context: int = 0  # Actual context dim (may include dummy)
        self._train_target_std: torch.Tensor | None = None  # Store target std for variance reg
        self._train_target_max: torch.Tensor | None = None  # Store max for clipping calibration
        self._original_scale_stats: dict[str, dict[str, float]] | None = None  # Original scale stats for clipping
        self._training_data: pd.DataFrame | None = None  # Store for full synthesis

    def _build_context_tensor(self, data: pd.DataFrame) -> torch.Tensor:
        if self.condition_vars:
            context_np = np.column_stack([data[var].values for var in self.condition_vars])
        else:
            context_np = np.zeros((len(data), 1))
        return torch.tensor(context_np, dtype=torch.float32)

    def _build_original_scale_stats(
        self,
        data: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for var in self.target_vars:
            values = np.asarray(data[var].values, dtype=float)
            positive_values = values[values > 0]
            if len(positive_values) > 0:
                stats[var] = {
                    "max": float(np.max(positive_values)),
                    "p99": float(np.percentile(positive_values, 99)),
                    "p999": float(np.percentile(positive_values, 99.9)),
                }
            else:
                stats[var] = {"max": 1.0, "p99": 1.0, "p999": 1.0}
        return stats

    def fit(
        self,
        data: pd.DataFrame,
        weight_col: str | None = "weight",
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        verbose: bool = True,
    ) -> Self:
        """
        Fit synthesizer on training data.

        Uses a two-stage approach:
        1. Binary models predict P(positive | context) for each variable
        2. Normalizing flow learns P(value | context) for positive cases

        Args:
            data: DataFrame with target and condition variables
            weight_col: Name of weight column (None if unweighted)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            verbose: Whether to print progress

        Returns:
            self
        """
        # Store training data for full synthesis mode
        self._training_data = data[self.condition_vars + self.target_vars].copy()

        # Prepare data dict for transforms
        data_dict = {col: data[col].values for col in data.columns}

        # Fit transforms on target variables
        self.transformer_ = MultiVariableTransformer(
            self.target_vars,
            zero_inflated=self.zero_inflated,
            log_transform=self.log_transform,
        )
        self.transformer_.fit(data_dict, weight_col=weight_col or "weight")

        # Transform target variables
        transformed = self.transformer_.transform(data_dict)

        # Store original scale statistics for adaptive clipping
        self._original_scale_stats = self._build_original_scale_stats(data)

        # Prepare tensors
        n_targets = len(self.target_vars)

        # Context tensor (handle empty condition_vars for unconditional generation)
        context = self._build_context_tensor(data)
        self._actual_n_context = len(self.condition_vars) or 1

        # Target tensor and observation mask
        # NaN values in ORIGINAL data indicate missing observations (from multi-survey stacking)
        # Check original data for NaN pattern first
        observation_mask_list = []
        has_missing = False

        for var in self.target_vars:
            raw_vals = data[var].values
            # Track which values are observed (not NaN) in ORIGINAL data
            is_observed = ~np.isnan(raw_vals)
            if not is_observed.all():
                has_missing = True
            observation_mask_list.append(is_observed.astype(np.float32))

        observation_mask_np = np.column_stack(observation_mask_list)

        # Now process transformed values
        targets_list = []
        for var in self.target_vars:
            vals = transformed[var].copy()
            # Replace NaN with 0 for tensor (masked loss will ignore these)
            vals = np.nan_to_num(vals, nan=0.0)
            targets_list.append(vals)

        targets_np = np.column_stack(targets_list)
        targets = torch.tensor(targets_np, dtype=torch.float32)

        # Observation mask: 1 = observed, 0 = missing (NaN in original data)
        observation_mask = torch.tensor(observation_mask_np, dtype=torch.float32) if has_missing else None

        # Compute observation frequencies for loss balancing
        if has_missing:
            obs_freq = observation_mask_np.mean(axis=0)
            # Inverse frequency weighting (more weight on rare observations)
            # Clamp to avoid division by zero and extreme weights
            self._obs_weights = torch.tensor(
                1.0 / np.clip(obs_freq, 0.1, 1.0),
                dtype=torch.float32
            )
            if verbose:
                print(f"  Observation frequencies: {dict(zip(self.target_vars, obs_freq.round(2)))}")
        else:
            self._obs_weights = None

        # Positive mask for each variable (only for observed values)
        positive_mask = torch.ones_like(targets)
        for i, var in enumerate(self.target_vars):
            # Use original data for positive check
            raw_vals = data[var].values
            is_positive = (raw_vals > 0).astype(np.float32)
            # NaN values should not be considered positive
            is_positive = np.where(np.isnan(raw_vals), 0.0, is_positive)
            positive_mask[:, i] = torch.tensor(is_positive)

        # Weights
        if weight_col and weight_col in data.columns:
            weights = torch.tensor(data[weight_col].values, dtype=torch.float32)
        else:
            weights = torch.ones(len(data), dtype=torch.float32)

        # Create normalizing flow
        # Use _actual_n_context which accounts for dummy context when condition_vars is empty
        self.flow_model_ = ConditionalMAF(
            n_features=n_targets,
            n_context=self._actual_n_context,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

        # Train flow on positive observations (with masked loss for missing data)
        self._train_flow(
            targets, context, weights, positive_mask,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
            observation_mask=observation_mask,
            dim_weights=self._obs_weights if has_missing else None,
        )

        # Train zero indicators
        self._train_zero_indicators(
            data, context,
            epochs=epochs // 2,
            learning_rate=learning_rate,
        )

        # Train discrete models if specified
        if self.discrete_vars:
            self._train_discrete(
                data, context,
                epochs=epochs // 2,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

        self.is_fitted_ = True
        return self

    def _train_flow(
        self,
        targets: torch.Tensor,
        context: torch.Tensor,
        weights: torch.Tensor,
        positive_mask: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        verbose: bool,
        observation_mask: torch.Tensor = None,
        dim_weights: torch.Tensor = None,
    ):
        """Train the normalizing flow model with variance regularization.

        Args:
            targets: Target variables [batch, n_targets]
            context: Context variables [batch, n_context]
            weights: Sample weights [batch]
            positive_mask: Which targets are positive (not zero) [batch, n_targets]
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print progress
            observation_mask: Optional mask for missing data [batch, n_targets]
                              1=observed, 0=missing (NaN in original data)
            dim_weights: Optional per-dimension weights for balancing sparse observations
        """
        self._dim_weights = dim_weights
        optimizer = torch.optim.Adam(
            self.flow_model_.parameters(), lr=learning_rate
        )

        # Combine positive_mask with observation_mask if provided
        # We only want to train on values that are both observed AND positive
        if observation_mask is not None:
            train_mask = positive_mask * observation_mask
        else:
            train_mask = positive_mask

        # For multi-survey data, train on all rows (using masked loss)
        # For single-survey data, can optionally filter to all-positive rows
        if observation_mask is not None:
            # Multi-survey mode: use all rows, masked loss handles missing
            train_targets = targets
            train_context = context
            train_weights = weights
            train_observation_mask = train_mask
        else:
            # Single-survey mode: filter to rows where all are positive
            all_positive = positive_mask.all(dim=1)
            if all_positive.sum() < 10:
                train_targets = targets
                train_context = context
                train_weights = weights
                train_observation_mask = None
            else:
                train_targets = targets[all_positive]
                train_context = context[all_positive]
                train_weights = weights[all_positive]
                train_observation_mask = None

        # Store target statistics for variance regularization and adaptive clipping
        self._train_target_std = train_targets.std(dim=0)
        self._train_target_mean = train_targets.mean(dim=0)
        self._train_target_max = train_targets.max(dim=0).values
        self._train_target_min = train_targets.min(dim=0).values

        if train_observation_mask is not None:
            dataset = TensorDataset(train_targets, train_context, train_weights, train_observation_mask)
        else:
            dataset = TensorDataset(train_targets, train_context, train_weights)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.training_history_ = []
        self._has_observation_mask = train_observation_mask is not None

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_var_loss = 0.0
            n_batches = 0

            for batch_data in loader:
                if self._has_observation_mask:
                    batch_targets, batch_context, batch_weights, batch_mask = batch_data
                else:
                    batch_targets, batch_context, batch_weights = batch_data
                    batch_mask = None

                optimizer.zero_grad()

                # Standard negative log-likelihood loss (with optional mask and dim weights)
                log_prob = self.flow_model_.log_prob(
                    batch_targets, batch_context,
                    mask=batch_mask,
                    dim_weights=self._dim_weights,
                )
                nll_loss = -(log_prob * batch_weights).sum() / batch_weights.sum()

                # Variance regularization: penalize log_scale being too small or too large
                # The log_scale parameters in MADE control variance - encourage them to stay near 0
                var_loss = torch.tensor(0.0)
                if self.variance_regularization > 0:
                    # Regularize the log_scale outputs to have variance matching target variance
                    # This is done by running forward pass and checking the log_det contribution
                    z = batch_targets
                    log_scale_sum = torch.tensor(0.0)
                    for layer in self.flow_model_.layers:
                        perm_idx = list(self.flow_model_.layers).index(layer)
                        perm = getattr(self.flow_model_, f"perm_{perm_idx}")
                        z = z[:, perm]
                        mu, log_scale = layer.made(z, batch_context)
                        # Penalize log_scale being too far from 0 (which means scale != 1)
                        # This helps prevent over/under-dispersion
                        log_scale_sum = log_scale_sum + (log_scale ** 2).mean()
                        z = (z - mu) * torch.exp(-log_scale)

                    var_loss = log_scale_sum / len(self.flow_model_.layers)

                # Combined loss
                loss = nll_loss + self.variance_regularization * var_loss

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.flow_model_.parameters(), 1.0)

                optimizer.step()

                epoch_loss += nll_loss.item()
                epoch_var_loss += var_loss.item() if isinstance(var_loss, torch.Tensor) else var_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_var_loss = epoch_var_loss / n_batches
            self.training_history_.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, NLL: {avg_loss:.4f}, VarLoss: {avg_var_loss:.4f}")

    def _train_zero_indicators(
        self,
        data: pd.DataFrame,
        context: torch.Tensor,
        epochs: int,
        learning_rate: float,
    ):
        """Train binary models for zero/non-zero indicators.

        Handles NaN values by only training on observed (non-NaN) samples.
        """
        self.zero_indicators_ = nn.ModuleDict()

        for var in self.target_vars:
            model = BinaryModel(
                n_context=self._actual_n_context,
                hidden_dim=self.hidden_dim // 2,
            )

            raw_vals = data[var].values
            # Only train on observed (non-NaN) values
            is_observed = ~np.isnan(raw_vals)
            if is_observed.sum() < 10:
                # Not enough observed values, use all
                is_observed = np.ones(len(raw_vals), dtype=bool)

            target = torch.tensor(
                (raw_vals[is_observed] > 0).astype(np.float32)
            ).unsqueeze(-1)
            var_context = context[is_observed]

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            for _ in range(epochs):
                optimizer.zero_grad()
                pred = model(var_context)
                loss = nn.functional.binary_cross_entropy(pred, target)
                loss.backward()
                optimizer.step()

            self.zero_indicators_[var] = model

    def _train_discrete(
        self,
        data: pd.DataFrame,
        context: torch.Tensor,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Train discrete variable models."""
        binary_vars = []
        categorical_vars = {}

        for var in self.discrete_vars:
            unique_vals = data[var].nunique()
            if unique_vals == 2:
                binary_vars.append(var)
            else:
                categorical_vars[var] = unique_vals

        self.discrete_model_ = DiscreteModelCollection(
            n_context=self._actual_n_context,
            binary_vars=binary_vars,
            categorical_vars=categorical_vars,
            hidden_dim=self.hidden_dim // 2,
        )

        targets = {
            var: torch.tensor(data[var].values, dtype=torch.long)
            for var in self.discrete_vars
        }

        optimizer = torch.optim.Adam(
            self.discrete_model_.parameters(), lr=learning_rate
        )

        for _ in range(epochs):
            optimizer.zero_grad()
            log_prob = self.discrete_model_.log_prob(context, targets)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic target variables for given conditions.

        Two-stage generation:
        1. Sample zero/non-zero indicators for each target variable
        2. For non-zero cases, sample from flow and inverse transform

        Args:
            conditions: DataFrame with condition variables
            seed: Random seed for reproducibility

        Returns:
            DataFrame with conditions + synthetic target variables
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare context tensor (handle empty condition_vars for unconditional generation)
        context = self._build_context_tensor(conditions)

        # Sample from flow (with optional clipping)
        with torch.no_grad():
            samples = self.flow_model_.sample(context, clip_z=self.sample_clipping)

        samples_np = samples.numpy()

        # Create dict with transformed values
        transformed_dict = {
            var: samples_np[:, i]
            for i, var in enumerate(self.target_vars)
        }

        # Inverse transform to original scale
        original_dict = self.transformer_.inverse_transform(transformed_dict)

        # Apply zero indicators
        with torch.no_grad():
            zero_indicators = self.zero_indicators_
            for var in self.target_vars:
                if zero_indicators is not None and var in zero_indicators:
                    prob_positive = zero_indicators[var](context).squeeze(-1)
                    is_positive = torch.bernoulli(prob_positive).numpy()
                    original_dict[var] = np.where(
                        is_positive > 0.5,
                        original_dict[var],
                        0.0
                    )

        # Ensure non-negative values and clip to training data range
        for var in self.target_vars:
            original_dict[var] = np.maximum(original_dict[var], 0)

            # Clip extreme values to slightly beyond training max
            # This prevents a few outliers from dominating variance
            if self._original_scale_stats and var in self._original_scale_stats:
                stats = self._original_scale_stats[var]
                # Allow up to 10% beyond the training max
                clip_max = stats['max'] * 1.1
                original_dict[var] = np.minimum(original_dict[var], clip_max)

        # Sample discrete variables
        if self.discrete_model_ is not None:
            with torch.no_grad():
                discrete_samples = self.discrete_model_.sample(context)
            for var in self.discrete_vars:
                original_dict[var] = discrete_samples[var].numpy().flatten()

        # Build result DataFrame
        result = conditions.copy()
        for var in self.target_vars:
            result[var] = original_dict[var]
        if self.discrete_model_ is not None:
            for var in self.discrete_vars:
                result[var] = original_dict[var]

        return result

    def sample(
        self,
        n: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate fully synthetic records (both conditions and targets).

        For full synthesis mode - samples conditions from training distribution,
        then generates targets conditioned on those.

        Args:
            n: Number of synthetic records to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with all variables (conditions + targets)
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if self._training_data is None:
            raise ValueError(
                "Full synthesis requires training data. "
                "Re-fit with store_training_data=True or use generate() with conditions."
            )

        if seed is not None:
            np.random.seed(seed)

        # Sample conditions from training distribution (with replacement)
        train_conditions = self._training_data[self.condition_vars]
        sampled_idx = np.random.choice(len(train_conditions), size=n, replace=True)
        conditions = train_conditions.iloc[sampled_idx].reset_index(drop=True)

        # Generate targets conditioned on sampled conditions
        return self.generate(conditions, seed=seed)

    def save(self, path: str | Path) -> None:
        """Save fitted model to disk."""
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        state = {
            "target_vars": self.target_vars,
            "condition_vars": self.condition_vars,
            "discrete_vars": self.discrete_vars,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "zero_inflated": self.zero_inflated,
            "log_transform": self.log_transform,
            "variance_regularization": self.variance_regularization,
            "sample_clipping": self.sample_clipping,
            "transformer": self.transformer_,
            "flow_state_dict": self.flow_model_.state_dict(),
            "zero_indicators_state_dict": (
                self.zero_indicators_.state_dict()
                if self.zero_indicators_ else None
            ),
            "discrete_state_dict": (
                self.discrete_model_.state_dict()
                if self.discrete_model_ else None
            ),
            "training_history": self.training_history_,
            "original_scale_stats": self._original_scale_stats,
        }

        torch.save(state, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load fitted model from disk."""
        state = torch.load(Path(path), weights_only=False)

        synth = cls(
            target_vars=state["target_vars"],
            condition_vars=state["condition_vars"],
            discrete_vars=state["discrete_vars"],
            n_layers=state["n_layers"],
            hidden_dim=state["hidden_dim"],
            zero_inflated=state["zero_inflated"],
            log_transform=state["log_transform"],
            variance_regularization=state.get("variance_regularization", 0.1),
            sample_clipping=state.get("sample_clipping", 3.0),
        )

        synth.transformer_ = state["transformer"]
        synth.training_history_ = state["training_history"]
        synth._original_scale_stats = state.get("original_scale_stats")

        # Reconstruct flow
        n_targets = len(state["target_vars"])
        n_context = len(state["condition_vars"])

        synth.flow_model_ = ConditionalMAF(
            n_features=n_targets,
            n_context=n_context,
            n_layers=state["n_layers"],
            hidden_dim=state["hidden_dim"],
        )
        synth.flow_model_.load_state_dict(state["flow_state_dict"])

        # Reconstruct zero indicators
        if state["zero_indicators_state_dict"]:
            synth.zero_indicators_ = nn.ModuleDict()
            for var in state["target_vars"]:
                synth.zero_indicators_[var] = BinaryModel(
                    n_context=n_context,
                    hidden_dim=state["hidden_dim"] // 2,
                )
            synth.zero_indicators_.load_state_dict(state["zero_indicators_state_dict"])

        # Reconstruct discrete model
        if state["discrete_state_dict"]:
            synth.discrete_model_ = DiscreteModelCollection(
                n_context=n_context,
                binary_vars=state["discrete_vars"],
                categorical_vars={},
                hidden_dim=state["hidden_dim"] // 2,
            )
            synth.discrete_model_.load_state_dict(state["discrete_state_dict"])

        synth.is_fitted_ = True
        return synth
