"""Unified panel evolution model.

Replaces separate transition classes (MarriageTransition, DivorceTransition,
Mortality, Disability) with a single autoregressive model:

    state[t+1] ~ state[t] + state[t-1] + ... + covariates

This treats all state transitions as conditional predictions, unifying:
- Binary transitions (marriage, divorce, death, disability)
- Continuous evolution (income, wealth)
- History-dependent dynamics (marriage duration affects divorce probability)

Key insight: there's no special "transition" logic - just P(Y_{t+1} | Y_t, Y_{t-1}, ..., X)

Example:
    >>> model = PanelEvolutionModel(
    ...     state_vars=["is_married", "income", "is_disabled"],
    ...     condition_vars=["age", "is_male", "education"],
    ...     lags=[1, 2, 3],
    ...     history_features={"is_married": ["duration", "ever"]},
    ... )
    >>> model.fit(psid_panel, epochs=100)
    >>>
    >>> # Simulate forward
    >>> next_year = model.simulate_step(current_state)
    >>> trajectory = model.simulate_trajectory(initial_state, n_steps=10)
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_lagged_features(
    df: pd.DataFrame,
    vars: list[str],
    lags: list[int],
    person_id_col: str = "person_id",
    period_col: str = "period",
) -> pd.DataFrame:
    """Create lagged features for panel data.

    Args:
        df: Panel DataFrame
        vars: Variables to create lags for
        lags: List of lag periods (e.g., [1, 2, 3])
        person_id_col: Column identifying individuals
        period_col: Column identifying time periods

    Returns:
        DataFrame with original columns plus lag columns (var_lag1, var_lag2, etc.)
    """
    result = df.copy()
    result = result.sort_values([person_id_col, period_col])

    for var in vars:
        if var not in df.columns:
            continue
        for lag in lags:
            col_name = f"{var}_lag{lag}"
            result[col_name] = result.groupby(person_id_col)[var].shift(lag)

    return result


def create_history_features(
    df: pd.DataFrame,
    state_var: str,
    feature_type: Literal["duration", "ever", "trend"],
    lookback: int = 3,
    person_id_col: str = "person_id",
    period_col: str = "period",
) -> pd.DataFrame:
    """Create derived history features from panel data.

    Args:
        df: Panel DataFrame
        state_var: Variable to compute history for
        feature_type: Type of history feature:
            - "duration": Consecutive periods in current state
            - "ever": Whether ever in state=1
            - "trend": Slope over lookback window
        lookback: Window size for trend calculation
        person_id_col: Column identifying individuals
        period_col: Column identifying time periods

    Returns:
        DataFrame with original columns plus history feature
    """
    result = df.copy()
    result = result.sort_values([person_id_col, period_col])

    if feature_type == "duration":
        # Count consecutive periods in state=1
        col_name = f"{state_var}_duration"

        def compute_duration(vals):
            durations = np.zeros(len(vals))
            current_duration = 0
            for i, v in enumerate(vals):
                if v == 1:
                    current_duration += 1
                    durations[i] = current_duration
                else:
                    current_duration = 0
                    durations[i] = 0
            return durations

        result[col_name] = result.groupby(person_id_col)[state_var].transform(
            lambda x: pd.Series(compute_duration(x.values), index=x.index)
        )

    elif feature_type == "ever":
        # Cumulative max (ever been in state=1)
        col_name = f"ever_{state_var}"
        result[col_name] = result.groupby(person_id_col)[state_var].cummax()

    elif feature_type == "trend":
        # Rolling slope over lookback window
        col_name = f"{state_var}_trend_{lookback}"

        def compute_trend(vals):
            trends = np.zeros(len(vals))
            for i in range(len(vals)):
                if i < lookback - 1:
                    trends[i] = np.nan
                else:
                    window = vals[i - lookback + 1:i + 1]
                    x = np.arange(lookback)
                    if np.std(window) > 0:
                        trends[i] = np.polyfit(x, window, 1)[0]
                    else:
                        trends[i] = 0
            return trends

        result[col_name] = result.groupby(person_id_col)[state_var].transform(
            lambda x: pd.Series(compute_trend(x.values), index=x.index)
        )

    return result


class PanelEvolutionNetwork(nn.Module):
    """Neural network for panel state evolution."""

    def __init__(
        self,
        input_dim: int,
        output_dims: dict[str, int],
        var_types: dict[str, str],
        hidden_dims: list[int] = [128, 64],
    ):
        super().__init__()

        self.var_types = var_types
        self.output_dims = output_dims

        # Shared trunk
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Per-variable output heads
        self.heads = nn.ModuleDict()
        for var, dim in output_dims.items():
            if var_types.get(var, "continuous") == "binary":
                # Binary: single logit
                self.heads[var] = nn.Linear(prev_dim, 1)
            else:
                # Continuous: mean and log_std
                self.heads[var] = nn.Linear(prev_dim, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning per-variable outputs."""
        h = self.trunk(x)
        outputs = {}
        for var in self.heads:
            outputs[var] = self.heads[var](h)
        return outputs


class PanelEvolutionModel:
    """Unified autoregressive model for panel state evolution.

    Learns P(state[t+1] | state[t], state[t-1], ..., covariates) for all
    state variables jointly, replacing separate transition classes.

    Attributes:
        state_vars: Variables to evolve (e.g., ["is_married", "income"])
        condition_vars: Static covariates (e.g., ["age", "is_male"])
        lags: How many periods back to look (e.g., [1, 2, 3])
        history_features: Derived features per variable (e.g., {"is_married": ["duration"]})
        var_types: Variable types ("binary" or "continuous")
    """

    def __init__(
        self,
        state_vars: list[str],
        condition_vars: list[str],
        lags: list[int] = [1],
        history_features: dict[str, list[str]] | None = None,
        var_types: dict[str, str] | None = None,
        hidden_dims: list[int] = [128, 64],
    ):
        self.state_vars = state_vars
        self.condition_vars = condition_vars
        self.lags = lags
        self.history_features = history_features or {}
        self.var_types = var_types or {}
        self.hidden_dims = hidden_dims

        self._is_fitted = False
        self._network = None
        self._feature_cols = None
        self._scaler_means = None
        self._scaler_stds = None

    def _build_features(
        self,
        df: pd.DataFrame,
        person_id_col: str,
        period_col: str,
    ) -> pd.DataFrame:
        """Build all features (lags + history) for training/prediction."""
        result = df.copy()

        # Add lagged features for state vars
        result = create_lagged_features(
            result,
            vars=self.state_vars,
            lags=self.lags,
            person_id_col=person_id_col,
            period_col=period_col,
        )

        # Add history features
        for var, feature_types in self.history_features.items():
            for ft in feature_types:
                result = create_history_features(
                    result,
                    state_var=var,
                    feature_type=ft,
                    person_id_col=person_id_col,
                    period_col=period_col,
                )

        return result

    def _get_feature_cols(self) -> list[str]:
        """Get list of all feature columns."""
        cols = list(self.condition_vars)

        # Add lag columns
        for var in self.state_vars:
            for lag in self.lags:
                cols.append(f"{var}_lag{lag}")

        # Add history feature columns
        for var, feature_types in self.history_features.items():
            for ft in feature_types:
                if ft == "duration":
                    cols.append(f"{var}_duration")
                elif ft == "ever":
                    cols.append(f"ever_{var}")
                elif ft == "trend":
                    cols.append(f"{var}_trend_3")  # Default lookback

        return cols

    def fit(
        self,
        data: pd.DataFrame,
        person_id_col: str = "person_id",
        period_col: str = "period",
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Self:
        """Fit the model on panel data.

        Args:
            data: Panel DataFrame with person-period observations
            person_id_col: Column identifying individuals
            period_col: Column identifying time periods
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            verbose: Whether to print progress

        Returns:
            self
        """
        # Build features
        df = self._build_features(data, person_id_col, period_col)

        # Get feature columns
        self._feature_cols = self._get_feature_cols()

        # Filter to rows with complete lag data
        available_cols = [c for c in self._feature_cols if c in df.columns]
        df_complete = df.dropna(subset=available_cols + self.state_vars)

        if len(df_complete) == 0:
            raise ValueError("No complete cases after creating lag features")

        # Prepare tensors
        X = df_complete[available_cols].values.astype(np.float32)

        # Normalize features
        self._scaler_means = X.mean(axis=0)
        self._scaler_stds = X.std(axis=0) + 1e-8
        X = (X - self._scaler_means) / self._scaler_stds

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Target tensors per variable
        Y_tensors = {}
        for var in self.state_vars:
            Y_tensors[var] = torch.tensor(
                df_complete[var].values.astype(np.float32),
                dtype=torch.float32
            )

        # Build network
        output_dims = {var: 1 for var in self.state_vars}
        self._network = PanelEvolutionNetwork(
            input_dim=len(available_cols),
            output_dims=output_dims,
            var_types=self.var_types,
            hidden_dims=self.hidden_dims,
        )

        # Training
        optimizer = torch.optim.Adam(self._network.parameters(), lr=lr)
        dataset = TensorDataset(X_tensor, *[Y_tensors[v] for v in self.state_vars])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x_batch = batch[0]
                y_batches = {var: batch[i + 1] for i, var in enumerate(self.state_vars)}

                optimizer.zero_grad()
                outputs = self._network(x_batch)

                loss = 0
                for var in self.state_vars:
                    y = y_batches[var]
                    out = outputs[var]

                    if self.var_types.get(var, "continuous") == "binary":
                        # Binary cross-entropy
                        loss += nn.functional.binary_cross_entropy_with_logits(
                            out.squeeze(), y
                        )
                    else:
                        # Gaussian NLL (mean, log_std)
                        mean = out[:, 0]
                        log_std = out[:, 1].clamp(-5, 5)
                        std = torch.exp(log_std)
                        loss += 0.5 * (((y - mean) / std) ** 2 + 2 * log_std).mean()

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

        self._is_fitted = True
        self._available_cols = available_cols
        return self

    def predict_proba(
        self,
        data: pd.DataFrame,
        person_id_col: str = "person_id",
        period_col: str = "period",
    ) -> pd.DataFrame:
        """Predict next-period probabilities/means.

        Args:
            data: Current state DataFrame (must include lag columns or raw panel)

        Returns:
            DataFrame with probability/mean columns for each state variable
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Build features if needed
        if f"{self.state_vars[0]}_lag1" not in data.columns:
            df = self._build_features(data, person_id_col, period_col)
        else:
            df = data.copy()

        # Get features
        X = df[self._available_cols].fillna(0).values.astype(np.float32)
        X = (X - self._scaler_means) / self._scaler_stds
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Predict
        self._network.eval()
        with torch.no_grad():
            outputs = self._network(X_tensor)

        result = df[[person_id_col, period_col]].copy()

        for var in self.state_vars:
            out = outputs[var].numpy()
            if self.var_types.get(var, "continuous") == "binary":
                result[f"{var}_prob"] = torch.sigmoid(torch.tensor(out)).numpy().squeeze()
            else:
                result[f"{var}_mean"] = out[:, 0]
                result[f"{var}_std"] = np.exp(np.clip(out[:, 1], -5, 5))

        return result

    def simulate_step(
        self,
        data: pd.DataFrame,
        person_id_col: str = "person_id",
        period_col: str = "period",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Simulate one step forward.

        Args:
            data: Current state DataFrame
            seed: Random seed

        Returns:
            DataFrame with simulated next-period states
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        probs = self.predict_proba(data, person_id_col, period_col)

        result = data[[person_id_col, period_col]].copy()
        result[period_col] = result[period_col] + 1

        for var in self.state_vars:
            if self.var_types.get(var, "continuous") == "binary":
                p = probs[f"{var}_prob"].values
                result[var] = (np.random.random(len(p)) < p).astype(int)
            else:
                mean = probs[f"{var}_mean"].values
                std = probs[f"{var}_std"].values
                result[var] = np.maximum(0, mean + std * np.random.randn(len(mean)))

        # Copy condition vars
        for var in self.condition_vars:
            if var in data.columns:
                result[var] = data[var].values
                # Increment age if present
                if var == "age":
                    result[var] = result[var] + 1

        return result

    def simulate_trajectory(
        self,
        initial_state: pd.DataFrame,
        n_steps: int,
        person_id_col: str = "person_id",
        period_col: str = "period",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Simulate multiple steps forward.

        Args:
            initial_state: Starting state DataFrame
            n_steps: Number of steps to simulate
            seed: Random seed

        Returns:
            DataFrame with full trajectory (initial + n_steps periods)
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        trajectories = [initial_state.copy()]
        current = initial_state.copy()

        # Build initial lag features
        for var in self.state_vars:
            for lag in self.lags:
                col = f"{var}_lag{lag}"
                if col not in current.columns:
                    current[col] = current[var] if var in current.columns else 0

        for step in range(n_steps):
            # Simulate next step
            next_state = self.simulate_step(current, person_id_col, period_col)

            # Update lag features for next iteration
            for var in self.state_vars:
                # Shift lags
                for lag in sorted(self.lags, reverse=True):
                    if lag > 1:
                        prev_col = f"{var}_lag{lag-1}"
                        next_state[f"{var}_lag{lag}"] = current[prev_col] if prev_col in current.columns else current[var]
                    else:
                        next_state[f"{var}_lag1"] = current[var] if var in current.columns else 0

            trajectories.append(next_state[[person_id_col, period_col] + self.state_vars + self.condition_vars])
            current = next_state

        return pd.concat(trajectories, ignore_index=True)
