"""Unified sequence synthesizer.

One model that:
- Takes full person history (variable length)
- Attends to all prior periods automatically (no manual lag config)
- Predicts any target vars given any context
- Same model handles imputation and evolution

This unifies:
- Cross-sectional imputation: P(missing_var | observed_vars)
- Panel evolution: P(state[t+1] | state[t], state[t-1], ...)
- Multi-source fusion: P(var | context_from_any_source)

All are just: P(target | available_context)

Example:
    >>> model = SequenceSynthesizer(
    ...     continuous_vars=["income", "wealth"],
    ...     binary_vars=["is_married", "is_employed"],
    ... )
    >>> model.fit(panel_data, epochs=100)
    >>>
    >>> # Evolution: predict next period from history
    >>> next_state = model.predict_next(person_history)
    >>>
    >>> # Imputation: fill missing vars using history
    >>> imputed = model.impute(record_with_missing, target_vars=["wealth"])
    >>>
    >>> # Generation: simulate trajectory
    >>> trajectory = model.generate_trajectory(initial, n_periods=10)
"""

import math
from typing import Self

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def prepare_sequences(
    df: pd.DataFrame,
    vars: list[str],
    person_id_col: str = "person_id",
    period_col: str = "period",
) -> list[dict]:
    """Convert panel DataFrame to list of sequences.

    Args:
        df: Panel data with person-period observations
        vars: Variables to include in sequences
        person_id_col: Column identifying individuals
        period_col: Column identifying time periods

    Returns:
        List of dicts, each with:
            - "person_id": identifier
            - "periods": list of period values (sorted)
            - "values": dict of var_name -> list of values
    """
    sequences = []

    for person_id, group in df.groupby(person_id_col):
        group = group.sort_values(period_col)

        seq = {
            "person_id": person_id,
            "periods": group[period_col].tolist(),
            "values": {},
        }

        for var in vars:
            if var in group.columns:
                seq["values"][var] = group[var].tolist()

        sequences.append(seq)

    return sequences


def collate_variable_length(
    sequences: list[dict],
    vars: list[str],
    pad_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Collate variable-length sequences into padded batch.

    Args:
        sequences: List of sequence dicts from prepare_sequences
        vars: Variables to include
        pad_value: Value for padding

    Returns:
        Dict with:
            - "values": (batch, max_len, n_vars) tensor
            - "mask": (batch, max_len) tensor, 1=real, 0=padded
            - "lengths": (batch,) tensor of sequence lengths
    """
    batch_size = len(sequences)
    n_vars = len(vars)

    # Get lengths
    lengths = [len(seq["periods"]) for seq in sequences]
    max_len = max(lengths)

    # Build tensors
    values = torch.full((batch_size, max_len, n_vars), pad_value)
    mask = torch.zeros(batch_size, max_len)

    for i, seq in enumerate(sequences):
        seq_len = lengths[i]
        mask[i, :seq_len] = 1

        for j, var in enumerate(vars):
            if var in seq["values"]:
                vals = seq["values"][var]
                for t, v in enumerate(vals):
                    if not (isinstance(v, float) and math.isnan(v)):
                        values[i, t, j] = v

    return {
        "values": values,
        "mask": mask,
        "lengths": torch.tensor(lengths),
    }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class SequenceTransformer(nn.Module):
    """Transformer for variable-length sequences."""

    def __init__(
        self,
        n_vars: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, n_vars * 2)  # mean and log_std per var
        self.n_vars = n_vars

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_vars) input sequence
            mask: (batch, seq_len) attention mask, 1=attend, 0=ignore

        Returns:
            means: (batch, seq_len, n_vars)
            log_stds: (batch, seq_len, n_vars)
        """
        # Project to model dim
        h = self.input_proj(x)
        h = self.pos_enc(h)

        # Create attention mask (True = ignore)
        if mask is not None:
            attn_mask = (mask == 0)
        else:
            attn_mask = None

        # Transformer
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        # Output projection
        out = self.output_proj(h)
        means = out[..., :self.n_vars]
        log_stds = out[..., self.n_vars:].clamp(-5, 5)

        return means, log_stds


class SequenceSynthesizer:
    """Unified sequence model for imputation and evolution.

    Uses Transformer to attend to full history automatically.
    No manual lag specification needed.

    Attributes:
        continuous_vars: Continuous variables (Gaussian output)
        binary_vars: Binary variables (Bernoulli output)
        static_vars: Variables that don't change over time
    """

    def __init__(
        self,
        continuous_vars: list[str],
        binary_vars: list[str] | None = None,
        static_vars: list[str] | None = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        self.continuous_vars = continuous_vars
        self.binary_vars = binary_vars or []
        self.static_vars = static_vars or []
        self.all_vars = continuous_vars + self.binary_vars

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self._transformer = None
        self._is_fitted = False
        self._var_means = None
        self._var_stds = None

    def fit(
        self,
        data: pd.DataFrame,
        person_id_col: str = "person_id",
        period_col: str = "period",
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Self:
        """Fit model on panel data.

        Args:
            data: Panel DataFrame with person-period observations
            person_id_col: Column identifying individuals
            period_col: Column identifying time periods
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress

        Returns:
            self
        """
        # Prepare sequences
        sequences = prepare_sequences(
            data,
            vars=self.all_vars,
            person_id_col=person_id_col,
            period_col=period_col,
        )

        if len(sequences) == 0:
            raise ValueError("No sequences found in data")

        # Compute normalization stats
        all_values = {var: [] for var in self.all_vars}
        for seq in sequences:
            for var in self.all_vars:
                if var in seq["values"]:
                    for v in seq["values"][var]:
                        if not (isinstance(v, float) and math.isnan(v)):
                            all_values[var].append(v)

        self._var_means = {var: np.mean(vals) if vals else 0 for var, vals in all_values.items()}
        self._var_stds = {var: np.std(vals) + 1e-8 if vals else 1 for var, vals in all_values.items()}

        # Normalize sequences
        for seq in sequences:
            for var in self.all_vars:
                if var in seq["values"]:
                    seq["values"][var] = [
                        (v - self._var_means[var]) / self._var_stds[var]
                        if not (isinstance(v, float) and math.isnan(v)) else v
                        for v in seq["values"][var]
                    ]

        # Build model
        self._transformer = SequenceTransformer(
            n_vars=len(self.all_vars),
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        optimizer = torch.optim.Adam(self._transformer.parameters(), lr=lr)

        # Training loop
        n_batches = (len(sequences) + batch_size - 1) // batch_size

        for epoch in range(epochs):
            np.random.shuffle(sequences)
            total_loss = 0

            for batch_idx in range(n_batches):
                batch_seqs = sequences[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                if len(batch_seqs) == 0:
                    continue

                batch = collate_variable_length(batch_seqs, self.all_vars)
                x = batch["values"]
                mask = batch["mask"]

                # Shift for next-step prediction: predict t+1 from t
                # Input: all but last, Target: all but first
                if x.size(1) < 2:
                    continue

                x_input = x[:, :-1, :]
                x_target = x[:, 1:, :]
                mask_input = mask[:, :-1]
                mask_target = mask[:, 1:]

                optimizer.zero_grad()

                means, log_stds = self._transformer(x_input, mask_input)

                # Compute loss only on valid (non-padded) positions
                loss = 0
                n_valid = 0

                for i, var in enumerate(self.all_vars):
                    var_mask = mask_target * (1 - torch.isnan(x_target[:, :, i]).float())

                    if var in self.binary_vars:
                        # Binary cross-entropy
                        probs = torch.sigmoid(means[:, :, i])
                        bce = -x_target[:, :, i] * torch.log(probs + 1e-8) - (1 - x_target[:, :, i]) * torch.log(1 - probs + 1e-8)
                        loss += (bce * var_mask).sum()
                    else:
                        # Gaussian NLL
                        stds = torch.exp(log_stds[:, :, i])
                        nll = 0.5 * (((x_target[:, :, i] - means[:, :, i]) / stds) ** 2 + 2 * log_stds[:, :, i])
                        loss += (nll * var_mask).sum()

                    n_valid += var_mask.sum()

                if n_valid > 0:
                    loss = loss / n_valid
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / n_batches:.4f}")

        self._is_fitted = True
        return self

    def predict_next(
        self,
        history: pd.DataFrame,
        person_id_col: str = "person_id",
        period_col: str = "period",
    ) -> dict[str, float]:
        """Predict next period from history.

        Args:
            history: DataFrame with person's history

        Returns:
            Dict of var_name -> predicted value
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # Prepare sequence
        sequences = prepare_sequences(history, vars=self.all_vars, person_id_col=person_id_col, period_col=period_col)
        if len(sequences) == 0:
            raise ValueError("No sequence found")

        seq = sequences[0]

        # Normalize
        for var in self.all_vars:
            if var in seq["values"]:
                seq["values"][var] = [
                    (v - self._var_means[var]) / self._var_stds[var]
                    if not (isinstance(v, float) and math.isnan(v)) else 0
                    for v in seq["values"][var]
                ]

        # Collate single sequence
        batch = collate_variable_length([seq], self.all_vars)
        x = batch["values"]
        mask = batch["mask"]

        # Predict
        self._transformer.eval()
        with torch.no_grad():
            means, log_stds = self._transformer(x, mask)

        # Get last position prediction
        last_idx = int(batch["lengths"][0]) - 1
        pred_means = means[0, last_idx].numpy()
        pred_stds = torch.exp(log_stds[0, last_idx]).numpy()

        # Sample and denormalize
        result = {}
        for i, var in enumerate(self.all_vars):
            if var in self.binary_vars:
                prob = 1 / (1 + np.exp(-pred_means[i]))
                result[var] = int(np.random.random() < prob)
            else:
                sample = pred_means[i] + pred_stds[i] * np.random.randn()
                result[var] = sample * self._var_stds[var] + self._var_means[var]
                if var in self.continuous_vars:
                    result[var] = max(0, result[var])

        return result

    def impute(
        self,
        data: pd.DataFrame,
        target_vars: list[str],
        person_id_col: str = "person_id",
        period_col: str = "period",
    ) -> pd.DataFrame:
        """Impute missing values using history.

        Args:
            data: DataFrame with person's history (may have NaN)
            target_vars: Variables to impute

        Returns:
            DataFrame with imputed values
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        result = data.copy()

        # For each row with missing target vars, impute
        for idx in result.index:
            row = result.loc[idx]
            needs_impute = any(pd.isna(row.get(var)) for var in target_vars if var in row.index)

            if needs_impute:
                # Get history up to this point
                if person_id_col in result.columns:
                    person_id = row[person_id_col]
                    history = result[result[person_id_col] == person_id]
                    history = history[history[period_col] <= row[period_col]]
                else:
                    history = result.loc[:idx]

                # Predict
                pred = self.predict_next(history, person_id_col, period_col)

                # Fill missing
                for var in target_vars:
                    if var in result.columns and pd.isna(result.loc[idx, var]):
                        result.loc[idx, var] = pred.get(var, 0)

        return result

    def generate_trajectory(
        self,
        initial_state: pd.DataFrame,
        n_periods: int,
        person_id_col: str = "person_id",
        period_col: str = "period",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Generate trajectory from initial state.

        Args:
            initial_state: Starting state (single row or short history)
            n_periods: Number of periods to generate
            seed: Random seed

        Returns:
            DataFrame with full trajectory
        """
        if seed is not None:
            np.random.seed(seed)

        trajectory = [initial_state.copy()]
        current_history = initial_state.copy()

        # Get initial period as number for incrementing
        if period_col in initial_state.columns:
            last_period = initial_state[period_col].iloc[-1]
            try:
                last_period_num = int(last_period)
                period_is_numeric = True
            except (TypeError, ValueError):
                period_is_numeric = False
                last_period_num = 0
        else:
            period_is_numeric = False
            last_period_num = 0

        for t in range(n_periods):
            # Predict next
            pred = self.predict_next(current_history, person_id_col, period_col)

            # Create next row
            next_row = initial_state.iloc[[-1]].copy()
            for var in self.all_vars:
                if var in next_row.columns and var in pred:
                    next_row[var] = pred[var]

            # Update period
            if period_col in next_row.columns:
                if period_is_numeric:
                    next_row[period_col] = str(last_period_num + t + 1)
                else:
                    next_row[period_col] = f"t+{t+1}"

            trajectory.append(next_row)

            # Update history
            current_history = pd.concat([current_history, next_row], ignore_index=True)

        return pd.concat(trajectory, ignore_index=True)
