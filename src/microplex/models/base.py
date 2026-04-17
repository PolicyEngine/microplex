"""
Base interface for synthesis models in popdgp.

All models must implement:
- fit(): Train on survey data
- generate(): Unconditional synthesis
- impute(): Conditional generation (given partial obs, sample the rest)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd


@dataclass
class SyntheticPopulation:
    """
    Output of synthesis: population with optional graph structure.

    This is what microplex ultimately produces.
    Calibration module can take this and add weights.
    """
    persons: pd.DataFrame          # person-period records
    households: pd.DataFrame | None = None  # household-period records
    edges: pd.DataFrame | None = None       # hh-person-period edges
    weights: np.ndarray | None = None       # sample weights (after calibration)

    @property
    def n_persons(self) -> int:
        return len(self.persons['person_id'].unique()) if 'person_id' in self.persons.columns else len(self.persons)

    @property
    def n_households(self) -> int:
        if self.households is not None:
            return len(self.households['household_id'].unique())
        return 0

    @property
    def n_periods(self) -> int:
        if 'period' in self.persons.columns:
            return self.persons['period'].nunique()
        return 1

    def to_cross_section(self, period: int = 0) -> pd.DataFrame:
        """Extract single period as cross-section."""
        if 'period' in self.persons.columns:
            return self.persons[self.persons['period'] == period].copy()
        return self.persons.copy()


@dataclass
class ImputationResult:
    """
    Output of conditional generation / imputation.

    Contains multiple samples per input row to represent uncertainty.
    """
    samples: pd.DataFrame    # n_input_rows * n_samples rows
    input_mask: pd.DataFrame # which columns were observed (input)
    n_samples: int

    def mean(self) -> pd.DataFrame:
        """Point estimate (mean across samples)."""
        return self.samples.groupby('_input_row_id').mean()

    def quantile(self, q: float) -> pd.DataFrame:
        """Get quantile across samples."""
        return self.samples.groupby('_input_row_id').quantile(q)

    def std(self) -> pd.DataFrame:
        """Uncertainty (std across samples)."""
        return self.samples.groupby('_input_row_id').std()


class BaseSynthesisModel(ABC):
    """
    Abstract base class for synthesis models.

    All models (MAF, VAE, Transformer, etc.) implement this interface.
    """

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame | None = None,
        **kwargs
    ) -> Self:
        """
        Fit the model to training data.

        Args:
            data: Training data (can be stacked from multiple surveys)
            mask: Boolean mask, True = observed. If None, all observed.
            **kwargs: Model-specific arguments

        Returns:
            self (for chaining)
        """
        pass

    @abstractmethod
    def generate(
        self,
        n: int,
        **kwargs
    ) -> SyntheticPopulation:
        """
        Unconditional generation: sample n complete records.

        This is what produces The Microplex.

        Args:
            n: Number of records to generate
            **kwargs: Model-specific arguments (e.g., T for trajectories)

        Returns:
            SyntheticPopulation with complete records
        """
        pass

    @abstractmethod
    def impute(
        self,
        partial_obs: pd.DataFrame,
        n_samples: int = 100,
        **kwargs
    ) -> ImputationResult:
        """
        Conditional generation: given partial observations, sample the rest.

        This is the microplex API service.

        Args:
            partial_obs: DataFrame with some columns filled, others NaN
            n_samples: Number of samples per input row (for uncertainty)
            **kwargs: Model-specific arguments

        Returns:
            ImputationResult with samples for each input row
        """
        pass

    @abstractmethod
    def log_prob(
        self,
        data: pd.DataFrame,
        mask: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        Compute log probability of data under the model.

        Useful for evaluation and anomaly detection.

        Args:
            data: Data to score
            mask: Which values are observed (if partial)

        Returns:
            Log probability for each row
        """
        pass

    def save(self, path: str) -> None:
        """Save model to disk."""
        import torch
        if hasattr(self, 'state_dict'):
            torch.save(self.state_dict(), path)
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load model from disk."""
        import torch
        model = cls()
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(torch.load(path))
        else:
            import pickle
            with open(path, 'rb') as f:
                model = pickle.load(f)
        return model


class BaseTrajectoryModel(BaseSynthesisModel):
    """
    Extended interface for models that handle trajectories.

    Adds methods for temporal generation and household structure.
    """

    @abstractmethod
    def generate_trajectories(
        self,
        n: int,
        T: int,
        **kwargs
    ) -> SyntheticPopulation:
        """
        Generate n trajectories of length T.

        Args:
            n: Number of individuals
            T: Number of time periods

        Returns:
            SyntheticPopulation with person-period records
        """
        pass

    def generate(self, n: int, T: int = 1, **kwargs) -> SyntheticPopulation:
        """Default generate uses generate_trajectories."""
        return self.generate_trajectories(n, T, **kwargs)


class BaseGraphModel(BaseTrajectoryModel):
    """
    Extended interface for models with household-person graph structure.

    Adds methods for graph-aware generation and event sampling.
    """

    @abstractmethod
    def generate_population(
        self,
        n_households: int,
        T: int,
        **kwargs
    ) -> SyntheticPopulation:
        """
        Generate population with household structure.

        Args:
            n_households: Number of households
            T: Number of time periods

        Returns:
            SyntheticPopulation with households, persons, and edges
        """
        pass

    @abstractmethod
    def sample_events(
        self,
        population: SyntheticPopulation,
        period: int,
    ) -> dict[str, Any]:
        """
        Sample life events for next period.

        Args:
            population: Current population state
            period: Current time period

        Returns:
            Dict of events {person_id: event_type}
        """
        pass
