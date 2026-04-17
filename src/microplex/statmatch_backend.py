"""
Statistical matching backend using py-statmatch.

Provides a Synthesizer-compatible interface for traditional statistical
matching methods (NND hot deck, random hot deck, etc.) as an alternative
to the neural normalizing flow approach.

This is an optional backend - install with: pip install microplex[statmatch]
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd

# Check if py-statmatch is available
HAS_STATMATCH = importlib.util.find_spec("statmatch") is not None
if HAS_STATMATCH:
    from statmatch import nnd_hotdeck

if TYPE_CHECKING:
    from .synthesizer import Synthesizer


def _check_statmatch_installed():
    """Raise ImportError with helpful message if py-statmatch not installed."""
    if not HAS_STATMATCH:
        raise ImportError(
            "py-statmatch is required for statistical matching. "
            "Install with: pip install microplex[statmatch]"
        )


class StatMatchSynthesizer:
    """
    Statistical matching synthesizer using py-statmatch.

    Uses nearest neighbor distance (NND) hot deck matching to impute
    target variables from a donor dataset to recipients based on
    shared matching variables.

    This is a traditional statistical matching approach - simpler and
    more interpretable than neural methods, but may not preserve
    complex joint distributions as well.

    Example:
        >>> synth = StatMatchSynthesizer(
        ...     target_vars=["income", "expenditure"],
        ...     match_vars=["age", "education", "region"],
        ... )
        >>> synth.fit(donor_data)
        >>> synthetic = synth.generate(recipient_data)
    """

    def __init__(
        self,
        target_vars: list[str],
        match_vars: list[str],
        dist_fun: str = "euclidean",
        constrained: bool = False,
        k: int | None = None,
    ):
        """
        Initialize statistical matching synthesizer.

        Args:
            target_vars: Variables to impute from donor to recipient
            match_vars: Variables to match on (must exist in both datasets)
            dist_fun: Distance function ("euclidean", "manhattan", "mahalanobis")
            constrained: If True, limit how many times each donor can be used
            k: Max times each donor can be used (only if constrained=True)
        """
        _check_statmatch_installed()

        self.target_vars = target_vars
        self.match_vars = match_vars
        self.dist_fun = dist_fun
        self.constrained = constrained
        self.k = k

        self._donor_data: pd.DataFrame | None = None
        self.is_fitted_ = False

    def fit(
        self,
        data: pd.DataFrame,
        weight_col: str | None = None,
        **kwargs,  # Accept but ignore neural-specific params
    ) -> Self:
        """
        Fit the synthesizer by storing donor data.

        For statistical matching, "fitting" just means storing the
        donor dataset that will be matched against recipients.

        Args:
            data: Donor dataset with target and match variables
            weight_col: Optional weight column (used in weighted matching)
            **kwargs: Ignored (for API compatibility with neural Synthesizer)

        Returns:
            self
        """
        # Validate columns exist
        required_cols = self.target_vars + self.match_vars
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns in donor data: {missing}")

        # Store donor data
        self._donor_data = data.copy()
        self._weight_col = weight_col
        self.is_fitted_ = True

        return self

    def generate(
        self,
        conditions: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic target variables via statistical matching.

        Finds nearest neighbor donors for each recipient record and
        imputes target variable values from matched donors.

        Args:
            conditions: Recipient data with match variables
            seed: Random seed for reproducibility (for tie-breaking)

        Returns:
            DataFrame with conditions + imputed target variables
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)

        # Validate match variables exist in recipient data
        missing = [c for c in self.match_vars if c not in conditions.columns]
        if missing:
            raise ValueError(f"Missing match variables in conditions: {missing}")

        # Get weights if available
        don_weights = None
        if self._weight_col and self._weight_col in self._donor_data.columns:
            don_weights = self._donor_data[self._weight_col].values

        # Run NND hot deck matching
        result = nnd_hotdeck(
            data_rec=conditions,
            data_don=self._donor_data,
            match_vars=self.match_vars,
            dist_fun=self.dist_fun,
            k=self.k if self.constrained else None,
            don_weights=don_weights,
        )

        # Get donor indices
        donor_indices = result["noad.index"]

        # Create output DataFrame
        output = conditions.copy()

        # Impute target variables from matched donors
        for var in self.target_vars:
            output[var] = self._donor_data[var].iloc[donor_indices].values

        return output

    def sample(
        self,
        n: int,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate fully synthetic records.

        Samples recipient conditions from donor data, then matches
        back to donors (effectively resampling with perturbation).

        Args:
            n: Number of synthetic records to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with all variables
        """
        if not self.is_fitted_:
            raise ValueError("Synthesizer not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)

        # Sample conditions from donor data
        sampled_idx = np.random.choice(
            len(self._donor_data), size=n, replace=True
        )
        conditions = self._donor_data[self.match_vars].iloc[sampled_idx].reset_index(drop=True)

        # Generate via matching (this adds some noise vs direct copying)
        return self.generate(conditions, seed=seed)


def create_synthesizer(
    method: str = "neural",
    target_vars: list[str] | None = None,
    condition_vars: list[str] | None = None,
    match_vars: list[str] | None = None,
    **kwargs,
) -> StatMatchSynthesizer | Synthesizer:
    """
    Factory function to create a synthesizer with the specified method.

    Args:
        method: "neural" (default) or "statistical_matching"
        target_vars: Variables to synthesize
        condition_vars: Variables to condition on (neural method)
        match_vars: Variables to match on (statistical matching method)
        **kwargs: Additional arguments passed to the synthesizer

    Returns:
        Synthesizer instance (either neural or statistical matching)

    Example:
        >>> # Neural synthesis (default)
        >>> synth = create_synthesizer(
        ...     method="neural",
        ...     target_vars=["income"],
        ...     condition_vars=["age", "education"],
        ... )
        >>>
        >>> # Statistical matching
        >>> synth = create_synthesizer(
        ...     method="statistical_matching",
        ...     target_vars=["income"],
        ...     match_vars=["age", "education"],
        ... )
    """
    if method == "neural":
        from .synthesizer import Synthesizer
        return Synthesizer(
            target_vars=target_vars,
            condition_vars=condition_vars or [],
            **kwargs,
        )
    elif method in ("statistical_matching", "statmatch", "nnd_hotdeck"):
        return StatMatchSynthesizer(
            target_vars=target_vars,
            match_vars=match_vars or condition_vars or [],
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Options: 'neural', 'statistical_matching'"
        )
