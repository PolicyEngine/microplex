"""Multi-survey fusion pipeline.

Provides a high-level API for synthesizing complete microdata
from multiple surveys with complementary coverage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from .harmonize import (
    COMMON_SCHEMA,
    apply_inverse_transform,
    apply_transform,
    harmonize_surveys,
    stack_surveys,
)
from .masked_maf import MaskedMAF


@dataclass
class FusionConfig:
    """Configuration for multi-survey fusion synthesis."""

    # Model architecture
    n_layers: int = 6
    hidden_dim: int = 128

    # Training
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Sampling - use 2.0 to avoid extreme outliers
    clip_z: float = 2.0

    # Device
    device: str = "cpu"


@dataclass
class FusionResult:
    """Result of fusion synthesis."""

    synthetic: pd.DataFrame
    model: MaskedMAF
    config: FusionConfig
    training_time: float
    variable_names: list[str]
    observation_rates: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path):
        """Save synthetic data and model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.synthetic.to_parquet(path / "synthetic.parquet")
        self.model.save(str(path / "model"))

        # Save metadata
        import json
        meta = {
            "n_records": len(self.synthetic),
            "variables": self.variable_names,
            "observation_rates": self.observation_rates,
            "training_time": self.training_time,
            "config": {
                "n_layers": self.config.n_layers,
                "hidden_dim": self.config.hidden_dim,
                "epochs": self.config.epochs,
            },
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


class FusionSynthesizer:
    """Synthesizes complete microdata from multiple partial surveys.

    Uses masked autoregressive flows to learn the joint distribution
    from stacked survey data where each survey has different observed
    variables.

    Example:
        >>> from microplex.fusion import FusionSynthesizer
        >>> synth = FusionSynthesizer()
        >>> synth.add_survey("cps", cps_data)
        >>> synth.add_survey("puf", puf_data)
        >>> result = synth.fit_generate(n_samples=200_000)
        >>> synthetic = result.synthetic
    """

    def __init__(self, config: FusionConfig | None = None):
        """Initialize synthesizer.

        Args:
            config: Fusion configuration. Uses defaults if not provided.
        """
        self.config = config or FusionConfig()
        self.surveys: dict[str, pd.DataFrame] = {}
        self.model: MaskedMAF | None = None
        self._harmonized: dict[str, pd.DataFrame] | None = None
        self._stacked: pd.DataFrame | None = None
        self._mask: np.ndarray | None = None
        self._variable_names: list[str] | None = None
        self._active_var_names: list[str] | None = None
        self._active_var_indices: list[int] | None = None

    def add_survey(self, name: str, data: pd.DataFrame) -> Self:
        """Add a survey to the fusion.

        Args:
            name: Survey identifier ("cps" or "puf")
            data: Survey data with columns matching expected schema

        Returns:
            self for method chaining
        """
        self.surveys[name] = data
        # Reset cached data
        self._harmonized = None
        self._stacked = None
        self._mask = None
        self._active_var_names = None
        self._active_var_indices = None
        return self

    def harmonize(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Harmonize and stack surveys.

        Returns:
            stacked: Stacked DataFrame with all surveys
            mask: Boolean observation mask [n_records, n_variables]
        """
        if not self.surveys:
            raise ValueError("No surveys added. Use add_survey() first.")

        self._harmonized = harmonize_surveys(self.surveys)
        self._stacked, self._mask = stack_surveys(
            self._harmonized, normalize_weights=True
        )
        self._variable_names = list(COMMON_SCHEMA.keys())

        return self._stacked, self._mask

    def fit(self, verbose: bool = True) -> MaskedMAF:
        """Fit the masked MAF on harmonized surveys.

        Args:
            verbose: Print training progress

        Returns:
            Fitted MaskedMAF model
        """
        if self._stacked is None:
            self.harmonize()

        # Prepare training data (filters to active variables)
        X, mask = self._prepare_training_data()

        # Get sample weights
        weights = None
        if "weight" in self._stacked.columns:
            weights = self._stacked["weight"].values

        # Create and fit model on ACTIVE variables only
        self.model = MaskedMAF(
            n_features=len(self._active_var_names),
            n_layers=self.config.n_layers,
            hidden_dim=self.config.hidden_dim,
        )

        self.model.fit(
            X=X,
            mask=mask,
            sample_weights=weights,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            device=self.config.device,
            verbose=verbose,
        )

        return self.model

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic population.

        Args:
            n_samples: Number of records to generate

        Returns:
            DataFrame with complete synthetic population
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Generate samples for ACTIVE variables only
        samples = self.model.sample(
            n_samples=n_samples,
            clip_z=self.config.clip_z,
            device=self.config.device,
        )

        # Convert to DataFrame using active variable names
        result = pd.DataFrame(samples, columns=self._active_var_names)

        # Apply inverse transforms and clipping for active variables
        for var in self._active_var_names:
            spec = COMMON_SCHEMA.get(var, {"transform": "none"})
            transform = spec.get("transform", "none")

            # For log-transformed vars, clip in log space to prevent exp overflow
            # exp(14) ≈ 1.2M which is reasonable for max income
            if transform == "log1p":
                result[var] = result[var].clip(lower=0, upper=14)
            elif transform == "signed_log":
                result[var] = result[var].clip(lower=-14, upper=14)

            # Apply inverse transform
            if spec["type"] != "binary":
                result[var] = apply_inverse_transform(result[var].values, transform)

            # Clip to schema-defined ranges
            if "min" in spec:
                result[var] = result[var].clip(lower=spec["min"])
            if "max" in spec:
                result[var] = result[var].clip(upper=spec["max"])

            # Round discrete variables
            if spec["type"] == "discrete":
                result[var] = result[var].round().astype(int)
            elif spec["type"] == "binary":
                result[var] = (result[var] > 0.5).astype(int)

        # Add inactive variables as 0
        for var in self._variable_names:
            if var not in self._active_var_names:
                result[var] = 0.0

        # Add uniform weights (to be calibrated later)
        result["weight"] = 1.0

        return result

    def fit_generate(
        self,
        n_samples: int,
        verbose: bool = True,
    ) -> FusionResult:
        """Fit model and generate synthetic population.

        Convenience method combining fit() and generate().

        Args:
            n_samples: Number of synthetic records
            verbose: Print progress

        Returns:
            FusionResult with synthetic data and model
        """
        start_time = time.time()

        if verbose:
            print("Harmonizing surveys...")
        self.harmonize()

        if verbose:
            print(f"\nTraining MaskedMAF ({self.config.n_layers} layers)...")
        self.fit(verbose=verbose)

        if verbose:
            print(f"\nGenerating {n_samples:,} synthetic records...")
        synthetic = self.generate(n_samples)

        training_time = time.time() - start_time

        # Compute observation rates
        obs_rates = {}
        for i, var in enumerate(self._variable_names):
            obs_rates[var] = float(self._mask[:, i].mean())

        return FusionResult(
            synthetic=synthetic,
            model=self.model,
            config=self.config,
            training_time=training_time,
            variable_names=self._variable_names,
            observation_rates=obs_rates,
        )

    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare normalized training arrays.

        Only includes variables with at least some observations (>0% observed).
        Variables with 0% observation rate cannot be learned from and are excluded.
        """
        n_records = len(self._stacked)

        # Filter to variables with at least some observations
        obs_rates = self._mask.mean(axis=0)
        self._active_var_indices = [i for i in range(len(self._variable_names)) if obs_rates[i] > 0]
        self._active_var_names = [self._variable_names[i] for i in self._active_var_indices]

        n_features = len(self._active_var_names)
        print(f"\nActive variables for training: {n_features}/{len(self._variable_names)}")
        print(f"  Excluded (0% observed): {[self._variable_names[i] for i in range(len(self._variable_names)) if obs_rates[i] == 0][:5]}...")

        X = np.zeros((n_records, n_features), dtype=np.float32)
        active_mask = np.zeros((n_records, n_features), dtype=bool)

        for j, i in enumerate(self._active_var_indices):
            var = self._variable_names[i]
            values = self._stacked[var].values.copy()
            observed = self._mask[:, i]

            # Replace NaN with 0 before transform
            values = np.where(observed, values, 0)

            # Apply transform
            spec = COMMON_SCHEMA.get(var, {"transform": "none"})
            if spec["type"] != "binary":
                values = apply_transform(values, spec.get("transform", "none"))

            X[:, j] = values
            active_mask[:, j] = observed

        # Clean up any remaining NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, active_mask


def load_cps_for_fusion(year: int = 2023) -> pd.DataFrame:
    """Load CPS ASEC ready for fusion.

    Args:
        year: CPS ASEC year

    Returns:
        DataFrame with harmonization-ready columns
    """
    from ..data_sources.cps import load_cps_asec

    dataset = load_cps_asec(year=year, download=True)

    # Convert to pandas
    persons = dataset.persons.to_pandas()
    households = dataset.households.to_pandas()

    # Merge
    df = persons.merge(
        households[["household_id", "state_fips", "household_weight"]],
        on="household_id",
        how="left",
    )

    # Rename to match schema
    rename_map = {
        "wage_income": "employment_income",
        "self_employment_income": "self_employment_income",
        "interest_income": "interest_income",
        "dividend_income": "dividend_income",
        "rental_income": "rental_income",
        "social_security": "social_security",
        "unemployment_compensation": "unemployment_compensation",
        "weight": "person_weight",
    }
    df = df.rename(columns=rename_map)

    # Derived columns
    if "sex" in df.columns:
        df["is_male"] = (df["sex"] == 1).astype(float)

    if "marital_status" in df.columns:
        df["is_married"] = df["marital_status"].isin([1, 2]).astype(float)

    df["weight"] = df["person_weight"]

    return df


def load_puf_for_fusion(target_year: int = 2024) -> pd.DataFrame | None:
    """Load IRS PUF ready for fusion.

    Args:
        target_year: Year to uprate to

    Returns:
        DataFrame with harmonization-ready columns, or None if unavailable
    """
    try:
        from ..data_sources.puf import load_puf
        return load_puf(target_year=target_year, expand_persons=True)
    except Exception as e:
        print(f"Warning: Could not load PUF: {e}")
        return None


def synthesize_from_surveys(
    n_samples: int = 200_000,
    cps_year: int = 2023,
    puf_target_year: int = 2024,
    config: FusionConfig | None = None,
    include_puf: bool = True,
    verbose: bool = True,
) -> FusionResult:
    """Synthesize complete population from CPS + PUF.

    High-level convenience function for the full pipeline.

    Args:
        n_samples: Number of synthetic records
        cps_year: CPS ASEC year
        puf_target_year: Year to uprate PUF to
        config: Model configuration
        include_puf: Whether to include PUF (requires HuggingFace access)
        verbose: Print progress

    Returns:
        FusionResult with synthetic population

    Example:
        >>> from microplex.fusion import synthesize_from_surveys, FusionConfig
        >>> config = FusionConfig(device="mps", epochs=50)
        >>> result = synthesize_from_surveys(n_samples=100_000, config=config)
        >>> print(f"Generated {len(result.synthetic):,} records")
    """
    if verbose:
        print("=" * 60)
        print("MULTI-SURVEY FUSION SYNTHESIS")
        print("=" * 60)

    synth = FusionSynthesizer(config=config)

    # Load CPS
    if verbose:
        print(f"\nLoading CPS ASEC {cps_year}...")
    cps = load_cps_for_fusion(year=cps_year)
    synth.add_survey("cps", cps)
    if verbose:
        print(f"  {len(cps):,} records")

    # Load PUF if requested
    if include_puf:
        if verbose:
            print(f"\nLoading IRS PUF (uprated to {puf_target_year})...")
        puf = load_puf_for_fusion(target_year=puf_target_year)
        if puf is not None:
            synth.add_survey("puf", puf)
            if verbose:
                print(f"  {len(puf):,} records")

    # Run pipeline
    result = synth.fit_generate(n_samples=n_samples, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print(f"COMPLETE: {len(result.synthetic):,} synthetic records")
        print(f"Training time: {result.training_time:.1f}s")
        print("=" * 60)

    return result
