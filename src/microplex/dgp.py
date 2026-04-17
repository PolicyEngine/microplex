"""Population Data Generating Process (DGP) learning from multiple partial surveys.

The core idea:
- Multiple surveys each observe different columns of the same underlying population
- We learn a unified generative model P(all_columns) from these partial views
- Then sample from this model to create synthetic populations

This is NOT:
- Statistical matching (overfits by pasting columns from similar records)
- Imputation (fills in missing for existing records)

This IS:
- Learning the true joint distribution from incomplete observations
- Generating entirely new records that could plausibly exist in the population
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    RandomForestQuantileRegressor = None


@dataclass
class Survey:
    """A survey with partial observations of the population."""

    name: str
    data: pd.DataFrame
    columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.columns:
            self.columns = list(self.data.columns)


@dataclass
class EvalResult:
    """Evaluation results against a holdout survey."""

    survey_name: str
    coverage: float      # Fraction of holdout with nearby synthetic
    precision: float     # Fraction of synthetic within holdout manifold
    recall: float        # Fraction of holdout within synthetic manifold
    density: float
    n_holdout: int
    n_synthetic: int
    columns_evaluated: List[str]


def compute_prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Compute Precision, Recall, Density, Coverage via canonical prdc library.

    Delegates to Naeem et al. (2020) reference implementation. Standardizes
    inputs first for consistent distance computation.
    """
    from prdc import compute_prdc as _prdc
    from sklearn.preprocessing import StandardScaler

    if len(real) < k + 1 or len(fake) < k + 1:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

    scaler = StandardScaler()
    real_s = scaler.fit_transform(real)
    fake_s = scaler.transform(fake)

    metrics = _prdc(real_s, fake_s, nearest_k=k)

    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "density": float(metrics["density"]),
        "coverage": float(metrics["coverage"]),
    }


class PopulationDGP:
    """Learn data generating process from multiple partial surveys.

    Each survey observes different columns of the same underlying population.
    We learn a generative model that can sample complete records.

    Example:
        >>> dgp = PopulationDGP()
        >>> dgp.fit(
        ...     surveys=[
        ...         Survey("CPS", cps_df),      # Has income, demographics
        ...         Survey("SCF", scf_df),      # Has wealth, assets
        ...         Survey("PUF", puf_df),      # Has taxes, deductions
        ...     ],
        ...     shared_cols=["age", "income", "filing_status"],
        ... )
        >>> synthetic = dgp.generate(n=1_000_000)
        >>> dgp.evaluate(holdouts={"CPS": cps_holdout, "SCF": scf_holdout})
    """

    name = "PopulationDGP"  # Can be overridden for display

    def __init__(
        self,
        n_estimators: int = 100,
        zero_inflation_threshold: float = 0.1,
        quantiles: List[float] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.zero_inflation_threshold = zero_inflation_threshold
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self.random_state = random_state

        # Learned models
        self.shared_cols_: List[str] = []
        self.all_cols_: List[str] = []
        self.col_to_survey_: Dict[str, str] = {}  # Which survey teaches each column
        self.models_: Dict[str, object] = {}  # column -> model
        self.is_zero_inflated_: Dict[str, bool] = {}
        self.zero_classifiers_: Dict[str, object] = {}
        self.col_stats_: Dict[str, Dict] = {}  # For standardization

        # Training data reference (for shared column sampling)
        self.shared_data_: Optional[pd.DataFrame] = None

    def fit(
        self,
        surveys: List[Survey],
        shared_cols: List[str],
    ) -> "PopulationDGP":
        """Learn generative model from multiple partial surveys.

        Args:
            surveys: List of Survey objects, each with different columns
            shared_cols: Columns that appear in all (or most) surveys

        Returns:
            self
        """
        if RandomForestQuantileRegressor is None:
            raise ImportError("quantile-forest required: pip install quantile-forest")

        self.shared_cols_ = list(shared_cols)
        rng = np.random.RandomState(self.random_state)

        # Collect all columns and figure out which survey teaches each
        all_cols = set(shared_cols)
        for survey in surveys:
            for col in survey.columns:
                if col not in all_cols:
                    all_cols.add(col)
                    self.col_to_survey_[col] = survey.name

        self.all_cols_ = list(all_cols)

        # Pool shared columns from all surveys for sampling base
        shared_dfs = []
        for survey in surveys:
            available_shared = [c for c in shared_cols if c in survey.data.columns]
            if len(available_shared) == len(shared_cols):
                shared_dfs.append(survey.data[shared_cols].copy())

        if shared_dfs:
            self.shared_data_ = pd.concat(shared_dfs, ignore_index=True)
        else:
            # Use first survey's shared columns
            self.shared_data_ = surveys[0].data[shared_cols].copy()

        # Learn model for each non-shared column
        survey_lookup = {s.name: s for s in surveys}

        for col in self.all_cols_:
            if col in shared_cols:
                continue

            survey_name = self.col_to_survey_[col]
            survey = survey_lookup[survey_name]

            # Get training data
            available_shared = [c for c in shared_cols if c in survey.data.columns]
            X = survey.data[available_shared].values
            y = survey.data[col].values

            # Check for zero-inflation
            min_val = y.min()
            at_min = np.isclose(y, min_val, atol=1e-6)
            zero_frac = at_min.sum() / len(y)

            self.is_zero_inflated_[col] = zero_frac >= self.zero_inflation_threshold
            self.col_stats_[col] = {"min": min_val, "zero_frac": zero_frac}

            if self.is_zero_inflated_[col] and at_min.sum() >= 10:
                # Two-stage model
                # Stage 1: Classifier for zero vs non-zero
                clf = RandomForestClassifier(
                    n_estimators=50,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                clf.fit(X, (~at_min).astype(int))
                self.zero_classifiers_[col] = clf

                # Stage 2: QRF on non-zero values
                if (~at_min).sum() >= 10:
                    qrf = RandomForestQuantileRegressor(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        n_jobs=-1,
                    )
                    qrf.fit(X[~at_min], y[~at_min])
                    self.models_[col] = qrf
            else:
                # Standard QRF
                qrf = RandomForestQuantileRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                qrf.fit(X, y)
                self.models_[col] = qrf

        return self

    def generate(
        self,
        n: int,
        noise_scale: float = 0.1,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate n synthetic records from learned DGP.

        Args:
            n: Number of records to generate
            noise_scale: Noise added to shared variables (breaks exact matches)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with all columns
        """
        rng = np.random.RandomState(seed or self.random_state)

        # Sample shared variables (bootstrap + noise)
        sample_idx = rng.choice(len(self.shared_data_), size=n, replace=True)
        shared_values = self.shared_data_.iloc[sample_idx].values.copy()

        if noise_scale > 0:
            shared_values += rng.normal(0, noise_scale, shared_values.shape)

        synthetic = pd.DataFrame(shared_values, columns=self.shared_cols_)

        # Generate each non-shared column
        for col in self.all_cols_:
            if col in self.shared_cols_:
                continue

            X = shared_values

            if self.is_zero_inflated_.get(col, False):
                # Two-stage generation
                results = np.full(n, self.col_stats_[col]["min"])

                if col in self.zero_classifiers_:
                    # Probabilistic zero/non-zero
                    clf = self.zero_classifiers_[col]
                    proba = clf.predict_proba(X)
                    if proba.shape[1] == 1:
                        # Single class: all zero or all non-zero
                        only_class = clf.classes_[0]
                        probs = np.full(n, float(only_class))
                    else:
                        probs = proba[:, 1]
                    is_nonzero = rng.random(n) < probs

                    if col in self.models_ and is_nonzero.sum() > 0:
                        qrf = self.models_[col]
                        preds = qrf.predict(X[is_nonzero], quantiles=self.quantiles)
                        q_choices = rng.choice(len(self.quantiles), size=is_nonzero.sum())
                        results[is_nonzero] = preds[np.arange(is_nonzero.sum()), q_choices]

                synthetic[col] = results
            else:
                # Standard QRF with quantile sampling
                if col in self.models_:
                    qrf = self.models_[col]
                    preds = qrf.predict(X, quantiles=self.quantiles)
                    q_choices = rng.choice(len(self.quantiles), size=n)
                    synthetic[col] = preds[np.arange(n), q_choices]

        return synthetic

    def evaluate(
        self,
        holdouts: Dict[str, pd.DataFrame],
        n_synthetic: Optional[int] = None,
        k: int = 5,
    ) -> Dict[str, EvalResult]:
        """Evaluate synthetic data against holdout surveys.

        For each holdout, we evaluate coverage on the columns that survey observes.

        Args:
            holdouts: Dict mapping survey name to holdout DataFrame
            n_synthetic: Number of synthetic records to generate (default: sum of holdouts)
            k: Number of neighbors for PRDC metrics

        Returns:
            Dict mapping survey name to EvalResult
        """
        if n_synthetic is None:
            n_synthetic = sum(len(h) for h in holdouts.values())

        synthetic = self.generate(n=n_synthetic)
        results = {}

        for name, holdout in holdouts.items():
            # Evaluate on columns present in this holdout
            eval_cols = [c for c in holdout.columns if c in synthetic.columns]

            if len(eval_cols) < 2:
                continue

            holdout_vals = holdout[eval_cols].values
            synth_vals = synthetic[eval_cols].values

            # Handle any NaN
            holdout_mask = ~np.isnan(holdout_vals).any(axis=1)
            synth_mask = ~np.isnan(synth_vals).any(axis=1)

            prdc = compute_prdc(
                holdout_vals[holdout_mask],
                synth_vals[synth_mask],
                k=k,
            )

            results[name] = EvalResult(
                survey_name=name,
                coverage=prdc["coverage"],
                precision=prdc["precision"],
                recall=prdc["recall"],
                density=prdc["density"],
                n_holdout=holdout_mask.sum(),
                n_synthetic=synth_mask.sum(),
                columns_evaluated=eval_cols,
            )

        return results

    def summary(self, eval_results: Dict[str, EvalResult]) -> str:
        """Pretty-print evaluation results."""
        lines = [
            "Population DGP Evaluation",
            "=" * 60,
            f"{'Survey':<15} {'Coverage':>10} {'Precision':>10} {'Recall':>10} {'Cols':>8}",
            "-" * 60,
        ]

        avg_coverage = 0
        for name, result in eval_results.items():
            lines.append(
                f"{name:<15} {result.coverage:>10.1%} {result.precision:>10.1%} "
                f"{result.recall:>10.1%} {len(result.columns_evaluated):>8}"
            )
            avg_coverage += result.coverage

        if eval_results:
            avg_coverage /= len(eval_results)
            lines.append("-" * 60)
            lines.append(f"{'Average':<15} {avg_coverage:>10.1%}")

        lines.append("=" * 60)
        return "\n".join(lines)


def run_multi_source_benchmark(
    surveys: List[Survey],
    shared_cols: List[str],
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[PopulationDGP, Dict[str, EvalResult]]:
    """Run full benchmark: train DGP, evaluate on holdouts from each survey.

    Args:
        surveys: List of Survey objects
        shared_cols: Columns shared across surveys
        holdout_frac: Fraction of each survey to hold out
        seed: Random seed

    Returns:
        Tuple of (trained DGP, evaluation results)
    """
    rng = np.random.RandomState(seed)

    # Split each survey into train/holdout
    train_surveys = []
    holdouts = {}

    for survey in surveys:
        n = len(survey.data)
        n_holdout = int(n * holdout_frac)
        indices = rng.permutation(n)

        train_data = survey.data.iloc[indices[n_holdout:]].reset_index(drop=True)
        holdout_data = survey.data.iloc[indices[:n_holdout]].reset_index(drop=True)

        train_surveys.append(Survey(survey.name, train_data, survey.columns))
        holdouts[survey.name] = holdout_data

    # Train DGP
    dgp = PopulationDGP(random_state=seed)
    dgp.fit(train_surveys, shared_cols)

    # Evaluate
    results = dgp.evaluate(holdouts)

    print(dgp.summary(results))

    return dgp, results
