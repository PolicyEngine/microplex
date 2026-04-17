"""Reweighting method comparison benchmark.

Compares multiple calibration/reweighting approaches on target-matching accuracy:
- IPF (Iterative Proportional Fitting / raking)
- Chi-square distance minimization
- Entropy balancing (KL divergence)
- L1 sparse (linear programming)
- L2 sparse (quadratic programming)
- L0 sparse (iterative reweighted L1)
- SparseCalibrator (FISTA-based cross-category selection)
- HardConcreteCalibrator (differentiable L0 via Hard Concrete distribution)

Evaluation metrics:
- Mean/max relative error against targets
- Weight coefficient of variation
- Sparsity (fraction of zero weights)
- Elapsed time
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# --- Protocol ---


@runtime_checkable
class ReweightingMethod(Protocol):
    """Protocol for reweighting methods in the benchmark."""

    name: str

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
    ) -> "ReweightingMethod": ...

    def get_weights(self) -> np.ndarray: ...


# --- Result dataclasses ---


@dataclass
class TargetError:
    """Error for one target constraint."""

    target_name: str
    target_value: float
    actual_value: float
    relative_error: float  # as fraction (0.05 = 5%)

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "target_value": float(self.target_value),
            "actual_value": float(self.actual_value),
            "relative_error": float(self.relative_error),
        }


@dataclass
class ReweightingMethodResult:
    """Results for one reweighting method."""

    method_name: str
    mean_relative_error: float
    max_relative_error: float
    weight_cv: float
    sparsity: float
    elapsed_seconds: float
    per_target_errors: list[TargetError] = field(default_factory=list)

    @classmethod
    def from_evaluation(
        cls,
        method_name: str,
        data: pd.DataFrame,
        weights: np.ndarray,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> "ReweightingMethodResult":
        """Compute evaluation metrics from weights and targets."""
        errors = []

        # Evaluate marginal targets
        for var, var_targets in marginal_targets.items():
            for cat, target in var_targets.items():
                mask = data[var] == cat
                actual = float(weights[mask].sum())
                rel_err = abs(actual - target) / target if target > 0 else 0.0
                errors.append(TargetError(
                    target_name=f"{var}={cat}",
                    target_value=target,
                    actual_value=actual,
                    relative_error=rel_err,
                ))

        # Evaluate continuous targets
        if continuous_targets:
            for var, target in continuous_targets.items():
                if var in data.columns:
                    actual = float((weights * data[var].values).sum())
                    rel_err = abs(actual - target) / abs(target) if target != 0 else 0.0
                    errors.append(TargetError(
                        target_name=var,
                        target_value=target,
                        actual_value=actual,
                        relative_error=rel_err,
                    ))

        rel_errors = [e.relative_error for e in errors]
        mean_err = float(np.mean(rel_errors)) if rel_errors else 0.0
        max_err = float(max(rel_errors)) if rel_errors else 0.0

        # Weight statistics
        mean_w = weights.mean()
        cv = float(weights.std() / mean_w) if mean_w > 0 else 0.0
        sparsity = float((weights < 1e-9).sum() / len(weights))

        return cls(
            method_name=method_name,
            mean_relative_error=mean_err,
            max_relative_error=max_err,
            weight_cv=cv,
            sparsity=sparsity,
            elapsed_seconds=elapsed_seconds,
            per_target_errors=errors,
        )

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "mean_relative_error": round(self.mean_relative_error, 6),
            "max_relative_error": round(self.max_relative_error, 6),
            "weight_cv": round(self.weight_cv, 4),
            "sparsity": round(self.sparsity, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "per_target_errors": [e.to_dict() for e in self.per_target_errors],
        }


@dataclass
class ReweightingBenchmarkResult:
    """Results from comparing all reweighting methods."""

    method_results: list[ReweightingMethodResult]
    seed: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "methods": {mr.method_name: mr.to_dict() for mr in self.method_results},
        }

    def summary(self) -> str:
        lines = [
            "Reweighting Method Benchmark",
            "=" * 85,
            f"{'Method':<20} {'Mean Error':>10} {'Max Error':>10} "
            f"{'Weight CV':>10} {'Sparsity':>10} {'Time':>10}",
            "-" * 85,
        ]
        for mr in sorted(self.method_results, key=lambda x: x.mean_relative_error):
            lines.append(
                f"{mr.method_name:<20} {mr.mean_relative_error:>10.2%} "
                f"{mr.max_relative_error:>10.2%} {mr.weight_cv:>10.3f} "
                f"{mr.sparsity:>10.1%} {mr.elapsed_seconds:>9.1f}s"
            )
        lines.append("=" * 85)
        return "\n".join(lines)


# --- Method wrappers ---
# Each wraps either Calibrator or Reweighter from microplex.


class _CalibratorMethodBase:
    """Base for methods using microplex.calibration.Calibrator."""

    name: str = "CalibratorBase"
    _method: str = "ipf"

    def __init__(self, **calibrator_kwargs):
        self._kwargs = calibrator_kwargs
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
    ) -> "_CalibratorMethodBase":
        from microplex.calibration import Calibrator

        cal = Calibrator(method=self._method, **self._kwargs)
        cal.fit(data, marginal_targets, continuous_targets)
        self._weights = cal.weights_
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("Not fitted. Call fit() first.")
        return self._weights


class IPFMethod(_CalibratorMethodBase):
    """Iterative Proportional Fitting (raking)."""

    name = "IPF"
    _method = "ipf"


class Chi2Method(_CalibratorMethodBase):
    """Chi-square distance minimization."""

    name = "Chi2"
    _method = "chi2"


class EntropyMethod(_CalibratorMethodBase):
    """Entropy balancing (KL divergence)."""

    name = "Entropy"
    _method = "entropy"


class _ReweighterMethodBase:
    """Base for methods using microplex.reweighting.Reweighter.

    Note: Reweighter only supports categorical targets (not continuous).
    Continuous targets are silently ignored.
    """

    name: str = "ReweighterBase"
    _sparsity: str = "l1"

    def __init__(self, **reweighter_kwargs):
        self._kwargs = reweighter_kwargs
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
    ) -> "_ReweighterMethodBase":
        from microplex.reweighting import Reweighter

        rw = Reweighter(sparsity=self._sparsity, **self._kwargs)
        rw.fit(data, marginal_targets)
        self._weights = rw.weights_
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("Not fitted. Call fit() first.")
        return self._weights


class L1SparseMethod(_ReweighterMethodBase):
    """L1 sparse reweighting (linear programming)."""

    name = "L1-Sparse"
    _sparsity = "l1"


class L2SparseMethod(_ReweighterMethodBase):
    """L2 sparse reweighting (quadratic programming)."""

    name = "L2-Sparse"
    _sparsity = "l2"


class L0SparseMethod(_ReweighterMethodBase):
    """L0 sparse reweighting (iterative reweighted L1)."""

    name = "L0-Sparse"
    _sparsity = "l0"


class SparseCalibratorMethod:
    """FISTA-based sparse calibration with cross-category selection."""

    name = "SparseCalibrator"

    def __init__(self, sparsity_weight: float = 0.01, **kwargs):
        self._sparsity_weight = sparsity_weight
        self._kwargs = kwargs
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
    ) -> "SparseCalibratorMethod":
        from microplex.calibration import SparseCalibrator

        cal = SparseCalibrator(sparsity_weight=self._sparsity_weight, **self._kwargs)
        cal.fit(data, marginal_targets, continuous_targets)
        self._weights = cal.weights_
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("Not fitted. Call fit() first.")
        return self._weights


class HardConcreteMethod:
    """L0-regularized calibration using Hard Concrete distribution."""

    name = "HardConcrete"

    def __init__(self, lambda_l0: float = 1e-5, epochs: int = 2000, **kwargs):
        self._lambda_l0 = lambda_l0
        self._epochs = epochs
        self._kwargs = kwargs
        self._weights: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
    ) -> "HardConcreteMethod":
        from microplex.calibration import HardConcreteCalibrator

        cal = HardConcreteCalibrator(
            lambda_l0=self._lambda_l0, epochs=self._epochs, **self._kwargs
        )
        cal.fit(data, marginal_targets, continuous_targets)
        self._weights = cal.weights_
        return self

    def get_weights(self) -> np.ndarray:
        if self._weights is None:
            raise ValueError("Not fitted. Call fit() first.")
        return self._weights


# --- Benchmark Runner ---


def get_default_reweighting_methods() -> list:
    """Return all available reweighting methods with reasonable defaults."""
    methods = [
        IPFMethod(),
        Chi2Method(),
        EntropyMethod(),
        L1SparseMethod(),
        L2SparseMethod(),
        L0SparseMethod(),
        SparseCalibratorMethod(sparsity_weight=0.01),
    ]

    # Add HardConcrete if l0-python is available
    try:
        import l0
        methods.append(HardConcreteMethod(lambda_l0=1e-4, epochs=2000))
    except ImportError:
        pass

    return methods


class ReweightingBenchmarkRunner:
    """Run reweighting method comparison benchmark.

    Fits each method on the same data + targets and evaluates
    target-matching accuracy and weight properties.

    Usage:
        runner = ReweightingBenchmarkRunner()
        result = runner.run(
            data=synthetic_df,
            marginal_targets={"state": {"CA": 1000, "NY": 500}},
        )
        print(result.summary())
    """

    def __init__(self, methods: list = None):
        self.methods = methods if methods is not None else get_default_reweighting_methods()

    def run(
        self,
        data: pd.DataFrame,
        marginal_targets: dict[str, dict[str, float]],
        continuous_targets: dict[str, float] | None = None,
        seed: int = 42,
    ) -> ReweightingBenchmarkResult:
        """Run benchmark on all methods.

        Args:
            data: DataFrame with microdata (must include columns matching targets)
            marginal_targets: Categorical targets {var: {category: count}}
            continuous_targets: Continuous targets {var: total}
            seed: Random seed

        Returns:
            ReweightingBenchmarkResult with per-method metrics
        """
        method_results = []

        for method in self.methods:
            print(f"\n--- {method.name} ---")
            t0 = time.time()

            try:
                method.fit(data, marginal_targets, continuous_targets)
                elapsed = time.time() - t0
                weights = method.get_weights()

                result = ReweightingMethodResult.from_evaluation(
                    method_name=method.name,
                    data=data,
                    weights=weights,
                    marginal_targets=marginal_targets,
                    continuous_targets=continuous_targets,
                    elapsed_seconds=elapsed,
                )
                method_results.append(result)
                print(f"  Mean error: {result.mean_relative_error:.2%} "
                      f"Max error: {result.max_relative_error:.2%} "
                      f"CV: {result.weight_cv:.3f} "
                      f"Sparsity: {result.sparsity:.1%} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                # Record failure as high-error result
                method_results.append(ReweightingMethodResult(
                    method_name=method.name,
                    mean_relative_error=float("inf"),
                    max_relative_error=float("inf"),
                    weight_cv=0.0,
                    sparsity=0.0,
                    elapsed_seconds=time.time() - t0,
                ))

        return ReweightingBenchmarkResult(
            method_results=method_results,
            seed=seed,
        )
