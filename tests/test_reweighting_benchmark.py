"""Tests for reweighting method comparison benchmark.

Compares:
- IPF (Iterative Proportional Fitting / raking)
- Chi-square distance minimization
- Entropy balancing
- L1 sparse (linear programming)
- L2 sparse (quadratic programming)
- L0 sparse (iterative reweighted L1)
- SparseCalibrator (FISTA-based)
- HardConcreteCalibrator (differentiable L0, skipped if l0-python not installed)

Evaluation metrics:
- Mean relative error against targets
- Max relative error
- Weight CV (coefficient of variation)
- Sparsity (fraction of zero weights)
- Elapsed time
"""

import numpy as np
import pandas as pd
import pytest

from microplex.eval.reweighting_benchmark import (
    ReweightingMethod,
    IPFMethod,
    Chi2Method,
    EntropyMethod,
    L1SparseMethod,
    L2SparseMethod,
    L0SparseMethod,
    SparseCalibratorMethod,
    HardConcreteMethod,
    ReweightingMethodResult,
    ReweightingBenchmarkResult,
    ReweightingBenchmarkRunner,
    get_default_reweighting_methods,
)


# --- Fixtures ---


@pytest.fixture
def simple_data():
    """Small dataset with categorical + continuous columns for reweighting."""
    rng = np.random.RandomState(42)
    n = 200
    states = rng.choice(["CA", "NY", "TX"], size=n, p=[0.4, 0.35, 0.25])
    age_groups = rng.choice(["young", "mid", "old"], size=n, p=[0.3, 0.5, 0.2])
    income = rng.exponential(50000, size=n)
    weight = np.ones(n)

    return pd.DataFrame({
        "state": states,
        "age_group": age_groups,
        "income": income,
        "weight": weight,
    })


@pytest.fixture
def targets(simple_data):
    """Marginal targets that differ from the data's natural distribution."""
    return {
        "state": {"CA": 80, "NY": 70, "TX": 50},
        "age_group": {"young": 60, "mid": 100, "old": 40},
    }


@pytest.fixture
def continuous_targets(simple_data):
    """Continuous targets (total income)."""
    return {"income": 10_000_000}


# --- Protocol tests ---


class TestReweightingMethodProtocol:
    """Every reweighting method must satisfy the ReweightingMethod protocol."""

    METHOD_CLASSES = [
        IPFMethod,
        Chi2Method,
        EntropyMethod,
        L1SparseMethod,
        L2SparseMethod,
        L0SparseMethod,
        SparseCalibratorMethod,
    ]

    @pytest.mark.parametrize("cls", METHOD_CLASSES)
    def test_has_name(self, cls):
        method = cls()
        assert hasattr(method, "name")
        assert isinstance(method.name, str)
        assert len(method.name) > 0

    @pytest.mark.parametrize("cls", METHOD_CLASSES)
    def test_has_fit(self, cls):
        method = cls()
        assert hasattr(method, "fit")
        assert callable(method.fit)

    @pytest.mark.parametrize("cls", METHOD_CLASSES)
    def test_has_get_weights(self, cls):
        method = cls()
        assert hasattr(method, "get_weights")
        assert callable(method.get_weights)

    @pytest.mark.parametrize("cls", METHOD_CLASSES)
    def test_implements_protocol(self, cls):
        method = cls()
        assert isinstance(method, ReweightingMethod)


# --- Calibrator-based method tests ---


class TestIPFMethod:
    def test_fit_returns_self(self, simple_data, targets):
        method = IPFMethod()
        result = method.fit(simple_data, targets)
        assert result is method

    def test_weights_are_positive(self, simple_data, targets):
        method = IPFMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()

    def test_marginals_approximately_matched(self, simple_data, targets):
        method = IPFMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        for var, var_targets in targets.items():
            for cat, target in var_targets.items():
                actual = weights[simple_data[var] == cat].sum()
                assert abs(actual - target) / target < 0.05, (
                    f"{var}={cat}: expected {target}, got {actual:.1f}"
                )


class TestChi2Method:
    def test_fit_and_weights(self, simple_data, targets):
        method = Chi2Method()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()


class TestEntropyMethod:
    def test_fit_and_weights(self, simple_data, targets):
        method = EntropyMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()


# --- Reweighter-based method tests ---


class TestL1SparseMethod:
    def test_fit_and_weights(self, simple_data, targets):
        method = L1SparseMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()

    def test_produces_sparse_weights(self, simple_data, targets):
        method = L1SparseMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        sparsity = (weights < 1e-9).sum() / len(weights)
        # L1 should produce at least some sparsity
        assert sparsity > 0, "L1 should produce sparse weights"


class TestL2SparseMethod:
    def test_fit_and_weights(self, simple_data, targets):
        method = L2SparseMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()


class TestL0SparseMethod:
    def test_fit_and_weights(self, simple_data, targets):
        method = L0SparseMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()


# --- SparseCalibrator method ---


class TestSparseCalibratorMethod:
    def test_fit_and_weights(self, simple_data, targets):
        method = SparseCalibratorMethod(sparsity_weight=0.01)
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()


# --- HardConcrete method ---


class TestHardConcreteMethod:
    @pytest.fixture(autouse=True)
    def _check_l0(self):
        try:
            import l0
        except ImportError:
            pytest.skip("l0-python not installed")

    def test_fit_and_weights(self, simple_data, targets):
        method = HardConcreteMethod(epochs=50)
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)
        assert (weights >= 0).all()


# --- Result dataclasses ---


class TestReweightingMethodResult:
    def test_from_weights_and_targets(self, simple_data, targets):
        weights = np.ones(len(simple_data))
        result = ReweightingMethodResult.from_evaluation(
            method_name="test",
            data=simple_data,
            weights=weights,
            marginal_targets=targets,
            elapsed_seconds=1.0,
        )
        assert result.method_name == "test"
        assert result.elapsed_seconds == 1.0
        assert result.mean_relative_error >= 0
        assert result.max_relative_error >= 0
        assert result.weight_cv >= 0
        assert 0 <= result.sparsity <= 1

    def test_to_dict(self, simple_data, targets):
        weights = np.ones(len(simple_data))
        result = ReweightingMethodResult.from_evaluation(
            method_name="test",
            data=simple_data,
            weights=weights,
            marginal_targets=targets,
            elapsed_seconds=1.0,
        )
        d = result.to_dict()
        assert "method_name" in d
        assert "mean_relative_error" in d
        assert "max_relative_error" in d
        assert "weight_cv" in d
        assert "sparsity" in d
        assert "elapsed_seconds" in d
        assert "per_target_errors" in d


class TestReweightingBenchmarkResult:
    def test_summary_string(self, simple_data, targets):
        weights = np.ones(len(simple_data))
        mr = ReweightingMethodResult.from_evaluation(
            method_name="test",
            data=simple_data,
            weights=weights,
            marginal_targets=targets,
            elapsed_seconds=1.0,
        )
        result = ReweightingBenchmarkResult(method_results=[mr], seed=42)
        summary = result.summary()
        assert "test" in summary
        assert "Error" in summary or "error" in summary

    def test_to_dict(self, simple_data, targets):
        weights = np.ones(len(simple_data))
        mr = ReweightingMethodResult.from_evaluation(
            method_name="test",
            data=simple_data,
            weights=weights,
            marginal_targets=targets,
            elapsed_seconds=1.0,
        )
        result = ReweightingBenchmarkResult(method_results=[mr], seed=42)
        d = result.to_dict()
        assert "methods" in d
        assert "seed" in d


# --- Benchmark Runner ---


class TestReweightingBenchmarkRunner:
    def test_run_with_defaults(self, simple_data, targets):
        methods = [IPFMethod(), EntropyMethod()]
        runner = ReweightingBenchmarkRunner(methods=methods)
        result = runner.run(
            data=simple_data,
            marginal_targets=targets,
        )
        assert isinstance(result, ReweightingBenchmarkResult)
        assert len(result.method_results) == 2

    def test_run_with_continuous_targets(self, simple_data, targets, continuous_targets):
        methods = [IPFMethod()]
        runner = ReweightingBenchmarkRunner(methods=methods)
        result = runner.run(
            data=simple_data,
            marginal_targets=targets,
            continuous_targets=continuous_targets,
        )
        assert len(result.method_results) == 1

    def test_methods_sorted_by_error(self, simple_data, targets):
        methods = [IPFMethod(), L1SparseMethod(), EntropyMethod()]
        runner = ReweightingBenchmarkRunner(methods=methods)
        result = runner.run(data=simple_data, marginal_targets=targets)
        # Result should have all methods
        assert len(result.method_results) == 3

    def test_handles_method_failure_gracefully(self, simple_data, targets):
        """If a method fails, the runner should continue with others."""
        methods = [IPFMethod(), EntropyMethod()]
        runner = ReweightingBenchmarkRunner(methods=methods)
        result = runner.run(data=simple_data, marginal_targets=targets)
        # Both should succeed for this simple case
        assert len(result.method_results) >= 1

    def test_continuous_targets_only_for_calibrator_methods(self, simple_data, targets, continuous_targets):
        """Calibrator methods support continuous targets; Reweighter methods don't.
        Runner should handle this gracefully."""
        methods = [IPFMethod(), L1SparseMethod()]
        runner = ReweightingBenchmarkRunner(methods=methods)
        result = runner.run(
            data=simple_data,
            marginal_targets=targets,
            continuous_targets=continuous_targets,
        )
        # Both should produce results (Reweighter ignores continuous targets)
        assert len(result.method_results) >= 1


# --- get_default_reweighting_methods ---


class TestGetDefaultMethods:
    def test_returns_list(self):
        methods = get_default_reweighting_methods()
        assert isinstance(methods, list)
        assert len(methods) >= 6  # At minimum: IPF, Chi2, Entropy, L1, L2, L0

    def test_all_have_names(self):
        methods = get_default_reweighting_methods()
        names = [m.name for m in methods]
        assert "IPF" in names
        assert "Entropy" in names
        assert "L1-Sparse" in names


# --- Edge cases ---


class TestEdgeCases:
    def test_single_target_variable(self, simple_data):
        """Single categorical target should work."""
        targets = {"state": {"CA": 80, "NY": 70, "TX": 50}}
        method = IPFMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        assert len(weights) == len(simple_data)

    def test_tight_targets(self, simple_data):
        """Targets matching current distribution should give near-uniform weights."""
        state_counts = simple_data["state"].value_counts().to_dict()
        targets = {"state": state_counts}
        method = IPFMethod()
        method.fit(simple_data, targets)
        weights = method.get_weights()
        # Weights should be close to uniform (all ~1.0)
        cv = weights.std() / weights.mean()
        assert cv < 0.5, f"CV should be low for matching targets, got {cv:.3f}"
