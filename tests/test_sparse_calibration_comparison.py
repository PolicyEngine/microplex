"""
Benchmark comparing sparse calibration approaches:
1. SparseCalibrator (cross-category selection + IPF)
2. HardConcreteCalibrator (gradient descent with L0)

Run with: pytest tests/test_sparse_calibration_comparison.py -v -s
"""

import importlib.util
import time

import numpy as np
import pandas as pd
import pytest

from microplex.calibration import HardConcreteCalibrator, SparseCalibrator


def _require_l0() -> None:
    if importlib.util.find_spec("l0") is None:
        pytest.skip("l0-python not installed")


def test_hard_concrete_supports_explicit_linear_constraints():
    """HardConcreteCalibrator should accept explicit linear target rows."""
    _require_l0()

    from microplex.calibration import LinearConstraint

    data = pd.DataFrame({"weight": [1.0, 1.0]})
    constraints = (
        LinearConstraint(
            name="row1",
            coefficients=np.array([1.0, 0.0]),
            target=1.0,
        ),
        LinearConstraint(
            name="row2",
            coefficients=np.array([0.0, 1.0]),
            target=2.0,
        ),
    )

    calibrator = HardConcreteCalibrator(lambda_l0=1e-7, epochs=200, lr=0.05)
    result = calibrator.fit_transform(
        data,
        {},
        weight_col="weight",
        linear_constraints=constraints,
    )
    validation = calibrator.validate(result)

    assert validation["converged"] is True
    assert set(validation["linear_errors"]) == {"row1", "row2"}
    assert validation["max_error"] < 0.05


def generate_synthetic_population(n_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic population with known structure."""
    np.random.seed(seed)

    # States with different population sizes
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
    state_probs = np.array([0.12, 0.09, 0.07, 0.06, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03])
    state_probs = state_probs / state_probs.sum()

    # Age groups
    age_groups = ["0-17", "18-34", "35-54", "55-64", "65+"]
    age_probs = [0.22, 0.22, 0.26, 0.13, 0.17]

    # Income brackets
    income_brackets = ["<25k", "25-50k", "50-100k", "100k+"]
    income_probs = [0.20, 0.25, 0.35, 0.20]

    data = pd.DataFrame({
        "state": np.random.choice(states, n_records, p=state_probs),
        "age_group": np.random.choice(age_groups, n_records, p=age_probs),
        "income_bracket": np.random.choice(income_brackets, n_records, p=income_probs),
        "income": np.random.lognormal(10.5, 1.0, n_records),  # Continuous income
        "weight": np.ones(n_records),
    })

    return data


def compute_targets(data: pd.DataFrame, scale: float = 1.0) -> tuple:
    """Compute realistic calibration targets from data."""
    marginal_targets = {}

    # State targets
    marginal_targets["state"] = {}
    for state in data["state"].unique():
        count = (data["state"] == state).sum()
        marginal_targets["state"][state] = count * scale

    # Age targets
    marginal_targets["age_group"] = {}
    for age in data["age_group"].unique():
        count = (data["age_group"] == age).sum()
        marginal_targets["age_group"][age] = count * scale

    # Income bracket targets
    marginal_targets["income_bracket"] = {}
    for bracket in data["income_bracket"].unique():
        count = (data["income_bracket"] == bracket).sum()
        marginal_targets["income_bracket"][bracket] = count * scale

    # Continuous target: total income
    continuous_targets = {
        "income": data["income"].sum() * scale
    }

    return marginal_targets, continuous_targets


class TestSparseCalibrationComparison:
    """Compare SparseCalibrator vs HardConcreteCalibrator."""

    @pytest.fixture
    def population(self):
        """Generate test population."""
        return generate_synthetic_population(n_records=5000)

    @pytest.fixture
    def targets(self, population):
        """Compute targets from population."""
        return compute_targets(population)

    def test_sparse_calibrator_basic(self, population, targets):
        """Test SparseCalibrator (cross-category selection)."""
        marginal_targets, continuous_targets = targets

        calibrator = SparseCalibrator(target_sparsity=0.5)
        start = time.time()
        result = calibrator.fit_transform(
            population, marginal_targets, continuous_targets
        )
        elapsed = time.time() - start

        validation = calibrator.validate(result)

        print("\n=== SparseCalibrator (Cross-Category) ===")
        print(f"Time: {elapsed:.2f}s")
        print(f"Sparsity: {calibrator.get_sparsity():.1%}")
        print(f"Non-zero records: {calibrator.get_n_nonzero()}")
        print(f"Max error: {validation['max_error']:.2%}")
        print(f"Mean error: {validation['mean_error']:.2%}")

        # Should achieve target sparsity approximately
        assert calibrator.get_sparsity() > 0.3, "Should achieve significant sparsity"
        # Should have reasonable accuracy
        assert validation["max_error"] < 0.5, "Max error should be reasonable"

    def test_hard_concrete_calibrator_basic(self, population, targets):
        """Test HardConcreteCalibrator (gradient descent L0)."""
        _require_l0()
        marginal_targets, continuous_targets = targets

        calibrator = HardConcreteCalibrator(
            lambda_l0=1e-4,
            epochs=500,
            lr=0.1,
            verbose=True,
            verbose_freq=100,
        )
        start = time.time()
        result = calibrator.fit_transform(
            population, marginal_targets, continuous_targets
        )
        elapsed = time.time() - start

        validation = calibrator.validate(result)

        print("\n=== HardConcreteCalibrator (Gradient L0) ===")
        print(f"Time: {elapsed:.2f}s")
        print(f"Sparsity: {calibrator.get_sparsity():.1%}")
        print(f"Non-zero records: {calibrator.get_n_nonzero()}")
        print(f"Max error: {validation['max_error']:.2%}")
        print(f"Mean error: {validation['mean_error']:.2%}")

        # Should achieve some sparsity
        assert calibrator.get_sparsity() > 0.1, "Should achieve some sparsity"

    def test_comparison_at_same_sparsity(self, population, targets):
        """Compare both methods targeting similar sparsity levels."""
        _require_l0()
        marginal_targets, continuous_targets = targets
        target_sparsity = 0.7  # 70% sparse

        # Cross-category approach
        sparse_cal = SparseCalibrator(target_sparsity=target_sparsity)
        start1 = time.time()
        result1 = sparse_cal.fit_transform(population, marginal_targets, continuous_targets)
        time1 = time.time() - start1
        val1 = sparse_cal.validate(result1)

        # Hard Concrete approach - tune lambda for similar sparsity
        # Higher lambda = more sparsity
        hc_cal = HardConcreteCalibrator(
            lambda_l0=5e-4,  # Tuned for ~70% sparsity
            epochs=1000,
            lr=0.1,
        )
        start2 = time.time()
        result2 = hc_cal.fit_transform(population, marginal_targets, continuous_targets)
        time2 = time.time() - start2
        val2 = hc_cal.validate(result2)

        print(f"\n=== Comparison at ~{target_sparsity:.0%} target sparsity ===")
        print("\nCross-Category Selection:")
        print(f"  Time: {time1:.2f}s")
        print(f"  Sparsity: {sparse_cal.get_sparsity():.1%}")
        print(f"  Max error: {val1['max_error']:.2%}")
        print(f"  Mean error: {val1['mean_error']:.2%}")

        print("\nHard Concrete L0:")
        print(f"  Time: {time2:.2f}s")
        print(f"  Sparsity: {hc_cal.get_sparsity():.1%}")
        print(f"  Max error: {val2['max_error']:.2%}")
        print(f"  Mean error: {val2['mean_error']:.2%}")

        print("\n=== Summary ===")
        print(f"Speed advantage: Cross-Category {time2/time1:.1f}x faster")
        print(f"Accuracy advantage: {'Hard Concrete' if val2['mean_error'] < val1['mean_error'] else 'Cross-Category'}")

    def test_scaling_behavior(self):
        """Test how both methods scale with population size."""
        _require_l0()
        sizes = [1000, 5000, 10000]
        results = []

        for n in sizes:
            pop = generate_synthetic_population(n_records=n)
            marginal_targets, continuous_targets = compute_targets(pop)

            # Cross-category
            sparse_cal = SparseCalibrator(target_sparsity=0.5)
            start = time.time()
            result = sparse_cal.fit_transform(pop, marginal_targets, continuous_targets)
            time_cc = time.time() - start
            val_cc = sparse_cal.validate(result)

            # Hard Concrete (fewer epochs for speed)
            hc_cal = HardConcreteCalibrator(lambda_l0=1e-4, epochs=200, lr=0.1)
            start = time.time()
            result = hc_cal.fit_transform(pop, marginal_targets, continuous_targets)
            time_hc = time.time() - start
            val_hc = hc_cal.validate(result)

            results.append({
                "n": n,
                "cc_time": time_cc,
                "cc_error": val_cc["mean_error"],
                "cc_sparsity": sparse_cal.get_sparsity(),
                "hc_time": time_hc,
                "hc_error": val_hc["mean_error"],
                "hc_sparsity": hc_cal.get_sparsity(),
            })

        print("\n=== Scaling Comparison ===")
        print(f"{'N':>8} | {'CC Time':>8} | {'CC Err':>8} | {'HC Time':>8} | {'HC Err':>8}")
        print("-" * 50)
        for r in results:
            print(f"{r['n']:>8} | {r['cc_time']:>7.2f}s | {r['cc_error']:>7.1%} | {r['hc_time']:>7.2f}s | {r['hc_error']:>7.1%}")


if __name__ == "__main__":
    # Run comparison directly
    pop = generate_synthetic_population(n_records=5000)
    marginal_targets, continuous_targets = compute_targets(pop)

    print("=" * 60)
    print("SPARSE CALIBRATION COMPARISON")
    print("=" * 60)

    # Cross-category
    print("\n--- Cross-Category Selection ---")
    sparse_cal = SparseCalibrator(target_sparsity=0.7)
    start = time.time()
    result1 = sparse_cal.fit_transform(pop, marginal_targets, continuous_targets)
    time1 = time.time() - start
    val1 = sparse_cal.validate(result1)
    print(f"Time: {time1:.2f}s")
    print(f"Sparsity: {sparse_cal.get_sparsity():.1%}")
    print(f"Max error: {val1['max_error']:.2%}")
    print(f"Mean error: {val1['mean_error']:.2%}")

    # Hard Concrete
    print("\n--- Hard Concrete L0 ---")
    hc_cal = HardConcreteCalibrator(
        lambda_l0=5e-4,
        epochs=1000,
        lr=0.1,
        verbose=True,
        verbose_freq=200,
    )
    start = time.time()
    result2 = hc_cal.fit_transform(pop, marginal_targets, continuous_targets)
    time2 = time.time() - start
    val2 = hc_cal.validate(result2)
    print(f"Time: {time2:.2f}s")
    print(f"Sparsity: {hc_cal.get_sparsity():.1%}")
    print(f"Max error: {val2['max_error']:.2%}")
    print(f"Mean error: {val2['mean_error']:.2%}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cross-Category: {time1:.2f}s, {sparse_cal.get_sparsity():.0%} sparse, {val1['mean_error']:.1%} error")
    print(f"Hard Concrete:  {time2:.2f}s, {hc_cal.get_sparsity():.0%} sparse, {val2['mean_error']:.1%} error")
