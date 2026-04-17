"""
Tests for CPS synthetic microdata generation.

TDD tests that verify the synthetic microdata generator:
1. Can load/create CPS summary statistics
2. Generates data matching marginal distributions
3. Preserves correlations between variables
4. Runs fast with vectorized numpy operations
"""

import numpy as np
import pandas as pd
import pytest


class TestCPSSummaryStats:
    """Test CPS summary statistics loading and creation."""

    def test_create_from_dataframe(self):
        """Should create summary stats from a DataFrame."""
        from microplex.cps_synthetic import CPSSummaryStats

        # Create sample CPS-like data
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
            "age": np.random.randint(18, 80, n),
            "filing_status": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),
            "num_children": np.random.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        })
        # Set zero income for some
        data.loc[np.random.rand(n) < 0.15, "employment_income"] = 0

        stats = CPSSummaryStats.from_dataframe(data)

        assert stats is not None
        assert "employment_income" in stats.variables
        assert "age" in stats.variables

    def test_summary_stats_has_means(self):
        """Summary stats should include means for continuous variables."""
        from microplex.cps_synthetic import CPSSummaryStats

        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
            "age": np.random.randint(18, 80, n),
        })

        stats = CPSSummaryStats.from_dataframe(data)

        assert hasattr(stats, "means")
        assert "employment_income" in stats.means
        assert "age" in stats.means

    def test_summary_stats_has_quantiles(self):
        """Summary stats should include decile quantiles."""
        from microplex.cps_synthetic import CPSSummaryStats

        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
        })

        stats = CPSSummaryStats.from_dataframe(data)

        assert hasattr(stats, "quantiles")
        assert "employment_income" in stats.quantiles
        # Should have decile quantiles (10%, 20%, ..., 90%)
        assert len(stats.quantiles["employment_income"]) >= 9

    def test_summary_stats_has_correlations(self):
        """Summary stats should include correlation matrix."""
        from microplex.cps_synthetic import CPSSummaryStats

        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
            "age": np.random.randint(18, 80, n),
        })

        stats = CPSSummaryStats.from_dataframe(data)

        assert hasattr(stats, "correlation_matrix")
        assert stats.correlation_matrix.shape == (2, 2)

    def test_summary_stats_detects_discrete_vars(self):
        """Should detect and track discrete variables."""
        from microplex.cps_synthetic import CPSSummaryStats

        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
            "filing_status": np.random.choice([1, 2, 3, 4], n),
            "num_children": np.random.choice([0, 1, 2, 3], n),
        })

        stats = CPSSummaryStats.from_dataframe(data)

        assert hasattr(stats, "discrete_vars")
        assert "filing_status" in stats.discrete_vars
        assert "num_children" in stats.discrete_vars
        assert "employment_income" not in stats.discrete_vars


class TestCPSSyntheticGenerator:
    """Test synthetic data generation."""

    @pytest.fixture
    def sample_stats(self):
        """Create sample CPS summary stats."""
        from microplex.cps_synthetic import CPSSummaryStats

        np.random.seed(42)
        n = 2000

        # Create data with known correlations
        age = np.random.randint(18, 80, n)

        # Income depends on age
        base_income = np.random.lognormal(10, 1, n)
        income = base_income * (1 + 0.02 * (age - 18))
        income[np.random.rand(n) < 0.15] = 0

        data = pd.DataFrame({
            "employment_income": income,
            "age": age,
            "filing_status": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),
            "num_children": np.random.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        })

        return CPSSummaryStats.from_dataframe(data)

    def test_generate_returns_dataframe(self, sample_stats):
        """Generate should return a DataFrame."""
        from microplex.cps_synthetic import CPSSyntheticGenerator

        gen = CPSSyntheticGenerator(sample_stats)
        result = gen.generate(n=1000)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1000

    def test_generate_has_all_variables(self, sample_stats):
        """Generated data should include all variables."""
        from microplex.cps_synthetic import CPSSyntheticGenerator

        gen = CPSSyntheticGenerator(sample_stats)
        result = gen.generate(n=1000)

        assert "employment_income" in result.columns
        assert "age" in result.columns
        assert "filing_status" in result.columns
        assert "num_children" in result.columns

    def test_generate_is_reproducible_with_seed(self, sample_stats):
        """Generation should be reproducible with seed."""
        from microplex.cps_synthetic import CPSSyntheticGenerator

        gen = CPSSyntheticGenerator(sample_stats)
        result1 = gen.generate(n=100, seed=42)
        result2 = gen.generate(n=100, seed=42)

        pd.testing.assert_frame_equal(result1, result2)

    def test_generate_non_negative_income(self, sample_stats):
        """Income should be non-negative."""
        from microplex.cps_synthetic import CPSSyntheticGenerator

        gen = CPSSyntheticGenerator(sample_stats)
        result = gen.generate(n=1000, seed=42)

        assert (result["employment_income"] >= 0).all()

    def test_generate_valid_age_range(self, sample_stats):
        """Age should be in valid range."""
        from microplex.cps_synthetic import CPSSyntheticGenerator

        gen = CPSSyntheticGenerator(sample_stats)
        result = gen.generate(n=1000, seed=42)

        assert (result["age"] >= 0).all()
        assert (result["age"] <= 120).all()


class TestMarginalDistributionMatching:
    """Test that synthetic data matches marginal distributions."""

    @pytest.fixture
    def reference_data(self):
        """Create reference CPS-like data."""
        np.random.seed(42)
        n = 5000

        age = np.random.randint(18, 80, n)
        base_income = np.random.lognormal(10, 1, n)
        income = base_income * (1 + 0.02 * (age - 18))
        income[np.random.rand(n) < 0.15] = 0

        return pd.DataFrame({
            "employment_income": income,
            "age": age,
            "filing_status": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),
            "num_children": np.random.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        })

    def test_mean_within_tolerance(self, reference_data):
        """Synthetic mean should be within 15% of reference.

        Note: We use 15% tolerance because the copula approach with
        zero-inflation handling can have slightly higher mean error
        while still matching quantiles well.
        """
        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        stats = CPSSummaryStats.from_dataframe(reference_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=5000, seed=42)

        ref_mean = reference_data["employment_income"].mean()
        syn_mean = synthetic["employment_income"].mean()

        relative_error = abs(syn_mean - ref_mean) / ref_mean
        assert relative_error < 0.15, f"Mean relative error {relative_error:.2%} > 15%"

    def test_quantiles_match_deciles(self, reference_data):
        """Synthetic quantiles should match reference deciles."""
        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        stats = CPSSummaryStats.from_dataframe(reference_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=5000, seed=42)

        # Compare decile quantiles for income
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            ref_q = reference_data["employment_income"].quantile(q)
            syn_q = synthetic["employment_income"].quantile(q)

            if ref_q > 0:
                relative_error = abs(syn_q - ref_q) / ref_q
                assert relative_error < 0.20, (
                    f"Quantile {q} error {relative_error:.2%} > 20% "
                    f"(ref={ref_q:.0f}, syn={syn_q:.0f})"
                )

    def test_discrete_distribution_match(self, reference_data):
        """Discrete variable distributions should match."""
        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        stats = CPSSummaryStats.from_dataframe(reference_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=5000, seed=42)

        # Compare filing status distribution
        ref_dist = reference_data["filing_status"].value_counts(normalize=True)
        syn_dist = synthetic["filing_status"].value_counts(normalize=True)

        for cat in ref_dist.index:
            ref_prop = ref_dist[cat]
            syn_prop = syn_dist.get(cat, 0)
            diff = abs(syn_prop - ref_prop)
            assert diff < 0.05, f"Filing status {cat} diff {diff:.2%} > 5%"

    def test_zero_inflation_match(self, reference_data):
        """Zero fraction should match reference."""
        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        stats = CPSSummaryStats.from_dataframe(reference_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=5000, seed=42)

        ref_zero_frac = (reference_data["employment_income"] == 0).mean()
        syn_zero_frac = (synthetic["employment_income"] == 0).mean()

        diff = abs(syn_zero_frac - ref_zero_frac)
        assert diff < 0.05, f"Zero fraction diff {diff:.2%} > 5%"


class TestCorrelationPreservation:
    """Test that synthetic data preserves correlations."""

    @pytest.fixture
    def correlated_data(self):
        """Create data with known correlations."""
        np.random.seed(42)
        n = 5000

        # Create correlated age and income
        age = np.random.randint(18, 80, n)
        # Income increases with age (up to peak)
        age_effect = 1 + 0.03 * np.minimum(age - 18, 40) - 0.01 * np.maximum(age - 55, 0)
        income = np.random.lognormal(10, 0.8, n) * age_effect
        income[np.random.rand(n) < 0.15] = 0

        return pd.DataFrame({
            "employment_income": income,
            "age": age.astype(float),
            "filing_status": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),
            "num_children": np.random.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        })

    def test_correlation_sign_preserved(self, correlated_data):
        """Correlation sign should be preserved."""
        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        stats = CPSSummaryStats.from_dataframe(correlated_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=5000, seed=42)

        # Get positive income for correlation
        ref_positive = correlated_data[correlated_data["employment_income"] > 0]
        syn_positive = synthetic[synthetic["employment_income"] > 0]

        ref_corr = ref_positive["employment_income"].corr(ref_positive["age"])
        syn_corr = syn_positive["employment_income"].corr(syn_positive["age"])

        # Sign should match
        assert np.sign(ref_corr) == np.sign(syn_corr), (
            f"Correlation sign mismatch: ref={ref_corr:.3f}, syn={syn_corr:.3f}"
        )

    def test_correlation_magnitude_reasonable(self, correlated_data):
        """Correlation magnitude should be within reasonable range."""
        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        stats = CPSSummaryStats.from_dataframe(correlated_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=5000, seed=42)

        ref_positive = correlated_data[correlated_data["employment_income"] > 0]
        syn_positive = synthetic[synthetic["employment_income"] > 0]

        ref_corr = ref_positive["employment_income"].corr(ref_positive["age"])
        syn_corr = syn_positive["employment_income"].corr(syn_positive["age"])

        # Correlation should be within 0.3 of reference
        diff = abs(syn_corr - ref_corr)
        assert diff < 0.3, f"Correlation diff {diff:.3f} > 0.3"


class TestValidation:
    """Test validation functionality."""

    @pytest.fixture
    def reference_data(self):
        """Create reference data."""
        np.random.seed(42)
        n = 2000
        return pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
            "age": np.random.randint(18, 80, n),
            "filing_status": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),
            "num_children": np.random.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.2, 0.2, 0.1, 0.05]),
        })

    def test_validate_returns_metrics(self, reference_data):
        """Validation should return metrics dict."""
        from microplex.cps_synthetic import (
            CPSSummaryStats,
            CPSSyntheticGenerator,
            validate_synthetic,
        )

        stats = CPSSummaryStats.from_dataframe(reference_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=2000, seed=42)

        metrics = validate_synthetic(reference_data, synthetic)

        assert isinstance(metrics, dict)
        assert "ks_statistics" in metrics
        assert "mean_errors" in metrics

    def test_validate_includes_ks_stats(self, reference_data):
        """Validation should include KS statistics."""
        from microplex.cps_synthetic import (
            CPSSummaryStats,
            CPSSyntheticGenerator,
            validate_synthetic,
        )

        stats = CPSSummaryStats.from_dataframe(reference_data)
        gen = CPSSyntheticGenerator(stats)
        synthetic = gen.generate(n=2000, seed=42)

        metrics = validate_synthetic(reference_data, synthetic)

        assert "employment_income" in metrics["ks_statistics"]
        assert "age" in metrics["ks_statistics"]


class TestPerformance:
    """Test generation performance."""

    def test_generation_is_fast(self):
        """Generation should be fast (< 1s for 100k records)."""
        import time

        from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator

        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            "employment_income": np.random.lognormal(10, 1, n),
            "age": np.random.randint(18, 80, n),
            "filing_status": np.random.choice([1, 2, 3, 4], n),
            "num_children": np.random.choice([0, 1, 2, 3, 4], n),
        })

        stats = CPSSummaryStats.from_dataframe(data)
        gen = CPSSyntheticGenerator(stats)

        # Time generation of 100k records
        start = time.time()
        result = gen.generate(n=100_000, seed=42)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Generation took {elapsed:.2f}s > 1s"
        assert len(result) == 100_000
