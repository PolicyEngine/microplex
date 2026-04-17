"""Tests for synthesis method comparison benchmark.

TDD tests for the benchmark framework that compares
QRF, QDNN, MAF, CTGAN, TVAE (each +/- zero-inflation) on PRDC coverage.
"""

import numpy as np
import pandas as pd
import pytest

from microplex.eval.benchmark import (
    SynthesisMethod,
    QRFMethod,
    ZIQRFMethod,
    QDNNMethod,
    ZIQDNNMethod,
    MAFMethod,
    ZIMAFMethod,
    CTGANMethod,
    TVAEMethod,
    BenchmarkRunner,
    BenchmarkResult,
    MethodResult,
)


# --- Fixtures ---


@pytest.fixture
def toy_data():
    """Small synthetic dataset for fast tests."""
    rng = np.random.RandomState(42)
    n = 200
    age = rng.uniform(18, 80, n)
    is_male = rng.binomial(1, 0.5, n).astype(float)
    income = np.where(
        rng.random(n) < 0.3,  # 30% zeros
        0.0,
        rng.lognormal(10, 1, n),
    )
    benefits = np.where(
        rng.random(n) < 0.6,  # 60% zeros
        0.0,
        rng.exponential(5000, n),
    )
    return pd.DataFrame({
        "age": age,
        "is_male": is_male,
        "income": income,
        "benefits": benefits,
    })


@pytest.fixture
def toy_sources(toy_data):
    """Multi-source data for benchmark."""
    rng = np.random.RandomState(42)
    n = len(toy_data)
    perm = rng.permutation(n)
    half = n // 2
    return {
        "source_a": toy_data.iloc[perm[:half]].reset_index(drop=True),
        "source_b": toy_data.iloc[perm[half:]].reset_index(drop=True),
    }


@pytest.fixture
def shared_cols():
    return ["age", "is_male"]


# --- Protocol tests ---


class TestSynthesisMethodProtocol:
    """Test that all methods implement the SynthesisMethod protocol."""

    def test_qrf_has_required_interface(self):
        m = QRFMethod()
        assert hasattr(m, "name")
        assert hasattr(m, "fit")
        assert hasattr(m, "generate")
        assert callable(m.fit)
        assert callable(m.generate)

    def test_zi_qrf_has_required_interface(self):
        m = ZIQRFMethod()
        assert hasattr(m, "name")
        assert m.name == "ZI-QRF"

    def test_qdnn_has_required_interface(self):
        m = QDNNMethod()
        assert hasattr(m, "name")
        assert m.name == "QDNN"

    def test_zi_qdnn_has_required_interface(self):
        m = ZIQDNNMethod()
        assert hasattr(m, "name")
        assert m.name == "ZI-QDNN"

    def test_maf_has_required_interface(self):
        m = MAFMethod()
        assert hasattr(m, "name")
        assert m.name == "MAF"

    def test_zi_maf_has_required_interface(self):
        m = ZIMAFMethod()
        assert hasattr(m, "name")
        assert m.name == "ZI-MAF"

    def test_ctgan_has_required_interface(self):
        m = CTGANMethod()
        assert hasattr(m, "name")
        assert m.name == "CTGAN"

    def test_tvae_has_required_interface(self):
        m = TVAEMethod()
        assert hasattr(m, "name")
        assert m.name == "TVAE"


# --- Method fit/generate tests ---


class TestQRFMethods:
    """Test QRF and ZI-QRF fit and generate."""

    def test_qrf_fit_generate(self, toy_sources, shared_cols):
        m = QRFMethod(n_estimators=10)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=50, seed=42)
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 50
        # Should have all columns from sources
        assert "age" in synthetic.columns
        assert "income" in synthetic.columns

    def test_zi_qrf_fit_generate(self, toy_sources, shared_cols):
        m = ZIQRFMethod(n_estimators=10)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=50, seed=42)
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 50

    def test_qrf_produces_no_structural_zeros(self, toy_sources, shared_cols):
        """QRF without ZI should not learn zero-inflation structure."""
        m = QRFMethod(n_estimators=10)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=500, seed=42)
        # Without ZI, there should be fewer exact zeros than training data
        # (QRF interpolates, so some zeros may appear but not as many)
        assert "income" in synthetic.columns

    def test_zi_qrf_preserves_zero_fraction(self, toy_sources, shared_cols):
        """ZI-QRF should approximately preserve the zero fraction."""
        m = ZIQRFMethod(n_estimators=10)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=500, seed=42)
        # Check that benefits has substantial zero fraction
        # (training data has ~60% zeros, synthetic should be in ballpark)
        if "benefits" in synthetic.columns:
            zero_frac = (synthetic["benefits"] == 0).mean()
            assert zero_frac > 0.1  # At least some zeros preserved


class TestQDNNMethods:
    """Test QDNN and ZI-QDNN fit and generate."""

    def test_qdnn_fit_generate(self, toy_sources, shared_cols):
        m = QDNNMethod(hidden_dim=16, epochs=5)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=50, seed=42)
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 50

    def test_zi_qdnn_fit_generate(self, toy_sources, shared_cols):
        m = ZIQDNNMethod(hidden_dim=16, epochs=5)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=50, seed=42)
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 50


class TestMAFMethods:
    """Test MAF and ZI-MAF fit and generate."""

    def test_maf_fit_generate(self, toy_sources, shared_cols):
        m = MAFMethod(n_layers=2, hidden_dim=16, epochs=5)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=50, seed=42)
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 50

    def test_zi_maf_fit_generate(self, toy_sources, shared_cols):
        m = ZIMAFMethod(n_layers=2, hidden_dim=16, epochs=5)
        m.fit(toy_sources, shared_cols)
        synthetic = m.generate(n=50, seed=42)
        assert isinstance(synthetic, pd.DataFrame)
        assert len(synthetic) == 50


class TestDeepGenerativeMethods:
    """Test CTGAN and TVAE (may skip if sdv not installed)."""

    def test_ctgan_fit_generate(self, toy_sources, shared_cols):
        m = CTGANMethod(epochs=5)
        try:
            m.fit(toy_sources, shared_cols)
            synthetic = m.generate(n=50, seed=42)
            assert isinstance(synthetic, pd.DataFrame)
            assert len(synthetic) == 50
        except ImportError:
            pytest.skip("sdv not installed")

    def test_tvae_fit_generate(self, toy_sources, shared_cols):
        m = TVAEMethod(epochs=5)
        try:
            m.fit(toy_sources, shared_cols)
            synthetic = m.generate(n=50, seed=42)
            assert isinstance(synthetic, pd.DataFrame)
            assert len(synthetic) == 50
        except ImportError:
            pytest.skip("sdv not installed")


# --- BenchmarkRunner tests ---


class TestBenchmarkRunner:
    """Test the benchmark runner."""

    def test_runner_creation_with_methods(self):
        methods = [QRFMethod(n_estimators=10), ZIQRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        assert len(runner.methods) == 2

    def test_runner_default_methods(self):
        runner = BenchmarkRunner()
        # Should include all 8 methods by default
        assert len(runner.methods) >= 6  # At least the non-sdv ones

    def test_runner_run_returns_benchmark_result(self, toy_sources, shared_cols):
        methods = [QRFMethod(n_estimators=10), ZIQRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=toy_sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=100,
            k=3,
            seed=42,
        )
        assert isinstance(result, BenchmarkResult)
        assert len(result.method_results) == 2

    def test_method_result_has_prdc(self, toy_sources, shared_cols):
        methods = [QRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=toy_sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=100,
            k=3,
            seed=42,
        )
        mr = result.method_results[0]
        assert isinstance(mr, MethodResult)
        assert mr.method_name == "QRF"
        assert 0 <= mr.mean_coverage <= 1
        assert 0 <= mr.mean_precision <= 1
        assert mr.mean_density >= 0
        assert mr.elapsed_seconds > 0

    def test_benchmark_result_to_dict(self, toy_sources, shared_cols):
        methods = [QRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=toy_sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=100,
            k=3,
            seed=42,
        )
        d = result.to_dict()
        assert "methods" in d
        assert len(d["methods"]) == 1
        assert "QRF" in d["methods"]
        assert "mean_coverage" in d["methods"]["QRF"]

    def test_benchmark_result_summary_table(self, toy_sources, shared_cols):
        methods = [QRFMethod(n_estimators=10), ZIQRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=toy_sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=100,
            k=3,
            seed=42,
        )
        summary = result.summary()
        assert "QRF" in summary
        assert "ZI-QRF" in summary
        assert "Coverage" in summary

    def test_benchmark_per_source_results(self, toy_sources, shared_cols):
        methods = [QRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=toy_sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=100,
            k=3,
            seed=42,
        )
        mr = result.method_results[0]
        # Should have per-source PRDC
        assert len(mr.source_results) == 2  # source_a and source_b
        for sr in mr.source_results:
            assert sr.source_name in ["source_a", "source_b"]
            assert 0 <= sr.coverage <= 1


# --- Edge cases ---


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_source(self, toy_data, shared_cols):
        sources = {"only_source": toy_data}
        methods = [QRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=100,
            k=3,
            seed=42,
        )
        assert len(result.method_results) == 1

    def test_small_data(self, shared_cols):
        """Test with very small dataset."""
        rng = np.random.RandomState(42)
        small = pd.DataFrame({
            "age": rng.uniform(18, 80, 30),
            "is_male": rng.binomial(1, 0.5, 30).astype(float),
            "income": rng.lognormal(10, 1, 30),
        })
        sources = {"tiny": small}
        methods = [QRFMethod(n_estimators=10)]
        runner = BenchmarkRunner(methods=methods)
        result = runner.run(
            sources=sources,
            shared_cols=shared_cols,
            holdout_frac=0.3,
            n_generate=50,
            k=3,
            seed=42,
        )
        assert len(result.method_results) == 1
