"""Tests for unified evaluation harness."""

import numpy as np
import pandas as pd
import pytest

from microplex.eval.harness import (
    SynthesisEvalResult,
    ReweightingEvalResult,
    SourceCoverage,
    AggregateError,
    EvalHarness,
)


# --- Fixtures ---


def make_source(n: int, cols: list[str], name: str, seed: int = 42) -> pd.DataFrame:
    """Create a simple source DataFrame for testing."""
    rng = np.random.RandomState(seed)
    data = {col: rng.randn(n) for col in cols}
    data["_survey"] = name
    return pd.DataFrame(data)


def make_weighted_data(n: int, seed: int = 42) -> pd.DataFrame:
    """Create weighted microdata with columns matching target registry."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "age": rng.randint(0, 90, n),
            "employment_income": np.maximum(0, rng.normal(50000, 30000, n)),
            "self_employment_income": np.where(
                rng.random(n) < 0.85, 0, rng.exponential(20000, n)
            ),
            "social_security": np.where(
                rng.random(n) < 0.8, 0, rng.exponential(15000, n)
            ),
            "snap": np.where(rng.random(n) < 0.88, 0, rng.exponential(3000, n)),
            "state_fips": rng.choice(["06", "36", "48"], n),
            "weight": rng.uniform(100, 5000, n),
        }
    )
    return df


class FakeModel:
    """Minimal model with fit/generate for testing."""

    def __init__(self):
        self._train_data = None

    def fit(self, sources: dict[str, pd.DataFrame], shared_cols: list[str]):
        # Just store concatenated data for bootstrap generation
        all_dfs = []
        for df in sources.values():
            cols = [c for c in shared_cols if c in df.columns]
            all_dfs.append(df[cols])
        self._train_data = pd.concat(all_dfs, ignore_index=True)
        self._shared_cols = shared_cols
        return self

    def generate(self, n: int, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(self._train_data), size=n, replace=True)
        result = self._train_data.iloc[idx].reset_index(drop=True)
        # Add small noise to avoid exact matches
        for col in self._shared_cols:
            if col in result.columns and result[col].dtype in [
                np.float64,
                np.int64,
            ]:
                result[col] = result[col].astype(float) + rng.normal(
                    0, 0.01, len(result)
                )
        return result


# --- Synthesis eval tests ---


class TestSynthesisEval:
    def test_source_coverage_fields(self):
        sc = SourceCoverage(
            source_name="CPS",
            precision=0.8,
            recall=0.75,
            density=1.2,
            coverage=0.7,
            n_holdout=1000,
            n_synthetic=5000,
            columns_evaluated=["age", "income"],
        )
        assert sc.source_name == "CPS"
        assert sc.coverage == 0.7
        assert len(sc.columns_evaluated) == 2

    def test_synthesis_eval_result_mean_coverage(self):
        r = SynthesisEvalResult(
            source_coverages=[
                SourceCoverage("A", 0.8, 0.7, 1.0, 0.6, 100, 500, ["x"]),
                SourceCoverage("B", 0.9, 0.8, 1.1, 0.8, 100, 500, ["x"]),
            ],
            n_synthetic=500,
        )
        assert r.mean_coverage == pytest.approx(0.7, abs=0.01)

    def test_eval_harness_synthesis_runs(self):
        """EvalHarness.evaluate_synthesis returns per-source PRDC."""
        shared = ["x", "y"]
        sources = {
            "A": make_source(200, ["x", "y", "z"], "A", seed=1),
            "B": make_source(200, ["x", "y", "w"], "B", seed=2),
        }
        model = FakeModel()
        harness = EvalHarness()
        result = harness.evaluate_synthesis(
            model=model,
            sources=sources,
            shared_cols=shared,
            holdout_frac=0.2,
            k=3,
        )
        assert isinstance(result, SynthesisEvalResult)
        assert len(result.source_coverages) == 2
        names = {sc.source_name for sc in result.source_coverages}
        assert names == {"A", "B"}
        # Coverage should be between 0 and 1
        for sc in result.source_coverages:
            assert 0 <= sc.coverage <= 1
            assert 0 <= sc.precision <= 1
            assert 0 <= sc.recall <= 1
            assert sc.n_holdout > 0

    def test_synthesis_eval_with_missing_cols(self):
        """Sources with different column sets are handled correctly."""
        sources = {
            "A": make_source(200, ["x", "y"], "A", seed=1),
            "B": make_source(200, ["x", "z"], "B", seed=2),
        }
        model = FakeModel()
        harness = EvalHarness()
        result = harness.evaluate_synthesis(
            model=model,
            sources=sources,
            shared_cols=["x"],
            holdout_frac=0.2,
            k=3,
        )
        # Both sources should still be evaluated (on shared cols)
        assert len(result.source_coverages) >= 1


# --- Reweighting eval tests ---


class TestReweightingEval:
    def test_aggregate_error_fields(self):
        ae = AggregateError(
            target_name="employment_income",
            category="income",
            target_value=9e12,
            computed_value=8.5e12,
            relative_error=5.56,
            absolute_error=5e11,
        )
        assert ae.relative_error == pytest.approx(5.56)

    def test_reweighting_eval_result_mean_error(self):
        r = ReweightingEvalResult(
            aggregate_errors=[
                AggregateError("a", "income", 100, 95, 5.0, 5),
                AggregateError("b", "income", 200, 180, 10.0, 20),
            ],
            n_targets=2,
            n_matched=2,
        )
        assert r.mean_relative_error == pytest.approx(7.5)
        assert r.max_relative_error == pytest.approx(10.0)

    def test_reweighting_eval_by_category(self):
        r = ReweightingEvalResult(
            aggregate_errors=[
                AggregateError("a", "income", 100, 95, 5.0, 5),
                AggregateError("b", "benefits", 200, 180, 10.0, 20),
                AggregateError("c", "income", 300, 290, 3.33, 10),
            ],
            n_targets=3,
            n_matched=3,
        )
        by_cat = r.errors_by_category()
        assert "income" in by_cat
        assert "benefits" in by_cat
        assert len(by_cat["income"]) == 2
        assert len(by_cat["benefits"]) == 1

    def test_eval_harness_reweighting_runs(self):
        """EvalHarness.evaluate_reweighting computes errors vs targets."""
        df = make_weighted_data(1000, seed=42)
        harness = EvalHarness()
        result = harness.evaluate_reweighting(
            data=df,
            weight_col="weight",
        )
        assert isinstance(result, ReweightingEvalResult)
        # Should have matched at least some targets
        assert result.n_matched > 0
        # Errors should be non-negative
        for ae in result.aggregate_errors:
            assert ae.relative_error >= 0


# --- Full eval tests ---


class TestFullEval:
    def test_report_json_serializable(self):
        """Full eval result can be serialized to dict."""
        synth_result = SynthesisEvalResult(
            source_coverages=[
                SourceCoverage("A", 0.8, 0.7, 1.0, 0.65, 100, 500, ["x", "y"]),
            ],
            n_synthetic=500,
        )
        rw_result = ReweightingEvalResult(
            aggregate_errors=[
                AggregateError("emp_inc", "income", 9e12, 8.5e12, 5.56, 5e11),
            ],
            n_targets=1,
            n_matched=1,
        )

        report = {
            "synthesis": synth_result.to_dict(),
            "reweighting": rw_result.to_dict(),
        }
        # Should be JSON-serializable (no numpy types)
        import json

        json_str = json.dumps(report)
        assert "mean_coverage" in json_str
        assert "mean_relative_error" in json_str
