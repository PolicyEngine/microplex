"""
Tests for the Reweighter class.

TDD tests that verify sparse reweighting functionality:
1. Initialize with optimization backend
2. Fit weights to match population targets
3. Transform data by applying weights
4. Handle geographic hierarchies (state, county, tract)
5. L0/L1 sparsity optimization
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest


class TestReweighterInit:
    """Test Reweighter initialization."""

    def test_basic_initialization(self):
        """Should initialize with default backend."""
        from microplex import Reweighter

        reweighter = Reweighter()

        assert reweighter.backend in ["scipy", "cvxpy"]
        assert not reweighter.is_fitted_

    def test_explicit_backend(self):
        """Should accept explicit backend choice."""
        from microplex import Reweighter

        reweighter = Reweighter(backend="scipy")

        assert reweighter.backend == "scipy"

    def test_invalid_backend_raises_error(self):
        """Should raise error for invalid backend."""
        from microplex import Reweighter

        with pytest.raises(ValueError, match="backend"):
            Reweighter(backend="invalid")

    def test_sparsity_parameter(self):
        """Should accept sparsity parameter."""
        from microplex import Reweighter

        reweighter = Reweighter(sparsity="l0")

        assert reweighter.sparsity == "l0"


class TestReweighterFit:
    """Test fitting weights to population targets."""

    @pytest.fixture
    def sample_data(self):
        """Create sample synthetic microdata."""
        np.random.seed(42)
        n = 1000

        return pd.DataFrame({
            "state": np.random.choice(["CA", "NY", "TX"], n),
            "age_group": np.random.choice(["0-17", "18-64", "65+"], n),
            "income": np.random.lognormal(10, 1, n),
            "weight": np.ones(n),  # Initial uniform weights
        })

    @pytest.fixture
    def simple_targets(self):
        """Create simple population targets."""
        # Target: 60% CA, 25% NY, 15% TX
        return {
            "state": {
                "CA": 600,
                "NY": 250,
                "TX": 150,
            }
        }

    def test_fit_completes(self, sample_data, simple_targets):
        """Fit should complete without errors."""
        from microplex import Reweighter

        reweighter = Reweighter()
        reweighter.fit(sample_data, simple_targets)

        assert reweighter.is_fitted_

    def test_fit_computes_weights(self, sample_data, simple_targets):
        """Fit should compute weight array."""
        from microplex import Reweighter

        reweighter = Reweighter()
        reweighter.fit(sample_data, simple_targets)

        assert reweighter.weights_ is not None
        assert len(reweighter.weights_) == len(sample_data)

    def test_fit_weights_are_sparse(self, sample_data, simple_targets):
        """With L0 optimization, many weights should be zero."""
        from microplex import Reweighter

        reweighter = Reweighter(sparsity="l0")
        reweighter.fit(sample_data, simple_targets)

        n_nonzero = np.sum(reweighter.weights_ > 1e-6)
        # Should use far fewer records than total
        assert n_nonzero < len(sample_data) * 0.5

    def test_fit_matches_targets(self, sample_data, simple_targets):
        """Weighted data should match target margins."""
        from microplex import Reweighter

        reweighter = Reweighter()
        reweighter.fit(sample_data, simple_targets)

        # Apply weights
        weighted = sample_data.copy()
        weighted["weight"] = reweighter.weights_

        # Check state targets
        state_counts = weighted.groupby("state")["weight"].sum()

        np.testing.assert_allclose(
            state_counts["CA"], simple_targets["state"]["CA"], rtol=0.01
        )
        np.testing.assert_allclose(
            state_counts["NY"], simple_targets["state"]["NY"], rtol=0.01
        )
        np.testing.assert_allclose(
            state_counts["TX"], simple_targets["state"]["TX"], rtol=0.01
        )

    def test_fit_with_multiple_margins(self, sample_data):
        """Should handle multiple margin constraints."""
        from microplex import Reweighter

        targets = {
            "state": {"CA": 600, "NY": 250, "TX": 150},
            "age_group": {"0-17": 200, "18-64": 600, "65+": 200},
        }

        reweighter = Reweighter()
        reweighter.fit(sample_data, targets)

        weighted = sample_data.copy()
        weighted["weight"] = reweighter.weights_

        # Check both margins
        state_counts = weighted.groupby("state")["weight"].sum()
        age_counts = weighted.groupby("age_group")["weight"].sum()

        np.testing.assert_allclose(state_counts["CA"], 600, rtol=0.01)
        np.testing.assert_allclose(age_counts["18-64"], 600, rtol=0.01)

    def test_fit_preserves_total_population(self, sample_data, simple_targets):
        """Total weighted population should match target total."""
        from microplex import Reweighter

        reweighter = Reweighter()
        reweighter.fit(sample_data, simple_targets)

        total_target = sum(simple_targets["state"].values())
        total_weighted = reweighter.weights_.sum()

        np.testing.assert_allclose(total_weighted, total_target, rtol=0.01)


class TestReweighterTransform:
    """Test applying weights to data."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 500

        return pd.DataFrame({
            "state": np.random.choice(["CA", "NY"], n),
            "income": np.random.lognormal(10, 1, n),
            "weight": np.ones(n),
        })

    @pytest.fixture
    def fitted_reweighter(self, sample_data):
        """Return a fitted reweighter."""
        from microplex import Reweighter

        targets = {"state": {"CA": 300, "NY": 200}}

        reweighter = Reweighter()
        reweighter.fit(sample_data, targets)
        return reweighter

    def test_transform_returns_dataframe(self, fitted_reweighter, sample_data):
        """Transform should return a DataFrame."""
        result = fitted_reweighter.transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    def test_transform_updates_weights(self, fitted_reweighter, sample_data):
        """Transform should update weight column."""
        result = fitted_reweighter.transform(sample_data)

        # Weights should be different from original
        assert not np.allclose(result["weight"].values, sample_data["weight"].values)

        # Should match fitted weights
        np.testing.assert_array_equal(result["weight"].values, fitted_reweighter.weights_)

    def test_transform_preserves_data(self, fitted_reweighter, sample_data):
        """Transform should preserve non-weight columns."""
        result = fitted_reweighter.transform(sample_data)

        pd.testing.assert_series_equal(
            result["state"], sample_data["state"], check_names=False
        )
        pd.testing.assert_series_equal(
            result["income"], sample_data["income"], check_names=False
        )

    def test_transform_filters_zero_weights(self, fitted_reweighter, sample_data):
        """Transform with drop_zeros=True should remove zero-weight records."""
        result = fitted_reweighter.transform(sample_data, drop_zeros=True)

        # All remaining records should have positive weight
        assert (result["weight"] > 0).all()

        # Should be fewer records than original
        assert len(result) <= len(sample_data)


class TestReweighterFitTransform:
    """Test fit_transform convenience method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 300

        return pd.DataFrame({
            "state": np.random.choice(["CA", "TX"], n),
            "age": np.random.randint(18, 80, n),
            "weight": np.ones(n),
        })

    def test_fit_transform_returns_weighted_data(self, sample_data):
        """fit_transform should fit and apply weights in one call."""
        from microplex import Reweighter

        targets = {"state": {"CA": 200, "TX": 100}}

        reweighter = Reweighter()
        result = reweighter.fit_transform(sample_data, targets)

        assert isinstance(result, pd.DataFrame)
        assert reweighter.is_fitted_
        assert len(result) == len(sample_data)

    def test_fit_transform_matches_transform(self, sample_data):
        """fit_transform should give same result as fit then transform."""
        from microplex import Reweighter

        targets = {"state": {"CA": 200, "TX": 100}}

        # Method 1: fit_transform
        reweighter1 = Reweighter(backend="scipy")
        result1 = reweighter1.fit_transform(sample_data, targets)

        # Method 2: fit then transform
        reweighter2 = Reweighter(backend="scipy")
        reweighter2.fit(sample_data, targets)
        result2 = reweighter2.transform(sample_data)

        np.testing.assert_array_almost_equal(
            result1["weight"].values, result2["weight"].values
        )


class TestGeographicTargeting:
    """Test geographic hierarchy handling."""

    @pytest.fixture
    def hierarchical_data(self):
        """Create data with state/county hierarchy."""
        np.random.seed(42)
        n = 1000

        # Define counties within states
        state_county_map = {
            "CA": ["Los Angeles", "San Diego", "San Francisco"],
            "NY": ["New York", "Kings", "Queens"],
        }

        states = []
        counties = []

        for _ in range(n):
            state = np.random.choice(list(state_county_map.keys()))
            county = np.random.choice(state_county_map[state])
            states.append(state)
            counties.append(county)

        return pd.DataFrame({
            "state": states,
            "county": counties,
            "income": np.random.lognormal(10, 1, n),
            "weight": np.ones(n),
        })

    def test_hierarchical_targets(self, hierarchical_data):
        """Should handle hierarchical geographic targets."""
        from microplex import Reweighter

        targets = {
            "state": {"CA": 600, "NY": 400},
            "county": {
                "Los Angeles": 300,
                "San Diego": 150,
                "San Francisco": 150,
                "New York": 200,
                "Kings": 100,
                "Queens": 100,
            },
        }

        reweighter = Reweighter()
        result = reweighter.fit_transform(hierarchical_data, targets)

        # Check state-level targets
        state_totals = result.groupby("state")["weight"].sum()
        np.testing.assert_allclose(state_totals["CA"], 600, rtol=0.01)
        np.testing.assert_allclose(state_totals["NY"], 400, rtol=0.01)

        # Check county-level targets
        county_totals = result.groupby("county")["weight"].sum()
        np.testing.assert_allclose(county_totals["Los Angeles"], 300, rtol=0.02)
        np.testing.assert_allclose(county_totals["New York"], 200, rtol=0.02)


class TestOptimizationBackends:
    """Test different optimization backends."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 200

        return pd.DataFrame({
            "state": np.random.choice(["CA", "NY"], n),
            "age_group": np.random.choice(["young", "old"], n),
            "weight": np.ones(n),
        })

    @pytest.fixture
    def targets(self):
        """Create targets."""
        return {
            "state": {"CA": 120, "NY": 80},
            "age_group": {"young": 100, "old": 100},
        }

    def test_scipy_backend(self, sample_data, targets):
        """scipy backend should work."""
        from microplex import Reweighter

        reweighter = Reweighter(backend="scipy")
        result = reweighter.fit_transform(sample_data, targets)

        state_totals = result.groupby("state")["weight"].sum()
        np.testing.assert_allclose(state_totals["CA"], 120, rtol=0.01)

    def test_cvxpy_backend_if_available(self, sample_data, targets):
        """cvxpy backend should work if installed."""
        if importlib.util.find_spec("cvxpy") is None:
            pytest.skip("cvxpy not installed")

        from microplex import Reweighter

        reweighter = Reweighter(backend="cvxpy")
        result = reweighter.fit_transform(sample_data, targets)

        state_totals = result.groupby("state")["weight"].sum()
        np.testing.assert_allclose(state_totals["CA"], 120, rtol=0.01)

    def test_backends_give_similar_results(self, sample_data, targets):
        """Different backends should give similar results."""
        if importlib.util.find_spec("cvxpy") is None:
            pytest.skip("cvxpy not installed")

        from microplex import Reweighter

        reweighter_scipy = Reweighter(backend="scipy")
        result_scipy = reweighter_scipy.fit_transform(sample_data, targets)

        reweighter_cvxpy = Reweighter(backend="cvxpy")
        result_cvxpy = reweighter_cvxpy.fit_transform(sample_data, targets)

        # Both should match targets closely
        state_scipy = result_scipy.groupby("state")["weight"].sum()
        state_cvxpy = result_cvxpy.groupby("state")["weight"].sum()

        np.testing.assert_allclose(state_scipy["CA"], state_cvxpy["CA"], rtol=0.05)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unfitted_transform_raises_error(self):
        """Transform without fit should raise error."""
        from microplex import Reweighter

        reweighter = Reweighter()
        data = pd.DataFrame({"state": ["CA"], "weight": [1.0]})

        with pytest.raises(ValueError, match="not fitted"):
            reweighter.transform(data)

    def test_mismatched_data_raises_error(self):
        """Transform with different data structure should raise error."""
        from microplex import Reweighter

        train_data = pd.DataFrame({
            "state": ["CA", "NY", "TX"],
            "weight": [1, 1, 1],
        })
        targets = {"state": {"CA": 1, "NY": 1, "TX": 1}}

        reweighter = Reweighter()
        reweighter.fit(train_data, targets)

        # Different length
        test_data = pd.DataFrame({
            "state": ["CA", "NY"],
            "weight": [1, 1],
        })

        with pytest.raises(ValueError, match="length"):
            reweighter.transform(test_data)

    def test_missing_category_in_targets(self):
        """Should handle case where data has categories not in targets."""
        from microplex import Reweighter

        data = pd.DataFrame({
            "state": ["CA", "NY", "TX", "FL"],
            "weight": [1, 1, 1, 1],
        })

        # Targets missing FL
        targets = {"state": {"CA": 1, "NY": 1, "TX": 1}}

        reweighter = Reweighter()

        # Should either raise informative error or handle gracefully
        # (implementation decision - document expected behavior)
        with pytest.raises((ValueError, KeyError)):
            reweighter.fit(data, targets)

    def test_zero_target_margin(self):
        """Should handle zero targets gracefully."""
        from microplex import Reweighter

        data = pd.DataFrame({
            "state": ["CA", "CA", "NY", "NY"],
            "weight": [1, 1, 1, 1],
        })

        targets = {"state": {"CA": 0, "NY": 4}}  # Zero target for CA

        reweighter = Reweighter()
        result = reweighter.fit_transform(data, targets)

        # CA records should get zero weight
        ca_weight = result[result["state"] == "CA"]["weight"].sum()
        ny_weight = result[result["state"] == "NY"]["weight"].sum()

        assert ca_weight < 0.01  # Effectively zero
        np.testing.assert_allclose(ny_weight, 4, rtol=0.01)


class TestSparsityComparison:
    """Test different sparsity objectives."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for sparsity comparison."""
        np.random.seed(42)
        n = 500

        return pd.DataFrame({
            "state": np.random.choice(["CA", "NY", "TX"], n),
            "weight": np.ones(n),
        })

    @pytest.fixture
    def targets(self):
        """Create targets."""
        return {"state": {"CA": 250, "NY": 150, "TX": 100}}

    def test_l0_is_sparser_than_l1(self, sample_data, targets):
        """L0 optimization should produce sparser weights than L1."""
        from microplex import Reweighter

        reweighter_l0 = Reweighter(sparsity="l0")
        result_l0 = reweighter_l0.fit_transform(sample_data, targets)

        reweighter_l1 = Reweighter(sparsity="l1")
        result_l1 = reweighter_l1.fit_transform(sample_data, targets)

        n_nonzero_l0 = (result_l0["weight"] > 1e-6).sum()
        n_nonzero_l1 = (result_l1["weight"] > 1e-6).sum()

        # L0 should use fewer or equal records (with simple margins, both may be minimal)
        assert n_nonzero_l0 <= n_nonzero_l1

        # Both should be sparse (use small fraction of total records)
        assert n_nonzero_l0 < len(sample_data) * 0.1

    def test_l2_uses_all_records(self, sample_data, targets):
        """L2 optimization should give nonzero weight to most/all records."""
        from microplex import Reweighter

        reweighter_l2 = Reweighter(sparsity="l2")
        result_l2 = reweighter_l2.fit_transform(sample_data, targets)

        n_nonzero = (result_l2["weight"] > 1e-6).sum()

        # L2 should use most records (not sparse)
        assert n_nonzero > len(sample_data) * 0.8
