"""Tests for SparseCalibrator."""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, 'src')

from microplex.calibration import LinearConstraint, SparseCalibrator


class TestSparseCalibrator:
    """Test sparse calibration with L0 selection."""

    def test_categorical_only_achieves_target_sparsity(self):
        """With only categorical targets, should achieve exact target sparsity."""
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'state': np.random.choice(['CA', 'NY', 'TX'], n, p=[0.4, 0.35, 0.25]),
        })
        targets = {'state': {'CA': 200, 'NY': 175, 'TX': 125}}

        sc = SparseCalibrator(target_sparsity=0.5, max_iter=500)
        sc.fit(data, targets)

        # Should achieve ~50% sparsity
        assert 0.45 <= sc.get_sparsity() <= 0.55

        # Should have perfect calibration (since we keep enough records)
        val = sc.validate(data)
        assert val['max_error'] < 0.01  # < 1% error

    def test_categorical_only_high_sparsity(self):
        """High sparsity (80%) should still achieve good calibration."""
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'state': np.random.choice(['CA', 'NY', 'TX'], n, p=[0.4, 0.35, 0.25]),
        })
        targets = {'state': {'CA': 200, 'NY': 175, 'TX': 125}}

        sc = SparseCalibrator(target_sparsity=0.8, max_iter=500)
        sc.fit(data, targets)

        assert 0.75 <= sc.get_sparsity() <= 0.85
        val = sc.validate(data)
        assert val['max_error'] < 0.01

    def test_with_continuous_targets(self):
        """Continuous targets should be calibrated within tolerance."""
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'state': np.random.choice(['CA', 'NY', 'TX'], n, p=[0.4, 0.35, 0.25]),
            'income': np.random.exponential(50000, n),
        })
        targets_cat = {'state': {'CA': 200, 'NY': 175, 'TX': 125}}
        targets_cont = {'income': data['income'].sum() * 0.9}

        sc = SparseCalibrator(target_sparsity=0.5, max_iter=500)
        sc.fit(data, targets_cat, targets_cont)

        assert 0.45 <= sc.get_sparsity() <= 0.55

        val = sc.validate(data)
        # With continuous targets and 50% sparsity, allow up to 15% error
        # (This is the tradeoff: sparsity for accuracy on continuous vars)
        # Categorical targets should still be accurate
        for name, info in val['targets'].items():
            if '=' in name:  # Categorical
                assert info['error'] < 0.05, f"{name} error {info['error']:.1%} > 5%"
            else:  # Continuous
                assert info['error'] < 0.15, f"{name} error {info['error']:.1%} > 15%"

    def test_sparsity_weight_parameter(self):
        """sparsity_weight should control sparsity level."""
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'state': np.random.choice(['CA', 'NY', 'TX'], n, p=[0.4, 0.35, 0.25]),
        })
        targets = {'state': {'CA': 200, 'NY': 175, 'TX': 125}}

        # Low λ = low sparsity
        sc_low = SparseCalibrator(sparsity_weight=0.1, max_iter=500)
        sc_low.fit(data, targets)

        # High λ = high sparsity
        sc_high = SparseCalibrator(sparsity_weight=2.0, max_iter=500)
        sc_high.fit(data, targets)

        assert sc_high.get_sparsity() > sc_low.get_sparsity()

    def test_all_targets_met_at_low_sparsity(self):
        """At low sparsity (10%), should hit all targets accurately."""
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'state': np.random.choice(['CA', 'NY', 'TX'], n, p=[0.4, 0.35, 0.25]),
        })
        targets = {'state': {'CA': 400, 'NY': 350, 'TX': 250}}

        sc = SparseCalibrator(target_sparsity=0.1, max_iter=500)
        sc.fit(data, targets)

        val = sc.validate(data)

        # Each target should be met precisely
        for name, info in val['targets'].items():
            assert info['error'] < 0.01, f"{name} error = {info['error']:.1%}"

    def test_supports_explicit_linear_constraints(self):
        """SparseCalibrator should calibrate directly against explicit linear rows."""
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

        calibrator = SparseCalibrator(target_sparsity=0.0, max_iter=500)
        result = calibrator.fit_transform(
            data,
            {},
            weight_col="weight",
            linear_constraints=constraints,
        )
        validation = calibrator.validate(result)

        assert validation["converged"] is True
        assert set(validation["linear_errors"]) == {"row1", "row2"}
        assert validation["max_error"] < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
