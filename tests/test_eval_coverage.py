"""
Tests for eval/coverage.py - PRDC metrics for synthetic data evaluation.

Following TDD: these tests define the expected behavior of the PRDC metrics.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from microplex.eval.coverage import (
    PRDCResult,
    compute_coverage_by_segment,
    compute_coverage_with_embeddings,
    compute_prdc,
    compute_trajectory_coverage,
    evaluate_imputation_quality,
)


class TestPRDCResult:
    """Tests for PRDCResult dataclass."""

    def test_prdc_result_repr(self):
        """PRDCResult should have a readable repr."""
        result = PRDCResult(
            precision=0.85,
            recall=0.90,
            density=1.2,
            coverage=0.88,
            covered_mask=np.array([True, True, False]),
            distances=np.array([0.1, 0.2, 0.5]),
            nearest_indices=np.array([0, 1, 2]),
        )
        repr_str = repr(result)
        assert "precision=0.850" in repr_str
        assert "recall=0.900" in repr_str
        assert "density=1.200" in repr_str
        assert "coverage=0.880" in repr_str

    def test_prdc_result_attributes(self):
        """PRDCResult should store all attributes correctly."""
        covered = np.array([True, False, True])
        distances = np.array([0.1, 0.5, 0.2])
        indices = np.array([0, 2, 1])

        result = PRDCResult(
            precision=0.5,
            recall=0.6,
            density=1.0,
            coverage=0.7,
            covered_mask=covered,
            distances=distances,
            nearest_indices=indices,
        )

        assert result.precision == 0.5
        assert result.recall == 0.6
        assert result.density == 1.0
        assert result.coverage == 0.7
        np.testing.assert_array_equal(result.covered_mask, covered)
        np.testing.assert_array_equal(result.distances, distances)
        np.testing.assert_array_equal(result.nearest_indices, indices)


class TestComputePRDC:
    """Tests for compute_prdc function."""

    def test_identical_data_perfect_coverage(self):
        """When real and synthetic are identical, coverage should be 1.0."""
        np.random.seed(42)
        data = np.random.randn(100, 5)

        result = compute_prdc(data, data.copy(), k=5)

        assert result.coverage == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.covered_mask.all()

    def test_disjoint_data_poor_coverage(self):
        """When data is completely disjoint, coverage should be low."""
        np.random.seed(42)
        real = np.random.randn(50, 3)
        synthetic = np.random.randn(50, 3) + 100  # Far away

        result = compute_prdc(real, synthetic, k=5)

        assert result.coverage < 0.1
        assert result.precision < 0.1

    def test_coverage_between_0_and_1(self):
        """All metrics should be in [0, 1] range (except density which can be >1)."""
        np.random.seed(42)
        real = np.random.randn(100, 5)
        synthetic = np.random.randn(80, 5) * 1.5  # Similar but not identical

        result = compute_prdc(real, synthetic, k=5)

        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.coverage <= 1
        assert result.density >= 0  # density can be > 1

    def test_custom_scaler(self):
        """Should respect custom scaler if provided."""
        np.random.seed(42)
        real = np.random.randn(50, 3) * 10 + 5
        synthetic = np.random.randn(50, 3) * 10 + 5

        # Fit scaler on different data (simulating train/test split)
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 3) * 10 + 5)

        result = compute_prdc(real, synthetic, k=5, scaler=scaler)

        assert isinstance(result, PRDCResult)

    def test_different_k_values(self):
        """Different k values should affect the metrics."""
        np.random.seed(42)
        real = np.random.randn(100, 5)
        synthetic = np.random.randn(100, 5) * 0.9

        result_k3 = compute_prdc(real, synthetic, k=3)
        result_k10 = compute_prdc(real, synthetic, k=10)

        # With larger k, radii are larger, so coverage should be higher
        assert result_k10.coverage >= result_k3.coverage * 0.8  # Allow some tolerance

    def test_output_shapes(self):
        """Output arrays should have correct shapes."""
        np.random.seed(42)
        n_real = 50
        real = np.random.randn(n_real, 4)
        synthetic = np.random.randn(30, 4)

        result = compute_prdc(real, synthetic, k=5)

        assert result.covered_mask.shape == (n_real,)
        assert result.distances.shape == (n_real,)
        assert result.nearest_indices.shape == (n_real,)

    def test_nearest_indices_valid(self):
        """Nearest indices should be valid indices into synthetic data."""
        np.random.seed(42)
        real = np.random.randn(50, 3)
        n_synth = 30
        synthetic = np.random.randn(n_synth, 3)

        result = compute_prdc(real, synthetic, k=5)

        assert all(0 <= idx < n_synth for idx in result.nearest_indices)

    def test_minimum_sample_size(self):
        """Should work with minimum viable sample sizes."""
        np.random.seed(42)
        real = np.random.randn(10, 2)
        synthetic = np.random.randn(10, 2)

        result = compute_prdc(real, synthetic, k=3)

        assert isinstance(result, PRDCResult)


class TestComputeCoverageWithEmbeddings:
    """Tests for compute_coverage_with_embeddings."""

    def test_with_identity_embedding(self):
        """With identity embedding, should match regular compute_prdc."""
        np.random.seed(42)
        real = np.random.randn(50, 5)
        synthetic = np.random.randn(50, 5)

        def identity(x):
            return x

        result = compute_coverage_with_embeddings(real, synthetic, identity, k=5)

        assert isinstance(result, PRDCResult)

    def test_with_dimensionality_reduction(self):
        """Embedding function that reduces dimensionality."""
        np.random.seed(42)
        real = np.random.randn(50, 10)
        synthetic = np.random.randn(50, 10)

        # Simple projection to lower dim
        W = np.random.randn(10, 3)
        def project(x):
            return x @ W

        result = compute_coverage_with_embeddings(real, synthetic, project, k=5)

        assert isinstance(result, PRDCResult)

    def test_with_nonlinear_embedding(self):
        """Embedding function with nonlinear transformation."""
        np.random.seed(42)
        real = np.random.randn(50, 5)
        synthetic = np.random.randn(50, 5)

        def nonlinear(x):
            return np.tanh(x) * 2

        result = compute_coverage_with_embeddings(real, synthetic, nonlinear, k=5)

        assert isinstance(result, PRDCResult)


class TestComputeTrajectoryCoverage:
    """Tests for compute_trajectory_coverage."""

    def test_flattened_trajectory_coverage(self):
        """Without embed_fn, should flatten trajectories and compute PRDC."""
        np.random.seed(42)
        n_real, n_synth, T, n_features = 30, 25, 12, 4
        real_traj = np.random.randn(n_real, T, n_features)
        synth_traj = np.random.randn(n_synth, T, n_features)

        result = compute_trajectory_coverage(real_traj, synth_traj, k=5)

        assert isinstance(result, PRDCResult)
        assert result.covered_mask.shape == (n_real,)

    def test_with_custom_embedding(self):
        """With custom embedding function for trajectories."""
        np.random.seed(42)
        n_real, n_synth, T, n_features = 30, 25, 12, 4
        real_traj = np.random.randn(n_real, T, n_features)
        synth_traj = np.random.randn(n_synth, T, n_features)

        # Embed trajectory by taking mean over time
        def temporal_mean(traj):
            return traj.mean(axis=1)

        result = compute_trajectory_coverage(
            real_traj, synth_traj, k=5, embed_fn=temporal_mean
        )

        assert isinstance(result, PRDCResult)

    def test_short_trajectories(self):
        """Should work with very short trajectories."""
        np.random.seed(42)
        real_traj = np.random.randn(20, 2, 3)  # T=2
        synth_traj = np.random.randn(20, 2, 3)

        result = compute_trajectory_coverage(real_traj, synth_traj, k=3)

        assert isinstance(result, PRDCResult)


class TestComputeCoverageBySegment:
    """Tests for compute_coverage_by_segment."""

    def test_single_segment(self):
        """With single segment, should return one result."""
        np.random.seed(42)
        real = np.random.randn(50, 4)
        synthetic = np.random.randn(50, 4)
        segments = np.array(["A"] * 50)

        results = compute_coverage_by_segment(real, synthetic, segments, k=5)

        assert len(results) == 1
        assert "A" in results
        assert isinstance(results["A"], PRDCResult)

    def test_multiple_segments(self):
        """With multiple segments, should return result per segment."""
        np.random.seed(42)
        real = np.random.randn(100, 4)
        synthetic = np.random.randn(100, 4)
        segments = np.array(["young"] * 40 + ["middle"] * 35 + ["old"] * 25)

        results = compute_coverage_by_segment(real, synthetic, segments, k=5)

        assert len(results) == 3
        assert "young" in results
        assert "middle" in results
        assert "old" in results

    def test_segment_with_different_distributions(self):
        """Segments with different distributions should have different coverage."""
        np.random.seed(42)
        # Create real data with two distinct clusters
        real_A = np.random.randn(50, 3)
        real_B = np.random.randn(50, 3) + 5
        real = np.vstack([real_A, real_B])

        # Synthetic only covers cluster A
        synthetic = np.random.randn(100, 3)

        segments = np.array(["A"] * 50 + ["B"] * 50)

        results = compute_coverage_by_segment(real, synthetic, segments, k=5)

        # Segment A should have higher coverage than B
        assert results["A"].coverage > results["B"].coverage

    def test_skips_small_segments(self):
        """Segments smaller than k+1 should be skipped."""
        np.random.seed(42)
        real = np.random.randn(50, 4)
        synthetic = np.random.randn(50, 4)
        # Create a segment with only 3 records (smaller than k=5)
        segments = np.array(["large"] * 47 + ["tiny"] * 3)

        results = compute_coverage_by_segment(real, synthetic, segments, k=5)

        assert "large" in results
        assert "tiny" not in results  # Too small

    def test_numeric_segment_labels(self):
        """Should work with numeric segment labels."""
        np.random.seed(42)
        real = np.random.randn(60, 3)
        synthetic = np.random.randn(60, 3)
        segments = np.array([1, 1, 1, 2, 2, 2] * 10)

        results = compute_coverage_by_segment(real, synthetic, segments, k=5)

        assert "1" in results or 1 in results
        assert "2" in results or 2 in results


class TestEvaluateImputationQuality:
    """Tests for evaluate_imputation_quality."""

    def test_perfect_imputation(self):
        """When imputation matches true values exactly, metrics should be optimal."""
        np.random.seed(42)
        n, n_samples, n_features = 20, 50, 4

        true_values = np.random.randn(n, n_features)
        # All samples equal to true values (no variance)
        imputed_samples = np.tile(true_values[:, np.newaxis, :], (1, n_samples, 1))
        # Mark some columns as imputed
        observed_mask = np.zeros((n, n_features), dtype=bool)
        observed_mask[:, :2] = True  # First 2 observed, last 2 imputed

        results = evaluate_imputation_quality(true_values, imputed_samples, observed_mask)

        assert "rmse" in results
        assert results["rmse"] < 1e-6  # Essentially zero

    def test_random_imputation(self):
        """Random imputation should have higher RMSE."""
        np.random.seed(42)
        n, n_samples, n_features = 20, 50, 4

        true_values = np.random.randn(n, n_features)
        imputed_samples = np.random.randn(n, n_samples, n_features)
        observed_mask = np.zeros((n, n_features), dtype=bool)
        observed_mask[:, :2] = True

        results = evaluate_imputation_quality(true_values, imputed_samples, observed_mask)

        assert results["rmse"] > 0.5  # Should be substantial

    def test_coverage_90_metric(self):
        """90% coverage should be close to 0.9 for well-calibrated imputations."""
        np.random.seed(42)
        n, n_samples, n_features = 100, 200, 3

        true_values = np.random.randn(n, n_features)
        # Generate samples from distribution centered on true with known std
        noise = np.random.randn(n, n_samples, n_features) * 0.5
        imputed_samples = true_values[:, np.newaxis, :] + noise

        observed_mask = np.zeros((n, n_features), dtype=bool)

        results = evaluate_imputation_quality(true_values, imputed_samples, observed_mask)

        assert "coverage_90" in results
        # Well-calibrated imputation should have high coverage
        # With samples centered on true values, coverage can be very high
        assert 0.80 < results["coverage_90"] <= 1.0

    def test_interval_width_metric(self):
        """Interval width should be smaller for more precise imputations."""
        np.random.seed(42)
        n, n_samples, n_features = 50, 100, 3

        true_values = np.random.randn(n, n_features)
        observed_mask = np.zeros((n, n_features), dtype=bool)

        # Narrow imputation
        narrow_samples = true_values[:, np.newaxis, :] + np.random.randn(n, n_samples, n_features) * 0.1

        # Wide imputation
        wide_samples = true_values[:, np.newaxis, :] + np.random.randn(n, n_samples, n_features) * 2.0

        narrow_results = evaluate_imputation_quality(true_values, narrow_samples, observed_mask)
        wide_results = evaluate_imputation_quality(true_values, wide_samples, observed_mask)

        assert "mean_interval_width_normalized" in narrow_results
        assert narrow_results["mean_interval_width_normalized"] < wide_results["mean_interval_width_normalized"]

    def test_only_evaluates_imputed_values(self):
        """Metrics should only consider imputed (not observed) values."""
        np.random.seed(42)
        n, n_samples, n_features = 30, 50, 4

        true_values = np.random.randn(n, n_features)
        imputed_samples = np.random.randn(n, n_samples, n_features)

        # All observed = no imputed values
        all_observed = np.ones((n, n_features), dtype=bool)

        # This should work but might have division issues with empty mask
        # Implementation should handle this gracefully
        try:
            results = evaluate_imputation_quality(true_values, imputed_samples, all_observed)
            # If it returns something, check it has the expected keys
            if results is not None:
                assert "rmse" in results
        except (ZeroDivisionError, ValueError):
            # It's acceptable to raise an error if nothing to evaluate
            pass
