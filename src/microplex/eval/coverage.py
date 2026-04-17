"""
Coverage evaluation metrics for synthetic data.

Implements PRDC (Precision, Recall, Density, Coverage) metrics
with support for both raw feature space and learned embeddings.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Callable, Tuple
from dataclasses import dataclass


@dataclass
class PRDCResult:
    """Results from PRDC computation."""
    precision: float      # Fraction of synthetic near real
    recall: float         # Fraction of real near synthetic
    density: float        # Average density of real near synthetic
    coverage: float       # Fraction of real "covered" by synthetic

    # Per-record details
    covered_mask: np.ndarray      # Which real records are covered
    distances: np.ndarray         # Distance to nearest synthetic
    nearest_indices: np.ndarray   # Index of nearest synthetic

    def __repr__(self):
        return (
            f"PRDC(precision={self.precision:.3f}, recall={self.recall:.3f}, "
            f"density={self.density:.3f}, coverage={self.coverage:.3f})"
        )


def compute_prdc(
    real: np.ndarray,
    synthetic: np.ndarray,
    k: int = 5,
    scaler: Optional[StandardScaler] = None,
) -> PRDCResult:
    """
    Compute Precision, Recall, Density, Coverage metrics.

    Delegates the four scalar metrics to the canonical ``prdc`` library
    (Naeem et al. 2020) and additionally computes per-record detail arrays
    (covered_mask, distances, nearest_indices) used by downstream code.

    Args:
        real: (n_real, n_features) real data
        synthetic: (n_synth, n_features) synthetic data
        k: Number of neighbors for radius computation
        scaler: Optional scaler. If None, fits StandardScaler on real.

    Returns:
        PRDCResult with all metrics and per-record arrays
    """
    from prdc import compute_prdc as _prdc

    # Scale data
    if scaler is None:
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real)
    else:
        real_scaled = scaler.transform(real)
    synth_scaled = scaler.transform(synthetic)

    # Canonical PRDC metrics
    metrics = _prdc(real_scaled, synth_scaled, nearest_k=k)

    # Per-record detail arrays (not provided by the prdc library)
    # Distance from each real point to its nearest synthetic neighbour,
    # plus the real manifold radii needed for the covered_mask.
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real_scaled)
    real_dists, _ = nn_real.kneighbors(real_scaled)
    real_radii = real_dists[:, -1]

    nn_synth_1 = NearestNeighbors(n_neighbors=1).fit(synth_scaled)
    real_to_synth, nearest_synth = nn_synth_1.kneighbors(real_scaled)
    real_to_synth = real_to_synth[:, 0]
    nearest_synth = nearest_synth[:, 0]

    covered = real_to_synth <= real_radii

    return PRDCResult(
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        density=float(metrics["density"]),
        coverage=float(metrics["coverage"]),
        covered_mask=covered,
        distances=real_to_synth,
        nearest_indices=nearest_synth,
    )


def compute_coverage_with_embeddings(
    real: np.ndarray,
    synthetic: np.ndarray,
    embed_fn: Callable[[np.ndarray], np.ndarray],
    k: int = 5,
) -> PRDCResult:
    """
    Compute coverage in a learned embedding space.

    Args:
        real: (n_real, ...) real data (any shape)
        synthetic: (n_synth, ...) synthetic data
        embed_fn: Function that maps data to embeddings
        k: Number of neighbors

    Returns:
        PRDCResult computed in embedding space
    """
    real_emb = embed_fn(real)
    synth_emb = embed_fn(synthetic)

    return compute_prdc(real_emb, synth_emb, k=k)


def compute_trajectory_coverage(
    real_trajectories: np.ndarray,
    synthetic_trajectories: np.ndarray,
    k: int = 5,
    embed_fn: Optional[Callable] = None,
) -> PRDCResult:
    """
    Compute coverage for trajectory data.

    Flattens trajectories to vectors unless embed_fn provided.

    Args:
        real_trajectories: (n_real, T, n_features)
        synthetic_trajectories: (n_synth, T, n_features)
        k: Number of neighbors
        embed_fn: Optional embedding function

    Returns:
        PRDCResult
    """
    if embed_fn is not None:
        return compute_coverage_with_embeddings(
            real_trajectories, synthetic_trajectories, embed_fn, k
        )

    # Flatten trajectories
    n_real = len(real_trajectories)
    n_synth = len(synthetic_trajectories)

    real_flat = real_trajectories.reshape(n_real, -1)
    synth_flat = synthetic_trajectories.reshape(n_synth, -1)

    return compute_prdc(real_flat, synth_flat, k=k)


def compute_coverage_by_segment(
    real: np.ndarray,
    synthetic: np.ndarray,
    segment_col: np.ndarray,
    k: int = 5,
) -> Dict[str, PRDCResult]:
    """
    Compute coverage broken down by segment.

    Useful for understanding where coverage is good/bad
    (e.g., by age group, wealth quintile).

    Args:
        real: (n_real, n_features)
        synthetic: (n_synth, n_features)
        segment_col: (n_real,) segment labels for real data
        k: Number of neighbors

    Returns:
        Dict mapping segment -> PRDCResult
    """
    results = {}

    for segment in np.unique(segment_col):
        mask = segment_col == segment
        real_segment = real[mask]

        if len(real_segment) >= k + 1:
            # Use full synthetic but only score real segment
            result = compute_prdc(real_segment, synthetic, k=min(k, len(real_segment) - 1))
            results[str(segment)] = result

    return results


def evaluate_imputation_quality(
    true_values: np.ndarray,
    imputed_samples: np.ndarray,
    observed_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate quality of imputation / conditional generation.

    Args:
        true_values: (n, n_features) ground truth
        imputed_samples: (n, n_samples, n_features) multiple imputations
        observed_mask: (n, n_features) True = was observed (not imputed)

    Returns:
        Dict with RMSE, coverage, interval width metrics
    """
    n, n_samples, n_features = imputed_samples.shape
    imputed_mask = ~observed_mask

    results = {}

    # Mean imputation RMSE
    mean_imputed = imputed_samples.mean(axis=1)
    rmse = np.sqrt(((true_values - mean_imputed) ** 2 * imputed_mask).sum() / imputed_mask.sum())
    results['rmse'] = rmse

    # 90% interval coverage
    lower = np.percentile(imputed_samples, 5, axis=1)
    upper = np.percentile(imputed_samples, 95, axis=1)
    in_interval = (true_values >= lower) & (true_values <= upper)
    coverage_90 = (in_interval * imputed_mask).sum() / imputed_mask.sum()
    results['coverage_90'] = coverage_90

    # Mean interval width (normalized by std of true values)
    interval_width = (upper - lower) * imputed_mask
    true_std = np.std(true_values, axis=0, keepdims=True)
    normalized_width = (interval_width / (true_std + 1e-8)).sum() / imputed_mask.sum()
    results['mean_interval_width_normalized'] = normalized_width

    return results
