"""
Evaluation metrics for synthetic data quality.

Includes PRDC (Precision, Recall, Density, Coverage) metrics,
imputation quality metrics, and the unified evaluation harness.
"""

from .coverage import (
    PRDCResult,
    compute_prdc,
    compute_coverage_with_embeddings,
    compute_trajectory_coverage,
    compute_coverage_by_segment,
    evaluate_imputation_quality,
)
from .harness import (
    EvalHarness,
    SynthesisEvalResult,
    ReweightingEvalResult,
    SourceCoverage,
    AggregateError,
)

__all__ = [
    "PRDCResult",
    "compute_prdc",
    "compute_coverage_with_embeddings",
    "compute_trajectory_coverage",
    "compute_coverage_by_segment",
    "evaluate_imputation_quality",
    "EvalHarness",
    "SynthesisEvalResult",
    "ReweightingEvalResult",
    "SourceCoverage",
    "AggregateError",
]
