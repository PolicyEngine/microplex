"""
microplex: Microdata synthesis and reweighting using normalizing flows.

A library for creating rich, calibrated microdata through:
- Conditional synthesis (demographics → outcomes)
- Reweighting to population targets
- Zero-inflated distributions (common in economic/health data)
- Joint correlations between variables
- Hierarchical structures (households, firms, etc.)

Example:
    >>> from microplex import Synthesizer
    >>> synth = Synthesizer(
    ...     target_vars=["income", "expenditure"],
    ...     condition_vars=["age", "education"],
    ... )
    >>> synth.fit(training_data)
    >>> synthetic = synth.generate(new_demographics)
"""

from microplex.calibration import (
    Calibrator,
    HardConcreteCalibrator,
    LinearConstraint,
    SparseCalibrator,
)
from microplex.reweighting import Reweighter
from microplex.synthesizer import Synthesizer

# Default sparse calibrator: Cross-Category + IPF achieves exact target matching
# with controllable sparsity. HardConcreteCalibrator available for differentiable
# pipelines or custom loss functions.
DefaultSparseCalibrator = SparseCalibrator

# Core data models (from cosilico-microdata merge)
from microplex.core import (
    # Variables
    BenefitUnit,
    DataType,
    Entity,
    # Entities
    EntityType,
    Family,
    FilingStatus,
    HardConcreteGate,
    Household,
    LegalReference,
    Period,
    # Periods
    PeriodType,
    Person,
    Record,
    RecordType,
    ResolutionConfig,
    # Resolution
    ResolutionLevel,
    SourceArchetype,
    SourceColumnManifest,
    SourceColumnValueType,
    SourceManifest,
    SourceObservationManifest,
    SourceProvider,
    SourceQuery,
    SPMUnit,
    StaticSourceProvider,
    TaxUnit,
    Variable,
    VariableRegistry,
    VariableRole,
    apply_source_query,
    compress_dataset,
    for_api,
    for_browser,
    for_research,
    load_source_manifest,
)

# Multi-source DGP (Data Generating Process)
from microplex.dgp import (
    EvalResult,
    PopulationDGP,
    Survey,
    run_multi_source_benchmark,
)
from microplex.discrete import (
    BinaryModel,
    CategoricalModel,
    DiscreteModelCollection,
)
from microplex.flows import MADE, AffineCouplingLayer, ConditionalMAF

# Fusion (multi-survey synthesis)
from microplex.fusion import (
    COMMON_SCHEMA,
    FusionConfig,
    FusionResult,
    FusionSynthesizer,
    MaskedMAF,
    harmonize_surveys,
    stack_surveys,
    synthesize_from_surveys,
)
from microplex.geography import (
    AtomicGeographyCrosswalk,
    GeographyAssignmentPlan,
    GeographyProvider,
    GeographyQuery,
    ProbabilisticAtomicGeographyAssigner,
    StaticGeographyProvider,
    materialize_geographies,
    nearest_numeric_partition_key,
)
from microplex.hierarchical import (
    HierarchicalSynthesizer,
    HouseholdSchema,
)
from microplex.statmatch_backend import (
    HAS_STATMATCH,
    StatMatchSynthesizer,
    create_synthesizer,
)
from microplex.targets import (
    BatchBenchmarkResultEvaluator,
    BenchmarkArtifactValidationResult,
    BenchmarkComparison,
    BenchmarkResult,
    BenchmarkResultEvaluator,
    BenchmarkSliceComparison,
    BenchmarkSliceSpec,
    BenchmarkSuiteResult,
    EntityTableBinding,
    EntityTableBundle,
    EntityTableBundleReweightingResult,
    FilterOperator,
    MetricComparisonSummary,
    SparseTargetCalibrationDiagnostics,
    SparseTargetConstraint,
    StaticTargetProvider,
    TargetAggregation,
    TargetConstraintCompilationResult,
    TargetDelta,
    TargetFilter,
    TargetGroupSummary,
    TargetMetric,
    TargetProvider,
    TargetQuery,
    TargetReweightingConstraint,
    TargetReweightingDiagnostics,
    TargetSet,
    TargetSpec,
    UnsupportedTarget,
    apply_target_query,
    assert_valid_benchmark_artifact_manifest,
    build_benchmark_result,
    build_benchmark_suite_from_payloads,
    build_benchmark_suite_from_results,
    build_benchmark_suite_result,
    build_grouped_summaries,
    build_target_mask,
    calibrate_sparse_target_weights,
    compare_benchmark_results,
    compare_metric_sets,
    compile_entity_table_bundle_target_constraints,
    compile_sparse_target_constraints,
    compile_target_reweighting_constraints,
    constraint_abs_relative_error,
    evaluate_benchmark_slice_payloads,
    evaluate_benchmark_slice_results,
    filter_nonempty_benchmark_slices,
    load_benchmark_slice_target_sets,
    max_abs_relative_error,
    mean_abs_relative_error,
    normalize_metric_payload,
    relative_error_ratio,
    reweight_entity_table_bundle_targets,
    reweight_to_target_constraints,
    sparse_constraint_abs_rel_error,
    union_target_sets,
    validate_benchmark_artifact_manifest,
)
from microplex.targets import (
    apply_filter as apply_target_filter,
)
from microplex.targets import (
    numeric_series as target_numeric_series,
)
from microplex.transforms import (
    LogTransform,
    MultiVariableTransformer,
    Standardizer,
    VariableTransformer,
    ZeroInflatedTransform,
)
from microplex.transitions import (
    DisabilityOnset,
    DisabilityRecovery,
    DisabilityTransitionModel,
    DivorceTransition,
    MarriageTransition,
    Mortality,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "Synthesizer",
    "Reweighter",
    "Calibrator",
    "LinearConstraint",
    "SparseCalibrator",
    "HardConcreteCalibrator",
    "DefaultSparseCalibrator",
    # Statistical matching (optional backend)
    "StatMatchSynthesizer",
    "create_synthesizer",
    "HAS_STATMATCH",
    # Hierarchical
    "HierarchicalSynthesizer",
    "HouseholdSchema",
    # Geography
    "AtomicGeographyCrosswalk",
    "GeographyAssignmentPlan",
    "GeographyQuery",
    "GeographyProvider",
    "ProbabilisticAtomicGeographyAssigner",
    "StaticGeographyProvider",
    "materialize_geographies",
    "nearest_numeric_partition_key",
    # Targets
    "FilterOperator",
    "TargetAggregation",
    "TargetFilter",
    "TargetProvider",
    "TargetQuery",
    "StaticTargetProvider",
    "apply_target_query",
    "BenchmarkArtifactValidationResult",
    "TargetSet",
    "TargetSpec",
    "UnsupportedTarget",
    "BenchmarkResult",
    "BenchmarkComparison",
    "BenchmarkSliceSpec",
    "BenchmarkSliceComparison",
    "BenchmarkSuiteResult",
    "BenchmarkResultEvaluator",
    "BatchBenchmarkResultEvaluator",
    "TargetMetric",
    "TargetDelta",
    "TargetGroupSummary",
    "MetricComparisonSummary",
    "normalize_metric_payload",
    "relative_error_ratio",
    "mean_abs_relative_error",
    "max_abs_relative_error",
    "validate_benchmark_artifact_manifest",
    "assert_valid_benchmark_artifact_manifest",
    "build_benchmark_result",
    "evaluate_benchmark_slice_results",
    "evaluate_benchmark_slice_payloads",
    "load_benchmark_slice_target_sets",
    "filter_nonempty_benchmark_slices",
    "union_target_sets",
    "build_benchmark_suite_from_results",
    "build_benchmark_suite_from_payloads",
    "build_benchmark_suite_result",
    "compare_benchmark_results",
    "compare_metric_sets",
    "build_grouped_summaries",
    "EntityTableBinding",
    "EntityTableBundle",
    "EntityTableBundleReweightingResult",
    "SparseTargetConstraint",
    "SparseTargetCalibrationDiagnostics",
    "TargetReweightingConstraint",
    "TargetReweightingDiagnostics",
    "TargetConstraintCompilationResult",
    "compile_entity_table_bundle_target_constraints",
    "compile_sparse_target_constraints",
    "calibrate_sparse_target_weights",
    "sparse_constraint_abs_rel_error",
    "compile_target_reweighting_constraints",
    "reweight_entity_table_bundle_targets",
    "reweight_to_target_constraints",
    "constraint_abs_relative_error",
    "build_target_mask",
    "apply_target_filter",
    "target_numeric_series",
    # Transforms
    "ZeroInflatedTransform",
    "LogTransform",
    "Standardizer",
    "VariableTransformer",
    "MultiVariableTransformer",
    # Flows
    "ConditionalMAF",
    "MADE",
    "AffineCouplingLayer",
    # Discrete
    "BinaryModel",
    "CategoricalModel",
    "DiscreteModelCollection",
    # Transitions
    "Mortality",
    "DisabilityOnset",
    "DisabilityRecovery",
    "DisabilityTransitionModel",
    "MarriageTransition",
    "DivorceTransition",
    # Core entities (from cosilico-microdata)
    "EntityType",
    "FilingStatus",
    "RecordType",
    "Entity",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "BenefitUnit",
    "SPMUnit",
    "Record",
    "SourceQuery",
    "SourceArchetype",
    "SourceColumnValueType",
    "SourceColumnManifest",
    "SourceObservationManifest",
    "SourceManifest",
    "SourceProvider",
    "StaticSourceProvider",
    "apply_source_query",
    "load_source_manifest",
    # Variables
    "DataType",
    "VariableRole",
    "LegalReference",
    "Variable",
    "VariableRegistry",
    # Periods
    "PeriodType",
    "Period",
    # Resolution
    "ResolutionLevel",
    "ResolutionConfig",
    "HardConcreteGate",
    "compress_dataset",
    "for_browser",
    "for_api",
    "for_research",
    # Fusion (multi-survey synthesis)
    "harmonize_surveys",
    "stack_surveys",
    "COMMON_SCHEMA",
    "MaskedMAF",
    "FusionConfig",
    "FusionResult",
    "FusionSynthesizer",
    "synthesize_from_surveys",
    # Multi-source DGP
    "PopulationDGP",
    "Survey",
    "EvalResult",
    "run_multi_source_benchmark",
]
