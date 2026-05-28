"""
microplex: Microdata synthesis and reweighting using normalizing flows.

A country-agnostic library for creating rich, calibrated microdata through:
- Conditional synthesis (demographics → outcomes)
- Reweighting to population targets
- Zero-inflated distributions (common in economic/health data)
- Joint correlations between variables
- Hierarchical structures (households, firms, etc.)
- Longitudinal / panel synthesis with trajectory models

Country-specific primitives (CPS, PUF, SOI, SSA life tables, census GEOIDs,
PolicyEngine-US parity helpers) live in country-pack packages such as
`microplex-us` and are not re-exported here.

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
    SparseCalibrator,
)
from microplex.core import (
    DataType,
    Entity,
    EntityType,
    Family,
    FilingStatus,
    HardConcreteGate,
    Household,
    LegalReference,
    Period,
    PeriodType,
    Person,
    Record,
    RecordType,
    ResolutionConfig,
    ResolutionLevel,
    SPMUnit,
    TaxUnit,
    Variable,
    VariableRegistry,
    VariableRole,
    compress_dataset,
    for_api,
    for_browser,
    for_research,
)
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
from microplex.fusion import (
    FusionConfig,
    FusionResult,
    FusionSynthesizer,
    MaskedMAF,
    synthesize_from_surveys,
)
from microplex.hierarchical import (
    HierarchicalSynthesizer,
    HouseholdSchema,
)
from microplex.reweighting import Reweighter
from microplex.statmatch_backend import (
    HAS_STATMATCH,
    StatMatchSynthesizer,
    create_synthesizer,
)
from microplex.synthesizer import Synthesizer
from microplex.tax_units import (
    PreservedTaxUnitTables,
    build_preserved_tax_unit_tables,
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

# Default sparse calibrator: Cross-Category + IPF achieves exact target matching
# with controllable sparsity. HardConcreteCalibrator available for differentiable
# pipelines or custom loss functions.
DefaultSparseCalibrator = SparseCalibrator

__version__ = "0.2.0"

__all__ = [
    # Core synthesis
    "Synthesizer",
    "HierarchicalSynthesizer",
    "HouseholdSchema",
    "PreservedTaxUnitTables",
    "build_preserved_tax_unit_tables",
    # Calibration
    "Reweighter",
    "Calibrator",
    "SparseCalibrator",
    "HardConcreteCalibrator",
    "DefaultSparseCalibrator",
    # Statistical matching (optional backend)
    "StatMatchSynthesizer",
    "create_synthesizer",
    "HAS_STATMATCH",
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
    # Core entities
    "EntityType",
    "FilingStatus",
    "RecordType",
    "Entity",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "SPMUnit",
    "Record",
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
