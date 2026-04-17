"""Multi-survey fusion for microplex.

Combines CPS + PUF (and optionally other surveys) into a single
stacked dataset with missing value indicators for masked MAF training.

Example:
    >>> from microplex.fusion import FusionSynthesizer, FusionConfig
    >>> synth = FusionSynthesizer(FusionConfig(device="mps"))
    >>> synth.add_survey("cps", cps_data)
    >>> synth.add_survey("puf", puf_data)
    >>> result = synth.fit_generate(n_samples=200_000)
    >>> synthetic = result.synthetic

Or use the high-level convenience function:
    >>> from microplex.fusion import synthesize_from_surveys
    >>> result = synthesize_from_surveys(n_samples=200_000, device="mps")
"""

from .harmonize import (
    COMMON_SCHEMA,
    CPS_MAPPING,
    PUF_MAPPING,
    apply_inverse_transform,
    apply_transform,
    harmonize_surveys,
    stack_surveys,
)
from .masked_maf import MaskedMAF
from .multi_source_fusion import MultiSourceFusion
from .pipeline import (
    FusionConfig,
    FusionResult,
    FusionSynthesizer,
    load_cps_for_fusion,
    load_puf_for_fusion,
    synthesize_from_surveys,
)
from .planning import FusionPlan, VariableCoverage

__all__ = [
    # Low-level harmonization
    "harmonize_surveys",
    "stack_surveys",
    "COMMON_SCHEMA",
    "CPS_MAPPING",
    "PUF_MAPPING",
    "apply_transform",
    "apply_inverse_transform",
    # Masked MAF model
    "MaskedMAF",
    "VariableCoverage",
    "FusionPlan",
    # High-level pipeline
    "FusionConfig",
    "FusionResult",
    "FusionSynthesizer",
    "load_cps_for_fusion",
    "load_puf_for_fusion",
    "synthesize_from_surveys",
    # Impute-stack fusion
    "MultiSourceFusion",
]
