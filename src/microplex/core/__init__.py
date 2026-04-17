"""
Core data models for microplex.

This module provides the foundational data structures for microdata representation:
- Entity types (Person, TaxUnit, Household, Family, BenefitUnit, SPMUnit, Record)
- Variable definitions with legal references
- Period arithmetic
- Multi-resolution dataset generation
"""

from microplex.core.entities import (
    BenefitUnit,
    Entity,
    EntityType,
    Family,
    FilingStatus,
    Household,
    Person,
    Record,
    RecordType,
    SPMUnit,
    TaxUnit,
)
from microplex.core.periods import (
    Period,
    PeriodType,
)
from microplex.core.resolution import (
    HardConcreteGate,
    ResolutionConfig,
    ResolutionLevel,
    compress_dataset,
    for_api,
    for_browser,
    for_research,
)
from microplex.core.source_manifests import (
    SourceColumnManifest,
    SourceColumnValueType,
    SourceManifest,
    SourceObservationManifest,
    load_source_manifest,
)
from microplex.core.sources import (
    EntityObservation,
    EntityRelationship,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceAdapter,
    SourceArchetype,
    SourceDescriptor,
    SourceProvider,
    SourceQuery,
    SourceVariableCapability,
    StaticSourceProvider,
    TimeStructure,
    apply_source_query,
)
from microplex.core.variables import (
    DataType,
    LegalReference,
    Variable,
    VariableRegistry,
    VariableRole,
)

__all__ = [
    # Entities
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
    # Sources
    "TimeStructure",
    "Shareability",
    "SourceArchetype",
    "RelationshipCardinality",
    "EntityObservation",
    "SourceVariableCapability",
    "SourceDescriptor",
    "EntityRelationship",
    "ObservationFrame",
    "SourceQuery",
    "SourceProvider",
    "StaticSourceProvider",
    "apply_source_query",
    "SourceAdapter",
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
    "SourceColumnValueType",
    "SourceColumnManifest",
    "SourceObservationManifest",
    "SourceManifest",
    "load_source_manifest",
    "compress_dataset",
    "for_browser",
    "for_api",
    "for_research",
]
