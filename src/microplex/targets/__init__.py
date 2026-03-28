"""Target primitives for microplex."""

from microplex.targets.database import Target, TargetCategory, TargetsDatabase
from microplex.targets.provider import (
    StaticTargetProvider,
    TargetProvider,
    TargetQuery,
    apply_target_query,
)
from microplex.targets.spec import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetSet,
    TargetSpec,
)

__all__ = [
    "TargetsDatabase",
    "Target",
    "TargetCategory",
    "FilterOperator",
    "TargetAggregation",
    "TargetFilter",
    "TargetProvider",
    "TargetQuery",
    "StaticTargetProvider",
    "apply_target_query",
    "TargetSet",
    "TargetSpec",
]
