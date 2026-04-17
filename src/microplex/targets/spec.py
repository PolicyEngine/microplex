"""Canonical target specification primitives for microplex."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from microplex.core import EntityType


class FilterOperator(str, Enum):
    """Supported operators for target filters."""

    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not_in"


class TargetAggregation(str, Enum):
    """Supported target aggregation modes."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"


@dataclass(frozen=True)
class TargetFilter:
    """A boolean filter over a materialized feature."""

    feature: str
    operator: FilterOperator | str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator", FilterOperator(self.operator))


@dataclass(frozen=True)
class TargetSpec:
    """Canonical representation of a calibration target."""

    name: str
    entity: EntityType | str
    value: float
    period: int | str
    measure: str | None = None
    aggregation: TargetAggregation | str = TargetAggregation.SUM
    filters: tuple[TargetFilter, ...] = ()
    tolerance: float | None = None
    source: str | None = None
    units: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entity = self.entity
        if not isinstance(entity, EntityType):
            entity = EntityType(entity)
        aggregation = self.aggregation
        if not isinstance(aggregation, TargetAggregation):
            aggregation = TargetAggregation(aggregation)

        normalized_filters = tuple(
            target_filter
            if isinstance(target_filter, TargetFilter)
            else TargetFilter(**target_filter)
            for target_filter in self.filters
        )

        if aggregation is TargetAggregation.COUNT and self.measure is not None:
            raise ValueError("Count targets must not define a measure column")

        object.__setattr__(self, "entity", entity)
        object.__setattr__(self, "aggregation", aggregation)
        object.__setattr__(self, "filters", normalized_filters)

    @property
    def required_features(self) -> tuple[str, ...]:
        """Features that must be materialized to evaluate this target."""
        features = []
        if self.measure is not None:
            features.append(self.measure)
        features.extend(target_filter.feature for target_filter in self.filters)
        ordered_unique = dict.fromkeys(features)
        return tuple(ordered_unique)


@dataclass
class TargetSet:
    """Collection helpers for canonical target specs."""

    targets: list[TargetSpec] = field(default_factory=list)

    def add(self, target: TargetSpec) -> None:
        self.targets.append(target)

    def add_many(self, targets: list[TargetSpec]) -> None:
        self.targets.extend(targets)

    def for_entity(self, entity: EntityType | str) -> list[TargetSpec]:
        entity_type = entity if isinstance(entity, EntityType) else EntityType(entity)
        return [target for target in self.targets if target.entity is entity_type]

    def for_period(self, period: int | str) -> list[TargetSpec]:
        return [target for target in self.targets if target.period == period]

    def required_features(self, entity: EntityType | str | None = None) -> tuple[str, ...]:
        relevant_targets = self.targets if entity is None else self.for_entity(entity)
        features: list[str] = []
        for target in relevant_targets:
            features.extend(target.required_features)
        ordered_unique = dict.fromkeys(features)
        return tuple(ordered_unique)
