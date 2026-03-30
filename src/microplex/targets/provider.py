"""Provider abstractions for canonical target specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from microplex.core import EntityType
from microplex.targets.spec import TargetSet, TargetSpec


@dataclass(frozen=True)
class TargetQuery:
    """Generic query parameters for loading canonical targets."""

    period: int | str | None = None
    entity: EntityType | str | None = None
    names: tuple[str, ...] = ()
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    provider_filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entity = self.entity
        if entity is not None and not isinstance(entity, EntityType):
            entity = EntityType(entity)
        object.__setattr__(self, "entity", entity)
        object.__setattr__(self, "names", tuple(self.names))


def apply_target_query(
    targets: TargetSet | list[TargetSpec],
    query: TargetQuery | None = None,
) -> TargetSet:
    """Filter a canonical target collection using generic query semantics."""
    target_set = targets if isinstance(targets, TargetSet) else TargetSet(list(targets))
    if query is None:
        return target_set

    selected: list[TargetSpec] = []
    for target in target_set.targets:
        if query.period is not None and target.period != query.period:
            continue
        if query.entity is not None and target.entity is not query.entity:
            continue
        if query.names and target.name not in query.names:
            continue
        if not _matches_metadata(target, query.metadata_filters):
            continue
        selected.append(target)
    return TargetSet(selected)


@runtime_checkable
class TargetProvider(Protocol):
    """Protocol for loading canonical target sets."""

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Return a canonical target set for the requested slice."""


@dataclass
class StaticTargetProvider:
    """A provider backed by an in-memory canonical target set."""

    target_set: TargetSet = field(default_factory=TargetSet)

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        return apply_target_query(self.target_set, query)


def _matches_metadata(target: TargetSpec, metadata_filters: dict[str, Any]) -> bool:
    for key, expected in metadata_filters.items():
        if target.metadata.get(key) != expected:
            return False
    return True
