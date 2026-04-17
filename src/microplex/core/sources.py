"""Source and observation metadata for multientity fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from microplex.core.entities import EntityType


class TimeStructure(Enum):
    """How observations are distributed across time."""

    CROSS_SECTION = "cross_section"
    REPEATED_CROSS_SECTION = "repeated_cross_section"
    PANEL = "panel"
    EVENT_HISTORY = "event_history"


class Shareability(Enum):
    """Whether source microdata can appear directly in released artifacts."""

    PUBLIC = "public"
    RESTRICTED = "restricted"
    NON_SHAREABLE = "non_shareable"

    @property
    def allows_direct_release(self) -> bool:
        """Whether this source can be directly represented in public outputs."""
        return self is Shareability.PUBLIC


class SourceArchetype(Enum):
    """Cross-country source role used for planning analogous survey families."""

    HOUSEHOLD_INCOME = "household_income"
    TAX_MICRODATA = "tax_microdata"
    WEALTH = "wealth"
    CONSUMPTION = "consumption"
    LONGITUDINAL_SOCIOECONOMIC = "longitudinal_socioeconomic"


class RelationshipCardinality(Enum):
    """Cardinality from parent entity to child entity."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass(frozen=True)
class SourceVariableCapability:
    """How one source variable should be used during fusion and imputation."""

    authoritative: bool = True
    usable_as_condition: bool = True
    notes: str | None = None


@dataclass(frozen=True)
class EntityObservation:
    """Observed variables for one entity within a source."""

    entity: EntityType
    key_column: str
    variable_names: tuple[str, ...]
    weight_column: str | None = None
    period_column: str | None = None

    def __post_init__(self) -> None:
        if not self.key_column:
            raise ValueError("EntityObservation.key_column must be non-empty")
        if not self.variable_names:
            raise ValueError("EntityObservation.variable_names must be non-empty")
        if len(set(self.variable_names)) != len(self.variable_names):
            raise ValueError(
                f"Duplicate variables declared for entity '{self.entity.value}'"
            )


@dataclass(frozen=True)
class SourceDescriptor:
    """Metadata describing one source as a partial view of the population."""

    name: str
    shareability: Shareability
    time_structure: TimeStructure
    observations: tuple[EntityObservation, ...]
    archetype: SourceArchetype | None = None
    population: str | None = None
    description: str | None = None
    variable_capabilities: Mapping[str, SourceVariableCapability] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SourceDescriptor.name must be non-empty")
        if not self.observations:
            raise ValueError("SourceDescriptor.observations must be non-empty")
        entities = [observation.entity for observation in self.observations]
        if len(set(entities)) != len(entities):
            raise ValueError(
                f"Source '{self.name}' declares the same entity more than once"
            )
        unknown_capabilities = set(self.variable_capabilities) - self.all_variable_names
        if unknown_capabilities:
            missing = ", ".join(sorted(unknown_capabilities))
            raise ValueError(
                f"Source '{self.name}' declares capabilities for unknown variables: {missing}"
            )

    @property
    def observed_entities(self) -> tuple[EntityType, ...]:
        """Entities observed by this source."""
        return tuple(observation.entity for observation in self.observations)

    @property
    def all_variable_names(self) -> frozenset[str]:
        """All variables observed across every entity table."""
        return frozenset(
            variable
            for observation in self.observations
            for variable in observation.variable_names
        )

    def observation_for(self, entity: EntityType) -> EntityObservation:
        """Return the observation metadata for one entity."""
        for observation in self.observations:
            if observation.entity is entity:
                return observation
        raise KeyError(
            f"Source '{self.name}' does not observe entity '{entity.value}'"
        )

    def variables_for(self, entity: EntityType) -> frozenset[str]:
        """Return the variables observed for one entity."""
        return frozenset(self.observation_for(entity).variable_names)

    def observes(self, variable_name: str, entity: EntityType | None = None) -> bool:
        """Whether this source observes a variable."""
        if entity is not None:
            return variable_name in self.variables_for(entity)
        return any(variable_name in observation.variable_names for observation in self.observations)

    def capability_for(self, variable_name: str) -> SourceVariableCapability:
        """Return usage metadata for one variable, defaulting to permissive behavior."""
        return self.variable_capabilities.get(variable_name, SourceVariableCapability())

    def is_authoritative_for(self, variable_name: str) -> bool:
        """Whether the source should be trusted to donate this variable."""
        return self.capability_for(variable_name).authoritative

    def allows_conditioning_on(self, variable_name: str) -> bool:
        """Whether the variable is semantically valid as a shared conditioning feature."""
        return self.capability_for(variable_name).usable_as_condition


@dataclass(frozen=True)
class EntityRelationship:
    """Relationship between two observed entity tables."""

    parent_entity: EntityType
    child_entity: EntityType
    parent_key: str
    child_key: str
    cardinality: RelationshipCardinality = RelationshipCardinality.ONE_TO_MANY

    def __post_init__(self) -> None:
        if self.parent_entity is self.child_entity:
            raise ValueError("EntityRelationship must connect different entities")
        if not self.parent_key or not self.child_key:
            raise ValueError("EntityRelationship keys must be non-empty")


@dataclass
class ObservationFrame:
    """Observed tables and relationships for one source realization."""

    source: SourceDescriptor
    tables: Mapping[EntityType, pd.DataFrame]
    relationships: tuple[EntityRelationship, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        """Validate table schemas, primary keys, and foreign-key relationships."""
        for observation in self.source.observations:
            table = self.tables.get(observation.entity)
            if table is None:
                raise ValueError(
                    f"Source '{self.source.name}' is missing table for "
                    f"entity '{observation.entity.value}'"
                )

            required_columns = set(observation.variable_names) | {observation.key_column}
            if observation.weight_column is not None:
                required_columns.add(observation.weight_column)
            if observation.period_column is not None:
                required_columns.add(observation.period_column)

            missing_columns = required_columns - set(table.columns)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(
                    f"Source '{self.source.name}' entity '{observation.entity.value}' "
                    f"is missing columns: {missing}"
                )

            if table[observation.key_column].isna().any():
                raise ValueError(
                    f"Source '{self.source.name}' entity '{observation.entity.value}' "
                    "contains null primary keys"
                )
            if table[observation.key_column].duplicated().any():
                raise ValueError(
                    f"Source '{self.source.name}' entity '{observation.entity.value}' "
                    "contains duplicate primary keys"
                )

        for relationship in self.relationships:
            parent_table = self._table_for(relationship.parent_entity)
            child_table = self._table_for(relationship.child_entity)
            self._require_columns(
                relationship=relationship,
                table=parent_table,
                entity=relationship.parent_entity,
                columns=(relationship.parent_key,),
            )
            self._require_columns(
                relationship=relationship,
                table=child_table,
                entity=relationship.child_entity,
                columns=(relationship.child_key,),
            )

            parent_keys = set(parent_table[relationship.parent_key].dropna())
            child_keys = set(child_table[relationship.child_key].dropna())
            missing_parent_keys = sorted(child_keys - parent_keys)
            if missing_parent_keys:
                raise ValueError(
                    "Relationship "
                    f"{relationship.child_entity.value}->{relationship.parent_entity.value} "
                    f"has missing parent keys: {missing_parent_keys}"
                )

            if relationship.cardinality is RelationshipCardinality.ONE_TO_ONE:
                duplicates = child_table[relationship.child_key].dropna().duplicated()
                if duplicates.any():
                    raise ValueError(
                        "Relationship "
                        f"{relationship.child_entity.value}->{relationship.parent_entity.value} "
                        "violates one-to-one cardinality"
                    )

    def observation_mask(self, entity: EntityType) -> pd.DataFrame:
        """Return a boolean observation mask for one entity table."""
        observation = self.source.observation_for(entity)
        table = self._table_for(entity)
        mask = table.loc[:, list(observation.variable_names)].notna().copy()
        mask.index = pd.Index(table[observation.key_column], name=observation.key_column)
        return mask

    def _table_for(self, entity: EntityType) -> pd.DataFrame:
        table = self.tables.get(entity)
        if table is None:
            raise ValueError(
                f"Source '{self.source.name}' is missing table for entity '{entity.value}'"
            )
        return table

    def _require_columns(
        self,
        relationship: EntityRelationship,
        table: pd.DataFrame,
        entity: EntityType,
        columns: Sequence[str],
    ) -> None:
        missing_columns = set(columns) - set(table.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Relationship "
                f"{relationship.child_entity.value}->{relationship.parent_entity.value} "
                f"entity '{entity.value}' is missing columns: {missing}"
            )


@dataclass(frozen=True)
class SourceQuery:
    """Generic query parameters for loading observation frames."""

    period: int | str | None = None
    provider_filters: dict[str, Any] = field(default_factory=dict)


def apply_source_query(
    frame: ObservationFrame,
    query: SourceQuery | None = None,
) -> ObservationFrame:
    """Filter an observation frame using generic query semantics."""
    if query is None or query.period is None:
        return frame

    filtered_tables: dict[EntityType, pd.DataFrame] = {}
    for observation in frame.source.observations:
        table = frame.tables[observation.entity]
        if observation.period_column is None:
            filtered_tables[observation.entity] = table.copy()
            continue
        filtered_tables[observation.entity] = table.loc[
            table[observation.period_column] == query.period
        ].copy()

    filtered = ObservationFrame(
        source=frame.source,
        tables=filtered_tables,
        relationships=frame.relationships,
    )
    filtered.validate()
    return filtered


@runtime_checkable
class SourceProvider(Protocol):
    """Protocol for providers that materialize observation frames."""

    @property
    def descriptor(self) -> SourceDescriptor:
        """Return metadata describing the source."""

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        """Load the source into a validated observation frame."""


@dataclass
class StaticSourceProvider:
    """A provider backed by an in-memory observation frame."""

    frame: ObservationFrame

    @property
    def descriptor(self) -> SourceDescriptor:
        return self.frame.source

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        self.frame.validate()
        return apply_source_query(self.frame, query)


class SourceAdapter(Protocol):
    """Protocol for adapters that materialize observation frames."""

    @property
    def descriptor(self) -> SourceDescriptor:
        """Return metadata describing the source."""

    def load(self) -> ObservationFrame:
        """Load the source into a validated observation frame."""
