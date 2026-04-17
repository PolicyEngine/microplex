"""Typed source-manifest loader for externalized provider specs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from microplex.core.entities import EntityType
from microplex.core.sources import SourceArchetype


class SourceColumnValueType(Enum):
    """How a raw source column should be coerced during canonical mapping."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class SourceColumnManifest:
    """One raw-to-canonical column mapping."""

    raw_column: str
    canonical_name: str
    value_type: SourceColumnValueType = SourceColumnValueType.NUMERIC


@dataclass(frozen=True)
class SourceObservationManifest:
    """Manifest for one observed entity table."""

    entity: EntityType
    key_column: str
    table_name: str | None = None
    weight_column: str | None = None
    period_column: str | None = None
    excluded_columns: tuple[str, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    columns: tuple[SourceColumnManifest, ...] = ()

    def observed_variable_names(
        self,
        frame_columns: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        """Return canonical observed variables for this entity."""
        reserved = {self.key_column}
        if self.weight_column is not None:
            reserved.add(self.weight_column)
        if self.period_column is not None:
            reserved.add(self.period_column)
        reserved.update(self.excluded_columns)
        if self.columns:
            return tuple(
                column.canonical_name
                for column in self.columns
                if column.canonical_name not in reserved
            )
        if frame_columns is None:
            raise ValueError(
                "frame_columns must be provided when manifest columns are implicit"
            )
        return tuple(column for column in frame_columns if column not in reserved)


@dataclass(frozen=True)
class SourceManifest:
    """Typed manifest for one source-provider family."""

    name: str
    archetype: SourceArchetype
    population: str | None = None
    description: str | None = None
    observations: tuple[SourceObservationManifest, ...] = ()

    def observation_for(self, entity: EntityType) -> SourceObservationManifest:
        """Return the manifest entry for one entity."""
        for observation in self.observations:
            if observation.entity is entity:
                return observation
        raise KeyError(f"Manifest '{self.name}' has no entity '{entity.value}'")


def load_source_manifest(path: str | Path) -> SourceManifest:
    """Load a typed source manifest from JSON."""
    payload = json.loads(Path(path).read_text())
    observations = tuple(
        SourceObservationManifest(
            entity=EntityType(observation_payload["entity"]),
            key_column=observation_payload["key_column"],
            table_name=observation_payload.get("table_name"),
            weight_column=observation_payload.get("weight_column"),
            period_column=observation_payload.get("period_column"),
            excluded_columns=tuple(observation_payload.get("excluded_columns", ())),
            aliases=dict(observation_payload.get("aliases", {})),
            columns=tuple(
                SourceColumnManifest(
                    raw_column=column_payload["raw_column"],
                    canonical_name=column_payload["canonical_name"],
                    value_type=SourceColumnValueType(
                        column_payload.get("value_type", "numeric")
                    ),
                )
                for column_payload in observation_payload.get("columns", ())
            ),
        )
        for observation_payload in payload["observations"]
    )
    return SourceManifest(
        name=payload["name"],
        archetype=SourceArchetype(payload["archetype"]),
        population=payload.get("population"),
        description=payload.get("description"),
        observations=observations,
    )
