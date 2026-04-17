"""Source-symmetric planning for multientity fusion."""

from __future__ import annotations

from dataclasses import dataclass

from microplex.core import EntityType
from microplex.core.sources import (
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    TimeStructure,
)


@dataclass(frozen=True)
class VariableCoverage:
    """Coverage metadata for one entity-scoped variable."""

    entity: EntityType
    variable_name: str
    sources: tuple[str, ...]
    shareabilities: frozenset[Shareability]
    time_structures: frozenset[TimeStructure]

    @property
    def publicly_observed(self) -> bool:
        """Whether at least one public source observes this variable."""
        return Shareability.PUBLIC in self.shareabilities

    @property
    def requires_synthetic_release(self) -> bool:
        """Whether public release must synthesize this variable."""
        return not self.publicly_observed


@dataclass(frozen=True)
class FusionPlan:
    """A source-symmetric plan describing multientity fusion coverage."""

    source_names: tuple[str, ...]
    source_archetypes: dict[str, SourceArchetype | None]
    coverage: dict[EntityType, dict[str, VariableCoverage]]

    @classmethod
    def from_sources(cls, sources: list[SourceDescriptor]) -> FusionPlan:
        """Build a fusion plan from a set of source descriptors."""
        if not sources:
            raise ValueError("FusionPlan requires at least one source")

        source_names = [source.name for source in sources]
        if len(set(source_names)) != len(source_names):
            raise ValueError("FusionPlan source names must be unique")

        by_variable: dict[
            tuple[EntityType, str],
            dict[str, set[str] | set[Shareability] | set[TimeStructure]],
        ] = {}

        for source in sources:
            for observation in source.observations:
                for variable_name in observation.variable_names:
                    key = (observation.entity, variable_name)
                    entry = by_variable.setdefault(
                        key,
                        {
                            "sources": set(),
                            "shareabilities": set(),
                            "time_structures": set(),
                        },
                    )
                    entry["sources"].add(source.name)
                    entry["shareabilities"].add(source.shareability)
                    entry["time_structures"].add(source.time_structure)

        coverage: dict[EntityType, dict[str, VariableCoverage]] = {}
        for (entity, variable_name), entry in sorted(
            by_variable.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        ):
            entity_coverage = coverage.setdefault(entity, {})
            entity_coverage[variable_name] = VariableCoverage(
                entity=entity,
                variable_name=variable_name,
                sources=tuple(sorted(entry["sources"])),
                shareabilities=frozenset(entry["shareabilities"]),
                time_structures=frozenset(entry["time_structures"]),
            )

        return cls(
            source_names=tuple(source_names),
            source_archetypes={
                source.name: source.archetype for source in sources
            },
            coverage=coverage,
        )

    @property
    def output_entities(self) -> tuple[EntityType, ...]:
        """Entities that appear in the planned fusion output."""
        return tuple(self.coverage.keys())

    def variables_for(self, entity: EntityType) -> frozenset[str]:
        """Variables covered for one entity."""
        entity_coverage = self.coverage.get(entity, {})
        return frozenset(entity_coverage)

    def sources_for_archetype(
        self,
        archetype: SourceArchetype,
    ) -> tuple[str, ...]:
        """Return source names registered to one cross-country archetype."""
        return tuple(
            source_name
            for source_name in self.source_names
            if self.source_archetypes.get(source_name) is archetype
        )

    def variable_plan(
        self,
        entity: EntityType,
        variable_name: str,
    ) -> VariableCoverage:
        """Return coverage metadata for one variable."""
        entity_coverage = self.coverage.get(entity)
        if entity_coverage is None or variable_name not in entity_coverage:
            raise KeyError(
                f"Fusion plan has no coverage for {entity.value}.{variable_name}"
            )
        return entity_coverage[variable_name]

    def variables_requiring_synthetic_release(
        self,
        entity: EntityType,
    ) -> frozenset[str]:
        """Variables that must be synthesized for public release."""
        entity_coverage = self.coverage.get(entity, {})
        return frozenset(
            variable_name
            for variable_name, variable_coverage in entity_coverage.items()
            if variable_coverage.requires_synthetic_release
        )
