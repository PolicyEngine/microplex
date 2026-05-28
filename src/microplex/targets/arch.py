"""Neutral helpers for Arch target artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARCH_CONSUMER_FACT_SCHEMA_VERSION = "arch.consumer_fact.v1"


@dataclass(frozen=True)
class ArchConsumerFact:
    """Neutral view over one Arch consumer-contract fact row."""

    row: Mapping[str, Any]
    path: str | None = None
    line_number: int | None = None

    @property
    def concept(self) -> str | None:
        """Return the canonical or observed concept for this fact."""
        return arch_consumer_fact_concept(self.row)

    @property
    def period(self) -> int:
        """Return the calendar/model year represented by this fact."""
        return arch_consumer_fact_period(self.row)

    @property
    def value(self) -> float:
        """Return the fact's numeric value."""
        return arch_consumer_fact_numeric_value(self.row.get("value"))

    @property
    def geography(self) -> Mapping[str, Any]:
        """Return the fact geography payload."""
        return mapping_value(self.row.get("geography"))

    @property
    def source(self) -> Mapping[str, Any]:
        """Return the source metadata payload."""
        return mapping_value(self.row.get("source"))

    @property
    def source_record_id(self) -> str | None:
        """Return the source record ID from lineage metadata, when present."""
        return arch_consumer_fact_source_record_id(self.row)


def load_arch_consumer_fact_jsonl_rows(
    paths: Iterable[str | Path],
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
) -> tuple[dict[str, Any], ...]:
    """Load validated Arch consumer fact JSONL rows from one or more files."""
    rows: list[dict[str, Any]] = []
    for pathlike in paths:
        path = Path(pathlike)
        for fact in iter_arch_consumer_facts(
            path,
            period=period,
            schema_version=schema_version,
        ):
            rows.append(dict(fact.row))
    return tuple(rows)


def load_arch_consumer_facts(
    paths: Iterable[str | Path],
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
) -> tuple[ArchConsumerFact, ...]:
    """Load validated Arch consumer facts from one or more JSONL files."""
    facts: list[ArchConsumerFact] = []
    for path in paths:
        facts.extend(
            iter_arch_consumer_facts(
                path,
                period=period,
                schema_version=schema_version,
            )
        )
    return tuple(facts)


def iter_arch_consumer_facts(
    pathlike: str | Path,
    *,
    period: int | None = None,
    schema_version: str = ARCH_CONSUMER_FACT_SCHEMA_VERSION,
) -> Iterable[ArchConsumerFact]:
    """Yield validated Arch consumer facts from one JSONL file."""
    path = Path(pathlike)
    with path.open() as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            observed_schema_version = row.get("schema_version")
            if observed_schema_version != schema_version:
                raise ValueError(
                    "Unsupported Arch consumer fact schema "
                    f"{observed_schema_version!r} in {path} line {line_number}; "
                    f"expected {schema_version!r}."
                )
            if period is not None and arch_consumer_fact_period(row) != int(period):
                continue
            yield ArchConsumerFact(
                row=row,
                path=str(path),
                line_number=line_number,
            )


def arch_consumer_fact_concept(row: Mapping[str, Any]) -> str | None:
    """Return a row's canonical concept, falling back to source concept."""
    concept_alignment = mapping_value(row.get("concept_alignment"))
    observed_measure = mapping_value(row.get("observed_measure"))
    concept = concept_alignment.get("canonical_concept") or observed_measure.get(
        "source_concept"
    )
    return str(concept) if concept is not None else None


def arch_consumer_fact_period(row: Mapping[str, Any]) -> int:
    """Return a consumer fact period as an integer year."""
    period = mapping_value(row.get("period"))
    value = period["value"]
    if period.get("type") == "month" and isinstance(value, str):
        return int(value.split("-", maxsplit=1)[0])
    return int(value)


def arch_consumer_fact_source_record_id(row: Mapping[str, Any]) -> str | None:
    """Return a source record ID from a consumer fact lineage payload."""
    lineage = mapping_value(row.get("lineage"))
    source_record_id = lineage.get("source_record_id")
    return str(source_record_id) if source_record_id is not None else None


def arch_consumer_fact_numeric_value(value: Any) -> float:
    """Return a numeric consumer fact value."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Arch consumer fact value is not numeric: {value!r}")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"Arch consumer fact value is not numeric: {value!r}")


def mapping_value(value: Any) -> Mapping[str, Any]:
    """Return a mapping payload, or an empty mapping for malformed/empty values."""
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "ARCH_CONSUMER_FACT_SCHEMA_VERSION",
    "ArchConsumerFact",
    "arch_consumer_fact_concept",
    "arch_consumer_fact_numeric_value",
    "arch_consumer_fact_period",
    "arch_consumer_fact_source_record_id",
    "iter_arch_consumer_facts",
    "load_arch_consumer_fact_jsonl_rows",
    "load_arch_consumer_facts",
    "mapping_value",
]
