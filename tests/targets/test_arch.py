"""Tests for neutral Arch target artifact helpers."""

from __future__ import annotations

import json

import pytest

from microplex.targets import (
    ArchConsumerFact,
    arch_consumer_fact_concept,
    arch_consumer_fact_numeric_value,
    arch_consumer_fact_period,
    arch_consumer_fact_source_record_id,
    load_arch_consumer_fact_jsonl_rows,
    load_arch_consumer_facts,
)


def _consumer_fact(
    key: str,
    *,
    period: dict | None = None,
    concept_alignment: dict | None = None,
) -> dict:
    return {
        "schema_version": "arch.consumer_fact.v1",
        "aggregate_fact_key": f"arch.aggregate_fact.v2:{key}",
        "semantic_fact_key": f"arch.semantic_fact.v2:{key}",
        "value": "123.5",
        "period": period or {"type": "calendar_year", "value": 2024},
        "geography": {"level": "country", "id": "0100000US", "name": "US"},
        "observed_measure": {
            "source_concept": "publisher.population",
            "source_name": "publisher",
            "source_table": "Table 1",
            "unit": "count",
        },
        "concept_alignment": concept_alignment or {},
        "source": {"source_name": "publisher", "source_table": "Table 1"},
        "lineage": {
            "source_record_id": f"publisher.{key}",
            "source_cell_keys": [f"arch.source_cell.v1:{key}"],
        },
    }


def test_load_arch_consumer_facts_validates_and_parses_rows(tmp_path) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    _consumer_fact(
                        "state",
                        concept_alignment={
                            "canonical_concept": "canonical.population",
                        },
                    ),
                    sort_keys=True,
                ),
                "",
                json.dumps(
                    _consumer_fact(
                        "month",
                        period={"type": "month", "value": "2025-01"},
                    ),
                    sort_keys=True,
                ),
            ]
        )
        + "\n"
    )

    rows = load_arch_consumer_fact_jsonl_rows((path,), period=2024)
    facts = load_arch_consumer_facts((path,))

    assert len(rows) == 1
    assert len(facts) == 2
    assert isinstance(facts[0], ArchConsumerFact)
    assert facts[0].concept == "canonical.population"
    assert facts[0].period == 2024
    assert facts[0].value == 123.5
    assert facts[0].source_record_id == "publisher.state"
    assert facts[0].path == str(path)
    assert facts[0].line_number == 1
    assert facts[1].period == 2025


def test_arch_consumer_fact_accessors_fall_back_to_observed_concept() -> None:
    row = _consumer_fact("fallback")

    assert arch_consumer_fact_concept(row) == "publisher.population"
    assert arch_consumer_fact_period(row) == 2024
    assert arch_consumer_fact_source_record_id(row) == "publisher.fallback"
    assert arch_consumer_fact_numeric_value("42") == 42


def test_load_arch_consumer_facts_rejects_wrong_schema(tmp_path) -> None:
    path = tmp_path / "consumer_facts.jsonl"
    row = _consumer_fact("bad")
    row["schema_version"] = "arch.consumer_fact.v0"
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="line 1"):
        load_arch_consumer_fact_jsonl_rows((path,))


def test_arch_consumer_fact_numeric_value_rejects_bool() -> None:
    with pytest.raises(ValueError, match="not numeric"):
        arch_consumer_fact_numeric_value(True)
