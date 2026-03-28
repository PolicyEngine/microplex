"""Tests for source-symmetric multientity fusion planning."""

import json

import pandas as pd
import pytest

from microplex.core import (
    BenefitUnit,
    EntityObservation,
    EntityRelationship,
    EntityType,
    ObservationFrame,
    Record,
    RecordType,
    RelationshipCardinality,
    Shareability,
    SourceArchetype,
    SourceColumnValueType,
    SourceDescriptor,
    SourceManifest,
    SourceProvider,
    SourceQuery,
    SourceVariableCapability,
    StaticSourceProvider,
    TimeStructure,
    load_source_manifest,
)


def test_record_is_first_class_entity_type():
    assert EntityType.RECORD.value == "record"

    record = Record(id="job-1", person_id="p1", record_type=RecordType.W2)

    assert record.entity_type is EntityType.RECORD


def test_benefit_unit_is_first_class_entity_type():
    assert EntityType.BENEFIT_UNIT.value == "benefit_unit"

    benefit_unit = BenefitUnit(id="bu-1", member_ids=["p1", "p2"], head_id="p1")

    assert benefit_unit.entity_type is EntityType.BENEFIT_UNIT
    assert benefit_unit.entity_type.is_group


def test_source_descriptor_tracks_variables_by_entity():
    descriptor = SourceDescriptor(
        name="cps",
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        archetype=SourceArchetype.HOUSEHOLD_INCOME,
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age", "employment_income"),
            ),
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips", "rent"),
                weight_column="household_weight",
            ),
        ),
    )

    assert set(descriptor.observed_entities) == {
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
    }
    assert descriptor.variables_for(EntityType.PERSON) == frozenset(
        {"age", "employment_income"}
    )
    assert descriptor.archetype is SourceArchetype.HOUSEHOLD_INCOME
    assert descriptor.variables_for(EntityType.HOUSEHOLD) == frozenset(
        {"state_fips", "rent"}
    )
    assert descriptor.observes("rent", entity=EntityType.HOUSEHOLD)
    assert not descriptor.observes("rent", entity=EntityType.PERSON)


def test_source_descriptor_tracks_variable_capabilities():
    descriptor = SourceDescriptor(
        name="puf_like",
        shareability=Shareability.RESTRICTED,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips",),
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age", "taxable_interest_income"),
            ),
        ),
        variable_capabilities={
            "state_fips": SourceVariableCapability(usable_as_condition=False),
            "taxable_interest_income": SourceVariableCapability(authoritative=True),
        },
    )

    assert descriptor.all_variable_names == frozenset(
        {"state_fips", "age", "taxable_interest_income"}
    )
    assert not descriptor.allows_conditioning_on("state_fips")
    assert descriptor.is_authoritative_for("taxable_interest_income")
    assert descriptor.is_authoritative_for("age")


def test_load_source_manifest_reads_typed_json(tmp_path):
    manifest_path = tmp_path / "spi.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "uk_spi",
                "archetype": "tax_microdata",
                "population": "UK tax units",
                "description": "Tax-unit source",
                "observations": [
                    {
                        "entity": "tax_unit",
                        "key_column": "tax_unit_id",
                        "weight_column": "weight",
                        "period_column": "year",
                        "columns": [
                            {
                                "raw_column": "FACT",
                                "canonical_name": "weight",
                            },
                            {
                                "raw_column": "DIVIDENDS",
                                "canonical_name": "dividend_income",
                                "value_type": "numeric",
                            },
                            {
                                "raw_column": "GORCODE",
                                "canonical_name": "region_code",
                                "value_type": "categorical",
                            },
                        ],
                    }
                ],
            }
        )
    )

    manifest = load_source_manifest(manifest_path)

    assert isinstance(manifest, SourceManifest)
    assert manifest.archetype is SourceArchetype.TAX_MICRODATA
    observation = manifest.observation_for(EntityType.TAX_UNIT)
    assert observation.weight_column == "weight"
    assert observation.observed_variable_names() == ("dividend_income", "region_code")
    assert observation.columns[-1].value_type is SourceColumnValueType.CATEGORICAL


def test_source_descriptor_rejects_unknown_capability_variables():
    with pytest.raises(ValueError, match="unknown variables"):
        SourceDescriptor(
            name="bad",
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.REPEATED_CROSS_SECTION,
            observations=(
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=("age",),
                ),
            ),
            variable_capabilities={
                "state_fips": SourceVariableCapability(usable_as_condition=False),
            },
        )


def test_observation_frame_validates_relationships_and_builds_masks():
    descriptor = SourceDescriptor(
        name="cps",
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age", "employment_income"),
            ),
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips", "rent"),
            ),
        ),
    )
    households = pd.DataFrame(
        {
            "household_id": ["h1"],
            "state_fips": ["06"],
            "rent": [2400.0],
        }
    )
    persons = pd.DataFrame(
        {
            "person_id": ["p1", "p2"],
            "household_id": ["h1", "h1"],
            "age": [35, None],
            "employment_income": [50_000.0, 0.0],
        }
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: households,
            EntityType.PERSON: persons,
        },
        relationships=(
            EntityRelationship(
                parent_entity=EntityType.HOUSEHOLD,
                child_entity=EntityType.PERSON,
                parent_key="household_id",
                child_key="household_id",
                cardinality=RelationshipCardinality.ONE_TO_MANY,
            ),
        ),
    )

    frame.validate()
    mask = frame.observation_mask(EntityType.PERSON)

    assert mask.index.tolist() == ["p1", "p2"]
    assert list(mask.columns) == ["age", "employment_income"]
    assert mask["age"].tolist() == [True, False]
    assert mask["employment_income"].tolist() == [True, True]


def test_observation_frame_rejects_missing_parent_keys():
    descriptor = SourceDescriptor(
        name="cps",
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age",),
            ),
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips",),
            ),
        ),
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: pd.DataFrame(
                {
                    "household_id": ["h1"],
                    "state_fips": ["06"],
                }
            ),
            EntityType.PERSON: pd.DataFrame(
                {
                    "person_id": ["p1", "p2"],
                    "household_id": ["h1", "missing"],
                    "age": [35, 41],
                }
            ),
        },
        relationships=(
            EntityRelationship(
                parent_entity=EntityType.HOUSEHOLD,
                child_entity=EntityType.PERSON,
                parent_key="household_id",
                child_key="household_id",
                cardinality=RelationshipCardinality.ONE_TO_MANY,
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing parent keys"):
        frame.validate()


def test_static_source_provider_filters_on_period_columns():
    descriptor = SourceDescriptor(
        name="survey",
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.REPEATED_CROSS_SECTION,
        observations=(
            EntityObservation(
                entity=EntityType.HOUSEHOLD,
                key_column="household_id",
                variable_names=("state_fips",),
                period_column="year",
            ),
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age",),
                period_column="year",
            ),
        ),
    )
    frame = ObservationFrame(
        source=descriptor,
        tables={
            EntityType.HOUSEHOLD: pd.DataFrame(
                {
                    "household_id": ["h1", "h2"],
                    "state_fips": ["06", "36"],
                    "year": [2024, 2023],
                }
            ),
            EntityType.PERSON: pd.DataFrame(
                {
                    "person_id": ["p1", "p2"],
                    "household_id": ["h1", "h2"],
                    "age": [35, 41],
                    "year": [2024, 2023],
                }
            ),
        },
        relationships=(
            EntityRelationship(
                parent_entity=EntityType.HOUSEHOLD,
                child_entity=EntityType.PERSON,
                parent_key="household_id",
                child_key="household_id",
            ),
        ),
    )
    provider = StaticSourceProvider(frame)

    assert isinstance(provider, SourceProvider)
    filtered = provider.load_frame(SourceQuery(period=2024))

    assert filtered.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == ["h1"]
    assert filtered.tables[EntityType.PERSON]["person_id"].tolist() == ["p1"]
