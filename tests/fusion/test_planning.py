"""Tests for source-symmetric fusion planning."""

from microplex.core import (
    EntityObservation,
    EntityType,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    TimeStructure,
)
from microplex.fusion import FusionPlan


def test_fusion_plan_is_source_symmetric_and_release_aware():
    cps = SourceDescriptor(
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
                variable_names=("state_fips",),
            ),
        ),
    )
    puf = SourceDescriptor(
        name="puf",
        shareability=Shareability.NON_SHAREABLE,
        time_structure=TimeStructure.CROSS_SECTION,
        archetype=SourceArchetype.TAX_MICRODATA,
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age",),
            ),
            EntityObservation(
                entity=EntityType.TAX_UNIT,
                key_column="tax_unit_id",
                variable_names=("adjusted_gross_income", "capital_gains"),
            ),
        ),
    )
    sipp = SourceDescriptor(
        name="sipp",
        shareability=Shareability.PUBLIC,
        time_structure=TimeStructure.PANEL,
        archetype=SourceArchetype.LONGITUDINAL_SOCIOECONOMIC,
        observations=(
            EntityObservation(
                entity=EntityType.PERSON,
                key_column="person_id",
                variable_names=("age", "employment_income", "is_disabled"),
            ),
            EntityObservation(
                entity=EntityType.RECORD,
                key_column="record_id",
                variable_names=("job_hours", "job_wages"),
            ),
        ),
    )

    plan = FusionPlan.from_sources([cps, puf, sipp])

    assert set(plan.output_entities) == {
        EntityType.PERSON,
        EntityType.HOUSEHOLD,
        EntityType.TAX_UNIT,
        EntityType.RECORD,
    }

    employment_income = plan.variable_plan(
        entity=EntityType.PERSON,
        variable_name="employment_income",
    )
    assert set(employment_income.sources) == {"cps", "sipp"}
    assert employment_income.publicly_observed
    assert not employment_income.requires_synthetic_release
    assert plan.sources_for_archetype(SourceArchetype.HOUSEHOLD_INCOME) == ("cps",)
    assert plan.sources_for_archetype(SourceArchetype.TAX_MICRODATA) == ("puf",)

    adjusted_gross_income = plan.variable_plan(
        entity=EntityType.TAX_UNIT,
        variable_name="adjusted_gross_income",
    )
    assert set(adjusted_gross_income.sources) == {"puf"}
    assert not adjusted_gross_income.publicly_observed
    assert adjusted_gross_income.requires_synthetic_release
    assert plan.variables_requiring_synthetic_release(EntityType.TAX_UNIT) == (
        frozenset({"adjusted_gross_income", "capital_gains"})
    )

    job_hours = plan.variable_plan(
        entity=EntityType.RECORD,
        variable_name="job_hours",
    )
    assert set(job_hours.sources) == {"sipp"}
    assert job_hours.publicly_observed
