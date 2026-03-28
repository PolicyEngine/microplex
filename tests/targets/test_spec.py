"""Tests for the canonical target specification primitives."""

from microplex.core import EntityType
from microplex.targets import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetSet,
    TargetSpec,
)


class TestTargetFilter:
    def test_operator_normalizes_from_string(self):
        target_filter = TargetFilter(feature="snap", operator=">", value=0)

        assert target_filter.operator is FilterOperator.GT


class TestTargetSpec:
    def test_entity_and_aggregation_normalize_from_strings(self):
        target = TargetSpec(
            name="snap_recipients",
            entity="spm_unit",
            value=100.0,
            period=2024,
            aggregation="count",
            filters=(TargetFilter(feature="snap", operator=">", value=0),),
        )

        assert target.entity is EntityType.SPM_UNIT
        assert target.aggregation is TargetAggregation.COUNT

    def test_count_target_rejects_measure(self):
        try:
            TargetSpec(
                name="bad_target",
                entity=EntityType.HOUSEHOLD,
                value=1.0,
                period=2024,
                measure="income",
                aggregation=TargetAggregation.COUNT,
            )
        except ValueError as exc:
            assert "Count targets" in str(exc)
        else:
            raise AssertionError("Expected ValueError for count target with measure")

    def test_required_features_deduplicates_and_preserves_order(self):
        target = TargetSpec(
            name="california_snap",
            entity=EntityType.HOUSEHOLD,
            value=1_000.0,
            period=2024,
            measure="snap",
            aggregation=TargetAggregation.SUM,
            filters=(
                TargetFilter(feature="state_fips", operator="==", value="06"),
                TargetFilter(feature="snap", operator=">", value=0),
            ),
        )

        assert target.required_features == ("snap", "state_fips")


class TestTargetSet:
    def test_collection_helpers(self):
        targets = TargetSet(
            targets=[
                TargetSpec(
                    name="households",
                    entity=EntityType.HOUSEHOLD,
                    value=10.0,
                    period=2024,
                    aggregation=TargetAggregation.COUNT,
                ),
                TargetSpec(
                    name="people",
                    entity=EntityType.PERSON,
                    value=20.0,
                    period=2025,
                    aggregation=TargetAggregation.COUNT,
                ),
            ]
        )

        assert len(targets.for_entity(EntityType.HOUSEHOLD)) == 1
        assert len(targets.for_period(2025)) == 1
        assert targets.required_features() == ()
