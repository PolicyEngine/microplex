"""Tests for canonical target providers."""

from microplex.core import EntityType
from microplex.targets import (
    StaticTargetProvider,
    TargetAggregation,
    TargetProvider,
    TargetQuery,
    TargetSet,
    TargetSpec,
    apply_target_query,
)


def test_apply_target_query_filters_target_set():
    target_set = TargetSet(
        [
            TargetSpec(
                name="ca_people",
                entity=EntityType.PERSON,
                value=2.0,
                period=2024,
                aggregation=TargetAggregation.COUNT,
                metadata={"kind": "admin"},
            ),
            TargetSpec(
                name="ny_people",
                entity=EntityType.PERSON,
                value=3.0,
                period=2023,
                aggregation=TargetAggregation.COUNT,
                metadata={"kind": "survey"},
            ),
        ]
    )

    selected = apply_target_query(
        target_set,
        TargetQuery(
            period=2024,
            entity=EntityType.PERSON,
            names=("ca_people",),
            metadata_filters={"kind": "admin"},
        ),
    )

    assert selected.targets == [target_set.targets[0]]


def test_static_target_provider_implements_protocol():
    target = TargetSpec(
        name="ca_people",
        entity=EntityType.PERSON,
        value=2.0,
        period=2024,
        aggregation=TargetAggregation.COUNT,
    )
    provider = StaticTargetProvider(TargetSet([target]))

    assert isinstance(provider, TargetProvider)
    assert provider.load_target_set(TargetQuery(entity=EntityType.PERSON)).targets == [target]
