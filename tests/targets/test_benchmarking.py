from __future__ import annotations

import pytest

from microplex.core import EntityType
from microplex.targets import (
    BenchmarkComparison,
    BenchmarkResultEvaluator,
    BenchmarkSliceComparison,
    BenchmarkSliceSpec,
    StaticTargetProvider,
    TargetAggregation,
    TargetQuery,
    TargetSet,
    TargetSpec,
    UnsupportedTarget,
    build_benchmark_result,
    build_benchmark_suite_from_payloads,
    build_benchmark_suite_from_results,
    build_benchmark_suite_result,
    build_grouped_summaries,
    compare_benchmark_results,
    compare_metric_sets,
    evaluate_benchmark_slice_payloads,
    evaluate_benchmark_slice_results,
    filter_nonempty_benchmark_slices,
    load_benchmark_slice_target_sets,
    normalize_metric_payload,
    union_target_sets,
)


class _StaticBenchmarkEvaluator:
    def __init__(self, estimates: dict[str, float]):
        self.estimates = dict(estimates)

    def evaluate_target_set(
        self,
        target_set: TargetSet,
        slice_spec: BenchmarkSliceSpec,
    ):
        target = target_set.targets[0]
        return build_benchmark_result(
            label="candidate",
            time_period=2024,
            metrics=[
                normalize_metric_payload(
                    {
                        "name": target.name,
                        "estimate": self.estimates[slice_spec.name],
                        "target": target.value,
                    }
                )
            ],
            target_count=1,
        )


class _StaticBatchBenchmarkEvaluator:
    def __init__(self, estimates: dict[str, float]):
        self.estimates = dict(estimates)

    def evaluate_target_sets(
        self,
        target_sets: dict[str, TargetSet],
        slices: tuple[BenchmarkSliceSpec, ...],
    ):
        return {
            slice_spec.name: build_benchmark_result(
                label="candidate",
                time_period=2024,
                metrics=[
                    normalize_metric_payload(
                        {
                            "name": target_sets[slice_spec.name].targets[0].name,
                            "estimate": self.estimates[slice_spec.name],
                            "target": target_sets[slice_spec.name].targets[0].value,
                        }
                    )
                ],
                target_count=1,
            )
            for slice_spec in slices
        }


def test_normalize_metric_payload_uses_finite_zero_target_denominator():
    metric = normalize_metric_payload(
        {
            "name": "zero_target",
            "estimate": 2.0,
            "target": 0.0,
        }
    )

    assert metric.rel_error == 2.0
    assert metric.abs_rel_error == 2.0


def test_compare_metric_sets_uses_common_target_intersection():
    candidate = [
        normalize_metric_payload(
            {"name": "shared", "estimate": 8.0, "target": 10.0, "metadata": {"source": "a"}}
        ),
        normalize_metric_payload(
            {"name": "candidate_only", "estimate": 100.0, "target": 10.0}
        ),
    ]
    baseline = [
        normalize_metric_payload(
            {"name": "shared", "estimate": 9.0, "target": 10.0, "metadata": {"source": "a"}}
        ),
        normalize_metric_payload(
            {"name": "baseline_only", "estimate": 0.0, "target": 10.0}
        ),
    ]

    summary = compare_metric_sets(candidate, baseline)

    assert summary.common_target_count == 1
    assert summary.candidate_excluded_target_count == 1
    assert summary.baseline_excluded_target_count == 1
    assert summary.mean_abs_relative_error_delta == pytest.approx(0.1)
    assert summary.target_win_rate == 0.0

    grouped = build_grouped_summaries(summary.deltas, group_fields=("source",))
    assert grouped["source"][0].group_value == "a"
    assert grouped["source"][0].target_count == 1


def test_compare_metric_sets_raises_on_zero_common_targets():
    candidate = [normalize_metric_payload({"name": "a", "estimate": 1.0, "target": 1.0})]
    baseline = [normalize_metric_payload({"name": "b", "estimate": 1.0, "target": 1.0})]

    with pytest.raises(ValueError, match="zero common supported targets"):
        compare_metric_sets(candidate, baseline)


def test_compare_benchmark_results_builds_serializable_report(tmp_path):
    candidate = build_benchmark_result(
        label="candidate",
        time_period=2024,
        metrics=[
            normalize_metric_payload(
                {"name": "shared", "estimate": 8.0, "target": 10.0, "metadata": {"source": "a"}}
            )
        ],
        target_count=2,
    )
    baseline = build_benchmark_result(
        label="baseline",
        time_period=2024,
        metrics=[
            normalize_metric_payload(
                {"name": "shared", "estimate": 9.0, "target": 10.0, "metadata": {"source": "a"}}
            )
        ],
        target_count=3,
    )

    comparison = compare_benchmark_results(candidate, baseline)
    payload = comparison.to_dict()

    assert comparison.mean_abs_relative_error_delta == pytest.approx(0.1)
    assert comparison.common_target_count == 1
    assert comparison.grouped_summaries == {}
    assert payload["candidate"]["supported_target_count"] == 1
    assert payload["baseline"]["target_count"] == 3
    assert payload["metadata"]["candidate_excluded_target_count"] == 0
    loaded = comparison.load(comparison.save(tmp_path / "comparison.json"))
    assert loaded.common_target_count == 1


def test_build_benchmark_result_save_writes_serializable_payload(tmp_path):
    result = build_benchmark_result(
        label="candidate",
        time_period=2024,
        metrics=[
            normalize_metric_payload(
                {"name": "shared", "estimate": 8.0, "target": 10.0, "metadata": {"source": "a"}}
            )
        ],
        target_count=1,
    )

    output_path = result.save(tmp_path / "benchmark_result.json")

    assert output_path.exists()
    payload = output_path.read_text()
    assert '"label": "candidate"' in payload


def test_build_benchmark_suite_result_summarizes_slice_comparisons(tmp_path):
    candidate = build_benchmark_result(
        label="candidate",
        time_period=2024,
        metrics=[
            normalize_metric_payload({"name": "shared", "estimate": 8.0, "target": 10.0}),
        ],
        target_count=1,
    )
    baseline = build_benchmark_result(
        label="baseline",
        time_period=2024,
        metrics=[
            normalize_metric_payload({"name": "shared", "estimate": 9.0, "target": 10.0}),
        ],
        target_count=1,
    )
    comparison = compare_benchmark_results(candidate, baseline)

    suite = build_benchmark_suite_result(
        candidate_label="candidate",
        baseline_label="baseline",
        period=2024,
        slice_results=[
            BenchmarkSliceComparison(
                slice=BenchmarkSliceSpec(
                    name="all_targets",
                    query=TargetQuery(period=2024),
                    description="All targets",
                    tags=("benchmark",),
                ),
                comparison=comparison,
            )
        ],
    )

    output_path = suite.save(tmp_path / "benchmark_suite.json")
    loaded = suite.load(output_path)
    payload = output_path.read_text()

    assert suite.mean_abs_relative_error_delta == pytest.approx(0.1)
    assert suite.slice_win_rate == 0.0
    assert suite.target_win_rate == 0.0
    assert suite.supported_target_rate == 1.0
    assert suite.baseline_supported_target_rate == 1.0
    assert loaded.slice_results[0].slice.query is not None
    assert loaded.slice_results[0].slice.query.period == 2024
    assert '"candidate_label": "candidate"' in payload
    assert '"query"' in payload


def test_benchmark_suite_result_preserves_duplicate_target_names_across_slices():
    candidate_wins = build_benchmark_result(
        label="candidate",
        time_period=2024,
        metrics=[
            normalize_metric_payload({"name": "shared", "estimate": 9.0, "target": 10.0}),
        ],
        target_count=1,
    )
    baseline_wins = build_benchmark_result(
        label="baseline",
        time_period=2024,
        metrics=[
            normalize_metric_payload({"name": "shared", "estimate": 8.0, "target": 10.0}),
        ],
        target_count=1,
    )
    candidate_unsupported = build_benchmark_result(
        label="candidate",
        time_period=2024,
        metrics=[],
        target_count=1,
        unsupported_targets=[UnsupportedTarget(name="shared", reason="unsupported")],
    )
    baseline_supported = build_benchmark_result(
        label="baseline",
        time_period=2024,
        metrics=[
            normalize_metric_payload({"name": "shared", "estimate": 7.0, "target": 10.0}),
        ],
        target_count=1,
    )
    suite = build_benchmark_suite_result(
        candidate_label="candidate",
        baseline_label="baseline",
        period=2024,
        slice_results=[
            BenchmarkSliceComparison(
                slice=BenchmarkSliceSpec(name="slice_a", tags=("a",)),
                comparison=compare_benchmark_results(candidate_wins, baseline_wins),
            ),
            BenchmarkSliceComparison(
                slice=BenchmarkSliceSpec(name="slice_b", tags=("b",)),
                comparison=BenchmarkComparison(
                    candidate=candidate_unsupported,
                    baseline=baseline_supported,
                    mean_abs_relative_error_delta=0.0,
                    target_win_rate=0.0,
                    common_target_count=0,
                ),
            ),
        ],
    )

    assert suite.target_win_rate == 1.0
    assert suite.supported_target_rate == 0.5
    assert suite.baseline_supported_target_rate == 1.0


def test_benchmark_slice_helpers_load_filter_and_union_targets():
    provider = StaticTargetProvider(
        TargetSet(
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=100.0,
                    period=2024,
                    measure="snap",
                    aggregation=TargetAggregation.SUM,
                ),
                TargetSpec(
                    name="population",
                    entity=EntityType.HOUSEHOLD,
                    value=2.0,
                    period=2024,
                    aggregation=TargetAggregation.COUNT,
                ),
            ]
        )
    )
    slices = (
        BenchmarkSliceSpec(
            name="snap",
            query=TargetQuery(period=2024, names=("snap_total",)),
        ),
        BenchmarkSliceSpec(
            name="missing",
            query=TargetQuery(period=2024, names=("unknown",)),
        ),
    )

    loaded = load_benchmark_slice_target_sets(provider, slices)
    filtered = filter_nonempty_benchmark_slices(provider, slices)
    union = union_target_sets({"snap": loaded["snap"]})

    assert tuple(loaded) == ("snap", "missing")
    assert tuple(target.name for target in loaded["snap"].targets) == ("snap_total",)
    assert not loaded["missing"].targets
    assert tuple(slice_spec.name for slice_spec in filtered) == ("snap",)
    assert tuple(target.name for target in union.targets) == ("snap_total",)


def test_benchmark_payload_helpers_build_suite_from_custom_payloads():
    slices = (
        BenchmarkSliceSpec(
            name="snap",
            query=TargetQuery(period=2024, names=("snap_total",)),
        ),
    )
    target_sets = {
        "snap": TargetSet(
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=100.0,
                    period=2024,
                    measure="snap",
                    aggregation=TargetAggregation.SUM,
                )
            ]
        )
    }

    candidate_payloads = evaluate_benchmark_slice_payloads(
        target_sets,
        slices,
        evaluator=lambda target_set, _slice: {"targets": target_set, "estimate": 95.0},
    )
    baseline_payloads = evaluate_benchmark_slice_payloads(
        target_sets,
        slices,
        batch_evaluator=lambda loaded_target_sets, _slices: {
            "snap": {"targets": loaded_target_sets["snap"], "estimate": 90.0}
        },
    )

    suite = build_benchmark_suite_from_payloads(
        slices=slices,
        candidate_payloads=candidate_payloads,
        baseline_payloads=baseline_payloads,
        candidate_result_getter=lambda payload: build_benchmark_result(
            label="candidate",
            time_period=2024,
            metrics=[
                normalize_metric_payload(
                    {
                        "name": payload["targets"].targets[0].name,
                        "estimate": payload["estimate"],
                        "target": payload["targets"].targets[0].value,
                        "metadata": {"source": "candidate"},
                    }
                )
            ],
            target_count=1,
        ),
        baseline_result_getter=lambda payload: build_benchmark_result(
            label="baseline",
            time_period=2024,
            metrics=[
                normalize_metric_payload(
                    {
                        "name": payload["targets"].targets[0].name,
                        "estimate": payload["estimate"],
                        "target": payload["targets"].targets[0].value,
                        "metadata": {"source": "baseline"},
                    }
                )
            ],
            target_count=1,
        ),
        candidate_label="candidate",
        baseline_label="baseline",
        period=2024,
        group_fields=("source",),
    )

    assert suite.mean_abs_relative_error_delta == pytest.approx(-0.05)
    assert suite.target_win_rate == 1.0
    assert suite.slice_results[0].slice.query is not None
    assert "source" in suite.slice_results[0].comparison.grouped_summaries


def test_benchmark_result_helpers_evaluate_and_build_suite():
    slices = (
        BenchmarkSliceSpec(
            name="snap",
            query=TargetQuery(period=2024, names=("snap_total",)),
        ),
    )
    target_sets = {
        "snap": TargetSet(
            [
                TargetSpec(
                    name="snap_total",
                    entity=EntityType.HOUSEHOLD,
                    value=100.0,
                    period=2024,
                    measure="snap",
                    aggregation=TargetAggregation.SUM,
                )
            ]
        )
    }

    evaluator = _StaticBenchmarkEvaluator({"snap": 95.0})
    baseline_evaluator = _StaticBatchBenchmarkEvaluator({"snap": 90.0})

    assert isinstance(evaluator, BenchmarkResultEvaluator)
    candidate_results = evaluate_benchmark_slice_results(
        target_sets,
        slices,
        evaluator=evaluator,
    )
    baseline_results = evaluate_benchmark_slice_results(
        target_sets,
        slices,
        batch_evaluator=baseline_evaluator,
    )
    suite = build_benchmark_suite_from_results(
        slices=slices,
        candidate_results=candidate_results,
        baseline_results=baseline_results,
        candidate_label="candidate",
        baseline_label="baseline",
        period=2024,
    )

    assert suite.mean_abs_relative_error_delta == pytest.approx(-0.05)
    assert suite.target_win_rate == 1.0


def test_evaluate_benchmark_slice_results_raises_when_result_missing():
    slices = (BenchmarkSliceSpec(name="snap"),)
    target_sets = {"snap": TargetSet([])}

    class _MissingBatchEvaluator:
        def evaluate_target_sets(self, target_sets, slices):
            _ = target_sets, slices
            return {}

    with pytest.raises(ValueError, match="did not return results"):
        evaluate_benchmark_slice_results(
            target_sets,
            slices,
            batch_evaluator=_MissingBatchEvaluator(),
        )


def test_union_target_sets_rejects_conflicting_duplicate_names():
    with pytest.raises(ValueError, match="conflicting definitions"):
        union_target_sets(
            {
                "left": TargetSet(
                    [
                        TargetSpec(
                            name="snap_total",
                            entity=EntityType.HOUSEHOLD,
                            value=100.0,
                            period=2024,
                            measure="snap",
                            aggregation=TargetAggregation.SUM,
                        )
                    ]
                ),
                "right": TargetSet(
                    [
                        TargetSpec(
                            name="snap_total",
                            entity=EntityType.HOUSEHOLD,
                            value=200.0,
                            period=2024,
                            measure="snap",
                            aggregation=TargetAggregation.SUM,
                        )
                    ]
                ),
            }
        )
