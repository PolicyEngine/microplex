"""Shared benchmark metric normalization and comparison helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from microplex.targets.provider import TargetProvider, TargetQuery
from microplex.targets.spec import TargetSet, TargetSpec

_RELATIVE_ERROR_DENOMINATOR_FLOOR = 1.0


@dataclass(frozen=True)
class TargetMetric:
    """Normalized benchmark metric for one target."""

    name: str
    estimate: float
    target: float
    error: float
    abs_error: float
    rel_error: float
    abs_rel_error: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetDelta:
    """Candidate-vs-baseline comparison for one common target."""

    name: str
    candidate_abs_rel_error: float
    baseline_abs_rel_error: float
    abs_rel_error_delta: float
    candidate_estimate: float
    baseline_estimate: float
    target: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetGroupSummary:
    """Grouped benchmark comparison summary for one metadata bucket."""

    group_field: str
    group_value: str
    target_count: int
    candidate_mean_abs_relative_error: float
    baseline_mean_abs_relative_error: float
    mean_abs_relative_error_delta: float
    target_win_rate: float


@dataclass(frozen=True)
class MetricComparisonSummary:
    """Common-target benchmark comparison summary."""

    deltas: list[TargetDelta]
    common_target_count: int
    target_win_rate: float
    mean_abs_relative_error_delta: float
    candidate_common_mean_abs_relative_error: float
    baseline_common_mean_abs_relative_error: float
    candidate_excluded_target_count: int
    baseline_excluded_target_count: int


@dataclass(frozen=True)
class UnsupportedTarget:
    """Unsupported target with an optional reason and metadata."""

    name: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BenchmarkResultEvaluator(Protocol):
    """Protocol for evaluating one canonical target slice into a benchmark result."""

    def evaluate_target_set(
        self,
        target_set: TargetSet,
        slice_spec: BenchmarkSliceSpec,
    ) -> BenchmarkResult:
        """Evaluate a single named target slice."""


@runtime_checkable
class BatchBenchmarkResultEvaluator(Protocol):
    """Protocol for batch evaluation of canonical target slices."""

    def evaluate_target_sets(
        self,
        target_sets: dict[str, TargetSet],
        slices: tuple[BenchmarkSliceSpec, ...],
    ) -> dict[str, BenchmarkResult]:
        """Evaluate multiple named target slices in one batch."""


@dataclass
class BenchmarkResult:
    """Normalized benchmark result built from evaluated metrics."""

    dataset_path: str | None = None
    label: str | None = None
    time_period: int | str | None = None
    target_count: int = 0
    supported_target_count: int | None = None
    unsupported_target_count: int | None = None
    mean_abs_relative_error: float = 0.0
    max_abs_relative_error: float = 0.0
    metrics: list[TargetMetric] = field(default_factory=list)
    unsupported_targets: list[UnsupportedTarget] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "dataset_path": self.dataset_path,
            "label": self.label,
            "time_period": self.time_period,
            "target_count": self.target_count,
            "mean_abs_relative_error": self.mean_abs_relative_error,
            "max_abs_relative_error": self.max_abs_relative_error,
            "metrics": [_asdict(metric) for metric in self.metrics],
            "metadata": self.metadata,
        }
        if self.supported_target_count is not None:
            payload["supported_target_count"] = self.supported_target_count
        if self.unsupported_target_count is not None:
            payload["unsupported_target_count"] = self.unsupported_target_count
        if self.unsupported_targets:
            payload["unsupported_targets"] = [
                _asdict(item) for item in self.unsupported_targets
            ]
        return payload

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkResult:
        return cls(
            dataset_path=payload.get("dataset_path"),
            label=payload.get("label"),
            time_period=payload.get("time_period"),
            target_count=int(payload.get("target_count", 0)),
            supported_target_count=payload.get("supported_target_count"),
            unsupported_target_count=payload.get("unsupported_target_count"),
            mean_abs_relative_error=float(payload.get("mean_abs_relative_error", 0.0)),
            max_abs_relative_error=float(payload.get("max_abs_relative_error", 0.0)),
            metrics=[
                TargetMetric(
                    name=item["name"],
                    estimate=float(item["estimate"]),
                    target=float(item["target"]),
                    error=float(item["error"]),
                    abs_error=float(item["abs_error"]),
                    rel_error=float(item["rel_error"]),
                    abs_rel_error=float(item["abs_rel_error"]),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload.get("metrics", [])
            ],
            unsupported_targets=[
                UnsupportedTarget(
                    name=item["name"],
                    reason=item["reason"],
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload.get("unsupported_targets", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class BenchmarkComparison:
    """Normalized candidate-vs-baseline benchmark comparison."""

    candidate: BenchmarkResult
    baseline: BenchmarkResult
    mean_abs_relative_error_delta: float
    target_win_rate: float
    common_target_count: int
    deltas: list[TargetDelta] = field(default_factory=list)
    grouped_summaries: dict[str, list[TargetGroupSummary]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
            "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta,
            "target_win_rate": self.target_win_rate,
            "common_target_count": self.common_target_count,
            "deltas": [_asdict(delta) for delta in self.deltas],
            "grouped_summaries": {
                field_name: [_asdict(summary) for summary in summaries]
                for field_name, summaries in self.grouped_summaries.items()
            },
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkComparison:
        return cls(
            candidate=BenchmarkResult.from_dict(payload["candidate"]),
            baseline=BenchmarkResult.from_dict(payload["baseline"]),
            mean_abs_relative_error_delta=float(payload["mean_abs_relative_error_delta"]),
            target_win_rate=float(payload["target_win_rate"]),
            common_target_count=int(payload["common_target_count"]),
            deltas=[
                TargetDelta(
                    name=item["name"],
                    candidate_abs_rel_error=float(item["candidate_abs_rel_error"]),
                    baseline_abs_rel_error=float(item["baseline_abs_rel_error"]),
                    abs_rel_error_delta=float(item["abs_rel_error_delta"]),
                    candidate_estimate=float(item["candidate_estimate"]),
                    baseline_estimate=float(item["baseline_estimate"]),
                    target=float(item["target"]),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload.get("deltas", [])
            ],
            grouped_summaries={
                field_name: [
                    TargetGroupSummary(
                        group_field=item["group_field"],
                        group_value=item["group_value"],
                        target_count=int(item["target_count"]),
                        candidate_mean_abs_relative_error=float(
                            item["candidate_mean_abs_relative_error"]
                        ),
                        baseline_mean_abs_relative_error=float(
                            item["baseline_mean_abs_relative_error"]
                        ),
                        mean_abs_relative_error_delta=float(
                            item["mean_abs_relative_error_delta"]
                        ),
                        target_win_rate=float(item["target_win_rate"]),
                    )
                    for item in summaries
                ]
                for field_name, summaries in payload.get("grouped_summaries", {}).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkComparison:
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class BenchmarkSliceSpec:
    """Named benchmark slice metadata."""

    name: str
    query: TargetQuery | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        query = self.query
        if query is not None and not isinstance(query, TargetQuery):
            query = TargetQuery(**query)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "tags", tuple(self.tags))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query": _target_query_to_dict(self.query),
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkSliceSpec:
        return cls(
            name=payload["name"],
            query=_target_query_from_dict(payload.get("query")),
            description=payload.get("description"),
            tags=tuple(payload.get("tags", [])),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class BenchmarkSliceComparison:
    """Benchmark comparison paired with one named slice."""

    slice: BenchmarkSliceSpec
    comparison: BenchmarkComparison

    @property
    def candidate_mean_abs_relative_error(self) -> float:
        return self.comparison.candidate.mean_abs_relative_error

    @property
    def baseline_mean_abs_relative_error(self) -> float:
        return self.comparison.baseline.mean_abs_relative_error

    @property
    def mean_abs_relative_error_delta(self) -> float:
        return self.comparison.mean_abs_relative_error_delta

    @property
    def candidate_beats_baseline(self) -> bool:
        return self.mean_abs_relative_error_delta < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice.to_dict(),
            "comparison": self.comparison.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkSliceComparison:
        return cls(
            slice=BenchmarkSliceSpec.from_dict(payload["slice"]),
            comparison=BenchmarkComparison.from_dict(payload["comparison"]),
        )


@dataclass
class BenchmarkSuiteResult:
    """Named collection of benchmark slice comparisons."""

    candidate_label: str
    baseline_label: str
    period: int | str
    slice_results: list[BenchmarkSliceComparison] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_mean_abs_relative_error(self) -> float | None:
        errors = [
            result.candidate_mean_abs_relative_error for result in self.slice_results
        ]
        if not errors:
            return None
        return float(sum(errors) / len(errors))

    @property
    def baseline_mean_abs_relative_error(self) -> float | None:
        errors = [
            result.baseline_mean_abs_relative_error for result in self.slice_results
        ]
        if not errors:
            return None
        return float(sum(errors) / len(errors))

    @property
    def mean_abs_relative_error_delta(self) -> float | None:
        candidate_error = self.candidate_mean_abs_relative_error
        baseline_error = self.baseline_mean_abs_relative_error
        if candidate_error is None or baseline_error is None:
            return None
        return candidate_error - baseline_error

    @property
    def slice_win_rate(self) -> float | None:
        return self.slice_win_rate_for_tag(None)

    @property
    def target_win_rate(self) -> float | None:
        return self.target_win_rate_for_tag(None)

    @property
    def supported_target_rate(self) -> float | None:
        return self.supported_target_rate_for_tag(None)

    @property
    def baseline_supported_target_rate(self) -> float | None:
        return self.baseline_supported_target_rate_for_tag(None)

    def candidate_mean_abs_relative_error_for_tag(self, tag: str | None) -> float | None:
        return self._mean_abs_relative_error(tag=tag, kind="candidate")

    def baseline_mean_abs_relative_error_for_tag(self, tag: str | None) -> float | None:
        return self._mean_abs_relative_error(tag=tag, kind="baseline")

    def mean_abs_relative_error_delta_for_tag(self, tag: str | None) -> float | None:
        candidate_error = self.candidate_mean_abs_relative_error_for_tag(tag)
        baseline_error = self.baseline_mean_abs_relative_error_for_tag(tag)
        if candidate_error is None or baseline_error is None:
            return None
        return candidate_error - baseline_error

    def slice_win_rate_for_tag(self, tag: str | None) -> float | None:
        slice_results = self._slice_results_for_tag(tag)
        if not slice_results:
            return None
        wins = sum(result.candidate_beats_baseline for result in slice_results)
        return float(wins / len(slice_results))

    def target_win_rate_for_tag(self, tag: str | None) -> float | None:
        candidate_records = self._target_records_for_tag(tag, kind="candidate")
        baseline_records = self._target_records_for_tag(tag, kind="baseline")
        wins = 0
        comparisons = 0
        for name, candidate_record in candidate_records.items():
            baseline_record = baseline_records.get(name)
            if baseline_record is None:
                continue
            candidate_error = candidate_record["abs_rel_error"]
            baseline_error = baseline_record["abs_rel_error"]
            if candidate_error is None or baseline_error is None:
                continue
            comparisons += 1
            wins += int(candidate_error < baseline_error)
        if comparisons == 0:
            return None
        return wins / comparisons

    def supported_target_rate_for_tag(self, tag: str | None) -> float | None:
        return self._supported_target_rate_for_tag(tag, kind="candidate")

    def baseline_supported_target_rate_for_tag(self, tag: str | None) -> float | None:
        return self._supported_target_rate_for_tag(tag, kind="baseline")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_label": self.candidate_label,
            "baseline_label": self.baseline_label,
            "period": self.period,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "summary": {
                "candidate_mean_abs_relative_error": self.candidate_mean_abs_relative_error,
                "baseline_mean_abs_relative_error": self.baseline_mean_abs_relative_error,
                "mean_abs_relative_error_delta": self.mean_abs_relative_error_delta,
                "slice_win_rate": self.slice_win_rate,
                "target_win_rate": self.target_win_rate,
                "supported_target_rate": self.supported_target_rate,
                "baseline_supported_target_rate": self.baseline_supported_target_rate,
            },
            "slices": [result.to_dict() for result in self.slice_results],
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkSuiteResult:
        return cls(
            candidate_label=payload["candidate_label"],
            baseline_label=payload["baseline_label"],
            period=payload["period"],
            created_at=payload["created_at"],
            metadata=dict(payload.get("metadata", {})),
            slice_results=[
                BenchmarkSliceComparison.from_dict(item)
                for item in payload.get("slices", [])
            ],
        )

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkSuiteResult:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def _slice_results_for_tag(self, tag: str | None) -> list[BenchmarkSliceComparison]:
        if tag is None:
            return list(self.slice_results)
        return [result for result in self.slice_results if tag in result.slice.tags]

    def _mean_abs_relative_error(
        self,
        *,
        tag: str | None,
        kind: str,
    ) -> float | None:
        errors: list[float] = []
        for result in self._slice_results_for_tag(tag):
            value = (
                result.candidate_mean_abs_relative_error
                if kind == "candidate"
                else result.baseline_mean_abs_relative_error
            )
            errors.append(value)
        if not errors:
            return None
        return float(sum(errors) / len(errors))

    def _supported_target_rate_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> float | None:
        records = self._target_records_for_tag(tag, kind=kind)
        if not records:
            return None
        supported = sum(1 for record in records.values() if record["supported"])
        return supported / len(records)

    def _target_records_for_tag(
        self,
        tag: str | None,
        *,
        kind: str,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for result in self._slice_results_for_tag(tag):
            benchmark_result = (
                result.comparison.candidate
                if kind == "candidate"
                else result.comparison.baseline
            )
            for item in benchmark_result.unsupported_targets:
                record_key = (result.slice.name, item.name)
                records.setdefault(
                    record_key,
                    {
                        "supported": False,
                        "abs_rel_error": None,
                    },
                )
            for metric in benchmark_result.metrics:
                records[(result.slice.name, metric.name)] = {
                    "supported": True,
                    "abs_rel_error": metric.abs_rel_error,
                }
        return records


def normalize_metric_payload(metric_payload: dict[str, Any]) -> TargetMetric:
    """Build a normalized metric object from raw benchmark payload data."""
    estimate = float(metric_payload["estimate"])
    target = float(metric_payload["target"])
    error = estimate - target
    rel_error = relative_error_ratio(estimate, target)
    return TargetMetric(
        name=str(metric_payload["name"]),
        estimate=estimate,
        target=target,
        error=error,
        abs_error=abs(error),
        rel_error=rel_error,
        abs_rel_error=abs(rel_error),
        metadata=dict(metric_payload.get("metadata", {})),
    )


def relative_error_ratio(estimate: float, target: float) -> float:
    """Compute a finite relative error ratio with a denominator floor."""
    denominator = max(abs(float(target)), _RELATIVE_ERROR_DENOMINATOR_FLOOR)
    return (float(estimate) - float(target)) / denominator


def mean_abs_relative_error(metrics: list[TargetMetric]) -> float:
    """Mean absolute relative error for a metric collection."""
    if not metrics:
        return 0.0
    return float(sum(metric.abs_rel_error for metric in metrics) / len(metrics))


def max_abs_relative_error(metrics: list[TargetMetric]) -> float:
    """Maximum absolute relative error for a metric collection."""
    if not metrics:
        return 0.0
    return float(max(metric.abs_rel_error for metric in metrics))


def compare_metric_sets(
    candidate_metrics: list[TargetMetric],
    baseline_metrics: list[TargetMetric],
) -> MetricComparisonSummary:
    """Compare candidate and baseline metrics on their common supported targets."""
    candidate_by_name = {metric.name: metric for metric in candidate_metrics}
    baseline_by_name = {metric.name: metric for metric in baseline_metrics}
    common_names = sorted(set(candidate_by_name) & set(baseline_by_name))
    if not common_names:
        raise ValueError("Cannot compare benchmark results with zero common supported targets")

    deltas = [
        TargetDelta(
            name=name,
            candidate_abs_rel_error=candidate_by_name[name].abs_rel_error,
            baseline_abs_rel_error=baseline_by_name[name].abs_rel_error,
            abs_rel_error_delta=(
                candidate_by_name[name].abs_rel_error
                - baseline_by_name[name].abs_rel_error
            ),
            candidate_estimate=candidate_by_name[name].estimate,
            baseline_estimate=baseline_by_name[name].estimate,
            target=candidate_by_name[name].target,
            metadata=dict(candidate_by_name[name].metadata),
        )
        for name in common_names
    ]
    wins = sum(
        delta.candidate_abs_rel_error < delta.baseline_abs_rel_error
        for delta in deltas
    )
    candidate_common_mean = mean_abs_relative_error(
        [candidate_by_name[name] for name in common_names]
    )
    baseline_common_mean = mean_abs_relative_error(
        [baseline_by_name[name] for name in common_names]
    )
    return MetricComparisonSummary(
        deltas=deltas,
        common_target_count=len(deltas),
        target_win_rate=(wins / len(deltas)),
        mean_abs_relative_error_delta=(candidate_common_mean - baseline_common_mean),
        candidate_common_mean_abs_relative_error=candidate_common_mean,
        baseline_common_mean_abs_relative_error=baseline_common_mean,
        candidate_excluded_target_count=max(len(candidate_metrics) - len(deltas), 0),
        baseline_excluded_target_count=max(len(baseline_metrics) - len(deltas), 0),
    )


def build_grouped_summaries(
    deltas: list[TargetDelta],
    *,
    group_fields: tuple[str, ...] | None = None,
) -> dict[str, list[TargetGroupSummary]]:
    """Build grouped summaries from per-target deltas."""
    if not group_fields:
        return {}
    grouped: dict[str, list[TargetGroupSummary]] = {}
    for field_name in group_fields:
        summaries = _summaries_for_field(deltas, field_name)
        if summaries:
            grouped[field_name] = summaries
    return grouped


def build_benchmark_result(
    *,
    metrics: list[TargetMetric],
    dataset_path: str | None = None,
    label: str | None = None,
    time_period: int | str | None = None,
    target_count: int | None = None,
    unsupported_targets: list[UnsupportedTarget] | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Build a normalized benchmark result from evaluated metrics."""
    unsupported = list(unsupported_targets or [])
    supported_count = len(metrics)
    total_target_count = (
        target_count if target_count is not None else supported_count + len(unsupported)
    )
    return BenchmarkResult(
        dataset_path=dataset_path,
        label=label,
        time_period=time_period,
        target_count=total_target_count,
        supported_target_count=supported_count,
        unsupported_target_count=len(unsupported),
        mean_abs_relative_error=mean_abs_relative_error(metrics),
        max_abs_relative_error=max_abs_relative_error(metrics),
        metrics=list(metrics),
        unsupported_targets=unsupported,
        metadata=dict(metadata or {}),
    )


def compare_benchmark_results(
    candidate: BenchmarkResult,
    baseline: BenchmarkResult,
    *,
    metadata: dict[str, Any] | None = None,
    group_fields: tuple[str, ...] | None = None,
) -> BenchmarkComparison:
    """Compare two normalized benchmark results on their common supported targets."""
    summary = compare_metric_sets(candidate.metrics, baseline.metrics)
    comparison_metadata = {
        **dict(metadata or {}),
        "candidate_metric_count": len(candidate.metrics),
        "baseline_metric_count": len(baseline.metrics),
        "candidate_excluded_target_count": summary.candidate_excluded_target_count,
        "baseline_excluded_target_count": summary.baseline_excluded_target_count,
        "candidate_common_mean_abs_relative_error": (
            summary.candidate_common_mean_abs_relative_error
        ),
        "baseline_common_mean_abs_relative_error": (
            summary.baseline_common_mean_abs_relative_error
        ),
    }
    return BenchmarkComparison(
        candidate=candidate,
        baseline=baseline,
        mean_abs_relative_error_delta=summary.mean_abs_relative_error_delta,
        target_win_rate=summary.target_win_rate,
        common_target_count=summary.common_target_count,
        deltas=summary.deltas,
        grouped_summaries=build_grouped_summaries(summary.deltas, group_fields=group_fields),
        metadata=comparison_metadata,
    )


def build_benchmark_suite_result(
    *,
    candidate_label: str,
    baseline_label: str,
    period: int | str,
    slice_results: list[BenchmarkSliceComparison],
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkSuiteResult:
    """Build a named benchmark suite from slice comparisons."""
    return BenchmarkSuiteResult(
        candidate_label=candidate_label,
        baseline_label=baseline_label,
        period=period,
        slice_results=list(slice_results),
        created_at=(
            created_at
            if created_at is not None
            else datetime.now(UTC).replace(microsecond=0).isoformat()
        ),
        metadata=dict(metadata or {}),
    )


def load_benchmark_slice_target_sets(
    provider: TargetProvider,
    slices: list[BenchmarkSliceSpec] | tuple[BenchmarkSliceSpec, ...],
    *,
    loader: Callable[[TargetProvider, TargetQuery | None], TargetSet] | None = None,
) -> dict[str, TargetSet]:
    """Resolve benchmark slices to canonical target sets."""
    target_loader = loader or (
        lambda effective_provider, query: effective_provider.load_target_set(query)
    )
    return {
        slice_spec.name: target_loader(provider, slice_spec.query)
        for slice_spec in slices
    }


def filter_nonempty_benchmark_slices(
    provider: TargetProvider,
    slices: list[BenchmarkSliceSpec] | tuple[BenchmarkSliceSpec, ...],
    *,
    loader: Callable[[TargetProvider, TargetQuery | None], TargetSet] | None = None,
) -> tuple[BenchmarkSliceSpec, ...]:
    """Drop slices that resolve to no canonical targets."""
    target_loader = loader or (
        lambda effective_provider, query: effective_provider.load_target_set(query)
    )
    return tuple(
        slice_spec
        for slice_spec in slices
        if target_loader(provider, slice_spec.query).targets
    )


def union_target_sets(target_sets: dict[str, TargetSet]) -> TargetSet:
    """Union multiple target sets while rejecting conflicting duplicate names."""
    union = TargetSet()
    seen_signatures: dict[str, tuple[object, ...]] = {}
    for target_set in target_sets.values():
        for target in target_set.targets:
            signature = _target_signature(target)
            existing = seen_signatures.get(target.name)
            if existing is None:
                seen_signatures[target.name] = signature
                union.add(target)
                continue
            if existing != signature:
                raise ValueError(
                    "Benchmark target-set union encountered conflicting "
                    f"definitions for target '{target.name}'"
                )
    return union


def evaluate_benchmark_slice_payloads(
    target_sets: dict[str, TargetSet],
    slices: list[BenchmarkSliceSpec] | tuple[BenchmarkSliceSpec, ...],
    *,
    evaluator: Callable[[TargetSet, BenchmarkSliceSpec], Any] | None = None,
    batch_evaluator: (
        Callable[[dict[str, TargetSet], tuple[BenchmarkSliceSpec, ...]], dict[str, Any]]
        | None
    ) = None,
) -> dict[str, Any]:
    """Evaluate one payload per named benchmark slice from preloaded target sets."""
    if evaluator is None and batch_evaluator is None:
        raise ValueError("Benchmark slice evaluation requires an evaluator")

    slice_specs = tuple(slices)
    if batch_evaluator is not None:
        payloads = batch_evaluator(target_sets, slice_specs)
    else:
        assert evaluator is not None
        payloads = {
            slice_spec.name: evaluator(target_sets[slice_spec.name], slice_spec)
            for slice_spec in slice_specs
        }

    missing = [
        slice_spec.name
        for slice_spec in slice_specs
        if slice_spec.name not in payloads
    ]
    if missing:
        raise ValueError(
            "Benchmark slice evaluator did not return payloads for: "
            + ", ".join(sorted(missing))
        )
    return payloads


def evaluate_benchmark_slice_results(
    target_sets: dict[str, TargetSet],
    slices: list[BenchmarkSliceSpec] | tuple[BenchmarkSliceSpec, ...],
    *,
    evaluator: BenchmarkResultEvaluator | None = None,
    batch_evaluator: BatchBenchmarkResultEvaluator | None = None,
) -> dict[str, BenchmarkResult]:
    """Evaluate one benchmark result per named slice using a result-oriented adapter."""
    if evaluator is None and batch_evaluator is None:
        raise ValueError("Benchmark slice evaluation requires an evaluator")

    slice_specs = tuple(slices)
    if batch_evaluator is not None:
        results = batch_evaluator.evaluate_target_sets(target_sets, slice_specs)
    else:
        assert evaluator is not None
        results = {
            slice_spec.name: evaluator.evaluate_target_set(
                target_sets[slice_spec.name],
                slice_spec,
            )
            for slice_spec in slice_specs
        }

    missing = [
        slice_spec.name
        for slice_spec in slice_specs
        if slice_spec.name not in results
    ]
    if missing:
        raise ValueError(
            "Benchmark slice evaluator did not return results for: "
            + ", ".join(sorted(missing))
        )
    return results


def build_benchmark_suite_from_results(
    *,
    slices: list[BenchmarkSliceSpec] | tuple[BenchmarkSliceSpec, ...],
    candidate_results: dict[str, BenchmarkResult],
    baseline_results: dict[str, BenchmarkResult],
    candidate_label: str,
    baseline_label: str,
    period: int | str,
    group_fields: tuple[str, ...] | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkSuiteResult:
    """Build a benchmark suite by comparing per-slice benchmark results."""
    slice_comparisons = [
        BenchmarkSliceComparison(
            slice=slice_spec,
            comparison=compare_benchmark_results(
                candidate_results[slice_spec.name],
                baseline_results[slice_spec.name],
                group_fields=group_fields,
            ),
        )
        for slice_spec in slices
    ]
    return build_benchmark_suite_result(
        candidate_label=candidate_label,
        baseline_label=baseline_label,
        period=period,
        slice_results=slice_comparisons,
        created_at=created_at,
        metadata=metadata,
    )


def build_benchmark_suite_from_payloads(
    *,
    slices: list[BenchmarkSliceSpec] | tuple[BenchmarkSliceSpec, ...],
    candidate_payloads: dict[str, Any],
    baseline_payloads: dict[str, Any],
    candidate_result_getter: Callable[[Any], BenchmarkResult],
    baseline_result_getter: Callable[[Any], BenchmarkResult] | None = None,
    candidate_label: str,
    baseline_label: str,
    period: int | str,
    group_fields: tuple[str, ...] | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkSuiteResult:
    """Build a benchmark suite by comparing candidate and baseline payloads per slice."""
    baseline_getter = baseline_result_getter or candidate_result_getter
    return build_benchmark_suite_from_results(
        slices=slices,
        candidate_results={
            slice_spec.name: candidate_result_getter(candidate_payloads[slice_spec.name])
            for slice_spec in slices
        },
        baseline_results={
            slice_spec.name: baseline_getter(baseline_payloads[slice_spec.name])
            for slice_spec in slices
        },
        candidate_label=candidate_label,
        baseline_label=baseline_label,
        period=period,
        group_fields=group_fields,
        created_at=created_at,
        metadata=metadata,
    )


def _summaries_for_field(
    deltas: list[TargetDelta],
    field_name: str,
) -> list[TargetGroupSummary]:
    buckets: dict[str, list[TargetDelta]] = {}
    for delta in deltas:
        value = delta.metadata.get(field_name)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(delta)
    summaries: list[TargetGroupSummary] = []
    for group_value, group_deltas in sorted(buckets.items()):
        target_count = len(group_deltas)
        candidate_mean = sum(
            delta.candidate_abs_rel_error for delta in group_deltas
        ) / target_count
        baseline_mean = sum(
            delta.baseline_abs_rel_error for delta in group_deltas
        ) / target_count
        wins = sum(
            delta.candidate_abs_rel_error < delta.baseline_abs_rel_error
            for delta in group_deltas
        )
        summaries.append(
            TargetGroupSummary(
                group_field=field_name,
                group_value=group_value,
                target_count=target_count,
                candidate_mean_abs_relative_error=candidate_mean,
                baseline_mean_abs_relative_error=baseline_mean,
                mean_abs_relative_error_delta=(candidate_mean - baseline_mean),
                target_win_rate=(wins / target_count),
            )
        )
    return summaries


def _asdict(value: Any) -> dict[str, Any]:
    return dict(value.__dict__)


def _target_query_to_dict(query: TargetQuery | None) -> dict[str, Any] | None:
    if query is None:
        return None
    return {
        "period": query.period,
        "entity": query.entity.value if query.entity is not None else None,
        "names": list(query.names),
        "metadata_filters": dict(query.metadata_filters),
        "provider_filters": dict(query.provider_filters),
    }


def _target_query_from_dict(payload: dict[str, Any] | None) -> TargetQuery | None:
    if payload is None:
        return None
    return TargetQuery(
        period=payload.get("period"),
        entity=payload.get("entity"),
        names=tuple(payload.get("names", [])),
        metadata_filters=dict(payload.get("metadata_filters", {})),
        provider_filters=dict(payload.get("provider_filters", {})),
    )


def _target_signature(target: TargetSpec) -> tuple[object, ...]:
    return (
        target.name,
        target.entity.value,
        target.period,
        target.measure,
        target.aggregation.value,
        tuple(
            (
                target_filter.feature,
                target_filter.operator.value,
                _freeze_value(target_filter.value),
            )
            for target_filter in target.filters
        ),
        float(target.value),
        target.tolerance,
        target.source,
        target.units,
        target.description,
        _freeze_value(target.metadata),
    )


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze_value(item) for item in value)
    return value
