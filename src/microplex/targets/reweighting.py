"""Generic target-driven record reweighting helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.targets.benchmarking import relative_error_ratio
from microplex.targets.bundles import EntityTableBundle
from microplex.targets.spec import FilterOperator, TargetAggregation, TargetSpec
from microplex.telemetry import (
    CalibrationEpochEvent,
    CalibrationTargetEvent,
    TelemetryWriter,
    effective_sample_size,
)


@dataclass(frozen=True)
class TargetReweightingConstraint:
    """A linear target constraint over an underlying weight vector."""

    name: str
    entity: EntityType
    weight_indexes: np.ndarray
    coefficients: np.ndarray
    target: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indexes = np.asarray(self.weight_indexes, dtype=int)
        coefficients = np.asarray(self.coefficients, dtype=float)
        if indexes.ndim != 1 or coefficients.ndim != 1:
            raise ValueError(
                "TargetReweightingConstraint arrays must be one-dimensional"
            )
        if len(indexes) != len(coefficients):
            raise ValueError(
                "weight_indexes and coefficients must have the same length"
            )
        object.__setattr__(self, "weight_indexes", indexes)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "target", float(self.target))


@dataclass(frozen=True)
class TargetReweightingDiagnostics:
    """Diagnostics for a target-driven reweighting run."""

    target_count: int
    constraint_count: int
    iterations: int
    converged: bool
    mean_abs_relative_error: float
    max_abs_relative_error: float


@dataclass(frozen=True)
class TargetConstraintCompilationResult:
    """Compiled and skipped targets for a given reweighting request."""

    constraints: tuple[TargetReweightingConstraint, ...]
    skipped_targets: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EntityTableBundleReweightingResult:
    """Result of reweighting a shared entity-table bundle."""

    bundle: EntityTableBundle
    compilation: TargetConstraintCompilationResult
    diagnostics: TargetReweightingDiagnostics


SparseTargetConstraint = TargetReweightingConstraint
SparseTargetCalibrationDiagnostics = TargetReweightingDiagnostics


def compile_target_reweighting_constraints(
    *,
    targets: list[TargetSpec],
    entity_frames: dict[EntityType, pd.DataFrame],
    entity_weight_indexes: dict[EntityType, pd.Series | np.ndarray],
) -> TargetConstraintCompilationResult:
    """Compile canonical targets into linear constraints over a shared weight vector."""
    constraints: list[TargetReweightingConstraint] = []
    skipped: list[tuple[str, str]] = []

    for target in targets:
        frame = entity_frames.get(target.entity)
        weight_indexes = entity_weight_indexes.get(target.entity)
        if frame is None or weight_indexes is None:
            skipped.append((target.name, "missing_entity_frame"))
            continue
        aligned_weight_indexes = _coerce_weight_indexes(weight_indexes, len(frame))
        missing_features = [
            feature
            for feature in target.required_features
            if feature not in frame.columns
        ]
        if missing_features:
            skipped.append(
                (target.name, f"missing_features:{','.join(sorted(missing_features))}")
            )
            continue

        mask = _build_target_mask(frame, target)
        coefficients = _target_coefficients(frame, target, mask)
        if coefficients is None:
            skipped.append((target.name, "unsupported_target"))
            continue
        active = coefficients != 0.0
        if not active.any():
            skipped.append((target.name, "zero_support"))
            continue
        grouped = (
            pd.DataFrame(
                {
                    "weight_index": aligned_weight_indexes[active],
                    "coefficient": coefficients.loc[active],
                }
            )
            .groupby("weight_index", dropna=False)["coefficient"]
            .sum()
        )
        metadata = dict(target.metadata)
        if target.source is not None:
            metadata.setdefault("source", target.source)
        metadata.setdefault("period", str(target.period))
        metadata.setdefault("aggregation", target.aggregation.value)
        constraints.append(
            TargetReweightingConstraint(
                name=target.name,
                entity=target.entity,
                weight_indexes=grouped.index.to_numpy(dtype=int),
                coefficients=grouped.to_numpy(dtype=float),
                target=_constraint_target_value(target),
                metadata=metadata,
            )
        )

    return TargetConstraintCompilationResult(
        constraints=tuple(constraints),
        skipped_targets=tuple(skipped),
    )


def compile_entity_table_bundle_target_constraints(
    bundle: EntityTableBundle,
    *,
    targets: list[TargetSpec],
) -> TargetConstraintCompilationResult:
    """Compile targets against a shared entity-table bundle."""
    return compile_target_reweighting_constraints(
        targets=targets,
        entity_frames=bundle.entity_frames(),
        entity_weight_indexes=bundle.entity_weight_indexes(),
    )


def reweight_to_target_constraints(
    initial_weights: pd.Series | np.ndarray,
    *,
    constraints: list[TargetReweightingConstraint]
    | tuple[TargetReweightingConstraint, ...],
    max_iter: int = 8,
    tol: float = 1e-4,
    factor_bounds: tuple[float, float] = (0.5, 2.0),
    telemetry_writer: TelemetryWriter | None = None,
    run_id: str | None = None,
    calibration_id: str = "target_reweighting",
) -> tuple[np.ndarray, TargetReweightingDiagnostics]:
    """Apply multiplicative updates to match compiled linear target constraints."""
    if (
        telemetry_writer is not None
        and getattr(telemetry_writer, "enabled", True)
        and not run_id
    ):
        raise ValueError("run_id is required when telemetry_writer is provided")

    weights = np.asarray(initial_weights, dtype=float).copy()
    lower_factor, upper_factor = factor_bounds
    converged = False
    iterations = 0
    compiled = tuple(constraints)
    skipped_nonpositive_positive_target = False

    if not compiled:
        diagnostics = TargetReweightingDiagnostics(
            target_count=0,
            constraint_count=0,
            iterations=0,
            converged=True,
            mean_abs_relative_error=0.0,
            max_abs_relative_error=0.0,
        )
        return weights, diagnostics

    for iteration in range(max_iter):
        max_change = 0.0
        skipped_nonpositive_positive_target = False
        for constraint in compiled:
            current = float(
                np.dot(weights[constraint.weight_indexes], constraint.coefficients)
            )
            target_value = float(constraint.target)
            if target_value == 0.0:
                current_abs = abs(current)
                if current_abs <= 0.0:
                    continue
                factor = float(
                    np.clip(
                        1.0 / (1.0 + current_abs),
                        lower_factor,
                        min(upper_factor, 1.0),
                    )
                )
            else:
                if current <= 0.0:
                    skipped_nonpositive_positive_target = True
                    continue
                factor = float(
                    np.clip(target_value / current, lower_factor, upper_factor)
                )
            weights[constraint.weight_indexes] *= factor
            max_change = max(max_change, abs(factor - 1.0))
        iterations = iteration + 1
        _emit_reweighting_epoch_telemetry(
            telemetry_writer,
            run_id=run_id,
            calibration_id=calibration_id,
            epoch=iterations,
            constraints=compiled,
            weights=weights,
        )
        if max_change < tol:
            converged = True
            break

    if skipped_nonpositive_positive_target:
        converged = False

    errors = [
        constraint_abs_relative_error(constraint, weights) for constraint in compiled
    ]
    diagnostics = TargetReweightingDiagnostics(
        target_count=len(compiled),
        constraint_count=len(compiled),
        iterations=iterations,
        converged=converged,
        mean_abs_relative_error=float(np.mean(errors)) if errors else 0.0,
        max_abs_relative_error=float(np.max(errors)) if errors else 0.0,
    )
    _emit_reweighting_target_telemetry(
        telemetry_writer,
        run_id=run_id,
        calibration_id=calibration_id,
        epoch_or_final="final",
        constraints=compiled,
        weights=weights,
    )
    return weights, diagnostics


def reweight_entity_table_bundle_targets(
    bundle: EntityTableBundle,
    *,
    targets: list[TargetSpec],
    max_iter: int = 8,
    tol: float = 1e-4,
    factor_bounds: tuple[float, float] = (0.5, 2.0),
    telemetry_writer: TelemetryWriter | None = None,
    run_id: str | None = None,
    calibration_id: str = "target_reweighting",
) -> EntityTableBundleReweightingResult:
    """Compile and reweight a shared entity-table bundle in one step."""
    compilation = compile_entity_table_bundle_target_constraints(
        bundle,
        targets=targets,
    )
    weights, diagnostics = reweight_to_target_constraints(
        bundle.initial_weights(),
        constraints=compilation.constraints,
        max_iter=max_iter,
        tol=tol,
        factor_bounds=factor_bounds,
        telemetry_writer=telemetry_writer,
        run_id=run_id,
        calibration_id=calibration_id,
    )
    return EntityTableBundleReweightingResult(
        bundle=bundle.with_updated_weights(weights),
        compilation=compilation,
        diagnostics=diagnostics,
    )


def compile_sparse_target_constraints(
    *,
    targets: list[TargetSpec],
    feature_tables: dict[EntityType, pd.DataFrame],
    weight_unit_index: pd.Series,
    entity_weight_id_columns: dict[EntityType, str],
) -> list[TargetReweightingConstraint]:
    """Compatibility wrapper that maps entity ids onto weight indexes before compiling."""
    entity_weight_indexes: dict[EntityType, pd.Series] = {}
    for entity, frame in feature_tables.items():
        weight_id_column = entity_weight_id_columns.get(entity)
        if weight_id_column is None or weight_id_column not in frame.columns:
            continue
        entity_weight_indexes[entity] = pd.to_numeric(
            frame[weight_id_column].map(weight_unit_index),
            errors="coerce",
        )
    return list(
        compile_target_reweighting_constraints(
            targets=targets,
            entity_frames=feature_tables,
            entity_weight_indexes=entity_weight_indexes,
        ).constraints
    )


def calibrate_sparse_target_weights(
    initial_weights: pd.Series | np.ndarray,
    *,
    constraints: list[TargetReweightingConstraint]
    | tuple[TargetReweightingConstraint, ...],
    target_count: int | None = None,
    max_iter: int = 8,
    tol: float = 1e-4,
    factor_bounds: tuple[float, float] = (0.5, 2.0),
    telemetry_writer: TelemetryWriter | None = None,
    run_id: str | None = None,
    calibration_id: str = "sparse_target_calibration",
) -> tuple[np.ndarray, TargetReweightingDiagnostics]:
    """Compatibility wrapper around target reweighting."""
    weights, diagnostics = reweight_to_target_constraints(
        initial_weights,
        constraints=constraints,
        max_iter=max_iter,
        tol=tol,
        factor_bounds=factor_bounds,
        telemetry_writer=telemetry_writer,
        run_id=run_id,
        calibration_id=calibration_id,
    )
    if target_count is None:
        return weights, diagnostics
    return weights, TargetReweightingDiagnostics(
        target_count=target_count,
        constraint_count=diagnostics.constraint_count,
        iterations=diagnostics.iterations,
        converged=diagnostics.converged,
        mean_abs_relative_error=diagnostics.mean_abs_relative_error,
        max_abs_relative_error=diagnostics.max_abs_relative_error,
    )


def constraint_abs_relative_error(
    constraint: TargetReweightingConstraint,
    weights: np.ndarray,
) -> float:
    """Compute absolute relative error for one compiled constraint."""
    estimate = float(
        np.dot(weights[constraint.weight_indexes], constraint.coefficients)
    )
    return abs(relative_error_ratio(estimate, constraint.target))


def sparse_constraint_abs_rel_error(
    constraint: TargetReweightingConstraint,
    weights: np.ndarray,
) -> float:
    """Compatibility alias for sparse constraint relative error."""
    return constraint_abs_relative_error(constraint, weights)


def _emit_reweighting_epoch_telemetry(
    telemetry_writer: TelemetryWriter | None,
    *,
    run_id: str | None,
    calibration_id: str,
    epoch: int,
    constraints: tuple[TargetReweightingConstraint, ...],
    weights: np.ndarray,
) -> None:
    if telemetry_writer is None or run_id is None:
        return
    errors = [
        constraint_abs_relative_error(constraint, weights) for constraint in constraints
    ]
    data_loss = float(np.mean(errors)) if errors else 0.0
    telemetry_writer.emit(
        CalibrationEpochEvent(
            run_id=run_id,
            calibration_id=calibration_id,
            epoch=epoch,
            objective=data_loss,
            data_loss=data_loss,
            l0_penalty=0.0,
            l2_penalty=0.0,
            nonzero_weights=int(np.count_nonzero(weights > 0.0)),
            ess=effective_sample_size(weights),
        )
    )


def _emit_reweighting_target_telemetry(
    telemetry_writer: TelemetryWriter | None,
    *,
    run_id: str | None,
    calibration_id: str,
    epoch_or_final: int | str,
    constraints: tuple[TargetReweightingConstraint, ...],
    weights: np.ndarray,
) -> None:
    if telemetry_writer is None or run_id is None:
        return
    events = []
    for constraint in constraints:
        estimate = float(
            np.dot(weights[constraint.weight_indexes], constraint.coefficients)
        )
        relative_error = relative_error_ratio(estimate, constraint.target)
        events.append(
            CalibrationTargetEvent(
                run_id=run_id,
                calibration_id=calibration_id,
                epoch_or_final=epoch_or_final,
                target_name=constraint.name,
                family=_metadata_scalar(constraint.metadata, "family"),
                split=_metadata_scalar(constraint.metadata, "split"),
                source=_metadata_scalar(constraint.metadata, "source"),
                geography=_metadata_scalar(constraint.metadata, "geography"),
                target_value=float(constraint.target),
                estimate=estimate,
                relative_error=float(relative_error),
                weighted_term=float(abs(relative_error)),
                in_loss_function=True,
                support_status=_metadata_scalar(
                    constraint.metadata,
                    "support_status",
                    default="included",
                ),
            )
        )
    telemetry_writer.emit_many(events)


def _metadata_scalar(
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str | None:
    value = metadata.get(key, default)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float):
        return str(value)
    return default


def _coerce_weight_indexes(
    values: pd.Series | np.ndarray,
    expected_length: int,
) -> np.ndarray:
    series = pd.Series(values)
    if len(series) != expected_length:
        raise ValueError("entity_weight_indexes must align to the entity frame length")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("entity_weight_indexes must be fully numeric after alignment")
    return numeric.to_numpy(dtype=int)


def _build_target_mask(frame: pd.DataFrame, target: TargetSpec) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for target_filter in target.filters:
        values = frame[target_filter.feature]
        mask = mask & _apply_filter(values, target_filter.operator, target_filter.value)
    return mask.fillna(False)


def _apply_filter(
    values: pd.Series,
    operator: FilterOperator,
    expected: Any,
) -> pd.Series:
    if operator is FilterOperator.EQ:
        return _equals_filter(values, expected)
    if operator is FilterOperator.NE:
        return ~_equals_filter(values, expected)
    if operator is FilterOperator.IN:
        return _isin_filter(values, expected)
    if operator is FilterOperator.NOT_IN:
        return ~_isin_filter(values, expected)
    numeric_values = _numeric_series(values)
    expected_value = float(expected)
    if operator is FilterOperator.GT:
        return numeric_values > expected_value
    if operator is FilterOperator.GTE:
        return numeric_values >= expected_value
    if operator is FilterOperator.LT:
        return numeric_values < expected_value
    if operator is FilterOperator.LTE:
        return numeric_values <= expected_value
    raise ValueError(f"Unsupported operator: {operator}")


def _target_coefficients(
    frame: pd.DataFrame,
    target: TargetSpec,
    mask: pd.Series,
) -> pd.Series | None:
    mask_values = mask.astype(float)
    if target.aggregation is TargetAggregation.COUNT:
        return mask_values
    if target.measure is None:
        return None
    measure_values = _numeric_series(frame[target.measure]).fillna(0.0)
    if target.aggregation is TargetAggregation.SUM:
        return mask_values * measure_values
    if target.aggregation is TargetAggregation.MEAN:
        return mask_values * (measure_values - float(target.value))
    return None


def _constraint_target_value(target: TargetSpec) -> float:
    if target.aggregation is TargetAggregation.MEAN:
        return 0.0
    return float(target.value)


def _equals_filter(values: pd.Series, expected: Any) -> pd.Series:
    if pd.isna(expected):
        return values.isna()
    return values.eq(expected)


def _isin_filter(values: pd.Series, expected: Any) -> pd.Series:
    expected_values = list(expected)
    non_null_expected = [item for item in expected_values if not pd.isna(item)]
    mask = values.isin(non_null_expected)
    if len(non_null_expected) != len(expected_values):
        mask = mask | values.isna()
    return mask


def _numeric_series(values: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce")


# Backward-compatible sparse target calibration surface.
build_target_mask = _build_target_mask
apply_filter = _apply_filter
numeric_series = _numeric_series
