"""Generic target providers for tabular rollup artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from microplex.core import EntityType
from microplex.targets.provider import TargetQuery, apply_target_query
from microplex.targets.spec import (
    FilterOperator,
    TargetAggregation,
    TargetFilter,
    TargetSet,
    TargetSpec,
)


@dataclass(frozen=True)
class TabularRollupSpec:
    """One rollup target family available from a tabular artifact."""

    geo_level: str
    source_column: str | None
    filter_feature: str | None
    group_name: str
    name_prefix: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class TabularRollupTargetProvider:
    """Build target specs by grouping an input table to configured rollups."""

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        data_path: str | Path | None = None,
        data_loader: Callable[[str | Path | None], pd.DataFrame] | None = None,
        prepare_data: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        rollups: Mapping[str, TabularRollupSpec],
        value_column: str,
        variable: str,
        entity: EntityType | str,
        period: int | str,
        source: str | None = None,
        units: str | None = None,
        aggregation: TargetAggregation | str = TargetAggregation.COUNT,
        measure: str | None = None,
        default_geo_levels: Iterable[str] | None = None,
        variable_aliases: Iterable[str] = (),
        base_metadata: Mapping[str, Any] | None = None,
        min_value: float | None = None,
        normalize_geographic_id: Callable[[Any], str] | None = None,
    ) -> None:
        self._data = data
        self.data_path = Path(data_path) if data_path is not None else None
        self.data_loader = data_loader or _read_parquet
        self.prepare_data = prepare_data
        self.rollups = dict(rollups)
        self.value_column = value_column
        self.variable = variable
        self.entity = entity
        self.period = period
        self.source = source
        self.units = units
        self.aggregation = aggregation
        self.measure = measure
        self.default_geo_levels = tuple(default_geo_levels or self.rollups)
        self.variable_aliases = tuple(variable_aliases)
        self.base_metadata = dict(base_metadata or {})
        self.min_value = min_value
        self.normalize_geographic_id = normalize_geographic_id or normalize_rollup_id

    def load_target_set(self, query: TargetQuery | None = None) -> TargetSet:
        """Load tabular rollup targets for the requested provider filters."""
        query = query or TargetQuery()
        provider_filters = dict(query.provider_filters)
        variables = as_string_tuple(provider_filters.get("variables"))
        allowed_variables = {self.variable, *self.variable_aliases}
        if variables and not any(
            variable in allowed_variables for variable in variables
        ):
            return TargetSet([])

        geo_levels = resolve_rollup_keys(
            provider_filters.get("geo_levels")
            if "geo_levels" in provider_filters
            else provider_filters.get("geographic_levels"),
            rollups=self.rollups,
            default_geo_levels=self.default_geo_levels,
        )
        geographic_ids = as_string_tuple(provider_filters.get("geographic_ids"))
        targets = build_tabular_rollup_targets(
            self._load_data(),
            rollups=self.rollups,
            value_column=self.value_column,
            variable=self.variable,
            entity=self.entity,
            period=self.period,
            source=self.source,
            units=self.units,
            aggregation=self.aggregation,
            measure=self.measure,
            geo_levels=geo_levels,
            geographic_ids=geographic_ids or None,
            base_metadata=self.base_metadata,
            min_value=self.min_value,
            normalize_geographic_id=self.normalize_geographic_id,
        )
        return apply_target_query(
            TargetSet(targets),
            TargetQuery(
                period=query.period,
                entity=query.entity,
                names=query.names,
                metadata_filters=query.metadata_filters,
            ),
        )

    def _load_data(self) -> pd.DataFrame:
        data = (
            self._data.copy()
            if self._data is not None
            else self.data_loader(self.data_path)
        )
        if self.prepare_data is not None:
            data = self.prepare_data(data)
        return data


def build_tabular_rollup_targets(
    data: pd.DataFrame,
    *,
    rollups: Mapping[str, TabularRollupSpec],
    value_column: str,
    variable: str,
    entity: EntityType | str,
    period: int | str,
    source: str | None = None,
    units: str | None = None,
    aggregation: TargetAggregation | str = TargetAggregation.COUNT,
    measure: str | None = None,
    geo_levels: Iterable[str] | None = None,
    geographic_ids: Iterable[str] | None = None,
    base_metadata: Mapping[str, Any] | None = None,
    min_value: float | None = None,
    normalize_geographic_id: Callable[[Any], str] | None = None,
) -> list[TargetSpec]:
    """Roll a tabular artifact into canonical target specs."""
    if value_column not in data.columns:
        raise ValueError(f"Tabular rollup data must include {value_column!r}")
    normalize_id = normalize_geographic_id or normalize_rollup_id
    rollup_keys = resolve_rollup_keys(
        geo_levels,
        rollups=rollups,
        default_geo_levels=tuple(rollups),
    )
    selected_geographic_ids = (
        {normalize_id(value) for value in geographic_ids}
        if geographic_ids is not None
        else None
    )
    values = data.copy()
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    targets: list[TargetSpec] = []
    for rollup_key in rollup_keys:
        rollup = rollups[rollup_key]
        if rollup.source_column is None:
            if selected_geographic_ids:
                continue
            value = float(values[value_column].sum())
            if min_value is not None and value <= min_value:
                continue
            targets.append(
                tabular_rollup_target(
                    rollup=rollup,
                    geographic_id=None,
                    value=value,
                    variable=variable,
                    entity=entity,
                    period=period,
                    source=source,
                    units=units,
                    aggregation=aggregation,
                    measure=measure,
                    base_metadata=base_metadata,
                )
            )
            continue
        if rollup.source_column not in values.columns:
            continue
        rollup_values = values[[rollup.source_column, value_column]].dropna(
            subset=[rollup.source_column, value_column]
        )
        rollup_values[rollup.source_column] = rollup_values[rollup.source_column].map(
            normalize_id
        )
        rollup_values = rollup_values[rollup_values[rollup.source_column].astype(bool)]
        if selected_geographic_ids is not None:
            rollup_values = rollup_values[
                rollup_values[rollup.source_column].isin(selected_geographic_ids)
            ]
        grouped = rollup_values.groupby(rollup.source_column, dropna=True)[
            value_column
        ].sum()
        for geographic_id, value in grouped.items():
            target_value = float(value)
            if min_value is not None and target_value <= min_value:
                continue
            targets.append(
                tabular_rollup_target(
                    rollup=rollup,
                    geographic_id=str(geographic_id),
                    value=target_value,
                    variable=variable,
                    entity=entity,
                    period=period,
                    source=source,
                    units=units,
                    aggregation=aggregation,
                    measure=measure,
                    base_metadata=base_metadata,
                )
            )
    return targets


def tabular_rollup_target(
    *,
    rollup: TabularRollupSpec,
    geographic_id: str | None,
    value: float,
    variable: str,
    entity: EntityType | str,
    period: int | str,
    source: str | None = None,
    units: str | None = None,
    aggregation: TargetAggregation | str = TargetAggregation.COUNT,
    measure: str | None = None,
    base_metadata: Mapping[str, Any] | None = None,
) -> TargetSpec:
    """Build one canonical target from a tabular rollup cell."""
    filters = (
        ()
        if geographic_id is None or rollup.filter_feature is None
        else (
            TargetFilter(
                feature=rollup.filter_feature,
                operator=FilterOperator.EQ,
                value=geographic_id,
            ),
        )
    )
    name = rollup.name_prefix
    if geographic_id is not None:
        name = f"{name}_{target_name_fragment(geographic_id)}"
    return TargetSpec(
        name=name,
        entity=entity,
        value=value,
        period=period,
        measure=measure,
        aggregation=aggregation,
        filters=filters,
        source=source,
        units=units,
        metadata={
            "variable": variable,
            "geo_level": rollup.geo_level,
            "geographic_id": geographic_id,
            "target_group": rollup.group_name,
            "tabular_rollup": True,
            **dict(base_metadata or {}),
            **dict(rollup.metadata),
        },
    )


def resolve_rollup_keys(
    geo_levels: Iterable[str] | Any | None,
    *,
    rollups: Mapping[str, TabularRollupSpec],
    default_geo_levels: Iterable[str],
) -> tuple[str, ...]:
    """Resolve requested rollup keys and validate them against available specs."""
    requested = (
        as_string_tuple(geo_levels)
        if geo_levels is not None
        else tuple(default_geo_levels)
    )
    if requested == ("all",):
        requested = tuple(rollups)
    unknown = sorted(set(requested) - set(rollups))
    if unknown:
        raise ValueError(f"Unsupported tabular rollup geo levels: {unknown}")
    return requested


def as_string_tuple(value: Any) -> tuple[str, ...]:
    """Coerce a scalar or iterable provider filter value to a string tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def normalize_rollup_id(value: Any) -> str:
    """Normalize common pandas scalar values into stable target geography IDs."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def target_name_fragment(geographic_id: str) -> str:
    """Return a stable target-name fragment for one geography ID."""
    return (
        geographic_id.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def _read_parquet(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        raise ValueError("A data_path or custom data_loader is required")
    return pd.read_parquet(path)


__all__ = [
    "TabularRollupSpec",
    "TabularRollupTargetProvider",
    "as_string_tuple",
    "build_tabular_rollup_targets",
    "normalize_rollup_id",
    "resolve_rollup_keys",
    "tabular_rollup_target",
    "target_name_fragment",
]
