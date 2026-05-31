"""Run and calibration telemetry primitives for Microplex.

The module is intentionally country-agnostic. Country packages can enrich
target metadata before creating these events, while core owns the event shapes,
privacy guardrails, and local / remote writer plumbing.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol

import httpx
import numpy as np
import pandas as pd

TELEMETRY_SCHEMA_VERSION = "microplex.telemetry.v1"


JsonRecord = dict[str, Any]

_SUPABASE_TABLES = {
    "run": "runs",
    "stage": "run_stages",
    "calibration_epoch": "calibration_epochs",
    "calibration_target": "calibration_targets",
    "artifact": "artifacts",
}

_SUPABASE_TABLE_COLUMNS = {
    "run": (
        "run_id",
        "build_id",
        "engine",
        "period",
        "created_at",
        "code_ref",
        "config_hash",
        "incognito",
        "status",
    ),
    "stage": (
        "run_id",
        "stage",
        "status",
        "started_at",
        "completed_at",
        "elapsed_seconds",
        "rss_mb",
        "notes",
    ),
    "calibration_epoch": (
        "run_id",
        "calibration_id",
        "epoch",
        "objective",
        "data_loss",
        "l0_penalty",
        "l2_penalty",
        "nonzero_weights",
        "ess",
        "timestamp",
    ),
    "calibration_target": (
        "run_id",
        "calibration_id",
        "epoch_or_final",
        "target_name",
        "family",
        "split",
        "source",
        "geography",
        "target_value",
        "estimate",
        "relative_error",
        "weighted_term",
        "in_loss_function",
        "support_status",
    ),
    "artifact": (
        "run_id",
        "artifact_kind",
        "path_or_uri",
        "sha256",
        "size_bytes",
        "created_at",
    ),
}


class TelemetryEvent(Protocol):
    """Serializable Microplex telemetry event."""

    event_type: ClassVar[str]
    run_id: str

    def to_record(self) -> JsonRecord:
        """Return a JSON-safe telemetry record."""


@dataclass(frozen=True)
class RunEvent:
    """Run lifecycle metadata."""

    run_id: str
    status: str
    build_id: str | None = None
    engine: str | None = None
    period: int | str | None = None
    created_at: str | None = None
    code_ref: str | None = None
    config_hash: str | None = None
    incognito: bool = False
    emitted_at: str | None = None

    event_type: ClassVar[str] = "run"

    def to_record(self) -> JsonRecord:
        return _event_record(self)


@dataclass(frozen=True)
class StageEvent:
    """Build-stage lifecycle metadata."""

    run_id: str
    stage: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float | None = None
    rss_mb: float | None = None
    notes: str | None = None
    emitted_at: str | None = None

    event_type: ClassVar[str] = "stage"

    def to_record(self) -> JsonRecord:
        return _event_record(self)


@dataclass(frozen=True)
class CalibrationEpochEvent:
    """Aggregate metrics from one calibration epoch or iteration."""

    run_id: str
    calibration_id: str
    epoch: int
    objective: float | None = None
    data_loss: float | None = None
    l0_penalty: float | None = None
    l2_penalty: float | None = None
    nonzero_weights: int | None = None
    ess: float | None = None
    timestamp: str | None = None
    emitted_at: str | None = None

    event_type: ClassVar[str] = "calibration_epoch"

    def to_record(self) -> JsonRecord:
        return _event_record(self)


@dataclass(frozen=True)
class CalibrationTargetEvent:
    """Per-target final or epoch-level calibration diagnostic."""

    run_id: str
    calibration_id: str
    epoch_or_final: int | str
    target_name: str
    target_value: float
    estimate: float
    relative_error: float
    family: str | None = None
    split: str | None = None
    source: str | None = None
    geography: str | None = None
    weighted_term: float | None = None
    in_loss_function: bool = True
    support_status: str | None = None
    emitted_at: str | None = None

    event_type: ClassVar[str] = "calibration_target"

    def to_record(self) -> JsonRecord:
        return _event_record(self)


@dataclass(frozen=True)
class ArtifactEvent:
    """Artifact reference emitted by a Microplex run."""

    run_id: str
    artifact_kind: str
    path_or_uri: str
    sha256: str | None = None
    size_bytes: int | None = None
    created_at: str | None = None
    emitted_at: str | None = None

    event_type: ClassVar[str] = "artifact"

    def to_record(self) -> JsonRecord:
        return _event_record(self)


class TelemetryWriter(Protocol):
    """Telemetry writer protocol shared by local and remote sinks."""

    def emit(self, event: TelemetryEvent | Mapping[str, Any]) -> None:
        """Write one telemetry event."""

    def emit_many(self, events: Iterable[TelemetryEvent | Mapping[str, Any]]) -> None:
        """Write multiple telemetry events."""


class NullTelemetryWriter:
    """Telemetry writer that intentionally drops all events."""

    def emit(self, event: TelemetryEvent | Mapping[str, Any]) -> None:
        return None

    def emit_many(self, events: Iterable[TelemetryEvent | Mapping[str, Any]]) -> None:
        for event in events:
            self.emit(event)


class LocalTelemetryWriter:
    """Append-only JSONL telemetry writer for local runs and CI artifacts."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        incognito: bool = False,
        remote_upload_enabled: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.incognito = bool(incognito)
        self.remote_upload_enabled = bool(remote_upload_enabled and not incognito)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def emit(self, event: TelemetryEvent | Mapping[str, Any]) -> None:
        record = normalize_telemetry_event(event)
        _append_jsonl(self.output_dir / "events.jsonl", record)
        typed_path = self.output_dir / _event_file_name(record["event_type"])
        _append_jsonl(typed_path, record)

    def emit_many(self, events: Iterable[TelemetryEvent | Mapping[str, Any]]) -> None:
        for event in events:
            self.emit(event)

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "created_at": utc_now(),
            "incognito": self.incognito,
            "remote_upload_enabled": self.remote_upload_enabled,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


class SupabaseTelemetryWriter:
    """Supabase REST writer for append-only Microplex telemetry events.

    By default events are written to typed tables named by their event shape
    (`runs`, `run_stages`, `calibration_epochs`, `calibration_targets`, and
    `artifacts`). Passing `table=` switches to a single generic event table with
    `event_type`, `run_id`, `emitted_at`, and `payload` columns.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        *,
        table: str | None = None,
        table_prefix: str = "",
        table_map: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not supabase_url:
            raise ValueError("supabase_url is required")
        if not supabase_key:
            raise ValueError("supabase_key is required")
        self.supabase_url = supabase_url.rstrip("/")
        self.table = table
        self.table_prefix = table_prefix
        self.table_map = dict(_SUPABASE_TABLES | dict(table_map or {}))
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    @classmethod
    def from_env(
        cls,
        *,
        table: str | None = None,
        table_prefix: str | None = None,
        client: httpx.Client | None = None,
    ) -> SupabaseTelemetryWriter:
        """Build a Supabase writer from Microplex telemetry environment vars."""
        url = os.environ.get("MICROPLEX_SUPABASE_URL")
        key = os.environ.get("MICROPLEX_SUPABASE_SERVICE_ROLE_KEY") or os.environ.get(
            "MICROPLEX_SUPABASE_ANON_KEY"
        )
        if not url or not key:
            raise ValueError(
                "MICROPLEX_SUPABASE_URL and a Microplex Supabase key are required"
            )
        return cls(
            url,
            key,
            table=table
            or os.environ.get("MICROPLEX_TELEMETRY_EVENT_TABLE")
            or os.environ.get("MICROPLEX_TELEMETRY_TABLE"),
            table_prefix=table_prefix
            if table_prefix is not None
            else os.environ.get("MICROPLEX_TELEMETRY_TABLE_PREFIX", ""),
            client=client,
        )

    def emit(self, event: TelemetryEvent | Mapping[str, Any]) -> None:
        record = normalize_telemetry_event(event)
        self._post_rows(self._table_for(record), self._row_for(record))

    def emit_many(self, events: Iterable[TelemetryEvent | Mapping[str, Any]]) -> None:
        records = [normalize_telemetry_event(event) for event in events]
        if not records:
            return
        if self.table is not None:
            self._post_rows(
                self.table,
                [
                    {
                        "event_type": record["event_type"],
                        "run_id": record.get("run_id"),
                        "emitted_at": record["emitted_at"],
                        "payload": record,
                    }
                    for record in records
                ],
            )
            return

        rows_by_table: dict[str, list[JsonRecord]] = {}
        for record in records:
            rows_by_table.setdefault(self._table_for(record), []).append(
                _typed_supabase_row(record)
            )
        for table, rows in rows_by_table.items():
            self._post_rows(table, rows)

    def _row_for(self, record: JsonRecord) -> JsonRecord:
        if self.table is None:
            return _typed_supabase_row(record)
        return {
            "event_type": record["event_type"],
            "run_id": record.get("run_id"),
            "emitted_at": record["emitted_at"],
            "payload": record,
        }

    def _table_for(self, record: JsonRecord) -> str:
        if self.table is not None:
            return self.table
        event_type = str(record["event_type"])
        table = self.table_map.get(event_type, f"{event_type}s")
        return f"{self.table_prefix}{table}"

    def _post_rows(self, table: str, rows: JsonRecord | list[JsonRecord]) -> None:
        response = self._client.post(
            f"{self.supabase_url}/rest/v1/{table}",
            headers=self._headers,
            json=rows,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Supabase telemetry write failed with HTTP {response.status_code}"
            )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class CompositeTelemetryWriter:
    """Fan-out writer for local + remote telemetry sinks."""

    def __init__(self, writers: Iterable[TelemetryWriter]) -> None:
        self.writers = tuple(writers)

    def emit(self, event: TelemetryEvent | Mapping[str, Any]) -> None:
        for writer in self.writers:
            writer.emit(event)

    def emit_many(self, events: Iterable[TelemetryEvent | Mapping[str, Any]]) -> None:
        buffered = tuple(events)
        for writer in self.writers:
            writer.emit_many(buffered)


def build_telemetry_writer(
    output_dir: str | Path | None = None,
    *,
    upload: bool = False,
    incognito: bool = False,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
    table: str | None = None,
    table_prefix: str | None = None,
) -> TelemetryWriter:
    """Create a local, remote, composite, or null telemetry writer.

    `incognito=True` is a hard remote-write off switch. Local artifacts still
    mark the run as incognito so dashboards can distinguish private runs.
    """
    writers: list[TelemetryWriter] = []
    if output_dir is not None:
        writers.append(
            LocalTelemetryWriter(
                output_dir,
                incognito=incognito,
                remote_upload_enabled=upload,
            )
        )

    if upload and not incognito:
        resolved_url = supabase_url or os.environ.get("MICROPLEX_SUPABASE_URL")
        resolved_key = (
            supabase_key
            or os.environ.get("MICROPLEX_SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("MICROPLEX_SUPABASE_ANON_KEY")
        )
        if not resolved_url or not resolved_key:
            raise ValueError(
                "Remote telemetry upload requested without Supabase credentials"
            )
        writers.append(
            SupabaseTelemetryWriter(
                resolved_url,
                resolved_key,
                table=table
                or os.environ.get("MICROPLEX_TELEMETRY_EVENT_TABLE")
                or os.environ.get("MICROPLEX_TELEMETRY_TABLE"),
                table_prefix=table_prefix
                if table_prefix is not None
                else os.environ.get("MICROPLEX_TELEMETRY_TABLE_PREFIX", ""),
            )
        )

    if not writers:
        return NullTelemetryWriter()
    if len(writers) == 1:
        return writers[0]
    return CompositeTelemetryWriter(writers)


def normalize_telemetry_event(event: TelemetryEvent | Mapping[str, Any]) -> JsonRecord:
    """Normalize an event object or mapping into a JSON-safe record."""
    if isinstance(event, Mapping):
        record = dict(event)
        record.setdefault("event_type", "event")
        record.setdefault("emitted_at", utc_now())
        return _json_safe_record(record)
    return _json_safe_record(event.to_record())


def effective_sample_size(weights: np.ndarray | pd.Series | list[float]) -> float:
    """Kish effective sample size for a vector of non-negative weights."""
    values = np.asarray(weights, dtype=float)
    denominator = float(np.sum(values**2))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(values)) ** 2
    return numerator / denominator


def utc_now() -> str:
    """Current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _event_record(event: Any) -> JsonRecord:
    payload = asdict(event)
    payload["event_type"] = event.event_type
    payload.setdefault("emitted_at", None)
    if payload["emitted_at"] is None:
        payload["emitted_at"] = utc_now()
    if "timestamp" in payload and payload["timestamp"] is None:
        payload["timestamp"] = payload["emitted_at"]
    if isinstance(event, RunEvent) and payload.get("created_at") is None:
        payload["created_at"] = payload["emitted_at"]
    return _json_safe_record(payload)


def _json_safe_record(record: Mapping[str, Any]) -> JsonRecord:
    return {key: _json_safe_value(value, key) for key, value in record.items()}


def _json_safe_value(value: Any, path: str) -> Any:
    if isinstance(value, pd.DataFrame | pd.Series | pd.Index):
        raise TypeError(f"Telemetry payload {path!r} contains row-level pandas data")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe_value(value.item(), path)
        raise TypeError(f"Telemetry payload {path!r} contains row-level array data")
    if isinstance(value, np.generic):
        return _json_safe_value(value.item(), path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe_value(asdict(value), path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_value(nested_value, f"{path}.{key}")
            for key, nested_value in value.items()
        }
    if isinstance(value, list | tuple):
        return [
            _json_safe_value(nested_value, f"{path}[{index}]")
            for index, nested_value in enumerate(value)
        ]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str | int | bool) or value is None:
        return value
    raise TypeError(
        f"Telemetry payload {path!r} has unsupported value type {type(value).__name__}"
    )


def _append_jsonl(path: Path, record: JsonRecord) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def _event_file_name(event_type: str) -> str:
    return {
        "run": "runs.jsonl",
        "stage": "run_stages.jsonl",
        "calibration_epoch": "calibration_epochs.jsonl",
        "calibration_target": "calibration_targets.jsonl",
        "artifact": "artifacts.jsonl",
    }.get(event_type, f"{event_type}s.jsonl")


def _typed_supabase_row(record: JsonRecord) -> JsonRecord:
    columns = _SUPABASE_TABLE_COLUMNS.get(str(record["event_type"]))
    if columns is None:
        return {
            "event_type": record["event_type"],
            "run_id": record.get("run_id"),
            "emitted_at": record["emitted_at"],
            "payload": record,
        }
    return {column: record[column] for column in columns if column in record}


__all__ = [
    "ArtifactEvent",
    "CalibrationEpochEvent",
    "CalibrationTargetEvent",
    "CompositeTelemetryWriter",
    "LocalTelemetryWriter",
    "NullTelemetryWriter",
    "RunEvent",
    "StageEvent",
    "SupabaseTelemetryWriter",
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryEvent",
    "TelemetryWriter",
    "build_telemetry_writer",
    "effective_sample_size",
    "normalize_telemetry_event",
    "utc_now",
]
