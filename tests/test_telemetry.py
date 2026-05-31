from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from microplex.telemetry import (
    CalibrationEpochEvent,
    LocalTelemetryWriter,
    RunEvent,
    StageEvent,
    SupabaseTelemetryWriter,
    build_telemetry_writer,
    normalize_telemetry_event,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_local_telemetry_writer_appends_events_and_manifest(tmp_path):
    writer = LocalTelemetryWriter(tmp_path / "telemetry", incognito=True)

    writer.emit(RunEvent(run_id="run-1", status="started", engine="microplex"))
    writer.emit(
        StageEvent(
            run_id="run-1",
            stage="calibration",
            status="completed",
            elapsed_seconds=1.5,
            rss_mb=128.0,
        )
    )

    manifest = json.loads((tmp_path / "telemetry" / "manifest.json").read_text())
    events = _read_jsonl(tmp_path / "telemetry" / "events.jsonl")
    run_events = _read_jsonl(tmp_path / "telemetry" / "runs.jsonl")
    stage_events = _read_jsonl(tmp_path / "telemetry" / "run_stages.jsonl")

    assert manifest["incognito"] is True
    assert manifest["remote_upload_enabled"] is False
    assert [event["event_type"] for event in events] == ["run", "stage"]
    assert run_events[0]["run_id"] == "run-1"
    assert stage_events[0]["stage"] == "calibration"


def test_build_telemetry_writer_incognito_disables_remote_upload(tmp_path):
    writer = build_telemetry_writer(
        tmp_path / "telemetry",
        upload=True,
        incognito=True,
    )

    writer.emit(RunEvent(run_id="run-1", status="started", incognito=True))

    manifest = json.loads((tmp_path / "telemetry" / "manifest.json").read_text())
    assert manifest["incognito"] is True
    assert manifest["remote_upload_enabled"] is False
    assert (tmp_path / "telemetry" / "events.jsonl").exists()


def test_supabase_telemetry_writer_posts_append_only_event():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    writer = SupabaseTelemetryWriter(
        "https://example.supabase.co",
        "secret-key",
        table="telemetry_events",
        client=client,
    )

    writer.emit(
        CalibrationEpochEvent(
            run_id="run-1",
            calibration_id="cal-1",
            epoch=7,
            objective=0.12,
            data_loss=0.12,
            nonzero_weights=42,
            ess=35.5,
        )
    )

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://example.supabase.co/rest/v1/telemetry_events"
    assert request.headers["apikey"] == "secret-key"
    body = json.loads(request.content)
    assert body["event_type"] == "calibration_epoch"
    assert body["run_id"] == "run-1"
    assert body["payload"]["epoch"] == 7


def test_supabase_telemetry_writer_posts_typed_table_by_default():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(201)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    writer = SupabaseTelemetryWriter(
        "https://example.supabase.co",
        "secret-key",
        client=client,
    )

    writer.emit(
        CalibrationEpochEvent(
            run_id="run-1",
            calibration_id="cal-1",
            epoch=8,
            objective=0.08,
        )
    )

    assert len(requests) == 1
    assert str(requests[0].url) == (
        "https://example.supabase.co/rest/v1/calibration_epochs"
    )
    body = json.loads(requests[0].content)
    assert body["event_type"] == "calibration_epoch"
    assert body["epoch"] == 8
    assert "payload" not in body


def test_telemetry_rejects_row_level_payloads():
    with pytest.raises(TypeError, match="row-level pandas data"):
        normalize_telemetry_event(
            {
                "event_type": "bad",
                "run_id": "run-1",
                "rows": pd.DataFrame({"person_id": [1, 2]}),
            }
        )
