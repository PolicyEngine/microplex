"""Tests for shared benchmark artifact validation."""

from __future__ import annotations

from microplex.targets import (
    assert_valid_benchmark_artifact_manifest,
    validate_benchmark_artifact_manifest,
)


def test_validate_benchmark_artifact_manifest_accepts_valid_manifest(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "candidate.h5").write_text("candidate")
    (artifact_dir / "comparison.json").write_text("{}")

    manifest = {
        "created_at": "2026-03-29T00:00:00+00:00",
        "config": {"benchmark_mode": "direct"},
        "artifacts": {
            "candidate_dataset": "candidate.h5",
            "comparison": "comparison.json",
        },
        "benchmark": {
            "candidate_mean_abs_relative_error": 0.5,
            "baseline_mean_abs_relative_error": 0.6,
            "mean_abs_relative_error_delta": -0.1,
        },
    }

    result = validate_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_dir,
        summary_section="benchmark",
        required_artifact_keys=("candidate_dataset", "comparison"),
        required_summary_keys=(
            "candidate_mean_abs_relative_error",
            "baseline_mean_abs_relative_error",
            "mean_abs_relative_error_delta",
        ),
    )

    assert result.is_valid
    assert result.errors == ()
    assert_valid_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_dir,
        summary_section="benchmark",
        required_artifact_keys=("candidate_dataset", "comparison"),
        required_summary_keys=(
            "candidate_mean_abs_relative_error",
            "baseline_mean_abs_relative_error",
            "mean_abs_relative_error_delta",
        ),
    )


def test_validate_benchmark_artifact_manifest_rejects_missing_files(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    manifest = {
        "created_at": "2026-03-29T00:00:00+00:00",
        "config": {},
        "artifacts": {
            "candidate_dataset": "candidate.h5",
            "comparison": "comparison.json",
        },
        "benchmark": {
            "candidate_mean_abs_relative_error": 0.5,
            "baseline_mean_abs_relative_error": 0.6,
            "mean_abs_relative_error_delta": -0.1,
        },
    }

    result = validate_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_dir,
        summary_section="benchmark",
        required_artifact_keys=("candidate_dataset", "comparison"),
        required_summary_keys=("candidate_mean_abs_relative_error",),
    )

    assert not result.is_valid
    assert any("missing file" in error for error in result.errors)


def test_validate_benchmark_artifact_manifest_rejects_missing_summary_keys(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "candidate.h5").write_text("candidate")
    (artifact_dir / "comparison.json").write_text("{}")

    manifest = {
        "created_at": "2026-03-29T00:00:00+00:00",
        "config": {},
        "artifacts": {
            "candidate_dataset": "candidate.h5",
            "comparison": "comparison.json",
        },
        "benchmark": {},
    }

    result = validate_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_dir,
        summary_section="benchmark",
        required_artifact_keys=("candidate_dataset", "comparison"),
        required_summary_keys=("candidate_mean_abs_relative_error",),
    )

    assert not result.is_valid
    assert any("candidate_mean_abs_relative_error" in error for error in result.errors)
