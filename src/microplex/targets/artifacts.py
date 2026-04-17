"""Shared manifest validation for benchmarked artifact bundles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkArtifactValidationResult:
    """Validation outcome for one benchmarked artifact manifest."""

    artifact_dir: str
    manifest_path: str | None = None
    summary_section: str | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.is_valid:
            return
        context = self.manifest_path or self.artifact_dir
        message = "; ".join(self.errors)
        raise ValueError(f"Invalid benchmark artifact manifest for '{context}': {message}")


def validate_benchmark_artifact_manifest(
    manifest: dict[str, Any],
    *,
    artifact_dir: str | Path,
    manifest_path: str | Path | None = None,
    summary_section: str | None = None,
    required_artifact_keys: Iterable[str] = (),
    required_summary_keys: Iterable[str] = (),
    required_top_level_keys: Iterable[str] = ("created_at", "config", "artifacts"),
) -> BenchmarkArtifactValidationResult:
    """Validate one benchmark-artifact manifest against the shared contract."""

    artifact_root = Path(artifact_dir)
    manifest_location = str(Path(manifest_path)) if manifest_path is not None else None
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(manifest, dict):
        return BenchmarkArtifactValidationResult(
            artifact_dir=str(artifact_root),
            manifest_path=manifest_location,
            summary_section=summary_section,
            errors=("manifest payload must be a dict",),
        )

    for key in required_top_level_keys:
        if key not in manifest:
            errors.append(f"missing top-level key '{key}'")

    config_payload = manifest.get("config")
    if config_payload is not None and not isinstance(config_payload, dict):
        errors.append("top-level key 'config' must be a dict")

    artifacts_payload = manifest.get("artifacts")
    if artifacts_payload is None:
        artifacts_payload = {}
    if not isinstance(artifacts_payload, dict):
        errors.append("top-level key 'artifacts' must be a dict")
        artifacts_payload = {}

    resolved_summary_section = summary_section
    if resolved_summary_section is None:
        available_sections = [
            key
            for key in ("benchmark", "policyengine_harness")
            if isinstance(manifest.get(key), dict)
        ]
        if len(available_sections) == 1:
            resolved_summary_section = available_sections[0]
        elif len(available_sections) > 1:
            warnings.append(
                "multiple summary sections found; pass summary_section explicitly to"
                " enforce one contract"
            )

    if resolved_summary_section is not None:
        summary_payload = manifest.get(resolved_summary_section)
        if not isinstance(summary_payload, dict):
            errors.append(f"summary section '{resolved_summary_section}' must be a dict")
        else:
            for key in required_summary_keys:
                if key not in summary_payload or summary_payload[key] is None:
                    errors.append(
                        f"summary section '{resolved_summary_section}' is missing"
                        f" required key '{key}'"
                    )

    for artifact_key in required_artifact_keys:
        value = artifacts_payload.get(artifact_key)
        if not isinstance(value, str) or not value:
            errors.append(f"artifacts['{artifact_key}'] must be a non-empty string")

    for artifact_key, value in artifacts_payload.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            errors.append(f"artifacts['{artifact_key}'] must be a non-empty string or null")
            continue
        artifact_path = Path(value)
        if not artifact_path.is_absolute():
            artifact_path = artifact_root / artifact_path
        if not artifact_path.exists():
            errors.append(
                f"artifacts['{artifact_key}'] references missing file '{artifact_path}'"
            )

    return BenchmarkArtifactValidationResult(
        artifact_dir=str(artifact_root),
        manifest_path=manifest_location,
        summary_section=resolved_summary_section,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def assert_valid_benchmark_artifact_manifest(
    manifest: dict[str, Any],
    *,
    artifact_dir: str | Path,
    manifest_path: str | Path | None = None,
    summary_section: str | None = None,
    required_artifact_keys: Iterable[str] = (),
    required_summary_keys: Iterable[str] = (),
    required_top_level_keys: Iterable[str] = ("created_at", "config", "artifacts"),
) -> BenchmarkArtifactValidationResult:
    """Validate one manifest and raise on contract violations."""

    result = validate_benchmark_artifact_manifest(
        manifest,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        summary_section=summary_section,
        required_artifact_keys=required_artifact_keys,
        required_summary_keys=required_summary_keys,
        required_top_level_keys=required_top_level_keys,
    )
    result.raise_for_errors()
    return result
