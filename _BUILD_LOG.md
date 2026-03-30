# _BUILD_LOG.md

Append-only notes for agents working in `microplex` core.

## 2026-03-28

- Generalization push moved benchmark logic deeper into core.
- Added and adopted shared target-layer modules:
  - `src/microplex/targets/bundles.py`
  - `src/microplex/targets/reweighting.py`
  - `src/microplex/targets/benchmarking.py`
  - `src/microplex/targets/provider.py`
  - `src/microplex/targets/spec.py`
- Core now owns:
  - shared benchmark metric math
  - comparison math on common target intersections
  - benchmark result/comparison/suite serialization
  - payload-based slice evaluation helpers
  - result-oriented slice evaluation and suite builders
- Fixed target-record aggregation so duplicate target names across slices no longer silently last-writer-win.
- Unified zero-target relative-error handling by routing reweighting diagnostics through shared `relative_error_ratio(...)`.
- Removed core-owned grouped-summary metadata defaults. Country packs must now pass `group_fields` explicitly.
- Public assigner partition APIs were added in `src/microplex/geography.py` so UK no longer reaches into private assigner internals.
- Legacy US-specific targets DB implementation was moved out of core into `microplex-us`; core `targets/database.py` is now compatibility territory and should not expand.

## Current review bar

- Do not accept changes that reintroduce country-specific assumptions into core.
- Benchmark math changes require explicit test coverage.
- Prefer adding focused regressions over broad integration-only confidence.

## Known remaining risks

- UK and US still diverge at the PolicyEngine execution layer.
- Suite-level metrics and composite-loss semantics are not identical across countries; do not present them as interchangeable without context.
- Any future change to benchmark/result normalization should be checked in both country packs immediately.

## 2026-03-29

- Added a shared benchmark-artifact manifest contract in:
  - `src/microplex/targets/artifacts.py`
- Core now exposes:
  - `validate_benchmark_artifact_manifest(...)`
  - `assert_valid_benchmark_artifact_manifest(...)`
  - `BenchmarkArtifactValidationResult`
- The contract is intentionally small:
  - required top-level manifest structure
  - explicit summary section (`benchmark` or `policyengine_harness`)
  - required summary keys for benchmarked bundles
  - existence checks for referenced artifact files
- This is meant to centralize enforcement, not force US and UK into one identical payload schema.
