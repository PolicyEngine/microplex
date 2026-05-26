# AGENTS.md

This repo is the shared `microplex` core. Treat it as the canonical home for abstractions that must work for both `microplex-us` and `microplex-uk`.

## Default posture

- Prefer moving reusable behavior into core over duplicating it in country packs.
- Keep country assumptions out of core package surfaces unless they are explicitly legacy-compat shims.
- When a seam is country-specific, make that explicit and leave it in the country repo instead of encoding it implicitly in shared types.
- Findings-first review standard: prioritize correctness, benchmark validity, abstraction drift, and missing tests.

## Current architectural intent

- Core owns shared targets abstractions:
  - target specs and queries
  - target providers/protocols
  - reweighting constraints and bundle helpers
  - benchmark metric math
  - benchmark result/comparison/suite envelopes
  - result-oriented slice and suite builders
- Country packs should be thin adapters around:
  - raw source parsing
  - PolicyEngine execution/materialization
  - geography assets and country-specific target providers

## Review checklist

When reviewing recent changes here, check:

1. Did this change actually remove duplication from both country packs, or only move code around?
2. Is the new abstraction country-agnostic, or is it secretly US-shaped or UK-shaped?
3. Can the same API support both PE-US multi-slice union evaluation and PE-UK per-dataset evaluation?
4. Are benchmark numbers still apples-to-apples after the change?
5. Are mutable shared objects or compatibility shims creating hidden footguns?

## Safe changes

- Adding shared types/protocols under `src/microplex/targets/`
- Tightening package-surface exports
- Adding focused core regressions under `tests/targets/` and `tests/test_package_surface.py`

## Be careful around

- `src/microplex/targets/database.py`
  - Legacy compatibility shim territory; do not expand it further.
- `src/microplex/geography.py`
  - Public assigner APIs are now used by country packs. Avoid reintroducing private-API dependencies.
- Any metric change in `src/microplex/targets/benchmarking.py`
  - This will affect both UK and US benchmark claims.

## Standard commands

- Ruff: `uv run --with duckdb ruff check src tests`
- Core target tests: `uv run --with duckdb pytest -q tests/targets/test_benchmarking.py tests/targets/test_reweighting.py`
- Package surface tests: `uv run --with duckdb pytest -q tests/test_package_surface.py`

## Claude/Codex review shortcut

For a quick review, read:

1. [`/Users/maxghenis/PolicyEngine/microplex/AGENTS.md`](/Users/maxghenis/PolicyEngine/microplex/AGENTS.md)
2. [`/Users/maxghenis/PolicyEngine/microplex/_WORKSPACE.md`](/Users/maxghenis/PolicyEngine/microplex/_WORKSPACE.md)
3. [`/Users/maxghenis/PolicyEngine/microplex/_BUILD_LOG.md`](/Users/maxghenis/PolicyEngine/microplex/_BUILD_LOG.md)

Then inspect changed files and return findings first.
