# _WORKSPACE.md

This file is the durable local context for `microplex` core.

## Repo role

`microplex` is the shared core for country packs. It should accumulate reusable abstractions, not country-specific policy logic or raw source parsing.

Sibling repos:

- [`/Users/maxghenis/PolicyEngine/microplex-us`](/Users/maxghenis/PolicyEngine/microplex-us)
- [`/Users/maxghenis/PolicyEngine/microplex-uk`](/Users/maxghenis/PolicyEngine/microplex-uk)

## Current shared seams

### Targets

- `src/microplex/targets/spec.py`
- `src/microplex/targets/provider.py`
- `src/microplex/targets/bundles.py`
- `src/microplex/targets/reweighting.py`
- `src/microplex/targets/benchmarking.py`

These now cover:

- target specs and queries
- target providers
- household-linked entity bundles for reweighting
- normalized metric construction
- benchmark result/comparison/suite serialization
- payload-based and result-based slice/suite builders

### Geography

- `src/microplex/geography.py`

Public partition helpers on `ProbabilisticAtomicGeographyAssigner` now exist specifically so country packs do not reach into private assigner internals.

## Current architectural boundary

Core should own:

- benchmark math and data structures
- reweighting math and bundle interfaces
- common query/filter semantics
- generic suite/result orchestration

Country packs should own:

- PolicyEngine adapters
- raw source manifests/parsers
- geography assets/crosswalks
- target registries that are country-specific

## Known open questions

1. The next likely extraction is a fuller PE adapter contract:
   - materialize required features
   - evaluate target set
   - return `BenchmarkResult`
2. Core should avoid hard-baking US tax-unit assumptions. US tax filing units may eventually need to be policy-endogenous.
3. UK and US still differ in execution architecture:
   - US is mostly in-process with richer caching
   - UK still uses subprocess-isolated PE execution

## High-signal tests

- `tests/targets/test_benchmarking.py`
- `tests/targets/test_reweighting.py`
- `tests/test_package_surface.py`
- `tests/test_geography.py`

## Working rule

If a change helps both UK and US, prefer landing it here first and making country packs adapt downward.
