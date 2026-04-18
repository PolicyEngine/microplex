"""Upstream-home tests for the country-agnostic microcalibrate adapter.

The adapter wraps `microcalibrate.Calibration` with the legacy
`Calibrator.fit_transform` interface so country packages (microplex-us,
microplex-uk, etc.) share one identity-preserving calibrator instead of
duplicating the glue code.

microcalibrate is an optional extra; these tests import it via the
`microplex[calibrate]` route and skip gracefully when unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microplex.calibration import LinearConstraint

microcalibrate = pytest.importorskip("microcalibrate")


def _toy_data(n_records: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n_records),
            "weight": np.ones(n_records),
        }
    )


def _age_band(
    data: pd.DataFrame, name: str, low: int, high: int, target: float
) -> LinearConstraint:
    mask = (data["age"] >= low) & (data["age"] < high)
    return LinearConstraint(
        name=name,
        coefficients=mask.astype(float).to_numpy(),
        target=target,
    )


class TestImportSurface:
    """The adapter must be importable from the canonical upstream location."""

    def test_importable_from_microplex_calibration(self) -> None:
        from microplex.calibration import (
            MicrocalibrateAdapter,
            MicrocalibrateAdapterConfig,
        )

        assert MicrocalibrateAdapter is not None
        assert MicrocalibrateAdapterConfig is not None

    def test_default_config_sets_batch_size(self) -> None:
        """Default batch_size is set so v7-scale fits stay under RSS budget."""
        from microplex.calibration import MicrocalibrateAdapterConfig

        config = MicrocalibrateAdapterConfig()
        assert config.batch_size is not None
        assert config.batch_size > 0


class TestConvergesOnSmallProblem:
    """End-to-end check: fit_transform moves weights toward targets."""

    def test_three_age_bands_converge(self) -> None:
        from microplex.calibration import (
            MicrocalibrateAdapter,
            MicrocalibrateAdapterConfig,
        )

        data = _toy_data(n_records=300)
        constraints = (
            _age_band(data, "age_18_30", 18, 30, 60.0),
            _age_band(data, "age_30_45", 30, 45, 90.0),
            _age_band(data, "age_45_70", 45, 70, 150.0),
        )
        adapter = MicrocalibrateAdapter(
            MicrocalibrateAdapterConfig(
                epochs=400, learning_rate=0.05, noise_level=0.0
            )
        )
        result = adapter.fit_transform(data, linear_constraints=constraints)
        validation = adapter.validate(result)
        assert validation["max_error"] < 0.1, validation
