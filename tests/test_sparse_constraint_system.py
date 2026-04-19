"""Sparse build path for the linear constraint system.

`_build_linear_constraint_system` currently materializes a dense numpy
`A` of shape `(n_targets, n_records)` via `np.vstack(rows)`. At
microplex-us v7 scale (1.5M records x ~4k+ constraints, mostly
marginal indicators that are 95 %+ zero) this dense materialization
eats ~24 GB and pushes the process past jetsam. Downstream L0
calibrators then immediately convert to CSR via `sp.csr_matrix(A)`,
which already produces the correct sparse result — we just got there
through a dense intermediate that wastes the memory.

These tests pin a sparse-native builder (`_build_sparse_constraint_system`)
that returns `(X_csr, b, names, n_categorical)` without ever
materializing the dense matrix. Semantics must agree with the dense
version up to float32 rounding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from scipy import sparse

from microplex.calibration import (
    LinearConstraint,
    _build_linear_constraint_system,
)


def _toy_frame(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "state": rng.choice(["CA", "NY", "TX", "FL"], size=n),
            "age_group": rng.choice(["0-17", "18-64", "65+"], size=n),
            "income": rng.lognormal(10, 1, size=n),
        }
    )


class TestSparseBuilderExists:
    """The sparse builder must be importable from microplex.calibration."""

    def test_sparse_builder_is_importable(self) -> None:
        from microplex.calibration import _build_sparse_constraint_system

        assert callable(_build_sparse_constraint_system)


class TestSparseBuilderMatchesDense:
    """Sparse builder output must equal dense builder output numerically."""

    def _build_both(self, data, marginal, continuous, linear):
        from microplex.calibration import _build_sparse_constraint_system

        A_dense, b_dense, names_dense, n_cat_dense = (
            _build_linear_constraint_system(
                data, marginal, continuous, linear
            )
        )
        X_sparse, b_sparse, names_sparse, n_cat_sparse = (
            _build_sparse_constraint_system(
                data, marginal, continuous, linear
            )
        )
        return (
            A_dense,
            b_dense,
            names_dense,
            n_cat_dense,
            X_sparse,
            b_sparse,
            names_sparse,
            n_cat_sparse,
        )

    def test_marginal_constraints_only(self) -> None:
        data = _toy_frame()
        marginal = {
            "state": {
                "CA": 280,
                "NY": 230,
                "TX": 260,
                "FL": 230,
            },
            "age_group": {"0-17": 230, "18-64": 570, "65+": 200},
        }
        (
            A_dense,
            b_dense,
            names_dense,
            n_cat_dense,
            X_sparse,
            b_sparse,
            names_sparse,
            n_cat_sparse,
        ) = self._build_both(data, marginal, None, ())

        assert sparse.issparse(X_sparse), type(X_sparse)
        assert X_sparse.shape == A_dense.shape
        np.testing.assert_allclose(
            X_sparse.toarray(), A_dense, rtol=0, atol=0
        )
        np.testing.assert_array_equal(b_sparse, b_dense)
        assert names_sparse == names_dense
        assert n_cat_sparse == n_cat_dense

    def test_continuous_and_linear_constraints(self) -> None:
        data = _toy_frame()
        marginal = {"state": {"CA": 250, "NY": 250, "TX": 250, "FL": 250}}
        continuous = {"income": 20_000_000.0}
        linear = (
            LinearConstraint(
                name="old_and_rich",
                coefficients=(
                    (data["age_group"] == "65+").astype(float)
                    * data["income"]
                ).to_numpy(),
                target=500_000.0,
            ),
        )
        (
            A_dense,
            b_dense,
            names_dense,
            n_cat_dense,
            X_sparse,
            b_sparse,
            names_sparse,
            n_cat_sparse,
        ) = self._build_both(data, marginal, continuous, linear)

        assert sparse.issparse(X_sparse)
        np.testing.assert_allclose(
            X_sparse.toarray(), A_dense, rtol=1e-12, atol=0
        )
        np.testing.assert_array_equal(b_sparse, b_dense)
        assert names_sparse == names_dense
        assert n_cat_sparse == n_cat_dense


class TestSparseBuilderActuallySparsifies:
    """Marginal indicators must actually be stored sparsely."""

    def test_marginal_density_is_low(self) -> None:
        """With 4 state × 3 age-group categories, density should be ~1/4 per target."""
        from microplex.calibration import _build_sparse_constraint_system

        data = _toy_frame(n=10_000, seed=1)
        marginal = {
            "state": {"CA": 2_500, "NY": 2_500, "TX": 2_500, "FL": 2_500},
            "age_group": {
                "0-17": 2_500,
                "18-64": 5_500,
                "65+": 2_000,
            },
        }
        X_sparse, _b, _names, _n = _build_sparse_constraint_system(
            data, marginal, None, ()
        )
        density = X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1])
        # Upper bound: with 4-state + 3-age indicators the weighted
        # density is <= max(1/4, 1/3) ≈ 0.33. Assert meaningfully less
        # than dense so we know CSR is doing its job.
        assert density < 0.45, density
