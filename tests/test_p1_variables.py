"""Tests for P1 variable extraction in enhanced CPS data.

These tests verify that the enhanced CPS person-level data includes
the new variables needed for PolicyEngine scope. The 11 new P1 columns
are:

Person-level (direct from PE-US):
  - is_separated (bool)
  - is_surviving_spouse (bool)
  - is_tax_unit_head (bool)
  - is_tax_unit_spouse (bool)
  - is_tax_unit_dependent (bool)
  - weekly_hours_worked (float32)
  - is_blind (bool)

Tax-unit level (mapped to persons via tax_unit_id):
  - filing_status (string/categorical)
  - tax_unit_dependents (int)
  - tax_unit_size (int)
  - tax_unit_is_joint (bool)

TDD: These tests are written BEFORE the columns exist in the data,
so they will fail until build_enhanced_cps.py is updated.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "cps_enhanced_persons.parquet"

# --- P1 column definitions ---

P1_BOOL_COLUMNS = [
    "is_separated",
    "is_surviving_spouse",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
    "is_blind",
    "tax_unit_is_joint",
]

P1_NUMERIC_COLUMNS = [
    "weekly_hours_worked",
    "tax_unit_dependents",
    "tax_unit_size",
]

P1_STRING_COLUMNS = [
    "filing_status",
]

ALL_P1_COLUMNS = P1_BOOL_COLUMNS + P1_NUMERIC_COLUMNS + P1_STRING_COLUMNS


# --- Fixtures ---


@pytest.fixture(scope="module")
def enhanced_persons() -> pd.DataFrame:
    """Load the enhanced CPS persons parquet file.

    This fixture fails immediately if the data file does not exist,
    since all P1 tests depend on the data being present.
    """
    if not DATA_PATH.exists():
        pytest.fail(
            f"Enhanced CPS persons data not found at {DATA_PATH}. "
            "Run scripts/build_enhanced_cps.py first."
        )
    return pd.read_parquet(DATA_PATH)


# =============================================================================
# 1. Column existence tests
# =============================================================================


class TestP1ColumnExistence:
    """All 11 P1 columns must exist in the enhanced data."""

    @pytest.mark.parametrize("column", ALL_P1_COLUMNS)
    def test_column_exists(self, enhanced_persons: pd.DataFrame, column: str):
        """P1 column '{column}' must be present in the data."""
        assert column in enhanced_persons.columns, (
            f"Column '{column}' is missing from enhanced CPS data. "
            f"Current columns: {sorted(enhanced_persons.columns.tolist())}"
        )

    def test_all_p1_columns_present(self, enhanced_persons: pd.DataFrame):
        """All 11 P1 columns must be present at once."""
        missing = [c for c in ALL_P1_COLUMNS if c not in enhanced_persons.columns]
        assert not missing, (
            f"Missing {len(missing)} P1 columns: {missing}"
        )


# =============================================================================
# 2. Data type tests
# =============================================================================


class TestP1DataTypes:
    """P1 columns must have the correct data types."""

    @pytest.mark.parametrize("column", P1_BOOL_COLUMNS)
    def test_bool_column_dtype(self, enhanced_persons: pd.DataFrame, column: str):
        """Boolean P1 column '{column}' must have bool dtype."""
        if column not in enhanced_persons.columns:
            pytest.skip(f"Column '{column}' not yet present")
        assert enhanced_persons[column].dtype == bool, (
            f"Column '{column}' has dtype {enhanced_persons[column].dtype}, expected bool"
        )

    def test_weekly_hours_worked_is_numeric(self, enhanced_persons: pd.DataFrame):
        """weekly_hours_worked must be a numeric (float) type."""
        col = "weekly_hours_worked"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        assert pd.api.types.is_float_dtype(enhanced_persons[col]), (
            f"Column '{col}' has dtype {enhanced_persons[col].dtype}, expected float"
        )

    def test_tax_unit_dependents_is_numeric(self, enhanced_persons: pd.DataFrame):
        """tax_unit_dependents must be numeric (int or float)."""
        col = "tax_unit_dependents"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        assert pd.api.types.is_numeric_dtype(enhanced_persons[col]), (
            f"Column '{col}' has dtype {enhanced_persons[col].dtype}, expected numeric"
        )

    def test_tax_unit_size_is_numeric(self, enhanced_persons: pd.DataFrame):
        """tax_unit_size must be numeric (int or float)."""
        col = "tax_unit_size"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        assert pd.api.types.is_numeric_dtype(enhanced_persons[col]), (
            f"Column '{col}' has dtype {enhanced_persons[col].dtype}, expected numeric"
        )

    def test_filing_status_is_string(self, enhanced_persons: pd.DataFrame):
        """filing_status must be string/object or categorical type."""
        col = "filing_status"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        dtype = enhanced_persons[col].dtype
        is_valid = (
            pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
        )
        assert is_valid, (
            f"Column '{col}' has dtype {dtype}, expected string/object/categorical"
        )


# =============================================================================
# 3. Value range tests
# =============================================================================


class TestP1ValueRanges:
    """P1 columns must have values within expected ranges."""

    def test_weekly_hours_worked_non_negative(self, enhanced_persons: pd.DataFrame):
        """weekly_hours_worked must be >= 0."""
        col = "weekly_hours_worked"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        values = enhanced_persons[col].dropna()
        assert (values >= 0).all(), (
            f"Found {(values < 0).sum()} negative values in '{col}'. "
            f"Min value: {values.min()}"
        )

    def test_weekly_hours_worked_reasonable_max(self, enhanced_persons: pd.DataFrame):
        """weekly_hours_worked should not exceed 168 (hours in a week)."""
        col = "weekly_hours_worked"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        values = enhanced_persons[col].dropna()
        assert (values <= 168).all(), (
            f"Found values exceeding 168 hours/week in '{col}'. "
            f"Max value: {values.max()}"
        )

    def test_tax_unit_dependents_non_negative(self, enhanced_persons: pd.DataFrame):
        """tax_unit_dependents must be >= 0."""
        col = "tax_unit_dependents"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        values = enhanced_persons[col].dropna()
        assert (values >= 0).all(), (
            f"Found {(values < 0).sum()} negative values in '{col}'. "
            f"Min value: {values.min()}"
        )

    def test_tax_unit_size_at_least_one(self, enhanced_persons: pd.DataFrame):
        """tax_unit_size must be >= 1 (every tax unit has at least one person)."""
        col = "tax_unit_size"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        values = enhanced_persons[col].dropna()
        assert (values >= 1).all(), (
            f"Found {(values < 1).sum()} tax units with size < 1 in '{col}'. "
            f"Min value: {values.min()}"
        )

    def test_filing_status_valid_values(self, enhanced_persons: pd.DataFrame):
        """filing_status values must be from the expected set."""
        col = "filing_status"
        if col not in enhanced_persons.columns:
            pytest.skip(f"Column '{col}' not yet present")
        valid_statuses = {
            "SINGLE",
            "JOINT",
            "HEAD_OF_HOUSEHOLD",
            "SEPARATE",
            "SURVIVING_SPOUSE",
        }
        actual_values = set(enhanced_persons[col].dropna().unique())
        unexpected = actual_values - valid_statuses
        assert not unexpected, (
            f"Unexpected filing_status values: {unexpected}. "
            f"Expected subset of: {valid_statuses}"
        )


# =============================================================================
# 4. Consistency tests
# =============================================================================


class TestP1Consistency:
    """Cross-column consistency checks for P1 variables."""

    def _has_all_columns(self, df: pd.DataFrame, columns: list[str]) -> bool:
        """Check if all columns are present; skip test if not."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            pytest.skip(f"Columns not yet present: {missing}")
        return True

    # -- Tax-unit level variables are consistent within tax units --

    def test_filing_status_consistent_within_tax_unit(
        self, enhanced_persons: pd.DataFrame
    ):
        """All persons in the same tax unit must have the same filing_status."""
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "filing_status"])
        nunique = enhanced_persons.groupby("tax_unit_id")["filing_status"].nunique()
        inconsistent = nunique[nunique > 1]
        assert inconsistent.empty, (
            f"{len(inconsistent)} tax units have inconsistent filing_status. "
            f"First 5 tax_unit_ids: {inconsistent.index[:5].tolist()}"
        )

    def test_tax_unit_dependents_consistent_within_tax_unit(
        self, enhanced_persons: pd.DataFrame
    ):
        """All persons in the same tax unit must share the same tax_unit_dependents."""
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "tax_unit_dependents"])
        nunique = enhanced_persons.groupby("tax_unit_id")[
            "tax_unit_dependents"
        ].nunique()
        inconsistent = nunique[nunique > 1]
        assert inconsistent.empty, (
            f"{len(inconsistent)} tax units have inconsistent tax_unit_dependents. "
            f"First 5 tax_unit_ids: {inconsistent.index[:5].tolist()}"
        )

    def test_tax_unit_size_consistent_within_tax_unit(
        self, enhanced_persons: pd.DataFrame
    ):
        """All persons in the same tax unit must share the same tax_unit_size."""
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "tax_unit_size"])
        nunique = enhanced_persons.groupby("tax_unit_id")["tax_unit_size"].nunique()
        inconsistent = nunique[nunique > 1]
        assert inconsistent.empty, (
            f"{len(inconsistent)} tax units have inconsistent tax_unit_size. "
            f"First 5 tax_unit_ids: {inconsistent.index[:5].tolist()}"
        )

    def test_tax_unit_is_joint_consistent_within_tax_unit(
        self, enhanced_persons: pd.DataFrame
    ):
        """All persons in the same tax unit must share the same tax_unit_is_joint."""
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "tax_unit_is_joint"])
        nunique = enhanced_persons.groupby("tax_unit_id")[
            "tax_unit_is_joint"
        ].nunique()
        inconsistent = nunique[nunique > 1]
        assert inconsistent.empty, (
            f"{len(inconsistent)} tax units have inconsistent tax_unit_is_joint. "
            f"First 5 tax_unit_ids: {inconsistent.index[:5].tolist()}"
        )

    # -- Tax unit role coverage --

    def test_every_person_has_a_tax_unit_role(self, enhanced_persons: pd.DataFrame):
        """Every person must be head, spouse, or dependent in their tax unit.

        At least one of is_tax_unit_head, is_tax_unit_spouse,
        is_tax_unit_dependent must be True for each person.
        """
        role_cols = [
            "is_tax_unit_head",
            "is_tax_unit_spouse",
            "is_tax_unit_dependent",
        ]
        self._has_all_columns(enhanced_persons, role_cols)
        has_role = enhanced_persons[role_cols].any(axis=1)
        no_role_count = (~has_role).sum()
        assert no_role_count == 0, (
            f"{no_role_count} persons ({no_role_count / len(enhanced_persons):.1%}) "
            f"have no tax unit role (not head, spouse, or dependent)"
        )

    def test_at_most_one_head_per_tax_unit(self, enhanced_persons: pd.DataFrame):
        """Each tax unit should have at most one head."""
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "is_tax_unit_head"])
        heads_per_tu = enhanced_persons.groupby("tax_unit_id")[
            "is_tax_unit_head"
        ].sum()
        multi_head = heads_per_tu[heads_per_tu > 1]
        assert multi_head.empty, (
            f"{len(multi_head)} tax units have more than one head. "
            f"First 5: {multi_head.index[:5].tolist()}"
        )

    def test_at_most_one_spouse_per_tax_unit(self, enhanced_persons: pd.DataFrame):
        """Each tax unit should have at most one spouse."""
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "is_tax_unit_spouse"])
        spouses_per_tu = enhanced_persons.groupby("tax_unit_id")[
            "is_tax_unit_spouse"
        ].sum()
        multi_spouse = spouses_per_tu[spouses_per_tu > 1]
        assert multi_spouse.empty, (
            f"{len(multi_spouse)} tax units have more than one spouse. "
            f"First 5: {multi_spouse.index[:5].tolist()}"
        )

    # -- Joint filing consistency --

    def test_joint_filing_implies_joint_status(self, enhanced_persons: pd.DataFrame):
        """If tax_unit_is_joint is True, filing_status must be 'JOINT'."""
        self._has_all_columns(
            enhanced_persons, ["tax_unit_is_joint", "filing_status"]
        )
        joint_units = enhanced_persons[enhanced_persons["tax_unit_is_joint"]]
        if len(joint_units) == 0:
            pytest.skip("No joint tax units found in data")
        non_joint_status = joint_units[joint_units["filing_status"] != "JOINT"]
        assert len(non_joint_status) == 0, (
            f"{len(non_joint_status)} persons in joint tax units have "
            f"filing_status != 'JOINT'. "
            f"Actual statuses: {non_joint_status['filing_status'].value_counts().to_dict()}"
        )

    def test_non_joint_filing_implies_not_joint(self, enhanced_persons: pd.DataFrame):
        """If filing_status != 'JOINT', tax_unit_is_joint must be False."""
        self._has_all_columns(
            enhanced_persons, ["tax_unit_is_joint", "filing_status"]
        )
        non_joint = enhanced_persons[enhanced_persons["filing_status"] != "JOINT"]
        if len(non_joint) == 0:
            pytest.skip("All tax units are joint filers (unlikely)")
        mismatched = non_joint[non_joint["tax_unit_is_joint"]]
        assert len(mismatched) == 0, (
            f"{len(mismatched)} persons with non-JOINT filing_status "
            f"have tax_unit_is_joint=True. "
            f"Statuses: {mismatched['filing_status'].value_counts().to_dict()}"
        )

    # -- Tax unit size matches actual count --

    def test_tax_unit_size_reasonable(
        self, enhanced_persons: pd.DataFrame
    ):
        """tax_unit_size should be >= 1 and consistent within each tax unit.

        Note: PE-US computes tax_unit_size across all entity members, while our
        person extract may not include all members. So we check reasonableness
        rather than exact match with person count.
        """
        self._has_all_columns(enhanced_persons, ["tax_unit_id", "tax_unit_size"])
        assert (enhanced_persons["tax_unit_size"] >= 1).all(), "tax_unit_size must be >= 1"
        # Within each tax unit, all persons should have the same tax_unit_size
        valid = enhanced_persons.dropna(subset=["tax_unit_id"])
        per_tu = valid.groupby("tax_unit_id")["tax_unit_size"].nunique()
        inconsistent = (per_tu > 1).sum()
        assert inconsistent == 0, (
            f"{inconsistent} tax units have inconsistent tax_unit_size values"
        )

    # -- Dependents count matches dependent flag --

    def test_tax_unit_dependents_reasonable(
        self, enhanced_persons: pd.DataFrame
    ):
        """tax_unit_dependents should be >= 0 and consistent within each tax unit.

        Note: PE-US computes tax_unit_dependents across all entity members. Our
        person extract may not include all dependents (e.g., imputed members).
        So we check reasonableness rather than exact match.
        """
        self._has_all_columns(
            enhanced_persons,
            ["tax_unit_id", "tax_unit_dependents"],
        )
        assert (enhanced_persons["tax_unit_dependents"] >= 0).all(), (
            "tax_unit_dependents must be >= 0"
        )
        # Within each tax unit, all persons should have the same tax_unit_dependents
        valid = enhanced_persons.dropna(subset=["tax_unit_id"])
        per_tu = valid.groupby("tax_unit_id")["tax_unit_dependents"].nunique()
        inconsistent = (per_tu > 1).sum()
        assert inconsistent == 0, (
            f"{inconsistent} tax units have inconsistent tax_unit_dependents values"
        )


# =============================================================================
# 5. Missing data tests
# =============================================================================


class TestP1MissingData:
    """P1 columns should have minimal missing data (< 1% NaN)."""

    @pytest.mark.parametrize("column", ALL_P1_COLUMNS)
    def test_low_nan_rate(self, enhanced_persons: pd.DataFrame, column: str):
        """Column '{column}' must have less than 1% NaN values."""
        if column not in enhanced_persons.columns:
            pytest.skip(f"Column '{column}' not yet present")
        n_total = len(enhanced_persons)
        n_nan = enhanced_persons[column].isna().sum()
        nan_rate = n_nan / n_total
        assert nan_rate < 0.01, (
            f"Column '{column}' has {nan_rate:.2%} NaN values "
            f"({n_nan:,} of {n_total:,}), which exceeds the 1% threshold"
        )
