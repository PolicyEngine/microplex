from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreservedTaxUnitTables:
    tax_units: pd.DataFrame
    persons: pd.DataFrame
    preserved_households: set[Any]


def build_preserved_tax_unit_tables(
    persons: pd.DataFrame,
    *,
    household_id_col: str = "household_id",
    tax_unit_id_col: str = "tax_unit_id",
    person_id_col: str = "person_id",
    income_col: str = "income",
    complete_households_only: bool = True,
    deduplicated_sum_columns: Sequence[str] = (
        "health_savings_account_ald",
        "unrecaptured_section_1250_gain",
    ),
) -> PreservedTaxUnitTables:
    """Build tax-unit rows from existing person-level tax-unit IDs.

    Use this when tax-unit membership already exists and should be preserved.
    It normalizes household/tax-unit pairs to dense integer IDs, then
    aggregates with NumPy reductions instead of pandas groupby/MultiIndex
    construction. Use :class:`microplex.hierarchical.TaxUnitOptimizer` when
    memberships need to be inferred from household composition.
    """

    _require_columns(
        persons,
        [person_id_col, household_id_col, tax_unit_id_col],
    )
    if persons.empty:
        return PreservedTaxUnitTables(
            tax_units=_empty_tax_units(),
            persons=persons.copy(),
            preserved_households=set(),
        )

    has_tax_unit_id = _non_missing_id_mask(persons[tax_unit_id_col])
    if complete_households_only:
        row_mask, preserved_households = _complete_household_row_mask(
            persons[household_id_col],
            has_tax_unit_id,
        )
    else:
        if not bool(has_tax_unit_id.all()):
            raise ValueError(
                "tax_unit_id contains missing values; pass "
                "complete_households_only=True to drop incomplete households"
            )
        row_mask = np.ones(len(persons), dtype=bool)
        preserved_households = set(persons[household_id_col].tolist())

    person_rows = persons.loc[row_mask].copy()
    if person_rows.empty:
        return PreservedTaxUnitTables(
            tax_units=_empty_tax_units(),
            persons=person_rows,
            preserved_households=set(),
        )

    unit_codes = _factorize_household_entity_pairs(
        person_rows[household_id_col],
        person_rows[tax_unit_id_col],
    )
    n_units = int(unit_codes.max()) + 1
    first_pos = _first_positions(unit_codes, n_units)
    tax_unit_ids = np.arange(1, n_units + 1, dtype=np.int64)

    person_rows[tax_unit_id_col] = tax_unit_ids[unit_codes]
    household_ids = person_rows[household_id_col].to_numpy()
    tax_units = pd.DataFrame(
        {
            "tax_unit_id": tax_unit_ids,
            "household_id": household_ids[first_pos],
            "total_income": _group_sum(
                _numeric_column(person_rows, income_col),
                unit_codes,
                n_units,
            ),
            "tax_liability": np.zeros(n_units, dtype=float),
            "n_dependents": _tax_unit_dependent_count(
                person_rows,
                unit_codes,
                n_units,
            ),
            "filing_status": _tax_unit_filing_status(
                person_rows,
                unit_codes,
                n_units,
                first_pos,
            ),
        }
    )

    for column in deduplicated_sum_columns:
        if column in person_rows.columns:
            tax_units[column] = _deduplicating_group_sum(
                _numeric_column(person_rows, column),
                unit_codes,
                n_units,
            )

    _sync_joint_separated_flags(tax_units, person_rows, unit_codes)
    _add_mortgage_columns(tax_units, person_rows, unit_codes, n_units)
    return PreservedTaxUnitTables(
        tax_units=tax_units,
        persons=person_rows,
        preserved_households=preserved_households,
    )


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        joined = ", ".join(sorted(missing))
        raise KeyError(f"Missing required column(s): {joined}")


def _empty_tax_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tax_unit_id": np.array([], dtype=np.int64),
            "household_id": np.array([], dtype=np.int64),
            "total_income": np.array([], dtype=float),
            "tax_liability": np.array([], dtype=float),
            "n_dependents": np.array([], dtype=np.int64),
            "filing_status": np.array([], dtype=object),
        }
    )


def _non_missing_id_mask(values: pd.Series) -> np.ndarray:
    mask = values.notna().to_numpy(dtype=bool, copy=True)
    if (
        pd.api.types.is_object_dtype(values.dtype)
        or pd.api.types.is_string_dtype(values.dtype)
    ):
        text = values.astype("string")
        mask &= text.str.strip().ne("").fillna(False).to_numpy(dtype=bool)
    return mask


def _complete_household_row_mask(
    household_ids: pd.Series,
    has_entity_id: np.ndarray,
) -> tuple[np.ndarray, set[Any]]:
    household_codes, household_uniques = pd.factorize(
        household_ids,
        sort=False,
    )
    valid_household = household_codes >= 0
    complete_by_household = np.ones(len(household_uniques), dtype=bool)
    np.logical_and.at(
        complete_by_household,
        household_codes[valid_household],
        has_entity_id[valid_household],
    )

    row_mask = np.zeros(len(household_ids), dtype=bool)
    row_mask[valid_household] = complete_by_household[
        household_codes[valid_household]
    ]
    preserved = set(pd.Series(household_uniques[complete_by_household]).tolist())
    return row_mask, preserved


def _factorize_household_entity_pairs(
    household_ids: pd.Series,
    entity_ids: pd.Series,
) -> np.ndarray:
    household_codes, household_uniques = pd.factorize(
        household_ids,
        sort=False,
    )
    entity_codes, entity_uniques = pd.factorize(entity_ids, sort=False)
    if bool((household_codes < 0).any() or (entity_codes < 0).any()):
        raise ValueError("Cannot factorize missing household/entity IDs")

    base = int(len(entity_uniques))
    if base == 0:
        return np.array([], dtype=np.int64)
    if len(household_uniques) > np.iinfo(np.int64).max // base:
        raise OverflowError("Too many household/entity pairs to encode")

    pair_keys = household_codes.astype(np.int64) * base + entity_codes
    pair_codes, _ = pd.factorize(pair_keys, sort=False)
    return pair_codes.astype(np.int64, copy=False)


def _first_positions(group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    _, first_pos = np.unique(group_codes, return_index=True)
    if len(first_pos) != n_groups:
        raise ValueError("Group codes are not dense")
    return first_pos


def _numeric_column(
    df: pd.DataFrame,
    column: str,
    *,
    default: float = 0.0,
) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), default, dtype=float)
    return (
        pd.to_numeric(df[column], errors="coerce")
        .fillna(default)
        .to_numpy(dtype=float)
    )


def _group_sum(
    values: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    return np.bincount(group_codes, weights=values, minlength=n_groups)


def _group_max(
    values: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    result = np.full(n_groups, -np.inf, dtype=float)
    np.maximum.at(result, group_codes, values)
    result[~np.isfinite(result)] = 0.0
    return result


def _tax_unit_dependent_count(
    person_rows: pd.DataFrame,
    unit_codes: np.ndarray,
    n_units: int,
) -> np.ndarray:
    if "tax_unit_count_dependents" in person_rows.columns:
        return _group_max(
            _numeric_column(person_rows, "tax_unit_count_dependents"),
            unit_codes,
            n_units,
        ).astype(np.int64)
    if "is_tax_unit_dependent" in person_rows.columns:
        dependents = _numeric_column(
            person_rows,
            "is_tax_unit_dependent",
        ) > 0
        return np.bincount(
            unit_codes,
            weights=dependents.astype(np.int64),
            minlength=n_units,
        ).astype(np.int64)
    return np.zeros(n_units, dtype=np.int64)


def _tax_unit_filing_status(
    person_rows: pd.DataFrame,
    unit_codes: np.ndarray,
    n_units: int,
    first_pos: np.ndarray,
) -> np.ndarray:
    if "filing_status" in person_rows.columns:
        raw = person_rows["filing_status"].to_numpy(dtype=object)[first_pos]
        return _normalize_filing_status(raw)
    if "tax_unit_is_joint" in person_rows.columns:
        joint = _group_max(
            _numeric_column(person_rows, "tax_unit_is_joint"),
            unit_codes,
            n_units,
        ) > 0
        return np.where(joint, "JOINT", "SINGLE")
    return np.full(n_units, "SINGLE", dtype=object)


def _normalize_filing_status(values: Sequence[Any]) -> np.ndarray:
    result = []
    for value in values:
        text = str(value).strip().upper() if value is not None else ""
        text = text.replace(" ", "_")
        if text in {"JOINT", "MARRIED_FILING_JOINTLY"}:
            result.append("JOINT")
        elif text in {"SEPARATE", "MARRIED_FILING_SEPARATELY"}:
            result.append("SEPARATE")
        elif text in {"HEAD_OF_HOUSEHOLD", "HEAD"}:
            result.append("HEAD_OF_HOUSEHOLD")
        elif text in {"SURVIVING_SPOUSE", "WIDOW", "WIDOWER"}:
            result.append("SURVIVING_SPOUSE")
        else:
            result.append("SINGLE")
    return np.asarray(result, dtype=object)


def _deduplicating_group_sum(
    values: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    summed = _group_sum(values, group_codes, n_groups)
    nonzero = ~np.isclose(values, 0.0)
    nonzero_count = np.bincount(
        group_codes,
        weights=nonzero.astype(np.int64),
        minlength=n_groups,
    )
    nonzero_codes = group_codes[nonzero]
    nonzero_values = values[nonzero]
    minimum = np.full(n_groups, np.inf, dtype=float)
    maximum = np.full(n_groups, -np.inf, dtype=float)
    if len(nonzero_values):
        np.minimum.at(minimum, nonzero_codes, nonzero_values)
        np.maximum.at(maximum, nonzero_codes, nonzero_values)

    repeated_same_value = (
        nonzero_count > 1
    ) & np.isfinite(minimum) & np.isclose(minimum, maximum)
    result = summed.copy()
    result[repeated_same_value] = maximum[repeated_same_value]
    return result


def _sync_joint_separated_flags(
    tax_units: pd.DataFrame,
    person_rows: pd.DataFrame,
    unit_codes: np.ndarray,
) -> None:
    if "is_separated" not in person_rows.columns:
        return
    if not (
        "is_tax_unit_head" in person_rows.columns
        or "is_tax_unit_spouse" in person_rows.columns
    ):
        return

    joint_units = (
        tax_units["filing_status"].astype("string").str.upper().eq("JOINT")
    ).to_numpy(dtype=bool)
    if not bool(np.any(joint_units)):
        return

    head = _numeric_column(person_rows, "is_tax_unit_head")
    spouse = _numeric_column(person_rows, "is_tax_unit_spouse")
    filer_rows = (head > 0.0) | (spouse > 0.0)
    person_rows.loc[filer_rows & joint_units[unit_codes], "is_separated"] = False


def _add_mortgage_columns(
    tax_units: pd.DataFrame,
    person_rows: pd.DataFrame,
    unit_codes: np.ndarray,
    n_units: int,
) -> None:
    interest_column = None
    if "first_home_mortgage_interest" in person_rows.columns:
        interest_column = "first_home_mortgage_interest"
    elif "home_mortgage_interest" in person_rows.columns:
        interest_column = "home_mortgage_interest"
    if interest_column is None:
        return

    interest = _group_sum(
        _numeric_column(person_rows, interest_column),
        unit_codes,
        n_units,
    )
    if not bool(np.any(interest > 0)):
        return

    tax_units["first_home_mortgage_interest"] = interest
    balance = _group_sum(
        _numeric_column(person_rows, "first_home_mortgage_balance"),
        unit_codes,
        n_units,
    )
    tax_units["first_home_mortgage_balance"] = np.where(
        balance > 0.0,
        balance,
        np.maximum(interest, 1.0),
    )

    if "first_home_mortgage_origination_year" not in person_rows.columns:
        return
    years = _numeric_column(person_rows, "first_home_mortgage_origination_year")
    positive_pos = np.flatnonzero(years > 0)
    if not len(positive_pos):
        return
    first_positive_pos = np.full(n_units, len(years), dtype=np.int64)
    np.minimum.at(first_positive_pos, unit_codes[positive_pos], positive_pos)
    has_year = first_positive_pos < len(years)
    result = np.zeros(n_units, dtype=float)
    result[has_year] = years[first_positive_pos[has_year]]
    tax_units["first_home_mortgage_origination_year"] = result
