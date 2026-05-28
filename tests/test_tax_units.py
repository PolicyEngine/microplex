import numpy as np
import pandas as pd
import pytest

from microplex.tax_units import build_preserved_tax_unit_tables


def test_preserved_tax_unit_tables_split_reused_string_ids_by_household():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "household_id": [10, 10, 20],
            "tax_unit_id": ["100:clone", "100:clone", "100:clone"],
            "income": [60_000.0, 15_000.0, 25_000.0],
            "tax_unit_is_joint": [1.0, 1.0, 0.0],
            "is_tax_unit_dependent": [0.0, 0.0, 0.0],
            "health_savings_account_ald": [60.0, 15.0, 5.0],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    assert result.persons["tax_unit_id"].tolist() == [1, 1, 2]
    assert result.preserved_households == {10, 20}
    tax_units = result.tax_units.sort_values("tax_unit_id").reset_index(
        drop=True
    )
    assert tax_units["household_id"].tolist() == [10, 20]
    assert tax_units["filing_status"].tolist() == ["JOINT", "SINGLE"]
    assert tax_units["total_income"].tolist() == [75_000.0, 25_000.0]
    assert tax_units["n_dependents"].tolist() == [0, 0]
    assert tax_units["health_savings_account_ald"].tolist() == [75.0, 5.0]


def test_preserved_tax_unit_tables_drop_incomplete_households():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "household_id": [10, 10, 20, 20],
            "tax_unit_id": ["a", None, "b", "b"],
            "income": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    assert result.preserved_households == {20}
    assert result.persons["person_id"].tolist() == [3, 4]
    assert result.tax_units["total_income"].tolist() == [70.0]


def test_preserved_tax_unit_tables_use_dependency_count_when_available():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "household_id": [10, 10, 10],
            "tax_unit_id": [100, 100, 100],
            "income": [10.0, 0.0, 0.0],
            "tax_unit_count_dependents": [2, 2, 2],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    assert result.tax_units["n_dependents"].tolist() == [2]


def test_preserved_tax_unit_tables_deduplicate_repeated_unit_values():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 10],
            "tax_unit_id": [100, 100],
            "income": [10.0, 20.0],
            "health_savings_account_ald": [500.0, 500.0],
            "unrecaptured_section_1250_gain": [100.0, 25.0],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    row = result.tax_units.iloc[0]
    assert row["health_savings_account_ald"] == 500.0
    assert row["unrecaptured_section_1250_gain"] == 125.0


def test_preserved_tax_unit_tables_clear_separated_flag_for_joint_filers():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 10],
            "tax_unit_id": ["joint", "joint"],
            "income": [10.0, 20.0],
            "tax_unit_is_joint": [1.0, 1.0],
            "is_tax_unit_head": [1.0, 0.0],
            "is_tax_unit_spouse": [0.0, 1.0],
            "is_separated": [True, True],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    assert result.tax_units["filing_status"].tolist() == ["JOINT"]
    assert result.persons["is_separated"].tolist() == [False, False]


def test_preserved_tax_unit_tables_add_mortgage_balance_floor():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2],
            "household_id": [10, 10],
            "tax_unit_id": [100, 100],
            "income": [10.0, 20.0],
            "home_mortgage_interest": [250.0, 50.0],
            "first_home_mortgage_origination_year": [0.0, 2020.0],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    row = result.tax_units.iloc[0]
    assert row["first_home_mortgage_interest"] == 300.0
    assert row["first_home_mortgage_balance"] == 300.0
    assert row["first_home_mortgage_origination_year"] == 2020.0


def test_preserved_tax_unit_tables_do_not_call_pandas_groupby(monkeypatch):
    def fail_groupby(self, *args, **kwargs):
        raise AssertionError("groupby should not be used in this hot path")

    monkeypatch.setattr(pd.DataFrame, "groupby", fail_groupby)
    monkeypatch.setattr(pd.Series, "groupby", fail_groupby)
    persons = pd.DataFrame(
        {
            "person_id": np.arange(6),
            "household_id": [1, 1, 2, 2, 3, 3],
            "tax_unit_id": ["a", "a", "a", "a", "b", "b"],
            "income": [1, 2, 3, 4, 5, 6],
            "is_tax_unit_dependent": [0, 1, 0, 0, 0, 1],
        }
    )

    result = build_preserved_tax_unit_tables(persons)

    assert result.persons["tax_unit_id"].tolist() == [1, 1, 2, 2, 3, 3]
    assert result.tax_units["total_income"].tolist() == [3.0, 7.0, 11.0]


def test_preserved_tax_unit_tables_require_complete_ids_when_not_dropping():
    persons = pd.DataFrame(
        {
            "person_id": [1],
            "household_id": [10],
            "tax_unit_id": [None],
        }
    )

    with pytest.raises(ValueError):
        build_preserved_tax_unit_tables(
            persons,
            complete_households_only=False,
        )
