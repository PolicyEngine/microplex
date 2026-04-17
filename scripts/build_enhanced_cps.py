#!/usr/bin/env python3
"""Build enhanced CPS microdata with full income/benefit columns from PE-US."""

import pandas as pd
import numpy as np
from pathlib import Path
from policyengine_us import Microsimulation


def build_enhanced_cps(year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build enhanced CPS person and household data from PE-US.

    Returns:
        Tuple of (persons_df, households_df)
    """
    print(f"Loading PE-US microsimulation for {year}...")
    sim = Microsimulation()

    # Person-level variables
    person_vars = [
        # IDs and weights
        'person_id', 'household_id', 'tax_unit_id', 'spm_unit_id',
        'person_weight',

        # Demographics
        'age', 'is_male', 'race', 'is_hispanic',
        'is_disabled', 'is_blind',
        'is_separated', 'is_surviving_spouse',

        # Tax unit role
        'is_tax_unit_head', 'is_tax_unit_spouse', 'is_tax_unit_dependent',

        # Employment
        'employment_income', 'self_employment_income',
        'weekly_hours_worked',

        # Investment income
        'dividend_income', 'interest_income', 'rental_income',
        'short_term_capital_gains', 'long_term_capital_gains',

        # Retirement income
        'social_security', 'pension_income',
        'taxable_pension_income', 'tax_exempt_pension_income',

        # Other income
        'ssi', 'unemployment_compensation', 'alimony_income',
        'farm_income', 'partnership_s_corp_income',

        # Benefits
        'snap', 'tanf', 'wic',
        'medicaid',

        # Tax credits
        'eitc', 'ctc', 'cdcc',

        # Geography
        'state_fips',
    ]

    # Household-level variables
    household_vars = [
        'household_id', 'household_weight',
        'household_size', 'household_income',
        'state_fips',
    ]

    # Load person data
    print("Loading person variables...")
    person_data = {}
    for var in person_vars:
        try:
            vals = sim.calculate(var, year)
            person_data[var] = vals
        except Exception as e:
            print(f"  Warning: {var} not available: {e}")

    persons_df = pd.DataFrame(person_data)
    print(f"  Persons: {len(persons_df):,} rows, {len(persons_df.columns)} columns")

    # Tax-unit level variables (need entity mapping to person level)
    print("Loading tax-unit variables...")
    tax_unit_vars = [
        'filing_status', 'tax_unit_dependents',
        'tax_unit_size', 'tax_unit_is_joint',
    ]
    tu_data = {}
    tu_data['tax_unit_id'] = sim.calculate('tax_unit_id', year)
    for var in tax_unit_vars:
        try:
            vals = sim.calculate(var, year)
            # filing_status is a PE Enum — convert to string labels
            if var == 'filing_status':
                vals = vals.decode() if hasattr(vals, 'decode') else np.array(vals).astype(str)
            tu_data[var] = vals
        except Exception as e:
            print(f"  Warning: {var} not available: {e}")

    tu_df = pd.DataFrame(tu_data)
    # Drop duplicate tax_unit rows (one row per tax unit)
    tu_df = tu_df.drop_duplicates(subset='tax_unit_id')
    print(f"  Tax units: {len(tu_df):,} rows")

    # Merge tax-unit variables onto person data
    if 'tax_unit_id' in persons_df.columns:
        persons_df = persons_df.merge(
            tu_df, on='tax_unit_id', how='left', suffixes=('', '_tu')
        )
        n_unmatched = persons_df['filing_status'].isna().sum()
        print(f"  After tax-unit merge: {len(persons_df):,} rows, {len(persons_df.columns)} columns")
        print(f"  Unmatched persons (NaN tax_unit_id): {n_unmatched:,}")

        # Fill defaults for persons without a tax unit (non-filers, dependents)
        if 'filing_status' in persons_df.columns:
            persons_df['filing_status'] = persons_df['filing_status'].fillna('SINGLE')
        if 'tax_unit_dependents' in persons_df.columns:
            persons_df['tax_unit_dependents'] = persons_df['tax_unit_dependents'].fillna(0).astype(int)
        if 'tax_unit_size' in persons_df.columns:
            persons_df['tax_unit_size'] = persons_df['tax_unit_size'].fillna(1).astype(int)
        if 'tax_unit_is_joint' in persons_df.columns:
            persons_df['tax_unit_is_joint'] = persons_df['tax_unit_is_joint'].fillna(False).astype(bool)

    # Derive is_married (approximate: joint filers or tax unit spouses)
    if 'filing_status' in persons_df.columns:
        persons_df['is_married'] = (
            (persons_df['filing_status'] == 'JOINT')
            | persons_df.get('is_tax_unit_spouse', pd.Series(False, index=persons_df.index)).astype(bool)
        )
    elif 'is_tax_unit_spouse' in persons_df.columns:
        persons_df['is_married'] = persons_df['is_tax_unit_spouse'].astype(bool)

    print(f"  Final persons: {len(persons_df):,} rows, {len(persons_df.columns)} columns")

    # Load household data
    print("Loading household variables...")
    household_data = {}
    for var in household_vars:
        try:
            vals = sim.calculate(var, year)
            household_data[var] = vals
        except Exception as e:
            print(f"  Warning: {var} not available: {e}")

    households_df = pd.DataFrame(household_data)
    print(f"  Households: {len(households_df):,} rows, {len(households_df.columns)} columns")

    return persons_df, households_df


def compute_calibration_totals(persons_df: pd.DataFrame) -> dict:
    """Compute weighted totals for calibration targets.

    Returns:
        Dict of target_name -> (computed_value, pe_target_value, error_pct)
    """
    from microplex.pe_targets import PETargets

    pe = PETargets()
    pe_national = pe.get_national_targets()

    # Weight column
    weight = persons_df.get('person_weight', pd.Series([1] * len(persons_df)))

    # Mapping of PE target names to our columns
    income_map = {
        'employment_income': 'employment_income',
        'self_employment_income': 'self_employment_income',
        'social_security': 'social_security',
        'dividend_income': 'dividend_income',
        'interest_income': 'interest_income',
        'rental_income': 'rental_income',
        'pension_income': 'pension_income',
        'ssi': 'ssi',
        'unemployment_compensation': 'unemployment_compensation',
    }

    results = {}

    for pe_name, col_name in income_map.items():
        if col_name in persons_df.columns:
            computed = (persons_df[col_name] * weight).sum()

            # Find PE target
            pe_row = pe_national[pe_national['name'] == pe_name]
            if not pe_row.empty:
                target = pe_row.iloc[0]['value']
                error = abs(computed - target) / target * 100
                results[pe_name] = {
                    'computed': computed,
                    'target': target,
                    'error_pct': error
                }

    return results


def main():
    # Build enhanced CPS
    persons_df, households_df = build_enhanced_cps(2024)

    # Save to parquet
    out_dir = Path("data")
    persons_df.to_parquet(out_dir / "cps_enhanced_persons.parquet", index=False)
    households_df.to_parquet(out_dir / "cps_enhanced_households.parquet", index=False)

    print(f"\n✅ Saved enhanced CPS data")
    print(f"   Persons: {out_dir / 'cps_enhanced_persons.parquet'}")
    print(f"   Households: {out_dir / 'cps_enhanced_households.parquet'}")

    # Compute and compare calibration totals
    print("\n=== CALIBRATION COMPARISON ===")
    results = compute_calibration_totals(persons_df)

    print(f"\n{'Variable':<30} {'Computed':>15} {'Target':>15} {'Error':>10}")
    print("-" * 75)

    for name, vals in sorted(results.items(), key=lambda x: -x[1]['target']):
        computed = vals['computed']
        target = vals['target']
        error = vals['error_pct']

        comp_str = f"${computed/1e9:.1f}B"
        tgt_str = f"${target/1e9:.1f}B"
        err_str = f"{error:.1f}%"

        print(f"{name:<30} {comp_str:>15} {tgt_str:>15} {err_str:>10}")


if __name__ == "__main__":
    main()
