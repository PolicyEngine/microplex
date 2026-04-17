#!/usr/bin/env python3
"""Generate target coverage report comparing microplex capabilities vs PE targets."""


import pandas as pd


def load_targets():
    """Load both target sets."""
    mp = pd.read_parquet("data/targets.parquet")
    pe = pd.read_parquet("data/pe_parity_targets.parquet")
    return mp, pe


def get_cps_columns():
    """Get available CPS microdata columns."""
    cps = pd.read_parquet("data/cps_asec_persons.parquet")
    return list(cps.columns)


def generate_report():
    """Generate comprehensive coverage report."""
    mp, pe = load_targets()
    cps_cols = get_cps_columns()

    print("=" * 80)
    print("MICROPLEX TARGET COVERAGE REPORT")
    print("=" * 80)

    # Summary stats
    mp_national = mp[mp['geography'].fillna('national') != 'state'] if 'geography' in mp.columns else mp
    mp_state = mp[mp['geography'] == 'state'] if 'geography' in mp.columns else pd.DataFrame()

    pe_national = pe[pe['geography'] == 'national']
    pe_state = pe[pe['geography'] == 'state']

    print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│                           TARGET INVENTORY                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                        Microplex              PolicyEngine                 │
│ ────────────────────   ─────────              ────────────                 │
│ Total targets:           {len(mp):>5,}                   {len(pe):>5,}                 │
│ National:                {len(mp_national):>5,}                   {len(pe_national):>5,}                 │
│ State-level:             {len(mp_state):>5,}                   {len(pe_state):>5,}                 │
└────────────────────────────────────────────────────────────────────────────┘
""")

    # Microplex target breakdown
    print("\n📊 MICROPLEX TARGETS BY CATEGORY")
    print("-" * 80)
    mp_cats = mp['category'].value_counts()
    for cat, count in mp_cats.items():
        print(f"  {cat:40} {count:>6}")

    # PE target breakdown
    print("\n📊 POLICYENGINE TARGETS BY CATEGORY")
    print("-" * 80)
    pe_cats = pe['category'].value_counts()
    for cat, count in pe_cats.items():
        print(f"  {cat:40} {count:>6}")

    # Gap analysis
    print("\n" + "=" * 80)
    print("GAP ANALYSIS: WHAT MICROPLEX NEEDS TO ADD")
    print("=" * 80)

    # Map PE categories to what microplex can compute
    compute_map = {
        'irs_soi': {
            'employment_income': ['WSAL_VAL', 'ERN_VAL'],  # wages
            'self_employment_income': ['SEMP_VAL'],
            'social_security': ['SS_VAL'],
            'unemployment_compensation': ['UC_VAL'],
            'taxable_pension_income': ['RET_VAL1', 'RET_VAL2'],
            'taxable_interest_income': ['INT_VAL'],
            'qualified_dividend_income': ['DIV_VAL'],
            'rental_income': ['RNT_VAL'],
        },
        'snap': {
            'participation': ['SNAP_YN', 'HFDVAL'],
        },
        'ssi': {
            'participation': ['SSI_VAL'],
        },
        'social_security': {
            'participation': ['SS_VAL'],
        },
        'medicaid': {
            'enrollment': ['MEDICAID'],  # if available
        }
    }

    print("\n✅ CAN COMPUTE FROM CPS (have microdata columns):")
    computable = []
    for pe_cat, variables in compute_map.items():
        for var, cols in variables.items():
            available = [c for c in cols if c in cps_cols]
            if available:
                computable.append((pe_cat, var, available))
                print(f"   {pe_cat}/{var}: {available}")

    print(f"\n   Total computable: {len(computable)} variables")

    print("\n⚠️ NEED IMPUTATION (not in CPS or limited):")
    limited = [
        ('irs_soi/long_term_capital_gains', 'No detailed capital gains in CPS'),
        ('irs_soi/short_term_capital_gains', 'No capital gains split in CPS'),
        ('irs_soi/partnership_s_corp_income', 'Limited business income detail'),
        ('irs_soi/farm_income', 'FARM_VAL may be limited'),
        ('irs_soi/alimony_income', 'OI_VAL may be limited'),
        ('medicaid/by_eligibility_group', 'Need to model eligibility'),
        ('chip/enrollment', 'Need child health insurance flag'),
        ('aca/enrollment', 'Need marketplace enrollment flag'),
    ]
    for item, reason in limited:
        print(f"   {item}: {reason}")

    # CPS column summary
    print("\n" + "=" * 80)
    print("CPS ASEC COLUMN INVENTORY (for target computation)")
    print("=" * 80)

    # Income columns
    income_cols = [c for c in cps_cols if any(x in c for x in ['VAL', 'INC', 'ERN', 'WAG'])]
    print(f"\n📈 Income columns ({len(income_cols)}):")
    for c in sorted(income_cols)[:20]:
        print(f"   {c}")
    if len(income_cols) > 20:
        print(f"   ... and {len(income_cols) - 20} more")

    # Benefit columns
    benefit_cols = [c for c in cps_cols if any(x in c.upper() for x in ['SNAP', 'SSI', 'SS_', 'MEDICAID', 'MEDICARE', 'WIC', 'TANF'])]
    print(f"\n🏥 Benefit columns ({len(benefit_cols)}):")
    for c in sorted(benefit_cols):
        print(f"   {c}")

    # Demographic columns
    demo_cols = [c for c in cps_cols if any(x in c.upper() for x in ['AGE', 'SEX', 'RACE', 'HISP', 'MAR', 'EDUC'])]
    print(f"\n👥 Demographic columns ({len(demo_cols)}):")
    for c in sorted(demo_cols):
        print(f"   {c}")

    # Create combined target universe
    print("\n" + "=" * 80)
    print("COMBINED TARGET UNIVERSE FOR CALIBRATION")
    print("=" * 80)

    # Count unique targets by type
    target_types = {
        'Geographic (CD/State/SLDU)': len(mp[mp['category'] == 'population']),
        'Age distribution': len(mp[mp['category'] == 'age_distribution']),
        'AGI distribution': len(mp[mp['category'] == 'agi_distribution']),
        'IRS SOI income totals': len(pe[pe['category'] == 'irs_soi']),
        'Benefit program totals': len(pe[pe['category'].isin(['snap', 'ssi', 'social_security', 'cbo_aggregates'])]),
        'Medicaid/CHIP by state': len(pe[pe['category'].isin(['medicaid', 'chip'])]),
        'ACA enrollment by state': len(pe[pe['category'] == 'aca']),
    }

    total = sum(target_types.values())
    print(f"\nTotal unique target dimensions: {total:,}")
    for name, count in target_types.items():
        pct = count / total * 100
        print(f"  {name:40} {count:>6} ({pct:5.1f}%)")

    return mp, pe


if __name__ == "__main__":
    mp, pe = generate_report()
