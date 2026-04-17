#!/usr/bin/env python3
"""Compare Microplex targets to PolicyEngine calibration targets."""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

# States list
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]


def parse_pe_calibration():
    """Parse all PolicyEngine calibration files."""
    cal_dir = Path("/opt/homebrew/lib/python3.13/site-packages/policyengine_us/parameters/calibration")
    pe_targets = []

    for yaml_file in cal_dir.rglob("*.yaml"):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            if not data:
                continue

            rel_path = yaml_file.relative_to(cal_dir)
            category = str(rel_path.parent).replace("/", ".")
            name = yaml_file.stem

            metadata = data.get('metadata', {})
            unit = metadata.get('unit', 'unknown')
            description = data.get('description', '')

            # Check if this has a 'values' key (national) or state keys directly
            if 'values' in data:
                # National target
                values = data['values']
                latest_date = max(values.keys())
                latest_value = values[latest_date]

                pe_targets.append({
                    'category': category,
                    'name': name,
                    'value': latest_value,
                    'year': latest_date.year if isinstance(latest_date, date) else int(str(latest_date)[:4]),
                    'unit': unit,
                    'geography': 'national',
                    'state': None,
                    'description': description
                })
            else:
                # State-level - look for state code keys
                for key, val in data.items():
                    if key in STATES and isinstance(val, dict):
                        # This is state-level data
                        latest_date = max(val.keys())
                        latest_value = val[latest_date]

                        pe_targets.append({
                            'category': category,
                            'name': name,
                            'value': latest_value,
                            'year': latest_date.year if isinstance(latest_date, date) else int(str(latest_date)[:4]),
                            'unit': unit,
                            'geography': 'state',
                            'state': key,
                            'description': description
                        })
        except Exception as e:
            print(f"Error parsing {yaml_file}: {e}")

    return pd.DataFrame(pe_targets)


def load_microplex_targets():
    """Load existing microplex targets."""
    return pd.read_parquet("data/targets.parquet")


def compare_targets():
    """Compare microplex and PE targets."""
    pe_df = parse_pe_calibration()
    mp_df = load_microplex_targets()

    print("=" * 80)
    print("POLICYENGINE VS MICROPLEX TARGET COMPARISON")
    print("=" * 80)

    # PE summary
    print(f"\n📊 PolicyEngine Calibration Targets: {len(pe_df)} total")
    print("\nBy category and geography:")
    print(pe_df.groupby(['category', 'geography']).size().to_string())

    # Microplex summary
    print(f"\n📊 Microplex Targets: {len(mp_df)} total")
    print("\nBy category:")
    print(mp_df['category'].value_counts().to_string())

    # Coverage comparison
    print("\n" + "=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)

    # Map PE categories to target types
    pe_df['category'].unique()

    coverage = []

    # National aggregates
    pe_national = pe_df[pe_df['geography'] == 'national']
    for _, row in pe_national.iterrows():
        name = row['name']
        val = row['value']

        # Try to find matching microplex target
        mp_match = None
        if 'snap' in name.lower():
            mp_match = mp_df[mp_df['name'].str.contains('snap', case=False)]
        elif 'ssi' in name.lower():
            mp_match = mp_df[mp_df['name'].str.contains('ssi', case=False)]
        elif 'medicaid' in name.lower():
            mp_match = mp_df[mp_df['name'].str.contains('medicaid', case=False)]
        elif 'eitc' in name.lower():
            mp_match = mp_df[mp_df['name'].str.contains('eitc', case=False)]
        elif 'population' in name.lower() or 'total' in name.lower():
            mp_match = mp_df[mp_df['name'].str.contains('population', case=False)]
        elif 'employment_income' in name.lower():
            mp_match = mp_df[mp_df['name'].str.contains('employment|wage', case=False)]

        mp_has = len(mp_match) > 0 if mp_match is not None else False

        coverage.append({
            'pe_category': row['category'],
            'pe_name': name,
            'pe_value': val,
            'pe_unit': row['unit'],
            'microplex_has': '✓' if mp_has else '✗',
            'geography': 'national'
        })

    # Print coverage table
    coverage_df = pd.DataFrame(coverage)

    print("\n📋 National Target Coverage:")
    print("-" * 80)

    for cat in coverage_df['pe_category'].unique():
        cat_df = coverage_df[coverage_df['pe_category'] == cat]
        has_count = (cat_df['microplex_has'] == '✓').sum()
        total = len(cat_df)
        print(f"\n{cat} ({has_count}/{total}):")

        for _, row in cat_df.iterrows():
            val = row['pe_value']
            if isinstance(val, (int, float)) and val > 1e9:
                val_str = f"${val/1e9:.1f}B"
            elif isinstance(val, (int, float)) and val > 1e6:
                val_str = f"{val/1e6:.1f}M"
            else:
                val_str = str(val)

            print(f"  {row['microplex_has']} {row['pe_name']}: {val_str}")

    # State-level targets
    pe_state = pe_df[pe_df['geography'] == 'state']
    state_cats = pe_state['name'].unique()

    print("\n" + "-" * 80)
    print(f"\n📋 State-Level Targets ({len(pe_state)} total, {len(state_cats)} types):")

    for cat in state_cats:
        cat_df = pe_state[pe_state['name'] == cat]
        print(f"  ✗ {cat}: {len(cat_df)} states")

    # Gap analysis
    print("\n" + "=" * 80)
    print("GAP ANALYSIS - TARGETS TO ADD")
    print("=" * 80)

    gaps = []

    # Income sources from IRS SOI
    print("\n📌 IRS SOI Income Sources (17 targets):")
    irs_targets = pe_national[pe_national['category'] == 'gov.irs.soi']
    for _, row in irs_targets.iterrows():
        val = row['value']
        val_str = f"${val/1e9:.1f}B" if val > 1e9 else f"${val/1e6:.1f}M"
        print(f"   - {row['name']}: {val_str}")
        gaps.append({'type': 'irs_soi', 'name': row['name'], 'value': val})

    # Benefit programs
    print("\n📌 Benefit Program Aggregates:")
    for _, row in pe_national.iterrows():
        if row['category'] in ['gov.cbo', 'gov.usda.snap', 'gov.ssa.ssi',
                               'gov.ssa.social_security', 'gov.treasury.tax_expenditures']:
            val = row['value']
            if isinstance(val, (int, float)):
                val_str = f"${val/1e9:.1f}B" if val > 1e9 else f"{val/1e6:.1f}M"
            else:
                val_str = str(val)
            print(f"   - {row['name']}: {val_str}")

    # State-level
    print(f"\n📌 State-Level Targets ({len(state_cats)} types × 51 states = {len(pe_state)}):")
    for cat in state_cats:
        print(f"   - {cat}")

    return pe_df, mp_df, coverage_df


if __name__ == "__main__":
    pe_df, mp_df, coverage_df = compare_targets()

    # Save comparison
    pe_df.to_parquet("data/pe_calibration_targets.parquet", index=False)
    print("\n✅ Saved PE targets to data/pe_calibration_targets.parquet")
