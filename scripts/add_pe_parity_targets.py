#!/usr/bin/env python3
"""Add PolicyEngine-parity targets to microplex."""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

# States
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]


def parse_pe_yaml(yaml_file):
    """Parse a PE calibration YAML file."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    metadata = data.get('metadata', {})
    unit = metadata.get('unit', 'unknown')
    description = data.get('description', '')
    name = yaml_file.stem

    targets = []

    if 'values' in data:
        # National target
        values = data['values']
        latest_date = max(values.keys())
        latest_value = values[latest_date]

        targets.append({
            'name': name,
            'value': latest_value,
            'year': latest_date.year if isinstance(latest_date, date) else int(str(latest_date)[:4]),
            'unit': unit,
            'geography': 'national',
            'state': None,
            'source': 'policyengine',
            'description': description
        })
    else:
        # State-level data
        for key, val in data.items():
            if key in STATES and isinstance(val, dict):
                latest_date = max(val.keys())
                latest_value = val[latest_date]

                targets.append({
                    'name': f"{name}_{key}",
                    'value': latest_value,
                    'year': latest_date.year if isinstance(latest_date, date) else int(str(latest_date)[:4]),
                    'unit': unit,
                    'geography': 'state',
                    'state': key,
                    'source': 'policyengine',
                    'description': description
                })

    return targets


def get_all_pe_targets():
    """Get all PE calibration targets."""
    cal_dir = Path("/opt/homebrew/lib/python3.13/site-packages/policyengine_us/parameters/calibration")

    all_targets = []

    # Mapping of PE paths to microplex categories
    category_map = {
        'gov.irs.soi': 'irs_soi',
        'gov.cbo': 'cbo_aggregates',
        'gov.usda.snap': 'snap',
        'gov.ssa.ssi': 'ssi',
        'gov.ssa.social_security': 'social_security',
        'gov.treasury.tax_expenditures': 'tax_expenditures',
        'gov.census.populations': 'census_population',
        'gov.hhs.medicaid': 'medicaid',
        'gov.hhs.cms.chip': 'chip',
        'gov.aca': 'aca',
        'gov.hhs.medicare': 'medicare',
        'gov.hhs.cms': 'cms',
    }

    for yaml_file in cal_dir.rglob("*.yaml"):
        try:
            rel_path = str(yaml_file.relative_to(cal_dir).parent).replace("/", ".")

            # Find category
            category = 'other'
            for path_prefix, cat in category_map.items():
                if rel_path.startswith(path_prefix):
                    category = cat
                    break

            targets = parse_pe_yaml(yaml_file)
            for t in targets:
                t['category'] = category
                t['pe_path'] = rel_path

            all_targets.extend(targets)
        except Exception as e:
            print(f"Error parsing {yaml_file}: {e}")

    return pd.DataFrame(all_targets)


def merge_with_existing():
    """Merge PE targets with existing microplex targets."""
    # Load existing
    existing = pd.read_parquet("data/targets.parquet")
    print(f"Existing microplex targets: {len(existing)}")

    # Get PE targets
    pe_targets = get_all_pe_targets()
    print(f"PolicyEngine targets: {len(pe_targets)}")

    # Rename columns to match existing schema
    pe_targets = pe_targets.rename(columns={
        'source': 'source',
    })

    # Add missing columns
    for col in existing.columns:
        if col not in pe_targets.columns:
            pe_targets[col] = None

    # Select only columns that exist in both
    list(set(existing.columns) & set(pe_targets.columns))

    # For now, just save PE targets separately
    pe_targets.to_parquet("data/pe_parity_targets.parquet", index=False)
    print(f"\n✅ Saved {len(pe_targets)} PE-parity targets to data/pe_parity_targets.parquet")

    return pe_targets


def print_summary(df):
    """Print target summary."""
    print("\n" + "=" * 80)
    print("PE-PARITY TARGETS SUMMARY")
    print("=" * 80)

    print("\nBy category:")
    print(df.groupby('category').size().sort_values(ascending=False).to_string())

    print("\nBy geography:")
    print(df.groupby('geography').size().to_string())

    print("\nNational targets by category:")
    national = df[df['geography'] == 'national']
    for cat in national['category'].unique():
        cat_df = national[national['category'] == cat]
        print(f"\n{cat}:")
        for _, row in cat_df.iterrows():
            val = row['value']
            if isinstance(val, (int, float)):
                if abs(val) > 1e9:
                    val_str = f"${val/1e9:.1f}B"
                elif abs(val) > 1e6:
                    val_str = f"{val/1e6:.1f}M"
                else:
                    val_str = f"{val:,.0f}"
            else:
                val_str = str(val)
            print(f"  - {row['name']}: {val_str}")


if __name__ == "__main__":
    pe_targets = merge_with_existing()
    print_summary(pe_targets)
