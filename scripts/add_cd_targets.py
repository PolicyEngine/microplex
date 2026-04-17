"""
Add Congressional District targets and PUMA-to-CD mapping.

Sources:
- CD populations: Census ACS 5-year 2022 (Table B01003)
- PUMA-CD crosswalk: Census PUMA-CD allocation factors
"""

from pathlib import Path

import pandas as pd
import requests

# State FIPS to abbreviation
FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def fetch_cd_populations() -> pd.DataFrame:
    """Fetch congressional district populations from Census ACS 5-year."""
    print("Fetching CD populations from Census API...")

    url = "https://api.census.gov/data/2022/acs/acs5"
    params = {
        "get": "NAME,B01003_001E",  # Total population
        "for": "congressional district:*",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Parse response
    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # Convert to proper types
    df["population"] = pd.to_numeric(df["B01003_001E"])
    df["state_fips"] = pd.to_numeric(df["state"])
    df["cd_number"] = df["congressional district"]

    # Create CD identifier: state_abbrev-cd_number (e.g., "CA-12")
    df["state_abbrev"] = df["state_fips"].map(FIPS_TO_STATE)
    df["cd_id"] = df["state_abbrev"] + "-" + df["cd_number"]

    # Handle at-large districts (CD 00)
    df.loc[df["cd_number"] == "00", "cd_id"] = df.loc[df["cd_number"] == "00", "state_abbrev"] + "-AL"

    # Filter out territories (Puerto Rico, etc.)
    df = df[df["state_abbrev"].notna()].copy()

    print(f"  Got {len(df)} congressional districts")
    print(f"  Total population: {df['population'].sum():,.0f}")

    return df[["cd_id", "state_fips", "state_abbrev", "cd_number", "population", "NAME"]]


def create_cd_targets(cd_pops: pd.DataFrame) -> pd.DataFrame:
    """Create target records for congressional districts."""
    targets = []

    for _, row in cd_pops.iterrows():
        targets.append({
            "name": f"population_{row['cd_id']}",
            "category": "population",
            "value": float(row["population"]),
            "year": 2022,
            "source": "Census ACS 5-year 2022",
            "geography": row["cd_id"],
            "state_fips": None,  # Use None like other targets, geography has CD info
            "filing_status": None,
            "agi_lower": None,
            "agi_upper": None,
            "is_count": True,
            "rac_variable": None,
            "rac_statute": None,
            "microdata_column": "cd_id",
        })

    return pd.DataFrame(targets)


def create_state_cd_probabilities(cd_pops: pd.DataFrame) -> pd.DataFrame:
    """Create probability distribution of CDs within each state.

    Since CPS only has state_fips (no PUMA), we assign households to CDs
    probabilistically based on CD population shares within each state.

    Returns DataFrame with columns: state_fips, cd_id, prob
    """
    print("Creating state-to-CD probability mapping...")

    # Calculate CD population share within each state
    state_totals = cd_pops.groupby("state_fips")["population"].sum().reset_index()
    state_totals.columns = ["state_fips", "state_total_pop"]

    cd_probs = cd_pops.merge(state_totals, on="state_fips")
    cd_probs["prob"] = cd_probs["population"] / cd_probs["state_total_pop"]

    print(f"  States with CDs: {cd_probs['state_fips'].nunique()}")
    print(f"  Total CDs: {len(cd_probs)}")

    # Verify probabilities sum to 1 within each state
    prob_check = cd_probs.groupby("state_fips")["prob"].sum()
    assert (prob_check.round(6) == 1.0).all(), "Probabilities don't sum to 1"

    return cd_probs[["state_fips", "cd_id", "prob", "population"]]


def main():
    data_dir = Path(__file__).parent.parent / "data"

    # 1. Fetch CD populations
    cd_pops = fetch_cd_populations()

    # 2. Create CD targets
    cd_targets = create_cd_targets(cd_pops)
    print(f"\nCreated {len(cd_targets)} CD population targets")

    # 3. Create state-to-CD probability mapping
    # (Since CPS lacks PUMA, we use probabilistic CD assignment)
    cd_probs = create_state_cd_probabilities(cd_pops)
    cd_probs.to_parquet(data_dir / "state_cd_probabilities.parquet", index=False)
    print(f"Saved state-CD probabilities to {data_dir / 'state_cd_probabilities.parquet'}")

    # 4. Load existing targets and append
    existing_targets = pd.read_parquet(data_dir / "targets.parquet")
    print(f"\nExisting targets: {len(existing_targets)}")

    # Remove any existing CD targets (in case we're re-running)
    # CD geographies look like "CA-12" or "WY-AL" (at-large)
    existing_targets = existing_targets[
        ~existing_targets["geography"].str.match(r"^[A-Z]{2}-\d{2}$|^[A-Z]{2}-AL$", na=False)
    ].copy()
    print(f"After removing existing CD targets: {len(existing_targets)}")

    # Append CD targets
    all_targets = pd.concat([existing_targets, cd_targets], ignore_index=True)
    print(f"Total targets after adding CDs: {len(all_targets)}")

    # 5. Save updated targets
    all_targets.to_parquet(data_dir / "targets.parquet", index=False)
    print(f"Saved to {data_dir / 'targets.parquet'}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Congressional districts: {len(cd_targets)}")
    print(f"Total population in CDs: {cd_pops['population'].sum():,.0f}")
    print(f"Average CD population: {cd_pops['population'].mean():,.0f}")
    print(f"Min CD population: {cd_pops['population'].min():,.0f} ({cd_pops.loc[cd_pops['population'].idxmin(), 'cd_id']})")
    print(f"Max CD population: {cd_pops['population'].max():,.0f} ({cd_pops.loc[cd_pops['population'].idxmax(), 'cd_id']})")


if __name__ == "__main__":
    main()
