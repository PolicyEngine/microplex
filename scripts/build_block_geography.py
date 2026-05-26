#!/usr/bin/env python3
"""
Build block-level geography data for bottom-up synthesis.

Census blocks are the atomic geographic unit. All other geographies
(tract, county, CD, ZCTA) can be derived from block assignments.

Data sources:
- Block populations: Census PL 94-171 Redistricting Data (2020)
- Block-to-CD: Census Block Assignment Files

Storage:
- Raw Census data → arch/data/microdata/census_blocks/ (synced to R2)
- Processed tables → Supabase PostgreSQL
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime

# Optional Supabase for processed data
try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

# State FIPS codes (50 states + DC)
STATE_FIPS = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12',
    '13', '15', '16', '17', '18', '19', '20', '21', '22', '23',
    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
    '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
    '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56'
]

FIPS_TO_STATE = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA',
    '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC', '12': 'FL',
    '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME',
    '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
    '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH',
    '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
    '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
    '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY',
}


def fetch_state_counties(state_fips: str) -> list:
    """Get list of counties in a state."""
    url = 'https://api.census.gov/data/2020/dec/pl'
    params = {
        'get': 'NAME',
        'for': 'county:*',
        'in': f'state:{state_fips}',
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [row[2] for row in data[1:]]  # county codes
    except Exception as e:
        return []


def fetch_county_blocks(state_fips: str, county_fips: str) -> pd.DataFrame:
    """Fetch block populations for a county from Census API."""
    url = 'https://api.census.gov/data/2020/dec/pl'
    params = {
        'get': 'P1_001N',  # Total population
        'for': 'block:*',
        'in': f'state:{state_fips} county:{county_fips}',
    }

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Parse response
        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)

        # Build GEOID
        df['geoid'] = df['state'] + df['county'] + df['tract'] + df['block']
        df['population'] = pd.to_numeric(df['P1_001N'])
        df['state_fips'] = df['state']

        return df[['geoid', 'state_fips', 'county', 'tract', 'block', 'population']]

    except Exception as e:
        return pd.DataFrame()


def fetch_state_blocks(state_fips: str) -> pd.DataFrame:
    """Fetch block populations for a state by iterating over counties."""
    counties = fetch_state_counties(state_fips)
    if not counties:
        print(f"  No counties for {state_fips}", end=" ")
        return pd.DataFrame()

    all_county_blocks = []
    for county in counties:
        county_df = fetch_county_blocks(state_fips, county)
        if len(county_df) > 0:
            all_county_blocks.append(county_df)
        time.sleep(0.1)  # Rate limit

    if all_county_blocks:
        return pd.concat(all_county_blocks, ignore_index=True)
    return pd.DataFrame()


def fetch_block_to_cd_crosswalk() -> pd.DataFrame:
    """Fetch block-to-CD assignment from Census.

    Census provides Block Assignment Files (BAF) that map each block to its CD.
    Uses the 118th Congress tract-to-CD relationship file.
    """
    print("Fetching block-to-CD crosswalk...")

    # Tract-to-CD crosswalk for 118th Congress
    url = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/cd-sld/tab20_cd11820_tract20_natl.txt"

    try:
        df = pd.read_csv(url, delimiter='|', dtype=str)
        print(f"  Got {len(df)} tract-to-CD mappings")

        # Extract from GEOID columns:
        # GEOID_CD118_20 = SSCC (state + district)
        # GEOID_TRACT_20 = SSCCCTTTTTT (state + county + tract)
        df['tract_geoid'] = df['GEOID_TRACT_20']
        df['state_fips'] = df['GEOID_TRACT_20'].str[:2]
        df['cd'] = df['GEOID_CD118_20'].str[2:4]  # District number

        # Create CD identifier
        df['state_abbrev'] = df['state_fips'].map(FIPS_TO_STATE)
        df['cd_id'] = df['state_abbrev'] + '-' + df['cd']

        # Handle at-large (district 00 or 98)
        df.loc[df['cd'] == '00', 'cd_id'] = df.loc[df['cd'] == '00', 'state_abbrev'] + '-AL'
        df.loc[df['cd'] == '98', 'cd_id'] = df.loc[df['cd'] == '98', 'state_abbrev'] + '-AL'
        df.loc[df['cd'] == 'ZZ', 'cd_id'] = df.loc[df['cd'] == 'ZZ', 'state_abbrev'] + '-ZZ'

        return df[['tract_geoid', 'cd_id', 'state_fips']].drop_duplicates()

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_tract_to_sld_crosswalk() -> pd.DataFrame:
    """Fetch tract-to-SLD (State Legislative District) assignments from Census.

    Returns crosswalk for both upper (state senate) and lower (state house) chambers.
    Uses 2024 SLD boundaries.

    Note: Tracts can be split across multiple SLDs. We assign each tract to the
    SLD that contains the largest area of that tract (AREALAND_PART column).
    """
    print("Fetching tract-to-SLD crosswalks...")

    result_dfs = []

    # State Senate (Upper chamber) - SLDU
    url_upper = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/cd-sld/tab20_sldu202420_tract20_natl.txt"
    try:
        df = pd.read_csv(url_upper, delimiter='|', dtype=str)
        print(f"  Got {len(df)} tract-to-SLDU mappings")

        df['tract_geoid'] = df['GEOID_TRACT_20']
        df['state_fips'] = df['GEOID_TRACT_20'].str[:2]
        df['sldu'] = df['GEOID_SLDU2024_20'].str[2:]  # District number after state
        df['area'] = pd.to_numeric(df['AREALAND_PART'], errors='coerce')

        # Create SLD identifier: STATE-SLDU-XXX
        df['state_abbrev'] = df['state_fips'].map(FIPS_TO_STATE)
        df['sldu_id'] = df['state_abbrev'] + '-SLDU-' + df['sldu']

        # For tracts in multiple SLDs, keep only the SLD with the most area
        df = df.sort_values('area', ascending=False)
        df = df.drop_duplicates('tract_geoid', keep='first')
        print(f"    Deduplicated to {len(df)} unique tracts")

        result_dfs.append(df[['tract_geoid', 'sldu_id', 'state_fips']])

    except Exception as e:
        print(f"  Error fetching SLDU: {e}")

    # State House (Lower chamber) - SLDL
    url_lower = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/cd-sld/tab20_sldl202420_tract20_natl.txt"
    try:
        df = pd.read_csv(url_lower, delimiter='|', dtype=str)
        print(f"  Got {len(df)} tract-to-SLDL mappings")

        df['tract_geoid'] = df['GEOID_TRACT_20']
        df['state_fips'] = df['GEOID_TRACT_20'].str[:2]
        df['sldl'] = df['GEOID_SLDL2024_20'].str[2:]  # District number after state
        df['area'] = pd.to_numeric(df['AREALAND_PART'], errors='coerce')

        # Create SLD identifier: STATE-SLDL-XXX
        df['state_abbrev'] = df['state_fips'].map(FIPS_TO_STATE)
        df['sldl_id'] = df['state_abbrev'] + '-SLDL-' + df['sldl']

        # For tracts in multiple SLDs, keep only the SLD with the most area
        df = df.sort_values('area', ascending=False)
        df = df.drop_duplicates('tract_geoid', keep='first')
        print(f"    Deduplicated to {len(df)} unique tracts")

        result_dfs.append(df[['tract_geoid', 'sldl_id', 'state_fips']])

    except Exception as e:
        print(f"  Error fetching SLDL: {e}")

    if len(result_dfs) == 2:
        # Merge upper and lower on tract
        merged = result_dfs[0].merge(
            result_dfs[1][['tract_geoid', 'sldl_id']],
            on='tract_geoid',
            how='outer'
        )
        return merged

    elif len(result_dfs) == 1:
        return result_dfs[0]

    return pd.DataFrame()


def build_block_probabilities(blocks: pd.DataFrame) -> pd.DataFrame:
    """Build probability distribution for block sampling.

    Returns DataFrame with block geoid, population, and cumulative probability
    for efficient sampling.
    """
    # Filter to populated blocks
    populated = blocks[blocks['population'] > 0].copy()

    # Calculate probabilities within each state
    state_totals = populated.groupby('state_fips')['population'].sum()
    populated['state_total'] = populated['state_fips'].map(state_totals)
    populated['prob'] = populated['population'] / populated['state_total']

    # Also calculate national probability
    national_total = populated['population'].sum()
    populated['national_prob'] = populated['population'] / national_total

    return populated


def upload_to_supabase(df: pd.DataFrame, table_name: str) -> bool:
    """Upload processed data to Supabase PostgreSQL."""
    db_url = os.environ.get("POLICYENGINE_SUPABASE_DB_URL") or os.environ.get(
        "COSILICO_SUPABASE_DB_URL"
    )
    if not db_url or not HAVE_PSYCOPG2:
        print(f"  Skipping Supabase upload (no connection)")
        return False

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Create table if needed (for block_probabilities)
        if table_name == "block_probabilities":
            cur.execute("""
                CREATE TABLE IF NOT EXISTS block_probabilities (
                    geoid TEXT PRIMARY KEY,
                    state_fips TEXT NOT NULL,
                    county TEXT,
                    tract TEXT,
                    block TEXT,
                    population INTEGER,
                    cd_id TEXT,
                    prob REAL,
                    national_prob REAL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_block_probs_state ON block_probabilities(state_fips);
                CREATE INDEX IF NOT EXISTS idx_block_probs_cd ON block_probabilities(cd_id);
            """)

        # Truncate and reload
        cur.execute(f"TRUNCATE TABLE {table_name}")

        # Insert in batches
        cols = list(df.columns)
        values_template = ",".join(["%s"] * len(cols))
        insert_sql = f"INSERT INTO {table_name} ({','.join(cols)}) VALUES ({values_template})"

        batch_size = 10000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            cur.executemany(insert_sql, batch.values.tolist())
            print(f"    Uploaded {min(i+batch_size, len(df)):,}/{len(df):,} rows", end="\r")

        conn.commit()
        cur.close()
        conn.close()
        print(f"  Uploaded {len(df):,} rows to {table_name}")
        return True

    except Exception as e:
        print(f"  Supabase upload error: {e}")
        return False


def main():
    # Arch directory for raw Census data
    arch_dir = Path(__file__).parent.parent.parent / "arch" / "data" / "microdata" / "census_blocks"
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Local data directory for processed outputs
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("BUILDING BLOCK-LEVEL GEOGRAPHY DATA")
    print("=" * 70)
    print(f"Raw data: {arch_dir}")
    print(f"Processed data: {data_dir}")

    # Check if we already have cached raw data in arch
    raw_blocks_path = arch_dir / "pl94171_blocks_2020.parquet"
    if raw_blocks_path.exists():
        print(f"\nLoading cached raw block data from {raw_blocks_path}")
        blocks = pd.read_parquet(raw_blocks_path)
        print(f"  Loaded {len(blocks):,} blocks")
    else:
        # Fetch block populations for all states
        print(f"\nFetching block populations for {len(STATE_FIPS)} states...")
        print("(This may take several minutes)")

        all_blocks = []
        for i, state_fips in enumerate(STATE_FIPS):
            state_abbrev = FIPS_TO_STATE[state_fips]
            print(f"  [{i+1}/{len(STATE_FIPS)}] {state_abbrev}...", end=" ", flush=True)

            start = time.time()
            state_blocks = fetch_state_blocks(state_fips)
            elapsed = time.time() - start

            if len(state_blocks) > 0:
                all_blocks.append(state_blocks)
                pop = state_blocks['population'].sum()
                print(f"{len(state_blocks):,} blocks, {pop:,} pop ({elapsed:.1f}s)")
            else:
                print("FAILED")

            # Rate limiting
            time.sleep(0.5)

        blocks = pd.concat(all_blocks, ignore_index=True)
        print(f"\nTotal: {len(blocks):,} blocks")

        # Save raw data to arch
        blocks.to_parquet(raw_blocks_path, index=False)
        print(f"Saved raw data to {raw_blocks_path}")

        # Save metadata
        metadata = {
            "source": "Census PL 94-171 Redistricting Data",
            "year": 2020,
            "api_endpoint": "https://api.census.gov/data/2020/dec/pl",
            "fetched_at": datetime.now().isoformat(),
            "n_blocks": len(blocks),
            "n_states": blocks["state_fips"].nunique(),
            "total_population": int(blocks["population"].sum()),
        }
        with open(arch_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {arch_dir / 'metadata.json'}")

    # Summary statistics
    print(f"\nBlock statistics:")
    print(f"  Total blocks: {len(blocks):,}")
    print(f"  Populated blocks: {(blocks['population'] > 0).sum():,}")
    print(f"  Total population: {blocks['population'].sum():,}")
    print(f"  States: {blocks['state_fips'].nunique()}")

    # Fetch tract-to-CD crosswalk (skip if already present)
    if 'cd_id' not in blocks.columns:
        tract_cd = fetch_block_to_cd_crosswalk()

        if len(tract_cd) > 0:
            # Add CD to blocks via tract
            blocks['tract_geoid'] = blocks['geoid'].str[:11]
            blocks = blocks.merge(tract_cd[['tract_geoid', 'cd_id']], on='tract_geoid', how='left')

            # Fill missing CDs (some tracts might not have CD assignments)
            missing_cd = blocks['cd_id'].isna().sum()
            if missing_cd > 0:
                print(f"  Blocks without CD assignment: {missing_cd:,}")
    else:
        print("CD assignment already present in cached data")

    # Fetch tract-to-SLD crosswalk (skip if already present)
    need_sldu = 'sldu_id' not in blocks.columns
    need_sldl = 'sldl_id' not in blocks.columns

    if need_sldu or need_sldl:
        tract_sld = fetch_tract_to_sld_crosswalk()

        if len(tract_sld) > 0:
            # Add SLD to blocks via tract
            if 'tract_geoid' not in blocks.columns:
                blocks['tract_geoid'] = blocks['geoid'].str[:11]

            # Only merge columns we need
            sld_cols = ['tract_geoid']
            if need_sldu and 'sldu_id' in tract_sld.columns:
                sld_cols.append('sldu_id')
            if need_sldl and 'sldl_id' in tract_sld.columns:
                sld_cols.append('sldl_id')

            if len(sld_cols) > 1:  # More than just tract_geoid
                blocks = blocks.merge(tract_sld[sld_cols], on='tract_geoid', how='left')

                if 'sldu_id' in blocks.columns:
                    missing_sldu = blocks['sldu_id'].isna().sum()
                    if missing_sldu > 0:
                        print(f"  Blocks without SLDU assignment: {missing_sldu:,}")
                if 'sldl_id' in blocks.columns:
                    missing_sldl = blocks['sldl_id'].isna().sum()
                    if missing_sldl > 0:
                        print(f"  Blocks without SLDL assignment: {missing_sldl:,}")
    else:
        print("SLD assignment already present in cached data")

    # Save with CD and SLD back to arch
    blocks.to_parquet(raw_blocks_path, index=False)
    print(f"\nSaved blocks with CD/SLD assignments to {raw_blocks_path}")

    # Build probability lookup for efficient sampling
    print("\nBuilding block probability lookup...")
    block_probs = build_block_probabilities(blocks)
    print(f"  Populated blocks: {len(block_probs):,}")

    # Save probability lookup locally
    probs_path = data_dir / "block_probabilities.parquet"
    block_probs.to_parquet(probs_path, index=False)
    print(f"Saved locally to {probs_path}")

    # Upload to Supabase
    print("\nUploading processed data to Supabase...")
    upload_to_supabase(block_probs, "block_probabilities")

    # Summary by state
    print("\nBlocks by state (top 10):")
    state_counts = blocks.groupby('state_fips').agg({
        'geoid': 'count',
        'population': 'sum'
    }).sort_values('population', ascending=False)

    for state_fips, row in state_counts.head(10).iterrows():
        state_abbrev = FIPS_TO_STATE.get(state_fips, state_fips)
        print(f"  {state_abbrev}: {row['geoid']:,} blocks, {row['population']:,} pop")

    # Summary by legislative districts
    if 'cd_id' in blocks.columns:
        print(f"\nCDs covered: {blocks['cd_id'].dropna().nunique()}")
    if 'sldu_id' in blocks.columns:
        print(f"State Senate (SLDU) districts: {blocks['sldu_id'].dropna().nunique()}")
    if 'sldl_id' in blocks.columns:
        print(f"State House (SLDL) districts: {blocks['sldl_id'].dropna().nunique()}")

    print("\n" + "=" * 70)
    print("BLOCK GEOGRAPHY BUILD COMPLETE")
    print("=" * 70)
    print(f"Raw data:       {raw_blocks_path}")
    print(f"Processed data: {probs_path}")
    print(f"Supabase table: block_probabilities")


if __name__ == "__main__":
    main()
