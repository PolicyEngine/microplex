"""
Download and process CPS ASEC microdata from Census Bureau.

This script downloads the CPS ASEC (Current Population Survey Annual Social and
Economic Supplement) public use microdata and processes it into household-level
and person-level DataFrames suitable for HierarchicalSynthesizer.

Usage:
    python scripts/download_cps_asec.py [--year 2024] [--force]

Data source:
    https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd


# CPS ASEC CSV download URLs by year
CPS_ASEC_URLS = {
    2024: "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
}

# Column mappings for standardized variable names
# These map from CPS variable names to our schema names
PERSON_COLUMN_MAP = {
    # Core identifiers
    "PH_SEQ": "household_id",
    "P_SEQ": "person_seq",
    "PERIDNUM": "person_id",
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",  # 1=Male, 2=Female
    "A_HGA": "education",  # Educational attainment
    "A_MARITL": "marital_status",
    "A_ENRLW": "enrollment_status",  # School enrollment
    "PERRP": "relationship_to_head",
    # Income (person-level)
    "PTOTVAL": "total_income",
    "PEARNVAL": "earned_income",
    "WSAL_VAL": "wage_salary_income",
    "SEMP_VAL": "self_employment_income",
    "SS_VAL": "social_security_income",
    "SSI_VAL": "ssi_income",
    "UC_VAL": "unemployment_comp",
    "VET_VAL": "veterans_benefits",
    "DIV_VAL": "dividend_income",
    "INT_VAL": "interest_income",
    "RNT_VAL": "rental_income",
    "RETVAL": "retirement_income",
    # Employment
    "A_WKSTAT": "work_status",
    "A_CLSWKR": "class_of_worker",
    "A_MJIND": "industry",
    "A_MJOCC": "occupation",
    "WKSWORK": "weeks_worked",
    "HRSWK": "hours_per_week",
    "PEIOOCC": "detailed_occupation",
    # Health insurance
    "COV": "has_health_insurance",
    "COV_HI": "has_private_insurance",
    "HIMCAIDYN": "has_medicaid",
    "HIMCARYN": "has_medicare",
    # Weights
    "A_FNLWGT": "person_weight",
    "MARSUPWT": "supplement_weight",
}

HOUSEHOLD_COLUMN_MAP = {
    # Identifiers
    "H_SEQ": "household_id",
    # Location
    "GESTFIPS": "state_fips",
    "GTCBSA": "metro_area",
    "GTCO": "county",
    # Housing
    "H_TENURE": "tenure",  # 1=Owned, 2=Rented
    "H_NUMPER": "n_persons",
    "H_TYPE": "household_type",
    "HRNUMHOU": "num_units_in_structure",
    # Weights
    "HSUP_WGT": "hh_weight",
    "HWHHWGT": "household_weight",
}

FAMILY_COLUMN_MAP = {
    "FH_SEQ": "household_id",
    "FFPOS": "family_seq",
    "FTOT_R": "family_type",
    "FPERSONS": "family_n_persons",
    "FOWNU18": "family_n_children_under_18",
    "FOWNU6": "family_n_children_under_6",
}


def download_cps_asec(year: int, cache_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download CPS ASEC data for a given year.

    Args:
        year: Survey year (e.g., 2024)
        cache_dir: Directory to cache downloaded files (default: data/raw)

    Returns:
        Tuple of (person_df, household_df, family_df)
    """
    if year not in CPS_ASEC_URLS:
        raise ValueError(f"Year {year} not supported. Available years: {list(CPS_ASEC_URLS.keys())}")

    url = CPS_ASEC_URLS[year]

    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"cps_asec_{year}.zip"

    # Download if not cached
    if not cache_file.exists():
        print(f"Downloading CPS ASEC {year} from Census Bureau...")
        print(f"  URL: {url}")

        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (microplex research)"})
        response = urlopen(req, timeout=300)
        content = response.read()

        with open(cache_file, "wb") as f:
            f.write(content)

        print(f"  Downloaded {len(content) / 1e6:.1f} MB")
    else:
        print(f"Using cached file: {cache_file}")

    # Extract and read CSV files from ZIP
    print("Extracting CSV files...")

    with zipfile.ZipFile(cache_file, "r") as zf:
        file_list = zf.namelist()
        print(f"  Found {len(file_list)} files in archive")

        # Find the person, household, and family files
        person_file = None
        household_file = None
        family_file = None

        for fname in file_list:
            fname_lower = fname.lower()
            if ("pers" in fname_lower or fname_lower.startswith("pppub")) and fname_lower.endswith(".csv"):
                person_file = fname
            elif ("hhld" in fname_lower or fname_lower.startswith("hhpub")) and fname_lower.endswith(".csv"):
                household_file = fname
            elif ("fam" in fname_lower or fname_lower.startswith("ffpub")) and fname_lower.endswith(".csv"):
                family_file = fname

        if person_file is None:
            raise ValueError(f"Could not find person file in archive. Files: {file_list}")

        print(f"  Reading person file: {person_file}")
        with zf.open(person_file) as f:
            person_df = pd.read_csv(f, low_memory=False)

        if household_file:
            print(f"  Reading household file: {household_file}")
            with zf.open(household_file) as f:
                household_df = pd.read_csv(f, low_memory=False)
        else:
            household_df = pd.DataFrame()

        if family_file:
            print(f"  Reading family file: {family_file}")
            with zf.open(family_file) as f:
                family_df = pd.read_csv(f, low_memory=False)
        else:
            family_df = pd.DataFrame()

    print(f"  Person records: {len(person_df):,}")
    print(f"  Household records: {len(household_df):,}")
    print(f"  Family records: {len(family_df):,}")

    return person_df, household_df, family_df


def process_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw person-level CPS data.

    Args:
        df: Raw person DataFrame from CPS

    Returns:
        Processed DataFrame with standardized columns
    """
    # Select and rename columns
    available_cols = {k: v for k, v in PERSON_COLUMN_MAP.items() if k in df.columns}
    result = df[list(available_cols.keys())].rename(columns=available_cols)

    # Create unique person_id if not present
    if "person_id" not in result.columns or result["person_id"].isna().all():
        result["person_id"] = range(len(result))

    # Create combined income variable
    income_cols = ["wage_salary_income", "self_employment_income", "social_security_income",
                   "ssi_income", "unemployment_comp", "dividend_income", "interest_income",
                   "rental_income", "retirement_income"]
    available_income = [c for c in income_cols if c in result.columns]
    if available_income:
        result["income"] = result[available_income].fillna(0).sum(axis=1)
    elif "total_income" in result.columns:
        result["income"] = result["total_income"].fillna(0)
    else:
        result["income"] = 0

    # Create employment status (simplified)
    if "work_status" in result.columns:
        # A_WKSTAT: 1-3 = working, 4-5 = unemployed, 6-7 = not in labor force
        result["employment_status"] = np.where(
            result["work_status"].isin([1, 2, 3]), 1,  # Employed
            np.where(result["work_status"].isin([4, 5]), 2,  # Unemployed
                     0)  # Not in labor force
        )
    else:
        result["employment_status"] = 0

    # Map education to categorical levels
    if "education" in result.columns:
        # A_HGA ranges from 0-46 with various educational attainments
        # Simplify to 1-4 scale (Less than HS, HS, Some college, Bachelor+)
        result["education_level"] = pd.cut(
            result["education"].fillna(0),
            bins=[-1, 38, 39, 42, 100],
            labels=[1, 2, 3, 4]
        ).astype(float).fillna(1).astype(int)

    # Select final columns for synthesis
    output_cols = [
        "person_id", "household_id", "age", "sex", "income",
        "employment_status", "education_level", "relationship_to_head",
    ]

    # Add weight columns if available
    for w in ["person_weight", "supplement_weight"]:
        if w in result.columns:
            output_cols.append(w)

    available_output = [c for c in output_cols if c in result.columns]
    result = result[available_output].copy()

    # Rename education_level to education for schema compatibility
    if "education_level" in result.columns:
        result = result.rename(columns={"education_level": "education"})

    return result


def process_household_data(hh_df: pd.DataFrame, person_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process household-level CPS data.

    Creates household-level summary from both the household file and person data.

    Args:
        hh_df: Raw household DataFrame from CPS
        person_df: Processed person DataFrame

    Returns:
        Processed DataFrame with one row per household
    """
    # Get household ID column
    hh_id_col = "household_id" if "household_id" in person_df.columns else "PH_SEQ"

    # Aggregate person data to household level
    hh_agg = person_df.groupby("household_id").agg({
        "age": [
            ("n_persons", "count"),
            ("n_adults", lambda x: (x >= 18).sum()),
            ("n_children", lambda x: (x < 18).sum()),
        ],
    })
    hh_agg.columns = ["n_persons", "n_adults", "n_children"]
    hh_agg = hh_agg.reset_index()

    # Merge with household file if available
    if len(hh_df) > 0:
        available_hh_cols = {k: v for k, v in HOUSEHOLD_COLUMN_MAP.items() if k in hh_df.columns}
        hh_from_file = hh_df[list(available_hh_cols.keys())].rename(columns=available_hh_cols)

        # Merge on household_id
        if "household_id" in hh_from_file.columns:
            hh_agg = hh_agg.merge(hh_from_file, on="household_id", how="left")

    # Fill missing values for state_fips and tenure from person data
    # (sometimes these are on person records)
    if "state_fips" not in hh_agg.columns or hh_agg["state_fips"].isna().all():
        if "GESTFIPS" in person_df.columns if hasattr(person_df, "columns") else False:
            state_map = person_df.groupby("household_id")["GESTFIPS"].first()
            hh_agg["state_fips"] = hh_agg["household_id"].map(state_map)

    # Default tenure to 1 (owned) if not available
    if "tenure" not in hh_agg.columns:
        hh_agg["tenure"] = 1

    # Default state_fips to 0 if not available
    if "state_fips" not in hh_agg.columns:
        hh_agg["state_fips"] = 0

    # Get household weight - use max of person weights in household
    if "person_weight" in person_df.columns or "supplement_weight" in person_df.columns:
        weight_col = "supplement_weight" if "supplement_weight" in person_df.columns else "person_weight"
        weight_map = person_df.groupby("household_id")[weight_col].first()
        hh_agg["hh_weight"] = hh_agg["household_id"].map(weight_map).fillna(1)
    elif "hh_weight" not in hh_agg.columns:
        hh_agg["hh_weight"] = 1

    # Select final columns
    output_cols = ["household_id", "n_persons", "n_adults", "n_children",
                   "state_fips", "tenure", "hh_weight"]
    available_output = [c for c in output_cols if c in hh_agg.columns]

    return hh_agg[available_output]


def create_sample_data(n_households: int = 1000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic sample data for testing when CPS download fails.

    This generates realistic-looking household and person data based on
    CPS ASEC distributions.

    Args:
        n_households: Number of households to generate
        seed: Random seed

    Returns:
        Tuple of (household_df, person_df)
    """
    np.random.seed(seed)

    # Generate household composition
    # Distribution based on CPS ASEC averages
    n_persons = np.random.choice([1, 2, 3, 4, 5, 6], n_households,
                                  p=[0.28, 0.34, 0.16, 0.13, 0.06, 0.03])

    households = pd.DataFrame({
        "household_id": np.arange(n_households),
        "n_persons": n_persons,
        "state_fips": np.random.choice(range(1, 57), n_households),
        "tenure": np.random.choice([1, 2, 3], n_households, p=[0.65, 0.34, 0.01]),
        "hh_weight": np.random.lognormal(8, 0.5, n_households),
    })

    # Derive n_adults and n_children from n_persons
    households["n_children"] = np.minimum(
        np.random.binomial(households["n_persons"], 0.3),
        households["n_persons"] - 1  # At least 1 adult
    )
    households["n_adults"] = households["n_persons"] - households["n_children"]

    # Generate persons
    persons = []
    person_id = 0

    for _, hh in households.iterrows():
        hh_id = hh["household_id"]
        n_adults_hh = int(hh["n_adults"])
        n_children_hh = int(hh["n_children"])

        # Generate adults
        for i in range(n_adults_hh):
            age = np.random.randint(18, 85)
            education = np.random.choice([1, 2, 3, 4], p=[0.10, 0.28, 0.30, 0.32])

            # Income depends on age and education
            base_income = np.random.lognormal(10.5, 1.0)
            age_factor = 1 + 0.02 * min(age - 18, 30) - 0.01 * max(age - 55, 0)
            edu_factor = 1 + 0.3 * education
            income = base_income * age_factor * edu_factor

            # 15% have zero income
            if np.random.random() < 0.15:
                income = 0

            persons.append({
                "person_id": person_id,
                "household_id": hh_id,
                "age": age,
                "sex": np.random.choice([1, 2]),
                "income": income,
                "employment_status": np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                "education": education,
                "relationship_to_head": 1 if i == 0 else (2 if i == 1 else 3),
            })
            person_id += 1

        # Generate children
        for i in range(n_children_hh):
            persons.append({
                "person_id": person_id,
                "household_id": hh_id,
                "age": np.random.randint(0, 18),
                "sex": np.random.choice([1, 2]),
                "income": 0,
                "employment_status": 0,
                "education": 1,
                "relationship_to_head": 4,  # Child
            })
            person_id += 1

    persons_df = pd.DataFrame(persons)

    return households, persons_df


def main():
    parser = argparse.ArgumentParser(description="Download and process CPS ASEC microdata")
    parser.add_argument("--year", type=int, default=2024, help="Survey year (default: 2024)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--sample", action="store_true", help="Generate sample data instead of downloading")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for parquet files")
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        print("Generating sample CPS-like data...")
        households, persons = create_sample_data(n_households=10000, seed=42)
    else:
        try:
            # Download and process real CPS data
            raw_persons, raw_households, raw_families = download_cps_asec(args.year)

            print("\nProcessing person data...")
            persons = process_person_data(raw_persons)

            print("Processing household data...")
            households = process_household_data(raw_households, persons)

        except Exception as e:
            print(f"\nError downloading CPS data: {e}")
            print("Falling back to sample data generation...")
            households, persons = create_sample_data(n_households=10000, seed=42)

    # Save to parquet
    print(f"\nSaving processed data to {output_dir}...")

    hh_path = output_dir / "cps_asec_households.parquet"
    persons_path = output_dir / "cps_asec_persons.parquet"

    households.to_parquet(hh_path, index=False)
    persons.to_parquet(persons_path, index=False)

    print(f"  Households: {hh_path} ({len(households):,} records)")
    print(f"  Persons: {persons_path} ({len(persons):,} records)")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"\nHouseholds ({len(households):,} records):")
    print(households.describe())
    print(f"\nPersons ({len(persons):,} records):")
    print(persons.describe())

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
