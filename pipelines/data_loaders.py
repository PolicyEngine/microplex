"""Data loaders for US survey data.

Loads and harmonizes CPS, PUF, SIPP, and PSID for multi-survey fusion.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Try to import HuggingFace hub for downloading data
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Data paths
COSILICO_DATA = Path("/Users/maxghenis/CosilicoAI/cosilico-data-sources")
STORAGE_FOLDER = COSILICO_DATA / "storage"
PSID_DATA_DIR = Path("/Users/maxghenis/CosilicoAI/psid/psid_data")


# PSID variable mappings by year (codes change each survey wave)
# Maps label patterns to year-specific column codes
PSID_VAR_BY_YEAR = {
    2021: {
        "age": "ER78017",           # AGE OF REFERENCE PERSON
        "sex": "ER78018",           # SEX OF REFERENCE PERSON (1=male, 2=female)
        "state_fips": "ER78004",    # CURRENT STATE
        "marital_status": "ER78025", # REFERENCE PERSON MARITAL STATUS
        "labor_income": "ER81642",   # LABOR INCOME OF REF PERSON-2020
        "interest_income": "ER81647", # REF PERSON INTEREST INCOME-2020
        "dividend_income": "ER81645", # REF PERSON DIVIDENDS-2020
        "rental_income": "ER81643",   # REF PERSON RENT INCOME-2020
        "social_security": "ER81769", # REF PERSON SOCIAL SECURITY INCOME-2020
        "total_family_income": "ER81775",  # TOTAL FAMILY INCOME-2020
        "taxable_income": "ER81679",  # REF PERSN AND SPOUSE TAXABLE INCOME-2020
        "food_stamps": "ER78848",     # VALUE OF FOOD STAMPS LAST YEAR
        "interview_number": "ER78002", # Interview number (family ID)
        "weight": "ER78244",          # Family weight (if exists)
    },
    2023: {
        "age": "ER82018",           # AGE OF REFERENCE PERSON
        "sex": "ER82019",           # SEX OF REFERENCE PERSON
        "state_fips": "ER82004",    # CURRENT STATE
        "marital_status": "ER82026", # REFERENCE PERSON MARITAL STATUS
        "interview_number": "ER82002", # Interview number
        # Note: Income variables for 2023 would be ER85xxx or similar
        # Need to verify from codebook when available
    },
}


# Variable harmonization mapping
# Maps survey-specific variable names to common names
# Note: CPS uses PolicyEngine's H5 format with specific variable names
# PSID uses dynamic column names set during load_psid()
VARIABLE_MAPPING = {
    # Demographics
    "age": {
        "cps": "age",
        "puf": "age",
        "sipp": "TAGE",
        "psid": "age",  # Harmonized during load
    },
    "is_male": {
        "cps": "sex",  # Already converted from is_female during load
        "puf": "is_male",
        "sipp": "ESEX",  # 1=male, 2=female
        "psid": "is_male",  # Harmonized during load
    },
    "state_fips": {
        "cps": "state_fips",  # Household-level in CPS
        "puf": None,  # PUF doesn't have state
        "sipp": None,
        "psid": "state_fips",  # PSID has state
    },
    "marital_status": {
        "cps": "marital_status",
        "puf": "filing_status",
        "sipp": "EMS",
        "psid": "marital_status",
    },
    # Income
    "wage_income": {
        "cps": "employment_income",  # CPS uses employment_income
        "puf": "employment_income",
        "sipp": "TPTOTINC",  # Total person income
        "psid": "labor_income",  # PSID labor income
    },
    "self_employment_income": {
        "cps": "self_employment_income",
        "puf": "self_employment_income",
        "sipp": None,
        "psid": None,  # PSID has business income but needs separate handling
    },
    "interest_income": {
        "cps": "interest_income",  # From taxable_interest_income
        "puf": "taxable_interest_income",
        "sipp": None,
        "psid": "interest_income",
    },
    "dividend_income": {
        "cps": "dividend_income",  # From qualified_dividend_income
        "puf": "qualified_dividend_income",
        "sipp": None,
        "psid": "dividend_income",
    },
    "social_security_income": {
        "cps": "social_security_income",
        "puf": "social_security",
        "sipp": None,
        "psid": "social_security",
    },
    "unemployment_compensation": {
        "cps": "unemployment_compensation",
        "puf": "taxable_unemployment_compensation",
        "sipp": None,
        "psid": None,
    },
    # Investment/Asset income
    "rental_income": {
        "cps": "rental_income",
        "puf": "rental_income",
        "sipp": None,
        "psid": "rental_income",
    },
    "capital_gains": {
        "cps": None,
        "puf": "long_term_capital_gains",
        "sipp": None,
        "psid": None,
    },
    "farm_income": {
        "cps": "farm_income",
        "puf": None,
        "sipp": None,
        "psid": None,
    },
    # SIPP-specific
    "tip_income": {
        "cps": None,
        "puf": None,
        "sipp": "tip_income",  # Derived column
        "psid": None,
    },
    # PSID-specific
    "total_family_income": {
        "cps": None,
        "puf": None,
        "sipp": None,
        "psid": "total_family_income",
    },
    "food_stamps": {
        "cps": None,
        "puf": None,
        "sipp": None,
        "psid": "food_stamps",
    },
}


def load_cps(
    path: Path | None = None,
    sample_frac: float = 1.0,
    seed: int = 42,
    use_policyengine: bool = True,
) -> pd.DataFrame:
    """Load CPS data.

    Args:
        path: Path to CPS parquet file (if not using PolicyEngine)
        sample_frac: Fraction of data to sample
        seed: Random seed
        use_policyengine: If True, load from policyengine_us_data package

    Returns:
        DataFrame with CPS data
    """
    if use_policyengine:
        try:
            from policyengine_us_data import CPS_2024

            print("Loading CPS via policyengine_us_data...")
            cps = CPS_2024()

            # Person-level variables
            person_vars = {
                "age": "age",
                "is_female": "is_female",  # Will convert to is_male
                "employment_income": "employment_income",
                "self_employment_income": "self_employment_income",
                "taxable_interest_income": "interest_income",
                "qualified_dividend_income": "dividend_income",
                "rental_income": "rental_income",
                "farm_income": "farm_income",
                "person_household_id": "person_household_id",
            }

            # Household-level variables (need to be joined via person_household_id)
            household_vars = {
                "state_fips": "state_fips",
                "household_weight": "household_weight",
                "household_id": "household_id",
            }

            data = {}
            for cps_name, common_name in person_vars.items():
                try:
                    data[common_name] = np.array(cps.load(cps_name))
                except Exception:
                    pass

            # Build DataFrame
            if "age" not in data:
                print("  Warning: Could not load CPS age variable")
                return pd.DataFrame()

            n_persons = len(data["age"])
            df = pd.DataFrame(index=range(n_persons))

            for col, values in data.items():
                df[col] = values

            # Load household-level data and join to persons
            try:
                hh_ids = np.array(cps.load("household_id"))  # Household-level
                person_hh_ids = data.get("person_household_id")  # Person-level

                if person_hh_ids is not None:
                    # Create household lookup
                    hh_data = {"household_id": hh_ids}
                    for cps_name, common_name in household_vars.items():
                        try:
                            hh_data[common_name] = np.array(cps.load(cps_name))
                        except Exception:
                            pass

                    hh_df = pd.DataFrame(hh_data)

                    # Join to persons using person_household_id
                    df = df.merge(
                        hh_df,
                        left_on="person_household_id",
                        right_on="household_id",
                        how="left"
                    )
            except Exception as e:
                print(f"  Warning: Could not join household data: {e}")

            # Convert is_female to is_male
            if "is_female" in df.columns:
                df["sex"] = (~df["is_female"].astype(bool)).astype(float)
                df.drop("is_female", axis=1, inplace=True)

            # Weight
            if "household_weight" in df.columns:
                df["weight"] = df["household_weight"].fillna(1.0)
            else:
                df["weight"] = 1.0

        except ImportError:
            print("  policyengine_us_data not available, trying parquet file...")
            use_policyengine = False

    if not use_policyengine:
        if path is None:
            path = COSILICO_DATA / "micro/us/cps_2024.parquet"

        if not path.exists():
            print(f"  Warning: CPS file not found: {path}")
            return pd.DataFrame()

        print(f"Loading CPS from {path}...")
        df = pd.read_parquet(path)

        # Weight column
        if "march_supplement_weight" in df.columns:
            df["weight"] = df["march_supplement_weight"]
        elif "weight" not in df.columns:
            df["weight"] = 1.0

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Add survey identifier
    df["_survey"] = "cps"

    print(f"  Loaded {len(df):,} CPS records")
    return df


def load_puf(
    year: int = 2024,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load PUF data from HuggingFace or local cache."""
    if not HF_AVAILABLE:
        print("  Warning: huggingface_hub not installed, skipping PUF")
        return pd.DataFrame()

    filename = f"puf_{year}.h5"
    local_path = STORAGE_FOLDER / filename

    if not local_path.exists():
        print(f"Downloading PUF {year} from HuggingFace...")
        STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="policyengine/irs-soi-puf",
                filename=filename,
                repo_type="model",
                local_dir=STORAGE_FOLDER,
            )
        except Exception as e:
            print(f"  Warning: Could not download PUF: {e}")
            return pd.DataFrame()

    print(f"Loading PUF from {local_path}...")
    try:
        df = pd.read_hdf(local_path)
    except Exception as e:
        print(f"  Warning: Could not load PUF: {e}")
        return pd.DataFrame()

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Add survey identifier
    df["_survey"] = "puf"

    # Weight column
    if "household_weight" in df.columns:
        df["weight"] = df["household_weight"]
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    print(f"  Loaded {len(df):,} PUF records")
    return df


def load_sipp(
    year: int = 2023,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load SIPP data from HuggingFace or local cache."""
    if not HF_AVAILABLE:
        print("  Warning: huggingface_hub not installed, skipping SIPP")
        return pd.DataFrame()

    filename = f"pu{year}_slim.csv"
    local_path = STORAGE_FOLDER / filename

    if not local_path.exists():
        print(f"Downloading SIPP {year} from HuggingFace...")
        STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="PolicyEngine/policyengine-us-data",
                filename=filename,
                repo_type="model",
                local_dir=STORAGE_FOLDER,
            )
        except Exception as e:
            print(f"  Warning: Could not download SIPP: {e}")
            return pd.DataFrame()

    print(f"Loading SIPP from {local_path}...")
    try:
        df = pd.read_csv(local_path)
    except Exception as e:
        print(f"  Warning: Could not load SIPP: {e}")
        return pd.DataFrame()

    # Derive tip income if columns exist
    tip_cols = [c for c in df.columns if "TXAMT" in c]
    if tip_cols:
        df["tip_income"] = df[tip_cols].fillna(0).sum(axis=1) * 12

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Add survey identifier
    df["_survey"] = "sipp"

    # Weight column
    if "WPFINWGT" in df.columns:
        df["weight"] = df["WPFINWGT"]
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    print(f"  Loaded {len(df):,} SIPP records")
    return df


def parse_psid_do_file(do_file: Path) -> dict[str, tuple[int, int]]:
    """Parse PSID Stata .do file to extract column specifications.

    Returns dict mapping variable name -> (start, end) positions.
    """
    content = do_file.read_text()

    # Find the infix section
    infix_match = re.search(r'infix\s+(.*?)using', content, re.DOTALL | re.IGNORECASE)
    if not infix_match:
        raise ValueError(f"Could not find infix specification in {do_file}")

    infix_section = infix_match.group(1)

    # Parse column specs: [long] VARNAME start - end
    pattern = r'(?:long\s+)?(\w+)\s+(\d+)\s*-\s*(\d+)'

    columns = {}
    for match in re.finditer(pattern, infix_section):
        var_name = match.group(1).upper()
        start = int(match.group(2))
        end = int(match.group(3))
        columns[var_name] = (start, end)

    return columns


def load_psid(
    year: int = 2021,
    data_dir: Path | None = None,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load PSID family file and harmonize to common schema.

    PSID uses different variable codes each survey year, so we map them
    to a common schema during load.

    Args:
        year: Survey year (2021 or 2023)
        data_dir: Directory containing PSID files (defaults to PSID_DATA_DIR)
        sample_frac: Fraction of data to sample
        seed: Random seed for sampling

    Returns:
        DataFrame with harmonized variable names
    """
    if data_dir is None:
        data_dir = PSID_DATA_DIR

    data_dir = Path(data_dir)

    # Find data file
    patterns = [
        f"FAM{year}ER.txt",
        f"fam{year}er.txt",
        f"FAM{year}ER.dta",
        f"fam{year}er.dta",
    ]

    data_file = None
    for pattern in patterns:
        path = data_dir / pattern
        if path.exists():
            data_file = path
            break

    if data_file is None:
        print(f"  Warning: PSID {year} data not found in {data_dir}")
        return pd.DataFrame()

    print(f"Loading PSID {year} from {data_file}...")

    # Get year-specific variable mapping
    var_map = PSID_VAR_BY_YEAR.get(year, {})
    if not var_map:
        print(f"  Warning: No variable mapping for PSID {year}")
        return pd.DataFrame()

    # Load based on file type
    if data_file.suffix.lower() == ".dta":
        df = pd.read_stata(data_file)
        df.columns = df.columns.str.upper()
    elif data_file.suffix.lower() == ".txt":
        # Need .do file for fixed-width parsing
        do_file = data_file.with_suffix(".do")
        if not do_file.exists():
            do_file = data_dir / f"FAM{year}ER.do"
        if not do_file.exists():
            print(f"  Warning: .do file not found for {data_file}")
            return pd.DataFrame()

        # Parse column specs from .do file
        col_specs = parse_psid_do_file(do_file)

        # Only load the columns we need
        needed_cols = [v.upper() for v in var_map.values() if v]
        available = [c for c in needed_cols if c in col_specs]

        if not available:
            print(f"  Warning: No mapped columns found in PSID {year}")
            return pd.DataFrame()

        # Build colspecs for pandas
        colspecs = [(col_specs[c][0] - 1, col_specs[c][1]) for c in available]
        names = available

        df = pd.read_fwf(
            data_file,
            colspecs=colspecs,
            names=names,
            dtype=str,
        )

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"  Warning: Unsupported file format: {data_file.suffix}")
        return pd.DataFrame()

    # Harmonize column names to common schema
    result = pd.DataFrame(index=df.index)

    for common_name, psid_code in var_map.items():
        if psid_code and psid_code.upper() in df.columns:
            if common_name == "sex":
                # Convert 1=male, 2=female to is_male boolean
                result["is_male"] = (df[psid_code.upper()] == 1).astype(float)
            else:
                result[common_name] = df[psid_code.upper()]

    # Add metadata
    result["_survey"] = "psid"
    result["year"] = year

    # Weight handling
    if "weight" in result.columns:
        pass  # Already have it
    elif "ER78244" in df.columns:  # 2021 family weight
        result["weight"] = df["ER78244"]
    else:
        result["weight"] = 1.0

    if sample_frac < 1.0:
        result = result.sample(frac=sample_frac, random_state=seed)

    print(f"  Loaded {len(result):,} PSID records with {len(result.columns)} columns")

    return result


def harmonize_variable(
    df: pd.DataFrame,
    common_name: str,
    survey: str,
) -> pd.Series:
    """Extract and harmonize a variable from survey data."""
    mapping = VARIABLE_MAPPING.get(common_name, {})
    survey_name = mapping.get(survey)

    if survey_name is None or survey_name not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    values = df[survey_name].copy()

    # Apply survey-specific transformations
    if common_name == "is_male":
        if survey == "cps":
            # CPS: 1=male, 2=female
            values = (values == 1).astype(float)
        elif survey == "sipp":
            # SIPP: 1=male, 2=female
            values = (values == 1).astype(float)
        elif survey == "psid":
            # PSID already harmonized to is_male during load
            pass

    return values


def stack_surveys(
    surveys: dict[str, pd.DataFrame],
    variables: list[str],
) -> pd.DataFrame:
    """Stack multiple surveys into single DataFrame with NaN for missing vars.

    This is the key function for multi-survey fusion. Variables not present
    in a survey are set to NaN, and microplex's masked loss will only
    compute loss on observed (non-NaN) values.
    """
    print("\nStacking surveys for multi-survey fusion...")

    stacked_rows = []
    for survey_name, df in surveys.items():
        if len(df) == 0:
            continue

        print(f"  Processing {survey_name}: {len(df):,} records")

        # Create harmonized DataFrame
        harmonized = pd.DataFrame(index=df.index)

        for var in variables:
            harmonized[var] = harmonize_variable(df, var, survey_name)

        # Add weight and survey identifier
        harmonized["weight"] = df["weight"]
        harmonized["_survey"] = survey_name

        stacked_rows.append(harmonized)

    result = pd.concat(stacked_rows, ignore_index=True)
    print(f"\nStacked total: {len(result):,} records")

    # Report missing data pattern
    print("\nMissing data pattern:")
    for var in variables:
        n_observed = result[var].notna().sum()
        pct = 100 * n_observed / len(result)
        print(f"  {var}: {n_observed:,} observed ({pct:.1f}%)")

    return result


def load_all_surveys(
    cps_path: Path | None = None,
    psid_dir: Path | None = None,
    include_psid: bool = True,
    include_puf: bool = True,
    sample_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load and stack all available surveys.

    Args:
        cps_path: Path to CPS parquet file
        psid_dir: Directory containing PSID files
        include_psid: Whether to include PSID
        include_puf: Whether to include PUF
        sample_frac: Fraction of data to sample
        seed: Random seed

    Returns:
        stacked: Combined DataFrame with harmonized variables
        surveys: Dict of individual survey DataFrames
    """
    surveys = {}

    # Load CPS
    cps = load_cps(cps_path, sample_frac=sample_frac, seed=seed)
    if len(cps) > 0:
        surveys["cps"] = cps

    # Load PUF
    if include_puf:
        puf = load_puf(sample_frac=sample_frac, seed=seed)
        if len(puf) > 0:
            surveys["puf"] = puf

    # Load SIPP
    sipp = load_sipp(sample_frac=sample_frac, seed=seed)
    if len(sipp) > 0:
        surveys["sipp"] = sipp

    # Load PSID
    if include_psid:
        psid = load_psid(year=2021, data_dir=psid_dir, sample_frac=sample_frac, seed=seed)
        if len(psid) > 0:
            surveys["psid"] = psid

    # Define common variables to harmonize
    common_vars = [
        "age",
        "is_male",
        "state_fips",
        "marital_status",
        "wage_income",
        "self_employment_income",
        "interest_income",
        "dividend_income",
        "social_security_income",
        "unemployment_compensation",
        "rental_income",
        "capital_gains",
        "farm_income",
        "tip_income",
        "total_family_income",
        "food_stamps",
    ]

    # Stack surveys
    stacked = stack_surveys(surveys, common_vars)

    return stacked, surveys


if __name__ == "__main__":
    # Test loading
    stacked, surveys = load_all_surveys(sample_frac=0.01)
    print(f"\nLoaded {len(surveys)} surveys")
    print(f"Total stacked records: {len(stacked):,}")
