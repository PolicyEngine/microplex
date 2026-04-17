"""Build comprehensive stacked dataset with SIPP lags for transition modeling.

Creates a multi-survey dataset (CPS + SIPP + PSID) with:
- Demographic variables (age, sex, race, marital status, education)
- Income variables (wages, self-employment, investment, transfers)
- SIPP panel lags for transition modeling (job loss, job gain, income changes)
- Proper type handling for all columns
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.data_loaders import STORAGE_FOLDER, load_psid

start_time = time.time()


def load_sipp_with_lags(sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Load SIPP with panel lags for transition modeling."""
    print("Loading SIPP with panel structure...")

    local_path = STORAGE_FOLDER / "pu2023_slim.csv"
    if not local_path.exists():
        print(f"  SIPP file not found: {local_path}")
        return pd.DataFrame()

    df = pd.read_csv(local_path)
    print(f"  Raw SIPP: {len(df):,} rows, {len(df.columns)} cols")

    # Identify key columns
    # Demographics
    demo_cols = {
        'SSUID': 'household_id',
        'PNUM': 'person_num',
        'MONTHCODE': 'month',
        'SWAVE': 'wave',
        'TAGE': 'age',
        'ESEX': 'sex',  # 1=male, 2=female
        'ERACE': 'race',
        'EORIGIN': 'hispanic',
        'EEDUC': 'education',
        'RRPCP': 'relationship',
        'EMS': 'marital_status',
    }

    # Income columns (multiple jobs)
    income_cols = {
        'TPTOTINC': 'total_income',
        'TJB1_MSUM': 'job1_income',
        'TJB2_MSUM': 'job2_income',
        'TJB3_MSUM': 'job3_income',
        'TPMISCSRCE1': 'tip_income',
    }

    # Job characteristics
    job_cols = {
        'TJB1_OCC': 'job1_occ',
        'TJB1_IND': 'job1_ind',
        'TJB2_OCC': 'job2_occ',
        'TJB2_IND': 'job2_ind',
    }

    # Weight
    weight_cols = {
        'WPFINWGT': 'weight',
    }

    # Collect available columns
    all_mappings = {**demo_cols, **income_cols, **job_cols, **weight_cols}
    available = {k: v for k, v in all_mappings.items() if k in df.columns}

    # Select and rename
    result = df[[c for c in available.keys()]].copy()
    result.columns = [available[c] for c in result.columns]

    # Create person_id (string for consistency)
    if 'household_id' in result.columns and 'person_num' in result.columns:
        result['person_id'] = result['household_id'].astype(str) + '_' + result['person_num'].astype(str)
    else:
        result['person_id'] = [f'sipp_{i}' for i in range(len(result))]

    # Convert sex to is_male
    if 'sex' in result.columns:
        result['is_male'] = (result['sex'] == 1).astype(int)
        result.drop('sex', axis=1, inplace=True)

    # Sort by person_id and wave/month for lag computation
    sort_cols = ['person_id']
    if 'wave' in result.columns:
        sort_cols.append('wave')
    if 'month' in result.columns:
        sort_cols.append('month')
    result = result.sort_values(sort_cols)

    # Compute lags within person
    print("  Computing panel lags...")
    lag_cols = ['job1_income', 'total_income', 'job1_occ']
    for col in lag_cols:
        if col in result.columns:
            result[f'{col}_lag1'] = result.groupby('person_id')[col].shift(1)

    # Compute transitions
    if 'job1_income' in result.columns and 'job1_income_lag1' in result.columns:
        result['job1_income_change'] = result['job1_income'] - result['job1_income_lag1']
        # Job loss: had income last period, none this period
        result['job_loss'] = ((result['job1_income_lag1'] > 0) & (result['job1_income'] == 0)).astype(int)
        # Job gain: no income last period, has income this period
        result['job_gain'] = ((result['job1_income_lag1'] == 0) & (result['job1_income'] > 0)).astype(int)

    result['_survey'] = 'sipp'

    if sample_frac < 1.0:
        result = result.sample(frac=sample_frac, random_state=seed)

    print(f"  SIPP with lags: {len(result):,} rows, {len(result.columns)} cols")
    return result


def load_cps_comprehensive(sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Load CPS with comprehensive variables."""
    print("Loading CPS...")

    try:
        from policyengine_us_data import CPS_2024
        cps = CPS_2024()

        # Variables to load
        person_vars = [
            'age', 'is_female',
            'employment_income', 'self_employment_income',
            'taxable_interest_income', 'qualified_dividend_income',
            'rental_income', 'farm_income',
            'social_security', 'ssi',
            'person_household_id', 'person_id',
        ]

        household_vars = [
            'state_fips', 'household_weight', 'household_id',
        ]

        data = {}
        for var in person_vars:
            try:
                data[var] = np.array(cps.load(var))
            except Exception:
                pass

        df = pd.DataFrame(data)

        # Join household data
        try:
            hh_data = {var: np.array(cps.load(var)) for var in household_vars if var != 'household_id'}
            hh_data['household_id'] = np.array(cps.load('household_id'))
            hh_df = pd.DataFrame(hh_data)

            if 'person_household_id' in df.columns:
                df = df.merge(hh_df, left_on='person_household_id', right_on='household_id', how='left')
        except Exception as e:
            print(f"  Could not join household data: {e}")

        # Harmonize
        if 'is_female' in df.columns:
            df['is_male'] = (~df['is_female'].astype(bool)).astype(int)
            df.drop('is_female', axis=1, inplace=True)

        # Rename income columns
        rename_map = {
            'employment_income': 'wage_income',
            'taxable_interest_income': 'interest_income',
            'qualified_dividend_income': 'dividend_income',
            'household_weight': 'weight',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure person_id is string
        if 'person_id' in df.columns:
            df['person_id'] = df['person_id'].astype(str)
        else:
            df['person_id'] = [f'cps_{i}' for i in range(len(df))]

        if 'weight' not in df.columns:
            df['weight'] = 1.0

        df['_survey'] = 'cps'

        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=seed)

        print(f"  CPS: {len(df):,} rows, {len(df.columns)} cols")
        return df

    except Exception as e:
        print(f"  CPS load error: {e}")
        return pd.DataFrame()


def load_psid_comprehensive(sample_frac: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Load PSID with comprehensive variables."""
    print("Loading PSID...")

    df = load_psid(year=2021, sample_frac=sample_frac, seed=seed)

    if len(df) == 0:
        return df

    # Rename columns
    rename_map = {
        'labor_income': 'wage_income',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure person_id is string
    if 'interview_number' in df.columns:
        df['person_id'] = df['interview_number'].astype(str)
    else:
        df['person_id'] = [f'psid_{i}' for i in range(len(df))]

    print(f"  PSID: {len(df):,} rows, {len(df.columns)} cols")
    return df


def stack_comprehensive(surveys: dict) -> pd.DataFrame:
    """Stack surveys with proper type handling."""
    print("\nStacking surveys...")

    # Get union of all columns
    all_cols = set()
    for df in surveys.values():
        all_cols.update(df.columns)

    # Remove internal columns
    internal = {'person_household_id', 'household_id', 'person_num', 'index'}
    all_cols = all_cols - internal

    # Stack with NaN for missing columns
    stacked_parts = []
    for name, df in surveys.items():
        part = df.copy()
        for col in all_cols:
            if col not in part.columns:
                part[col] = np.nan
        stacked_parts.append(part[list(all_cols)])

    stacked = pd.concat(stacked_parts, ignore_index=True)

    # Ensure types are consistent
    print("  Ensuring consistent types...")

    # String columns
    str_cols = ['person_id', '_survey']
    for col in str_cols:
        if col in stacked.columns:
            stacked[col] = stacked[col].astype(str)

    # Integer columns (where NaN becomes -1 or stays as float)
    int_cols = ['age', 'is_male', 'race', 'hispanic', 'education', 'relationship',
                'marital_status', 'wave', 'month', 'state_fips',
                'job1_occ', 'job1_ind', 'job2_occ', 'job2_ind',
                'job_loss', 'job_gain']
    for col in int_cols:
        if col in stacked.columns:
            # Keep as float to preserve NaN
            stacked[col] = pd.to_numeric(stacked[col], errors='coerce')

    # Float columns
    float_cols = ['weight', 'wage_income', 'self_employment_income', 'interest_income',
                  'dividend_income', 'rental_income', 'farm_income', 'social_security',
                  'ssi', 'total_income', 'job1_income', 'job2_income', 'job3_income',
                  'tip_income', 'total_family_income', 'food_stamps', 'taxable_income',
                  'job1_income_lag1', 'total_income_lag1', 'job1_occ_lag1',
                  'job1_income_change']
    for col in float_cols:
        if col in stacked.columns:
            stacked[col] = pd.to_numeric(stacked[col], errors='coerce')

    return stacked


def main():
    print("="*60)
    print("Building Comprehensive Multi-Survey Stack")
    print("="*60)

    # Load each survey
    sipp = load_sipp_with_lags(sample_frac=1.0)
    cps = load_cps_comprehensive(sample_frac=1.0)
    psid = load_psid_comprehensive(sample_frac=1.0)

    surveys = {}
    if len(sipp) > 0:
        surveys['sipp'] = sipp
    if len(cps) > 0:
        surveys['cps'] = cps
    if len(psid) > 0:
        surveys['psid'] = psid

    if not surveys:
        print("ERROR: No surveys loaded!")
        return

    # Stack
    stacked = stack_comprehensive(surveys)

    print(f"\n{'='*60}")
    print(f"Final stacked dataset: {len(stacked):,} rows x {len(stacked.columns)} cols")
    print(f"{'='*60}")

    # Report columns by survey
    print("\nColumns by survey:")
    for survey_name in ['cps', 'sipp', 'psid']:
        mask = stacked['_survey'] == survey_name
        if mask.sum() == 0:
            continue
        subset = stacked[mask]
        observed_cols = [c for c in stacked.columns if c != '_survey' and subset[c].notna().any()]
        print(f"\n  {survey_name.upper()} ({mask.sum():,} rows):")
        print(f"    {observed_cols}")

    # Save
    output_path = Path(__file__).parent.parent / "data" / "stacked_comprehensive.parquet"
    output_path.parent.mkdir(exist_ok=True)

    print(f"\nSaving to {output_path}...")
    stacked.to_parquet(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1e6:.1f} MB")

    return stacked


if __name__ == "__main__":
    stacked = main()
