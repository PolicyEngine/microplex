"""Multi-source fusion experiment with SequenceSynthesizer.

Fuses SIPP, PSID, and CPS into a unified model that:
1. Loads each source
2. Converts to person-period format
3. Harmonizes variables
4. Trains SequenceSynthesizer on stacked data
5. Generates synthetic person-period microdata
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np
import pandas as pd
import torch

# Import SequenceSynthesizer
from microplex.models.sequence_synthesizer import SequenceSynthesizer
from pipelines.data_loaders import load_cps, load_sipp


def prepare_sipp_person_period(
    sipp_df: pd.DataFrame,
    min_periods: int = 6,
) -> pd.DataFrame:
    """Convert SIPP to person-period format.

    Args:
        sipp_df: Raw SIPP data
        min_periods: Minimum periods required per person

    Returns:
        DataFrame with columns:
            - person_id: Unique person identifier
            - period: YYYY-MM format
            - source: 'sipp'
            - (harmonized variables)
    """
    print("Converting SIPP to person-period format...")
    df = sipp_df.copy()

    # Create person_id from household + person number
    df['person_id'] = 'sipp_' + df['SSUID'].astype(str) + '_' + df['PNUM'].astype(str)

    # Create period from wave and month (YYYY-MM format)
    # SIPP 2022/2023 waves are ~2022-01 onward
    base_year = 2022
    df['period_month'] = (df['SWAVE'] - 1) * 12 + df['MONTHCODE']
    df['year'] = base_year + (df['period_month'] - 1) // 12
    df['month'] = ((df['period_month'] - 1) % 12) + 1
    df['period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)

    # Harmonize variables
    result = pd.DataFrame({
        'person_id': df['person_id'],
        'period': df['period'],
        'source': 'sipp',
        'age': df['TAGE'].fillna(0).astype(int),
        'is_male': (df['ESEX'] == 1).astype(int) if 'ESEX' in df.columns else np.nan,
        'total_income': df['TPTOTINC'].fillna(0).clip(lower=0),
        'is_married': (df['EMS'] == 1).astype(int) if 'EMS' in df.columns else np.nan,
    })

    # Add job incomes if available
    for i in range(1, 4):
        col = f'TJB{i}_MSUM'
        if col in df.columns:
            result[f'job{i}_income'] = df[col].fillna(0).clip(lower=0)
        else:
            result[f'job{i}_income'] = 0.0

    # Filter to people with enough periods
    periods_per_person = result.groupby('person_id')['period'].nunique()
    valid_persons = periods_per_person[periods_per_person >= min_periods].index
    result = result[result['person_id'].isin(valid_persons)]

    print(f"  SIPP: {result['person_id'].nunique():,} persons, {len(result):,} person-periods")
    return result


def prepare_cps_person_period(
    cps_df: pd.DataFrame,
    year: int = 2024,
) -> pd.DataFrame:
    """Convert CPS to person-period format.

    CPS is cross-sectional, so each person has exactly one period.

    Args:
        cps_df: Raw CPS data
        year: Survey year

    Returns:
        DataFrame in person-period format
    """
    print("Converting CPS to person-period format...")
    df = cps_df.copy()

    # Create person_id from available identifiers
    if 'household_id' in df.columns and 'person_number' in df.columns:
        df['person_id'] = 'cps_' + df['household_id'].astype(str) + '_' + df['person_number'].astype(str)
    else:
        # Create synthetic ID
        df['person_id'] = 'cps_' + df.index.astype(str)

    # CPS is annual, so period is just the year
    df['period'] = str(year)

    # Harmonize variables
    result = pd.DataFrame({
        'person_id': df['person_id'],
        'period': df['period'],
        'source': 'cps',
        'age': df['age'].fillna(0).astype(int) if 'age' in df.columns else np.nan,
        'is_male': (df['sex'] == 1).astype(int) if 'sex' in df.columns else np.nan,
        'total_income': df['total_income'].fillna(0).clip(lower=0) if 'total_income' in df.columns else 0.0,
        'is_married': (df['marital_status'] == 1).astype(int) if 'marital_status' in df.columns else np.nan,
    })

    # CPS-specific income components (SIPP doesn't have these)
    for col in ['dividend_income', 'interest_income', 'rental_income']:
        if col in df.columns:
            result[col] = df[col].fillna(0).clip(lower=0)
        else:
            result[col] = np.nan  # Mark as missing

    print(f"  CPS: {result['person_id'].nunique():,} persons, {len(result):,} person-periods")
    return result


def prepare_psid_person_period(
    data_dir: str | None = None,
    years: list | None = None,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """Load and convert PSID to person-period format.

    PSID is biennial panel, so periods are years (YYYY).

    Returns:
        DataFrame in person-period format
    """
    print("Loading PSID...")

    try:
        import psid

        # Use default data directory
        if data_dir is None:
            data_dir = './psid_data'

        # Check if PSID data exists
        from pathlib import Path
        psid_path = Path(data_dir)
        if not psid_path.exists() or not list(psid_path.glob("*.txt")):
            raise FileNotFoundError(f"PSID data not found in {data_dir}. "
                "PSID requires registration at https://psidonline.isr.umich.edu")

        # Define variables to extract
        family_vars = psid.FamilyVars({
            "total_family_income": psid.COMMON_VARIABLES.get("total_family_income", {}),
        })

        # Build panel
        if years is None:
            years = [2019, 2021]

        panel = psid.build_panel(
            data_dir=data_dir,
            years=years,
            family_vars=family_vars,
        )

        # Get DataFrame from panel
        df = panel.data if hasattr(panel, 'data') else pd.DataFrame()
        print(f"  Loaded {len(df):,} person-years from PSID")

        # Sample if requested
        if sample_frac < 1.0 and len(df) > 0:
            persons = df['person_id'].unique()
            sample_persons = np.random.choice(persons, size=int(len(persons) * sample_frac), replace=False)
            df = df[df['person_id'].isin(sample_persons)]
            print(f"  Sampled to {df['person_id'].nunique():,} persons")

    except Exception as e:
        print(f"  Warning: Could not load PSID: {e}")
        print("  Creating mock PSID data for testing...")

        # Create mock PSID data
        np.random.seed(42)
        n_persons = 500
        n_years = 10
        records = []
        for pid in range(n_persons):
            age_start = np.random.randint(25, 55)
            is_male = np.random.choice([0, 1])
            is_married = np.random.choice([0, 1])
            income = np.random.lognormal(10.5, 0.5)

            for y in range(n_years):
                year = 2010 + y * 2  # Biennial
                # Simple dynamics
                if not is_married and np.random.random() < 0.05:
                    is_married = 1
                elif is_married and np.random.random() < 0.02:
                    is_married = 0
                income *= (1 + np.random.normal(0.02, 0.05))

                records.append({
                    'person_id': f'psid_{pid}',
                    'period': str(year),
                    'source': 'psid',
                    'age': age_start + y * 2,
                    'is_male': is_male,
                    'total_income': max(0, income),
                    'is_married': is_married,
                })

        df = pd.DataFrame(records)
        print(f"  Mock PSID: {df['person_id'].nunique():,} persons, {len(df):,} person-periods")
        return df

    # Convert real PSID data
    df['person_id'] = 'psid_' + df['person_id'].astype(str)
    df['period'] = df['year'].astype(str)
    df['source'] = 'psid'

    # Harmonize variables (check what columns exist)
    result = pd.DataFrame({
        'person_id': df['person_id'],
        'period': df['period'],
        'source': 'psid',
    })

    # Map available columns
    col_mapping = {
        'age': 'age',
        'sex': 'is_male',  # Will need transformation
        'total_family_income': 'total_income',
        'marital_status': 'is_married',
    }

    for src_col, dst_col in col_mapping.items():
        if src_col in df.columns:
            if dst_col == 'is_male':
                result[dst_col] = (df[src_col] == 1).astype(int)  # 1=male in PSID
            elif dst_col == 'is_married':
                result[dst_col] = (df[src_col] == 1).astype(int)  # 1=married in PSID
            elif dst_col == 'total_income':
                result[dst_col] = df[src_col].fillna(0).clip(lower=0)
            else:
                result[dst_col] = df[src_col].fillna(0).astype(int)
        else:
            result[dst_col] = np.nan

    print(f"  PSID: {result['person_id'].nunique():,} persons, {len(result):,} person-periods")
    return result


def stack_sources(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack all sources into unified person-period format.

    Args:
        sources: Dict mapping source name to person-period DataFrame

    Returns:
        Stacked DataFrame with all sources
    """
    print("\nStacking sources...")

    # Get union of all columns
    all_cols = set()
    for df in sources.values():
        all_cols.update(df.columns)

    # Ensure each DataFrame has all columns
    dfs = []
    for name, df in sources.items():
        for col in all_cols:
            if col not in df.columns:
                df[col] = np.nan
        dfs.append(df)

    stacked = pd.concat(dfs, ignore_index=True)

    print(f"  Stacked: {stacked['person_id'].nunique():,} persons, {len(stacked):,} person-periods")
    print(f"  Sources: {stacked['source'].value_counts().to_dict()}")

    return stacked


def run_fusion_experiment(
    sample_frac: float = 0.1,
    epochs: int = 50,
    save_model: bool = True,
):
    """Run the full fusion experiment.

    Args:
        sample_frac: Fraction of data to use (for faster testing)
        epochs: Training epochs
        save_model: Whether to save the trained model
    """
    print("=" * 60)
    print("MULTI-SOURCE FUSION EXPERIMENT")
    print("=" * 60)
    print()

    # Load sources
    sources = {}

    # 1. SIPP (panel, monthly)
    print("Loading SIPP...")
    try:
        sipp_raw = load_sipp(sample_frac=sample_frac)
        if len(sipp_raw) > 0:
            sources['sipp'] = prepare_sipp_person_period(sipp_raw)
    except Exception as e:
        print(f"  Warning: Could not load SIPP: {e}")

    # 2. CPS (cross-sectional, annual)
    print("\nLoading CPS...")
    try:
        cps_raw = load_cps(sample_frac=sample_frac)
        if len(cps_raw) > 0:
            sources['cps'] = prepare_cps_person_period(cps_raw)
    except Exception as e:
        print(f"  Warning: Could not load CPS: {e}")

    # 3. PSID (panel, biennial)
    print("\nLoading PSID...")
    sources['psid'] = prepare_psid_person_period()

    if not sources:
        print("ERROR: No data sources loaded!")
        return None

    # Stack all sources
    stacked = stack_sources(sources)

    # Define variable sets
    continuous_vars = ['total_income']
    binary_vars = ['is_married']

    # Add job incomes if present (from SIPP)
    if 'job1_income' in stacked.columns and stacked['job1_income'].notna().any():
        continuous_vars.extend(['job1_income', 'job2_income', 'job3_income'])

    # Add CPS-specific income if present
    for col in ['dividend_income', 'interest_income']:
        if col in stacked.columns and stacked[col].notna().any():
            continuous_vars.append(col)

    print(f"\nContinuous vars: {continuous_vars}")
    print(f"Binary vars: {binary_vars}")

    # Train SequenceSynthesizer
    print("\n" + "=" * 60)
    print("TRAINING SEQUENCE SYNTHESIZER")
    print("=" * 60)

    model = SequenceSynthesizer(
        continuous_vars=continuous_vars,
        binary_vars=binary_vars,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    model.fit(
        stacked,
        person_id_col='person_id',
        period_col='period',
        epochs=epochs,
        batch_size=32,
        verbose=True,
    )

    # Save model
    if save_model:
        model_path = Path(__file__).parent / "sequence_fusion_model.pt"
        torch.save({
            'model_state': model._transformer.state_dict(),
            'config': {
                'continuous_vars': model.continuous_vars,
                'binary_vars': model.binary_vars,
                'd_model': model.d_model,
                'n_heads': model.n_heads,
                'n_layers': model.n_layers,
            },
            'var_means': model._var_means,
            'var_stds': model._var_stds,
        }, model_path)
        print(f"\nModel saved to {model_path}")

    # Generate synthetic data
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 60)

    # Get initial states from each source (last period for panel, only period for cross-sectional)
    initial_states = []
    for source_name, source_df in sources.items():
        # Get unique persons
        persons = source_df.groupby('person_id').last().reset_index()
        initial_states.append(persons.head(100))  # 100 from each source

    initial = pd.concat(initial_states, ignore_index=True)
    print(f"Initial states: {len(initial)} from {len(sources)} sources")

    # Generate trajectories per person
    print("Generating 10-period trajectories...")
    all_trajectories = []
    for person_id in initial['person_id'].unique():
        person_initial = initial[initial['person_id'] == person_id]
        traj = model.generate_trajectory(
            initial_state=person_initial,
            n_periods=10,
            seed=42,
        )
        all_trajectories.append(traj)
    trajectories = pd.concat(all_trajectories, ignore_index=True)

    print(f"Generated: {trajectories['person_id'].nunique()} persons, {len(trajectories)} person-periods")
    print("\nSample output (first 5 rows):")
    print(trajectories[['person_id', 'period', 'source', 'total_income', 'is_married']].head())

    # Evaluate coverage
    print("\n" + "=" * 60)
    print("COVERAGE EVALUATION")
    print("=" * 60)

    for var in continuous_vars[:3]:  # First 3 continuous vars
        if var not in trajectories.columns:
            continue
        synthetic_mean = trajectories[var].mean()
        synthetic_std = trajectories[var].std()

        for source_name, source_df in sources.items():
            if var in source_df.columns:
                real_mean = source_df[var].mean()
                real_std = source_df[var].std()
                print(f"  {var} ({source_name}): real={real_mean:.0f}±{real_std:.0f}, synth={synthetic_mean:.0f}±{synthetic_std:.0f}")

    # Save results
    results_path = Path(__file__).parent / "sequence_fusion_results.csv"
    trajectories.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    return trajectories


if __name__ == "__main__":
    # Run with 10% sample for faster testing
    trajectories = run_fusion_experiment(
        sample_frac=0.1,
        epochs=20,
        save_model=True,
    )
