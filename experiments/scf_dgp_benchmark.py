"""SCF Benchmark for PopulationDGP.

Uses SCF as ground truth to test multi-source population synthesis.
Creates artificial "surveys" with different columns, trains DGP, evaluates coverage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from microplex.dgp import Survey, run_multi_source_benchmark

CACHE_DIR = Path(__file__).parent / ".cache"


def download_scf(year: int = 2022) -> pd.DataFrame:
    """Download SCF summary extract."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"scf{year}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"https://www.federalreserve.gov/econres/files/scfp{year}s.zip"
    print(f"Downloading SCF {year}...")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        dta_files = [f for f in z.namelist() if f.endswith(".dta")]
        with z.open(dta_files[0]) as f:
            df = pd.read_stata(io.BytesIO(f.read()))

    df.to_parquet(cache_path)
    return df


def load_scf_surveys() -> tuple[list[Survey], list[str]]:
    """Load SCF and create artificial surveys with different columns.

    Returns:
        Tuple of (list of Survey objects, shared_cols)
    """
    raw = download_scf(2022)

    # Take first implicate
    if "y1" in raw.columns:
        raw["_implicate"] = raw["y1"] % 10
        df = raw[raw["_implicate"] == 1].copy()
    elif "Y1" in raw.columns:
        raw["_implicate"] = raw["Y1"] % 10
        df = raw[raw["_implicate"] == 1].copy()
    else:
        df = raw.copy()

    # Variable mapping (uppercase SCF names -> lowercase)
    var_map = {
        # Demographics (shared)
        "AGE": "age",
        "EDCL": "educ",
        "MARRIED": "married",
        "KIDS": "kids",
        # Income (Survey A focus)
        "INCOME": "income",
        "WAGEINC": "wageinc",
        "SSRETINC": "ssretinc",
        "INTDIVINC": "intdivinc",
        "BUSSEFARMINC": "bussefarminc",
        # Wealth summary (shared)
        "NETWORTH": "networth",
        "EQUITY": "equity",
        # Assets (Survey B focus)
        "ASSET": "asset",
        "FIN": "fin",
        "NFIN": "nfin",
        "LIQ": "liq",
        "CHECKING": "checking",
        "SAVING": "saving",
        "STOCKS": "stocks",
        "BOND": "bond",
        "RETQLIQ": "retqliq",
        "IRAKH": "irakh",
        # Debt (Survey B focus)
        "DEBT": "debt",
        "MRTHEL": "mrthel",
        "CCBAL": "ccbal",
        "INSTALL": "install",
        # Real assets
        "VEHIC": "vehic",
        "HOUSES": "houses",
    }

    raw_cols = {c.upper(): c for c in df.columns}
    selected = {}
    for scf_name, our_name in var_map.items():
        if scf_name in raw_cols:
            selected[raw_cols[scf_name]] = our_name

    df = df[list(selected.keys())].rename(columns=selected)

    # Clip negatives
    for col in df.columns:
        if col not in ["age", "educ", "married", "kids"]:
            df[col] = df[col].clip(lower=0)

    # Log transform and standardize
    monetary = [c for c in df.columns if c not in ["age", "educ", "married", "kids"]]
    for col in monetary:
        df[col] = np.log1p(df[col])

    for col in df.columns:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std

    df = df.reset_index(drop=True)

    # Define surveys with different column subsets
    shared_cols = ["age", "income", "networth", "equity"]

    survey_a_cols = shared_cols + [
        "educ", "married", "kids",
        "wageinc", "ssretinc", "intdivinc", "bussefarminc",
    ]

    survey_b_cols = shared_cols + [
        "asset", "fin", "nfin", "liq",
        "checking", "saving", "stocks", "bond",
        "retqliq", "irakh",
        "debt", "mrthel", "ccbal", "install",
        "vehic", "houses",
    ]

    # Filter to columns that exist
    survey_a_cols = [c for c in survey_a_cols if c in df.columns]
    survey_b_cols = [c for c in survey_b_cols if c in df.columns]

    # Sample different rows for each survey (simulating different populations)
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)

    n_a = n // 2
    n_b = n // 2

    survey_a = Survey("CPS_like", df.iloc[indices[:n_a]][survey_a_cols].reset_index(drop=True))
    survey_b = Survey("SCF_wealth", df.iloc[indices[n_a:n_a + n_b]][survey_b_cols].reset_index(drop=True))

    print(f"Survey A (CPS-like): {len(survey_a.data)} records, {len(survey_a_cols)} cols")
    print(f"Survey B (SCF wealth): {len(survey_b.data)} records, {len(survey_b_cols)} cols")
    print(f"Shared cols: {shared_cols}")

    return [survey_a, survey_b], shared_cols


def main():
    """Run SCF benchmark."""
    print("=" * 60)
    print("SCF Multi-Source Population Synthesis Benchmark")
    print("=" * 60)
    print()

    surveys, shared_cols = load_scf_surveys()

    print()
    print("Training PopulationDGP...")
    dgp, results = run_multi_source_benchmark(
        surveys=surveys,
        shared_cols=shared_cols,
        holdout_frac=0.2,
        seed=42,
    )

    print()
    print("Zero-inflated columns detected:")
    for col, is_zi in dgp.is_zero_inflated_.items():
        if is_zi:
            stats = dgp.col_stats_[col]
            print(f"  {col}: {stats['zero_frac']:.1%} zeros")

    # Summary
    print()
    print("Key insight: Coverage measures how well synthetic data spans real population")
    print("Higher coverage → calibration will need less extreme weights")


if __name__ == "__main__":
    main()
