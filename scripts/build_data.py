#!/usr/bin/env python3
"""Build the stacked multi-source dataset for benchmarks.

This script creates `data/stacked_comprehensive.parquet` from raw survey
data. The raw data must be obtained separately:

  1. CPS ASEC: Downloaded via PolicyEngine's enhanced CPS pipeline
     - Source: https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
     - Requires: policyengine-us-data package or raw Census downloads

  2. SIPP: Survey of Income and Program Participation
     - Source: https://www.census.gov/programs-surveys/sipp/data/datasets.html
     - Download: 2023 Public Use File (pu2023.csv)

  3. PSID: Panel Study of Income Dynamics
     - Source: https://psidonline.isr.umich.edu/
     - Requires: Registered access and data extract

Prerequisites:
    pip install microplex[benchmark]

Usage:
    python scripts/build_data.py --data-dir ./data

If you don't have access to the raw survey data, you can use the
pre-built dataset available at:
    https://huggingface.co/datasets/nikhil-woodruff/microplex-benchmark-data

    huggingface-cli download nikhil-woodruff/microplex-benchmark-data \\
        --local-dir ./data --include "stacked_comprehensive.parquet"
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build stacked multi-source dataset for benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Output directory for parquet files (default: data/)",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download pre-built dataset from HuggingFace instead of building from raw data",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "stacked_comprehensive.parquet"

    if args.download:
        _download_from_hf(output_path)
    else:
        _build_from_raw(data_dir, output_path)


def _download_from_hf(output_path: Path):
    """Download pre-built dataset from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub required: pip install huggingface_hub")
        sys.exit(1)

    print("Downloading stacked_comprehensive.parquet from HuggingFace...")
    path = hf_hub_download(
        repo_id="nikhil-woodruff/microplex-benchmark-data",
        filename="stacked_comprehensive.parquet",
        repo_type="dataset",
        local_dir=str(output_path.parent),
    )
    print(f"Downloaded to {path}")


def _build_from_raw(data_dir: Path, output_path: Path):
    """Build stacked dataset from raw survey files."""
    # Add project root to path for pipelines import
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from experiments.build_comprehensive_stack import main as build_main
    except ImportError as e:
        print(f"ERROR: Cannot import build pipeline: {e}")
        print("\nTo build from raw data, ensure:")
        print("  1. Raw survey files are in the expected locations")
        print("  2. pipelines/data_loaders.py paths are configured")
        print("\nAlternatively, use --download to get the pre-built dataset.")
        sys.exit(1)

    build_main()

    if output_path.exists():
        import pandas as pd
        df = pd.read_parquet(output_path)
        print(f"\nBuilt {output_path}: {len(df):,} rows, {len(df.columns)} columns")
    else:
        print(f"\nWARNING: Expected output {output_path} not found")


if __name__ == "__main__":
    main()
