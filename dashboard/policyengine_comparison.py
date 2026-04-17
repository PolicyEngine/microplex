"""Compare microplex to PolicyEngine-US-Data.

Detailed comparison against:
- Enhanced CPS (calibrated CPS with IRS imputation)
- PUF (IRS Public Use File)
- SIPP panels
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PE_US_DATA = Path("/Users/maxghenis/PolicyEngine/policyengine-us-data")
COSILICO_DATA = Path("/Users/maxghenis/CosilicoAI/cosilico-data-sources")


# Key variables to compare
COMPARISON_VARIABLES = {
    # Demographics
    "demographics": [
        "age",
        "is_male",
        "marital_status",
        "state_fips",
    ],
    # Income
    "income": [
        "wage_salary_income",
        "self_employment_income",
        "interest_income",
        "dividend_income",
        "capital_gains",
        "rental_income",
        "social_security_income",
        "unemployment_compensation",
    ],
    # Tax
    "tax": [
        "adjusted_gross_income",
        "taxable_income",
        "income_tax_before_credits",
        "earned_income_tax_credit",
        "child_tax_credit",
        "income_tax",
    ],
    # Benefits
    "benefits": [
        "snap",
        "ssi",
        "tanf",
        "wic",
        "medicaid",
    ],
}


def load_enhanced_cps(year: int = 2024) -> pd.DataFrame:
    """Load PolicyEngine Enhanced CPS."""
    try:
        # Try via policyengine_us_data package
        import sys
        sys.path.insert(0, str(PE_US_DATA))
        from policyengine_us_data import EnhancedCPS
        return EnhancedCPS.load(year)
    except Exception:
        # Fallback to local parquet
        path = COSILICO_DATA / f"micro/us/cps_{year}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"Could not load Enhanced CPS for {year}")


def load_puf(year: int = 2024) -> pd.DataFrame:
    """Load IRS PUF."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="policyengine/irs-soi-puf",
            filename=f"puf_{year}.h5",
            repo_type="model",
        )
        return pd.read_hdf(path)
    except Exception as e:
        print(f"Could not load PUF: {e}")
        return pd.DataFrame()


def compare_distributions(
    microplex: pd.DataFrame,
    reference: pd.DataFrame,
    variable: str,
    weight_col: str = "weight",
    n_quantiles: int = 10,
) -> dict:
    """Compare distribution of a variable between microplex and reference."""
    if variable not in microplex.columns or variable not in reference.columns:
        return {"error": f"Variable {variable} not in both datasets"}

    mp_vals = microplex[variable].fillna(0)
    ref_vals = reference[variable].fillna(0)

    mp_weights = microplex[weight_col] if weight_col in microplex.columns else pd.Series(1, index=microplex.index)
    ref_weights = reference[weight_col] if weight_col in reference.columns else pd.Series(1, index=reference.index)

    # Weighted statistics
    mp_mean = np.average(mp_vals, weights=mp_weights)
    ref_mean = np.average(ref_vals, weights=ref_weights)

    mp_std = np.sqrt(np.average((mp_vals - mp_mean) ** 2, weights=mp_weights))
    ref_std = np.sqrt(np.average((ref_vals - ref_mean) ** 2, weights=ref_weights))

    # Zero rates
    mp_zero = (mp_vals == 0).mean()
    ref_zero = (ref_vals == 0).mean()

    # Quantiles (unweighted for simplicity)
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    mp_quantiles = mp_vals.quantile(quantiles).tolist()
    ref_quantiles = ref_vals.quantile(quantiles).tolist()

    return {
        "variable": variable,
        "microplex_mean": mp_mean,
        "reference_mean": ref_mean,
        "mean_ratio": mp_mean / ref_mean if ref_mean != 0 else np.nan,
        "microplex_std": mp_std,
        "reference_std": ref_std,
        "std_ratio": mp_std / ref_std if ref_std != 0 else np.nan,
        "microplex_zero_rate": mp_zero,
        "reference_zero_rate": ref_zero,
        "microplex_quantiles": mp_quantiles,
        "reference_quantiles": ref_quantiles,
    }


def compare_all_variables(
    microplex: pd.DataFrame,
    reference: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Compare all common variables."""
    results = []

    for category, variables in COMPARISON_VARIABLES.items():
        for var in variables:
            comparison = compare_distributions(microplex, reference, var, weight_col)
            if "error" not in comparison:
                comparison["category"] = category
                results.append(comparison)

    return pd.DataFrame(results)


def generate_comparison_report(
    microplex: pd.DataFrame,
    reference: pd.DataFrame,
    reference_name: str = "Enhanced CPS",
    output_path: Path | None = None,
) -> str:
    """Generate markdown comparison report."""
    comparisons = compare_all_variables(microplex, reference)

    lines = [
        f"# Microplex vs {reference_name} Comparison",
        "",
        "## Summary Statistics",
        "",
        "| Category | Variable | Reference Mean | Microplex Mean | Ratio | Zero Rate Diff |",
        "|----------|----------|----------------|----------------|-------|----------------|",
    ]

    for _, row in comparisons.iterrows():
        ratio = row["mean_ratio"]
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
        zero_diff = row["microplex_zero_rate"] - row["reference_zero_rate"]
        zero_str = f"{zero_diff:+.1%}"
        lines.append(
            f"| {row['category']} | {row['variable']} | "
            f"{row['reference_mean']:,.0f} | {row['microplex_mean']:,.0f} | "
            f"{ratio_str} | {zero_str} |"
        )

    lines.extend([
        "",
        "## Distribution Comparison",
        "",
    ])

    # Add distribution details for key variables
    key_vars = ["wage_salary_income", "adjusted_gross_income", "income_tax"]
    for var in key_vars:
        var_data = comparisons[comparisons["variable"] == var]
        if len(var_data) > 0:
            row = var_data.iloc[0]
            lines.extend([
                f"### {var}",
                "",
                "| Quantile | Reference | Microplex |",
                "|----------|-----------|-----------|",
            ])
            quantiles = np.linspace(0, 1, 11)
            for i, q in enumerate(quantiles):
                ref_val = row["reference_quantiles"][i]
                mp_val = row["microplex_quantiles"][i]
                lines.append(f"| {q:.0%} | {ref_val:,.0f} | {mp_val:,.0f} |")
            lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report


def run_policyengine_comparison(
    microplex_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run full comparison against PolicyEngine data."""
    print("=" * 70)
    print("POLICYENGINE COMPARISON")
    print("=" * 70)

    # Load reference
    print("\n1. Loading Enhanced CPS...")
    try:
        reference = load_enhanced_cps(2024)
        print(f"   Loaded {len(reference):,} records")
    except Exception as e:
        print(f"   Error: {e}")
        return pd.DataFrame()

    # Load microplex
    print("\n2. Loading microplex...")
    if microplex_path and microplex_path.exists():
        microplex = pd.read_parquet(microplex_path)
    else:
        # Use CPS as proxy
        cps_path = COSILICO_DATA / "micro/us/cps_2024.parquet"
        if cps_path.exists():
            microplex = pd.read_parquet(cps_path)
            print(f"   Using CPS as proxy: {len(microplex):,} records")
        else:
            print("   No microplex data available")
            return pd.DataFrame()

    # Run comparison
    print("\n3. Comparing distributions...")
    comparisons = compare_all_variables(microplex, reference)
    print(f"   Compared {len(comparisons)} variables")

    # Generate report
    print("\n4. Generating report...")
    if output_path is None:
        output_path = Path("dashboard/pe_comparison.md")
    generate_comparison_report(microplex, reference, output_path=output_path)
    print(f"   Report saved to {output_path}")

    return comparisons


if __name__ == "__main__":
    run_policyengine_comparison()
