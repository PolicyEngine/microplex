"""Microplex Tracking Dashboard.

Compares microplex outputs against reference datasets:
- PolicyEngine-US-Data (Enhanced CPS, PUF)
- Yale Budget Lab microdata
- PSL Tax-Data
- IRS SOI statistics
- Census aggregates
- SSA administrative data

Usage:
    python -m dashboard.tracking --output dashboard/report.md
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Data source paths
POLICYENGINE_DATA = Path("/Users/maxghenis/PolicyEngine/arch-data")
PE_US_DATA = Path("/Users/maxghenis/PolicyEngine/policyengine-us-data")


@dataclass
class ValidationTarget:
    """A target metric for validation."""
    name: str
    value: float
    source: str
    year: int
    unit: str = ""
    category: str = "aggregate"
    notes: str = ""


@dataclass
class ComparisonResult:
    """Result of comparing microplex to a target."""
    target: ValidationTarget
    microplex_value: float
    absolute_error: float
    relative_error: float
    within_tolerance: bool


@dataclass
class DataCoverage:
    """Coverage metrics for a variable."""
    variable: str
    microplex_present: bool
    microplex_nonzero_rate: float
    reference_present: bool
    reference_nonzero_rate: float
    correlation: Optional[float] = None


class TrackingDashboard:
    """Dashboard for tracking microplex quality against reference data."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize dashboard.

        Args:
            tolerance: Relative error tolerance for "within tolerance" flag
        """
        self.tolerance = tolerance
        self.targets: Dict[str, List[ValidationTarget]] = {}
        self.results: List[ComparisonResult] = []
        self.coverage: List[DataCoverage] = []

    def add_target(self, target: ValidationTarget) -> None:
        """Add a validation target."""
        source = target.source
        if source not in self.targets:
            self.targets[source] = []
        self.targets[source].append(target)

    def add_targets_from_dict(self, targets: Dict[str, Any], source: str, year: int) -> None:
        """Add multiple targets from a dictionary."""
        for name, value in targets.items():
            if isinstance(value, dict):
                # Nested categories
                for subname, subvalue in value.items():
                    self.add_target(ValidationTarget(
                        name=f"{name}_{subname}",
                        value=subvalue,
                        source=source,
                        year=year,
                    ))
            else:
                self.add_target(ValidationTarget(
                    name=name,
                    value=value,
                    source=source,
                    year=year,
                ))

    def load_irs_soi_targets(self, year: int = 2021) -> None:
        """Load IRS SOI aggregate targets."""
        # Key aggregates from IRS Statistics of Income
        # Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns
        irs_targets = {
            # Returns and population
            "total_returns": 150_439_000,
            "total_agi": 14_713_000_000_000,  # $14.7T
            "total_taxable_income": 11_793_000_000_000,
            "total_income_tax": 2_227_000_000_000,

            # Income components
            "wages_and_salaries": 9_431_000_000_000,
            "taxable_interest": 117_000_000_000,
            "ordinary_dividends": 398_000_000_000,
            "capital_gains": 1_648_000_000_000,
            "business_income": 397_000_000_000,
            "ira_distributions": 490_000_000_000,
            "pensions_annuities": 917_000_000_000,
            "social_security_benefits": 413_000_000_000,

            # Deductions
            "total_itemized_deductions": 1_276_000_000_000,
            "state_local_taxes_deduction": 228_000_000_000,
            "mortgage_interest_deduction": 184_000_000_000,
            "charitable_contributions": 276_000_000_000,

            # Credits
            "total_child_tax_credit": 109_000_000_000,
            "total_eitc": 57_000_000_000,
        }

        self.add_targets_from_dict(irs_targets, "IRS SOI", year)

    def load_census_targets(self, year: int = 2023) -> None:
        """Load Census population targets."""
        census_targets = {
            "total_population": 334_914_000,
            "population_under_18": 72_800_000,
            "population_18_64": 205_000_000,
            "population_65_plus": 57_100_000,

            # Households
            "total_households": 131_200_000,
            "family_households": 83_900_000,
            "nonfamily_households": 47_300_000,

            # By state (top 5)
            "population_california": 39_029_000,
            "population_texas": 30_030_000,
            "population_florida": 22_611_000,
            "population_new_york": 19_677_000,
            "population_pennsylvania": 12_973_000,
        }

        self.add_targets_from_dict(census_targets, "Census", year)

    def load_ssa_targets(self, year: int = 2023) -> None:
        """Load SSA administrative targets."""
        # From SSA Annual Statistical Supplement
        ssa_targets = {
            # OASDI Beneficiaries
            "oasdi_beneficiaries_total": 66_700_000,
            "oasdi_retired_workers": 50_500_000,
            "oasdi_disabled_workers": 7_600_000,
            "oasdi_survivors": 5_800_000,
            "oasdi_children": 2_800_000,

            # Benefits paid
            "oasdi_benefits_total": 1_352_000_000_000,
            "oasdi_avg_monthly_benefit": 1_698,

            # SSI
            "ssi_recipients_total": 7_400_000,
            "ssi_benefits_total": 60_000_000_000,

            # Covered employment
            "covered_workers": 181_000_000,
            "covered_earnings_total": 10_500_000_000_000,
        }

        self.add_targets_from_dict(ssa_targets, "SSA", year)

    def load_yale_budget_lab_targets(self) -> None:
        """Load Yale Budget Lab data targets.

        Yale Budget Lab provides:
        - Distributional analysis data
        - Tax policy microsimulation benchmarks
        - Income distribution statistics
        """
        # TODO: Fetch from Yale Budget Lab API or published data
        # https://budgetlab.yale.edu/
        pass

    def load_psl_taxdata_targets(self) -> None:
        """Load PSL Tax-Data targets.

        PSL Tax-Data (taxdata) is the data preparation for Tax-Calculator.
        GitHub: https://github.com/PSLmodels/taxdata
        """
        # TODO: Integrate with PSL taxdata
        # Key metrics: PUF extrapolation factors, growth rates
        pass

    def compare_to_microplex(
        self,
        microplex: pd.DataFrame,
        weight_col: str = "weight",
    ) -> List[ComparisonResult]:
        """Compare microplex to all loaded targets."""
        results = []

        for source, targets in self.targets.items():
            for target in targets:
                value = self._compute_microplex_value(microplex, target, weight_col)
                if value is not None:
                    abs_err = abs(value - target.value)
                    rel_err = abs_err / target.value if target.value != 0 else 0
                    result = ComparisonResult(
                        target=target,
                        microplex_value=value,
                        absolute_error=abs_err,
                        relative_error=rel_err,
                        within_tolerance=rel_err <= self.tolerance,
                    )
                    results.append(result)

        self.results = results
        return results

    def _compute_microplex_value(
        self,
        data: pd.DataFrame,
        target: ValidationTarget,
        weight_col: str,
    ) -> Optional[float]:
        """Compute microplex aggregate for a target."""
        name = target.name.lower()
        weights = data[weight_col].values if weight_col in data.columns else np.ones(len(data))

        # Map target names to data columns
        column_mapping = {
            "total_population": lambda d, w: w.sum(),
            "total_returns": lambda d, w: w.sum(),  # Assumes tax unit level
            "total_agi": lambda d, w: (w * d.get("agi", d.get("adjusted_gross_income", 0))).sum(),
            "wages_and_salaries": lambda d, w: (w * d.get("wage_salary_income", d.get("wages", 0))).sum(),
            "social_security_benefits": lambda d, w: (w * d.get("social_security_income", 0)).sum(),
            "total_eitc": lambda d, w: (w * d.get("eitc", d.get("earned_income_tax_credit", 0))).sum(),
            "total_child_tax_credit": lambda d, w: (w * d.get("ctc", d.get("child_tax_credit", 0))).sum(),
        }

        # Try exact match first
        if name in column_mapping:
            try:
                return column_mapping[name](data, weights)
            except Exception:
                return None

        # Try column name match
        for col in data.columns:
            if name.replace("_", "") in col.replace("_", "").lower():
                try:
                    return (weights * data[col].fillna(0)).sum()
                except Exception:
                    pass

        return None

    def compute_coverage(
        self,
        microplex: pd.DataFrame,
        reference: pd.DataFrame,
        variables: Optional[List[str]] = None,
    ) -> List[DataCoverage]:
        """Compute coverage metrics comparing microplex to reference."""
        if variables is None:
            # Use intersection of columns
            variables = list(set(microplex.columns) & set(reference.columns))

        coverage = []
        for var in variables:
            mp_present = var in microplex.columns
            ref_present = var in reference.columns

            mp_nz = 0.0
            ref_nz = 0.0
            corr = None

            if mp_present:
                mp_nz = (microplex[var].fillna(0) != 0).mean()
            if ref_present:
                ref_nz = (reference[var].fillna(0) != 0).mean()

            if mp_present and ref_present:
                # Sample correlation if both have the variable
                try:
                    # Take random sample for efficiency
                    n = min(len(microplex), len(reference), 10000)
                    mp_sample = microplex[var].sample(n, random_state=42)
                    ref_sample = reference[var].sample(n, random_state=42)
                    corr = mp_sample.corr(ref_sample)
                except Exception:
                    pass

            coverage.append(DataCoverage(
                variable=var,
                microplex_present=mp_present,
                microplex_nonzero_rate=mp_nz,
                reference_present=ref_present,
                reference_nonzero_rate=ref_nz,
                correlation=corr,
            ))

        self.coverage = coverage
        return coverage

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate markdown report."""
        lines = [
            "# Microplex Tracking Dashboard",
            f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        ]

        # Summary
        if self.results:
            n_total = len(self.results)
            n_within = sum(1 for r in self.results if r.within_tolerance)
            lines.append("## Summary\n")
            lines.append(f"- **Total targets**: {n_total}")
            lines.append(f"- **Within tolerance ({self.tolerance:.0%})**: {n_within} ({n_within/n_total:.0%})")
            lines.append(f"- **Outside tolerance**: {n_total - n_within}\n")
        else:
            # Show targets without comparison
            lines.append("## Validation Targets\n")
            lines.append("_No microplex data loaded - showing targets only_\n")

            for source, targets in self.targets.items():
                lines.append(f"### {source}\n")
                lines.append("| Target | Value |")
                lines.append("|--------|-------|")
                for t in targets:
                    lines.append(f"| {t.name} | {t.value:,.0f} |")
                lines.append("")

        # Results by source
        if self.results:
            lines.append("## Comparison Results\n")

            by_source: Dict[str, List[ComparisonResult]] = {}
            for r in self.results:
                src = r.target.source
                if src not in by_source:
                    by_source[src] = []
                by_source[src].append(r)

            for source, results in by_source.items():
                lines.append(f"### {source}\n")
                lines.append("| Target | Reference | Microplex | Rel Error | Status |")
                lines.append("|--------|-----------|-----------|-----------|--------|")

                for r in sorted(results, key=lambda x: -x.relative_error):
                    status = "✅" if r.within_tolerance else "❌"
                    ref_fmt = f"{r.target.value:,.0f}"
                    mp_fmt = f"{r.microplex_value:,.0f}"
                    err_fmt = f"{r.relative_error:.1%}"
                    lines.append(f"| {r.target.name} | {ref_fmt} | {mp_fmt} | {err_fmt} | {status} |")

                lines.append("")

        # Coverage
        if self.coverage:
            lines.append("## Data Coverage\n")
            lines.append("| Variable | Microplex NZ% | Reference NZ% | Correlation |")
            lines.append("|----------|---------------|---------------|-------------|")

            for c in sorted(self.coverage, key=lambda x: x.variable):
                mp_nz = f"{c.microplex_nonzero_rate:.1%}" if c.microplex_present else "N/A"
                ref_nz = f"{c.reference_nonzero_rate:.1%}" if c.reference_present else "N/A"
                corr = f"{c.correlation:.3f}" if c.correlation is not None else "N/A"
                lines.append(f"| {c.variable} | {mp_nz} | {ref_nz} | {corr} |")

            lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)

        return report

    def to_json(self) -> Dict[str, Any]:
        """Export dashboard data as JSON."""
        return {
            "generated": datetime.now().isoformat(),
            "tolerance": self.tolerance,
            "results": [
                {
                    "name": r.target.name,
                    "source": r.target.source,
                    "year": r.target.year,
                    "reference_value": r.target.value,
                    "microplex_value": r.microplex_value,
                    "relative_error": r.relative_error,
                    "within_tolerance": r.within_tolerance,
                }
                for r in self.results
            ],
            "coverage": [
                {
                    "variable": c.variable,
                    "microplex_nonzero_rate": c.microplex_nonzero_rate,
                    "reference_nonzero_rate": c.reference_nonzero_rate,
                    "correlation": c.correlation,
                }
                for c in self.coverage
            ],
        }


def load_policyengine_enhanced_cps(year: int = 2024) -> pd.DataFrame:
    """Load PolicyEngine Enhanced CPS for comparison."""
    try:
        # Try loading from policyengine-us-data
        from policyengine_us_data import EnhancedCPS
        ecps = EnhancedCPS(year)
        return ecps.load()
    except ImportError:
        # Fallback to local parquet
        path = POLICYENGINE_DATA / f"micro/us/cps_{year}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        raise FileNotFoundError(f"Could not load Enhanced CPS for {year}")


def run_dashboard(
    microplex_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> TrackingDashboard:
    """Run the full tracking dashboard."""
    print("=" * 70)
    print("MICROPLEX TRACKING DASHBOARD")
    print("=" * 70)

    dashboard = TrackingDashboard(tolerance=0.10)  # 10% tolerance

    # Load reference targets
    print("\n1. Loading reference targets...")
    dashboard.load_irs_soi_targets(year=2021)
    dashboard.load_census_targets(year=2023)
    dashboard.load_ssa_targets(year=2023)
    print(f"   Loaded {sum(len(t) for t in dashboard.targets.values())} targets from {len(dashboard.targets)} sources")

    # Load microplex (or generate sample)
    print("\n2. Loading microplex data...")
    microplex = None
    try:
        if microplex_path and microplex_path.exists():
            microplex = pd.read_parquet(microplex_path)
            print(f"   Loaded {len(microplex):,} records from {microplex_path}")
        else:
            # Load CPS as proxy
            cps_path = POLICYENGINE_DATA / "micro/us/cps_2024.parquet"
            if cps_path.exists():
                microplex = pd.read_parquet(cps_path)
                print(f"   Using CPS as proxy: {len(microplex):,} records")
            else:
                print("   No microplex data available - generating targets report only")
    except ImportError as e:
        print(f"   Warning: {e}")
        print("   Install pyarrow: pip install pyarrow")
        print("   Generating targets report only...")
    except Exception as e:
        print(f"   Error loading data: {e}")

    if microplex is None:
        # Generate report with just targets
        print("\n3. Generating targets report...")
        if output_path is None:
            output_path = Path("dashboard/tracking_report.md")
        dashboard.generate_report(output_path)
        print(f"   Report saved to {output_path}")
        return dashboard

    # Run comparison
    print("\n3. Comparing to targets...")
    results = dashboard.compare_to_microplex(microplex, weight_col="weight")
    n_within = sum(1 for r in results if r.within_tolerance)
    print(f"   {n_within}/{len(results)} targets within 10% tolerance")

    # Load reference for coverage
    print("\n4. Computing coverage metrics...")
    try:
        reference = load_policyengine_enhanced_cps(2024)
        coverage = dashboard.compute_coverage(microplex, reference)
        print(f"   Computed coverage for {len(coverage)} variables")
    except Exception as e:
        print(f"   Could not load reference: {e}")

    # Generate report
    print("\n5. Generating report...")
    if output_path is None:
        output_path = Path("dashboard/tracking_report.md")
    report = dashboard.generate_report(output_path)
    print(f"   Report saved to {output_path}")

    # Save JSON
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(dashboard.to_json(), f, indent=2)
    print(f"   JSON saved to {json_path}")

    print("\n" + "=" * 70)
    print("DASHBOARD COMPLETE")
    print("=" * 70)

    return dashboard


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Microplex Tracking Dashboard")
    parser.add_argument("--microplex", type=Path, help="Path to microplex parquet file")
    parser.add_argument("--output", type=Path, default=Path("dashboard/tracking_report.md"),
                        help="Output path for markdown report")
    args = parser.parse_args()

    run_dashboard(microplex_path=args.microplex, output_path=args.output)
