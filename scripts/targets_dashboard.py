"""
Targets Comparison Dashboard

Compares calibration targets across data sources:
- Cosilico (our targets)
- PolicyEngine (policyengine_us_data)
- Yale Tax Simulator
- PSL Tax-Calculator
- IRS SOI (authoritative source)

Usage:
    python scripts/targets_dashboard.py --output targets_comparison.html
"""

import argparse
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class TargetSource:
    """A source of calibration targets."""
    name: str
    url: str
    description: str
    targets: Dict[str, Dict] = field(default_factory=dict)
    coverage: Dict[str, bool] = field(default_factory=dict)


# IRS SOI 2021 National Targets (authoritative)
IRS_SOI_2021 = {
    "returns_by_agi": {
        "no_agi": 13_992_100,
        "under_1": 1_686_440,
        "1_to_5k": 5_183_390,
        "5k_to_10k": 7_929_860,
        "10k_to_15k": 9_883_050,
        "15k_to_20k": 9_113_990,
        "20k_to_25k": 8_186_640,
        "25k_to_30k": 7_407_890,
        "30k_to_40k": 13_194_450,
        "40k_to_50k": 10_930_780,
        "50k_to_75k": 19_494_660,
        "75k_to_100k": 15_137_070,
        "100k_to_200k": 22_849_380,
        "200k_to_500k": 7_167_290,
        "500k_to_1m": 1_106_040,
        "1m_plus": 664_340,
    },
    "agi_by_bracket": {
        "under_1": -94_000_000_000,
        "1_to_5k": 15_000_000_000,
        "5k_to_10k": 59_000_000_000,
        "10k_to_15k": 123_000_000_000,
        "15k_to_20k": 160_000_000_000,
        "20k_to_25k": 184_000_000_000,
        "25k_to_30k": 204_000_000_000,
        "30k_to_40k": 461_000_000_000,
        "40k_to_50k": 492_000_000_000,
        "50k_to_75k": 1_210_000_000_000,
        "75k_to_100k": 1_316_000_000_000,
        "100k_to_200k": 3_187_000_000_000,
        "200k_to_500k": 2_161_000_000_000,
        "500k_to_1m": 762_000_000_000,
        "1m_plus": 4_466_000_000_000,
    },
    "total_returns": 153_774_320,
    "total_agi": 14_706_000_000_000,
    "eitc_claims": 31_000_000,
    "eitc_amount": 64_000_000_000,
    "ctc_claims": 48_000_000,
    "ctc_amount": 122_000_000_000,
}


def load_cosilico_targets(data_source_path: Path) -> TargetSource:
    """Load Cosilico targets from cosilico-data-sources."""
    source = TargetSource(
        name="Cosilico",
        url="https://github.com/CosilicoAI/cosilico-data-sources",
        description="Cosilico's calibration targets from IRS SOI and Census",
    )

    targets_dir = data_source_path / "data" / "targets"

    # Load state income distribution
    income_path = targets_dir / "state_income_distribution.parquet"
    if income_path.exists():
        df = pd.read_parquet(income_path)
        df_2021 = df[df["year"] == 2021] if "year" in df.columns else df

        # Aggregate to national
        national_returns = df_2021.groupby("agi_bracket")["target_returns"].sum()
        national_agi = df_2021.groupby("agi_bracket")["target_agi"].sum()

        source.targets["returns_by_agi"] = national_returns.to_dict()
        source.targets["agi_by_bracket"] = national_agi.to_dict()
        source.coverage["state_income_distribution"] = True
        source.coverage["national_agi_brackets"] = True

    # Load tax credits
    credits_path = targets_dir / "state_tax_credits.parquet"
    if credits_path.exists():
        df = pd.read_parquet(credits_path)
        df_2021 = df[df["year"] == 2021] if "year" in df.columns else df

        source.targets["eitc_claims"] = df_2021["eitc_claims"].sum()
        source.targets["eitc_amount"] = df_2021["eitc_amount"].sum()
        source.targets["ctc_claims"] = df_2021["ctc_claims"].sum()
        source.targets["ctc_amount"] = df_2021["ctc_amount"].sum()
        source.coverage["tax_credits"] = True

    # Load demographics
    demo_path = targets_dir / "state_demographics.parquet"
    if demo_path.exists():
        df = pd.read_parquet(demo_path)
        source.coverage["state_demographics"] = True
        source.coverage["population_by_age"] = True
        source.coverage["household_structure"] = True

    return source


def load_policyengine_targets() -> TargetSource:
    """Load PolicyEngine targets from policyengine_us_data."""
    source = TargetSource(
        name="PolicyEngine",
        url="https://github.com/PolicyEngine/policyengine-us-data",
        description="PolicyEngine's calibration targets for US microsimulation",
    )

    try:
        # Try to import PolicyEngine targets
        from policyengine_us_data.datasets.cps.calibration import targets as pe_targets
        source.targets = pe_targets
        source.coverage["national_agi_brackets"] = True
        source.coverage["tax_credits"] = True
    except ImportError:
        pass

    # Document known coverage based on codebase analysis
    source.coverage["national_agi_brackets"] = True
    source.coverage["state_income_distribution"] = True
    source.coverage["tax_credits"] = True
    source.coverage["demographics"] = True
    source.coverage["snap_participation"] = True
    source.coverage["medicaid_enrollment"] = True
    source.coverage["housing_subsidies"] = True

    # Known PolicyEngine targets (from their documentation)
    source.targets["returns_by_agi"] = {
        "Categories": "16 AGI brackets matching IRS SOI",
        "Source": "IRS Statistics of Income",
    }

    return source


def load_yale_targets() -> TargetSource:
    """Load Yale Tax Simulator targets (if available)."""
    source = TargetSource(
        name="Yale Tax Simulator",
        url="https://taxsimulator.org/",
        description="Yale's TAXSIM model targets",
    )

    # Yale TAXSIM coverage (from documentation)
    source.coverage["federal_tax_liability"] = True
    source.coverage["state_tax_liability"] = True
    source.coverage["fica_taxes"] = True
    source.coverage["marginal_rates"] = True

    # They use CPS with specific adjustments
    source.targets["data_source"] = "CPS ASEC with SOI adjustments"
    source.targets["years"] = "1960-2023"
    source.targets["granularity"] = "State-level for state taxes"

    return source


def load_psl_targets() -> TargetSource:
    """Load PSL Tax-Calculator targets."""
    source = TargetSource(
        name="PSL Tax-Calculator",
        url="https://github.com/PSLmodels/Tax-Calculator",
        description="Policy Simulation Library Tax-Calculator targets",
    )

    # PSL Tax-Calculator coverage
    source.coverage["federal_income_tax"] = True
    source.coverage["payroll_tax"] = True
    source.coverage["agi_distribution"] = True
    source.coverage["tax_expenditures"] = True

    # They use PUF primarily
    source.targets["data_source"] = "IRS PUF (primary) + CPS (benefits)"
    source.targets["granularity"] = "National only"
    source.targets["weights"] = "SOI-adjusted"

    return source


def load_taxdata_targets() -> TargetSource:
    """Load Tax-Data (PSL) targets."""
    source = TargetSource(
        name="Tax-Data",
        url="https://github.com/PSLmodels/taxdata",
        description="PSL Tax-Data project targets for weighting PUF/CPS",
    )

    # Tax-Data coverage
    source.coverage["agi_by_bracket"] = True
    source.coverage["income_sources"] = True
    source.coverage["deductions"] = True
    source.coverage["credits"] = True
    source.coverage["state_weights"] = False  # National only

    source.targets["calibration_method"] = "Linear programming (LP)"
    source.targets["data_sources"] = "IRS SOI, CBO, Census"

    return source


def compare_targets(sources: List[TargetSource]) -> pd.DataFrame:
    """Compare target coverage across sources."""
    all_coverage = set()
    for s in sources:
        all_coverage.update(s.coverage.keys())

    rows = []
    for target in sorted(all_coverage):
        row = {"Target": target}
        for s in sources:
            row[s.name] = "✓" if s.coverage.get(target, False) else "✗"
        rows.append(row)

    return pd.DataFrame(rows)


def compare_values(sources: List[TargetSource], irs_soi: Dict) -> pd.DataFrame:
    """Compare actual target values to IRS SOI ground truth."""
    rows = []

    # Compare total returns
    irs_total = irs_soi["total_returns"]
    for s in sources:
        if "returns_by_agi" in s.targets and isinstance(s.targets["returns_by_agi"], dict):
            if all(isinstance(v, (int, float)) for v in s.targets["returns_by_agi"].values()):
                total = sum(s.targets["returns_by_agi"].values())
                error = (total - irs_total) / irs_total * 100
                rows.append({
                    "Metric": "Total Returns",
                    "Source": s.name,
                    "Value": f"{total:,.0f}",
                    "IRS SOI": f"{irs_total:,.0f}",
                    "Error (%)": f"{error:+.1f}%",
                })

    # Compare EITC
    irs_eitc = irs_soi["eitc_amount"]
    for s in sources:
        if "eitc_amount" in s.targets and isinstance(s.targets["eitc_amount"], (int, float)):
            val = s.targets["eitc_amount"]
            error = (val - irs_eitc) / irs_eitc * 100
            rows.append({
                "Metric": "EITC Amount",
                "Source": s.name,
                "Value": f"${val/1e9:,.1f}B",
                "IRS SOI": f"${irs_eitc/1e9:,.1f}B",
                "Error (%)": f"{error:+.1f}%",
            })

    # Compare CTC
    irs_ctc = irs_soi["ctc_amount"]
    for s in sources:
        if "ctc_amount" in s.targets and isinstance(s.targets["ctc_amount"], (int, float)):
            val = s.targets["ctc_amount"]
            error = (val - irs_ctc) / irs_ctc * 100
            rows.append({
                "Metric": "CTC Amount",
                "Source": s.name,
                "Value": f"${val/1e9:,.1f}B",
                "IRS SOI": f"${irs_ctc/1e9:,.1f}B",
                "Error (%)": f"{error:+.1f}%",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def generate_html_dashboard(
    coverage_df: pd.DataFrame,
    values_df: pd.DataFrame,
    sources: List[TargetSource],
    output_path: Path,
):
    """Generate HTML dashboard comparing targets."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Microplex Targets Comparison Dashboard</title>
    <style>
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: #f8fafc;
            color: #1e293b;
        }
        h1 { color: #0f172a; border-bottom: 3px solid #0ea5e9; padding-bottom: 0.5rem; }
        h2 { color: #334155; margin-top: 2rem; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td {
            border: 1px solid #e2e8f0;
            padding: 0.75rem;
            text-align: left;
        }
        th { background: #0ea5e9; color: white; }
        tr:nth-child(even) { background: #f8fafc; }
        .check { color: #22c55e; font-weight: bold; }
        .cross { color: #ef4444; }
        .source-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .source-card h3 { margin-top: 0; color: #0f172a; }
        .source-card a { color: #0ea5e9; }
        .good { color: #22c55e; }
        .warn { color: #f59e0b; }
        .bad { color: #ef4444; }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #0ea5e9, #06b6d4);
            color: white;
            padding: 1rem;
            border-radius: 8px;
        }
        .metric-card .value { font-size: 1.5rem; font-weight: bold; }
        .metric-card .label { opacity: 0.8; }
    </style>
</head>
<body>
    <h1>🎯 Microplex Targets Comparison Dashboard</h1>

    <p>Comparing calibration targets across microsimulation data sources.
    Ground truth: <strong>IRS Statistics of Income (SOI) 2021</strong></p>

    <h2>📊 Key Metrics (IRS SOI 2021)</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="value">153.8M</div>
            <div class="label">Total Tax Returns</div>
        </div>
        <div class="metric-card">
            <div class="value">$14.7T</div>
            <div class="label">Total AGI</div>
        </div>
        <div class="metric-card">
            <div class="value">$64B</div>
            <div class="label">Total EITC</div>
        </div>
        <div class="metric-card">
            <div class="value">$122B</div>
            <div class="label">Total CTC</div>
        </div>
    </div>

    <h2>📦 Data Sources</h2>
"""

    for s in sources:
        html += f"""
    <div class="source-card">
        <h3>{s.name}</h3>
        <p>{s.description}</p>
        <p><a href="{s.url}" target="_blank">{s.url}</a></p>
    </div>
"""

    html += """
    <h2>✓ Target Coverage Comparison</h2>
"""

    # Coverage table
    html += coverage_df.to_html(index=False, classes="coverage-table", escape=False)
    html = html.replace("✓", '<span class="check">✓</span>')
    html = html.replace("✗", '<span class="cross">✗</span>')

    if len(values_df) > 0:
        html += """
    <h2>📈 Target Value Comparison</h2>
"""
        html += values_df.to_html(index=False, escape=False)

    html += """
    <h2>🗺️ Geographic Granularity</h2>
    <table>
        <tr><th>Source</th><th>National</th><th>State</th><th>County</th><th>ZIP</th></tr>
        <tr><td>Cosilico</td><td class="check">✓</td><td class="check">✓</td><td>🚧</td><td>🚧</td></tr>
        <tr><td>PolicyEngine</td><td class="check">✓</td><td class="check">✓</td><td class="cross">✗</td><td class="cross">✗</td></tr>
        <tr><td>Yale TAXSIM</td><td class="check">✓</td><td class="check">✓</td><td class="cross">✗</td><td class="cross">✗</td></tr>
        <tr><td>PSL Tax-Calculator</td><td class="check">✓</td><td class="cross">✗</td><td class="cross">✗</td><td class="cross">✗</td></tr>
    </table>

    <h2>📋 Target Categories</h2>
    <table>
        <tr><th>Category</th><th>Cosilico</th><th>PolicyEngine</th><th>Yale</th><th>PSL</th></tr>
        <tr>
            <td>Income Distribution (AGI brackets)</td>
            <td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td>
        </tr>
        <tr>
            <td>Tax Credits (EITC, CTC, ACTC)</td>
            <td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td>
        </tr>
        <tr>
            <td>Benefits (SNAP, Medicaid, Housing)</td>
            <td>🚧</td><td class="check">✓</td><td class="cross">✗</td><td class="cross">✗</td>
        </tr>
        <tr>
            <td>Demographics (age, family structure)</td>
            <td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="cross">✗</td>
        </tr>
        <tr>
            <td>Employment (wages, SE income)</td>
            <td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td>
        </tr>
        <tr>
            <td>Capital Income (interest, dividends, gains)</td>
            <td class="check">✓</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td>
        </tr>
        <tr>
            <td>Deductions (itemized, charitable)</td>
            <td>🚧</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td>
        </tr>
        <tr>
            <td>Business Income (S-corp, partnership)</td>
            <td>🚧</td><td class="check">✓</td><td class="check">✓</td><td class="check">✓</td>
        </tr>
    </table>

    <h2>🔧 Calibration Methods</h2>
    <table>
        <tr><th>Source</th><th>Method</th><th>Sparsity</th></tr>
        <tr>
            <td>Cosilico</td>
            <td>Cross-Category Selection + IPF (SparseCalibrator)</td>
            <td class="check">✓ Controllable</td>
        </tr>
        <tr>
            <td>PolicyEngine</td>
            <td>Gradient Descent with relative loss</td>
            <td class="cross">✗ Dense</td>
        </tr>
        <tr>
            <td>Yale TAXSIM</td>
            <td>Hot-deck imputation</td>
            <td>N/A</td>
        </tr>
        <tr>
            <td>PSL Tax-Data</td>
            <td>Linear Programming (LP)</td>
            <td class="cross">✗ Dense</td>
        </tr>
    </table>

    <footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #64748b;">
        Generated by <a href="https://github.com/PolicyEngine/microplex">Microplex</a> |
        Data: IRS SOI 2021, Census ACS, DOL |
        Last updated: 2024
    </footer>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"Dashboard saved to {output_path}")


def generate_json_comparison(sources: List[TargetSource], output_path: Path):
    """Export targets comparison as JSON for API use."""
    data = {
        "irs_soi_2021": IRS_SOI_2021,
        "sources": {},
    }

    for s in sources:
        data["sources"][s.name] = {
            "url": s.url,
            "description": s.description,
            "coverage": s.coverage,
            "targets": {k: v for k, v in s.targets.items()
                       if not isinstance(v, (pd.DataFrame, np.ndarray))},
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"JSON comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate targets comparison dashboard")
    parser.add_argument("--output", type=str, default="targets_comparison.html")
    parser.add_argument("--json", type=str, default=None, help="Also output JSON")
    args = parser.parse_args()

    # Find cosilico-data-sources
    script_dir = Path(__file__).parent
    data_source_path = script_dir.parent.parent / "cosilico-data-sources"

    print("Loading target sources...")

    sources = []

    # Load Cosilico targets
    if data_source_path.exists():
        cosilico = load_cosilico_targets(data_source_path)
        sources.append(cosilico)
        print(f"  ✓ {cosilico.name}: {len(cosilico.coverage)} coverage areas")
    else:
        print(f"  ✗ Cosilico: data sources not found at {data_source_path}")

    # Load other sources
    policyengine = load_policyengine_targets()
    sources.append(policyengine)
    print(f"  ✓ {policyengine.name}: {len(policyengine.coverage)} coverage areas")

    yale = load_yale_targets()
    sources.append(yale)
    print(f"  ✓ {yale.name}: {len(yale.coverage)} coverage areas")

    psl = load_psl_targets()
    sources.append(psl)
    print(f"  ✓ {psl.name}: {len(psl.coverage)} coverage areas")

    taxdata = load_taxdata_targets()
    sources.append(taxdata)
    print(f"  ✓ {taxdata.name}: {len(taxdata.coverage)} coverage areas")

    # Generate comparisons
    print("\nGenerating comparison tables...")
    coverage_df = compare_targets(sources)
    values_df = compare_values(sources, IRS_SOI_2021)

    # Generate HTML dashboard
    output_path = Path(args.output)
    generate_html_dashboard(coverage_df, values_df, sources, output_path)

    # Optionally generate JSON
    if args.json:
        generate_json_comparison(sources, Path(args.json))

    print("\n✅ Dashboard generation complete!")


if __name__ == "__main__":
    main()
