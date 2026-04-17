"""External data source targets for validation.

Sources:
- Yale Budget Lab: https://budgetlab.yale.edu/
- PSL Tax-Data: https://github.com/PSLmodels/taxdata
- Tax Foundation: https://taxfoundation.org/
- CBPP: https://www.cbpp.org/
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExternalTarget:
    """A target from an external data source."""
    name: str
    value: float
    source: str
    url: str
    year: int
    notes: str = ""


# Yale Budget Lab distributional analysis targets
YALE_BUDGET_LAB_TARGETS = {
    "source": "Yale Budget Lab",
    "url": "https://budgetlab.yale.edu/",
    "year": 2024,
    "targets": {
        # Income distribution (from Yale distributional tables)
        # These would be populated from their actual publications
        "bottom_20_income_share": 0.032,  # Bottom quintile share of income
        "top_1_income_share": 0.205,       # Top 1% share of income
        "gini_coefficient": 0.49,          # Pre-tax income Gini

        # Tax burden distribution
        "effective_tax_rate_bottom_20": 0.02,
        "effective_tax_rate_top_1": 0.24,
    },
}


# PSL Tax-Data growth factors and validation targets
PSL_TAXDATA_TARGETS = {
    "source": "PSL Tax-Data",
    "url": "https://github.com/PSLmodels/taxdata",
    "year": 2024,
    "targets": {
        # These are extrapolation/growth targets used in taxdata
        # See: https://github.com/PSLmodels/taxdata/blob/master/taxdata/growfactors.csv

        # 2024 projections (example values)
        "wage_growth_factor": 1.045,      # Wage growth from base year
        "interest_growth_factor": 1.02,
        "dividend_growth_factor": 1.03,
        "cg_growth_factor": 1.05,

        # Population weights
        "total_tax_units": 175_000_000,
    },
}


# Tax Foundation data
TAX_FOUNDATION_TARGETS = {
    "source": "Tax Foundation",
    "url": "https://taxfoundation.org/data/",
    "year": 2023,
    "targets": {
        # Federal tax statistics
        "total_federal_revenue": 4_439_000_000_000,
        "individual_income_tax_revenue": 2_176_000_000_000,
        "payroll_tax_revenue": 1_614_000_000_000,
        "corporate_income_tax_revenue": 420_000_000_000,

        # Tax rates
        "top_marginal_rate": 0.37,
        "corporate_rate": 0.21,

        # Filer statistics
        "returns_with_no_liability": 60_600_000,
        "returns_with_eitc": 31_000_000,
        "average_eitc_amount": 2_541,
    },
}


# CBPP (Center on Budget and Policy Priorities) benefit data
CBPP_TARGETS = {
    "source": "CBPP",
    "url": "https://www.cbpp.org/",
    "year": 2023,
    "targets": {
        # SNAP
        "snap_participants": 42_000_000,
        "snap_households": 21_600_000,
        "average_snap_benefit": 234,  # per person per month

        # SSI
        "ssi_recipients": 7_400_000,
        "average_ssi_payment": 674,  # per month

        # TANF
        "tanf_recipients": 2_000_000,
        "tanf_families": 700_000,

        # Housing assistance
        "housing_assistance_households": 5_200_000,
    },
}


# CBO (Congressional Budget Office) projections
CBO_TARGETS = {
    "source": "CBO",
    "url": "https://www.cbo.gov/",
    "year": 2024,
    "targets": {
        # Economic projections
        "gdp_nominal": 28_300_000_000_000,
        "labor_force": 166_000_000,
        "unemployment_rate": 0.039,

        # Budget projections
        "federal_outlays": 6_900_000_000_000,
        "social_security_outlays": 1_500_000_000_000,
        "medicare_outlays": 1_000_000_000_000,
        "medicaid_outlays": 600_000_000_000,
    },
}


def get_all_external_targets() -> dict[str, dict]:
    """Get all external validation targets."""
    return {
        "yale_budget_lab": YALE_BUDGET_LAB_TARGETS,
        "psl_taxdata": PSL_TAXDATA_TARGETS,
        "tax_foundation": TAX_FOUNDATION_TARGETS,
        "cbpp": CBPP_TARGETS,
        "cbo": CBO_TARGETS,
    }


def export_targets_json(output_path: Path) -> None:
    """Export all targets to JSON."""
    targets = get_all_external_targets()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(targets, f, indent=2)


def generate_targets_markdown(output_path: Path | None = None) -> str:
    """Generate markdown summary of all targets."""
    targets = get_all_external_targets()

    lines = [
        "# External Validation Targets",
        "",
        "Aggregate statistics from authoritative sources for validating microplex.",
        "",
    ]

    for source_key, source_data in targets.items():
        source = source_data["source"]
        url = source_data["url"]
        year = source_data["year"]

        lines.extend([
            f"## {source}",
            "",
            f"Source: [{url}]({url})",
            f"Year: {year}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ])

        for name, value in source_data["targets"].items():
            if isinstance(value, float) and value < 1:
                # Format as percentage
                value_str = f"{value:.1%}"
            elif isinstance(value, (int, float)) and value >= 1_000_000:
                # Format large numbers
                value_str = f"{value:,.0f}"
            else:
                value_str = f"{value:,.2f}" if isinstance(value, float) else str(value)

            lines.append(f"| {name.replace('_', ' ').title()} | {value_str} |")

        lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report


if __name__ == "__main__":
    # Generate targets documentation
    generate_targets_markdown(Path("dashboard/external_targets.md"))
    export_targets_json(Path("dashboard/external_targets.json"))
    print("Generated external targets documentation")
