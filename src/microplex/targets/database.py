"""
Calibration Targets Database

Stores and manages calibration targets from multiple sources,
with mappings to RAC variables for PolicyEngine integration.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class TargetCategory(Enum):
    """Categories of calibration targets."""
    # Income targets (IRS SOI)
    AGI_DISTRIBUTION = "agi_distribution"
    INCOME_SOURCES = "income_sources"
    DEDUCTIONS = "deductions"
    TAX_LIABILITY = "tax_liability"

    # Tax credits (IRS SOI)
    EITC = "eitc"
    CTC = "ctc"
    ACTC = "actc"
    OTHER_CREDITS = "other_credits"

    # Benefits (Admin data)
    SNAP = "snap"
    MEDICAID = "medicaid"
    HOUSING = "housing"
    SSI = "ssi"
    TANF = "tanf"
    UNEMPLOYMENT = "unemployment"

    # Demographics (Census)
    POPULATION = "population"
    HOUSEHOLD_STRUCTURE = "household_structure"
    AGE_DISTRIBUTION = "age_distribution"
    EMPLOYMENT = "employment"


@dataclass
class Target:
    """A calibration target."""
    name: str
    category: TargetCategory
    value: float
    year: int
    source: str
    source_url: str | None = None

    # Geographic scope
    geography: str = "US"  # US, state FIPS, county FIPS
    state_fips: str | None = None

    # Filtering dimensions
    filing_status: str | None = None  # All, Single, MFJ, MFS, HOH
    agi_lower: float = -np.inf
    agi_upper: float = np.inf

    # Target type
    is_count: bool = True  # True = count of units, False = dollar amount
    is_taxable_only: bool = False

    # RAC mapping
    rac_variable: str | None = None  # e.g., "adjusted_gross_income"
    rac_statute: str | None = None   # e.g., "26/62"

    # Microdata column mapping
    microdata_column: str | None = None  # Column in CPS/PUF

    # Metadata
    notes: str | None = None
    last_updated: str | None = None


@dataclass
class TargetsDatabase:
    """
    Database of calibration targets from multiple sources.

    Maintains parity with PolicyEngine targets while adding
    RAC variable mappings for PolicyEngine integration.
    """
    targets: list[Target] = field(default_factory=list)
    _by_category: dict[TargetCategory, list[Target]] = field(default_factory=dict)
    _by_geography: dict[str, list[Target]] = field(default_factory=dict)

    def add(self, target: Target):
        """Add a target to the database."""
        self.targets.append(target)

        # Index by category
        if target.category not in self._by_category:
            self._by_category[target.category] = []
        self._by_category[target.category].append(target)

        # Index by geography
        if target.geography not in self._by_geography:
            self._by_geography[target.geography] = []
        self._by_geography[target.geography].append(target)

    def add_many(self, targets: list[Target]):
        """Add multiple targets."""
        for t in targets:
            self.add(t)

    def get_by_category(self, category: TargetCategory) -> list[Target]:
        """Get all targets in a category."""
        return self._by_category.get(category, [])

    def get_by_geography(self, geography: str) -> list[Target]:
        """Get all targets for a geography."""
        return self._by_geography.get(geography, [])

    def get_national(self) -> list[Target]:
        """Get national-level targets."""
        return self.get_by_geography("US")

    def get_state(self, state_fips: str) -> list[Target]:
        """Get state-level targets."""
        return [t for t in self.targets if t.state_fips == state_fips]

    def get_with_rac_mapping(self) -> list[Target]:
        """Get targets that have RAC variable mappings."""
        return [t for t in self.targets if t.rac_variable is not None]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        rows = []
        for t in self.targets:
            rows.append({
                "name": t.name,
                "category": t.category.value,
                "value": t.value,
                "year": t.year,
                "source": t.source,
                "geography": t.geography,
                "state_fips": t.state_fips,
                "filing_status": t.filing_status,
                "agi_lower": t.agi_lower,
                "agi_upper": t.agi_upper,
                "is_count": t.is_count,
                "rac_variable": t.rac_variable,
                "rac_statute": t.rac_statute,
                "microdata_column": t.microdata_column,
            })
        return pd.DataFrame(rows)

    def to_calibration_format(
        self,
        geography: str = "US",
        year: int = 2021,
    ) -> tuple[dict[str, dict], dict[str, float]]:
        """
        Convert to microplex calibration format.

        Returns:
            marginal_targets: {variable: {category: target_count}}
            continuous_targets: {variable: target_sum}
        """
        marginal_targets = {}
        continuous_targets = {}

        for t in self.targets:
            if t.geography != geography or t.year != year:
                continue

            if t.microdata_column is None:
                continue

            if t.is_count:
                # Categorical target
                var = t.microdata_column
                if var not in marginal_targets:
                    marginal_targets[var] = {}

                # Create category key from AGI bracket
                if t.agi_lower != -np.inf or t.agi_upper != np.inf:
                    cat = f"{t.agi_lower:.0f}_to_{t.agi_upper:.0f}"
                else:
                    cat = "all"

                marginal_targets[var][cat] = t.value
            else:
                # Continuous target (sum)
                continuous_targets[t.microdata_column] = t.value

        return marginal_targets, continuous_targets

    def compare_to_policyengine(self, pe_targets: pd.DataFrame) -> pd.DataFrame:
        """Compare our targets to PolicyEngine's."""
        our_df = self.to_dataframe()

        # Merge on matching columns
        comparison = our_df.merge(
            pe_targets,
            left_on=["name", "year"],
            right_on=["Variable", "Year"],
            how="outer",
            suffixes=("_policyengine", "_pe"),
        )

        comparison["difference"] = comparison["value"] - comparison["Value"]
        comparison["pct_difference"] = comparison["difference"] / comparison["Value"] * 100

        return comparison

    def coverage_summary(self) -> dict[str, int]:
        """Summarize target coverage by category."""
        summary = {}
        for cat in TargetCategory:
            summary[cat.value] = len(self.get_by_category(cat))
        return summary

    def __len__(self) -> int:
        return len(self.targets)

    def __repr__(self) -> str:
        coverage = self.coverage_summary()
        non_zero = {k: v for k, v in coverage.items() if v > 0}
        return f"TargetsDatabase({len(self)} targets across {len(non_zero)} categories)"
