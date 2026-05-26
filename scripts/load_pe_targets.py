"""
Batch loader for PolicyEngine calibration targets to Supabase.

Uses batch upsert operations for 10-100x faster loading than individual inserts.
"""

import os
import time
import pandas as pd
import requests
from io import StringIO
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

# Supabase connection. Prefer PolicyEngine-owned secrets while accepting the
# old names during the migration window.
SUPABASE_URL = os.environ.get("POLICYENGINE_SUPABASE_URL") or os.environ.get(
    "SUPABASE_URL"
)
SUPABASE_KEY = os.environ.get(
    "POLICYENGINE_SUPABASE_SERVICE_KEY"
) or os.environ.get("COSILICO_SUPABASE_SERVICE_KEY")

PE_BASE = "https://raw.githubusercontent.com/PolicyEngine/policyengine-us-data/main/policyengine_us_data/storage/calibration_targets"

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY"
}
STATE_NAME_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}


class BatchSupabaseClient:
    """Supabase client optimized for batch operations."""

    def __init__(self, url: str, key: str, schema: str = "microplex"):
        if not url:
            raise ValueError(
                "POLICYENGINE_SUPABASE_URL must be set before loading "
                "PolicyEngine calibration targets."
            )
        if not key:
            raise ValueError(
                "POLICYENGINE_SUPABASE_SERVICE_KEY must be set before loading "
                "PolicyEngine calibration targets."
            )
        self.base_url = f"{url}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }
        self.session = requests.Session()

    def _request(self, method: str, url: str, max_retries: int = 5, **kwargs) -> requests.Response:
        """Make request with retry on transient errors."""
        kwargs.setdefault("timeout", 60)
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in [429, 500, 502, 503, 504]:
                    if attempt < max_retries - 1:
                        wait = min(2 ** attempt, 30)
                        print(f"    Retry {attempt+1}/{max_retries} after {resp.status_code}, waiting {wait}s")
                        time.sleep(wait)
                        continue
                return resp
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 30)
                    print(f"    Retry {attempt+1}/{max_retries} after {type(e).__name__}, waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise
        return resp

    def batch_upsert(self, table: str, records: List[Dict], on_conflict: str,
                     chunk_size: int = 500, ignore_duplicates: bool = False) -> List[Dict]:
        """Batch upsert records with chunking."""
        if not records:
            return []

        results = []
        # Try upsert first, fall back to insert with duplicate handling
        url = f"{self.base_url}/{table}"
        if on_conflict:
            url += f"?on_conflict={on_conflict}"

        # Use merge-duplicates for upsert behavior
        prefer = "resolution=merge-duplicates,return=representation"
        if ignore_duplicates:
            prefer = "resolution=ignore-duplicates,return=representation"
        headers = {**self.headers, "Prefer": prefer}

        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            resp = self._request("POST", url, headers=headers, json=chunk)

            # Handle 400 by trying without on_conflict (table may not have constraint)
            if resp.status_code == 400 and on_conflict:
                # Retry as simple insert (may fail on duplicates)
                url_simple = f"{self.base_url}/{table}"
                headers_simple = {**self.headers, "Prefer": "return=representation"}
                resp = self._request("POST", url_simple, headers=headers_simple, json=chunk)

                # If we get 409 (conflict), fall back to individual upserts
                if resp.status_code == 409:
                    print(f"    Falling back to individual upserts for {len(chunk)} records...")
                    for record in chunk:
                        self._upsert_single(table, record, on_conflict)
                    results.extend(chunk)  # Return the records we tried to insert
                    continue

            resp.raise_for_status()
            results.extend(resp.json())
            if len(records) > chunk_size:
                print(f"    Batch progress: {min(i + chunk_size, len(records))}/{len(records)}")

        return results

    def _upsert_single(self, table: str, record: Dict, on_conflict: str):
        """Upsert a single record by checking existence first."""
        # For targets, check if exists by composite key
        if table == "targets" and on_conflict:
            keys = on_conflict.split(",")
            filters = "&".join(f"{k}=eq.{record.get(k)}" for k in keys if record.get(k))
            check_url = f"{self.base_url}/{table}?select=id&{filters}"
            resp = self._request("GET", check_url, headers=self.headers)
            if resp.status_code == 200 and resp.json():
                # Update existing
                existing_id = resp.json()[0]["id"]
                update_url = f"{self.base_url}/{table}?id=eq.{existing_id}"
                self._request("PATCH", update_url, headers=self.headers, json=record)
                return

        # Insert new
        url = f"{self.base_url}/{table}"
        headers = {**self.headers, "Prefer": "return=representation"}
        self._request("POST", url, headers=headers, json=record)

    def batch_upsert_strata(self, strata: List[Dict], return_mapping: bool = False) -> Any:
        """Batch upsert strata with optional name->id mapping."""
        result = self.batch_upsert("strata", strata, "name,jurisdiction")
        if return_mapping:
            return {(r["name"], r["jurisdiction"]): r["id"] for r in result}
        return result

    def batch_upsert_targets(self, targets: List[Dict], chunk_size: int = 500) -> List[Dict]:
        """Batch upsert targets."""
        return self.batch_upsert("targets", targets, "source_id,stratum_id,variable,period",
                                 chunk_size=chunk_size)


@dataclass
class TargetCollector:
    """Collects targets and strata for batch insertion."""
    sources: Dict[Tuple[str, str, str], Dict] = field(default_factory=dict)  # (jurisdiction, institution, dataset) -> source data
    strata: Dict[Tuple[str, str], Dict] = field(default_factory=dict)  # (name, jurisdiction) -> stratum data
    targets: List[Dict] = field(default_factory=list)

    # Mappings populated after batch insert
    source_ids: Dict[Tuple[str, str, str], str] = field(default_factory=dict)
    stratum_ids: Dict[Tuple[str, str], str] = field(default_factory=dict)

    def add_source(self, jurisdiction: str, institution: str, dataset: str,
                   name: str, url: str = None) -> Tuple[str, str, str]:
        """Register a source, return key for later ID lookup."""
        key = (jurisdiction, institution, dataset)
        if key not in self.sources:
            self.sources[key] = {
                "jurisdiction": jurisdiction,
                "institution": institution,
                "dataset": dataset,
                "name": name,
                "url": url
            }
        return key

    def add_stratum(self, name: str, jurisdiction: str) -> Tuple[str, str]:
        """Register a stratum, return key for later ID lookup."""
        key = (name, jurisdiction)
        if key not in self.strata:
            self.strata[key] = {
                "name": name,
                "jurisdiction": jurisdiction,
                "description": name
            }
        return key

    def add_target(self, source_key: Tuple[str, str, str], stratum_key: Tuple[str, str],
                   variable: str, value: float, target_type: str = "amount",
                   period: int = 2024, notes: str = None):
        """Add a target (source/stratum resolved later)."""
        # Convert numpy types
        if hasattr(value, 'item'):
            value = value.item()
        value = float(value)

        self.targets.append({
            "_source_key": source_key,
            "_stratum_key": stratum_key,
            "variable": variable,
            "value": value,
            "target_type": target_type,
            "period": period,
            "notes": notes
        })


def fetch_csv(filename: str) -> pd.DataFrame:
    """Fetch CSV from PE repo."""
    url = f"{PE_BASE}/{filename}"
    print(f"  Fetching {filename}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def collect_irs_income_targets(collector: TargetCollector):
    """Collect IRS SOI income targets."""
    print("\n=== Collecting IRS Income Targets ===")
    source_key = collector.add_source(
        "us", "irs", "soi",
        "IRS Statistics of Income",
        "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
    )
    stratum_key = collector.add_stratum("US total population", "us")

    targets = {
        "employment_income": 9_022_400_000_000,
        "self_employment_income": 436_400_000_000,
        "taxable_pension_income": 827_600_000_000,
        "tax_exempt_pension_income": 580_400_000_000,
        "social_security": 774_000_000_000,
        "unemployment_compensation": 208_000_000_000,
        "dividend_income": 260_200_000_000,
        "interest_income": 127_400_000_000,
        "rental_income": 46_000_000_000,
        "long_term_capital_gains": 1_137_000_000_000,
        "short_term_capital_gains": -72_000_000_000,
        "partnership_s_corp_income": 976_000_000_000,
        "farm_income": -26_100_000_000,
        "alimony_income": 8_500_000_000,
    }

    for name, value in targets.items():
        collector.add_target(source_key, stratum_key, name, value, "amount", 2024)

    print(f"  Collected {len(targets)} income targets")


def collect_benefit_spending_targets(collector: TargetCollector):
    """Collect CBO benefit spending targets."""
    print("\n=== Collecting Benefit Spending Targets ===")
    source_key = collector.add_source(
        "us", "cbo", "budget_projections",
        "CBO Budget Projections",
        "https://www.cbo.gov/data/budget-economic-data"
    )
    stratum_key = collector.add_stratum("US total population", "us")

    targets = {
        "snap_spending": 103_100_000_000,
        "ssi_spending": 78_500_000_000,
        "social_security_spending": 2_623_800_000_000,
        "eitc_spending": 72_700_000_000,
        "unemployment_spending": 59_100_000_000,
        "medicaid_spending": 900_000_000_000,
        "aca_ptc_spending": 98_000_000_000,
    }

    for name, value in targets.items():
        collector.add_target(source_key, stratum_key, name, value, "amount", 2024)

    print(f"  Collected {len(targets)} benefit targets")


def collect_healthcare_targets(collector: TargetCollector):
    """Collect healthcare coverage targets."""
    print("\n=== Collecting Healthcare Targets ===")
    source_key = collector.add_source(
        "us", "hhs", "healthcare_coverage",
        "HHS Healthcare Coverage Estimates",
        "https://www.hhs.gov/"
    )
    stratum_key = collector.add_stratum("US total population", "us")

    targets = {
        "health_insurance_premiums": (385_000_000_000, "amount"),
        "other_medical_expenses": (278_000_000_000, "amount"),
        "medicare_part_b_premiums": (112_000_000_000, "amount"),
        "over_the_counter_health_expenses": (72_000_000_000, "amount"),
        "medicaid_enrollment": (72_429_055, "count"),
        "aca_enrollment": (19_743_689, "count"),
    }

    for name, (value, ttype) in targets.items():
        collector.add_target(source_key, stratum_key, name, value, ttype, 2024)

    print(f"  Collected {len(targets)} healthcare targets")


def collect_tax_targets(collector: TargetCollector):
    """Collect JCT tax expenditure targets."""
    print("\n=== Collecting Tax Expenditure Targets ===")
    source_key = collector.add_source(
        "us", "jct", "tax_expenditures",
        "JCT Tax Expenditure Estimates",
        "https://www.jct.gov/publications/tax-expenditure-estimates/"
    )
    stratum_key = collector.add_stratum("US total population", "us")

    targets = {
        "salt_deduction": 21_247_000_000,
        "medical_expense_deduction": 11_400_000_000,
        "charitable_deduction": 65_301_000_000,
        "interest_deduction": 24_800_000_000,
        "qbi_deduction": 63_100_000_000,
    }

    for name, value in targets.items():
        collector.add_target(source_key, stratum_key, name, value, "amount", 2024)

    print(f"  Collected {len(targets)} tax targets")


def collect_eitc_targets(collector: TargetCollector):
    """Collect EITC by number of children."""
    print("\n=== Collecting EITC Targets ===")
    df = fetch_csv("eitc.csv")

    source_key = collector.add_source(
        "us", "irs", "soi_eitc",
        "IRS SOI EITC Statistics",
        "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
    )

    for _, row in df.iterrows():
        n_children = int(row["count_children"])
        stratum_key = collector.add_stratum(
            f"Tax filers with {n_children} qualifying children", "us"
        )
        collector.add_target(source_key, stratum_key, "eitc_returns", row["eitc_returns"], "count", 2020)
        collector.add_target(source_key, stratum_key, "eitc_total", row["eitc_total"], "amount", 2020)

    print(f"  Collected {len(df) * 2} EITC targets")


def collect_medicaid_targets(collector: TargetCollector):
    """Collect Medicaid enrollment by state."""
    print("\n=== Collecting Medicaid Targets ===")
    df = fetch_csv("medicaid_enrollment_2024.csv")

    source_key = collector.add_source(
        "us", "hhs", "medicaid",
        "HHS Medicaid Enrollment Data",
        "https://www.medicaid.gov/medicaid/program-information/medicaid-and-chip-enrollment-data/index.html"
    )

    # National
    national_stratum = collector.add_stratum("US total population", "us")
    collector.add_target(source_key, national_stratum, "medicaid_enrollment",
                        df["enrollment"].sum(), "count", 2024)

    # By state
    for _, row in df.iterrows():
        state = row["state"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue
        stratum_key = collector.add_stratum(f"State {state} population", f"us-{state.lower()}")
        collector.add_target(source_key, stratum_key, "medicaid_enrollment", row["enrollment"], "count", 2024)

    print(f"  Collected {len(df) + 1} Medicaid targets")


def collect_snap_targets(collector: TargetCollector):
    """Collect SNAP by state."""
    print("\n=== Collecting SNAP Targets ===")
    df = fetch_csv("snap_state.csv")
    df["state_fips"] = df["GEO_ID"].str.extract(r"(\d{2})$")

    source_key = collector.add_source(
        "us", "usda", "snap",
        "USDA SNAP State Activity Report",
        "https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap"
    )

    # National
    national_stratum = collector.add_stratum("US total population", "us")
    collector.add_target(source_key, national_stratum, "snap_households", df["Households"].sum(), "count", 2024)
    collector.add_target(source_key, national_stratum, "snap_spending", df["Cost"].sum(), "amount", 2024)

    # By state
    for _, row in df.iterrows():
        fips = row["state_fips"]
        state = STATE_FIPS.get(fips)
        if not state:
            continue
        stratum_key = collector.add_stratum(f"State {state} population", f"us-{state.lower()}")
        collector.add_target(source_key, stratum_key, "snap_households", row["Households"], "count", 2024)
        collector.add_target(source_key, stratum_key, "snap_spending", row["Cost"], "amount", 2024)

    print(f"  Collected {(len(df) + 1) * 2} SNAP targets")


def collect_aca_targets(collector: TargetCollector):
    """Collect ACA enrollment by state."""
    print("\n=== Collecting ACA Targets ===")
    df = fetch_csv("aca_spending_and_enrollment_2024.csv")

    source_key = collector.add_source(
        "us", "cms", "aca_marketplace",
        "CMS ACA Marketplace Enrollment",
        "https://www.cms.gov/newsroom/fact-sheets/marketplace-2024-open-enrollment-period-report-final-national-snapshot"
    )

    # National
    national_stratum = collector.add_stratum("US total population", "us")
    collector.add_target(source_key, national_stratum, "aca_enrollment", df["enrollment"].sum(), "count", 2024)
    collector.add_target(source_key, national_stratum, "aca_ptc_spending", df["spending"].sum(), "amount", 2024)

    # By state
    for _, row in df.iterrows():
        state = row["state"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue
        stratum_key = collector.add_stratum(f"State {state} population", f"us-{state.lower()}")
        collector.add_target(source_key, stratum_key, "aca_enrollment", row["enrollment"], "count", 2024)
        collector.add_target(source_key, stratum_key, "aca_ptc_spending", row["spending"], "amount", 2024)

    print(f"  Collected {(len(df) + 1) * 2} ACA targets")


def collect_population_targets(collector: TargetCollector):
    """Collect population by state."""
    print("\n=== Collecting Population Targets ===")
    df = fetch_csv("population_by_state.csv")

    source_key = collector.add_source(
        "us", "census", "population_projections",
        "Census Bureau Population Projections",
        "https://www.census.gov/programs-surveys/popproj.html"
    )

    # National
    national_stratum = collector.add_stratum("US total population", "us")
    collector.add_target(source_key, national_stratum, "total_population", df["population"].sum(), "count", 2024)
    collector.add_target(source_key, national_stratum, "population_under_5", df["population_under_5"].sum(), "count", 2024)

    # By state
    for _, row in df.iterrows():
        state = row["state"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue
        stratum_key = collector.add_stratum(f"State {state} population", f"us-{state.lower()}")
        collector.add_target(source_key, stratum_key, "total_population", row["population"], "count", 2024)
        collector.add_target(source_key, stratum_key, "population_under_5", row["population_under_5"], "count", 2024)

    print(f"  Collected {(len(df) + 1) * 2} population targets")


def collect_age_state_targets(collector: TargetCollector):
    """Collect age distribution by state."""
    print("\n=== Collecting Age by State Targets ===")
    df = fetch_csv("age_state.csv")

    source_key = collector.add_source(
        "us", "census", "acs_age",
        "Census ACS Age Distribution",
        "https://data.census.gov/"
    )

    age_cols = [c for c in df.columns if c not in ["GEO_ID", "GEO_NAME"]]

    for _, row in df.iterrows():
        fips = row["GEO_ID"][-2:]
        state = STATE_FIPS.get(fips)
        if not state:
            continue

        stratum_key = collector.add_stratum(f"State {state} population", f"us-{state.lower()}")
        for age_col in age_cols:
            collector.add_target(source_key, stratum_key, f"population_age_{age_col}", row[age_col], "count", 2024)

    print(f"  Collected {len(df) * len(age_cols)} age-state targets")


def collect_agi_state_targets(collector: TargetCollector):
    """Collect AGI by state and bracket."""
    print("\n=== Collecting AGI by State Targets ===")
    df = fetch_csv("agi_state.csv")

    source_key = collector.add_source(
        "us", "irs", "soi_state",
        "IRS SOI State Data",
        "https://www.irs.gov/statistics/soi-tax-stats-historic-table-2"
    )

    count = 0
    for _, row in df.iterrows():
        geo_id = row["GEO_ID"]
        if pd.isna(geo_id) or not isinstance(geo_id, str):
            continue

        fips = geo_id[-2:]
        state = STATE_FIPS.get(fips)
        if not state:
            continue

        agi_lower = row["AGI_LOWER_BOUND"]
        agi_upper = row["AGI_UPPER_BOUND"]

        # Create stratum name
        is_neg_inf = pd.isna(agi_lower) or str(agi_lower) == "-inf"
        is_pos_inf = pd.isna(agi_upper) or str(agi_upper) == "inf"

        if is_neg_inf and is_pos_inf:
            agi_range = "all AGI"
        elif is_neg_inf:
            agi_range = f"AGI < ${float(agi_upper):,.0f}"
        elif is_pos_inf:
            agi_range = f"AGI >= ${float(agi_lower):,.0f}"
        else:
            agi_range = f"AGI ${float(agi_lower):,.0f}-${float(agi_upper):,.0f}"

        stratum_key = collector.add_stratum(f"State {state} {agi_range}", f"us-{state.lower()}")

        is_count = row["IS_COUNT"] if "IS_COUNT" in row else False
        target_type = "count" if is_count else "amount"
        variable = row["VARIABLE"] if "VARIABLE" in row else "adjusted_gross_income"

        collector.add_target(source_key, stratum_key, variable, row["VALUE"], target_type, 2024)
        count += 1

    print(f"  Collected {count} AGI-state targets")


def collect_real_estate_tax_targets(collector: TargetCollector):
    """Collect real estate taxes by state."""
    print("\n=== Collecting Real Estate Tax Targets ===")
    df = fetch_csv("real_estate_taxes_by_state_acs.csv")

    source_key = collector.add_source(
        "us", "census", "acs_real_estate_taxes",
        "Census ACS Real Estate Taxes",
        "https://data.census.gov/"
    )

    for _, row in df.iterrows():
        state = row["state_code"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue

        stratum_key = collector.add_stratum(f"State {state} population", f"us-{state.lower()}")
        value = row["real_estate_taxes_bn"] * 1e9
        collector.add_target(source_key, stratum_key, "real_estate_taxes", value, "amount", 2024)

    print(f"  Collected {len(df)} real estate tax targets")


def collect_soi_targets(collector: TargetCollector):
    """Collect SOI targets by AGI bracket."""
    print("\n=== Collecting SOI Targets by AGI Bracket ===")
    df = fetch_csv("soi_targets.csv")

    source_key = collector.add_source(
        "us", "irs", "soi_detailed",
        "IRS Statistics of Income - Detailed",
        "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
    )

    count = 0
    for (year, filing_status, agi_lower, agi_upper, variable), group in df.groupby(
        ["Year", "Filing status", "AGI lower bound", "AGI upper bound", "Variable"]
    ):
        is_neg_inf = str(agi_lower) == "-inf"
        is_pos_inf = str(agi_upper) == "inf"

        if is_neg_inf and is_pos_inf:
            agi_range = "all AGI"
        elif is_neg_inf:
            agi_range = f"AGI < ${float(agi_upper):,.0f}"
        elif is_pos_inf:
            agi_range = f"AGI >= ${float(agi_lower):,.0f}"
        else:
            agi_range = f"AGI ${float(agi_lower):,.0f}-${float(agi_upper):,.0f}"

        stratum_key = collector.add_stratum(f"Tax filers {filing_status} {agi_range}", "us")

        value = group["Value"].iloc[0]
        is_count = group["Count"].iloc[0] if "Count" in group.columns else False
        target_type = "count" if is_count else "amount"

        collector.add_target(source_key, stratum_key, variable, value, target_type, int(year))
        count += 1

    print(f"  Collected {count} SOI targets")


def collect_spm_agi_targets(collector: TargetCollector):
    """Collect SPM by AGI decile."""
    print("\n=== Collecting SPM by AGI Decile Targets ===")
    df = fetch_csv("spm_threshold_agi.csv")

    source_key = collector.add_source(
        "us", "census", "spm",
        "Census SPM Thresholds",
        "https://www.census.gov/topics/income-poverty/supplemental-poverty-measure.html"
    )

    for _, row in df.iterrows():
        decile = int(row["decile"])
        spm_lower = row["lower_spm_threshold"]
        spm_upper = row["upper_spm_threshold"]

        stratum_key = collector.add_stratum(f"Decile {decile} (SPM ${spm_lower:,.0f}-${spm_upper:,.0f})", "us")
        collector.add_target(source_key, stratum_key, "adjusted_gross_income", row["adjusted_gross_income"], "amount", 2024)
        collector.add_target(source_key, stratum_key, "count", row["count"], "count", 2024)

    print(f"  Collected {len(df) * 2} SPM-AGI targets")


def collect_healthcare_age_targets(collector: TargetCollector):
    """Collect healthcare spending by age."""
    print("\n=== Collecting Healthcare Spending by Age ===")
    df = fetch_csv("healthcare_spending.csv")

    source_key = collector.add_source(
        "us", "bls", "healthcare_spending_age",
        "BLS Consumer Expenditure Survey - Healthcare",
        "https://www.bls.gov/cex/"
    )

    count = 0
    for _, row in df.iterrows():
        age_lower = int(row["age_10_year_lower_bound"])

        if age_lower == 80:
            stratum_name = "Healthcare population age 80+"
        else:
            stratum_name = f"Healthcare population age {age_lower}-{age_lower + 9}"

        stratum_key = collector.add_stratum(stratum_name, "us")

        for col in ["health_insurance_premiums_without_medicare_part_b",
                    "over_the_counter_health_expenses", "other_medical_expenses",
                    "medicare_part_b_premiums"]:
            if col in row:
                collector.add_target(source_key, stratum_key, col, row[col], "amount", 2024)
                count += 1

    print(f"  Collected {count} healthcare-age targets")


def collect_census_projection_targets(collector: TargetCollector):
    """Collect Census projections by demographics."""
    print("\n=== Collecting Census Projection Targets ===")
    df = fetch_csv("np2023_d5_mid.csv")

    source_key = collector.add_source(
        "us", "census", "population_projections_detailed",
        "Census Population Projections 2023",
        "https://www.census.gov/programs-surveys/popproj.html"
    )

    race_map = {0: "All", 1: "White non-Hispanic", 2: "Black", 3: "AIAN",
                4: "Asian", 5: "NHPI", 6: "Two+", 7: "Hispanic"}
    sex_map = {0: "Both", 1: "Male", 2: "Female"}
    nativity_map = {0: "All", 1: "Native", 2: "Foreign-born"}

    for _, row in df.iterrows():
        year = int(row["YEAR"])
        nativity = int(row["NATIVITY"])
        race = int(row["RACE_HISP"])
        sex = int(row["SEX"])

        parts = []
        if nativity != 0:
            parts.append(nativity_map.get(nativity, f"Nativity {nativity}"))
        if race != 0:
            parts.append(race_map.get(race, f"Race {race}"))
        if sex != 0:
            parts.append(sex_map.get(sex, f"Sex {sex}"))

        base_name = " ".join(parts) if parts else "Total population"
        stratum_key = collector.add_stratum(f"Census projection {base_name} ({year})", "us")
        collector.add_target(source_key, stratum_key, "total_population", row["TOTAL_POP"], "count", year)

    print(f"  Collected {len(df)} census projection targets")


def batch_insert_all(collector: TargetCollector, client: BatchSupabaseClient):
    """Batch insert all collected data."""
    print("\n" + "=" * 70)
    print("BATCH INSERTING ALL DATA")
    print("=" * 70)

    # 1. Upsert sources
    print(f"\n  Upserting {len(collector.sources)} sources...")
    source_records = list(collector.sources.values())
    results = client.batch_upsert("sources", source_records, "jurisdiction,institution,dataset")
    for r in results:
        key = (r["jurisdiction"], r["institution"], r["dataset"])
        collector.source_ids[key] = r["id"]
    print(f"    Done - {len(results)} sources")

    # 2. Upsert strata
    print(f"\n  Upserting {len(collector.strata)} strata...")
    strata_records = list(collector.strata.values())
    results = client.batch_upsert("strata", strata_records, "name,jurisdiction")
    for r in results:
        key = (r["name"], r["jurisdiction"])
        collector.stratum_ids[key] = r["id"]
    print(f"    Done - {len(results)} strata")

    # 3. Upsert targets (resolve IDs first)
    print(f"\n  Preparing {len(collector.targets)} targets...")
    target_records = []
    for t in collector.targets:
        source_id = collector.source_ids.get(t["_source_key"])
        stratum_id = collector.stratum_ids.get(t["_stratum_key"])
        if not source_id or not stratum_id:
            print(f"    Warning: Missing ID for target {t['variable']}")
            continue
        target_records.append({
            "source_id": source_id,
            "stratum_id": stratum_id,
            "variable": t["variable"],
            "value": t["value"],
            "target_type": t["target_type"],
            "period": t["period"],
            "notes": t["notes"]
        })

    print(f"  Upserting {len(target_records)} targets...")
    results = client.batch_upsert("targets", target_records,
                                  "source_id,stratum_id,variable,period",
                                  chunk_size=500)
    print(f"    Done - {len(results)} targets")


def main():
    print("=" * 70)
    print("BATCH LOADING POLICYENGINE CALIBRATION TARGETS TO SUPABASE")
    print("=" * 70)

    start_time = time.time()
    client = BatchSupabaseClient(SUPABASE_URL, SUPABASE_KEY)
    collector = TargetCollector()

    # Collect all targets
    collect_irs_income_targets(collector)
    collect_benefit_spending_targets(collector)
    collect_healthcare_targets(collector)
    collect_tax_targets(collector)
    collect_eitc_targets(collector)
    collect_medicaid_targets(collector)
    collect_snap_targets(collector)
    collect_aca_targets(collector)
    collect_population_targets(collector)
    collect_age_state_targets(collector)
    collect_agi_state_targets(collector)
    collect_real_estate_tax_targets(collector)
    collect_soi_targets(collector)
    collect_spm_agi_targets(collector)
    collect_healthcare_age_targets(collector)
    collect_census_projection_targets(collector)

    # Summary before insert
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"  Sources: {len(collector.sources)}")
    print(f"  Strata: {len(collector.strata)}")
    print(f"  Targets: {len(collector.targets)}")

    # Batch insert
    batch_insert_all(collector, client)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"DONE! Completed in {elapsed:.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
