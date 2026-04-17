"""
Load calibration targets from Supabase.

Provides SupabaseTargetLoader for loading PE calibration targets from the
microplex Supabase schema and mapping them to CPS columns for calibration.
"""

import os
from typing import Any

import requests


class SupabaseTargetLoader:
    """Load calibration targets from Supabase."""

    # Mapping from Supabase variable names to CPS column names
    CPS_COLUMN_MAP = {
        # IRS Income targets
        "employment_income": "employment_income",
        "self_employment_income": "self_employment_income",
        "dividend_income": "dividend_income",
        "interest_income": "interest_income",
        "rental_income": "rental_income",
        "social_security": "social_security",
        "unemployment_compensation": "unemployment_compensation",
        "taxable_pension_income": "taxable_pension_income",
        "tax_exempt_pension_income": "tax_exempt_pension_income",
        "long_term_capital_gains": "long_term_capital_gains",
        "short_term_capital_gains": "short_term_capital_gains",
        "partnership_s_corp_income": "partnership_s_corp_income",
        "farm_income": "farm_income",
        "alimony_income": "alimony_income",
        # Benefit spending targets
        "snap_spending": "snap",
        "ssi_spending": "ssi",
        "eitc_spending": "eitc",
        "social_security_spending": "social_security",
        "unemployment_spending": "unemployment_compensation",
        # Benefit enrollment/count targets
        "medicaid_enrollment": "medicaid",
        "aca_enrollment": "aca",
        "snap_households": "snap",
        # Healthcare targets
        "health_insurance_premiums": "health_insurance_premiums",
        "other_medical_expenses": "medical_expenses",
    }

    # State FIPS to abbreviation
    STATE_FIPS = {
        "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca",
        "08": "co", "09": "ct", "10": "de", "11": "dc", "12": "fl",
        "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in",
        "19": "ia", "20": "ks", "21": "ky", "22": "la", "23": "me",
        "24": "md", "25": "ma", "26": "mi", "27": "mn", "28": "ms",
        "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh",
        "34": "nj", "35": "nm", "36": "ny", "37": "nc", "38": "nd",
        "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri",
        "45": "sc", "46": "sd", "47": "tn", "48": "tx", "49": "ut",
        "50": "vt", "51": "va", "53": "wa", "54": "wv", "55": "wi",
        "56": "wy"
    }

    def __init__(self, url: str = None, key: str = None, schema: str = "microplex"):
        """Initialize the loader.

        Args:
            url: Supabase URL. Defaults to SUPABASE_URL env var.
            key: Supabase key. Defaults to COSILICO_SUPABASE_SERVICE_KEY env var.
            schema: Schema to use. Defaults to 'microplex'.
        """
        self.url = url or os.environ.get(
            "SUPABASE_URL",
            "https://nsupqhfchdtqclomlrgs.supabase.co"
        )
        self.key = key or os.environ.get(
            "COSILICO_SUPABASE_SERVICE_KEY",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5zdXBxaGZjaGR0cWNsb21scmdzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NjkzMTEwOCwiZXhwIjoyMDgyNTA3MTA4fQ.IZX2C6dM6CCuxzBeg3zoZSA31p_jy9XLjdxjaE126BU"
        )
        self.base_url = f"{self.url}/rest/v1"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }
        self._cache = {}

    def _get(self, endpoint: str, params: dict = None, paginate: bool = True) -> list[dict]:
        """Make a GET request to Supabase with optional pagination.

        Args:
            endpoint: API endpoint.
            params: Query parameters.
            paginate: If True, fetch all results using pagination.

        Returns:
            List of result dicts.
        """
        url = f"{self.base_url}/{endpoint}"
        params = params or {}

        if not paginate:
            resp = requests.get(url, headers=self.headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        # Paginate to get all results
        all_results = []
        offset = 0
        limit = 1000  # Supabase default max

        while True:
            page_params = {**params, "limit": limit, "offset": offset}
            resp = requests.get(url, headers=self.headers, params=page_params, timeout=30)
            resp.raise_for_status()
            results = resp.json()

            if not results:
                break

            all_results.extend(results)
            offset += limit

            # If we got fewer than limit, we're done
            if len(results) < limit:
                break

        return all_results

    def load_all(self, period: int = None) -> list[dict]:
        """Load all targets with source and stratum info.

        Args:
            period: Optional year to filter by.

        Returns:
            List of target dicts with nested source and stratum info.
        """
        # Use PostgREST's embedded resources to join
        params = {
            "select": "id,variable,value,target_type,period,notes,source:sources(id,name,institution),stratum:strata(id,name,jurisdiction)",
        }
        if period:
            params["period"] = f"eq.{period}"

        return self._get("targets", params)

    def load_by_institution(self, institution: str, period: int = None) -> list[dict]:
        """Load targets from a specific institution.

        Args:
            institution: Institution name (e.g., 'IRS', 'Census', 'USDA').
            period: Optional year to filter by.

        Returns:
            List of target dicts.
        """
        # First get source IDs for this institution
        sources = self._get("sources", {"institution": f"eq.{institution}"})
        source_ids = [s["id"] for s in sources]

        if not source_ids:
            return []

        # Filter targets by source IDs
        params = {
            "select": "id,variable,value,target_type,period,notes,source:sources(id,name,institution),stratum:strata(id,name,jurisdiction)",
            "source_id": f"in.({','.join(source_ids)})",
        }
        if period:
            params["period"] = f"eq.{period}"

        return self._get("targets", params)

    def load_by_period(self, period: int) -> list[dict]:
        """Load targets for a specific year.

        Args:
            period: Year to filter by.

        Returns:
            List of target dicts.
        """
        return self.load_all(period=period)

    def get_cps_column_map(self) -> dict[str, str]:
        """Get the mapping from Supabase variable names to CPS columns.

        Returns:
            Dict mapping variable -> CPS column name.
        """
        return self.CPS_COLUMN_MAP.copy()

    def _parse_jurisdiction(self, jurisdiction: str) -> str | None:
        """Parse jurisdiction to get state code if applicable.

        Args:
            jurisdiction: Jurisdiction string (e.g., 'us', 'us-ca', 'us-06').

        Returns:
            State abbreviation if state-level, None for national.
        """
        if jurisdiction == "us" or jurisdiction == "us-national":
            return None

        # Handle us-XX format (state abbrev)
        if jurisdiction.startswith("us-") and len(jurisdiction) == 5:
            state = jurisdiction[3:].lower()
            if len(state) == 2:
                return state

        # Handle us-FIPS format
        if jurisdiction.startswith("us-") and len(jurisdiction) == 5:
            fips = jurisdiction[3:]
            return self.STATE_FIPS.get(fips)

        return None

    def build_calibration_constraints(
        self,
        period: int = 2024,
        include_states: bool = False,
        target_types: list[str] = None
    ) -> dict[str, float]:
        """Build calibration constraint dict from Supabase targets.

        Args:
            period: Year to get targets for.
            include_states: Whether to include state-level targets.
            target_types: List of target types to include ('amount', 'count').
                         Defaults to all.

        Returns:
            Dict mapping CPS column name to target value.
        """
        targets = self.load_all(period=period)
        constraints = {}

        for target in targets:
            variable = target["variable"]
            value = target["value"]
            target_type = target.get("target_type", "amount")
            stratum = target.get("stratum", {})
            jurisdiction = stratum.get("jurisdiction", "us")

            # Filter by target type
            if target_types and target_type not in target_types:
                continue

            # Map variable to CPS column
            cps_col = self.CPS_COLUMN_MAP.get(variable)
            if not cps_col:
                continue

            # Handle national vs state targets
            state = self._parse_jurisdiction(jurisdiction)

            if state and include_states:
                # State-level target: append state code
                key = f"{cps_col}_{state}"
                constraints[key] = value
            elif not state:
                # National target
                # Avoid duplicates (prefer first encountered)
                if cps_col not in constraints:
                    constraints[cps_col] = value

        return constraints

    def get_summary(self) -> dict[str, Any]:
        """Get summary of available targets in Supabase.

        Returns:
            Dict with counts by institution, variable, etc.
        """
        targets = self.load_all()

        by_institution = {}
        by_variable = {}
        by_type = {}

        for t in targets:
            # By institution
            inst = t.get("source", {}).get("institution", "Unknown")
            by_institution[inst] = by_institution.get(inst, 0) + 1

            # By variable
            var = t["variable"]
            by_variable[var] = by_variable.get(var, 0) + 1

            # By type
            tt = t.get("target_type", "amount")
            by_type[tt] = by_type.get(tt, 0) + 1

        return {
            "total": len(targets),
            "by_institution": by_institution,
            "by_variable": by_variable,
            "by_type": by_type,
        }
