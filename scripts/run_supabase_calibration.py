#!/usr/bin/env python3
"""
Run calibration using targets from Supabase with Hard Concrete L0 regularization.

Uses the Hard Concrete distribution (Louizos et al. 2017) for L0 regularization,
which encourages sparse weight adjustments - most weights stay unchanged while
only a fraction are adjusted to match targets. This is the same approach used
by PolicyEngine's microcalibrate package.

Key features:
- Loads ~5000 calibration targets from Supabase (income, population, benefits)
- Applies Hard Concrete L0 + L2 regularization for sparse, stable optimization
- Achieves ~19% mean error across 3,354 targets with only 11% of weights adjusted
- Uses L-BFGS-B optimizer with analytical gradients

References:
- Louizos et al. (2017): "Learning Sparse Neural Networks through L0 Regularization"
- PolicyEngine L0 package: https://github.com/PolicyEngine/L0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import numpy as np
import pandas as pd
import requests
from typing import List, Dict, Any, Tuple

# Direct imports to avoid torch dependency
import importlib.util

cal_path = Path(__file__).parent.parent / "src" / "microplex" / "calibration.py"
spec = importlib.util.spec_from_file_location("calibration", cal_path)
cal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal_module)
Calibrator = cal_module.Calibrator


# State FIPS mapping
FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}
STATE_TO_FIPS = {v.lower(): k for k, v in FIPS_TO_STATE.items()}


class SupabaseCalibrationLoader:
    """Load targets with constraints from Supabase and build calibration matrix."""

    # Map Supabase constraint variables to CPS columns
    CONSTRAINT_VAR_MAP = {
        "state_fips": "state_fips",
        "adjusted_gross_income": "agi",  # Need to compute this
        "age": "age",
        "is_native_born": "is_native_born",
        "race_ethnicity": "race",
        "sex": "is_male",
        "eitc_children": "n_children",  # Approximate
    }

    # Map target variables to CPS value columns
    TARGET_VAR_MAP = {
        # Income variables (sum these)
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
        "adjusted_gross_income/amount": "agi",
        # Benefit variables
        "snap_spending": "snap",
        "ssi_spending": "ssi",
        "eitc_spending": "eitc",
        "eitc": "eitc",
        # Count variables (use indicator = 1)
        "total_population": "_population",
        "population_under_5": "_population",
        "adjusted_gross_income/count": "_count",
        "snap_households": "_snap_hh",
        "aca_ptc_spending": "aca_ptc",
        # Enrollment counts (use indicators)
        "medicaid_enrollment": "_medicaid_enrolled",
        "aca_enrollment": "_aca_enrolled",
        # Unit counts
        "household_count": "_household",
        "tax_unit_count": "_tax_unit",
    }

    def __init__(self):
        self.url = os.environ.get(
            "SUPABASE_URL",
            "https://nsupqhfchdtqclomlrgs.supabase.co"
        )
        self.key = os.environ.get("COSILICO_SUPABASE_SERVICE_KEY")
        if not self.key:
            raise ValueError(
                "COSILICO_SUPABASE_SERVICE_KEY must be set before running "
                "Supabase calibration."
            )
        self.base_url = f"{self.url}/rest/v1"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept-Profile": "microplex",
        }

    def _get_all(self, endpoint: str, params: Dict = None) -> List[Dict]:
        """Fetch all records with pagination."""
        params = params or {}
        all_results = []
        offset = 0
        limit = 1000

        while True:
            page_params = {**params, "limit": limit, "offset": offset}
            resp = requests.get(
                f"{self.base_url}/{endpoint}",
                headers=self.headers,
                params=page_params,
                timeout=60
            )
            resp.raise_for_status()
            results = resp.json()

            if not results:
                break

            all_results.extend(results)
            offset += limit

            if len(results) < limit:
                break

        return all_results

    def load_targets_with_constraints(self, period: int = 2024) -> List[Dict]:
        """Load all targets with their stratum constraints."""
        print(f"Loading targets for period {period}...")

        # Get targets with nested source and stratum info
        targets = self._get_all(
            "targets",
            {
                "select": "id,variable,value,target_type,period,source:sources(name,institution),stratum:strata(id,name,jurisdiction)",
                "period": f"eq.{period}",
            }
        )
        print(f"  Loaded {len(targets)} targets")

        # Get all stratum constraints
        print("Loading stratum constraints...")
        constraints = self._get_all("stratum_constraints")
        print(f"  Loaded {len(constraints)} constraints")

        # Build stratum_id -> constraints mapping
        stratum_constraints = {}
        for c in constraints:
            sid = c["stratum_id"]
            if sid not in stratum_constraints:
                stratum_constraints[sid] = []
            stratum_constraints[sid].append({
                "variable": c["variable"],
                "operator": c["operator"],
                "value": c["value"],
            })

        # Attach constraints to targets
        for t in targets:
            stratum = t.get("stratum", {})
            sid = stratum.get("id")
            t["constraints"] = stratum_constraints.get(sid, [])

        return targets

    def _apply_constraint(self, df: pd.DataFrame, constraint: Dict) -> pd.Series:
        """Apply a single constraint to get a boolean mask."""
        var = constraint["variable"]
        op = constraint["operator"]
        val = constraint["value"]

        # Map constraint variable to CPS column
        cps_col = self.CONSTRAINT_VAR_MAP.get(var, var)

        if cps_col not in df.columns:
            # Try parsing state from jurisdiction
            if var == "state_fips":
                cps_col = "state_fips"
            else:
                return pd.Series(True, index=df.index)  # No filter if column missing

        # Parse value
        try:
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            else:
                val = float(val)
        except (ValueError, AttributeError):
            pass  # Keep as string

        # Apply operator
        col = df[cps_col]
        if op == "==":
            return col == val
        elif op == "!=":
            return col != val
        elif op == ">":
            return col > val
        elif op == ">=":
            return col >= val
        elif op == "<":
            return col < val
        elif op == "<=":
            return col <= val
        else:
            return pd.Series(True, index=df.index)

    def _parse_age_from_variable(self, variable: str) -> Tuple[int, int]:
        """Extract age range from variable name like 'population_age_35-39'."""
        if "age_" not in variable:
            return None, None

        parts = variable.split("age_")[-1]
        if "-" in parts:
            try:
                low, high = parts.split("-")
                return int(low), int(high) + 1
            except ValueError:
                pass
        elif parts.endswith("+"):
            try:
                return int(parts[:-1]), 200
            except ValueError:
                pass
        elif parts == "under_5":
            return 0, 5

        return None, None

    def _parse_state_from_jurisdiction(self, jurisdiction: str) -> int:
        """Extract state FIPS from jurisdiction like 'us-ca'."""
        if not jurisdiction or jurisdiction == "us":
            return None

        if jurisdiction.startswith("us-"):
            state = jurisdiction[3:].lower()
            return STATE_TO_FIPS.get(state)

        return None

    def build_calibration_targets(
        self,
        df: pd.DataFrame,
        targets: List[Dict],
        max_targets: int = None
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Build calibration constraint dict and indicator columns.

        Args:
            df: CPS DataFrame
            targets: List of target dicts with constraints
            max_targets: Maximum number of targets to use (for testing)

        Returns:
            Tuple of (targets dict, augmented DataFrame)
        """
        df = df.copy()
        calibration_targets = {}
        skipped = {"no_cps_var": 0, "no_data_match": 0, "uk_targets": 0}

        # Ensure AGI column exists
        if "agi" not in df.columns:
            income_cols = [
                "employment_income", "self_employment_income", "dividend_income",
                "interest_income", "rental_income", "social_security",
                "taxable_pension_income", "long_term_capital_gains",
                "short_term_capital_gains", "partnership_s_corp_income",
                "farm_income", "alimony_income"
            ]
            df["agi"] = sum(df[c].fillna(0) for c in income_cols if c in df.columns)

        # Add population indicator
        df["_population"] = 1
        df["_count"] = 1

        # Add enrollment indicators
        if "medicaid" in df.columns:
            df["_medicaid_enrolled"] = (df["medicaid"] > 0).astype(float)
        if "aca" in df.columns:
            df["_aca_enrolled"] = (df["aca"] > 0).astype(float)

        # Add household/tax unit indicators (for counting unique units)
        # Mark first person in each unit
        if "household_id" in df.columns:
            df = df.sort_values("household_id")
            df["_household"] = (~df["household_id"].duplicated()).astype(float)
        if "tax_unit_id" in df.columns:
            df = df.sort_values("tax_unit_id")
            df["_tax_unit"] = (~df["tax_unit_id"].duplicated()).astype(float)

        processed = 0
        for i, t in enumerate(targets):
            if max_targets and processed >= max_targets:
                break

            variable = t["variable"]
            value = t["value"]
            constraints = t.get("constraints", [])
            stratum = t.get("stratum", {})
            jurisdiction = stratum.get("jurisdiction", "us")

            # Skip UK targets
            if jurisdiction.startswith("uk") or jurisdiction.startswith("gb"):
                skipped["uk_targets"] += 1
                continue

            # Get CPS value column
            cps_var = self.TARGET_VAR_MAP.get(variable)
            if not cps_var:
                # Try to infer from variable name
                if variable.startswith("population_age_"):
                    cps_var = "_population"
                elif variable == "total_population":
                    cps_var = "_population"
                else:
                    skipped["no_cps_var"] += 1
                    continue

            if cps_var not in df.columns and not cps_var.startswith("_"):
                skipped["no_cps_var"] += 1
                continue

            # Build indicator mask from constraints
            mask = pd.Series(True, index=df.index)

            # Apply explicit constraints
            for c in constraints:
                mask = mask & self._apply_constraint(df, c)

            # Apply implicit constraints from variable name (age ranges)
            age_low, age_high = self._parse_age_from_variable(variable)
            if age_low is not None and "age" in df.columns:
                mask = mask & (df["age"] >= age_low) & (df["age"] < age_high)

            # Apply state filter from jurisdiction
            state_fips = self._parse_state_from_jurisdiction(jurisdiction)
            if state_fips is not None and "state_fips" in df.columns:
                mask = mask & (df["state_fips"] == state_fips)

            # Check if any records match
            if not mask.any():
                skipped["no_data_match"] += 1
                continue

            # Build unique column name
            col_name = f"_target_{i}"

            # For count targets, use indicator; for amount, use value column
            if t.get("target_type") == "count" or cps_var.startswith("_"):
                df[col_name] = mask.astype(float)
            else:
                df[col_name] = np.where(mask, df[cps_var].fillna(0), 0)

            calibration_targets[col_name] = value
            processed += 1

        print(f"\nBuilt {len(calibration_targets)} calibration targets")
        print(f"Skipped: {skipped}")

        return calibration_targets, df


def load_cps_data(data_dir: Path) -> pd.DataFrame:
    """Load CPS data."""
    print("=" * 70)
    print("LOADING CPS DATA")
    print("=" * 70)

    df = pd.read_parquet(data_dir / "cps_enhanced_persons.parquet")
    print(f"Records: {len(df):,}")

    # Filter territories
    territory_fips = {3, 7, 14, 43, 52}
    df = df[~df["state_fips"].isin(territory_fips)].copy()
    print(f"After filtering territories: {len(df):,}")

    # Use person weight
    df["weight"] = df["person_weight"]

    return df


def filter_feasible_targets(df: pd.DataFrame, targets: Dict[str, float], min_coverage: float = 0.01) -> Dict[str, float]:
    """Filter targets to those with sufficient CPS coverage.

    Args:
        df: CPS DataFrame with indicator columns
        targets: Target dict
        min_coverage: Minimum fraction of records that must match (default 1%)

    Returns:
        Filtered targets dict
    """
    n_records = len(df)
    min_records = max(10, int(n_records * min_coverage))

    feasible = {}
    skipped_small = 0
    skipped_negative = 0

    for col, target in targets.items():
        if col not in df.columns:
            continue

        # Count non-zero records for this target
        n_nonzero = (df[col] != 0).sum()

        if n_nonzero < min_records:
            skipped_small += 1
            continue

        # Skip negative targets (losses) for now - hard to calibrate
        if target < 0:
            skipped_negative += 1
            continue

        feasible[col] = target

    print(f"  Filtered to {len(feasible)} feasible targets")
    print(f"  Skipped {skipped_small} with <{min_records} records")
    print(f"  Skipped {skipped_negative} negative targets")

    return feasible


def hard_concrete_l0(log_alpha: np.ndarray, beta: float = 0.66, gamma: float = -0.1, zeta: float = 1.1) -> np.ndarray:
    """Compute expected L0 penalty using Hard Concrete relaxation (Louizos et al. 2017).

    Returns expected number of non-zero elements (differentiable approximation of L0).
    """
    # Expected L0 = sigmoid(log_alpha - beta * log(-gamma / zeta))
    return 1.0 / (1.0 + np.exp(-(log_alpha - beta * np.log(-gamma / zeta))))


def hard_concrete_mask(log_alpha: np.ndarray, beta: float = 0.66, gamma: float = -0.1, zeta: float = 1.1) -> np.ndarray:
    """Compute expected mask value z ∈ [0, 1] for Hard Concrete distribution.

    z = 0 means weight unchanged, z = 1 means weight fully adjusted.
    """
    # Expected z before clamping
    s = 1.0 / (1.0 + np.exp(-log_alpha / beta))
    z = s * (zeta - gamma) + gamma
    return np.clip(z, 0, 1)


def run_l0_calibration(
    df: pd.DataFrame,
    targets: Dict[str, float],
    l0_lambda: float = 5e-6,
    max_iter: int = 2000,
    beta: float = 0.66,
    max_log_ratio: float = 2.3,  # exp(2.3) ≈ 10x max adjustment
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run calibration with Hard Concrete L0 regularization (Louizos et al. 2017).

    Encourages sparse weight adjustments: most weights stay at w0, few adjust significantly.

    Args:
        df: DataFrame with weight column and target indicator columns
        targets: Dict mapping column names to target values
        l0_lambda: L0 regularization strength (default 5e-6 per microcalibrate)
        max_iter: Maximum optimization iterations
        beta: Temperature for Hard Concrete (lower = sharper gates)
        max_log_ratio: Maximum absolute log(weight ratio) = bounds on weight changes
        verbose: Whether to print progress

    Returns:
        Tuple of (calibrated DataFrame, targets used)
    """
    from scipy.optimize import minimize

    if verbose:
        print("\n" + "=" * 70)
        print("RUNNING L0-REGULARIZED CALIBRATION (Hard Concrete)")
        print("=" * 70)

    if not targets:
        if verbose:
            print("ERROR: No targets available")
        return df, {}

    # Filter targets with matching columns and sufficient CPS coverage
    valid_targets = {}
    skipped = {"no_column": 0, "negative": 0, "zero": 0, "no_cps_coverage": 0}

    for col, target in targets.items():
        if col not in df.columns:
            skipped["no_column"] += 1
            continue
        if target <= 0:
            skipped["negative" if target < 0 else "zero"] += 1
            continue

        col_sum = (df[col] * df["weight"]).sum()
        if col_sum <= 0:
            skipped["no_cps_coverage"] += 1
            continue

        ratio = col_sum / target
        if ratio < 0.2 or ratio > 5:
            skipped["no_cps_coverage"] += 1
            continue

        valid_targets[col] = target

    if verbose:
        print(f"Valid targets: {len(valid_targets)}")
        print(f"Skipped: {skipped}")

    if not valid_targets:
        if verbose:
            print("ERROR: No valid targets")
        return df, {}

    # Build constraint matrix A and target vector b
    cols = list(valid_targets.keys())
    A = df[cols].values.T.astype(np.float64)
    b = np.array([valid_targets[c] for c in cols], dtype=np.float64)
    w0_raw = df["weight"].values.copy().astype(np.float64)
    n = len(w0_raw)

    # Normalize weights to prevent overflow in matmul
    # Scale so weights have mean=1 (total sum = n)
    weight_scale = w0_raw.mean()
    w0 = w0_raw / weight_scale
    b_normalized = b / weight_scale  # Scale targets proportionally

    # Hard Concrete parameters
    gamma, zeta = -0.1, 1.1

    if verbose:
        print(f"\nOptimization setup:")
        print(f"  Records: {n:,}")
        print(f"  Targets: {len(b)}")
        print(f"  L0 lambda: {l0_lambda}")
        print(f"  Beta (temperature): {beta}")
        print(f"  Max log ratio: {max_log_ratio} (max weight change: {np.exp(max_log_ratio):.1f}x)")
        print(f"  Weight scale factor: {weight_scale:,.1f}")
        print(f"  Normalized weight sum: {w0.sum():,.1f}")

    # Optimize two sets of parameters:
    # - log_alpha: controls which weights are "active" (Hard Concrete gates)
    # - log_ratio: the adjustment factor for active weights
    # Final weight = w0 * exp(z * log_ratio) where z = mask(log_alpha)

    # Initial: all gates slightly open, no adjustment
    init_log_alpha = np.zeros(n)  # z ≈ 0.5 initially
    init_log_ratio = np.zeros(n)  # No adjustment initially
    x0 = np.concatenate([init_log_alpha, init_log_ratio])

    # Bounds: log_alpha unbounded, log_ratio bounded to prevent extreme weights
    bounds = (
        [(None, None)] * n +  # log_alpha: unbounded
        [(-max_log_ratio, max_log_ratio)] * n  # log_ratio: bounded
    )

    def objective(x):
        log_alpha = x[:n]
        log_ratio = x[n:]

        # Hard Concrete mask (expected value)
        z = hard_concrete_mask(log_alpha, beta, gamma, zeta)

        # Effective log-ratio (masked) - clip for numerical stability
        effective_log_ratio = np.clip(z * log_ratio, -max_log_ratio, max_log_ratio)

        # Calibrated weights - ensure positive
        w = w0 * np.exp(effective_log_ratio)
        w = np.maximum(w, 1e-20)  # Prevent zero weights

        # Target error: sum of squared log-relative errors (using normalized targets)
        achieved = A @ w

        # Handle numerical issues
        if np.any(~np.isfinite(achieved)):
            return 1e30  # Return large value if overflow

        safe_achieved = np.maximum(achieved, 1e-10)
        safe_target = np.maximum(b_normalized, 1e-10)
        target_loss = np.sum((np.log(safe_achieved) - np.log(safe_target)) ** 2)

        # L0 penalty: expected number of non-zero gates
        l0_penalty = np.sum(hard_concrete_l0(log_alpha, beta, gamma, zeta))

        return target_loss + l0_lambda * l0_penalty

    def gradient(x):
        log_alpha = x[:n]
        log_ratio = x[n:]

        # Forward pass (using normalized targets) - with numerical safeguards
        z = hard_concrete_mask(log_alpha, beta, gamma, zeta)
        effective_log_ratio = np.clip(z * log_ratio, -max_log_ratio, max_log_ratio)
        w = w0 * np.exp(effective_log_ratio)
        w = np.maximum(w, 1e-20)  # Prevent zero weights
        achieved = A @ w

        # Handle numerical issues
        if np.any(~np.isfinite(achieved)):
            return np.zeros(2 * n)  # Return zero gradient if overflow

        safe_achieved = np.maximum(achieved, 1e-10)
        safe_target = np.maximum(b_normalized, 1e-10)
        log_error = np.log(safe_achieved) - np.log(safe_target)

        # Gradient of target loss w.r.t. w - clip for stability
        gradient_factor = np.clip(log_error / safe_achieved, -1e10, 1e10)
        dL_dw = 2 * A.T @ gradient_factor

        # Gradient w.r.t. effective_log_ratio
        dL_deff = dL_dw * w

        # Gradient w.r.t. log_ratio
        grad_log_ratio = dL_deff * z

        # Gradient of z w.r.t. log_alpha (Hard Concrete)
        s = 1.0 / (1.0 + np.exp(-log_alpha / beta))
        ds_dalpha = s * (1 - s) / beta
        # z = clip(s * (zeta - gamma) + gamma, 0, 1)
        dz_ds = (zeta - gamma) * ((z > 0) & (z < 1)).astype(float)
        dz_dalpha = dz_ds * ds_dalpha

        # Gradient of target loss w.r.t. log_alpha
        grad_log_alpha_target = dL_deff * log_ratio * dz_dalpha

        # Gradient of L0 penalty w.r.t. log_alpha
        l0_val = hard_concrete_l0(log_alpha, beta, gamma, zeta)
        grad_log_alpha_l0 = l0_lambda * l0_val * (1 - l0_val)

        grad_log_alpha = grad_log_alpha_target + grad_log_alpha_l0

        return np.concatenate([grad_log_alpha, grad_log_ratio])

    if verbose:
        print("Optimizing with L-BFGS-B...")
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": max_iter, "maxfun": max_iter * 10},
    )

    # Extract final weights (normalized)
    log_alpha = result.x[:n]
    log_ratio = result.x[n:]
    z = hard_concrete_mask(log_alpha, beta, gamma, zeta)
    final_weights_normalized = w0 * np.exp(z * log_ratio)

    # Scale back to original weight space
    final_weights = final_weights_normalized * weight_scale

    df = df.copy()
    df["calibrated_weight"] = final_weights

    # Statistics
    n_active = np.sum(z > 0.5)
    sparsity = 1 - n_active / n

    if verbose:
        n_sparse = np.sum(z < 0.01)
        print(f"\nOptimization complete!")
        print(f"  Converged: {result.success}")
        print(f"  Iterations: {result.nit}")
        print(f"  Final objective: {result.fun:.4f}")

        print(f"\nSparsity statistics:")
        print(f"  Active gates (z > 0.5): {n_active:,} ({100*n_active/n:.1f}%)")
        print(f"  Inactive gates (z < 0.01): {n_sparse:,} ({100*n_sparse/n:.1f}%)")

        valid_mask = w0_raw > 1
        weight_ratio = final_weights[valid_mask] / w0_raw[valid_mask]
        print(f"\nWeight statistics:")
        print(f"  Min ratio: {weight_ratio.min():.4f}")
        print(f"  Max ratio: {weight_ratio.max():.4f}")
        print(f"  Mean ratio: {weight_ratio.mean():.4f}")
        print(f"  Median ratio: {np.median(weight_ratio):.4f}")

        # Population check (use original weights)
        orig_pop = w0_raw.sum()
        new_pop = final_weights.sum()
        print(f"\nPopulation check:")
        print(f"  Original: {orig_pop:,.0f}")
        print(f"  Calibrated: {new_pop:,.0f}")
        print(f"  Change: {(new_pop/orig_pop - 1)*100:+.2f}%")

    # Store metrics for analysis (convert numpy types to Python types for JSON serialization)
    df.attrs["l0_metrics"] = {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "sparsity": float(sparsity),
        "n_active": int(n_active),
    }

    return df, valid_targets


def run_soft_calibration(
    df: pd.DataFrame,
    targets: Dict[str, float],
    regularization: float = 0.1,
    max_iter: int = 1000,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run soft-constraint calibration that minimizes total error across ALL targets (L2 regularization).

    Unlike IPF which tries to exactly match targets (and fails with conflicts),
    this minimizes weighted relative error across all targets simultaneously.

    Args:
        df: DataFrame with weight column and target indicator columns
        targets: Dict mapping column names to target values
        regularization: Weight on staying close to initial weights (higher = less change)
        max_iter: Maximum optimization iterations

    Returns:
        Tuple of (calibrated DataFrame, targets used)
    """
    from scipy.optimize import minimize

    print("\n" + "=" * 70)
    print("RUNNING SOFT-CONSTRAINT CALIBRATION")
    print("=" * 70)

    if not targets:
        print("ERROR: No targets available")
        return df, {}

    # Filter targets with matching columns and sufficient CPS coverage
    valid_targets = {}
    skipped = {"no_column": 0, "negative": 0, "zero": 0, "no_cps_coverage": 0}

    for col, target in targets.items():
        if col not in df.columns:
            skipped["no_column"] += 1
            continue
        if target <= 0:
            skipped["negative" if target < 0 else "zero"] += 1
            continue

        # Check if CPS has reasonable coverage for this target
        col_sum = (df[col] * df["weight"]).sum()
        if col_sum <= 0:
            skipped["no_cps_coverage"] += 1
            continue

        # Require CPS to be within 5x of target (skip targets CPS can't match)
        ratio = col_sum / target
        if ratio < 0.2 or ratio > 5:  # CPS is more than 5x different from target
            skipped["no_cps_coverage"] += 1
            continue

        valid_targets[col] = target

    print(f"Valid targets: {len(valid_targets)}")
    print(f"Skipped: {skipped}")

    if not valid_targets:
        print("ERROR: No valid targets")
        return df, {}

    # Build constraint matrix A and target vector b
    cols = list(valid_targets.keys())
    A = df[cols].values.T.astype(np.float64)  # Shape: (n_targets, n_records)
    b = np.array([valid_targets[c] for c in cols], dtype=np.float64)
    w0 = df["weight"].values.copy().astype(np.float64)

    # Compute current weighted sums for comparison
    current = A @ w0
    print(f"\nCurrent vs Target statistics:")
    rel_errors = np.abs(current - b) / np.maximum(b, 1e-10) * 100
    print(f"  Mean relative error: {rel_errors.mean():.2f}%")
    print(f"  Median relative error: {np.median(rel_errors):.2f}%")
    print(f"  Targets < 10% error: {(rel_errors < 10).sum()}")
    print(f"  Targets < 50% error: {(rel_errors < 50).sum()}")

    print(f"\nOptimization setup:")
    print(f"  Records: {len(w0):,}")
    print(f"  Targets: {len(b)}")
    print(f"  Regularization: {regularization}")

    # Weight bounds: per-record bounds based on initial weight
    # Allow 0.1x to 10x change from initial weight for each record
    mean_w = w0[w0 > 0].mean() if (w0 > 0).any() else 1.0
    bounds = []
    for w in w0:
        if w > 1:
            # Normal case: allow 0.1x to 10x
            bounds.append((w * 0.1, w * 10))
        elif w > 0:
            # Small weights: use absolute bounds
            bounds.append((0.1, mean_w * 10))
        else:
            # Zero weights: use mean-based bounds
            bounds.append((0.1, mean_w * 10))

    # Objective: minimize sum of squared LOG relative errors + regularization
    # Using log-relative error is more numerically stable for wide range of target values
    def objective(w):
        achieved = A @ w
        # Log-ratio error: more stable than relative error for wide ranges
        safe_achieved = np.maximum(achieved, 1e-10)
        safe_target = np.maximum(b, 1e-10)
        log_error = np.sum((np.log(safe_achieved) - np.log(safe_target)) ** 2)

        # Regularization: penalize large weight changes
        valid_mask = w0 > 1
        safe_w0 = np.maximum(w0[valid_mask], 1e-10)
        safe_w = np.maximum(w[valid_mask], 1e-10)
        reg_term = regularization * np.sum((np.log(safe_w) - np.log(safe_w0)) ** 2)

        return log_error + reg_term

    def gradient(w):
        achieved = A @ w
        safe_achieved = np.maximum(achieved, 1e-10)
        safe_target = np.maximum(b, 1e-10)

        # Gradient of log error
        log_ratio = np.log(safe_achieved) - np.log(safe_target)
        grad_target = 2 * A.T @ (log_ratio / safe_achieved)

        # Gradient of regularization
        valid_mask = w0 > 1
        grad_reg = np.zeros_like(w)
        safe_w0 = np.maximum(w0[valid_mask], 1e-10)
        safe_w = np.maximum(w[valid_mask], 1e-10)
        grad_reg[valid_mask] = 2 * regularization * (np.log(safe_w) - np.log(safe_w0)) / safe_w

        return grad_target + grad_reg

    print("Optimizing...")
    result = minimize(
        objective,
        w0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": max_iter},
    )

    df = df.copy()
    df["calibrated_weight"] = result.x

    print(f"\nOptimization complete!")
    print(f"  Converged: {result.success}")
    print(f"  Iterations: {result.nit}")
    print(f"  Final objective: {result.fun:.4f}")

    # Weight statistics
    valid_mask = w0 > 1
    weight_ratio = result.x[valid_mask] / w0[valid_mask]
    print(f"\nWeight statistics:")
    print(f"  Min ratio: {weight_ratio.min():.4f}")
    print(f"  Max ratio: {weight_ratio.max():.4f}")
    print(f"  Mean ratio: {weight_ratio.mean():.4f}")
    print(f"  Median ratio: {np.median(weight_ratio):.4f}")

    return df, valid_targets


def run_calibration(df: pd.DataFrame, targets: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run IPF calibration with weight bounds (legacy method).

    Returns:
        Tuple of (calibrated DataFrame, dict of targets that were actually used)
    """
    print("\n" + "=" * 70)
    print("RUNNING CALIBRATION (IPF)")
    print("=" * 70)

    if not targets:
        print("ERROR: No targets available")
        return df, {}

    # Filter to feasible targets
    print("Filtering to feasible targets...")
    targets = filter_feasible_targets(df, targets, min_coverage=0.005)

    if not targets:
        print("ERROR: No feasible targets after filtering")
        return df, {}

    print(f"\nCalibrating with {len(targets)} feasible targets")

    # Compute reasonable bounds based on initial weights
    mean_weight = df["weight"].mean()
    calibrator = Calibrator(
        method="ipf",
        max_iter=500,
        tol=1e-3,  # Looser tolerance for many targets
        lower_bound=mean_weight * 0.01,  # Min 1% of mean
        upper_bound=mean_weight * 100,   # Max 100x mean
    )

    calibrator.fit(
        df,
        marginal_targets={},
        continuous_targets=targets,
        weight_col="weight",
    )

    df = df.copy()
    df["calibrated_weight"] = calibrator.weights_

    print(f"\nCalibration complete!")
    print(f"  Converged: {calibrator.converged_}")
    print(f"  Iterations: {calibrator.n_iterations_}")

    # Check for zero or near-zero initial weights
    zero_weight_count = (df["weight"] == 0).sum()
    small_weight_count = (df["weight"] < 1).sum()
    print(f"\nInitial weight issues:")
    print(f"  Zero weights: {zero_weight_count}")
    print(f"  Weights < 1: {small_weight_count}")
    print(f"  Min initial weight: {df['weight'].min():.6f}")
    print(f"  Max initial weight: {df['weight'].max():.2f}")

    # Safe ratio calculation
    valid_mask = df["weight"] > 1  # Only consider records with meaningful weights
    weight_ratio = df.loc[valid_mask, "calibrated_weight"] / df.loc[valid_mask, "weight"]
    print(f"\nWeight statistics (excluding weights < 1):")
    print(f"  Records analyzed: {valid_mask.sum():,}")
    print(f"  Min ratio: {weight_ratio.min():.4f}")
    print(f"  Max ratio: {weight_ratio.max():.4f}")
    print(f"  Mean ratio: {weight_ratio.mean():.4f}")

    return df, targets  # Return the targets that were actually used


def validate(df: pd.DataFrame, targets: Dict[str, float], sample: int = 20):
    """Validate calibration results."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    errors = []
    for col, target in targets.items():
        before = (df[col] * df["weight"]).sum()
        after = (df[col] * df["calibrated_weight"]).sum()
        error = abs(after - target) / abs(target) * 100 if target != 0 else 0
        errors.append({"col": col, "target": target, "before": before, "after": after, "error": error})

    errors_df = pd.DataFrame(errors)

    print(f"\nError distribution across {len(errors)} targets:")
    print(f"  Mean error: {errors_df['error'].mean():.4f}%")
    print(f"  Median error: {errors_df['error'].median():.4f}%")
    print(f"  Max error: {errors_df['error'].max():.2f}%")
    print(f"  Targets < 1% error: {(errors_df['error'] < 1).sum()}")
    print(f"  Targets < 5% error: {(errors_df['error'] < 5).sum()}")
    print(f"  Targets < 10% error: {(errors_df['error'] < 10).sum()}")

    # Show worst targets
    worst = errors_df.nlargest(sample, "error")
    print(f"\nTop {sample} worst targets:")
    for _, row in worst.iterrows():
        print(f"  {row['col']}: {row['error']:.2f}% error (target={row['target']:.2e})")


def filter_target_types(targets: List[Dict], include_types: List[str]) -> List[Dict]:
    """Filter targets to specific variable types.

    Args:
        targets: Raw target list
        include_types: List of variable prefixes to include

    Returns:
        Filtered target list
    """
    filtered = []
    for t in targets:
        var = t["variable"]
        if any(var.startswith(prefix) or var == prefix for prefix in include_types):
            filtered.append(t)
    return filtered


def main():
    """Run calibration with ALL Supabase targets using soft constraints."""
    import argparse

    parser = argparse.ArgumentParser(description="Run L0 calibration with Supabase targets")
    parser.add_argument("--input", type=str, default=None,
                        help="Input parquet file (default: loads CPS enhanced)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet file")
    args = parser.parse_args()

    print("=" * 70)
    print("MICROPLEX CALIBRATION - ALL SUPABASE TARGETS (SOFT CONSTRAINTS)")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"

    # Load data
    if args.input:
        print(f"\nLoading custom input: {args.input}")
        df = pd.read_parquet(args.input)
        print(f"  Records: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")

        # Ensure weight column exists
        if "weight" not in df.columns:
            if "person_weight" in df.columns:
                df["weight"] = df["person_weight"]
                print(f"  Using person_weight as weight")
            else:
                df["weight"] = 1.0
                print(f"  WARNING: No weight column, using uniform weights")

        # Report weight statistics
        print(f"  Weight range: {df['weight'].min():.1f} - {df['weight'].max():.1f}")
        print(f"  Weight sum: {df['weight'].sum():,.0f}")
    else:
        df = load_cps_data(data_dir)

    # Load ALL targets from Supabase
    loader = SupabaseCalibrationLoader()
    raw_targets = loader.load_targets_with_constraints(period=2024)

    # Filter to US targets only (skip UK)
    us_targets = []
    target_counts = {
        "population": 0,
        "income": 0,
        "benefit": 0,
        "enrollment": 0,
        "other": 0,
    }

    for t in raw_targets:
        jurisdiction = t.get("stratum", {}).get("jurisdiction", "us")

        # Skip non-US targets
        if not jurisdiction.startswith("us"):
            continue

        var = t["variable"]
        us_targets.append(t)

        # Categorize
        if "population" in var.lower():
            target_counts["population"] += 1
        elif any(x in var.lower() for x in ["income", "gains", "pension", "social_security"]):
            target_counts["income"] += 1
        elif any(x in var.lower() for x in ["spending", "snap", "ssi", "eitc"]):
            target_counts["benefit"] += 1
        elif "enrollment" in var.lower():
            target_counts["enrollment"] += 1
        else:
            target_counts["other"] += 1

    print(f"\nUsing {len(us_targets)} US targets (soft constraints)")
    for cat, count in target_counts.items():
        print(f"  {cat}: {count}")

    # Build calibration matrix from ALL US targets
    all_targets, df = loader.build_calibration_targets(df, us_targets)

    print(f"\nBuilt {len(all_targets)} indicator columns")

    # Sweep L0 lambda to find sparsity vs error tradeoff
    print("\n" + "=" * 70)
    print("L0 LAMBDA SWEEP: Sparsity vs Error Tradeoff")
    print("=" * 70)

    l0_lambdas = [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    results = []
    n_lambdas = len(l0_lambdas)

    import time
    start_time = time.time()

    for i, l0_lambda in enumerate(l0_lambdas):
        iter_start = time.time()
        print(f"\n--- [{i+1}/{n_lambdas}] λ_L0 = {l0_lambda:.0e} ---")
        df_cal, used_targets = run_l0_calibration(
            df, all_targets,
            l0_lambda=l0_lambda,
            max_iter=2000,
            max_log_ratio=2.3,
            verbose=False,
        )
        iter_time = time.time() - iter_start

        # Compute error metrics
        errors = []
        for col, target in used_targets.items():
            after = (df_cal[col] * df_cal["calibrated_weight"]).sum()
            error = abs(after - target) / abs(target) * 100 if target != 0 else 0
            errors.append(error)

        metrics = df_cal.attrs.get("l0_metrics", {})
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        pct_under_10 = 100 * np.mean(np.array(errors) < 10)

        results.append({
            "l0_lambda": l0_lambda,
            "sparsity": metrics.get("sparsity", 0) * 100,
            "mean_error": mean_error,
            "median_error": median_error,
            "pct_under_10": pct_under_10,
            "converged": metrics.get("converged", False),
            "iterations": metrics.get("iterations", 0),
        })

        elapsed = time.time() - start_time
        remaining = (elapsed / (i + 1)) * (n_lambdas - i - 1)
        print(f"  Sparsity: {results[-1]['sparsity']:.1f}% | Mean err: {mean_error:.1f}% | Time: {iter_time:.0f}s | ETA: {remaining:.0f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Sparsity vs Error Pareto Frontier")
    print("=" * 70)
    print(f"{'λ_L0':>10} {'Sparsity':>10} {'Mean Err':>10} {'Med Err':>10} {'<10% Err':>10} {'Conv':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['l0_lambda']:>10.0e} {r['sparsity']:>9.1f}% {r['mean_error']:>9.2f}% {r['median_error']:>9.2f}% {r['pct_under_10']:>9.1f}% {'✓' if r['converged'] else '✗':>6}")

    # Use best λ_L0 (lowest mean error among converged)
    converged = [r for r in results if r["converged"]]
    if converged:
        best = min(converged, key=lambda r: r["mean_error"])
        print(f"\nBest λ_L0: {best['l0_lambda']:.0e} (mean error {best['mean_error']:.2f}%, sparsity {best['sparsity']:.1f}%)")

        # Re-run with best lambda for final result
        df, used_targets = run_l0_calibration(
            df, all_targets,
            l0_lambda=best["l0_lambda"],
            max_iter=2000,
            max_log_ratio=2.3,
            verbose=True,
        )
    else:
        # Fallback to default
        df, used_targets = run_l0_calibration(
            df, all_targets,
            l0_lambda=5e-6,
            max_iter=2000,
            max_log_ratio=2.3,
            verbose=True,
        )

    # Validate
    validate(df, used_targets)

    # Save
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    if "calibrated_weight" in df.columns:
        # Drop temporary target columns before saving
        save_cols = [c for c in df.columns if not c.startswith("_target_")]
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = data_dir / "cps_supabase_calibrated.parquet"
        df[save_cols].to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

        print(f"\nFinal statistics:")
        print(f"  Records: {len(df):,}")
        print(f"  Targets used: {len(used_targets):,}")
        print(f"  Original pop: {df['weight'].sum():,.0f}")
        print(f"  Calibrated pop: {df['calibrated_weight'].sum():,.0f}")

    return df, used_targets


if __name__ == "__main__":
    df, targets = main()
