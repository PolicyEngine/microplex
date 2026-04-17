"""Survey harmonization for multi-survey fusion.

Defines common variable schema and functions to harmonize
CPS and PUF into a stacked dataset for masked MAF training.
"""

import numpy as np
import pandas as pd

# Common variable schema for all surveys
# Each variable has: type, transform, description
COMMON_SCHEMA = {
    # Demographics
    "age": {"type": "continuous", "transform": "none", "min": 0, "max": 120},
    "is_male": {"type": "binary", "transform": "none"},
    "is_married": {"type": "binary", "transform": "none"},
    "n_children": {"type": "discrete", "transform": "none", "min": 0, "max": 10},

    # Geography (CPS only - NaN in PUF)
    "state_fips": {"type": "discrete", "transform": "none"},

    # Filing status
    "is_joint_filer": {"type": "binary", "transform": "none"},
    "is_head_of_household": {"type": "binary", "transform": "none"},

    # Wage income (both surveys, different quality)
    "employment_income": {"type": "continuous", "transform": "log1p", "min": 0},
    "self_employment_income": {"type": "continuous", "transform": "signed_log", "unbounded": True},

    # Investment income (PUF better, CPS underreports)
    "interest_income": {"type": "continuous", "transform": "log1p", "min": 0},
    "dividend_income": {"type": "continuous", "transform": "log1p", "min": 0},
    "qualified_dividend_income": {"type": "continuous", "transform": "log1p", "min": 0},  # PUF only

    # Capital gains (PUF only - NaN in CPS)
    "short_term_capital_gains": {"type": "continuous", "transform": "signed_log", "unbounded": True},
    "long_term_capital_gains": {"type": "continuous", "transform": "signed_log", "unbounded": True},
    "capital_gains_distributions": {"type": "continuous", "transform": "log1p", "min": 0},

    # Pass-through income (PUF only)
    "partnership_s_corp_income": {"type": "continuous", "transform": "signed_log", "unbounded": True},
    "rental_income": {"type": "continuous", "transform": "signed_log", "unbounded": True},

    # Retirement income (both surveys)
    "social_security": {"type": "continuous", "transform": "log1p", "min": 0},
    "pension_income": {"type": "continuous", "transform": "log1p", "min": 0},
    "ira_distributions": {"type": "continuous", "transform": "log1p", "min": 0},

    # Other income
    "unemployment_compensation": {"type": "continuous", "transform": "log1p", "min": 0},
    "farm_income": {"type": "continuous", "transform": "signed_log", "unbounded": True},
    "alimony_income": {"type": "continuous", "transform": "log1p", "min": 0},

    # Expenses (for deduction computation by RAC)
    "medical_expense": {"type": "continuous", "transform": "log1p", "min": 0},
    "childcare_expense": {"type": "continuous", "transform": "log1p", "min": 0},
    "property_tax_paid": {"type": "continuous", "transform": "log1p", "min": 0},
    "mortgage_interest_paid": {"type": "continuous", "transform": "log1p", "min": 0},
    "charitable_contributions": {"type": "continuous", "transform": "log1p", "min": 0},
    "state_local_tax_paid": {"type": "continuous", "transform": "log1p", "min": 0},
    "student_loan_interest": {"type": "continuous", "transform": "log1p", "min": 0},

    # Housing (CPS only)
    "housing_cost": {"type": "continuous", "transform": "log1p", "min": 0},
}


# CPS variable mapping to common schema
CPS_MAPPING = {
    "age": "age",
    "is_male": lambda df: (df.get("sex", 2) == 1).astype(float) if "sex" in df.columns else np.nan,
    "is_married": lambda df: df.get("marital_status", 0).isin([1, 2]).astype(float) if "marital_status" in df.columns else np.nan,
    "n_children": "own_children_in_household",
    "state_fips": "state_fips",
    "is_joint_filer": lambda df: (df.get("marital_status", 0) == 1).astype(float),
    "is_head_of_household": lambda df: (df.get("relationship_to_head", 0) == 1).astype(float),
    "employment_income": "employment_income",
    "self_employment_income": "self_employment_income",
    "interest_income": "interest_income",
    "dividend_income": "dividend_income",
    "social_security": "social_security",
    "pension_income": "taxable_pension_income",
    "unemployment_compensation": "unemployment_compensation",
    "medical_expense": "medical_out_of_pocket",
    "childcare_expense": "childcare_expense",
    "housing_cost": "housing_subsidy",  # Or rent/mortgage
}


# PUF variable mapping to common schema
PUF_MAPPING = {
    "age": "age",
    "is_male": "is_male",
    "is_married": lambda df: (df.get("filing_status", "") == "JOINT").astype(float),
    "n_children": "exemptions_count",  # Approximate
    "is_joint_filer": lambda df: (df.get("filing_status", "") == "JOINT").astype(float),
    "is_head_of_household": lambda df: (df.get("filing_status", "") == "HEAD_OF_HOUSEHOLD").astype(float),
    "employment_income": "employment_income",
    "self_employment_income": "self_employment_income",
    "interest_income": "taxable_interest_income",
    "dividend_income": "ordinary_dividend_income",
    "qualified_dividend_income": "qualified_dividend_income",
    "short_term_capital_gains": "short_term_capital_gains",
    "long_term_capital_gains": "long_term_capital_gains",
    "capital_gains_distributions": "capital_gains_distributions",
    "partnership_s_corp_income": "partnership_s_corp_income",
    "rental_income": "rental_income",
    "social_security": "gross_social_security",
    "pension_income": "taxable_pension_income",
    "ira_distributions": "ira_distributions",
    "unemployment_compensation": "unemployment_compensation",
    "farm_income": "farm_income",
    "alimony_income": "alimony_received",
    "medical_expense": "medical_expense_agi_floor",
    "property_tax_paid": "real_estate_tax_paid",
    "mortgage_interest_paid": "mortgage_interest_paid",
    "charitable_contributions": lambda df: df.get("charitable_cash", 0) + df.get("charitable_noncash", 0),
    "state_local_tax_paid": "state_income_tax_paid",
    "student_loan_interest": "student_loan_interest",
}


def signed_log(x: np.ndarray) -> np.ndarray:
    """Signed log transform for values that can be negative.

    sign(x) * log(1 + |x|)
    """
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log(y: np.ndarray) -> np.ndarray:
    """Inverse of signed log transform."""
    return np.sign(y) * (np.exp(np.abs(y)) - 1)


def apply_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply transformation to values."""
    if transform == "none":
        return values
    elif transform == "log1p":
        return np.log1p(np.maximum(values, 0))
    elif transform == "signed_log":
        return signed_log(values)
    else:
        raise ValueError(f"Unknown transform: {transform}")


def apply_inverse_transform(values: np.ndarray, transform: str) -> np.ndarray:
    """Apply inverse transformation."""
    if transform == "none":
        return values
    elif transform == "log1p":
        return np.expm1(values)
    elif transform == "signed_log":
        return inverse_signed_log(values)
    else:
        raise ValueError(f"Unknown transform: {transform}")


def harmonize_cps(cps: pd.DataFrame) -> pd.DataFrame:
    """Harmonize CPS to common schema."""
    result = pd.DataFrame(index=cps.index)

    for common_var, spec in COMMON_SCHEMA.items():
        mapping = CPS_MAPPING.get(common_var)

        if mapping is None:
            # Variable not in CPS - mark as NaN
            result[common_var] = np.nan
        elif callable(mapping):
            # Lambda function to compute value
            try:
                result[common_var] = mapping(cps)
            except Exception:
                result[common_var] = np.nan
        elif isinstance(mapping, str):
            # Direct column mapping
            if mapping in cps.columns:
                result[common_var] = cps[mapping].fillna(0)
            else:
                result[common_var] = np.nan
        else:
            result[common_var] = np.nan

    # Add metadata
    result["_survey"] = "cps"
    result["weight"] = cps.get("weight", cps.get("person_weight", 1.0))

    # Copy household/tax unit IDs if present
    for id_col in ["household_id", "tax_unit_id", "person_id"]:
        if id_col in cps.columns:
            result[id_col] = cps[id_col]

    return result


def harmonize_puf(puf: pd.DataFrame) -> pd.DataFrame:
    """Harmonize PUF to common schema."""
    result = pd.DataFrame(index=puf.index)

    for common_var, spec in COMMON_SCHEMA.items():
        mapping = PUF_MAPPING.get(common_var)

        if mapping is None:
            # Variable not in PUF - mark as NaN
            result[common_var] = np.nan
        elif callable(mapping):
            # Lambda function to compute value
            try:
                result[common_var] = mapping(puf)
            except Exception:
                result[common_var] = np.nan
        elif isinstance(mapping, str):
            # Direct column mapping
            if mapping in puf.columns:
                result[common_var] = puf[mapping].fillna(0)
            else:
                result[common_var] = np.nan
        else:
            result[common_var] = np.nan

    # Add metadata
    result["_survey"] = "puf"
    result["weight"] = puf.get("weight", 1.0)

    # Copy IDs if present
    for id_col in ["tax_unit_id", "person_id"]:
        if id_col in puf.columns:
            result[id_col] = puf[id_col]

    return result


def harmonize_surveys(
    surveys: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Harmonize multiple surveys to common schema.

    Args:
        surveys: Dict of survey name to DataFrame
                 Supported: "cps", "puf"

    Returns:
        Dict of harmonized DataFrames
    """
    result = {}

    for name, df in surveys.items():
        print(f"Harmonizing {name}...")
        if name == "cps":
            result[name] = harmonize_cps(df)
        elif name == "puf":
            result[name] = harmonize_puf(df)
        else:
            raise ValueError(f"Unknown survey: {name}")

        print(f"  {len(result[name]):,} records")
        print(f"  Variables: {len([c for c in result[name].columns if not c.startswith('_')])}")

    return result


def stack_surveys(
    harmonized: dict[str, pd.DataFrame],
    normalize_weights: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stack harmonized surveys into single DataFrame.

    Args:
        harmonized: Dict of harmonized survey DataFrames
        normalize_weights: If True, normalize weights within each survey to sum to 1

    Returns:
        stacked: Combined DataFrame
        mask: Boolean array of shape (n_records, n_variables)
              True = observed, False = missing (NaN)
    """
    dfs = []

    for name, df in harmonized.items():
        df = df.copy()

        if normalize_weights and "weight" in df.columns:
            # Normalize so each survey contributes equally
            df["weight"] = df["weight"] / df["weight"].sum()

        dfs.append(df)

    stacked = pd.concat(dfs, ignore_index=True)

    # Create mask for common schema variables
    schema_vars = list(COMMON_SCHEMA.keys())
    mask = stacked[schema_vars].notna().values

    print("\nStacked data:")
    print(f"  Total records: {len(stacked):,}")
    print(f"  Schema variables: {len(schema_vars)}")

    # Report missing data pattern
    print("\nMissing data by variable:")
    for i, var in enumerate(schema_vars):
        n_observed = mask[:, i].sum()
        pct = 100 * n_observed / len(stacked)
        if pct < 100:
            print(f"  {var}: {n_observed:,} ({pct:.1f}%)")

    return stacked, mask


def transform_for_training(
    stacked: pd.DataFrame,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Transform stacked data for MAF training.

    Applies log transforms, normalizes, and handles NaN.

    Returns:
        X: Transformed feature array with 0 for missing values
        mask: Boolean mask (True = observed)
        variable_names: List of variable names in order
    """
    schema_vars = list(COMMON_SCHEMA.keys())
    n_records = len(stacked)
    n_vars = len(schema_vars)

    X = np.zeros((n_records, n_vars), dtype=np.float32)

    for i, var in enumerate(schema_vars):
        spec = COMMON_SCHEMA[var]
        values = stacked[var].values.copy()

        # Replace NaN with 0 before transform (mask will ignore these)
        observed = mask[:, i]
        values = np.where(observed, values, 0)

        # Apply transform
        if spec["type"] != "binary":
            values = apply_transform(values, spec.get("transform", "none"))

        X[:, i] = values

    # Normalize observed values per variable
    for i in range(n_vars):
        observed = mask[:, i]
        if observed.sum() > 0:
            obs_values = X[observed, i]
            mean = obs_values.mean()
            std = obs_values.std() + 1e-8
            X[:, i] = (X[:, i] - mean) / std

    return X, mask, schema_vars


if __name__ == "__main__":
    # Test harmonization
    print("Testing survey harmonization...")

    # Create dummy CPS data
    cps = pd.DataFrame({
        "age": [35, 42, 28, 65],
        "sex": [1, 2, 1, 2],
        "state_fips": [6, 36, 48, 12],
        "employment_income": [50000, 75000, 40000, 0],
        "self_employment_income": [0, 10000, 0, 0],
        "social_security": [0, 0, 0, 24000],
        "weight": [1000, 1500, 800, 1200],
    })

    # Create dummy PUF data
    puf = pd.DataFrame({
        "age": [45, 55, 38],
        "filing_status": ["JOINT", "SINGLE", "JOINT"],
        "employment_income": [150000, 80000, 200000],
        "long_term_capital_gains": [50000, 0, 100000],
        "partnership_s_corp_income": [0, 25000, 75000],
        "charitable_cash": [5000, 0, 10000],
        "weight": [500, 300, 400],
    })

    # Harmonize
    harmonized = harmonize_surveys({"cps": cps, "puf": puf})

    # Stack
    stacked, mask = stack_surveys(harmonized)

    print("\nStacked head:")
    print(stacked.head())

    print("\nMask shape:", mask.shape)
    print("Observed per row:", mask.sum(axis=1))
