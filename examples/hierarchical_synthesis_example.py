"""
Hierarchical Microdata Synthesis Example

Demonstrates the hybrid flattening + post-processing approach for
generating microdata with household/person hierarchy.
"""

import numpy as np
import pandas as pd

from microplex import Synthesizer

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_hierarchical_data(n_households=1000):
    """
    Create sample hierarchical microdata with realistic structure.

    Structure:
        Household (income, region)
          └── Persons (age, earnings, education)
    """
    households = []
    persons = []
    person_id = 0

    for hh_id in range(n_households):
        # Household characteristics
        region = np.random.choice(["Northeast", "South", "Midwest", "West"])
        household_size = np.random.choice([1, 2, 3, 4, 5], p=[0.28, 0.35, 0.15, 0.13, 0.09])

        # Generate persons in household
        household_members = []
        for _ in range(household_size):
            if len(household_members) == 0:
                # Head of household (25-75)
                age = np.random.randint(25, 76)
            elif len(household_members) == 1:
                # Spouse (similar age)
                head_age = household_members[0]["age"]
                age = max(18, head_age + np.random.randint(-10, 11))
            else:
                # Children (younger than head)
                head_age = household_members[0]["age"]
                age = max(0, np.random.randint(0, head_age - 18))

            # Education (age-dependent)
            if age < 18:
                education = "None"
            elif age < 22:
                education = np.random.choice(["High School", "Some College"], p=[0.5, 0.5])
            else:
                education = np.random.choice(
                    ["High School", "Some College", "Bachelor", "Graduate"],
                    p=[0.30, 0.25, 0.30, 0.15]
                )

            # Earnings (age and education dependent)
            if age < 16:
                earnings = 0
            elif education == "High School":
                earnings = max(0, np.random.lognormal(10, 0.8))
            elif education in ["Some College", "Bachelor"]:
                earnings = max(0, np.random.lognormal(10.5, 0.9))
            else:
                earnings = max(0, np.random.lognormal(11, 1.0))

            member = {
                "person_id": person_id,
                "household_id": hh_id,
                "age": age,
                "education": education,
                "earnings": earnings,
            }
            household_members.append(member)
            person_id += 1

        # Household income = sum of member earnings
        household_income = sum(m["earnings"] for m in household_members)

        # Add household_income to each person record
        for member in household_members:
            member["household_income"] = household_income
            member["region"] = region
            persons.append(member)

        households.append({
            "household_id": hh_id,
            "income": household_income,
            "size": household_size,
            "region": region,
        })

    return pd.DataFrame(households), pd.DataFrame(persons)


def enforce_hierarchy(
    data: pd.DataFrame,
    household_id: str = "household_id",
    shared_vars: list = None,
    max_spouse_age_diff: int = 20,
    min_parent_child_diff: int = 18,
) -> pd.DataFrame:
    """
    Enforce hierarchical consistency in flattened microdata.

    Args:
        data: Flattened microdata with person records
        household_id: Column identifying household membership
        shared_vars: Variables that must be identical within household
        max_spouse_age_diff: Maximum age difference between spouses
        min_parent_child_diff: Minimum age difference between parents and children

    Returns:
        Corrected microdata
    """
    if shared_vars is None:
        shared_vars = ["household_income", "region"]

    result = []

    for hh_id, group in data.groupby(household_id):
        hh = group.copy().reset_index(drop=True)

        # Enforce shared variables (use mean for numeric, mode for categorical)
        for var in shared_vars:
            if var in hh.columns:
                if hh[var].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    # Numeric: use mean
                    consistent_value = hh[var].mean()
                    hh[var] = consistent_value
                else:
                    # Categorical: use mode
                    consistent_value = hh[var].mode()[0]
                    hh[var] = consistent_value

        # Apply relationship constraints
        if len(hh) >= 2:
            # Spouse age constraint
            ages = hh["age"].values
            if abs(ages[0] - ages[1]) > max_spouse_age_diff:
                # Adjust second person's age to be within range
                adjustment = np.random.randint(-max_spouse_age_diff//2, max_spouse_age_diff//2)
                hh.loc[1, "age"] = max(18, ages[0] + adjustment)

        if len(hh) > 2:
            # Child age constraints
            parent_ages = hh["age"].values[:2]
            max_parent_age = max(parent_ages)

            for i in range(2, len(hh)):
                child_age = hh.loc[i, "age"]
                if child_age > max_parent_age - min_parent_child_diff:
                    # Adjust child age to be reasonable
                    hh.loc[i, "age"] = max(
                        0,
                        max_parent_age - min_parent_child_diff - np.random.randint(0, 10)
                    )

        result.append(hh)

    return pd.concat(result, ignore_index=True)


def main():
    """Demonstrate hierarchical synthesis workflow."""

    print("=" * 80)
    print("Hierarchical Microdata Synthesis Example")
    print("=" * 80)

    # 1. Create sample data
    print("\n1. Creating sample hierarchical data...")
    households_df, persons_df = create_sample_hierarchical_data(n_households=2000)

    print(f"   Created {len(households_df)} households with {len(persons_df)} persons")
    print(f"   Average household size: {len(persons_df) / len(households_df):.2f}")

    # 2. Encode categorical variables
    print("\n2. Encoding categorical variables...")
    # Create numeric codes for categorical variables
    education_map = {
        "None": 0,
        "High School": 1,
        "Some College": 2,
        "Bachelor": 3,
        "Graduate": 4,
    }
    region_map = {
        "Northeast": 0,
        "South": 1,
        "Midwest": 2,
        "West": 3,
    }

    persons_encoded = persons_df.copy()
    persons_encoded["education_code"] = persons_df["education"].map(education_map)
    persons_encoded["region_code"] = persons_df["region"].map(region_map)

    # 3. Train synthesizer on flattened data
    print("\n3. Training synthesizer on person-level data...")
    synth = Synthesizer(
        target_vars=["age", "earnings", "household_income"],
        condition_vars=["education_code", "region_code"],
        n_layers=4,
        hidden_dim=32,
        zero_inflated=True,
    )

    synth.fit(
        persons_encoded,
        epochs=50,
        batch_size=128,
        verbose=False,
    )
    print("   Training complete!")

    # 4. Generate synthetic persons
    print("\n4. Generating synthetic persons...")
    # Create new demographic conditions
    n_synthetic = 5000

    # Generate categorical conditions
    education_cats = np.random.choice(
        ["High School", "Some College", "Bachelor", "Graduate"],
        size=n_synthetic,
        p=[0.30, 0.25, 0.30, 0.15]
    )
    region_cats = np.random.choice(
        ["Northeast", "South", "Midwest", "West"],
        size=n_synthetic,
    )

    # Encode conditions
    synthetic_conditions = pd.DataFrame({
        "education": education_cats,
        "region": region_cats,
        "education_code": [education_map[e] for e in education_cats],
        "region_code": [region_map[r] for r in region_cats],
    })

    # Assign household IDs (simulate household sizes)
    household_sizes = np.random.choice([1, 2, 3, 4, 5], size=n_synthetic // 2,
                                      p=[0.28, 0.35, 0.15, 0.13, 0.09])
    household_ids = []
    for hh_id, size in enumerate(household_sizes):
        household_ids.extend([hh_id] * size)
    # Truncate or pad to match n_synthetic
    household_ids = household_ids[:n_synthetic]
    synthetic_conditions["household_id"] = household_ids

    synthetic = synth.generate(synthetic_conditions[["education_code", "region_code", "household_id"]], seed=42)
    # Add back categorical labels
    synthetic["education"] = synthetic_conditions["education"].values
    synthetic["region"] = synthetic_conditions["region"].values
    print(f"   Generated {len(synthetic)} synthetic person records")

    # 5. Check issues before correction
    print("\n5. Checking hierarchical consistency before correction...")
    issues_before = check_hierarchical_issues(synthetic)
    print(f"   Households with inconsistent household_income: {issues_before['inconsistent_hh_income']}")
    print(f"   Spouse pairs with >20 year age gap: {issues_before['large_spouse_age_diff']}")
    print(f"   Children older than (parent - 18): {issues_before['implausible_child_age']}")

    # 6. Enforce hierarchical consistency
    print("\n6. Enforcing hierarchical consistency...")
    synthetic_corrected = enforce_hierarchy(
        synthetic,
        household_id="household_id",
        shared_vars=["household_income", "region"],
        max_spouse_age_diff=20,
        min_parent_child_diff=18,
    )

    # 7. Check issues after correction
    print("\n7. Checking hierarchical consistency after correction...")
    issues_after = check_hierarchical_issues(synthetic_corrected)
    print(f"   Households with inconsistent household_income: {issues_after['inconsistent_hh_income']}")
    print(f"   Spouse pairs with >20 year age gap: {issues_after['large_spouse_age_diff']}")
    print(f"   Children older than (parent - 18): {issues_after['implausible_child_age']}")

    # 8. Summary statistics
    print("\n8. Comparing distributions:")
    print("\nOriginal data:")
    print(persons_df[["age", "earnings", "household_income"]].describe())
    print("\nSynthetic data (corrected):")
    print(synthetic_corrected[["age", "earnings", "household_income"]].describe())

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)

    return synthetic_corrected


def check_hierarchical_issues(data: pd.DataFrame) -> dict:
    """Check for common hierarchical consistency issues."""
    issues = {}

    # Inconsistent household_income
    inconsistent_count = 0
    for hh_id, group in data.groupby("household_id"):
        if group["household_income"].std() > 0.01:  # Allow small floating point errors
            inconsistent_count += 1
    issues["inconsistent_hh_income"] = inconsistent_count

    # Large spouse age differences
    large_spouse_diff = 0
    for hh_id, group in data.groupby("household_id"):
        if len(group) >= 2:
            ages = group["age"].values[:2]
            if abs(ages[0] - ages[1]) > 20:
                large_spouse_diff += 1
    issues["large_spouse_age_diff"] = large_spouse_diff

    # Implausible child ages
    implausible_child = 0
    for hh_id, group in data.groupby("household_id"):
        if len(group) > 2:
            parent_ages = group["age"].values[:2]
            max_parent = max(parent_ages)
            for child_age in group["age"].values[2:]:
                if child_age > max_parent - 18:
                    implausible_child += 1
                    break
    issues["implausible_child_age"] = implausible_child

    return issues


if __name__ == "__main__":
    synthetic_data = main()
