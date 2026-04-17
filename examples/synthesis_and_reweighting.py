"""
Example: End-to-end synthesis and reweighting workflow.

This demonstrates the full microplex pipeline:
1. Synthesize microdata using normalizing flows
2. Reweight synthetic data to match population targets
3. Compare distributions before and after reweighting
"""

import numpy as np
import pandas as pd

from microplex import Reweighter, Synthesizer


def create_training_data(n=5000, seed=42):
    """Create sample training data with demographics and income."""
    np.random.seed(seed)

    # Demographics
    age = np.random.randint(18, 80, n)
    education = np.random.choice([1, 2, 3, 4], n, p=[0.1, 0.3, 0.4, 0.2])
    state_code = np.random.choice([0, 1, 2], n, p=[0.35, 0.30, 0.35])  # 0=CA, 1=NY, 2=TX

    # Income depends on age and education
    base_income = np.random.lognormal(10, 1, n)
    age_factor = 1 + 0.015 * (age - 18)
    edu_factor = 1 + 0.25 * education
    income = base_income * age_factor * edu_factor

    # 10% have zero income
    income[np.random.random(n) < 0.1] = 0

    # Map state codes to names for display
    state_map = {0: "CA", 1: "NY", 2: "TX"}
    state = [state_map[code] for code in state_code]

    return pd.DataFrame({
        "age": age,
        "education": education,
        "state_code": state_code,
        "state": state,
        "income": income,
        "weight": np.ones(n),
    })


def create_demographics(n=10000, seed=123):
    """Create new demographics for synthesis."""
    np.random.seed(seed)

    state_code = np.random.choice([0, 1, 2], n)
    state_map = {0: "CA", 1: "NY", 2: "TX"}
    state = [state_map[code] for code in state_code]

    return pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "education": np.random.choice([1, 2, 3, 4], n),
        "state_code": state_code,
        "state": state,
    })


def main():
    print("=" * 60)
    print("Microplex: Synthesis + Reweighting Example")
    print("=" * 60)

    # Step 1: Train synthesizer
    print("\n[1/5] Creating training data...")
    training_data = create_training_data()
    print(f"  Training records: {len(training_data)}")
    print(f"  Mean income: ${training_data['income'].mean():,.0f}")

    print("\n[2/5] Training synthesizer...")
    synth = Synthesizer(
        target_vars=["income"],
        condition_vars=["age", "education", "state_code"],
        zero_inflated=True,
        log_transform=True,
    )
    synth.fit(training_data, epochs=50, verbose=False)
    print("  Synthesizer trained successfully")

    # Step 2: Generate synthetic microdata
    print("\n[3/5] Generating synthetic microdata...")
    demographics = create_demographics()
    synthetic = synth.generate(demographics, seed=42)
    print(f"  Synthetic records: {len(synthetic)}")
    print(f"  Mean synthetic income: ${synthetic['income'].mean():,.0f}")

    # Show initial state distribution
    print("\n  Initial state distribution:")
    state_counts = synthetic.groupby("state").size()
    for state, count in state_counts.items():
        pct = 100 * count / len(synthetic)
        print(f"    {state}: {count:,} ({pct:.1f}%)")

    # Step 3: Define population targets
    print("\n[4/5] Defining population targets...")
    targets = {
        "state": {
            "CA": 4000,  # 40% target
            "NY": 3000,  # 30% target
            "TX": 3000,  # 30% target
        }
    }
    print("  Target state distribution:")
    for state, target in targets["state"].items():
        pct = 100 * target / sum(targets["state"].values())
        print(f"    {state}: {target:,} ({pct:.1f}%)")

    # Step 4: Sparse reweighting
    print("\n[5/5] Applying sparse reweighting...")

    # Compare L0, L1, and L2 reweighting
    for sparsity in ["l0", "l1", "l2"]:
        print(f"\n  {sparsity.upper()} reweighting:")

        reweighter = Reweighter(sparsity=sparsity)
        weighted = reweighter.fit_transform(synthetic, targets, drop_zeros=False)

        # Get sparsity statistics
        stats = reweighter.get_sparsity_stats()
        print(f"    Records used: {stats['n_nonzero']} / {stats['n_records']}")
        print(f"    Sparsity: {100 * stats['sparsity']:.1f}%")
        print(f"    Max weight: {stats['max_weight']:.2f}")

        # Check target matching
        state_weights = weighted.groupby("state")["weight"].sum()
        print("    Target matching:")
        for state in ["CA", "NY", "TX"]:
            actual = state_weights[state]
            target = targets["state"][state]
            error = abs(actual - target) / target * 100
            print(f"      {state}: {actual:,.0f} (target: {target:,}, error: {error:.2f}%)")

    print("\n" + "=" * 60)
    print("Complete! Synthetic data calibrated to population targets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
