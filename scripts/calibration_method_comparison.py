"""
Sparsity Experiment v4: IPF vs Alternating IPF/GREG vs GD+L0.

Key addition: Alternating IPF/GREG calibration.
- IPF pass: Match categorical margins (state, CD populations)
- GREG pass: Regression adjustment for continuous totals (income, benefits)
- L1 sparsity: Applied via IRL1 (iteratively reweighted L1)
"""

import sys
sys.path.insert(0, '/Users/maxghenis/PolicyEngine/microplex/src')

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.optimize import nnls
import torch
import matplotlib.pyplot as plt

print("="*70)
print("CALIBRATION METHOD COMPARISON: IPF vs IPF/GREG vs GD+L0")
print("="*70)

# Load data - use synthetic file with matching CD IDs
synth = pd.read_parquet('/Users/maxghenis/PolicyEngine/microplex/data/microplex_synthetic_with_blocks.parquet')
synth['state_fips'] = synth['state_fips'].astype(str).str.zfill(2)
blocks = pd.read_parquet('/Users/maxghenis/PolicyEngine/microplex/data/block_probabilities.parquet')
print(f"Loaded {len(synth):,} households")

# =============================================================================
# BUILD TARGETS
# =============================================================================

# Geographic targets (categorical - good for IPF)
state_pops = blocks.groupby('state_fips')['population'].sum()
state_targets = {str(k).zfill(2): v for k, v in state_pops.items()}

cd_col = 'cd_id' if 'cd_id' in blocks.columns else 'cd_geoid'
cd_pops = blocks.groupby(cd_col)['population'].sum()
cd_targets = dict(cd_pops)

# Use consistent column name for synthetic data
synth_cd_col = 'cd_id' if 'cd_id' in synth.columns else 'cd_geoid'

# Income targets (continuous - better for GREG)
# Note: microplex_synthetic_with_blocks.parquet only has hh_income
income_cols = ['hh_income']
income_targets = {}
for col in income_cols:
    if col in synth.columns:
        total = (synth[col] * synth['weight']).sum()
        if total > 0:
            income_targets[col] = total

# Create income bracket targets for more continuous constraints
income_brackets = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, np.inf]
bracket_labels = [f"income_bracket_{i}" for i in range(len(income_brackets)-1)]
synth['income_bracket'] = pd.cut(synth['hh_income'], bins=income_brackets, labels=bracket_labels)

for bracket in bracket_labels:
    mask = synth['income_bracket'] == bracket
    total = (synth.loc[mask, 'hh_income'] * synth.loc[mask, 'weight']).sum()
    if total > 0:
        income_targets[bracket] = total

# Household size targets (continuous aggregates)
benefit_targets = {}
for col in ['n_persons', 'n_adults', 'n_children']:
    if col in synth.columns:
        total = (synth[col] * synth['weight']).sum()
        if total > 0:
            benefit_targets[col] = total

print(f"\nTargets:")
print(f"  States: {len(state_targets)} (categorical)")
print(f"  CDs: {len(cd_targets)} (categorical)")
print(f"  Income: {len(income_targets)} (continuous)")
print(f"  Benefits: {len(benefit_targets)} (continuous)")
print(f"  Total: {len(state_targets) + len(cd_targets) + len(income_targets) + len(benefit_targets)}")

# =============================================================================
# BUILD CONSTRAINT MATRICES
# =============================================================================

def build_categorical_constraints(df, state_targets, cd_targets, cd_col='cd_geoid'):
    """Build constraint matrix for categorical (count) targets."""
    n = len(df)
    rows, cols, vals = [], [], []
    targets = []
    names = []
    row_idx = 0

    # State constraints
    for state, target in state_targets.items():
        indices = np.where(df['state_fips'] == state)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0] * len(indices))
            targets.append(target)
            names.append(f"state_{state}")
            row_idx += 1

    # CD constraints
    for cd, target in cd_targets.items():
        indices = np.where(df[cd_col] == cd)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0] * len(indices))
            targets.append(target)
            names.append(f"cd_{cd}")
            row_idx += 1

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(row_idx, n))
    return A, np.array(targets), names


def build_continuous_constraints(df, income_targets, benefit_targets):
    """Build constraint matrix for continuous (sum) targets."""
    n = len(df)
    rows, cols, vals = [], [], []
    targets = []
    names = []
    row_idx = 0

    # Income constraints
    for name, target in income_targets.items():
        if name.startswith('income_bracket_'):
            # For bracket targets, use hh_income values for records in that bracket
            mask = df['income_bracket'] == name
            indices = np.where(mask)[0]
            if len(indices) > 0:
                inc_vals = df.loc[df.index[indices], 'hh_income'].values
                nonzero = inc_vals > 0
                if nonzero.sum() > 0:
                    rows.extend([row_idx] * nonzero.sum())
                    cols.extend(indices[nonzero])
                    vals.extend(inc_vals[nonzero])
                    targets.append(target)
                    names.append(name)
                    row_idx += 1
        elif name in df.columns:
            values = df[name].values
            nonzero = np.where(values > 0)[0]
            if len(nonzero) > 0:
                rows.extend([row_idx] * len(nonzero))
                cols.extend(nonzero)
                vals.extend(values[nonzero])
                targets.append(target)
                names.append(name)
                row_idx += 1

    # Benefit/demographic constraints
    for col, target in benefit_targets.items():
        if col in df.columns:
            values = df[col].values
            nonzero = np.where(values > 0)[0]
            if len(nonzero) > 0:
                rows.extend([row_idx] * len(nonzero))
                cols.extend(nonzero)
                vals.extend(values[nonzero])
                targets.append(target)
                names.append(col)
                row_idx += 1

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(row_idx, n))
    return A, np.array(targets), names


# Build separate constraint matrices
A_cat, b_cat, names_cat = build_categorical_constraints(synth, state_targets, cd_targets, synth_cd_col)
A_cont, b_cont, names_cont = build_continuous_constraints(synth, income_targets, benefit_targets)

# Combined matrix for evaluation
A_full = sparse.vstack([A_cat, A_cont])
b_full = np.concatenate([b_cat, b_cont])

print(f"\nConstraint matrices:")
print(f"  Categorical: {A_cat.shape[0]} targets × {A_cat.shape[1]} records")
print(f"  Continuous: {A_cont.shape[0]} targets × {A_cont.shape[1]} records")

# =============================================================================
# CALIBRATION METHODS
# =============================================================================

def ipf_calibrate(A, b, weights=None, max_iter=100, tol=1e-8):
    """IPF calibration for categorical constraints."""
    n = A.shape[1]
    if weights is None:
        weights = np.ones(n)
    else:
        weights = weights.copy()

    for _ in range(max_iter):
        old = weights.copy()
        for i in range(A.shape[0]):
            row = A.getrow(i)
            if row.nnz == 0:
                continue
            current = (weights[row.indices] * row.data).sum()
            if current > 1e-10:
                weights[row.indices] *= b[i] / current
        if np.max(np.abs(weights - old) / (old + 1e-10)) < tol:
            break
    return weights


def greg_calibrate(A, b, weights, max_iter=50, tol=1e-6, damp=0.5):
    """
    GREG (Generalized Regression Estimation) calibration.

    Adjusts weights to match continuous targets via regression:
    w_new = w * g  where g minimizes ||A @ (w*g) - b||^2

    Uses multiplicative adjustment to preserve non-negativity.

    Args:
        damp: Damping factor (0-1). Lower = more conservative updates.
    """
    n = A.shape[1]
    w = weights.copy()

    for iteration in range(max_iter):
        # Current predictions
        pred = A @ w

        # Residuals
        resid = b - pred

        # If close enough, stop
        rel_err = np.abs(resid) / (np.abs(b) + 1e-10)
        if rel_err.max() < tol:
            break

        # Proportional adjustment per constraint (damped)
        for i in range(A.shape[0]):
            row = A.getrow(i)
            if row.nnz == 0:
                continue

            current = (w[row.indices] * row.data).sum()
            if current > 1e-10:
                # Proportional adjustment
                factor = b[i] / current
                # Damped update to avoid oscillation
                factor = 1 + damp * (factor - 1)
                factor = np.clip(factor, 1 - damp, 1 + damp)
                w[row.indices] *= factor

    return w


def alternating_ipf_greg(A_cat, b_cat, A_cont, b_cont,
                         max_outer=20, max_inner=10, tol=1e-4):
    """
    Alternating IPF/GREG calibration.

    1. IPF pass: Match categorical margins
    2. GREG pass: Adjust for continuous totals (damped)
    3. Final IPF pass: Restore categorical margins
    Repeat until convergence.
    """
    n = A_cat.shape[1]
    weights = np.ones(n)

    for outer in range(max_outer):
        old_weights = weights.copy()

        # IPF pass for categorical constraints
        weights = ipf_calibrate(A_cat, b_cat, weights, max_iter=max_inner)

        # GREG pass for continuous constraints (very damped to not disrupt categorical)
        if A_cont.shape[0] > 0:
            weights = greg_calibrate(A_cont, b_cont, weights, max_iter=5, damp=0.1)

        # Final IPF pass to restore categorical margins
        weights = ipf_calibrate(A_cat, b_cat, weights, max_iter=max_inner)

        # Check convergence
        change = np.max(np.abs(weights - old_weights) / (old_weights + 1e-10))
        if change < tol:
            break

    return weights


def alternating_ipf_greg_sparse(A_cat, b_cat, A_cont, b_cont,
                                 target_sparsity=0.9, max_outer=20):
    """
    Alternating IPF/GREG with L1-style sparsity.

    After initial calibration, iteratively:
    1. Zero out smallest weights
    2. Re-calibrate remaining weights
    """
    n = A_cat.shape[1]

    # Initial calibration
    weights = alternating_ipf_greg(A_cat, b_cat, A_cont, b_cont, max_outer=max_outer)

    # Target number of non-zero weights
    n_target = int(n * (1 - target_sparsity))
    n_target = max(n_target, A_cat.shape[0] + A_cont.shape[0])  # Need at least as many as constraints

    # Iteratively remove smallest weights
    active = np.ones(n, dtype=bool)

    for iteration in range(50):  # Max iterations
        n_active = active.sum()
        if n_active <= n_target:
            break

        # Remove bottom 10% of active weights each iteration
        n_remove = max(1, int(n_active * 0.1))
        n_remove = min(n_remove, n_active - n_target)

        # Find smallest active weights
        active_indices = np.where(active)[0]
        active_weights = weights[active_indices]
        remove_idx = active_indices[np.argsort(active_weights)[:n_remove]]

        # Zero them out
        active[remove_idx] = False
        weights[remove_idx] = 0

        # Re-calibrate remaining weights
        A_cat_sub = A_cat[:, active]
        A_cont_sub = A_cont[:, active]

        sub_weights = alternating_ipf_greg(
            A_cat_sub, b_cat, A_cont_sub, b_cont, max_outer=10
        )
        weights[active] = sub_weights

    return weights


# =============================================================================
# EVALUATION
# =============================================================================

def compute_errors(weights, A_cat, b_cat, A_cont, b_cont):
    """Compute MAE by target type."""
    # Categorical (geographic) targets
    pred_cat = A_cat @ weights
    cat_err = np.abs(pred_cat - b_cat) / np.maximum(np.abs(b_cat), 1e-10) * 100

    # Continuous (income/benefit) targets
    pred_cont = A_cont @ weights
    cont_err = np.abs(pred_cont - b_cont) / np.maximum(np.abs(b_cont), 1e-10) * 100

    return {
        'categorical': cat_err.mean(),
        'continuous': cont_err.mean(),
        'overall': np.concatenate([cat_err, cont_err]).mean(),
    }


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

results = []

# Method 1: Pure IPF (subsampled for sparsity)
print("\n" + "="*70)
print("METHOD 1: IPF + Random Subsampling")
print("="*70)

for frac in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    n_sample = int(len(synth) * frac)
    np.random.seed(42)
    idx = np.random.choice(len(synth), n_sample, replace=False)

    A_cat_sub = A_cat[:, idx]
    A_cont_sub = A_cont[:, idx]

    w_sub = ipf_calibrate(A_cat_sub, b_cat)

    w_full = np.zeros(len(synth))
    w_full[idx] = w_sub

    errs = compute_errors(w_full, A_cat, b_cat, A_cont, b_cont)
    print(f"  {frac*100:5.1f}% ({n_sample:>6,}): cat={errs['categorical']:.2f}%, "
          f"cont={errs['continuous']:.2f}%, overall={errs['overall']:.2f}%")

    results.append({
        'method': 'IPF+Subsample',
        'n_records': n_sample,
        **errs
    })


# Method 2: Alternating IPF/GREG (subsampled for sparsity)
print("\n" + "="*70)
print("METHOD 2: Alternating IPF/GREG + Random Subsampling")
print("="*70)

for frac in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    n_sample = int(len(synth) * frac)
    np.random.seed(42)
    idx = np.random.choice(len(synth), n_sample, replace=False)

    A_cat_sub = A_cat[:, idx]
    A_cont_sub = A_cont[:, idx]

    w_sub = alternating_ipf_greg(A_cat_sub, b_cat, A_cont_sub, b_cont)

    w_full = np.zeros(len(synth))
    w_full[idx] = w_sub

    errs = compute_errors(w_full, A_cat, b_cat, A_cont, b_cont)
    print(f"  {frac*100:5.1f}% ({n_sample:>6,}): cat={errs['categorical']:.2f}%, "
          f"cont={errs['continuous']:.2f}%, overall={errs['overall']:.2f}%")

    results.append({
        'method': 'IPF/GREG+Subsample',
        'n_records': n_sample,
        **errs
    })


# Method 3: Alternating IPF/GREG with L1-style sparsity
print("\n" + "="*70)
print("METHOD 3: Alternating IPF/GREG + L1 Sparsity")
print("="*70)

for sparsity in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
    weights = alternating_ipf_greg_sparse(
        A_cat, b_cat, A_cont, b_cont, target_sparsity=sparsity
    )

    n_active = (weights > 0).sum()
    errs = compute_errors(weights, A_cat, b_cat, A_cont, b_cont)

    print(f"  {sparsity*100:5.1f}% sparse ({n_active:>6,}): cat={errs['categorical']:.2f}%, "
          f"cont={errs['continuous']:.2f}%, overall={errs['overall']:.2f}%")

    results.append({
        'method': 'IPF/GREG+L1',
        'n_records': n_active,
        **errs
    })


# Method 4: GD + L0 (if l0-python available)
print("\n" + "="*70)
print("METHOD 4: Gradient Descent + L0")
print("="*70)

try:
    from l0.calibration import SparseCalibrationWeights

    # Build normalized constraint matrix for GD
    A_norm_rows, A_norm_cols, A_norm_vals = [], [], []
    b_norm = []

    # Categorical (normalize by target)
    for i in range(A_cat.shape[0]):
        row = A_cat.getrow(i)
        target = b_cat[i]
        A_norm_rows.extend([len(b_norm)] * row.nnz)
        A_norm_cols.extend(row.indices)
        A_norm_vals.extend(row.data / target)
        b_norm.append(1.0)

    # Continuous (normalize by target)
    for i in range(A_cont.shape[0]):
        row = A_cont.getrow(i)
        target = b_cont[i]
        A_norm_rows.extend([len(b_norm)] * row.nnz)
        A_norm_cols.extend(row.indices)
        A_norm_vals.extend(row.data / target)
        b_norm.append(1.0)

    A_norm = sparse.csr_matrix(
        (A_norm_vals, (A_norm_rows, A_norm_cols)),
        shape=(len(b_norm), len(synth))
    )
    b_norm = np.array(b_norm)

    configs = [
        (0, 1.0),
        (1e-4, 0.9),
        (1e-3, 0.7),
        (5e-3, 0.5),
        (1e-2, 0.3),
        (5e-2, 0.2),
        (1e-1, 0.1),
        (5e-1, 0.05),
        (1.0, 0.02),
    ]

    for lam, init_keep in configs:
        model = SparseCalibrationWeights(n_features=len(synth), init_keep_prob=init_keep)
        model.fit(M=A_norm, y=b_norm, lambda_l0=lam, lr=0.3, epochs=500, verbose=False)

        with torch.no_grad():
            weights = model.get_weights(deterministic=True).cpu().numpy()

        n_active = (weights > 0).sum()
        errs = compute_errors(weights, A_cat, b_cat, A_cont, b_cont)

        print(f"  λ={lam:.0e}, init={init_keep:.2f}: {n_active:>6,} active, "
              f"cat={errs['categorical']:.2f}%, cont={errs['continuous']:.2f}%, "
              f"overall={errs['overall']:.2f}%")

        results.append({
            'method': 'GD+L0',
            'n_records': n_active,
            'lambda': lam,
            **errs
        })

except ImportError:
    print("  (l0-python not available, skipping GD+L0)")


# =============================================================================
# PLOT RESULTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Group by method
methods = ['IPF+Subsample', 'IPF/GREG+Subsample', 'IPF/GREG+L1', 'GD+L0']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
markers = ['o', '^', 's', 'D']

for ax, metric, title in [
    (axes[0], 'overall', 'Overall MAE'),
    (axes[1], 'categorical', 'Geographic Targets (Categorical)'),
    (axes[2], 'continuous', 'Income/Benefit Targets (Continuous)'),
]:
    for method, color, marker in zip(methods, colors, markers):
        data = [r for r in results if r['method'] == method]
        if data:
            x = [r['n_records'] for r in data]
            y = [r[metric] for r in data]
            ax.plot(x, y, f'{marker}-', label=method, linewidth=2, markersize=8, color=color)

    ax.set_xlabel('Number of Active Records', fontsize=12)
    ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1% error')

plt.suptitle('Calibration Methods: IPF vs IPF/GREG vs GD+L0\n'
             f'({len(state_targets)} states + {len(cd_targets)} CDs categorical, '
             f'{len(income_targets)} income + {len(benefit_targets)} benefit continuous)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/Users/maxghenis/PolicyEngine/microplex/docs/calibration_method_comparison.png',
            dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: docs/calibration_method_comparison.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Method':<22} {'Records':>10} {'Categorical':>12} {'Continuous':>12} {'Overall':>10}")
print("-"*68)
for r in sorted(results, key=lambda x: (-x['n_records'], x['method'])):
    print(f"{r['method']:<22} {r['n_records']:>10,} {r['categorical']:>11.2f}% "
          f"{r['continuous']:>11.2f}% {r['overall']:>9.2f}%")
