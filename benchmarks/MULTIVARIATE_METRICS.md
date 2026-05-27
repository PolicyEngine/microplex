# Multivariate Realism Metrics

## Overview

This document describes the multivariate realism metrics added to microplex benchmarks. These metrics go beyond univariate (marginal) distribution matching to assess whether synthetic records are **realistic in the full joint space**.

## Motivation

### The Problem with Univariate Metrics

Traditional synthetic data evaluation relies heavily on univariate metrics:
- KS statistics (marginal distributions)
- Correlation matrices (pairwise relationships)
- Zero-fraction accuracy

**Critical limitation**: A model can have perfect marginal distributions and correct pairwise correlations, but still produce **unrealistic joint records**.

### Examples of Joint Unrealism

Even with perfect marginals, synthetic data could contain:
- 80-year-olds with student debt
- Billionaires receiving food stamps
- Teenagers with $500k in retirement savings
- Families with 10 children earning $15k/year

These records are individually implausible even though each variable's marginal distribution might be correct.

## Multivariate Metrics

### 1. Authenticity Distance (Synthetic → Holdout)

**What it measures**: For each synthetic record, find the Euclidean distance to its nearest real record in the holdout set.

**Interpretation**:
- **Lower = more realistic** synthetic records
- Mean distance shows typical realism
- Min distance checks for privacy issues (too close = potential memorization)

**Key insight**: If synthetic records are far from all real records, they're likely unrealistic combinations.

### 2. Coverage Distance (Holdout → Synthetic)

**What it measures**: For each real holdout record, find the distance to its nearest synthetic record.

**Interpretation**:
- **Lower = better coverage** of the real data manifold
- High max distance indicates regions of the data space that synthetic data doesn't cover
- Should be similar to authenticity distance (if not, we have coverage gaps)

**Key insight**: Ensures we're not missing important regions of the real data distribution.

### 3. Privacy Distance Ratio

**What it measures**: Compare synthetic record distances to training data vs holdout data.

```
Ratio = distance_to_holdout / distance_to_train
```

**Interpretation**:
- **Ratio > 1**: Synthetic records are farther from training than holdout (good generalization)
- **Ratio ≈ 1**: Equal distance (ideal)
- **Ratio < 1**: Closer to training than holdout (overfitting risk)

Also track: What fraction of synthetic records are closer to training than holdout?
- **< 50%**: Good generalization
- **> 50%**: Potential overfitting/memorization

**Key insight**: Prevents models from simply memorizing training data.

### 4. Maximum Mean Discrepancy (MMD)

**What it measures**: Kernel-based two-sample test comparing entire multivariate distributions.

**Formula**:
```
MMD² = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
```
where k is an RBF kernel and X, Y are samples from real and synthetic distributions.

**Interpretation**:
- **MMD = 0** if and only if distributions are identical
- Higher MMD = more different distributions
- **Compare across methods** (lower is better)

**Key insight**: Proper multivariate distribution test with statistical guarantees.

### 5. Energy Distance

**What it measures**: Another multivariate two-sample test based on Euclidean distances.

**Formula**:
```
D(X,Y) = 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
```

**Interpretation**:
- **D = 0** if and only if distributions are identical
- Based on Euclidean distance (no kernel choice needed)
- More interpretable than MMD
- **Compare across methods** (lower is better)

**Key insight**: Measures expected distance between samples from two distributions.

## Implementation Details

### Normalization

All distances are computed on **normalized data** using StandardScaler fit on training data:
```python
scaler = StandardScaler()
train_norm = scaler.fit_transform(train[variables])
holdout_norm = scaler.transform(holdout[variables])
synthetic_norm = scaler.transform(synthetic[variables])
```

This is critical because variables have different scales (age in [18, 90], income in [0, 1e6], etc.).

### Efficient Computation

Uses `scipy.spatial.distance.cdist` for efficient pairwise distance computation:
```python
distances = cdist(source, target, metric='euclidean')
min_distances = np.min(distances, axis=1)
```

### Variables Included

Metrics computed on **all variables** (both conditioning and target):
- Age, education, region (demographics)
- Income, assets, debt, savings (economic outcomes)

This tests the full joint distribution, not just target variables.

## Benchmark Results

### Key Findings

**Best Multivariate Distribution Match**:
- **Microplex**: Lowest MMD (0.039) and Energy Distance (0.010)
- **CT-GAN**: Second best MMD (0.052) and Energy Distance (0.011)
- **QRF**: Worst MMD (0.164) and Energy Distance (0.108)

**Most Realistic Individual Records**:
- **QRF Sequential**: Lowest authenticity distance (0.306)
- **TVAE**: Second lowest (0.361)
- **Microplex**: Higher (0.702)

**Best Coverage**:
- **Microplex**: Lowest coverage distance (0.623)
- **Copula**: Second (0.722)
- **QRF**: Worst (0.997) - large coverage gaps

**Privacy/Overfitting Concerns**:
- **All methods** have 80-84% of synthetic records closer to training than holdout
- This is somewhat concerning but may reflect the nature of the data
- All methods have some records with min distance < 0.1 (privacy threshold)

### Interpretation

1. **Microplex excels at multivariate distribution matching**
   - Lowest MMD and Energy Distance
   - Best coverage of the data manifold
   - Produces records that are statistically correct in the joint space

2. **QRF produces individually realistic records but poor joint distribution**
   - Lowest authenticity distance (records close to real data)
   - But worst MMD/Energy Distance
   - Worst coverage (large gaps)
   - **Hypothesis**: Sequential imputation produces marginally correct but jointly unrealistic records

3. **CT-GAN balances both objectives well**
   - Low MMD/Energy Distance (second best)
   - Moderate authenticity distance
   - Good coverage

4. **Trade-off between individual realism and distributional correctness**
   - QRF: Individual records look realistic (close to training examples) but joint distribution is poor
   - Microplex: Joint distribution is correct but individual records may be farther from any single training example

## Usage

### Running the Benchmark

```bash
cd /Users/maxghenis/PolicyEngine/micro
source .venv/bin/activate
python benchmarks/run_multivariate_benchmark.py
```

### Using Metrics Standalone

```python
from multivariate_metrics import compute_multivariate_metrics

metrics = compute_multivariate_metrics(
    train_data=train_df,
    holdout_data=test_df,
    synthetic_data=synthetic_df,
    variables=['age', 'education', 'region', 'income', 'assets', 'debt', 'savings'],
    kernel_gamma=None,  # Auto: 1/n_features
)

print(f"Authenticity: {metrics['authenticity']['mean']:.4f}")
print(f"Coverage: {metrics['coverage']['mean']:.4f}")
print(f"Privacy Ratio: {metrics['privacy_ratio']['mean_ratio']:.4f}")
print(f"MMD: {metrics['mmd']:.6f}")
print(f"Energy Distance: {metrics['energy_distance']:.6f}")
```

### Comparing Multiple Methods

```python
from multivariate_metrics import compare_methods_multivariate

synthetic_datasets = {
    'microplex': microplex_synthetic,
    'qrf': qrf_synthetic,
    'ctgan': ctgan_synthetic,
}

comparison_df = compare_methods_multivariate(
    train_data=train_df,
    holdout_data=test_df,
    synthetic_datasets=synthetic_datasets,
    variables=all_variables,
)
```

## Visualization Outputs

The benchmark generates three visualizations:

1. **multivariate_comparison.png**: Main metrics comparison
   - Authenticity (Synth → Holdout)
   - Coverage (Holdout → Synth)
   - Privacy Ratio
   - MMD

2. **privacy_analysis.png**: Privacy and overfitting checks
   - Minimum authenticity distance (privacy risk if < 0.1)
   - Fraction of records closer to training than holdout

3. **distribution_tests.png**: Multivariate distribution tests
   - Maximum Mean Discrepancy (RBF kernel)
   - Energy Distance

## Future Work

### Potential Improvements

1. **Conditional multivariate metrics**: Compute MMD/Energy Distance within demographic subgroups
2. **Variable-specific distances**: Weight variables by importance
3. **Local outlier detection**: Use LOF or isolation forest to detect unrealistic records
4. **Semantic realism checks**: Domain-specific rules (e.g., "retired people should have low labor income")

### Integration with Existing Metrics

Multivariate metrics complement, not replace, existing metrics:
- **Marginal fidelity** (KS): Tests univariate distributions
- **Correlation error**: Tests pairwise relationships
- **Multivariate realism**: Tests joint plausibility
- **All three** are needed for comprehensive evaluation

## References

1. Gretton, A., et al. (2012). "A Kernel Two-Sample Test." JMLR.
2. Székely, G. J., & Rizzo, M. L. (2013). "Energy statistics: A class of statistics based on distances." Journal of Statistical Planning and Inference.
3. Hernandez, M., et al. (2022). "SynthCity: Facilitating innovative use cases of synthetic data." arXiv:2301.07573.

## Files

- `benchmarks/multivariate_metrics.py`: Core metric implementations
- `benchmarks/run_multivariate_benchmark.py`: Benchmark runner
- `benchmarks/results/multivariate_metrics.csv`: Numerical results
- `benchmarks/results/*.png`: Visualizations
- `benchmarks/MULTIVARIATE_METRICS.md`: This documentation
