# Distributional Quality Metrics for Microplex

## Overview

This document explains the new distributional quality metrics added to microplex benchmarks. These metrics address a critical gap in the previous benchmarks, which only used KS tests and correlation error.

## The Problem

Previous metrics (KS statistic, correlation error) don't test:

1. **Full conditional distribution capture** - Do we get the full distribution, or just the mode?
2. **Uncertainty calibration** - Are prediction intervals correctly calibrated?
3. **Variance preservation** - Does synthetic data have appropriate spread?

This is critical for microplex because normalizing flows can exhibit **mode collapse** - where the model learns to predict only the most likely value rather than the full distribution.

## New Metrics

### 1. Quantile Loss / Pinball Loss

**What it measures:** How well predicted quantiles match the true distribution.

**Formula:**
```
For quantile q:
  pinball_loss(y_true, y_pred, q) = mean(max(q * (y_true - y_pred), (q-1) * (y_true - y_pred)))
```

**Why it matters:**
- Tests if the model captures the **full distribution**, not just the median
- Asymmetric loss function penalizes errors differently based on quantile
- Lower values indicate better quantile matching

**Interpretation:**
- Compute for q = [0.1, 0.25, 0.5, 0.75, 0.9]
- Average across quantiles approximates CRPS

### 2. CRPS (Continuous Ranked Probability Score)

**What it measures:** Proper scoring rule for probabilistic forecasts.

**Formula:**
```
CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
```
where X, X' are independent samples from predicted distribution F.

**Why it matters:**
- **Proper scoring rule** - cannot be gamed by miscalibration
- Measures integral of squared differences between predicted CDF and true value
- CRPS = 0 for perfect predictions

**Interpretation:**
- Lower is better
- Equivalent to MAE for deterministic forecasts
- Penalizes both bias and miscalibration

### 3. Prediction Interval Coverage

**What it measures:** Calibration of uncertainty intervals.

**Process:**
1. Generate N samples for each conditioning context
2. Compute prediction intervals (50%, 80%, 90%) from samples
3. Check if true values fall within intervals
4. Compare actual coverage to target coverage

**Why it matters:**
- Well-calibrated model: 90% interval should contain 90% of true values
- Detects under-confidence (intervals too wide) or over-confidence (too narrow)

**Interpretation:**
- **Calibration error** = |target_coverage - actual_coverage|
- Should be close to 0 for well-calibrated models
- Also reports mean interval width

### 4. Variance Ratio

**What it measures:** Whether synthetic data has appropriate spread.

**Formula:**
```
variance_ratio = var(synthetic) / var(real)
```

**Why it matters:**
- **Ratio < 1:** Under-dispersed (mode collapse)
- **Ratio > 1:** Over-dispersed
- **Ratio ≈ 1:** Good variance matching

**Interpretation:**
- Should be in [0.8, 1.2] for good preservation
- QRF shows severe under-dispersion (0.0-0.1)
- microplex shows some over-dispersion (1.3-6.4)

### 5. Conditional Variance Preservation

**What it measures:** Within-group variance preservation across demographic subgroups.

**Process:**
1. Bin data by demographic variables (age, education, region)
2. Compute variance within each group
3. Compare synthetic vs real within-group variance
4. Average error across groups

**Why it matters:**
- Tests if model captures **heteroscedasticity**
- Example: Young people may have more variable income than old people
- Critical for policy analysis where subgroup behavior matters

**Interpretation:**
- Lower error = better preservation of conditional variance structure
- QRF: ~0.9 (good)
- microplex: ~5.6 (poor)

## Benchmark Results

### Key Findings

**Question 1: Does microplex capture full conditional distribution?**

**Partial.** Results show:
- **CRPS:** microplex best (27,980 vs 31,373 for QRF)
- **Variance ratios:** microplex is over-dispersed (1.3-6.4x real variance)
- **Prediction intervals:** Well-calibrated (90% interval error = 0.04)

**Conclusion:** microplex captures distributional uncertainty but over-estimates variance.

**Question 2: Is uncertainty calibrated correctly?**

**Yes.** Prediction interval calibration:
- **microplex:** 90% interval error = 0.04 (excellent)
- **QRF sequential:** 90% interval error = 0.46 (poor)
- **QRF zero-inflation:** 90% interval error = 0.15 (moderate)

**Conclusion:** microplex's normalizing flow produces well-calibrated uncertainty.

**Question 3: Does it collapse to modal predictions?**

**No.** Evidence:
- Variance ratios > 1 (over-dispersed, not collapsed)
- CRPS best among methods (captures full distribution)
- Quantile losses competitive across all quantiles

**Conclusion:** No mode collapse. If anything, microplex is **too** spread out.

### Comparison Table

| Metric | QRF Sequential | QRF Zero-Inflation | microplex | Winner |
|--------|----------------|--------------------|-----------|----|
| **CRPS** ↓ | 31,373 | 29,372 | **27,980** | microplex |
| **Variance Ratio Error** ↓ | **0.98** | 0.95 | 2.07 | QRF |
| **90% Interval Calibration** ↓ | 0.46 | 0.15 | **0.04** | microplex |
| **Conditional Variance Error** ↓ | 0.97 | **0.88** | 5.55 | QRF+ZI |
| **Mean Quantile Loss** ↓ | 15,801 | 14,710 | **13,827** | microplex |

**↓** = lower is better

### Critical Weaknesses Identified

**QRF (both variants):**
- **Severe under-dispersion:** Variance ratios 0.0-0.1 (100x too narrow!)
- **Poor calibration:** 90% intervals contain only 40-50% of true values
- **Mode collapse:** Predicting near-constant values within groups

**microplex:**
- **Over-dispersion:** Variance ratios 1.3-6.4 (too spread out)
- **Poor conditional variance:** Within-group variance error ~5.6
- **Wide intervals:** Prediction intervals much wider than necessary

## Recommendations

### For microplex

**Issue:** Over-dispersion and poor conditional variance preservation.

**Potential fixes:**
1. **Add variance regularization** to normalizing flow training
2. **Conditional batch normalization** to preserve variance structure
3. **Variance-matching loss** in addition to likelihood loss
4. **Tune temperature parameter** for sampling from flow

### For QRF

**Issue:** Severe under-dispersion and mode collapse.

**Root cause:** Sequential QRF uses median prediction (quantile=0.5) without noise injection that scales with conditional variance.

**Current behavior:**
```python
# QRF generates essentially deterministic predictions
predictions = model.predict(X)  # Always predicts median
predictions += np.random.normal(0, fixed_noise_scale)  # Fixed noise
```

**Why this fails:**
- Fixed noise scale doesn't adapt to conditional heteroscedasticity
- No mechanism to preserve within-group variance
- Sequential chaining propagates errors without variance correction

## Files Created

### Core Metrics Module
- `/Users/maxghenis/PolicyEngine/micro/benchmarks/metrics.py`
  - All distributional quality metrics
  - Comprehensive evaluation function
  - Report generation utilities

### Benchmark Scripts
- `/Users/maxghenis/PolicyEngine/micro/benchmarks/run_distributional_benchmark.py`
  - Runs full distributional quality benchmark
  - Generates visualizations and report
  - Compares QRF vs microplex

### Results
- `/Users/maxghenis/PolicyEngine/micro/benchmarks/results/distributional_quality.md`
  - Full analysis report
- `/Users/maxghenis/PolicyEngine/micro/benchmarks/results/distributional_metrics.json`
  - Raw metrics in JSON format
- `/Users/maxghenis/PolicyEngine/micro/benchmarks/results/distributional_*.png`
  - 5 visualization files

## Usage

```bash
cd /Users/maxghenis/PolicyEngine/micro

# Run full distributional benchmark
/Users/maxghenis/PolicyEngine/micro/.venv/bin/python benchmarks/run_distributional_benchmark.py

# Results will be saved to benchmarks/results/
```

## Next Steps

1. **Investigate microplex over-dispersion:**
   - Why are variance ratios 1.3-6.4x instead of ~1.0?
   - Is this a temperature/sampling issue?
   - Does it happen on real CPS data or just synthetic benchmarks?

2. **Add variance regularization:**
   - Implement variance-matching loss
   - Test temperature tuning for flow sampling
   - Consider conditional batch normalization

3. **Validate on real data:**
   - Run distributional metrics on CPS enhancement
   - Compare against IRS statistics for validation
   - Check if over-dispersion affects downstream policy analysis

4. **Fix QRF for fair comparison:**
   - Implement proper quantile sampling with variance scaling
   - Add heteroscedastic noise injection
   - Rerun benchmark with fixed QRF

## References

- **CRPS:** Gneiting & Raftery (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation"
- **Quantile Loss:** Koenker & Bassett (1978). "Regression Quantiles"
- **Calibration:** Gneiting et al. (2007). "Probabilistic forecasts, calibration and sharpness"

---

**Created:** December 25, 2024
**Location:** `/Users/maxghenis/PolicyEngine/micro/benchmarks/DISTRIBUTIONAL_METRICS.md`
