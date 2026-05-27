# Distributional Quality Benchmark: QRF vs microplex

**Date:** December 25, 2024

## Overview

This benchmark tests whether synthetic data methods capture the **full conditional distribution**, not just the mode. Key questions:

1. Does the method produce properly calibrated uncertainty?
2. Does it capture the full distribution (not collapse to mode)?
3. Are prediction intervals correctly calibrated?

## Executive Summary

- **Best CRPS (distributional forecast):** microplex
- **Best Variance Preservation:** qrf_zero_inflation
- **Best Conditional Variance:** qrf_zero_inflation

## Results Summary

| Method | Mean CRPS ↓ | Var Ratio Error ↓ | Cond Var Error ↓ |
|--------|-------------|-------------------|-------------------|
| qrf_sequential | 31372.93 | 0.9790 | 0.9727 |
| qrf_zero_inflation | 29371.72 | 0.9482 | 0.8753 |
| microplex | 27980.14 | 2.0730 | 5.5507 |

**↓** = lower is better

## Detailed Metrics

### 1. CRPS (Continuous Ranked Probability Score)

**What it measures:** Proper scoring rule for probabilistic forecasts. Measures integral of squared differences between predicted CDF and true value.

**Interpretation:** Lower is better. CRPS = 0 for perfect predictions.

**qrf_sequential:**
  - income: 32560.48
  - assets: 75154.58
  - debt: 10186.18
  - savings: 7590.48
  - **Mean:** 31372.93

**qrf_zero_inflation:**
  - income: 32530.77
  - assets: 69257.59
  - debt: 8657.36
  - savings: 7041.15
  - **Mean:** 29371.72

**microplex:**
  - income: 29002.76
  - assets: 68044.53
  - debt: 8169.22
  - savings: 6704.06
  - **Mean:** 27980.14

### 2. Variance Ratios (Synthetic / Real)

**What it measures:** Whether synthetic data has appropriate spread.

**Interpretation:**
- Ratio < 1: Under-dispersed (mode collapse)
- Ratio > 1: Over-dispersed
- Ratio ≈ 1: Good variance matching

**qrf_sequential:**
  - income: 0.063 ✗
  - assets: 0.003 ✗
  - debt: 0.000 ✗
  - savings: 0.017 ✗

**qrf_zero_inflation:**
  - income: 0.063 ✗
  - assets: 0.034 ✗
  - debt: 0.046 ✗
  - savings: 0.065 ✗

**microplex:**
  - income: 1.348 ✗
  - assets: 6.383 ✗
  - debt: 3.033 ✗
  - savings: 1.528 ✗

### 3. Prediction Interval Coverage

**What it measures:** Calibration of uncertainty intervals.

**Interpretation:** A well-calibrated model should have actual coverage match target. E.g., 90% interval should contain 90% of true values.

#### 50% Intervals

**qrf_sequential:**
  - income:
    - Target: 50.0%
    - Actual: 16.8%
    - Calibration error: 0.3320
    - Mean width: 16534.96
  - assets:
    - Target: 50.0%
    - Actual: 10.0%
    - Calibration error: 0.4000
    - Mean width: 12858.50
  - debt:
    - Target: 50.0%
    - Actual: 49.2%
    - Calibration error: 0.0080
    - Mean width: 107.39
  - savings:
    - Target: 50.0%
    - Actual: 12.6%
    - Calibration error: 0.3740
    - Mean width: 2158.97

**qrf_zero_inflation:**
  - income:
    - Target: 50.0%
    - Actual: 17.0%
    - Calibration error: 0.3300
    - Mean width: 16697.93
  - assets:
    - Target: 50.0%
    - Actual: 70.8%
    - Calibration error: 0.2080
    - Mean width: 60744.58
  - debt:
    - Target: 50.0%
    - Actual: 72.4%
    - Calibration error: 0.2240
    - Mean width: 9736.76
  - savings:
    - Target: 50.0%
    - Actual: 19.2%
    - Calibration error: 0.3080
    - Mean width: 3727.13

**microplex:**
  - income:
    - Target: 50.0%
    - Actual: 52.2%
    - Calibration error: 0.0220
    - Mean width: 61365.75
  - assets:
    - Target: 50.0%
    - Actual: 45.4%
    - Calibration error: 0.0460
    - Mean width: 115573.92
  - debt:
    - Target: 50.0%
    - Actual: 69.8%
    - Calibration error: 0.1980
    - Mean width: 7875.46
  - savings:
    - Target: 50.0%
    - Actual: 52.4%
    - Calibration error: 0.0240
    - Mean width: 15322.55

#### 80% Intervals

**qrf_sequential:**
  - income:
    - Target: 80.0%
    - Actual: 34.8%
    - Calibration error: 0.4520
    - Mean width: 35212.74
  - assets:
    - Target: 80.0%
    - Actual: 17.8%
    - Calibration error: 0.6220
    - Mean width: 26253.68
  - debt:
    - Target: 80.0%
    - Actual: 49.4%
    - Calibration error: 0.3060
    - Mean width: 206.42
  - savings:
    - Target: 80.0%
    - Actual: 22.0%
    - Calibration error: 0.5800
    - Mean width: 4063.68

**qrf_zero_inflation:**
  - income:
    - Target: 80.0%
    - Actual: 35.4%
    - Calibration error: 0.4460
    - Mean width: 35413.79
  - assets:
    - Target: 80.0%
    - Actual: 75.8%
    - Calibration error: 0.0420
    - Mean width: 80975.53
  - debt:
    - Target: 80.0%
    - Actual: 76.2%
    - Calibration error: 0.0380
    - Mean width: 12176.65
  - savings:
    - Target: 80.0%
    - Actual: 45.4%
    - Calibration error: 0.3460
    - Mean width: 8955.80

**microplex:**
  - income:
    - Target: 80.0%
    - Actual: 78.6%
    - Calibration error: 0.0140
    - Mean width: 133784.17
  - assets:
    - Target: 80.0%
    - Actual: 91.4%
    - Calibration error: 0.1140
    - Mean width: 334450.26
  - debt:
    - Target: 80.0%
    - Actual: 82.6%
    - Calibration error: 0.0260
    - Mean width: 18695.28
  - savings:
    - Target: 80.0%
    - Actual: 75.2%
    - Calibration error: 0.0480
    - Mean width: 33563.81

#### 90% Intervals

**qrf_sequential:**
  - income:
    - Target: 90.0%
    - Actual: 43.6%
    - Calibration error: 0.4640
    - Mean width: 46119.43
  - assets:
    - Target: 90.0%
    - Actual: 22.6%
    - Calibration error: 0.6740
    - Mean width: 34474.18
  - debt:
    - Target: 90.0%
    - Actual: 49.6%
    - Calibration error: 0.4040
    - Mean width: 335.90
  - savings:
    - Target: 90.0%
    - Actual: 27.6%
    - Calibration error: 0.6240
    - Mean width: 5227.66

**qrf_zero_inflation:**
  - income:
    - Target: 90.0%
    - Actual: 43.4%
    - Calibration error: 0.4660
    - Mean width: 46242.16
  - assets:
    - Target: 90.0%
    - Actual: 77.0%
    - Calibration error: 0.1300
    - Mean width: 93065.31
  - debt:
    - Target: 90.0%
    - Actual: 77.8%
    - Calibration error: 0.1220
    - Mean width: 13504.82
  - savings:
    - Target: 90.0%
    - Actual: 56.0%
    - Calibration error: 0.3400
    - Mean width: 11877.35

**microplex:**
  - income:
    - Target: 90.0%
    - Actual: 89.0%
    - Calibration error: 0.0100
    - Mean width: 196885.81
  - assets:
    - Target: 90.0%
    - Actual: 96.4%
    - Calibration error: 0.0640
    - Mean width: 594702.68
  - debt:
    - Target: 90.0%
    - Actual: 88.8%
    - Calibration error: 0.0120
    - Mean width: 29555.85
  - savings:
    - Target: 90.0%
    - Actual: 81.6%
    - Calibration error: 0.0840
    - Mean width: 46261.58

### 4. Conditional Variance Preservation

**What it measures:** Within-group variance preservation across demographic subgroups.

**Interpretation:** Tests if the model captures heteroscedasticity. Lower error = better preservation of conditional variance structure.

**qrf_sequential:**
  - income: 0.9357
  - assets: 0.9838
  - debt: 0.9967
  - savings: 0.9746
  - **Mean:** 0.9727

**qrf_zero_inflation:**
  - income: 0.9335
  - assets: 0.8336
  - debt: 0.8523
  - savings: 0.8820
  - **Mean:** 0.8753

**microplex:**
  - income: 1.7575
  - assets: 13.2266
  - debt: 4.6958
  - savings: 2.5229
  - **Mean:** 5.5507

### 5. Quantile Losses (Pinball Loss)

**What it measures:** How well predicted quantiles match true distribution.

**Interpretation:** Lower loss = better quantile matching. Tests if method captures full distribution (not just median).

**qrf_sequential:**

| Variable | q=0.10 | q=0.25 | q=0.50 | q=0.75 | q=0.90 | Mean |
|----------|--------|--------|--------|--------|--------|------|
| income | 7829 | 13535 | 19352 | 21720 | 19888 | 16465 |
| assets | 10126 | 22542 | 40199 | 55160 | 60970 | 37799 |
| debt | 1029 | 2572 | 5144 | 7686 | 9170 | 5120 |
| savings | 2093 | 3049 | 4219 | 4934 | 4802 | 3819 |

**qrf_zero_inflation:**

| Variable | q=0.10 | q=0.25 | q=0.50 | q=0.75 | q=0.90 | Mean |
|----------|--------|--------|--------|--------|--------|------|
| income | 7666 | 13529 | 19351 | 21692 | 19836 | 16415 |
| assets | 8186 | 20465 | 41590 | 51617 | 51173 | 34606 |
| debt | 1029 | 2572 | 5144 | 6459 | 6198 | 4280 |
| savings | 1895 | 3042 | 4181 | 4544 | 4026 | 3538 |

**microplex:**

| Variable | q=0.10 | q=0.25 | q=0.50 | q=0.75 | q=0.90 | Mean |
|----------|--------|--------|--------|--------|--------|------|
| income | 5770 | 12471 | 19351 | 19946 | 14181 | 14344 |
| assets | 8186 | 21215 | 41143 | 51718 | 45408 | 33534 |
| debt | 1029 | 2572 | 5144 | 6471 | 5400 | 4123 |
| savings | 1591 | 2917 | 4223 | 4452 | 3331 | 3303 |

## Key Findings

### Question 1: Does microplex capture full conditional distribution?

**Partial.** Some variables show variance ratio issues, suggesting potential under/over-dispersion.

### Question 2: Is uncertainty calibrated correctly?

**Yes.** Mean 90% interval calibration error is 0.0425, indicating well-calibrated uncertainty.

### Question 3: How does QRF compare?

**Comparison:**

- **qrf_sequential vs microplex:**
  - CRPS: 31372.93 vs 27980.14 (1.12x)
  - Variance error: 0.9790 vs 2.0730 (0.47x)

- **qrf_zero_inflation vs microplex:**
  - CRPS: 29371.72 vs 27980.14 (1.05x)
  - Variance error: 0.9482 vs 2.0730 (0.46x)

## Visualizations

1. `distributional_variance_ratios.png` - Variance preservation
2. `distributional_crps.png` - CRPS comparison
3. `distributional_calibration.png` - Prediction interval calibration
4. `distributional_conditional_variance.png` - Conditional variance
5. `distributional_quantile_losses.png` - Quantile loss heatmaps

## Reproducibility

```bash
cd /Users/maxghenis/PolicyEngine/micro
python benchmarks/run_distributional_benchmark.py
```

---

**Generated:** December 25, 2024
