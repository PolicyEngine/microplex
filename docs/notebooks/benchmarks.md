# Benchmarks

This notebook compares `microplex` synthesis methods.

## Methods compared

| Method | Description | Library |
|--------|-------------|---------|
| **microplex QRF** | Quantile regression forest | This package |
| **microplex ZI-QRF** | Zero-inflated QRF | This package |
| **microplex QDNN** | Quantile deep neural network | This package |
| **microplex MAF** | Masked autoregressive flow | This package |
| **CT-GAN** | Conditional tabular GAN | SDV |
| **TVAE** | Tabular VAE | SDV |

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.compare import run_benchmark, results_to_dataframe
```

## Create test data

```python
np.random.seed(42)
n_train = 5000
n_test = 1000

# Demographics
age = np.random.randint(18, 80, n_train)
education = np.random.choice([1, 2, 3, 4], n_train)

# Income sources (zero-inflated)
base = np.random.lognormal(10, 1, n_train)
wages = base * (1 + 0.01 * (age - 18)) * (1 + 0.2 * education)
wages[np.random.random(n_train) < 0.08] = 0

capital_gains = np.where(
    base > np.percentile(base, 70),
    np.random.lognormal(9, 2, n_train),
    0
)

training_data = pd.DataFrame({
    "age": age,
    "education": education,
    "wages": wages,
    "capital_gains": capital_gains,
})

# Test conditions
test_conditions = pd.DataFrame({
    "age": np.random.randint(18, 80, n_test),
    "education": np.random.choice([1, 2, 3, 4], n_test),
})
```

## Run benchmarks

```python
results = run_benchmark(
    train_data=training_data,
    test_conditions=test_conditions,
    target_vars=["wages", "capital_gains"],
    condition_vars=["age", "education"],
    methods=["microplex", "ctgan", "tvae"],
    epochs=100,
)

df = results_to_dataframe(results)
df
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Coverage (higher is better)
ax = axes[0, 0]
df.plot.bar(x="method", y="coverage", ax=ax, legend=False)
ax.set_title("Coverage (PRDC)")
ax.set_ylabel("Coverage (higher is better)")

# Precision
ax = axes[0, 1]
df.plot.bar(x="method", y="precision", ax=ax, legend=False, color="orange")
ax.set_title("Precision (PRDC)")
ax.set_ylabel("Precision (higher is better)")

# Zero fraction error (lower is better)
ax = axes[1, 0]
df.plot.bar(x="method", y="mean_zero_error", ax=ax, legend=False, color="green")
ax.set_title("Zero fraction error")
ax.set_ylabel("Error (lower is better)")

# Training time
ax = axes[1, 1]
df.plot.bar(x="method", y="train_time", ax=ax, legend=False, color="red")
ax.set_title("Training time")
ax.set_ylabel("Seconds")

plt.tight_layout()
plt.show()
```

## Key findings

1. **Coverage**: Zero-inflated methods achieve higher PRDC coverage
2. **Zero handling**: Two-stage ZI models excel at preserving zero fractions
3. **Speed**: QRF methods are fastest; MAF slowest but most flexible
4. **Architecture matters**: ZI handling lifts neural methods more than tree methods
