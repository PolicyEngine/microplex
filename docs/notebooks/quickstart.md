# Quickstart

This notebook demonstrates basic usage of `microplex`.

```python
# Install microplex
# !pip install microplex
```

```python
import numpy as np
import pandas as pd
from microplex import Synthesizer
```

## Create sample data

```python
np.random.seed(42)
n = 1000

# Demographics
age = np.random.randint(18, 80, n)
education = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2])

# Income (depends on demographics)
base_income = np.random.lognormal(10, 1, n)
income = base_income * (1 + 0.01 * (age - 18)) * (1 + 0.2 * education)
income[np.random.random(n) < 0.1] = 0  # 10% zero income

training_data = pd.DataFrame({
    "age": age,
    "education": education,
    "income": income,
    "weight": np.ones(n),
})

training_data.describe()
```

## Train synthesizer

```python
synth = Synthesizer(
    target_vars=["income"],
    condition_vars=["age", "education"],
)

synth.fit(training_data, epochs=50)
```

## Generate synthetic data

```python
# New demographics
test_conditions = pd.DataFrame({
    "age": [25, 35, 45, 55, 65],
    "education": [2, 3, 4, 4, 3],
})

synthetic = synth.generate(test_conditions, seed=42)
synthetic
```

## Validate results

```python
# Check zero fractions match
print(f"Training zero fraction: {(training_data['income'] == 0).mean():.2%}")
print(f"Synthetic zero fraction: {(synthetic['income'] == 0).mean():.2%}")
```

```python
# Generate larger sample for distribution comparison
large_conditions = training_data[["age", "education"]].copy()
large_synthetic = synth.generate(large_conditions)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Log income distribution
axes[0].hist(np.log1p(training_data["income"]), bins=50, alpha=0.5, label="Real")
axes[0].hist(np.log1p(large_synthetic["income"]), bins=50, alpha=0.5, label="Synthetic")
axes[0].set_xlabel("Log(1 + income)")
axes[0].legend()
axes[0].set_title("Income distribution")

# Age vs income
axes[1].scatter(training_data["age"], training_data["income"], alpha=0.3, label="Real")
axes[1].scatter(large_synthetic["age"], large_synthetic["income"], alpha=0.3, label="Synthetic")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Income")
axes[1].legend()
axes[1].set_title("Age vs income")

plt.tight_layout()
plt.show()
```
