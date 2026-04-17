# Tutorial

This tutorial walks through using `microplex` to synthesize income data.

## Problem Setup

Imagine you have:
- A **demographic survey** with many respondents but no income data
- A **tax dataset** with income data but fewer demographics

You want to impute income onto the demographic survey while preserving:
- The relationship between income and demographics (age, education, etc.)
- The joint distribution of multiple income sources
- Zero-inflation (many people have $0 for certain income types)

## Step 1: Load Data

```python
import pandas as pd
from microplex import Synthesizer

# Training data: has both demographics and income
training_data = pd.DataFrame({
    "age": [25, 35, 45, 55, 65],
    "education": [2, 3, 4, 4, 3],
    "wages": [30000, 50000, 80000, 90000, 0],
    "capital_gains": [0, 0, 5000, 20000, 50000],
    "weight": [1000, 1500, 2000, 1800, 1200],
})

# Target demographics: need income imputed
demographics = pd.DataFrame({
    "age": [30, 40, 50, 60],
    "education": [3, 4, 3, 2],
})
```

## Step 2: Configure Synthesizer

Specify which variables to synthesize and which to condition on:

```python
synth = Synthesizer(
    target_vars=["wages", "capital_gains"],  # What to generate
    condition_vars=["age", "education"],      # What to condition on
    zero_inflated=True,                       # Handle zero values
    n_layers=6,                               # Flow depth
    hidden_dim=64,                            # Network width
)
```

## Step 3: Fit Model

Train on the data where you have all variables:

```python
synth.fit(
    training_data,
    weight_col="weight",  # Use sample weights
    epochs=100,
    verbose=True,
)
```

Output:
```
Epoch 10/100, Loss: 5.2341
Epoch 20/100, Loss: 3.1234
...
Epoch 100/100, Loss: 2.0123
```

## Step 4: Generate Synthetic Data

Generate target variables for new demographics:

```python
synthetic = synth.generate(demographics, seed=42)
print(synthetic)
```

Output:
```
   age  education    wages  capital_gains
0   30          3  42156.7            0.0
1   40          4  68234.2         1234.5
2   50          3  55678.9            0.0
3   60          2  28901.2        12345.6
```

## Step 5: Validate Results

Check that the synthesis preserves key properties:

```python
# Zero fractions
print("Zero fraction (wages):")
print(f"  Training: {(training_data['wages'] == 0).mean():.2%}")
print(f"  Synthetic: {(synthetic['wages'] == 0).mean():.2%}")

# Correlations
print("\nCorrelation (age, wages):")
print(f"  Training: {training_data['age'].corr(training_data['wages']):.3f}")
print(f"  Synthetic: {synthetic['age'].corr(synthetic['wages']):.3f}")
```

## Step 6: Save and Load

Save the fitted model for later use:

```python
# Save
synth.save("income_synthesizer.pt")

# Load
loaded = Synthesizer.load("income_synthesizer.pt")
new_synthetic = loaded.generate(more_demographics)
```

## Advanced: Discrete Variables

For binary or categorical target variables:

```python
synth = Synthesizer(
    target_vars=["income"],
    condition_vars=["age", "education"],
    discrete_vars=["employed", "has_business"],  # Binary/categorical
)
```

## Tips

1. **Epochs**: Start with 100, increase if loss is still decreasing
2. **Zero-inflation**: Set `zero_inflated=True` for economic data
3. **Weights**: Always use sample weights for survey data
4. **Seed**: Use `seed` parameter for reproducible generation
