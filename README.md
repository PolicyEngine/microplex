# microplex

Multi-source microdata synthesis and survey reweighting.

[![PyPI](https://img.shields.io/pypi/v/microplex.svg)](https://pypi.org/project/microplex/)
[![Tests](https://github.com/CosilicoAI/microplex/actions/workflows/test.yml/badge.svg)](https://github.com/CosilicoAI/microplex/actions/workflows/test.yml)
[![Docs](https://github.com/CosilicoAI/microplex/actions/workflows/docs.yml/badge.svg)](https://cosilicoai.github.io/microplex)

## Overview

`microplex` creates rich, calibrated microdata through:

- **Conditional relationships**: Generate target variables given demographics
- **Zero-inflated distributions**: Handle variables that are 0 for many observations
- **Multi-source fusion**: Combine variables from surveys with different variable sets

## Installation

```bash
pip install microplex
```

## Quick Start

### Synthesis

```python
from microplex import Synthesizer
import pandas as pd

# Load training data with known target variables
training_data = pd.read_csv("survey_with_income.csv")

# Initialize synthesizer
synth = Synthesizer(
    target_vars=["income", "expenditure", "savings"],
    condition_vars=["age", "education", "region"],
)

# Fit on training data
synth.fit(training_data, weight_col="weight", epochs=100)

# Generate synthetic targets for new demographics
new_demographics = pd.read_csv("demographics_only.csv")
synthetic = synth.generate(new_demographics)
```

### Reweighting

```python
from microplex import Reweighter
import pandas as pd

# Load synthetic microdata
synthetic_data = pd.read_csv("synthetic_microdata.csv")

# Define population targets (official statistics)
targets = {
    "state": {"CA": 39_500_000, "NY": 19_500_000, "TX": 29_000_000},
    "age_group": {"0-17": 22_000_000, "18-64": 54_000_000, "65+": 12_000_000},
}

# Sparse reweighting to match targets
reweighter = Reweighter(sparsity="l0")  # Minimize number of records used
weighted_data = reweighter.fit_transform(synthetic_data, targets)

# Check sparsity
stats = reweighter.get_sparsity_stats()
print(f"Using {stats['n_nonzero']} of {stats['n_records']} records")
```

## Why `microplex`?

| Feature | microplex | synthpop |
|---------|-------|----------|
| Multi-source fusion | ✅ | ❌ |
| Zero-inflation handling | ✅ | ⚠️ |
| Multiple synthesis methods | ✅ (QRF, QDNN, MAF) | ✅ (CART) |
| Survey reweighting | ✅ (IPF, entropy, sparse) | ❌ |
| PRDC evaluation | ✅ | ❌ |

### Use Cases

**Synthesis:**
- **Survey enhancement**: Impute income variables from tax data onto census demographics
- **Privacy-preserving synthesis**: Generate synthetic data that preserves statistical properties without copying real records
- **Data fusion**: Combine variables from multiple surveys with different sample designs
- **Missing data imputation**: Fill in missing values conditioned on observed variables

**Reweighting:**
- **Population calibration**: Match synthetic microdata to official population statistics (Census, ACS)
- **Geographic targeting**: Ensure correct distributions at state, county, and tract levels
- **Sparse sampling**: Minimize number of records while preserving statistical accuracy
- **Survey weighting**: Adjust sample weights to match known population margins

## Architecture

### Synthesizer

```
┌─────────────────────────────────────────────────────────┐
│                      Synthesizer                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Training:                                               │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Training │───▶│ Transformer  │───▶│ Normalizing  │  │
│  │   Data   │    │ (log, std)   │    │    Flow      │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
│                                                          │
│  Generation:                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Context  │───▶│ Zero + Flow  │───▶│  Inverse     │  │
│  │  Vars    │    │   Sampling   │    │  Transform   │  │
│  └──────────┘    └──────────────┘    └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Reweighter

```
┌─────────────────────────────────────────────────────────┐
│                      Reweighter                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Optimization Problem:                                   │
│     min ||w||_p  subject to  A @ w = b                  │
│                                                          │
│  where:                                                  │
│    w: weight vector (decision variables)                │
│    A: constraint matrix (margin indicators)             │
│    b: target vector (population totals)                 │
│    p: sparsity norm (0, 1, or 2)                        │
│                                                          │
│  Backends:                                               │
│    - scipy:  L1/L2 (linprog, minimize)                  │
│    - cvxpy:  L0/L1/L2 (convex optimization)             │
│                                                          │
│  L0 Approximation:                                       │
│    Iterative Reweighted L1 (IRL1)                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Documentation

Full documentation at [cosilicoai.github.io/microplex](https://cosilicoai.github.io/microplex)

- [Tutorial](https://cosilicoai.github.io/microplex/tutorial.html)
- [API Reference](https://cosilicoai.github.io/microplex/api.html)
- [Benchmarks](https://cosilicoai.github.io/microplex/benchmarks.html)

## Benchmarks

See [benchmarks/](benchmarks/) for synthesis method comparisons:

- **QRF / ZI-QRF**: Quantile regression forests (with/without zero-inflation)
- **QDNN / ZI-QDNN**: Quantile deep neural networks
- **MAF / ZI-MAF**: Masked autoregressive flows

## Citation

```bibtex
@software{microplex2025,
  author = {Ghenis, Max},
  title = {microplex: Multi-source microdata synthesis and survey reweighting},
  year = {2025},
  url = {https://github.com/CosilicoAI/microplex}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
