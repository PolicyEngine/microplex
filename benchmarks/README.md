# microplex benchmarks

Synthesis method comparison using PRDC (Precision, Recall, Density, Coverage) metrics from [Naeem et al. (2020)](https://arxiv.org/abs/2002.09797), evaluated via the canonical [`prdc`](https://github.com/clovaai/generative-evaluation-prdc) library.

## Methods compared

Six synthesis methods — QRF, QDNN, and MAF, each with and without zero-inflation (ZI) handling — are evaluated against holdouts from three source surveys (SIPP, CPS ASEC, PSID).

| Method | Description |
|--------|-------------|
| **QRF** | Quantile regression forest |
| **ZI-QRF** | Zero-inflated QRF (two-stage hurdle) |
| **QDNN** | Quantile deep neural network (pinball loss) |
| **ZI-QDNN** | Zero-inflated QDNN |
| **MAF** | Masked autoregressive flow (1D per column) |
| **ZI-MAF** | Zero-inflated MAF |

## Running benchmarks

```bash
# Install dependencies
pip install microplex[benchmark]

# Single-seed run
python scripts/run_benchmark.py --output benchmarks/results/benchmark_full.json

# Multi-seed run (for paper)
python scripts/run_benchmark.py --n-seeds 10 --output benchmarks/results/benchmark_multi_seed.json

# Fast mode for testing
python scripts/run_benchmark.py --fast --output /tmp/benchmark_test.json
```

## Key findings

See the [paper](../paper/) for full analysis. Summary:

- **Zero-inflation handling matters more than base model choice** for economic data with mass-at-zero variables. ZI lifts MAF and QDNN coverage substantially while barely affecting QRF (which handles mixed distributions natively via leaf node composition).
- **Per-source coverage varies dramatically**: SIPP and CPS achieve meaningful coverage; PSID shows 0% coverage with only 2 shared conditioning variables (age, sex).
- **Speed-accuracy tradeoff**: ZI-QRF is fastest with competitive coverage; ZI-MAF is slowest but achieves the highest CPS coverage.

## Output structure

```
benchmarks/results/
├── benchmark_full.json          # Single-seed PRDC results
├── benchmark_multi_seed.json    # Multi-seed means +/- SE
├── reweighting_full.json        # Calibration method comparison
└── *.png                        # Visualization charts
```

## Citation

```bibtex
@software{microplex2025,
  author = {Ghenis, Max},
  title = {microplex: Multi-source microdata synthesis and survey reweighting},
  year = {2025},
  url = {https://github.com/CosilicoAI/microplex}
}
```
