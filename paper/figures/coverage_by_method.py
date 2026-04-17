#!/usr/bin/env python3
"""Generate grouped bar chart of coverage by method and source.

Reads benchmark_multi_seed.json and produces a figure suitable for the paper.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "results" / "benchmark_multi_seed.json"
OUTPUT_PATH = Path(__file__).parent / "coverage_by_method.pdf"

# Method display order and labels
METHOD_ORDER = ["QRF", "ZI-QRF", "QDNN", "ZI-QDNN", "MAF", "ZI-MAF"]
SOURCE_ORDER = ["sipp", "cps"]  # Exclude PSID (always 0%)
SOURCE_LABELS = {"sipp": "SIPP", "cps": "CPS ASEC"}


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    methods_data = data["methods"]
    n_seeds = data.get("n_seeds", 1)

    # Build arrays
    n_methods = len(METHOD_ORDER)
    n_sources = len(SOURCE_ORDER)
    means = np.zeros((n_methods, n_sources))
    ses = np.zeros((n_methods, n_sources))

    for i, method in enumerate(METHOD_ORDER):
        if method not in methods_data:
            continue
        for j, source in enumerate(SOURCE_ORDER):
            if source in methods_data[method]:
                means[i, j] = methods_data[method][source]["mean"]
                ses[i, j] = methods_data[method][source].get("se", 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(n_methods)
    width = 0.35
    colors = ["#4878CF", "#D65F5F"]

    for j, source in enumerate(SOURCE_ORDER):
        offset = (j - 0.5) * width
        bars = ax.bar(
            x + offset,
            means[:, j] * 100,
            width,
            yerr=ses[:, j] * 100,
            label=SOURCE_LABELS[source],
            color=colors[j],
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_ylabel("Coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=30, ha="right")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"PRDC coverage by method and source ({n_seeds} seeds)")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    # Also save PNG for previewing
    fig.savefig(OUTPUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Saved {OUTPUT_PATH}")
    print(f"Saved {OUTPUT_PATH.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
