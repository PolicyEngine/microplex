#!/usr/bin/env python3
"""Generate reweighting frontier figure: records used vs out-of-sample error.

Reads frontier data and produces a figure showing the accuracy-sparsity
tradeoff. SparseCalibrator (convex, deterministic) traces a reliable frontier.
HardConcrete (non-convex) is shown with error bars from multi-seed runs.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent.parent / "benchmarks" / "results"
FRONTIER_PATH = RESULTS_DIR / "reweighting_frontier.json"
HC_MULTISEED_PATH = RESULTS_DIR / "reweighting_frontier_hc_multiseed.json"
OUTPUT_PATH = Path(__file__).parent / "reweighting_frontier.pdf"


def main():
    with open(FRONTIER_PATH) as f:
        data = json.load(f)

    n_records = data["n_records"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # --- SparseCalibrator frontier (L1 family, convex/deterministic) ---
    sc = data["methods"]["SparseCalibrator"]
    sc_x = [p["n_active"] for p in sc]
    sc_y = [p["test_error"] for p in sc]
    order = np.argsort(sc_x)
    sc_x = [sc_x[i] for i in order]
    sc_y = [sc_y[i] for i in order]
    ax.plot(sc_x, sc_y, "s-", color="#2ca02c", linewidth=1.5, markersize=5,
            label=r"SparseCalibrator ($L_1$, convex)", zorder=3)

    # L1-Sparse endpoint
    if "L1-Sparse" in data["methods"]:
        l1 = data["methods"]["L1-Sparse"][0]
        ax.scatter(l1["n_active"], l1["test_error"], marker="s", color="#2ca02c",
                   s=100, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate("L1-Sparse\n(hard constraints)", (l1["n_active"], l1["test_error"]),
                    textcoords="offset points", xytext=(15, 8), fontsize=7,
                    color="#2ca02c", ha="left")

    # --- HardConcrete frontier (L0 family, non-convex, with error bars) ---
    if HC_MULTISEED_PATH.exists():
        with open(HC_MULTISEED_PATH) as f:
            hc_ms = json.load(f)
        hc_x = [p["n_active_mean"] for p in hc_ms]
        hc_y = [p["test_error_mean"] for p in hc_ms]
        hc_se = [p["test_error_se"] for p in hc_ms]
        order = np.argsort(hc_x)
        hc_x = [hc_x[i] for i in order]
        hc_y = [hc_y[i] for i in order]
        hc_se = [hc_se[i] for i in order]
        ax.errorbar(hc_x, hc_y, yerr=hc_se, fmt="o-", color="#1f77b4",
                    linewidth=1.5, markersize=5, capsize=3, capthick=1,
                    label=r"HardConcrete ($L_0$, non-convex)", zorder=3)
    else:
        # Fallback: single-seed data
        hc = data["methods"]["HardConcrete"]
        hc_x = [p["n_active"] for p in hc]
        hc_y = [p["test_error"] for p in hc]
        order = np.argsort(hc_x)
        hc_x = [hc_x[i] for i in order]
        hc_y = [hc_y[i] for i in order]
        ax.plot(hc_x, hc_y, "o-", color="#1f77b4", linewidth=1.5, markersize=5,
                label=r"HardConcrete ($L_0$, non-convex)", zorder=3)

    # L0-Sparse endpoint
    if "L0-Sparse" in data["methods"]:
        l0 = data["methods"]["L0-Sparse"][0]
        ax.scatter(l0["n_active"], l0["test_error"], marker="o", color="#1f77b4",
                   s=100, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate("L0-Sparse\n(hard constraints)", (l0["n_active"], l0["test_error"]),
                    textcoords="offset points", xytext=(15, -18), fontsize=7,
                    color="#1f77b4", ha="left")

    # --- Dense methods as reference points ---
    for name, marker, color in [
        ("IPF", "D", "#d62728"),
        ("Entropy", "^", "#ff7f0e"),
    ]:
        if name in data["methods"]:
            pts = data["methods"][name]
            for p in pts:
                ax.scatter(p["n_active"], p["test_error"], marker=marker, color=color,
                           s=80, zorder=4, label=name, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Active records (non-zero weight)", fontsize=11)
    ax.set_ylabel("Out-of-sample error\n(held-out sex margin)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(4, n_records * 1.3)

    # Compute y-axis limits from all data
    all_errors = []
    for method_data in data["methods"].values():
        for p in method_data:
            all_errors.append(p["test_error"])
    ax.set_ylim(0, max(all_errors) * 1.15)

    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add annotation for total records
    ax.axvline(n_records, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(n_records * 0.85, ax.get_ylim()[1] * 0.95, f"N={n_records:,}",
            ha="right", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH} and {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
