"""
Run comprehensive distributional quality benchmark with proper metrics.

This extends the QRF vs microplex comparison with metrics that test:
1. Full conditional distribution capture (not just mode)
2. Uncertainty calibration
3. Resemblance to real out-of-sample records

Adds:
- Quantile Loss / Pinball Loss
- CRPS (Continuous Ranked Probability Score)
- Prediction Interval Coverage
- Variance Ratios
- Conditional Variance Checks
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare_qrf import (
    SequentialQRF,
    SequentialQRFWithZeroInflation,
)
from metrics import (
    compute_comprehensive_distributional_metrics,
    print_distributional_metrics_report,
)
from run_benchmarks import generate_realistic_microdata

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def create_distributional_visualizations(
    all_metrics: dict,
    output_dir: Path,
):
    """Create visualizations for distributional metrics comparison."""

    methods = list(all_metrics.keys())
    target_vars = list(all_metrics[methods[0]]['variance_ratios'].keys())

    # Color scheme
    colors = {
        "qrf_sequential": "#e74c3c",
        "qrf_zero_inflation": "#f39c12",
        "microplex": "#27ae60"
    }

    # 1. Variance Ratios (should be close to 1.0)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(target_vars))
    width = 0.25

    for i, method in enumerate(methods):
        ratios = [all_metrics[method]['variance_ratios'][var] for var in target_vars]
        ax.bar(
            x + i * width,
            ratios,
            width,
            label=method,
            color=colors.get(method, 'steelblue'),
            alpha=0.8,
        )

    # Add ideal line at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=1.2, color='gray', linestyle=':', alpha=0.3)

    ax.set_ylabel("Variance Ratio (Synthetic / Real)", fontsize=11)
    ax.set_title("Variance Preservation (closer to 1.0 is better)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([v.capitalize() for v in target_vars])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(2.0, ax.get_ylim()[1]))

    plt.tight_layout()
    plt.savefig(output_dir / "distributional_variance_ratios.png", dpi=300, bbox_inches="tight")
    print(f"Saved variance ratios to {output_dir / 'distributional_variance_ratios.png'}")
    plt.close()

    # 2. CRPS (lower is better)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        crps_values = [all_metrics[method]['crps'][var] for var in target_vars]
        ax.bar(
            x + i * width,
            crps_values,
            width,
            label=method,
            color=colors.get(method, 'steelblue'),
            alpha=0.8,
        )

    ax.set_ylabel("CRPS (lower is better)", fontsize=11)
    ax.set_title("Continuous Ranked Probability Score", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([v.capitalize() for v in target_vars])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distributional_crps.png", dpi=300, bbox_inches="tight")
    print(f"Saved CRPS to {output_dir / 'distributional_crps.png'}")
    plt.close()

    # 3. Prediction Interval Calibration
    coverage_levels = [0.5, 0.8, 0.9]

    fig, axes = plt.subplots(1, len(coverage_levels), figsize=(16, 5))
    fig.suptitle("Prediction Interval Calibration", fontsize=14, fontweight='bold')

    for level_idx, level in enumerate(coverage_levels):
        ax = axes[level_idx]

        # Get calibration errors for each method
        for i, method in enumerate(methods):
            cal_errors = []
            for var in target_vars:
                interval_stats = all_metrics[method]['interval_coverage'][var][level]
                cal_errors.append(interval_stats['calibration_error'])

            ax.bar(
                x + i * width,
                cal_errors,
                width,
                label=method,
                color=colors.get(method, 'steelblue'),
                alpha=0.8,
            )

        ax.set_ylabel("Calibration Error", fontsize=11)
        ax.set_title(f"{level*100:.0f}% Interval (lower is better)", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([v.capitalize() for v in target_vars])
        if level_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distributional_calibration.png", dpi=300, bbox_inches="tight")
    print(f"Saved calibration to {output_dir / 'distributional_calibration.png'}")
    plt.close()

    # 4. Conditional Variance Errors
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        cond_var_errors = [all_metrics[method]['conditional_variance_errors'][var] for var in target_vars]
        ax.bar(
            x + i * width,
            cond_var_errors,
            width,
            label=method,
            color=colors.get(method, 'steelblue'),
            alpha=0.8,
        )

    ax.set_ylabel("Mean Within-Group Variance Error", fontsize=11)
    ax.set_title("Conditional Variance Preservation (lower is better)", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([v.capitalize() for v in target_vars])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distributional_conditional_variance.png", dpi=300, bbox_inches="tight")
    print(f"Saved conditional variance to {output_dir / 'distributional_conditional_variance.png'}")
    plt.close()

    # 5. Quantile Losses Heatmap
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    fig, axes = plt.subplots(1, len(methods), figsize=(16, 5))
    if len(methods) == 1:
        axes = [axes]

    fig.suptitle("Quantile Losses (Pinball Loss) - Lower is Better", fontsize=14, fontweight='bold')

    for method_idx, method in enumerate(methods):
        ax = axes[method_idx]

        # Build matrix: rows = variables, cols = quantiles
        matrix = np.zeros((len(target_vars), len(quantiles)))
        for i, var in enumerate(target_vars):
            for j, q in enumerate(quantiles):
                matrix[i, j] = all_metrics[method]['quantile_losses'][var][q]

        # Normalize for visualization (log scale helps)
        matrix_log = np.log10(matrix + 1)

        im = ax.imshow(matrix_log, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(np.arange(len(quantiles)))
        ax.set_yticks(np.arange(len(target_vars)))
        ax.set_xticklabels([f"{q:.2f}" for q in quantiles])
        ax.set_yticklabels([v.capitalize() for v in target_vars])
        ax.set_xlabel("Quantile", fontsize=10)
        if method_idx == 0:
            ax.set_ylabel("Variable", fontsize=10)
        ax.set_title(method, fontsize=11, color=colors.get(method, 'black'))

        # Add text annotations
        for i in range(len(target_vars)):
            for j in range(len(quantiles)):
                text = ax.text(j, i, f"{matrix[i, j]:.0f}",
                             ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax, label='log10(Loss + 1)')

    plt.tight_layout()
    plt.savefig(output_dir / "distributional_quantile_losses.png", dpi=300, bbox_inches="tight")
    print(f"Saved quantile losses to {output_dir / 'distributional_quantile_losses.png'}")
    plt.close()


def generate_distributional_markdown_report(
    all_metrics: dict,
    output_path: Path,
):
    """Generate comprehensive markdown report."""

    methods = list(all_metrics.keys())
    target_vars = list(all_metrics[methods[0]]['variance_ratios'].keys())

    with open(output_path, "w") as f:
        f.write("# Distributional Quality Benchmark: QRF vs microplex\n\n")
        f.write("**Date:** December 25, 2024\n\n")

        f.write("## Overview\n\n")
        f.write("This benchmark tests whether synthetic data methods capture the **full conditional distribution**, ")
        f.write("not just the mode. Key questions:\n\n")
        f.write("1. Does the method produce properly calibrated uncertainty?\n")
        f.write("2. Does it capture the full distribution (not collapse to mode)?\n")
        f.write("3. Are prediction intervals correctly calibrated?\n\n")

        f.write("## Executive Summary\n\n")

        # Find best for each metric
        best_crps = min(methods, key=lambda m: np.mean(list(all_metrics[m]['crps'].values())))
        best_var_ratio = min(
            methods,
            key=lambda m: np.mean([abs(1.0 - v) for v in all_metrics[m]['variance_ratios'].values()])
        )
        best_cond_var = min(
            methods,
            key=lambda m: np.mean(list(all_metrics[m]['conditional_variance_errors'].values()))
        )

        f.write(f"- **Best CRPS (distributional forecast):** {best_crps}\n")
        f.write(f"- **Best Variance Preservation:** {best_var_ratio}\n")
        f.write(f"- **Best Conditional Variance:** {best_cond_var}\n\n")

        # Summary table
        f.write("## Results Summary\n\n")
        f.write("| Method | Mean CRPS ↓ | Var Ratio Error ↓ | Cond Var Error ↓ |\n")
        f.write("|--------|-------------|-------------------|-------------------|\n")

        for method in methods:
            mean_crps = np.mean(list(all_metrics[method]['crps'].values()))
            var_ratio_error = np.mean([abs(1.0 - v) for v in all_metrics[method]['variance_ratios'].values()])
            cond_var_error = np.mean(list(all_metrics[method]['conditional_variance_errors'].values()))

            f.write(f"| {method} | {mean_crps:.2f} | {var_ratio_error:.4f} | {cond_var_error:.4f} |\n")

        f.write("\n**↓** = lower is better\n\n")

        # Detailed analysis
        f.write("## Detailed Metrics\n\n")

        f.write("### 1. CRPS (Continuous Ranked Probability Score)\n\n")
        f.write("**What it measures:** Proper scoring rule for probabilistic forecasts. ")
        f.write("Measures integral of squared differences between predicted CDF and true value.\n\n")
        f.write("**Interpretation:** Lower is better. CRPS = 0 for perfect predictions.\n\n")

        for method in methods:
            f.write(f"**{method}:**\n")
            for var in target_vars:
                crps = all_metrics[method]['crps'][var]
                f.write(f"  - {var}: {crps:.2f}\n")
            mean_crps = np.mean(list(all_metrics[method]['crps'].values()))
            f.write(f"  - **Mean:** {mean_crps:.2f}\n\n")

        f.write("### 2. Variance Ratios (Synthetic / Real)\n\n")
        f.write("**What it measures:** Whether synthetic data has appropriate spread.\n\n")
        f.write("**Interpretation:**\n")
        f.write("- Ratio < 1: Under-dispersed (mode collapse)\n")
        f.write("- Ratio > 1: Over-dispersed\n")
        f.write("- Ratio ≈ 1: Good variance matching\n\n")

        for method in methods:
            f.write(f"**{method}:**\n")
            for var in target_vars:
                ratio = all_metrics[method]['variance_ratios'][var]
                status = "✓" if 0.8 <= ratio <= 1.2 else "✗"
                f.write(f"  - {var}: {ratio:.3f} {status}\n")
            f.write("\n")

        f.write("### 3. Prediction Interval Coverage\n\n")
        f.write("**What it measures:** Calibration of uncertainty intervals.\n\n")
        f.write("**Interpretation:** A well-calibrated model should have actual coverage match target. ")
        f.write("E.g., 90% interval should contain 90% of true values.\n\n")

        coverage_levels = [0.5, 0.8, 0.9]

        for level in coverage_levels:
            f.write(f"#### {level*100:.0f}% Intervals\n\n")

            for method in methods:
                f.write(f"**{method}:**\n")
                for var in target_vars:
                    stats = all_metrics[method]['interval_coverage'][var][level]
                    f.write(f"  - {var}:\n")
                    f.write(f"    - Target: {stats['target']:.1%}\n")
                    f.write(f"    - Actual: {stats['actual']:.1%}\n")
                    f.write(f"    - Calibration error: {stats['calibration_error']:.4f}\n")
                    f.write(f"    - Mean width: {stats['interval_width']:.2f}\n")
                f.write("\n")

        f.write("### 4. Conditional Variance Preservation\n\n")
        f.write("**What it measures:** Within-group variance preservation across demographic subgroups.\n\n")
        f.write("**Interpretation:** Tests if the model captures heteroscedasticity. ")
        f.write("Lower error = better preservation of conditional variance structure.\n\n")

        for method in methods:
            f.write(f"**{method}:**\n")
            for var in target_vars:
                error = all_metrics[method]['conditional_variance_errors'][var]
                f.write(f"  - {var}: {error:.4f}\n")
            mean_error = np.mean(list(all_metrics[method]['conditional_variance_errors'].values()))
            f.write(f"  - **Mean:** {mean_error:.4f}\n\n")

        f.write("### 5. Quantile Losses (Pinball Loss)\n\n")
        f.write("**What it measures:** How well predicted quantiles match true distribution.\n\n")
        f.write("**Interpretation:** Lower loss = better quantile matching. ")
        f.write("Tests if method captures full distribution (not just median).\n\n")

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        for method in methods:
            f.write(f"**{method}:**\n\n")
            f.write("| Variable | q=0.10 | q=0.25 | q=0.50 | q=0.75 | q=0.90 | Mean |\n")
            f.write("|----------|--------|--------|--------|--------|--------|------|\n")

            for var in target_vars:
                losses = all_metrics[method]['quantile_losses'][var]
                loss_values = [losses[q] for q in quantiles]
                mean_loss = np.mean(loss_values)
                f.write(f"| {var} | {loss_values[0]:.0f} | {loss_values[1]:.0f} | "
                       f"{loss_values[2]:.0f} | {loss_values[3]:.0f} | {loss_values[4]:.0f} | "
                       f"{mean_loss:.0f} |\n")
            f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")

        f.write("### Question 1: Does microplex capture full conditional distribution?\n\n")
        micro_metrics = all_metrics.get("microplex")
        if micro_metrics:
            var_ratios = micro_metrics['variance_ratios']
            all_good = all(0.8 <= v <= 1.2 for v in var_ratios.values())
            if all_good:
                f.write("**Yes.** Variance ratios are all within [0.8, 1.2], indicating microplex ")
                f.write("captures the full distribution without mode collapse.\n\n")
            else:
                f.write("**Partial.** Some variables show variance ratio issues, suggesting ")
                f.write("potential under/over-dispersion.\n\n")
        else:
            f.write("microplex not tested.\n\n")

        f.write("### Question 2: Is uncertainty calibrated correctly?\n\n")
        if micro_metrics:
            # Check 90% interval calibration
            interval_stats = micro_metrics['interval_coverage']
            cal_errors = []
            for var in target_vars:
                cal_errors.append(interval_stats[var][0.9]['calibration_error'])
            mean_cal_error = np.mean(cal_errors)

            if mean_cal_error < 0.05:
                f.write(f"**Yes.** Mean 90% interval calibration error is {mean_cal_error:.4f}, ")
                f.write("indicating well-calibrated uncertainty.\n\n")
            else:
                f.write(f"**No.** Mean 90% interval calibration error is {mean_cal_error:.4f}, ")
                f.write("indicating mis-calibrated uncertainty.\n\n")
        else:
            f.write("microplex not tested.\n\n")

        f.write("### Question 3: How does QRF compare?\n\n")

        qrf_methods = [m for m in methods if 'qrf' in m]
        if qrf_methods and micro_metrics:
            f.write("**Comparison:**\n\n")

            for qrf_method in qrf_methods:
                qrf_crps = np.mean(list(all_metrics[qrf_method]['crps'].values()))
                micro_crps = np.mean(list(micro_metrics['crps'].values()))

                qrf_var_error = np.mean([abs(1.0 - v) for v in all_metrics[qrf_method]['variance_ratios'].values()])
                micro_var_error = np.mean([abs(1.0 - v) for v in micro_metrics['variance_ratios'].values()])

                f.write(f"- **{qrf_method} vs microplex:**\n")
                f.write(f"  - CRPS: {qrf_crps:.2f} vs {micro_crps:.2f} ")
                f.write(f"({qrf_crps/micro_crps:.2f}x)\n")
                f.write(f"  - Variance error: {qrf_var_error:.4f} vs {micro_var_error:.4f} ")
                f.write(f"({qrf_var_error/micro_var_error:.2f}x)\n\n")

        f.write("## Visualizations\n\n")
        f.write("1. `distributional_variance_ratios.png` - Variance preservation\n")
        f.write("2. `distributional_crps.png` - CRPS comparison\n")
        f.write("3. `distributional_calibration.png` - Prediction interval calibration\n")
        f.write("4. `distributional_conditional_variance.png` - Conditional variance\n")
        f.write("5. `distributional_quantile_losses.png` - Quantile loss heatmaps\n\n")

        f.write("## Reproducibility\n\n")
        f.write("```bash\n")
        f.write("cd /Users/maxghenis/PolicyEngine/micro\n")
        f.write("python benchmarks/run_distributional_benchmark.py\n")
        f.write("```\n\n")

        f.write("---\n\n")
        f.write("**Generated:** December 25, 2024\n")

    print(f"\nSaved markdown report to {output_path}")


def main():
    """Run distributional quality benchmark."""

    print("=" * 80)
    print("DISTRIBUTIONAL QUALITY BENCHMARK")
    print("=" * 80)

    # Configuration
    n_train = 5000
    n_test = 500
    epochs = 50
    n_samples_per_condition = 50  # For CRPS/interval metrics

    target_vars = ["income", "assets", "debt", "savings"]
    condition_vars = ["age", "education", "region"]

    # Output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate data
    print(f"\nGenerating realistic economic microdata...")
    print(f"  Training: {n_train:,} samples")
    print(f"  Test: {n_test:,} samples")

    full_data = generate_realistic_microdata(n_train + n_test, seed=42)
    train_data = full_data.iloc[:n_train].copy()
    test_data = full_data.iloc[n_train:].copy()

    print("\nData statistics:")
    print(f"  Income: ${train_data['income'].mean():,.0f} ± ${train_data['income'].std():,.0f}")
    print(f"  Assets: ${train_data['assets'].mean():,.0f} (zero: {(train_data['assets'] == 0).mean():.1%})")
    print(f"  Debt: ${train_data['debt'].mean():,.0f} (zero: {(train_data['debt'] == 0).mean():.1%})")

    # Storage for all metrics
    all_metrics = {}

    # 1. Benchmark QRF Sequential
    print("\n" + "=" * 80)
    print("BENCHMARKING: QRF Sequential")
    print("=" * 80)

    qrf_model = SequentialQRF(target_vars, condition_vars)
    print("Training...")
    qrf_model.fit(train_data, verbose=True)

    print("Computing distributional metrics...")
    all_metrics['qrf_sequential'] = compute_comprehensive_distributional_metrics(
        qrf_model,
        train_data,
        test_data,
        target_vars,
        condition_vars,
        n_samples_per_condition=n_samples_per_condition,
    )

    print_distributional_metrics_report(all_metrics['qrf_sequential'], target_vars)

    # 2. Benchmark QRF with Zero Inflation
    print("\n" + "=" * 80)
    print("BENCHMARKING: QRF with Zero Inflation")
    print("=" * 80)

    qrf_zi_model = SequentialQRFWithZeroInflation(target_vars, condition_vars)
    print("Training...")
    qrf_zi_model.fit(train_data, verbose=True)

    print("Computing distributional metrics...")
    all_metrics['qrf_zero_inflation'] = compute_comprehensive_distributional_metrics(
        qrf_zi_model,
        train_data,
        test_data,
        target_vars,
        condition_vars,
        n_samples_per_condition=n_samples_per_condition,
    )

    print_distributional_metrics_report(all_metrics['qrf_zero_inflation'], target_vars)

    # 3. Benchmark microplex
    print("\n" + "=" * 80)
    print("BENCHMARKING: microplex")
    print("=" * 80)

    try:
        from microplex import Synthesizer

        micro_model = Synthesizer(target_vars=target_vars, condition_vars=condition_vars)
        print("Training...")
        micro_model.fit(train_data, epochs=epochs, verbose=False)

        print("Computing distributional metrics...")
        all_metrics['microplex'] = compute_comprehensive_distributional_metrics(
            micro_model,
            train_data,
            test_data,
            target_vars,
            condition_vars,
            n_samples_per_condition=n_samples_per_condition,
        )

        print_distributional_metrics_report(all_metrics['microplex'], target_vars)

    except Exception as e:
        print(f"ERROR: microplex benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    create_distributional_visualizations(all_metrics, output_dir)

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    report_path = output_dir / "distributional_quality.md"
    generate_distributional_markdown_report(all_metrics, report_path)

    # Save metrics JSON
    import json

    metrics_json = output_dir / "distributional_metrics.json"
    with open(metrics_json, "w") as f:
        # Convert to serializable format
        serializable_metrics = {}
        for method, metrics in all_metrics.items():
            serializable_metrics[method] = {
                'variance_ratios': metrics['variance_ratios'],
                'conditional_variance_errors': metrics['conditional_variance_errors'],
                'crps': metrics['crps'],
                # Skip complex nested structures for now
            }
        json.dump(serializable_metrics, f, indent=2)
    print(f"Saved metrics JSON to {metrics_json}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - distributional_quality.md: Full analysis report")
    print(f"  - distributional_metrics.json: Raw metrics")
    print(f"  - distributional_*.png: Visualizations")


if __name__ == "__main__":
    main()
