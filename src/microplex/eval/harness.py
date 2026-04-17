"""Unified evaluation harness for microplex synthesis and reweighting.

Two eval axes:
1. Synthesis: multivariate coverage (PRDC) against holdout sets per donor source
2. Reweighting: loss across aggregate targets (SOI income, benefit spending, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# --- Protocols ---


class SynthesisModel(Protocol):
    """Any model with fit() and generate()."""

    def fit(
        self, sources: dict[str, pd.DataFrame], shared_cols: list[str]
    ) -> "SynthesisModel": ...

    def generate(self, n: int, seed: int = 42) -> pd.DataFrame: ...


# --- Result dataclasses ---


@dataclass
class SourceCoverage:
    """PRDC metrics for one donor source's holdout."""

    source_name: str
    precision: float
    recall: float
    density: float
    coverage: float
    n_holdout: int
    n_synthetic: int
    columns_evaluated: list[str]

    def to_dict(self) -> dict:
        return {
            "source": self.source_name,
            "precision": float(self.precision),
            "recall": float(self.recall),
            "density": float(self.density),
            "coverage": float(self.coverage),
            "n_holdout": self.n_holdout,
            "n_synthetic": self.n_synthetic,
            "columns": self.columns_evaluated,
        }


@dataclass
class SynthesisEvalResult:
    """Results from synthesis evaluation across all sources."""

    source_coverages: list[SourceCoverage]
    n_synthetic: int

    @property
    def mean_coverage(self) -> float:
        if not self.source_coverages:
            return 0.0
        return np.mean([sc.coverage for sc in self.source_coverages])

    @property
    def mean_precision(self) -> float:
        if not self.source_coverages:
            return 0.0
        return np.mean([sc.precision for sc in self.source_coverages])

    @property
    def mean_recall(self) -> float:
        if not self.source_coverages:
            return 0.0
        return np.mean([sc.recall for sc in self.source_coverages])

    def to_dict(self) -> dict:
        return {
            "mean_coverage": float(self.mean_coverage),
            "mean_precision": float(self.mean_precision),
            "mean_recall": float(self.mean_recall),
            "n_synthetic": self.n_synthetic,
            "sources": [sc.to_dict() for sc in self.source_coverages],
        }

    def summary(self) -> str:
        lines = [
            "Synthesis Evaluation",
            "=" * 70,
            f"{'Source':<12} {'Coverage':>10} {'Precision':>10} "
            f"{'Density':>10} {'Holdout':>8} {'Cols':>6}",
            "-" * 64,
        ]
        for sc in self.source_coverages:
            lines.append(
                f"{sc.source_name:<12} {sc.coverage:>10.1%} {sc.precision:>10.1%} "
                f"{sc.density:>10.2f} "
                f"{sc.n_holdout:>8,} {len(sc.columns_evaluated):>6}"
            )
        lines.append("-" * 64)
        lines.append(
            f"{'MEAN':<12} {self.mean_coverage:>10.1%} {self.mean_precision:>10.1%}"
        )
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class AggregateError:
    """Error for one aggregate target."""

    target_name: str
    category: str
    target_value: float
    computed_value: float
    relative_error: float  # percentage
    absolute_error: float

    def to_dict(self) -> dict:
        return {
            "target": self.target_name,
            "category": self.category,
            "target_value": float(self.target_value),
            "computed_value": float(self.computed_value),
            "relative_error_pct": float(self.relative_error),
            "absolute_error": float(self.absolute_error),
        }


@dataclass
class ReweightingEvalResult:
    """Results from reweighting evaluation against aggregate targets."""

    aggregate_errors: list[AggregateError]
    n_targets: int
    n_matched: int

    @property
    def mean_relative_error(self) -> float:
        if not self.aggregate_errors:
            return 0.0
        return np.mean([ae.relative_error for ae in self.aggregate_errors])

    @property
    def max_relative_error(self) -> float:
        if not self.aggregate_errors:
            return 0.0
        return max(ae.relative_error for ae in self.aggregate_errors)

    def errors_by_category(self) -> dict[str, list[AggregateError]]:
        result: dict[str, list[AggregateError]] = {}
        for ae in self.aggregate_errors:
            result.setdefault(ae.category, []).append(ae)
        return result

    def to_dict(self) -> dict:
        by_cat = {}
        for cat, errors in self.errors_by_category().items():
            by_cat[cat] = {
                "mean_error_pct": float(np.mean([e.relative_error for e in errors])),
                "max_error_pct": float(max(e.relative_error for e in errors)),
                "n_targets": len(errors),
            }
        return {
            "mean_relative_error": float(self.mean_relative_error),
            "max_relative_error": float(self.max_relative_error),
            "n_targets": self.n_targets,
            "n_matched": self.n_matched,
            "by_category": by_cat,
            "targets": [ae.to_dict() for ae in self.aggregate_errors],
        }

    def summary(self) -> str:
        lines = [
            "Reweighting Evaluation",
            "=" * 80,
            f"{'Target':<35} {'Computed':>14} {'Official':>14} {'Error':>8}",
            "-" * 80,
        ]

        # Group by category
        for cat, errors in sorted(self.errors_by_category().items()):
            lines.append(f"  [{cat.upper()}]")
            for ae in sorted(errors, key=lambda x: -x.relative_error):
                comp = _fmt_number(ae.computed_value)
                tgt = _fmt_number(ae.target_value)
                lines.append(
                    f"  {ae.target_name:<33} {comp:>14} {tgt:>14} "
                    f"{ae.relative_error:>7.1f}%"
                )
            cat_mean = np.mean([e.relative_error for e in errors])
            lines.append(f"  {'  category mean':<33} {'':>14} {'':>14} {cat_mean:>7.1f}%")
            lines.append("")

        lines.append("-" * 80)
        lines.append(
            f"{'OVERALL':<35} {self.n_matched}/{self.n_targets} targets matched"
            f"   mean={self.mean_relative_error:.1f}%  max={self.max_relative_error:.1f}%"
        )
        lines.append("=" * 80)
        return "\n".join(lines)


def _fmt_number(v: float) -> str:
    av = abs(v)
    if av >= 1e12:
        return f"${v / 1e12:.2f}T"
    if av >= 1e9:
        return f"${v / 1e9:.1f}B"
    if av >= 1e6:
        return f"{v / 1e6:.1f}M"
    return f"{v:,.0f}"


# --- PRDC computation (standalone, no dependency on eval.coverage) ---


def _compute_prdc(
    real: np.ndarray, synthetic: np.ndarray, k: int = 5
) -> dict[str, float]:
    """Compute Precision, Recall, Density, Coverage via canonical prdc library.

    Delegates to Naeem et al. (2020) reference implementation. Recall is
    kept in the return dict for API compatibility (it equals coverage in
    the k-NN formulation).
    """
    from prdc import compute_prdc as _prdc

    if len(real) < k + 1 or len(synthetic) < k + 1:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

    scaler = StandardScaler()
    real_s = scaler.fit_transform(real)
    synth_s = scaler.transform(synthetic)

    metrics = _prdc(real_s, synth_s, nearest_k=k)

    return {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "density": float(metrics["density"]),
        "coverage": float(metrics["coverage"]),
    }


# --- Main harness ---


class EvalHarness:
    """Unified evaluation for microplex synthesis and reweighting.

    Usage:
        harness = EvalHarness()

        # Synthesis eval: PRDC coverage per donor source
        synth_result = harness.evaluate_synthesis(
            model=my_model,
            sources={"CPS": cps_df, "SIPP": sipp_df},
            shared_cols=["age", "income", "is_male"],
        )

        # Reweighting eval: loss against aggregates
        rw_result = harness.evaluate_reweighting(
            data=weighted_df,
            weight_col="weight",
        )
    """

    def __init__(self, registry=None):
        """Initialize harness.

        Args:
            registry: TargetRegistry for reweighting eval. If None, uses default.
        """
        self._registry = registry

    @property
    def registry(self):
        if self._registry is None:
            from microplex.target_registry import get_registry

            self._registry = get_registry()
        return self._registry

    def evaluate_synthesis(
        self,
        model: Any,
        sources: dict[str, pd.DataFrame],
        shared_cols: list[str],
        holdout_frac: float = 0.2,
        n_synthetic: Optional[int] = None,
        k: int = 5,
        seed: int = 42,
    ) -> SynthesisEvalResult:
        """Evaluate synthesis quality via PRDC against holdout sets.

        For each source:
        1. Split into train/holdout
        2. Train model on train portions
        3. Generate synthetic data
        4. Compute PRDC on holdout columns

        Args:
            model: Object with fit(sources, shared_cols) and generate(n, seed)
            sources: name -> DataFrame for each donor source
            shared_cols: Columns present across sources (used for fitting)
            holdout_frac: Fraction of each source to hold out
            n_synthetic: Records to generate (default: sum of all holdouts)
            k: Number of neighbors for PRDC
            seed: Random seed

        Returns:
            SynthesisEvalResult with per-source PRDC
        """
        rng = np.random.RandomState(seed)

        # Split each source
        train_sources = {}
        holdouts = {}
        for name, df in sources.items():
            n = len(df)
            n_holdout = max(int(n * holdout_frac), k + 2)
            perm = rng.permutation(n)
            holdouts[name] = df.iloc[perm[:n_holdout]].reset_index(drop=True)
            train_sources[name] = df.iloc[perm[n_holdout:]].reset_index(drop=True)

        # Fit model on train data
        model.fit(train_sources, shared_cols)

        # Generate synthetic
        if n_synthetic is None:
            n_synthetic = sum(len(h) for h in holdouts.values())
        synthetic = model.generate(n=n_synthetic, seed=seed)

        # Evaluate per source
        coverages = []
        for name, holdout in holdouts.items():
            # Find columns present in both holdout and synthetic
            eval_cols = [
                c
                for c in holdout.columns
                if c in synthetic.columns and c != "_survey"
            ]
            # Only use numeric columns
            eval_cols = [
                c
                for c in eval_cols
                if holdout[c].dtype in [np.float64, np.int64, np.float32, np.int32]
            ]

            if len(eval_cols) < 1:
                continue

            holdout_vals = holdout[eval_cols].values.astype(float)
            synth_vals = synthetic[eval_cols].dropna().values.astype(float)

            # Drop NaN rows
            hold_mask = ~np.isnan(holdout_vals).any(axis=1)
            synth_mask = ~np.isnan(synth_vals).any(axis=1)
            holdout_clean = holdout_vals[hold_mask]
            synth_clean = synth_vals[synth_mask]

            if len(holdout_clean) < k + 2 or len(synth_clean) < k + 2:
                continue

            prdc = _compute_prdc(holdout_clean, synth_clean, k=k)

            coverages.append(
                SourceCoverage(
                    source_name=name,
                    precision=prdc["precision"],
                    recall=prdc["recall"],
                    density=prdc["density"],
                    coverage=prdc["coverage"],
                    n_holdout=len(holdout_clean),
                    n_synthetic=len(synth_clean),
                    columns_evaluated=eval_cols,
                )
            )

        return SynthesisEvalResult(
            source_coverages=coverages,
            n_synthetic=n_synthetic,
        )

    def evaluate_reweighting(
        self,
        data: pd.DataFrame,
        weight_col: str = "weight",
        categories: Optional[list[str]] = None,
    ) -> ReweightingEvalResult:
        """Evaluate weighted data against aggregate targets.

        Computes weighted sums/counts from data and compares to
        official targets from the registry.

        Args:
            data: Weighted microdata DataFrame
            weight_col: Column containing weights
            categories: Target categories to evaluate (None = all available)

        Returns:
            ReweightingEvalResult with per-target errors
        """
        all_targets = self.registry.get_all_targets()

        if categories:
            all_targets = [
                t for t in all_targets if t.category.value in categories
            ]

        weights = data[weight_col].values if weight_col in data.columns else np.ones(len(data))

        errors = []
        n_total = 0
        n_matched = 0

        for target in all_targets:
            # Skip zero-valued targets (placeholders)
            if target.value == 0:
                continue
            n_total += 1

            # Check if we can compute this target from the data
            if target.column and target.column not in data.columns:
                continue
            if target.filter_column and target.filter_column not in data.columns:
                continue

            # Compute weighted aggregate
            if target.filter_column and target.filter_value is not None:
                # Handle type mismatches (e.g., state_fips: float 6.0 vs string "06")
                col_vals = data[target.filter_column]
                filter_val = target.filter_value
                if col_vals.dtype in [np.float64, np.int64, np.float32, np.int32]:
                    try:
                        filter_val = float(filter_val)
                    except (ValueError, TypeError):
                        pass
                elif col_vals.dtype == object:
                    # Data is string, try matching as string
                    filter_val = str(filter_val)
                mask = (col_vals == filter_val).values
                if target.aggregation == "count" or target.column is None:
                    computed = float(np.sum(weights * mask))
                elif target.aggregation == "sum":
                    vals = data[target.column].fillna(0).values
                    computed = float(np.sum(weights * mask * vals))
                else:
                    computed = float(np.sum(weights * mask))
            elif target.column:
                col_data = data[target.column]
                # Skip non-numeric columns for aggregation
                if col_data.dtype == object or col_data.dtype.name == 'category':
                    continue
                vals = col_data.fillna(0).values
                if target.aggregation == "count":
                    computed = float(np.sum(weights * (vals > 0)))
                elif target.aggregation == "sum":
                    computed = float(np.sum(weights * vals))
                else:
                    computed = float(np.sum(weights * vals))
            else:
                # Total population count
                computed = float(np.sum(weights))

            abs_err = abs(computed - target.value)
            rel_err = abs_err / abs(target.value) * 100 if target.value != 0 else 0

            errors.append(
                AggregateError(
                    target_name=target.name,
                    category=target.category.value,
                    target_value=target.value,
                    computed_value=computed,
                    relative_error=rel_err,
                    absolute_error=abs_err,
                )
            )
            n_matched += 1

        return ReweightingEvalResult(
            aggregate_errors=errors,
            n_targets=n_total,
            n_matched=n_matched,
        )
