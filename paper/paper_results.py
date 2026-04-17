"""Paper results module — single source of truth for inline values.

All computed results referenced in the paper are derived here from
benchmark JSON output. Use {eval}`r.some_value` in MyST markdown.

Primary results come from multi-seed evaluation (benchmark_multi_seed.json)
with means ± standard errors. Single-seed results (benchmark_full.json)
are kept for backward compatibility.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MethodStats:
    name: str
    coverage: float
    precision: float
    density: float
    elapsed: float
    sipp_coverage: float
    cps_coverage: float
    psid_coverage: float
    # Multi-seed stats (None if single-seed only)
    sipp_se: float = 0.0
    cps_se: float = 0.0
    psid_se: float = 0.0
    n_seeds: int = 1

    @property
    def coverage_pct(self) -> str:
        return f"{self.coverage:.1%}"

    @property
    def precision_pct(self) -> str:
        return f"{self.precision:.1%}"

    @property
    def sipp_pct(self) -> str:
        if self.n_seeds > 1 and self.sipp_se > 0:
            return f"{self.sipp_coverage:.1%} ± {self.sipp_se:.1%}"
        return f"{self.sipp_coverage:.1%}"

    @property
    def cps_pct(self) -> str:
        if self.n_seeds > 1 and self.cps_se > 0:
            return f"{self.cps_coverage:.1%} ± {self.cps_se:.1%}"
        return f"{self.cps_coverage:.1%}"

    @property
    def psid_pct(self) -> str:
        return f"{self.psid_coverage:.1%}"

    @property
    def time_str(self) -> str:
        return f"{self.elapsed:.0f}s"


@dataclass
class ReweightingMethodStats:
    name: str
    mean_relative_error: float
    max_relative_error: float
    weight_cv: float
    sparsity: float
    elapsed: float
    train_mean_error: float = 0.0
    test_mean_error: float = 0.0

    @property
    def mean_error_pct(self) -> str:
        return f"{self.mean_relative_error:.1%}"

    @property
    def train_error_pct(self) -> str:
        return f"{self.train_mean_error:.1%}"

    @property
    def test_error_pct(self) -> str:
        return f"{self.test_mean_error:.1%}"

    @property
    def max_error_pct(self) -> str:
        return f"{self.max_relative_error:.1%}"

    @property
    def cv_str(self) -> str:
        return f"{self.weight_cv:.3f}"

    @property
    def sparsity_pct(self) -> str:
        return f"{self.sparsity:.1%}"

    @property
    def time_str(self) -> str:
        return f"{self.elapsed:.2f}s"


def _pct_change(old: float, new: float) -> float:
    """Percent reduction from old to new: (old - new) / old * 100."""
    return (old - new) / old * 100


@dataclass
class PaperResults:
    """All computed values for paper inline references."""

    # Synthesis method results
    qrf: MethodStats
    zi_qrf: MethodStats
    qdnn: MethodStats
    zi_qdnn: MethodStats
    maf: MethodStats
    zi_maf: MethodStats

    # Reweighting method results
    rw_ipf: ReweightingMethodStats
    rw_entropy: ReweightingMethodStats
    rw_sparse_cal: ReweightingMethodStats
    rw_l1: ReweightingMethodStats
    rw_l0: ReweightingMethodStats

    # Synthesis benchmark config
    k: int
    holdout_frac: float
    n_generate: int
    seed: int
    total_elapsed: float

    # Reweighting benchmark config
    rw_n_records: int = 5000
    rw_n_marginal_targets: int = 7
    rw_n_continuous_targets: int = 1
    rw_n_train_targets: int = 6
    rw_n_test_targets: int = 2

    # Data characteristics
    n_sipp: int = 476_744
    n_cps: int = 144_265
    n_psid: int = 9_207
    n_total: int = 630_216
    n_sources: int = 3

    # Multi-seed info
    n_seeds: int = 1
    max_rows_per_source: int = 20_000

    # Optional reweighting methods (may not be in all benchmark runs)
    rw_hardconcrete: ReweightingMethodStats = None

    # Synthesis derived comparisons
    @property
    def _synthesis_methods(self) -> list[MethodStats]:
        return [self.qrf, self.zi_qrf, self.qdnn, self.zi_qdnn, self.maf, self.zi_maf]

    def _coverage_lift(self, zi_method: MethodStats, base_method: MethodStats) -> str:
        lift = (zi_method.coverage - base_method.coverage) / base_method.coverage * 100
        return f"{lift:.0f}%"

    @property
    def best_method(self) -> str:
        return max(self._synthesis_methods, key=lambda m: m.coverage).name

    @property
    def best_coverage(self) -> str:
        return max(m.coverage for m in self._synthesis_methods)

    @property
    def best_coverage_pct(self) -> str:
        return f"{self.best_coverage:.1%}"

    @property
    def zi_maf_vs_maf_lift(self) -> str:
        return self._coverage_lift(self.zi_maf, self.maf)

    @property
    def zi_qdnn_vs_qdnn_lift(self) -> str:
        return self._coverage_lift(self.zi_qdnn, self.qdnn)

    @property
    def zi_qrf_vs_qrf_lift(self) -> str:
        return self._coverage_lift(self.zi_qrf, self.qrf)

    @property
    def zi_speedup_over_maf(self) -> str:
        ratio = self.zi_maf.elapsed / self.zi_qrf.elapsed
        return f"{ratio:.0f}x"

    @property
    def n_methods(self) -> int:
        return 6

    @property
    def total_elapsed_str(self) -> str:
        return f"{self.total_elapsed:.0f}s"

    # Reweighting derived comparisons
    @property
    def _calibration_methods(self) -> list[ReweightingMethodStats]:
        methods = [self.rw_ipf, self.rw_entropy, self.rw_sparse_cal]
        if self.rw_hardconcrete is not None:
            methods.append(self.rw_hardconcrete)
        return methods

    @property
    def best_rw_method(self) -> str:
        """Best reweighting method by lowest mean relative error (calibration methods only)."""
        return min(self._calibration_methods, key=lambda m: m.mean_relative_error).name

    @property
    def best_rw_error(self) -> str:
        return f"{min(m.mean_relative_error for m in self._calibration_methods):.1%}"

    @property
    def entropy_vs_ipf_error_reduction(self) -> str:
        reduction = _pct_change(self.rw_ipf.mean_relative_error, self.rw_entropy.mean_relative_error)
        return f"{reduction:.0f}%"

    @property
    def sparse_cal_cv_vs_ipf(self) -> str:
        reduction = _pct_change(self.rw_ipf.weight_cv, self.rw_sparse_cal.weight_cv)
        return f"{reduction:.0f}%"

    @property
    def rw_n_targets_total(self) -> int:
        return self.rw_n_marginal_targets + self.rw_n_continuous_targets


def _extract_method(data: dict, name: str, multi_seed_data: dict = None) -> MethodStats:
    m = data["methods"][name]
    source_map = {s["source"]: s for s in m["sources"]}

    # If multi-seed data available, use means ± SE from there
    sipp_cov = source_map.get("sipp", {}).get("coverage", 0)
    cps_cov = source_map.get("cps", {}).get("coverage", 0)
    psid_cov = source_map.get("psid", {}).get("coverage", 0)
    sipp_se = 0.0
    cps_se = 0.0
    psid_se = 0.0
    n_seeds = 1

    if multi_seed_data and name in multi_seed_data.get("methods", {}):
        ms = multi_seed_data["methods"][name]
        n_seeds = multi_seed_data.get("n_seeds", 1)
        if "sipp" in ms:
            sipp_cov = ms["sipp"]["mean"]
            sipp_se = ms["sipp"]["se"]
        if "cps" in ms:
            cps_cov = ms["cps"]["mean"]
            cps_se = ms["cps"]["se"]
        if "psid" in ms:
            psid_cov = ms["psid"]["mean"]
            psid_se = ms["psid"]["se"]

    # Recompute mean coverage from per-source values
    coverage = (sipp_cov + cps_cov + psid_cov) / 3

    return MethodStats(
        name=name,
        coverage=coverage,
        precision=m["mean_precision"],
        density=m.get("mean_density", 0),
        elapsed=m["elapsed_seconds"],
        sipp_coverage=sipp_cov,
        cps_coverage=cps_cov,
        psid_coverage=psid_cov,
        sipp_se=sipp_se,
        cps_se=cps_se,
        psid_se=psid_se,
        n_seeds=n_seeds,
    )


def _extract_rw_method(data: dict, key: str) -> ReweightingMethodStats:
    m = data["methods"][key]
    return ReweightingMethodStats(
        name=m["method_name"],
        mean_relative_error=m["mean_relative_error"],
        max_relative_error=m["max_relative_error"],
        weight_cv=m["weight_cv"],
        sparsity=m["sparsity"],
        elapsed=m["elapsed_seconds"],
        train_mean_error=m.get("train_mean_error", m["mean_relative_error"]),
        test_mean_error=m.get("test_mean_error", 0.0),
    )


def load_results(
    synthesis_path: str = None,
    reweighting_path: str = None,
    multi_seed_path: str = None,
) -> PaperResults:
    results_dir = Path(__file__).parent.parent / "benchmarks" / "results"

    if synthesis_path is None:
        synthesis_path = str(results_dir / "benchmark_full.json")
    if reweighting_path is None:
        reweighting_path = str(results_dir / "reweighting_full.json")
    if multi_seed_path is None:
        multi_seed_path = str(results_dir / "benchmark_multi_seed.json")

    with open(synthesis_path) as f:
        synth_data = json.load(f)

    with open(reweighting_path) as f:
        rw_data = json.load(f)

    # Load multi-seed data if available
    multi_seed_data = None
    ms_path = Path(multi_seed_path)
    if ms_path.exists():
        with open(ms_path) as f:
            multi_seed_data = json.load(f)

    n_seeds = multi_seed_data.get("n_seeds", 1) if multi_seed_data else 1

    return PaperResults(
        # Synthesis (using multi-seed means when available)
        qrf=_extract_method(synth_data, "QRF", multi_seed_data),
        zi_qrf=_extract_method(synth_data, "ZI-QRF", multi_seed_data),
        qdnn=_extract_method(synth_data, "QDNN", multi_seed_data),
        zi_qdnn=_extract_method(synth_data, "ZI-QDNN", multi_seed_data),
        maf=_extract_method(synth_data, "MAF", multi_seed_data),
        zi_maf=_extract_method(synth_data, "ZI-MAF", multi_seed_data),
        k=synth_data["k"],
        holdout_frac=synth_data["holdout_frac"],
        n_generate=synth_data["n_generate"],
        seed=synth_data["seed"],
        total_elapsed=synth_data["total_elapsed_seconds"],
        n_seeds=n_seeds,
        # Reweighting
        rw_ipf=_extract_rw_method(rw_data, "IPF"),
        rw_entropy=_extract_rw_method(rw_data, "Entropy"),
        rw_sparse_cal=_extract_rw_method(rw_data, "SparseCalibrator"),
        rw_l1=_extract_rw_method(rw_data, "L1-Sparse"),
        rw_l0=_extract_rw_method(rw_data, "L0-Sparse"),
        rw_hardconcrete=_extract_rw_method(rw_data, "HardConcrete") if "HardConcrete" in rw_data.get("methods", {}) else None,
        rw_n_records=rw_data["n_records"],
        rw_n_marginal_targets=rw_data["n_marginal_targets"],
        rw_n_continuous_targets=rw_data["n_continuous_targets"],
        rw_n_train_targets=rw_data.get("n_train_targets", 6),
        rw_n_test_targets=rw_data.get("n_test_targets", 2),
    )


# Module-level instance for {eval}`r.xxx` in MyST
r = load_results()
