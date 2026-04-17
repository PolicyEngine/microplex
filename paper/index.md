---
kernelspec:
  name: python3
  display_name: Python 3
---

# Microplex: Multi-source synthetic microdata via zero-inflated conditional models

**Max Ghenis**

max@cosilico.ai | Cosilico

```{code-cell} python
:tags: [remove-cell]

from paper_results import r
```

## Abstract

Government surveys observe different slices of the same population: the Current Population Survey (CPS) captures employment and income, the Survey of Income and Program Participation (SIPP) tracks employment dynamics, and the Panel Study of Income Dynamics (PSID) follows families longitudinally. No single survey observes all variables for all people. I present microplex, a framework for learning per-variable conditional distributions $P(v \mid V_{\text{shared}})$ from multiple surveys and generating synthetic records with complete variable coverage. Because each variable is modeled conditionally on shared demographics (age, sex), the resulting synthetic data preserves within-source marginals but does not learn cross-source correlations — a limitation I discuss. I compare six synthesis methods — quantile regression forests (QRF), quantile deep neural networks (QDNN), and masked autoregressive flows (MAF), each with and without zero-inflation (ZI) handling — using Precision, Density, and Coverage metrics adapted from {cite:t}`naeem2020reliable`, evaluated against holdouts from each source survey across {eval}`r.n_seeds` random seeds. ZI-QRF achieves the highest SIPP coverage ({eval}`r.zi_qrf.sipp_pct`) while ZI-MAF achieves the highest CPS coverage ({eval}`r.zi_maf.cps_pct`), but the key finding is architectural: zero-inflation handling lifts MAF coverage by {eval}`r.zi_maf_vs_maf_lift` and QDNN by {eval}`r.zi_qdnn_vs_qdnn_lift`, while barely affecting QRF ({eval}`r.zi_qrf_vs_qrf_lift`). I also compare five calibration methods for reweighting synthetic populations, finding that entropy balancing achieves the lowest mean relative error ({eval}`r.rw_entropy.mean_error_pct`). Code is available at [github.com/CosilicoAI/microplex](https://github.com/CosilicoAI/microplex).

## Introduction

Policy microsimulation requires detailed individual records spanning demographics, income, taxes, transfers, wealth, and health. No single survey covers all domains. The Current Population Survey (CPS) Annual Social and Economic Supplement (ASEC) {cite:p}`flood2023integrated` captures {eval}`f"{r.n_cps:,}"` persons with employment and income variables. The Survey of Income and Program Participation (SIPP) {cite:p}`census2023sipp` adds employment dynamics and income detail for {eval}`f"{r.n_sipp:,}"` persons. The Panel Study of Income Dynamics (PSID) {cite:p}`psid2023` provides longitudinal structure for {eval}`f"{r.n_psid:,}"` persons. Administrative sources (Internal Revenue Service Statistics of Income, Social Security Administration earnings records) cover entire populations but with narrower variable sets.

Current approaches to combining these sources — sequential imputation, statistical matching, or record linkage — suffer from well-documented limitations. Synthetic data approaches {cite:p}`rubin1993statistical,drechsler2011synthetic` and multiple imputation {cite:p}`raghunathan2003multiple` address disclosure concerns but typically operate on single surveys. Sequential chaining (e.g., imputing CPS variables onto the American Community Survey, then Public Use File variables onto CPS) loses joint distributional structure at each step {cite:p}`meinfelder2011simulation`. Statistical matching preserves marginals but distorts correlations {cite:p}`dorazio2006statistical`. Record linkage requires common identifiers rarely available across surveys.

I make three contributions:

1. **Multi-source conditional synthesis framework.** I formalize the problem of learning per-variable conditionals $P(v \mid V_{\text{shared}})$ from surveys that each observe different subsets of variables. This approach generates records with complete variable coverage, though it assumes conditional independence across sources given shared variables — a strong assumption whose implications I evaluate.

2. **Zero-inflation (ZI) as architectural choice.** I show that ZI handling — a two-stage model that separately predicts whether a variable is zero vs. its positive-value distribution — provides large coverage gains for neural methods (MAF: +{eval}`r.zi_maf_vs_maf_lift`; QDNN: +{eval}`r.zi_qdnn_vs_qdnn_lift`) while barely affecting tree-based methods (QRF: +{eval}`r.zi_qrf_vs_qrf_lift`), suggesting it is more impactful than the choice of base model for economic survey data with mass-at-zero variables.

3. **Cross-source holdout evaluation.** I evaluate synthetic data quality using Precision, Density, and Coverage metrics adapted from {cite:t}`naeem2020reliable`, computed against holdouts from each source survey separately, revealing that coverage varies dramatically across sources — a pattern obscured by aggregate metrics.

## Methods

### Problem formulation

Let $\mathcal{S} = \{S_1, \ldots, S_K\}$ be $K$ surveys, each observing a subset of variables $V_k \subset V$ for $n_k$ records drawn from the same population. A set of shared variables $V_{\text{shared}} = \bigcap_k V_k$ appears in all surveys (e.g., age, sex). For each non-shared variable $v \in V_k \setminus V_{\text{shared}}$, I learn $P(v \mid V_{\text{shared}})$ from survey $S_k$.

This factorization implies a conditional independence assumption: non-shared variables from different sources are independent given $V_{\text{shared}}$. Furthermore, variables within the same source are also generated independently conditional on $V_{\text{shared}}$, destroying within-source correlations that the original survey captures. The resulting synthetic joint distribution is $P(V) = P(V_{\text{shared}}) \prod_{k} \prod_{v \in V_k \setminus V_{\text{shared}}} P(v \mid V_{\text{shared}})$. This preserves each marginal conditional but does not capture either cross-source or within-source correlations beyond what the shared variables mediate. The quality of this approximation depends on the richness of the shared variable set — with only demographic variables, it is coarse; with employment, education, and filing status added, it would improve substantially.

To generate synthetic records, I:
1. Sample shared variables from the pooled empirical distribution (with small Gaussian perturbation, $\sigma=0.1$, to smooth the discrete sample)
2. For each non-shared variable, sample from its learned conditional distribution
3. Calibrate weights against administrative targets

### Zero-inflation handling

Economic variables exhibit mass-at-zero: many people have zero values for income sources, benefit receipts, or tax credits. For a variable $y$ with zero fraction $\pi_0 \geq \theta$ (I use $\theta = 0.1$), the two-stage hurdle model decomposes generation into:

$$
y \sim \begin{cases} 0 & \text{with probability } \hat{\pi}_0(x) \\ g(x) & \text{with probability } 1 - \hat{\pi}_0(x) \end{cases}
$$

where $\hat{\pi}_0(x)$ is a random forest classifier predicting zero vs. non-zero, and $g(x)$ is the base model (QRF, QDNN, or MAF) trained only on positive values. This two-part decomposition follows the hurdle model tradition {cite:p}`mullahy1986specification` rather than the zero-inflated count model of {cite:t}`lambert1992zero`, since the economic variables here are continuous with mass at zero.

### Base models

I compare three model families, each with and without zero-inflation:

**Quantile regression forest (QRF).** Following {cite:t}`meinshausen2006quantile`, I fit a random forest that learns the full conditional distribution $P(y \mid x)$ by retaining quantile information from training observations in leaf nodes. At generation time, I uniformly sample one of five pre-computed quantile levels $\tau \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$ and return the corresponding predicted quantile. The quantile range is truncated to $[0.1, 0.9]$ to avoid extreme tail values; this reduces tail coverage but improves stability.

**Quantile deep neural network (QDNN).** A multi-layer perceptron trained with pinball loss {cite:p}`koenker2001quantile` to predict quantiles $\hat{q}_\tau(x)$ for $\tau \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}$. At generation time, I uniformly sample one of these seven quantile levels and return the corresponding prediction. This discrete quantile grid limits the resolution of the generated distribution; QDNN also exhibits higher variance across seeds than QRF or MAF (see Results).

**Masked autoregressive flow (MAF).** A normalizing flow {cite:p}`papamakarios2017masked` that learns the full conditional density $p(y \mid x)$ via invertible transformations. In the benchmark implementation, each non-shared variable gets its own 1-dimensional conditional flow $p(v \mid V_{\text{shared}})$; the MAF does not learn cross-variable dependencies within a source, making the conditional independence assumption shared across all three method families. I apply log transformation to positive values before standardization and train with maximum likelihood. Generated values are clipped to non-negative via `max(x, 0)`, which for the non-ZI variant creates an artificial mass at zero that may inflate the apparent ZI benefit.

### Evaluation metrics

I evaluate using three of the four metrics from the Precision, Recall, Density, and Coverage (PRDC) framework of {cite:t}`naeem2020reliable`, originally developed for evaluating generative image models. In Naeem et al.'s $k$-nearest-neighbor formulation, recall and coverage both measure the fraction of real points with a synthetic neighbor within the real manifold's local radius, making them mathematically equivalent (see their Definitions 2 and 4). I therefore report only Precision, Density, and Coverage (PDC).

For each source survey $S_k$:
1. Subsample to at most {eval}`f"{r.max_rows_per_source:,}"` records per source (to keep computation tractable)
2. Split into 80% train / 20% holdout
3. Train all methods on the training portions
4. Generate synthetic records
5. Compute PDC on columns present in $S_k$

**Coverage** — the primary metric — measures the fraction of real holdout points that have at least one synthetic neighbor within their $k$-th nearest-neighbor radius (I use $k={eval}`r.k`$). The radius is adaptive and per-point: dense regions of the real manifold have smaller radii, sparse regions larger. All distances are computed in standardized Euclidean space.

All results are reported as means ± standard errors across {eval}`r.n_seeds` random seeds.

### Calibration via reweighting

After synthesis, I calibrate the microdata against administrative targets by adjusting record weights. I compare six methods spanning three families.

**Calibration methods** solve for weights that match both categorical marginals and continuous targets simultaneously. Iterative proportional fitting (IPF) {cite:p}`deming1940least` is the classical raking algorithm that alternately adjusts weights to match each marginal target. Entropy balancing {cite:p}`hainmueller2012entropy`, originally developed for causal inference but applicable to any moment-matching reweighting problem, minimizes the Kullback-Leibler divergence from the original weights subject to target constraints: $\min_w \sum_i w_i \log(w_i / w_i^0)$ s.t. $Aw = b$, where $A$ is the constraint matrix and $b$ the vector of target values. SparseCalibrator, building on calibration estimator theory {cite:p}`deville1992calibration`, selects a sparse subset of records via cross-category proportional sampling, then calibrates the selected subset using iterative proportional fitting to match both categorical and continuous targets.

**Sparse optimization methods** ($L_1$-sparse and $L_0$-sparse) minimize the weight norm subject to categorical constraints only, solving $\min_w \|w\|_p$ s.t. $Aw = b$ for subset selection rather than population calibration.

**Differentiable sparse calibration** (HardConcrete) uses Hard Concrete gates {cite:p}`louizos2018learning` to learn both which records to include and what weight to assign them, jointly optimizing $L_0$ sparsity and target-matching accuracy via gradient descent. The implementation wraps the `l0-python` package. A key implementation detail: the initial weights must be rescaled so that the initial constraint violation is small (within ~30% of targets), otherwise gradient descent in the log-weight parameterization fails to converge from an initial point thousands of times larger than the target.

## Data

I use three public-use surveys stacked into a common format ({eval}`f"{r.n_total:,}"` total records across all survey years):

| Source | Records | Variables | Domain |
|--------|--------:|----------:|--------|
| SIPP | {eval}`f"{r.n_sipp:,}"` | 24 | Employment dynamics, job transitions, income |
| CPS ASEC | {eval}`f"{r.n_cps:,}"` | 10 | Wage and non-wage income, demographics |
| PSID | {eval}`f"{r.n_psid:,}"` | 15 | Longitudinal income, wealth, program participation |

Shared conditioning variables across all sources: age and sex (encoded as binary `is_male`). SIPP-specific variables include multi-job income, occupation/industry codes, job transitions, education, race, and marital status. CPS-specific variables include wage, dividend, interest, rental, farm, and self-employment income. PSID-specific variables include food stamp receipt, Social Security income, taxable income, and total family income.

For the benchmark, each source is subsampled to at most {eval}`f"{r.max_rows_per_source:,}"` records before the train/holdout split, keeping computation tractable while retaining sufficient data for reliable coverage estimates. The full dataset is available at [huggingface.co/datasets/nikhil-woodruff/microplex-benchmark-data](https://huggingface.co/datasets/nikhil-woodruff/microplex-benchmark-data).

## Results

### Synthesis: method comparison

```{code-cell} python
:tags: [remove-input]

import json
import pandas as pd
from pathlib import Path

# Load multi-seed results for primary table
with open(Path("..") / "benchmarks" / "results" / "benchmark_multi_seed.json") as f:
    ms_data = json.load(f)

rows = []
for name, sources in ms_data["methods"].items():
    sipp = sources.get("sipp", {})
    cps = sources.get("cps", {})
    psid = sources.get("psid", {})
    rows.append({
        "Method": name,
        "SIPP cov.": f"{sipp.get('mean', 0):.1%} ± {sipp.get('se', 0):.1%}",
        "CPS cov.": f"{cps.get('mean', 0):.1%} ± {cps.get('se', 0):.1%}",
        "PSID cov.": f"{psid.get('mean', 0):.0%}",
    })
df = pd.DataFrame(rows).sort_values("SIPP cov.", ascending=False)
df.index = range(1, len(df) + 1)
df
```

```{figure} figures/coverage_by_method.png
:name: fig-coverage
:width: 100%

PRDC coverage by synthesis method and source survey. Error bars show standard errors across {eval}`r.n_seeds` random seeds. PSID (0% for all methods) is omitted.
```

Per-source coverage varies dramatically across surveys. ZI-QRF achieves the highest SIPP coverage ({eval}`r.zi_qrf.sipp_pct`), while ZI-MAF leads on CPS ({eval}`r.zi_maf.cps_pct`). PSID coverage is 0% for all methods, reflecting a fundamental limitation of the current shared variable set: with only 2 conditioning variables (age, sex) and 15 PSID-specific columns, the model cannot learn the 15-dimensional joint structure from demographics alone.

I report per-source results as the primary metrics rather than aggregating across sources, since averaging with a degenerate 0% source obscures the pattern. QDNN exhibits notably higher variance across seeds than the other methods (SIPP coverage standard deviation of 8.5 percentage points, vs. below 1.2 for all others), likely reflecting sensitivity of pinball-loss training to the train/holdout split. This instability should be considered when evaluating QDNN for production use.

### The zero-inflation effect

The differential impact of zero-inflation across model families is the most consistent pattern in the results:

| Base model | Without ZI | With ZI | Lift |
|-----------|-----------|---------|------|
| MAF | {eval}`r.maf.coverage_pct` | {eval}`r.zi_maf.coverage_pct` | +{eval}`r.zi_maf_vs_maf_lift` |
| QDNN | {eval}`r.qdnn.coverage_pct` | {eval}`r.zi_qdnn.coverage_pct` | +{eval}`r.zi_qdnn_vs_qdnn_lift` |
| QRF | {eval}`r.qrf.coverage_pct` | {eval}`r.zi_qrf.coverage_pct` | +{eval}`r.zi_qrf_vs_qrf_lift` |

MAF without zero-inflation ({eval}`r.maf.coverage_pct` mean coverage) performs worse than plain QRF ({eval}`r.qrf.coverage_pct`). Adding zero-inflation lifts MAF coverage substantially ({eval}`r.zi_maf.coverage_pct`). This is consistent with the hypothesis that normalizing flows cannot jointly model the zero mass and positive-value density as a single distribution, but perform well on the smooth positive-value conditional once freed from this burden. Note that the non-ZI MAF also clips generated values to non-negative via `max(x, 0)`, which creates an artificial zero mass; the ZI lift for MAF thus conflates a genuine methodological improvement with partial mitigation of this clipping artifact.

QRF is naturally robust to zero-inflation because quantile forests can represent mixed distributions — leaf nodes containing both zero and positive training observations produce quantile predictions that implicitly capture the zero mass. The minimal ZI lift for QRF (+{eval}`r.zi_qrf_vs_qrf_lift`) reflects that forests already handle this case.

### Speed-accuracy tradeoff

ZI-QRF completes in {eval}`r.zi_qrf.time_str`, compared to ZI-MAF's {eval}`r.zi_maf.time_str` (ZI-MAF is {eval}`r.zi_speedup_over_maf` slower). ZI-QRF achieves higher SIPP coverage ({eval}`r.zi_qrf.sipp_pct` vs. {eval}`r.zi_maf.sipp_pct`), while ZI-MAF has a {eval}`f"{r.zi_maf.cps_coverage - r.zi_qrf.cps_coverage:.0%}"` point CPS coverage advantage ({eval}`r.zi_maf.cps_pct` vs. {eval}`r.zi_qrf.cps_pct`). For production pipelines requiring frequent regeneration, ZI-QRF achieves comparable coverage at substantially lower computational cost.

### Reweighting calibration

I evaluate reweighting methods on {eval}`f"{r.rw_n_records:,}"` records using a train/test split to assess out-of-sample generalization. Methods are calibrated on {eval}`r.rw_n_train_targets` training targets (age group categories plus total population weight), then evaluated on {eval}`r.rw_n_test_targets` held-out test targets (sex categories) that were not used during calibration. Target values are perturbed from the sample distribution by 10-30% to simulate calibration to known population totals. This design measures whether calibrating on one set of demographic margins improves representativeness along dimensions not explicitly targeted — the relevant question for practical survey calibration.

```{code-cell} python
:tags: [remove-input]

import json
import pandas as pd
from pathlib import Path

with open(Path("..") / "benchmarks" / "results" / "reweighting_full.json") as f:
    rw_data = json.load(f)

rows = []
for name, m in rw_data["methods"].items():
    rows.append({
        "Method": name,
        "Train error": f"{m['train_mean_error']:.2%}",
        "Test error": f"{m['test_mean_error']:.1%}",
        "Weight CV": f"{m['weight_cv']:.3f}",
        "Sparsity": f"{m['sparsity']:.1%}",
        "Time (s)": f"{m['elapsed_seconds']:.2f}",
    })
df = pd.DataFrame(rows)
df = df.sort_values("Test error")
df.index = range(1, len(df) + 1)
df
```

All calibration methods (IPF, entropy, SparseCalibrator) achieve near-zero training error — they satisfy the age and weight constraints exactly. The key comparison is test error on the held-out sex margin, which measures generalization.

To characterize the accuracy-sparsity tradeoff, I sweep the regularization parameter for each sparse method (SparseCalibrator's $\lambda$ and HardConcrete's $\lambda_{L_0}$) across several orders of magnitude and plot out-of-sample error against the number of records with non-zero weight. HardConcrete uses a non-convex optimizer, so I report mean $\pm$ SE over 5 random seeds.

```{figure} figures/reweighting_frontier.png
:name: fig-reweighting-frontier
:width: 100%

Reweighting frontier: out-of-sample error on held-out sex margin vs. number of active records. SparseCalibrator ($L_1$, convex) traces a deterministic frontier; HardConcrete ($L_0$, non-convex) shows mean $\pm$ SE over 5 seeds. Hard-constraint endpoints ($L_1$-Sparse, $L_0$-Sparse) at 5 records are far off the efficient frontier.
```

{numref}`fig-reweighting-frontier` shows that SparseCalibrator ($L_1$, convex) dominates HardConcrete ($L_0$, non-convex) across the entire frontier. At high sparsity (~30 records), SparseCalibrator achieves ~9% test error — half that of dense methods — with zero variance because the underlying FISTA optimizer is deterministic. HardConcrete {cite:p}`louizos2018learning` matches dense methods (~18%) with $>$300 records, but its error bars explode at high sparsity (mean 32-46% with $\pm$10-14% SE below 100 records), reflecting the difficulty of non-convex optimization with very few degrees of freedom. HardConcrete uses differentiable $L_0$ gates based on the Hard Concrete distribution to jointly optimize which records to keep and what weights to assign, implemented via the `l0-python` package. An initial weight rescaling step is critical: without it, gradient descent fails to converge because survey weights (mean ~6,800) produce initial constraint violations of 5,000x or more.

Dense calibration methods (IPF, entropy) use all {eval}`f"{r.rw_n_records:,}"` records and achieve ~18% test error. SparseCalibrator at its default operating point produces {eval}`r.sparse_cal_cv_vs_ipf` lower weight coefficient of variation ({eval}`r.rw_sparse_cal.cv_str` vs. {eval}`r.rw_ipf.cv_str`), meaning smoother weights that are less likely to amplify noise in downstream estimates.

The hard-constraint $L_1$- and $L_0$-sparse methods ({eval}`r.rw_l1.test_error_pct` test error) sit far off the efficient frontier. They solve `min $\|w\|_p$ s.t. $Aw = b$` — pure sparsity with no accuracy tradeoff — selecting just 5 records that cannot maintain representativeness on held-out dimensions. SparseCalibrator and HardConcrete are the parameterized versions that interpolate between dense calibration and extreme sparsity.

## Discussion

### Zero-inflation as architectural choice

The most consistent finding is that zero-inflation handling provides large coverage gains for neural methods while barely affecting tree-based methods. A two-stage decomposition — random forest classifier {cite:p}`breiman2001random` for zero vs. non-zero, followed by a conditional model on positive values only — transforms underperforming MAF and QDNN methods into competitive performers. The zero-inflation lift exceeds the between-model-family differences for neural methods.

The mechanism follows from the structure of economic survey variables. Income sources (wages, dividends, transfers) are zero for large population fractions. Without ZI, a normalizing flow or neural network must simultaneously model: (a) the probability of being a recipient, and (b) the distribution of amounts conditional on receipt. These are distinct modeling tasks — one is a classification boundary, the other a continuous density estimation — and conflating them degrades both. Tree-based methods handle this naturally through leaf node composition.

### Limitations

With only age and sex as shared conditioning variables, the model cannot capture the covariance structure that depends on education, occupation, geography, and other demographics. The 0% PSID coverage across all methods demonstrates this shared variable bottleneck. Expanding shared variables is the highest-priority improvement.

Non-shared variables are generated independently conditional on shared variables — both across and within sources. The synthetic joint distribution is $\prod_v P(v \mid V_{\text{shared}})$, which preserves each marginal conditional but destroys all correlations not mediated by the shared variables. For microsimulation applications, the correlation between (e.g.) SIPP program participation and CPS income components is precisely what is needed. The current framework does not capture these relationships, and I do not evaluate cross-source correlation fidelity. A full joint model — via a unified latent space, conditional dependency chains, or copula-based approaches {cite:p}`dorazio2006statistical` — would address this at the cost of additional complexity.

The benchmark treats all survey records equally, ignoring complex sampling designs and survey weights. This biases the learned distributions toward oversampled strata and may not reflect the population distributions that practitioners need.

Current synthesis operates at the person level. Realistic microdata requires consistent household structure: spouses should have compatible incomes, dependents should be children, tax unit filing status should match household composition. Hierarchical synthesis and relationship pointers (spouse_person_id, parent_person_id) are planned for future work.

The PDC metrics adapted from computer vision may not capture the properties that survey statisticians prioritize, such as marginal distributional fidelity, cross-tabulation accuracy, or analytical validity (whether regressions on synthetic data replicate those on real data). Adding survey-standard evaluation metrics would strengthen the evaluation.

I exclude CTGAN and TVAE {cite:p}`xu2019modeling` from the current benchmark due to dependency constraints. Adding these baselines, along with recent diffusion-based methods like Forest Flow {cite:p}`jolicoeurmartineau2024generating`, would strengthen the comparison.

### Future work

The most impactful improvement would be expanding the shared variable set beyond age and sex to include employment status, education, marital status, filing status, and disability indicators. This would strengthen the conditioning bridge between sources and address the 0% PSID coverage. Second, hierarchical synthesis preserving household, tax unit, and person structure {cite:p}`gale2022simulating` would make the synthetic data usable for tax-benefit microsimulation. Third, adding diffusion-based methods (Forest Flow {cite:p}`jolicoeurmartineau2024generating`) and established baselines (CTGAN, TVAE) would strengthen the methodological comparison. Finally, incorporating survey weights into training and adding survey-standard evaluation metrics (marginal comparisons, analytical validity) would make the framework more relevant to survey statisticians.

## Conclusion

Microplex learns per-variable conditional distributions from multiple government surveys and generates synthetic records with complete variable coverage. The central empirical finding is that zero-inflation handling — separating zero/non-zero classification from positive-value density estimation — matters more than the choice of base model for neural methods. This two-stage decomposition lifts MAF coverage by {eval}`r.zi_maf_vs_maf_lift` and QDNN by {eval}`r.zi_qdnn_vs_qdnn_lift`, while barely affecting tree-based QRF (+{eval}`r.zi_qrf_vs_qrf_lift`). The pattern is stable across {eval}`r.n_seeds` random seeds, suggesting that practitioners working with economic survey data should implement zero-inflation handling before selecting a base model.

The framework has clear limitations in its current form: the conditional independence assumption and narrow shared variable set (age, sex) mean cross-source correlations are not captured, as demonstrated by the 0% PSID coverage. These limitations are addressable — expanding shared variables and modeling cross-source dependencies are the highest-priority improvements for making the synthetic data usable in production microsimulation.

## References

```{bibliography}
:style: unsrt
```
