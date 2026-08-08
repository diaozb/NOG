# Formal ZO experiment (Step ZO-5B)

This report is generated from the frozen 20-seed formal run. No formal seed was used for parameter selection.

## Protocol

- Methods: NOG-ZO, ME-DOL-ZO, DGFM, DGFM+.
- Formal seeds: 20 (0 through 19).
- Equal total-work budget per method and seed: 983,040.
- Workers: 8; evaluation every 4 communication-depth units.
- Evaluation bank: smooth B=256, data B=512.
- A hit requires 2 consecutive checkpoints at or below epsilon.
- Primary endpoint: paired communication-depth ratio, baseline / NOG-ZO.
- Work ratios are secondary descriptive endpoints.

Frozen candidates:

- NOG-ZO: {"M": 2, "eta": 0.01, "smooth_B": 8}
- ME-DOL-ZO: {"epoch_length": 12, "theory_multiplier": 60.0}
- DGFM: {"eta": 0.05}
- DGFM+: {"eta": 0.05}

## Representative first-hit results

Each cell reports **hits/20; conditional mean depth; conditional mean work**. Conditional means exclude non-hits and must not be interpreted as complete-sample means.

| epsilon | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---:|---:|---:|---:|---:|
| 0.2000 | 20/20; 103.4; 105,881.6 | 20/20; 165.0; 42,240.0 | 20/20; 367.6; 47,052.8 | 20/20; 364.0; 49,856.0 |
| 0.1000 | 20/20; 180.2; 184,524.8 | 20/20; 340.2; 87,091.2 | 20/20; 855.2; 109,465.6 | 20/20; 872.0; 118,950.4 |
| 0.0500 | 20/20; 214.4; 219,545.6 | 20/20; 496.2; 127,027.2 | 20/20; 1,296.0; 165,888.0 | 20/20; 1,410.0; 192,076.8 |
| 0.0200 | 20/20; 237.2; 242,892.8 | 20/20; 780.6; 199,833.6 | 20/20; 1,993.6; 255,180.8 | 0/20; —; — |
| 0.0180 | 20/20; 240.2; 245,964.8 | 20/20; 852.0; 218,112.0 | 20/20; 2,111.6; 270,284.8 | 0/20; —; — |
| 0.0160 | 18/20; 242.0; 247,808.0 | 20/20; 991.8; 253,900.8 | 20/20; 2,270.0; 290,560.0 | 0/20; —; — |
| 0.0150 | 11/20; 244.0; 249,856.0 | 20/20; 1,247.4; 319,334.4 | 20/20; 2,392.8; 306,278.4 | 0/20; —; — |
| 0.0140 | 1/20; 244.0; 249,856.0 | 20/20; 1,860.0; 476,160.0 | 20/20; 2,523.6; 323,020.8 | 0/20; —; — |
| 0.0130 | 0/20; —; — | 12/20; 2,270.0; 581,120.0 | 20/20; 2,697.2; 345,241.6 | 0/20; —; — |
| 0.0100 | 0/20; —; — | 0/20; —; — | 0/20; —; — | 0/20; —; — |
| 0.0090 | 0/20; —; — | 0/20; —; — | 0/20; —; — | 0/20; —; — |
| 0.0080 | 0/20; —; — | 0/20; —; — | 0/20; —; — | 0/20; —; — |
| 0.0050 | 0/20; —; — | 0/20; —; — | 0/20; —; — | 0/20; —; — |
| 0.0020 | 0/20; —; — | 0/20; —; — | 0/20; —; — | 0/20; —; — |

## Representative paired ratios

Ratios are computed within the same formal seed and then averaged. Only rows with all 20 paired hits support the complete-pair trend; partial rows are explicitly labelled.

| baseline | epsilon | paired hits | depth ratio (95% CI) | work ratio (95% CI) | status |
|---|---:|---:|---:|---:|---|
| ME-DOL-ZO | 0.2000 | 20/20 | 1.60 [1.57, 1.62] | 0.40 [0.39, 0.41] | complete |
| ME-DOL-ZO | 0.1000 | 20/20 | 1.89 [1.87, 1.91] | 0.47 [0.47, 0.48] | complete |
| ME-DOL-ZO | 0.0500 | 20/20 | 2.31 [2.28, 2.35] | 0.58 [0.57, 0.59] | complete |
| ME-DOL-ZO | 0.0200 | 20/20 | 3.29 [3.21, 3.37] | 0.82 [0.80, 0.84] | complete |
| ME-DOL-ZO | 0.0180 | 20/20 | 3.55 [3.45, 3.65] | 0.89 [0.86, 0.91] | complete |
| ME-DOL-ZO | 0.0160 | 18/20 | 4.15 [3.93, 4.39] | 1.04 [0.98, 1.10] | censored |
| ME-DOL-ZO | 0.0150 | 11/20 | 5.63 [4.88, 6.60] | 1.41 [1.23, 1.64] | censored |
| ME-DOL-ZO | 0.0140 | 1/20 | 4.62 [4.62, 4.62] | 1.16 [1.16, 1.16] | censored |
| ME-DOL-ZO | 0.0130 | 0/20 | — [—, —] | — [—, —] | censored |
| ME-DOL-ZO | 0.0100 | 0/20 | — [—, —] | — [—, —] | censored |
| ME-DOL-ZO | 0.0090 | 0/20 | — [—, —] | — [—, —] | censored |
| ME-DOL-ZO | 0.0080 | 0/20 | — [—, —] | — [—, —] | censored |
| ME-DOL-ZO | 0.0050 | 0/20 | — [—, —] | — [—, —] | censored |
| ME-DOL-ZO | 0.0020 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM | 0.2000 | 20/20 | 3.56 [3.51, 3.60] | 0.44 [0.44, 0.45] | complete |
| DGFM | 0.1000 | 20/20 | 4.75 [4.71, 4.78] | 0.59 [0.59, 0.60] | complete |
| DGFM | 0.0500 | 20/20 | 6.04 [5.99, 6.09] | 0.76 [0.75, 0.76] | complete |
| DGFM | 0.0200 | 20/20 | 8.40 [8.26, 8.53] | 1.05 [1.03, 1.07] | complete |
| DGFM | 0.0180 | 20/20 | 8.79 [8.63, 8.96] | 1.10 [1.08, 1.12] | complete |
| DGFM | 0.0160 | 18/20 | 9.39 [9.19, 9.59] | 1.17 [1.15, 1.20] | censored |
| DGFM | 0.0150 | 11/20 | 9.85 [9.59, 10.16] | 1.23 [1.20, 1.27] | censored |
| DGFM | 0.0140 | 1/20 | 11.28 [11.28, 11.28] | 1.41 [1.41, 1.41] | censored |
| DGFM | 0.0130 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM | 0.0100 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM | 0.0090 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM | 0.0080 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM | 0.0050 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM | 0.0020 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.2000 | 20/20 | 3.52 [3.39, 3.65] | 0.47 [0.45, 0.49] | complete |
| DGFM+ | 0.1000 | 20/20 | 4.84 [4.72, 4.95] | 0.64 [0.63, 0.66] | complete |
| DGFM+ | 0.0500 | 20/20 | 6.58 [6.37, 6.79] | 0.87 [0.85, 0.90] | complete |
| DGFM+ | 0.0200 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0180 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0160 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0150 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0140 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0130 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0100 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0090 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0080 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0050 | 0/20 | — [—, —] | — [—, —] | censored |
| DGFM+ | 0.0020 | 0/20 | — [—, —] | — [—, —] | censored |

## Trend diagnostics

The diagnostics below use only epsilon points at which all 20 same-seed pairs hit. A positive log-log slope or positive Spearman coefficient means the ratio tends to increase as epsilon decreases.

| baseline | complete points | depth slope | depth Spearman | work slope | work Spearman |
|---|---:|---:|---:|---:|---:|
| ME-DOL-ZO | 16 | 0.319 | 1.000 | 0.319 | 1.000 |
| DGFM | 16 | 0.363 | 1.000 | 0.363 | 1.000 |
| DGFM+ | 13 | 0.529 | 1.000 | 0.527 | 1.000 |

## Figures

![Formal hit rate, depth and work](figures/formal_hit_depth_work.png)

![Formal paired ratios](figures/formal_ratios.png)

In both figures, solid ratio curves require all 20 paired hits. Crosses denote partially censored points and are not used as complete-sample scaling evidence.

## Reproducibility artifacts

- formal_per_seed.csv: first confirmed hit or censor limit for every method, seed and epsilon.
- formal_summary.csv: hit counts and conditional/capped summaries.
- formal_ratios.csv: same-seed paired ratios with bootstrap 95% CIs.
- formal_trends.json: diagnostics on the complete-pair interval.
- audit.json: identity, parameter, monotonicity and budget checks.
- analysis_manifest.json: hashes of analysis inputs and outputs.
- theory_comparison.csv/json: Step ZO-5C theoretical ratio powers, observed
  slopes, and joint-seed bootstrap confidence intervals.
- STEP_ZO_5C_THEORY_COMPARISON.md: theory comparison, censoring boundary,
  paper-safe wording, and prohibited overclaims.
- anomaly_task_diagnostics.csv and anomaly_method_summary.csv: location of
  raw/confirmed floors, rebound diagnostics, and extension outlook.
- boundary_checkpoint_sensitivity.csv: one-checkpoint versus confirmed hit
  rates on a dense boundary grid.
- STEP_ZO_6A_ANOMALY_AUDIT.md: full anomaly and censoring diagnosis.
- anomaly_replication_comparison.csv/json: formal-versus-reserved-seed
  trajectory comparison and independent bootstrap intervals.
- STEP_ZO_6C_REPLICATION_DECISION.md: replicated anomaly conclusion and
  frozen-budget extension decision.

Regenerate from the repository root with:

    conda run -n NOG python -m src.distributed.zo_formal_analysis

## Interpretation boundary

This finite-budget simulation can support an empirical depth trend, but it does not prove the asymptotic exponents. At smaller epsilon values, non-hits are right-censored by the fixed budget. Conditional first-hit means in that region are susceptible to survivor bias; capped summaries are supplied only as budget-bound descriptions, not as true first-hit ratios.

The completed Step ZO-5C interpretation is available in
[STEP_ZO_5C_THEORY_COMPARISON.md](STEP_ZO_5C_THEORY_COMPARISON.md).

The completed Step ZO-6A diagnosis is available in
[STEP_ZO_6A_ANOMALY_AUDIT.md](STEP_ZO_6A_ANOMALY_AUDIT.md).

The completed Step ZO-6C replication decision is available in
[STEP_ZO_6C_REPLICATION_DECISION.md](STEP_ZO_6C_REPLICATION_DECISION.md).
