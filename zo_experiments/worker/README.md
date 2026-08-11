# Step ZO-8C: fixed-configuration logical-worker sensitivity

## 1. Scope and audit

This report merges the completed m=1,2,4 Step ZO-8B trajectories with the hash-verified m=8 formal trajectories. All four methods reuse their frozen d=100,m=8 parameters without worker-specific retuning. It is a logical single-process sensitivity experiment, not a wall-clock or multi-machine speedup benchmark.

- Audit: **pass**; 320/320 worker-method-seed tasks and 194,080 checkpoints.
- Logical workers: 1, 2, 4, 8; methods: NOG-ZO, ME-DOL-ZO, DGFM, DGFM+.
- Formal seeds: 0--19; maximum training work: 983,040 two-point SZO calls.
- A confirmed hit requires two consecutive method-independent evaluation checkpoints.
- Primary epsilon: 0.05. Epsilon 0.03 and below are descriptive/censored.
- Checkpoint intervals were work-aligned within method; ME-DOL remains restricted to complete epochs.

![Worker hit, depth, and work](figures/worker_hit_depth_work.png)

## 2. Primary confirmed first-hit results

Each cell is `hits/20; conditional mean depth; conditional mean total work; conditional mean per-worker work`. Conditional means must be read with hit counts.

| workers | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---:|---:|---:|---:|---:|
| 1 | 20/20; 214.2; 219,340.8; 219,340.8 | 20/20; 348.0; 11,136.0; 11,136.0 | 20/20; 1,385.6; 22,169.6; 22,169.6 | 20/20; 13,360.0; 227,132.8; 227,132.8 |
| 2 | 20/20; 214.2; 219,340.8; 109,670.4 | 20/20; 326.4; 20,889.6; 10,444.8 | 20/20; 1,355.2; 43,366.4; 21,683.2 | 18/20; 4,549.3; 154,762.7; 77,381.3 |
| 4 | 20/20; 214.0; 219,136.0; 54,784.0 | 20/20; 378.0; 48,384.0; 12,096.0 | 20/20; 1,304.0; 83,456.0; 20,864.0 | 20/20; 1,542.4; 105,011.2; 26,252.8 |
| 8 | 20/20; 214.4; 219,545.6; 27,443.2 | 20/20; 496.2; 127,027.2; 15,878.4 | 20/20; 1,296.0; 165,888.0; 20,736.0 | 20/20; 1,410.0; 192,076.8; 24,009.6 |

DGFM+ at m=2 has 18/20 hits. Its conditional first-hit means and ratios are therefore censored and may be optimistic; the two non-hits remain in the per-seed and capped summaries.

## 3. Same-seed ratios relative to one worker

![Within-method worker ratios](figures/worker_relative_to_m1.png)

Ratios compare each worker count with m=1 for the same method and seed. Confidence intervals are 2,000-repetition seed bootstrap intervals.

| method | workers | pairs | depth ratio (95% CI) | total-work ratio (95% CI) | per-worker ratio (95% CI) |
|---|---:|---:|---:|---:|---:|
| DGFM | 1 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| DGFM | 2 | 20/20 | 0.98 [0.94, 1.03] | 1.97 [1.89, 2.06] | 0.98 [0.94, 1.03] |
| DGFM | 4 | 20/20 | 0.95 [0.91, 0.98] | 3.79 [3.64, 3.94] | 0.95 [0.91, 0.98] |
| DGFM | 8 | 20/20 | 0.94 [0.91, 0.98] | 7.53 [7.26, 7.83] | 0.94 [0.91, 0.98] |
| DGFM+ | 1 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| DGFM+ | 2 | 18/20 | 0.32 [0.20, 0.49] | 0.65 [0.40, 0.99] | 0.32 [0.20, 0.50] |
| DGFM+ | 4 | 20/20 | 0.13 [0.11, 0.15] | 0.51 [0.44, 0.58] | 0.13 [0.11, 0.15] |
| DGFM+ | 8 | 20/20 | 0.12 [0.10, 0.13] | 0.92 [0.81, 1.04] | 0.12 [0.10, 0.13] |
| ME-DOL-ZO | 1 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| ME-DOL-ZO | 2 | 20/20 | 0.96 [0.89, 1.02] | 1.91 [1.78, 2.04] | 0.96 [0.89, 1.02] |
| ME-DOL-ZO | 4 | 20/20 | 1.11 [1.03, 1.19] | 4.44 [4.13, 4.77] | 1.11 [1.03, 1.19] |
| ME-DOL-ZO | 8 | 20/20 | 1.45 [1.37, 1.54] | 11.63 [10.87, 12.34] | 1.45 [1.36, 1.55] |
| NOG-ZO | 1 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| NOG-ZO | 2 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.50 [0.50, 0.50] |
| NOG-ZO | 4 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.25 [0.25, 0.25] |
| NOG-ZO | 8 | 20/20 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.13 [0.12, 0.13] |

## 4. Log-log sensitivity slopes

Slopes fit log(mean first-hit metric) against log(m). The bootstrap jointly resamples seeds across all four worker counts. These are empirical fixed-configuration sensitivities, not preregistered exact worker exponents.

| method | common complete seeds | depth slope (95% CI) | total-work slope (95% CI) | per-worker slope (95% CI) |
|---|---:|---:|---:|---:|
| NOG-ZO | 20/20 | 0.000 [-0.001, 0.002] | 0.000 [-0.001, 0.002] | -1.000 [-1.001, -0.998] |
| ME-DOL-ZO | 20/20 | 0.175 [0.146, 0.204] | 1.175 [1.146, 1.203] | 0.175 [0.147, 0.204] |
| DGFM | 20/20 | -0.034 [-0.051, -0.017] | 0.966 [0.949, 0.982] | -0.034 [-0.050, -0.018] |
| DGFM+ | 18/20 | -1.116 [-1.231, -0.982] | -0.115 [-0.235, 0.016] | -1.115 [-1.234, -0.980] |

## 5. Terminal accounting

![Terminal accounting](figures/worker_terminal_accounting.png)

Terminal total training work is 983,040 for NOG-ZO, ME-DOL-ZO, and DGFM. DGFM+ stops at the largest valid restart-aligned round and is within 576 calls of the same cap. Terminal per-worker work follows the expected near-1/m decomposition by construction. Evaluation work remains separate.

## 6. Interpretation and theory boundary

For NOG-ZO at epsilon=0.05, mean first-hit depth varies by only 1.002x and total work by 1.002x across m=1--8, while mean per-worker first-hit work falls to 0.125 of the m=1 value (approximately 1/8). This is the cleanest empirical result of the worker study.

The other methods use fixed per-worker training batches, whereas NOG-ZO splits a fixed global batch. Consequently their total first-hit work need not stay constant as m changes. This implementation distinction is part of the frozen protocol and must be disclosed; the experiment does not establish a universally fair wall-clock comparison across worker counts.

ME-DOL-ZO depth is mildly non-monotone at small m and rises at m=8. DGFM depth is nearly flat. DGFM+ depth falls strongly with m, but its m=2 estimate is censored. These observations are method sensitivity results, not evidence for exact asymptotic m powers.

## 7. Descriptive epsilon=0.03 boundary

At epsilon=0.03, ME-DOL-ZO/m=1 and DGFM+/m=2,4 have 0/20 confirmed hits; all remaining method-worker combinations have 20/20. No finite ratios or slopes are reported across that incomplete grid. Full capped and conditional values remain in worker_summary.csv.

## 8. Paper-safe claim

> Under one frozen configuration and fixed total training-work cap, NOG-ZO maintains essentially unchanged confirmed-first-hit communication depth and total work from one to eight logical workers, while its accounted per-worker work decreases approximately as 1/m. This is a logical work-decomposition result and not a measured multi-process speedup.

Do not claim real cluster speedup, worker-wise optimal tuning, exact worker exponents, or complete primary coverage without mentioning the two DGFM+/m=2 non-hits.

## 9. Reproduction

~~~bash
conda run -n NOG python -m src.distributed.zo_worker_analysis
~~~

Raw trajectories under outputs/distributed_zo are hash-audited and are not modified by this analysis.
