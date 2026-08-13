# Step ZO-7C: fixed-configuration dimension sensitivity

## 1. Scope and audit

This report merges the completed d=25,50,200 Step ZO-7B runs with the hash-verified d=100 formal trajectories. It does not retune any method. The result is a qualitative fixed-configuration sensitivity check, not an exact dimension-exponent experiment.

- Audit: **pass**; 320/320 dimension-method-seed tasks.
- Dimensions: 25, 50, 100, 200.
- Methods: NOG-ZO, ME-DOL-ZO, DGFM, DGFM+; formal seeds: 0--19.
- Maximum work per task: 983,040 two-point SZO calls; eight logical workers.
- A hit requires two consecutive method-independent evaluation checkpoints at or below epsilon.
- Primary epsilons: 0.05 and 0.03; both have complete 20/20 coverage for every method and dimension.

![Absolute dimension metrics](figures/dimension_hit_depth_work.png)

## 2. Absolute confirmed first-hit results

Each cell is `hits/20; mean depth; mean work`. All primary means are uncensored because all 20 seeds hit.

### epsilon = 0.05

| dimension | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---:|---:|---:|---:|---:|
| 25 | 20/20; 148.2; 151,756.8 | 20/20; 342.6; 87,705.6 | 20/20; 936.8; 119,910.4 | 20/20; 1,050.0; 143,078.4 |
| 50 | 20/20; 182.2; 186,572.8 | 20/20; 418.2; 107,059.2 | 20/20; 1,156.8; 148,070.4 | 20/20; 1,238.8; 168,896.0 |
| 100 | 20/20; 214.4; 219,545.6 | 20/20; 496.2; 127,027.2 | 20/20; 1,296.0; 165,888.0 | 20/20; 1,410.0; 192,076.8 |
| 200 | 20/20; 250.0; 256,000.0 | 20/20; 644.4; 164,966.4 | 20/20; 1,388.8; 177,766.4 | 20/20; 1,529.2; 208,332.8 |

### epsilon = 0.03

| dimension | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---:|---:|---:|---:|---:|
| 25 | 20/20; 159.4; 163,225.6 | 20/20; 438.6; 112,281.6 | 20/20; 1,205.2; 154,265.6 | 20/20; 1,730.8; 235,712.0 |
| 50 | 20/20; 194.8; 199,475.2 | 20/20; 543.0; 139,008.0 | 20/20; 1,480.8; 189,542.4 | 20/20; 1,893.2; 257,804.8 |
| 100 | 20/20; 229.2; 234,700.8 | 20/20; 622.8; 159,436.8 | 20/20; 1,652.4; 211,507.2 | 20/20; 2,609.6; 355,302.4 |
| 200 | 20/20; 266.0; 272,384.0 | 20/20; 784.8; 200,908.8 | 20/20; 1,774.0; 227,072.0 | 20/20; 2,604.8; 354,649.6 |

## 3. Same-seed baseline/NOG-ZO ratios

Ratios are means of the 20 within-seed ratios, not ratios of method means. Parentheses give bootstrap 95% confidence intervals.

![Paired dimension ratios](figures/dimension_ratios.png)

### epsilon = 0.05

| dimension | baseline | depth ratio (95% CI) | work ratio (95% CI) |
|---:|---|---:|---:|
| 25 | DGFM | 6.32 [6.21, 6.44] | 0.79 [0.78, 0.81] |
| 25 | DGFM+ | 7.08 [6.81, 7.38] | 0.94 [0.91, 0.98] |
| 25 | ME-DOL-ZO | 2.31 [2.27, 2.36] | 0.58 [0.57, 0.59] |
| 50 | DGFM | 6.35 [6.28, 6.42] | 0.79 [0.78, 0.80] |
| 50 | DGFM+ | 6.80 [6.54, 7.06] | 0.91 [0.87, 0.94] |
| 50 | ME-DOL-ZO | 2.30 [2.27, 2.32] | 0.57 [0.57, 0.58] |
| 100 | DGFM | 6.04 [5.99, 6.09] | 0.76 [0.75, 0.76] |
| 100 | DGFM+ | 6.58 [6.38, 6.78] | 0.87 [0.85, 0.90] |
| 100 | ME-DOL-ZO | 2.31 [2.28, 2.35] | 0.58 [0.57, 0.59] |
| 200 | DGFM | 5.56 [5.50, 5.61] | 0.69 [0.69, 0.70] |
| 200 | DGFM+ | 6.12 [6.00, 6.23] | 0.81 [0.80, 0.83] |
| 200 | ME-DOL-ZO | 2.58 [2.55, 2.61] | 0.64 [0.64, 0.65] |

### epsilon = 0.03

| dimension | baseline | depth ratio (95% CI) | work ratio (95% CI) |
|---:|---|---:|---:|
| 25 | DGFM | 7.56 [7.43, 7.71] | 0.95 [0.93, 0.96] |
| 25 | DGFM+ | 10.86 [9.85, 12.13] | 1.44 [1.31, 1.62] |
| 25 | ME-DOL-ZO | 2.75 [2.70, 2.80] | 0.69 [0.67, 0.70] |
| 50 | DGFM | 7.60 [7.48, 7.73] | 0.95 [0.93, 0.97] |
| 50 | DGFM+ | 9.72 [9.16, 10.29] | 1.29 [1.22, 1.37] |
| 50 | ME-DOL-ZO | 2.79 [2.73, 2.85] | 0.70 [0.68, 0.71] |
| 100 | DGFM | 7.21 [7.12, 7.31] | 0.90 [0.89, 0.91] |
| 100 | DGFM+ | 11.39 [10.32, 12.53] | 1.51 [1.37, 1.66] |
| 100 | ME-DOL-ZO | 2.72 [2.67, 2.76] | 0.68 [0.67, 0.69] |
| 200 | DGFM | 6.67 [6.59, 6.76] | 0.83 [0.82, 0.84] |
| 200 | DGFM+ | 9.79 [9.25, 10.41] | 1.30 [1.23, 1.38] |
| 200 | ME-DOL-ZO | 2.95 [2.92, 2.98] | 0.74 [0.73, 0.75] |

## 4. Dimension slopes and theory boundary

The empirical slope fits log(mean paired ratio) against log(d). Confidence intervals jointly resample the same 20 seed IDs across all four dimensions. The theory column is a reference power only; exact exponent recovery was not preregistered.

| baseline | epsilon | depth slope (95% CI) | theory | work slope (95% CI) | theory |
|---|---:|---:|---:|---:|---:|
| ME-DOL-ZO | 0.05 | 0.048 [0.037, 0.060] | 0.667 | 0.048 [0.037, 0.059] | 0.000 |
| ME-DOL-ZO | 0.03 | 0.026 [0.016, 0.037] | 0.667 | 0.026 [0.016, 0.038] | 0.000 |
| DGFM | 0.05 | -0.063 [-0.072, -0.054] | 1.167 | -0.063 [-0.071, -0.054] | 0.500 |
| DGFM | 0.03 | -0.062 [-0.071, -0.052] | 1.167 | -0.062 [-0.072, -0.052] | 0.500 |
| DGFM+ | 0.05 | -0.068 [-0.085, -0.052] | 1.167 | -0.069 [-0.085, -0.052] | 0.500 |
| DGFM+ | 0.03 | -0.022 [-0.081, 0.033] | 1.167 | -0.022 [-0.082, 0.034] | 0.500 |

## 5. Interpretation

Only 2/6 primary baseline/epsilon depth slopes have a bootstrap interval strictly above zero, and 0/6 intervals contain the corresponding worst-case reference power.

The absolute first-hit depth generally rises with dimension, but the relative baseline/NOG-ZO curves are mostly flat or non-monotone. Therefore this fixed-configuration experiment does **not** empirically recover the theoretical dimension powers. It remains useful as a reproducible sensitivity check showing that the epsilon-scale communication advantage is not created by a single d=100 run.

Likely reasons include frozen d=100 hyperparameters, finite dimension-independent work budgets, non-tight worst-case bounds, and a synthetic data generator whose instance geometry changes with d. A true exponent experiment would need dimension-aware theory-prescribed batch/step scaling and a separately frozen protocol.

## 6. Censoring below the primary range

At epsilon=0.02, NOG-ZO, ME-DOL-ZO, and DGFM retain 20/20 hits at all dimensions, while DGFM+ has 8/20 hits at d=25 and 0/20 at d=50,100,200. Those conditional DGFM+ values are censored and are excluded from dimension-slope fitting. Full descriptive hit rates and capped values are in dimension_summary.csv.

## 7. Paper-safe claim

> Under one frozen configuration, the primary thresholds remain fully attainable across d=25--200, and NOG-ZO retains lower communication depth than the three baselines. The relative dimension slopes do not recover the worst-case theoretical powers, so the study is reported only as fixed-configuration dimension sensitivity.

Do not state that these results verify exact dimension exponents or that parameter selection is dimension-wise optimal.

## 8. Reproduction

~~~bash
conda run -n NOG python -m src.distributed.zo_dimension_analysis
~~~

Raw trajectories remain under outputs/distributed_zo and are not modified by the analysis.
