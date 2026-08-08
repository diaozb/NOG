# Step ZO-5C: theory comparison and claim boundary

## 1. What this step evaluates

Step ZO-5C interprets the audited 20-seed formal experiment against Theorem parallel-ZO, Theorem dist-ZO, and Table res-dist in the paper source. It does not rerun trajectories or tune parameters.

The input audit passed 80/80 frozen method-seed tasks. The primary estimand is the same-seed baseline/NOG-ZO first-hit depth ratio.

## 2. Theoretical ratio directions

After fixing dimension, Goldstein radius, smoothness constants, and initial optimality gap, the leading epsilon powers imply:

| baseline/NOG-ZO | predicted depth-ratio power | predicted work-ratio power |
|---|---:|---:|
| ME-DOL-ZO | 4/3 | 0 |
| DGFM | 7/3 | 1 |
| DGFM+ | 4/3 | 0 |

A positive depth-ratio power predicts a growing relative depth advantage for NOG-ZO as epsilon decreases. A zero work-ratio power means only equal asymptotic order; it does not require a ratio of one and does not remove finite constants.

## 3. Formal observations on complete-pair intervals

Slope confidence intervals use 2,000 joint bootstrap resamples of the 20 formal seeds. Each resample preserves dependence across epsilon thresholds because all thresholds reuse the same trajectory.

| baseline | complete epsilon interval | depth ratio endpoints | observed depth slope (95% CI) | theory | work slope (95% CI) | theory |
|---|---|---:|---:|---:|---:|---:|
| ME-DOL-ZO | 0.200 to 0.018 | 1.60 to 3.55 | 0.319 [0.308, 0.331] | 1.333 | 0.319 [0.307, 0.330] | 0.000 |
| DGFM | 0.200 to 0.018 | 3.56 to 8.79 | 0.363 [0.357, 0.369] | 2.333 | 0.363 [0.357, 0.369] | 1.000 |
| DGFM+ | 0.200 to 0.030 | 3.52 to 11.39 | 0.529 [0.496, 0.561] | 1.333 | 0.527 [0.492, 0.559] | 0.000 |

All three observed depth slopes are positive, their bootstrap intervals exclude zero, and their mean curves have Spearman coefficient one against inverse epsilon. This is direct finite-instance evidence for the predicted direction of the relative communication-depth advantage.

The observed depth-slope intervals do not contain the worst-case theoretical ratio exponents. This does not contradict a Big-O upper bound, but it means the experiment does not verify the exact 5/3, 3, 4, 4/3, or 7/3 powers.

The work result is weaker. ME-DOL-ZO/NOG-ZO and DGFM+/NOG-ZO increase over their complete intervals even though their worst-case work orders have the same epsilon power. DGFM/NOG-ZO increases in the predicted direction but more slowly than the reference power one. Consequently, work should be reported as a secondary finite-budget tradeoff, not as validation of exact work scaling.

## 4. Censoring boundary

- ME-DOL-ZO/NOG-ZO and DGFM/NOG-ZO have 20/20 paired hits down to epsilon 0.018. NOG-ZO then has 18/20 hits at 0.016, 11/20 at 0.015, 1/20 at 0.014, and 0/20 at 0.013.
- DGFM+/NOG-ZO has 20/20 paired hits down to epsilon 0.030; DGFM+ has no confirmed hits at 0.025 or below.
- Ratios below these complete-pair boundaries are right-censored and are excluded from slope fitting. Conditional ratios there are survivor-biased.

## 5. Why exact exponent verification is out of scope

1. The theorem prescribes epsilon-dependent quantities, including the NOG-ZO oracle variance and batch size. The formal experiment instead freezes one pilot-selected configuration and reads all thresholds from one anytime trajectory.
2. Worst-case Big-O upper bounds need not be tight on one finite synthetic objective or over a one-decade accuracy range.
3. The measured endpoint is a fixed-bank Monte Carlo smoothed-gradient proxy, not the exact Goldstein subdifferential distance.
4. The experiment simulates dependency depth on a complete graph; it does not measure multi-machine latency.
5. The low-epsilon region is budget-censored.

## 6. Paper-safe conclusion

> Across the complete paired-hit regime, the communication-depth ratios of ME-DOL-ZO, DGFM, and DGFM+ relative to NOG-ZO increase monotonically as the target accuracy becomes more stringent. This finite-instance trend is qualitatively consistent with the improved epsilon dependence of NOG-ZO communication complexity. We do not interpret the experiment as verification of the exact worst-case exponents: the configurations are frozen across thresholds, the stationarity measure is a Monte Carlo proxy, and the smaller-epsilon observations are right-censored by the fixed training-work budget.

## 7. Claims that should not be made

- Do not claim that the experiment proves Theorem parallel-ZO or Theorem dist-ZO.
- Do not claim empirical recovery of the exact theoretical powers.
- Do not report hit-only ratios below the complete-pair boundary as unbiased comparisons.
- Do not claim constant empirical work ratios for ME-DOL-ZO or DGFM+ under this formal configuration.
- Do not describe the frozen pilot configurations as globally optimal hyperparameters.

## 8. Reproduction

    conda run -n NOG python -m src.distributed.zo_theory_interpretation

Machine-readable values are stored in theory_comparison.csv and theory_comparison.json. The underlying first-hit and ratio tables remain formal_per_seed.csv and formal_ratios.csv.
