# ZO experiments: pilot selection and formal results

> Status date: 2026-08-11. Pilot results below were used only for parameter
> selection. The independently seeded 20-seed formal results are reported
> separately and must not be mixed with the pilot estimates.

## 1. Current status

The fixed-work epsilon-scaling experiment, anomaly replication, and Steps
ZO-7A/7B/7C dimension sensitivity and ZO-8A/8B/8C logical-worker sensitivity
are complete. There is no ZO process running in the background. Real-process
and optional real-data checks remain future experiments.

| Stage | Status |
|---|---|
| ZO-2: implementation and theoretical accounting audit | Complete |
| ZO-3: scale and runtime calibration | Complete |
| ZO-4A: NOG-ZO dense pilot | Complete |
| ZO-4B: ME-DOL-ZO/DGFM/DGFM+ dense pilot | Complete (60/60) |
| ZO-4C: freeze four configurations and formal manifest | Complete |
| ZO-5A: 20-seed formal epsilon-scaling | Complete (80/80) |
| ZO-5B: audit, paired statistics, CIs and figures | Complete |
| ZO-5C: theory comparison and claim boundary | Complete |
| ZO-6A: anomaly and censoring audit | Complete |
| ZO-6B: frozen-config anomaly-seed replication | Complete (20/20) |
| ZO-6C: replication comparison and budget decision | Complete |
| ZO-7A: fixed-configuration dimension calibration | Complete (16/16) |
| ZO-7B: formal fixed-configuration dimension sensitivity | Complete (240/240) |
| ZO-7C: audited merge, ratios, CIs, figures, and report | Complete (320/320) |
| ZO-8A: logical-worker calibration | Complete (16/16) |
| ZO-8B: formal logical-worker sensitivity | Complete (240/240 new tasks) |
| ZO-8C: audited merge, CIs, figures, and report | Complete (320/320) |
| ZO-9: real-process and optional real-data checks | Not started |
| ZO-10: final report, paper figures, and packaging | Not started |

## 2. Formal result entry point

The complete Step ZO-5B report is [formal/README.md](formal/README.md). It
contains the frozen protocol, all epsilon points, same-seed baseline/NOG-ZO
ratios, bootstrap 95% confidence intervals, figures, and explicit censoring
boundaries. All three baseline/NOG-ZO mean depth ratios increase monotonically
as epsilon decreases on their complete 20-pair intervals. Below those intervals,
non-hits are reported as censored rather than as fictitious finite ratios.

![Formal paired depth and work ratios](formal/figures/formal_ratios.png)

The theory-aligned interpretation, joint-seed bootstrap slope intervals, and
paper-safe claim boundary are documented in
[formal/STEP_ZO_5C_THEORY_COMPARISON.md](formal/STEP_ZO_5C_THEORY_COMPARISON.md).

The trajectory-level rebound, censoring, and checkpoint-confirmation audit is
[formal/STEP_ZO_6A_ANOMALY_AUDIT.md](formal/STEP_ZO_6A_ANOMALY_AUDIT.md).

The reserved-seed replication reuses the frozen configurations on seeds
200--204 at the original work budget. Its isolated raw output directory is
`outputs/distributed_zo/zo_theory_validation/diagnostic/anomaly_seeds_fixed_work_983040/`;
these diagnostic observations will not be merged with the formal 20 seeds.

The completed formal-versus-anomaly comparison and the decision not to extend
the same frozen configurations for the current paper are documented in
[formal/STEP_ZO_6C_REPLICATION_DECISION.md](formal/STEP_ZO_6C_REPLICATION_DECISION.md).

The non-paper Step ZO-7A calibration completed at dimensions 25, 50, 100,
and 200 with calibration seed 300. It reuses the d=100 frozen candidates only
to measure hit coverage, stability, and runtime before a formal dimension
protocol is frozen. Raw outputs are isolated locally under
`outputs/distributed_zo/zo_theory_validation/dimension/calibration_fixed_params_work983040/`;
the compact protocol is [dimension_scaling_manifest.json](dimension_scaling_manifest.json).

The Step ZO-7B protocol is frozen in
[dimension_scaling_manifest.json](dimension_scaling_manifest.json). It runs
dimensions 25, 50, and 200 on 20 formal seeds and later reuses the audited
dimension-100 formal trajectories. The primary common thresholds are 0.05 and
0.03, and the fixed-configuration result is explicitly limited to qualitative
dimension sensitivity.

Step ZO-7C is complete. The full audit, absolute metrics, same-seed paired
ratios, joint-seed bootstrap intervals, figures, censoring table, and claim
boundary are in [dimension/README.md](dimension/README.md). All primary points
have 20/20 hits. NOG-ZO retains lower depth across d=25--200, but the relative
slopes do not recover the worst-case dimension powers; this result must remain
a fixed-configuration sensitivity statement.

![Dimension paired ratios](dimension/figures/dimension_ratios.png)

Step ZO-8 is complete. The frozen logical-worker protocol uses m=1,2,4,8,
20 formal seeds, and the same d=100 candidate parameters at every worker count.
The m=1,2,4 run contributes 240 new tasks and the hash-audited m=8 formal run
contributes 80 reused tasks. The complete report is
[worker/README.md](worker/README.md).

At epsilon=0.05, NOG-ZO has 20/20 hits at every worker count. Its mean
first-hit depth and total work remain essentially invariant from m=1 to m=8,
while mean per-worker work decreases from 219,340.8 to 27,443.2. The fitted
per-worker slope is -0.9997 with a 95% bootstrap interval
[-1.0011,-0.9984]. DGFM+/m=2 has 18/20 hits, so its conditional estimates are
explicitly censored. These are logical work-accounting results, not measured
multi-process speedups.

![Logical-worker accounting](worker/figures/worker_relative_to_m1.png)


## 3. Pilot protocol
The four methods are NOG-ZO, ME-DOL-ZO, DGFM, and DGFM+. They are evaluated
on

$$
F(x;\xi)=\max_{1\le r\le R}\sin(a_{\xi,r}^{\top}x+b_{\xi,r})
+\lambda\lVert x\rVert_1
$$

with $d=100$, $n=4096$, $R=1$, $\lambda=10^{-3}$, target Goldstein
radius $0.1$, eight logical workers, and a complete communication topology.
This is a single-process GPU simulation of the dependency graph, not a
multi-machine wall-clock experiment.

- Pilot seeds: 100--104 (five seeds).
- Reserved formal seeds: 0--19; these have not been used for selection.
- Maximum training budget per candidate: 983,040 two-point SZO calls.
- Two function values are counted for every two-point sample.
- Evaluation uses a method-independent fixed bank with 256 smoothing samples
  and 512 data samples; evaluation work is excluded from training work.
- The dense pilot evaluates every four underlying rounds, subject to each
  method's valid output/checkpoint structure.
- A threshold is a confirmed hit only when two consecutive checkpoints satisfy
  $\widehat{\operatorname{stat}}(x)\le\epsilon$.

All candidates receive the same maximum training-work budget. Their maximum
depths differ because batch sizes and communication dependency graphs differ.

## 4. Current best pilot candidates

“Lowest confirmed epsilon” below is the mean of the lowest confirmed threshold
reached by each pilot seed; smaller is better. It is a selection statistic,
not a formal accuracy guarantee.

| Method | Selected pilot parameters | Lowest confirmed epsilon | Minimum proxy | Final proxy | Maximum depth | Maximum work |
|---|---|---:|---:|---:|---:|---:|
| NOG-ZO | $M=2,\eta=0.01,B_{\mathrm{smooth}}=8$ | 0.01460 | 0.01387 | 0.02956 | 960 | 983,040 |
| ME-DOL-ZO | epoch length 12, multiplier 60 | 0.01380 | 0.01331 | 0.01458 | 3,840 | 983,040 |
| DGFM | $\eta=0.05$ | **0.01095** | **0.01067** | **0.01179** | 7,680 | 983,040 |
| DGFM+ | $\eta=0.05$ | 0.02900 | 0.02564 | 0.02805 | 7,224 | 983,040 |

![Preliminary ZO pilot snapshot](figures/zo_pilot_snapshot.png)

The depth and work curves are conditional on hit. They must be read together
with the hit-rate panel, especially below $\epsilon=0.016$.

## 5. Representative confirmed first-hit results

Each cell reports **hit count / 5; conditional mean depth; conditional mean
work**. A conditional mean excludes non-hitting seeds and is therefore not a
fair unconditional comparison when the hit count is below 5/5.

| $\epsilon$ | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---:|---:|---:|---:|---:|
| 0.0500 | 5/5; 215.2; 220,365 | 5/5; 492.0; 125,952 | 5/5; 1,300.8; 166,502 | 5/5; 1,332.8; 181,504 |
| 0.0300 | 5/5; 228.8; 234,291 | 5/5; 609.6; 156,058 | 5/5; 1,632.0; 208,896 | 5/5; 2,164.8; 294,912 |
| 0.0200 | 5/5; 236.8; 242,483 | 5/5; 780.0; 199,680 | 5/5; 1,963.2; 251,290 | 0/5; --; -- |
| 0.0180 | 5/5; 239.2; 244,941 | 5/5; 868.8; 222,413 | 5/5; 2,065.6; 264,397 | 0/5; --; -- |
| 0.0160 | 5/5; 241.6; 247,398 | 5/5; 1,034.4; 264,806 | 5/5; 2,284.8; 292,454 | 0/5; --; -- |
| 0.0150 | 4/5; 243.0; 248,832 | 5/5; 1,140.0; 291,840 | 5/5; 2,376.0; 304,128 | 0/5; --; -- |
| 0.0140 | 3/5; 244.0; 249,856 | 5/5; 2,008.8; 514,253 | 5/5; 2,576.0; 329,728 | 0/5; --; -- |
| 0.0130 | 0/5; --; -- | 1/5; 3,672.0; 940,032 | 5/5; 2,814.4; 360,243 | 0/5; --; -- |
| 0.0120 | 0/5; --; -- | 0/5; --; -- | 5/5; 3,230.4; 413,491 | 0/5; --; -- |
| 0.0110 | 0/5; --; -- | 0/5; --; -- | 3/5; 4,672.0; 598,016 | 0/5; --; -- |
| 0.0105 | 0/5; --; -- | 0/5; --; -- | 2/5; 5,996.0; 767,488 | 0/5; --; -- |

## 6. Preliminary interpretation

The most encouraging range is $\epsilon\in[0.016,0.02]$. At
$\epsilon=0.016$, all NOG-ZO seeds hit with mean depth 241.6, compared with
1,034.4 for ME-DOL-ZO and 2,284.8 for DGFM. NOG-ZO therefore uses about
4.3 times less depth than ME-DOL-ZO and 9.5 times less than DGFM, while its
mean work (247,398) remains comparable to ME-DOL-ZO (264,806) and DGFM
(292,454). This is qualitatively consistent with the intended depth/work
tradeoff.

There are also important limitations:

1. NOG-ZO drops from 5/5 hits at 0.016 to 4/5 at 0.015 and 3/5 at 0.014.
   Conditional depth below this boundary is survivor-biased.
2. NOG-ZO reaches a low proxy early but its final proxy rises to 0.02956.
   This indicates late-run instability or overshooting under the current fixed
   step size.
3. DGFM currently reaches the smallest thresholds most reliably, but with much
   larger depth.
4. DGFM+ is substantially weaker under the searched pilot configurations and
   does not reach 0.02 within the common work budget.
5. Because these five seeds were used for parameter selection, the table
   cannot be reported as an unbiased formal comparison.

Step ZO-4C froze these candidates before any formal trajectory was run. Formal
execution has now completed. Its results were not used to retune parameters;
any later anomaly rerun must retain the frozen configuration.

## 7. Files and reproduction

- Combined pilot snapshot: [pilot_snapshot.csv](pilot_snapshot.csv)
- Figure-generation script:
  [generate_pilot_snapshot.py](generate_pilot_snapshot.py)
- Full experiment plan: [ZO_plan.md](../ZO_plan.md)
- Frozen formal parameters:
  [frozen_parameters.json](frozen_parameters.json)
- Completed formal audit manifest: [formal/analysis_manifest.json](formal/analysis_manifest.json)
- Formal Step ZO-5B report:
  [formal/README.md](formal/README.md)
- Dimension Step ZO-7C report:
  [dimension/README.md](dimension/README.md)

Regenerate the current snapshot from the repository root:
- Worker Step ZO-8C report:
  [worker/README.md](worker/README.md)

~~~bash
conda run -n NOG python zo_experiments/generate_pilot_snapshot.py
~~~

The two raw results CSV files remain under their respective output directories
and are intentionally not duplicated here.
