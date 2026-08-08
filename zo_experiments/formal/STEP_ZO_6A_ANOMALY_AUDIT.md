# Step ZO-6A: anomaly and censoring audit

This audit reuses the frozen Step ZO-5A trajectories. It does not change parameters, discard seeds, or launch a new experiment.

Input integrity: 80/80 tasks passed the Step ZO-5B audit.

## 1. Diagnostic definitions

- Raw floor: the minimum proxy at any single checkpoint.
- Confirmed floor: the minimum, over adjacent checkpoint pairs, of the larger proxy in the pair. This is the smallest epsilon that can satisfy the preregistered two-consecutive-checkpoint rule.
- Early rebound: the confirmed floor occurs before half of the work budget and the final proxy exceeds it by more than 50 percent.
- Late floor: the confirmed floor occurs in the final 20 percent of the budget.
- Extension outlook is a trajectory diagnostic, not a guarantee of what an extended run would do.

## 2. Method-level diagnosis

| method | confirmed floor mean [min, max] | median floor position | final/floor | early rebound | late floor | tail improving | extension outlook |
|---|---:|---:|---:|---:|---:|---:|---|
| NOG-ZO | 0.01493 [0.01322, 0.01638] | 25.4% | 1.97 | 20/20 | 0/20 | 8/20 | extension_unlikely_without_stability_change |
| ME-DOL-ZO | 0.01271 [0.01197, 0.01335] | 73.0% | 1.12 | 0/20 | 9/20 | 11/20 | extension_plausible_but_unproven |
| DGFM | 0.01051 [0.01005, 0.01092] | 79.4% | 1.11 | 0/20 | 10/20 | 8/20 | extension_plausible_but_unproven |
| DGFM+ | 0.02676 [0.02470, 0.02927] | 76.7% | 1.08 | 0/20 | 10/20 | 10/20 | extension_plausible_but_unproven |

NOG-ZO is the clearest anomaly: its confirmed floor occurs near one quarter of the work budget for every seed, after which the proxy rebounds substantially. Merely appending more iterations to the same fixed configuration is therefore not supported as a likely cure by the existing trajectories.

The three baselines typically attain their floors later and end much closer to those floors. A larger budget may help some late-floor seeds, but this remains unproven and should be tested on reserved anomaly seeds before altering any formal claim.

## 3. Checkpoint-confirmation sensitivity

Each cell is one-checkpoint hits / confirmed two-checkpoint hits out of 20. A gap identifies transient crossings suppressed by the confirmation rule.

| epsilon | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |
|---:|---:|---:|---:|---:|
| 0.03000 | 20/20 | 20/20 | 20/20 | 20/20 |
| 0.02900 | 20/20 | 20/20 | 20/20 | 19/19 |
| 0.02800 | 20/20 | 20/20 | 20/20 | 17/17 |
| 0.02700 | 20/20 | 20/20 | 20/20 | 12/12 |
| 0.02600 | 20/20 | 20/20 | 20/20 | 6/6 |
| 0.02500 | 20/20 | 20/20 | 20/20 | 1/1 |
| 0.02000 | 20/20 | 20/20 | 20/20 | 0/0 |
| 0.01900 | 20/20 | 20/20 | 20/20 | 0/0 |
| 0.01800 | 20/20 | 20/20 | 20/20 | 0/0 |
| 0.01700 | 20/20 | 20/20 | 20/20 | 0/0 |
| 0.01650 | 20/20 | 20/20 | 20/20 | 0/0 |
| 0.01600 | 20/18 | 20/20 | 20/20 | 0/0 |
| 0.01550 | 18/16 | 20/20 | 20/20 | 0/0 |
| 0.01500 | 17/11 | 20/20 | 20/20 | 0/0 |
| 0.01450 | 12/6 | 20/20 | 20/20 | 0/0 |
| 0.01400 | 7/1 | 20/20 | 20/20 | 0/0 |
| 0.01350 | 2/1 | 20/20 | 20/20 | 0/0 |
| 0.01300 | 0/0 | 15/12 | 20/20 | 0/0 |
| 0.01200 | 0/0 | 3/1 | 20/20 | 0/0 |
| 0.01100 | 0/0 | 0/0 | 20/20 | 0/0 |
| 0.01050 | 0/0 | 0/0 | 12/10 | 0/0 |
| 0.01000 | 0/0 | 0/0 | 2/0 | 0/0 |

The confirmation rule removes isolated noisy crossings, but it does not create the main low-epsilon failure: the confirmed-floor distribution and the late trajectory behavior remain the primary constraints.

## 4. Figures

![Proxy trajectories](figures/anomaly_proxy_trajectories.png)

![Checkpoint sensitivity](figures/boundary_checkpoint_sensitivity.png)

## 5. Step ZO-6B recommendation

1. Keep the frozen formal results unchanged.
2. Run the same frozen configurations on reserved anomaly seeds 200 through 204 at the current budget to test whether the rebound and floor positions replicate.
3. For NOG-ZO, do not assume a simple budget extension will help; first confirm the early-rebound pattern on anomaly seeds.
4. For ME-DOL-ZO, DGFM, and DGFM+, an explicitly separate two-times-budget sensitivity run can be considered after replication.
5. Label every Step ZO-6B run diagnostic. It must not be merged with the 20-seed formal curves or used to retune them.

## 6. Reproduction

    conda run -n NOG python -m src.distributed.zo_anomaly_audit

Machine-readable outputs are anomaly_task_diagnostics.csv, boundary_checkpoint_sensitivity.csv, anomaly_method_summary.csv, and anomaly_audit.json.
