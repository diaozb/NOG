# Step 8 paper-candidate figure captions

## `runtime_vs_workers.pdf`

**Single-host CPU-process runtime under pilot-frozen full budgets.** Training and end-to-end times are shown for NOG-FO and ME-DOL-FO with $m\in\{1,2,4,8,16,32\}$ worker processes. Each point is the median of three measured repeats, and error bars span the observed minimum and maximum without outlier removal. Training time excludes process startup, fixed-bank evaluation, and serialization, whereas end-to-end time includes these costs. All runs use complete pilot-frozen budgets rather than first-hit stopping, and the ME-DOL configuration is shared across the three displayed tolerance labels.

## `nog_strong_scaling_speedup.pdf`

**Fixed-total-work strong-scaling diagnostic for NOG-FO.** Speedup is the ratio between the median training time at one worker and at $m$ workers. NOG's total SFO work is fixed across worker counts. Values below one show that, on this single-host CPU/Gloo implementation, process and collective overhead exceeds the available local parallelism benefit. The dashed line denotes ideal linear scaling. This diagnostic should not be generalized to multi-node or GPU collectives.

## `communication_fraction_vs_workers.pdf`

**Measured communication fraction within training time.** The plotted fraction is communication time divided by training time for the complete frozen budget. It excludes startup, evaluation, and serialization, and therefore is not a fraction of end-to-end time. The increase with worker count is consistent with the observed absence of positive single-host process scaling.

## `full_budget_method_comparison.pdf`

**Runtime and SFO-work ratios under unmatched frozen budgets.** The left panel reports the NOG-FO/ME-DOL-FO median training-time ratio, where values below one indicate that NOG completed its frozen budget faster. The right panel reports the corresponding total training SFO-work ratio. Because the accuracy-selected configurations are not work matched, the timing ratio is an implementation-level full-budget comparison rather than a work-matched time-to-stationarity speedup. ME-DOL is right-censored at $\epsilon=0.008$ in the formal accuracy experiment.
