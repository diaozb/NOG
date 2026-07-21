# Step 8D Runtime Figure Notes

## `runtime_vs_workers`

Each point is the median of three measured real-CPU-process repeats; error bars span the observed `[min, max]` without outlier removal. Training time excludes process startup, evaluation, and serialization, while end-to-end time includes them. Every curve uses the complete pilot-frozen budget, so this is not first-hit time-to-epsilon. The ME-DOL configuration is identical across the three epsilon labels and was physically measured once per worker/repeat before deterministic expansion.

## `nog_strong_scaling_speedup`

Speedup is `median training time at m=1 / median training time at m`. Only NOG is plotted because its total SFO work is fixed across worker counts. Values below one mean that the single-machine CPU/Gloo process and communication overhead exceeds the available local parallelism benefit; the plot is a process-level scaling diagnostic, not evidence about a multi-node cluster.

## `communication_fraction_vs_workers`

Communication fraction is measured communication time divided by training time. It excludes startup, evaluation, and serialization and therefore must not be interpreted as a fraction of end-to-end time.

## `full_budget_method_comparison`

A runtime ratio below one means that NOG completed its frozen full budget faster in this implementation. The adjacent work-ratio panel is essential: the accuracy-selected configurations are not work matched, and NOG uses substantially more empirical SFO calls. Therefore the runtime ratio does not establish finite-work parity or directly validate the asymptotic Work complexity in Section 5. The epsilon=0.008 ME-DOL accuracy result remains right-censored, so neither panel is an unconditional time-to-epsilon comparison.
