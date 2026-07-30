# Reproduce symmetric low-epsilon v5

Run from the repository root in the `NOG` Conda environment. Algorithm and
batch selection use pilot seeds 110--114. Formal seeds 20--39 and anomaly
confirmation seeds 40--49 are disjoint. All commands resume only artifacts
whose config fingerprint and SHA256 match.

```bash
conda run -n NOG python -m src.distributed.low_epsilon_runner pilot-algorithms
conda run -n NOG python -m src.distributed.low_epsilon_freeze freeze-algorithms
conda run -n NOG python -m src.distributed.low_epsilon_runner pilot-batches
conda run -n NOG python -m src.distributed.low_epsilon_freeze freeze-final
conda run -n NOG python -m src.distributed.low_epsilon_runner formal
conda run -n NOG python -m src.distributed.low_epsilon_audit
conda run -n NOG python -m src.distributed.low_epsilon_analysis
conda run -n NOG python -m src.distributed.low_epsilon_audit \
  --formal-root outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric/formal_extra \
  --extra
conda run -n NOG python -m src.distributed.low_epsilon_report
conda run -n NOG python -m src.distributed.low_epsilon_package
```

The analysis command automatically runs the already-frozen extra seeds when a
preregistered adjacent depth-ratio drop exceeds 20%. Raw trajectories are kept
locally because of size; the compact Git package contains audits, summaries,
figures, freeze manifests, and SHA256 manifests.
