# Reproduce the wide-epsilon theory validation

Run from the repository root with the `NOG` Conda environment.  Pilot seeds
100--104 select a monotone matched-work batch schedule; formal seeds 0--19 are
never used for selection.  The scheduler admits at most four eight-worker
tasks (32 worker processes) at once, and every command resumes valid partials.

```bash
conda run -n NOG python -m src.distributed.theory_validation_runner pilot-batch-grid
conda run -n NOG python -m src.distributed.theory_validation_freeze
conda run -n NOG python -m src.distributed.theory_validation_runner formal
conda run -n NOG python -m src.distributed.theory_validation_audit
conda run -n NOG python -m src.distributed.theory_validation_analysis
conda run -n NOG python -m src.distributed.theory_validation_report
conda run -n NOG python -m src.distributed.theory_validation_package
```

Raw process trajectories are intentionally excluded from Git because they are
large.  `frozen_parameters.json` and `analysis/analysis_manifest.json` retain
SHA256 hashes for every pilot/formal raw input used to make the package.
