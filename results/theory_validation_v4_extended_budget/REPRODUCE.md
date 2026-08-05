# Reproduce the v4 extended-budget continuation

Run from the repository root with the `NOG` Conda environment.  The pipeline
uses at most four tasks with eight workers each (32 worker processes), resumes
SHA/fingerprint-matching partials, audits every task, validates the original
v4 numerical prefix, and then creates the analysis tables and figures.

```bash
PYTHONPATH=. conda run -n NOG python \
  -m src.distributed.theory_validation_v4_extension_pipeline \
  --max-parallel-tasks 4
```

To run or resume only one stage:

```bash
PYTHONPATH=. conda run -n NOG python \
  scripts/run_v4_extended_budget_stage.py stage1 --max-parallel-tasks 4

PYTHONPATH=. conda run -n NOG python \
  scripts/run_v4_extended_budget_stage.py stage2 --max-parallel-tasks 4
```

Stage 1 originally used a 600-second process-launch management timeout.  Stage
2 uses 3,600 seconds because a complete task takes longer than ten minutes.
This timeout is excluded from the process fingerprint and does not alter the
algorithmic trajectory.  Exact per-stage configurations used for the packaged
results are retained as `nog_config_used.json` and `me_config_used.json`.

Raw trajectories remain under `outputs/distributed_cpu_fo_v4_extended_budget/`
and are excluded from Git due to size.  The analysis manifests record their
SHA256 hashes.
