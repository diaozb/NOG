# Reproduce ZO optimism ablation

Use the absolute CPU interpreter; do not run `conda activate`:

```bash
cd /data/diaozb/NOG
/root/miniconda3/envs/NOG/bin/python -m py_compile \
  src/distributed/algorithms.py \
  src/distributed/run_distributed_baselines.py \
  src/distributed/zo_refine_pilot.py \
  zo_optimism_ablation_20260825/run_ablation.py \
  zo_optimism_ablation_20260825/analyze_ablation.py

# Independent pilot: 3 datasets x 2 methods x 5 seeds = 30 CPU tasks
/root/miniconda3/envs/NOG/bin/python \
  zo_optimism_ablation_20260825/run_ablation.py --stage pilot --concurrency 4

# Formal: 3 datasets x 2 methods x 20 seeds = 120 CPU tasks
/root/miniconda3/envs/NOG/bin/python \
  zo_optimism_ablation_20260825/run_ablation.py --stage formal --concurrency 4

# Tables and curves
/root/miniconda3/envs/NOG/bin/python \
  zo_optimism_ablation_20260825/analyze_ablation.py
```

The runner is resumable: an existing per-task CSV is loaded rather than rerun. The formal seeds are fixed to 720--739 and must not be changed after the run begins.
