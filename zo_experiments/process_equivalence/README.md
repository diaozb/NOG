# ZO real-process equivalence audit

This directory is the compact, version-controlled evidence for Step ZO-9B.
The audit compares the logical single-process simulator with independent Gloo
CPU processes at 1, 2, and 8 workers for NOG-ZO and ME-DOL-ZO.

- Passed tasks: **6/6**.
- Largest checkpoint-wise absolute trajectory difference: **5.960e-08**.
- Every task passed trajectory, identity, rank-seed, independent-PID,
  one-thread-per-rank, exhaustive-shard, accounting, monotone-counter, and
  timing-sanity checks.

Claim boundary: this establishes numerical and oracle-work/communication-depth
accounting equivalence for NOG-ZO and ME-DOL-ZO. It is neither a cluster
speedup benchmark nor an implementation-equivalence claim for DGFM/DGFM+.

Reproduce with:

```bash
conda run -n NOG python -m src.distributed.cpu_zo_equivalence
conda run -n NOG python -m src.distributed.zo_audit_package
```
