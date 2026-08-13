"""Package compact ZO process-equivalence evidence for version control."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs/distributed_cpu_zo/equivalence"
DEFAULT_OUTPUT = ROOT / "zo_experiments/process_equivalence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    audit_path = source / "equivalence_audit.json"
    summary_path = source / "equivalence_summary.csv"
    audit = json.loads(audit_path.read_text())
    summary = pd.read_csv(summary_path)
    if audit.get("status") != "passed" or audit.get("passed_tasks") != 6:
        raise ValueError("Expected a passed 6/6 process-equivalence audit.")
    if len(summary) != 6 or not summary["passed"].astype(bool).all():
        raise ValueError("Equivalence summary does not contain six passed tasks.")
    if set(summary["method"]) != {"NOG-ZO", "ME-DOL-ZO"}:
        raise ValueError("Unexpected methods in process-equivalence audit.")
    if set(summary["workers"].astype(int)) != {1, 2, 8}:
        raise ValueError("Expected 1/2/8-worker equivalence tasks.")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audit_path, output / audit_path.name)
    shutil.copy2(summary_path, output / summary_path.name)
    maximum = float(summary["max_abs_trajectory_difference"].max())
    readme = f"""# ZO real-process equivalence audit

This directory is the compact, version-controlled evidence for Step ZO-9B.
The audit compares the logical single-process simulator with independent Gloo
CPU processes at 1, 2, and 8 workers for NOG-ZO and ME-DOL-ZO.

- Passed tasks: **6/6**.
- Largest checkpoint-wise absolute trajectory difference: **{maximum:.3e}**.
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
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(f"passed=6/6 max_abs_difference={maximum:.3e} saved={output}")


if __name__ == "__main__":
    main()
