"""Audit every completed wide-epsilon formal trajectory and its reuse mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_formal_analysis import _task_audit
from src.distributed.cpu_fo_tasks import atomic_write_json, utc_now
from src.distributed.epsilon_scaling_formal import build_formal_schedule


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_results(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    schedule = build_formal_schedule(cfg, root)
    completion = _load(root / "formal_completion.json")
    errors: List[str] = []
    if completion.get("status") != "complete":
        errors.append("formal completion is not complete")
    if completion.get("schedule_sha256") != schedule.get("schedule_sha256"):
        errors.append("completion/schedule hash mismatch")
    if completion.get("completed_tasks") != 120 or completion.get("failed_tasks") != 0:
        errors.append("formal completion counts are invalid")
    records = {row["task_id"]: row for row in completion.get("records", [])}
    if set(records) != {row["task_id"] for row in schedule["entries"]}:
        errors.append("formal task coverage mismatch")

    audit_rows = []
    logical_keys = set()
    trajectory_rows = 0
    observed_seeds = set()
    for entry in schedule["entries"]:
        record = records.get(entry["task_id"])
        if record is None:
            continue
        group_root = root / "formal" / entry["group_id"]
        group_id = entry["group_id"]
        prefix = f"formal/{group_id}/"
        completion_record = {
            "task": {
                "method": entry["method"],
                "formal_seed": entry["formal_seed"],
                "worker_count": entry["worker_count"],
            },
            "task_key": record["task_key"],
            "partial_path": record["partial_path"].removeprefix(prefix),
            "manifest_path": record["manifest_path"].removeprefix(prefix),
            "row_count": record["row_count"],
        }
        formal_config = {
            "formal_config_id": entry["group_id"],
            "method": entry["method"],
            "parameters": entry["parameters"],
            "rounds": entry["rounds"],
        }
        _, audit_row, rows = _task_audit(
            cfg, formal_config, group_root, completion_record
        )
        audit_row["region"] = entry["region"]
        audit_row["task_id"] = entry["task_id"]
        audit_rows.append(audit_row)
        trajectory_rows += len(rows)
        observed_seeds.add(int(entry["formal_seed"]))
        for epsilon in entry["epsilons"]:
            logical_keys.add((entry["method"], float(epsilon), int(entry["formal_seed"])))

    if len(audit_rows) != 120 or any(not row["passed"] for row in audit_rows):
        errors.append("one or more task-level audits failed")
    if observed_seeds != set(range(20)):
        errors.append("formal seed coverage mismatch")
    if observed_seeds & set(cfg["epsilon_scaling"]["pilot_seeds"]):
        errors.append("pilot seed leaked into formal results")
    if len(logical_keys) != 680:
        errors.append("epsilon reuse mapping is not exactly 680 rows")

    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "schedule_sha256": schedule["schedule_sha256"],
        "expected_tasks": 120,
        "audited_tasks": len(audit_rows),
        "passed_tasks": sum(bool(row["passed"]) for row in audit_rows),
        "failed_tasks": sum(not bool(row["passed"]) for row in audit_rows),
        "trajectory_rows": trajectory_rows,
        "logical_method_epsilon_seed_rows": len(logical_keys),
        "formal_seeds": sorted(observed_seeds),
        "pilot_seeds_excluded": cfg["epsilon_scaling"]["pilot_seeds"],
        "global_errors": errors,
        "historical_failed_attempts": len(list((root / "formal").glob("*/failures/*.json"))),
    }
    atomic_write_csv(root / "formal_work_accounting_audit.csv", audit_rows)
    atomic_write_json(root / "formal_result_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    report = audit_results(cfg, root)
    status = report["status"]
    passed = report["passed_tasks"]
    expected = report["expected_tasks"]
    logical = report["logical_method_epsilon_seed_rows"]
    print(f"status={status} tasks={passed}/{expected} logical_rows={logical}")
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
