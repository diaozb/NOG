"""Audit all new and reused wide-epsilon worker robustness trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_formal_analysis import _task_audit
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, object_sha256, utc_now


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    unhashed = dict(manifest)
    expected = unhashed.pop("manifest_sha256", None)
    if expected != object_sha256(unhashed):
        raise ValueError("Robustness manifest hash mismatch.")


def audit_robustness(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    manifest = _load(root / "robustness_manifest.json")
    _validate_manifest(manifest)
    formal_completion = _load(root / "formal_completion.json")
    formal_records = {row["task_id"]: row for row in formal_completion["records"]}
    stage_records: Dict[int, Dict[str, Dict[str, Any]]] = {}
    errors: List[str] = []
    for worker in (1, 2, 4):
        completion = _load(root / f"robustness_m{worker}_completion.json")
        if completion.get("status") != "complete" or completion.get("completed_tasks") != 60 or completion.get("failed_tasks") != 0:
            errors.append(f"m={worker} completion is invalid")
        if completion.get("robustness_manifest_sha256") != manifest["manifest_sha256"]:
            errors.append(f"m={worker} completion/manifest hash mismatch")
        stage_records[worker] = {row["task_id"]: row for row in completion["records"]}

    audit_rows = []
    observed_keys = set()
    trajectory_rows = 0
    reused_count = 0
    new_count = 0
    for entry in manifest["tasks"]:
        worker = int(entry["worker_count"])
        method = entry["method"]
        region = entry["region"]
        task_id = entry["task_id"]
        if entry["source"] == "reuse-formal":
            reused_count += 1
            record = formal_records[entry["source_task_id"]]
            group_id = f"{method}__{region}"
            group_root = root / "formal" / group_id
            prefix = f"formal/{group_id}/"
            if file_sha256(root / entry["source_partial_path"]) != entry["source_partial_sha256"]:
                errors.append(f"reused partial changed: {task_id}")
            if file_sha256(root / entry["source_manifest_path"]) != entry["source_manifest_sha256"]:
                errors.append(f"reused manifest changed: {task_id}")
        else:
            new_count += 1
            record = stage_records[worker].get(entry["task_id"])
            if record is None:
                errors.append(f"missing robustness task: {task_id}")
                continue
            group_id = f"{method}__{region}__m{worker}"
            group_root = root / "robustness" / f"m{worker}" / group_id
            prefix = f"robustness/m{worker}/{group_id}/"
        completion_record = {
            "task": {
                "method": entry["method"],
                "formal_seed": entry["formal_seed"],
                "worker_count": worker,
            },
            "task_key": record["task_key"],
            "partial_path": record["partial_path"].removeprefix(prefix),
            "manifest_path": record["manifest_path"].removeprefix(prefix),
            "row_count": record["row_count"],
        }
        formal_config = {
            "formal_config_id": entry["task_id"],
            "method": entry["method"],
            "parameters": entry["parameters"],
            "rounds": entry["rounds"],
        }
        _, audit_row, rows = _task_audit(cfg, formal_config, group_root, completion_record)
        audit_row.update(
            {
                "task_id": entry["task_id"],
                "region": entry["region"],
                "epsilon": entry["epsilon"],
                "source": entry["source"],
            }
        )
        audit_rows.append(audit_row)
        trajectory_rows += len(rows)
        observed_keys.add(
            (
                entry["method"],
                float(entry["epsilon"]),
                worker,
                int(entry["formal_seed"]),
            )
        )

    if new_count != 180 or reused_count != 60:
        errors.append("new/reused task counts are invalid")
    if len(audit_rows) != 240 or any(not row["passed"] for row in audit_rows):
        errors.append("one or more robustness task audits failed")
    if len(observed_keys) != 240:
        errors.append("robustness method/epsilon/worker/seed coverage mismatch")
    report = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "robustness_manifest_sha256": manifest["manifest_sha256"],
        "expected_tasks": 240,
        "audited_tasks": len(audit_rows),
        "passed_tasks": sum(bool(row["passed"]) for row in audit_rows),
        "failed_tasks": sum(not bool(row["passed"]) for row in audit_rows),
        "new_tasks": new_count,
        "reused_formal_tasks": reused_count,
        "trajectory_rows": trajectory_rows,
        "logical_method_epsilon_worker_seed_rows": len(observed_keys),
        "workers": manifest["workers"],
        "epsilons": manifest["epsilons"],
        "seeds": manifest["seeds"],
        "global_errors": errors,
        "historical_failed_attempts": len(list((root / "robustness").glob("m*/*/failures/*.json"))),
    }
    atomic_write_csv(root / "robustness_work_accounting_audit.csv", audit_rows)
    atomic_write_json(root / "robustness_result_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    report = audit_robustness(cfg, root)
    status = report["status"]
    passed = report["passed_tasks"]
    expected = report["expected_tasks"]
    trajectory_rows = report["trajectory_rows"]
    print(f"status={status} tasks={passed}/{expected} trajectory_rows={trajectory_rows}")
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
