"""Audit hashes, identities, trajectories, and work accounting for formal runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def audit(freeze_path: Path, formal_root: Path) -> Dict[str, Any]:
    freeze = _load(freeze_path)
    completion = _load(formal_root / "completion.json")
    seeds = {int(value) for value in freeze["formal_seeds"]}
    expected = (len(freeze["selected_batches"]) + 1) * len(seeds)
    errors = []
    task_rows = []
    if completion.get("status") != "complete":
        errors.append("formal completion status is not complete")
    if int(completion.get("completed_tasks", -1)) != expected:
        errors.append("formal completion task count does not match frozen schedule")
    if int(completion.get("failed_tasks", -1)) != 0:
        errors.append("formal completion contains failures")

    identities = set()
    for record in completion.get("records", []):
        label = str(record["label"])
        task = record["task"]
        seed = int(task["formal_seed"])
        method = str(task["method"])
        identity = (label, seed)
        local_errors = []
        if identity in identities:
            local_errors.append("duplicate label/seed identity")
        identities.add(identity)
        if seed not in seeds or int(task["worker_count"]) != 8:
            local_errors.append("task seed/worker does not match freeze")
        partial_path = Path(record["partial_path"])
        manifest_path = Path(record["manifest_path"])
        payload = _load(partial_path)
        manifest = _load(manifest_path)
        if manifest.get("status") != "complete":
            local_errors.append("task manifest is not complete")
        if manifest.get("partial_sha256") != file_sha256(partial_path):
            local_errors.append("partial SHA256 mismatch")
        if payload.get("config_sha256") != manifest.get("config_sha256"):
            local_errors.append("config SHA256 mismatch")
        if payload.get("task_fingerprint") != manifest.get("task_fingerprint"):
            local_errors.append("task fingerprint mismatch")
        if payload.get("method") != method or int(payload.get("formal_seed", -1)) != seed:
            local_errors.append("payload task identity mismatch")
        rank_metadata = payload.get("rank_metadata", [])
        if len(rank_metadata) != 8 or {int(row["rank"]) for row in rank_metadata} != set(range(8)):
            local_errors.append("rank metadata is not exactly ranks 0..7")
        child_pids = payload.get("launch", {}).get("child_pids", [])
        if len(child_pids) != 8 or len(set(child_pids)) != 8:
            local_errors.append("launch does not contain eight unique child PIDs")

        rows = payload.get("rows", [])
        if not rows:
            local_errors.append("payload has no trajectory rows")
        iterations = [int(row["iteration"]) for row in rows]
        depths = [int(row["depth"]) for row in rows]
        works = [int(row["total_work"]) for row in rows]
        if iterations != sorted(set(iterations)):
            local_errors.append("iterations are not unique and increasing")
        if any(right <= left for left, right in zip(depths, depths[1:])):
            local_errors.append("depth is not strictly increasing")
        if any(right <= left for left, right in zip(works, works[1:])):
            local_errors.append("total work is not strictly increasing")
        if method == "NOG-FO":
            batch = int(label.rsplit("-", 1)[1])
            expected_work = [depth * batch for depth in depths]
        elif method == "ME-DOL-FO":
            batch = 8
            expected_work = [depth * 8 for depth in depths]
        else:
            batch = -1
            expected_work = []
            local_errors.append("unexpected method")
        if works != expected_work:
            local_errors.append("total work formula mismatch")
        if any(int(row["per_worker_work_max"]) * 8 != int(row["total_work"]) for row in rows):
            local_errors.append("per-worker work does not sum to total work")
        if any(
            not math.isfinite(float(row["stat_proxy"])) or float(row["stat_proxy"]) < 0.0
            for row in rows
        ):
            local_errors.append("invalid stat_proxy")
        errors.extend(f"{label}/seed{seed}: {message}" for message in local_errors)
        task_rows.append(
            {
                "label": label,
                "method": method,
                "formal_seed": seed,
                "data_B_total": batch,
                "row_count": len(rows),
                "final_depth": depths[-1] if depths else None,
                "final_total_work": works[-1] if works else None,
                "passed": not local_errors,
            }
        )

    if len(identities) != expected:
        errors.append("unique task identity count does not match expectation")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "expected_tasks": expected,
        "audited_tasks": len(task_rows),
        "passed_tasks": sum(bool(row["passed"]) for row in task_rows),
        "failed_tasks": sum(not bool(row["passed"]) for row in task_rows),
        "checks": [
            "completion count and seed/config identity",
            "partial SHA256 and task/config fingerprints",
            "rank metadata and unique child processes",
            "strictly increasing iteration/depth/work",
            "exact total and per-worker SFO work accounting",
            "finite nonnegative stationarity proxy",
        ],
        "errors": errors,
        "tasks": sorted(task_rows, key=lambda row: (row["label"], row["formal_seed"])),
    }
    atomic_write_json(formal_root / "formal_result_audit.json", report)
    if errors:
        raise ValueError(f"Formal result audit failed with {len(errors)} error(s).")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/frozen_parameters.json",
    )
    parser.add_argument(
        "--formal-root",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/formal",
    )
    args = parser.parse_args()
    result = audit(Path(args.freeze), Path(args.formal_root))
    print(
        f"status={result['status']} passed={result['passed_tasks']}/"
        f"{result['expected_tasks']}"
    )


if __name__ == "__main__":
    main()
