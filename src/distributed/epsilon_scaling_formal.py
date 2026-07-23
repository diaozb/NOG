"""Prepare and run the frozen wide-epsilon formal trajectories."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import load_config, process_config_from_experiment
from src.distributed.cpu_fo_pilot import candidate_config
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    object_sha256,
    run_or_resume_task,
    utc_now,
)
from src.distributed.epsilon_scaling import validate_scaling_protocol


FORMAL_RUN_SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_hash(payload: Dict[str, Any], field: str) -> None:
    unhashed = dict(payload)
    expected = unhashed.pop(field, None)
    if expected != object_sha256(unhashed):
        raise ValueError(f"{field} mismatch.")


def build_formal_schedule(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Validate frozen inputs and build a deterministic zero-launch schedule."""
    validate_scaling_protocol(cfg)
    freeze = _load(root / "frozen_region_configs.json")
    manifest = _load(root / "formal_manifest.json")
    _validate_hash(freeze, "freeze_sha256")
    _validate_hash(manifest, "manifest_sha256")
    if freeze.get("status") != "frozen" or manifest.get("status") != "prepared":
        raise ValueError("Formal inputs are not frozen/prepared.")
    if manifest.get("freeze_sha256") != freeze.get("freeze_sha256"):
        raise ValueError("Formal manifest freeze reference mismatch.")
    if manifest.get("launches_started") != 0:
        raise ValueError("Prepared formal manifest unexpectedly records launches.")

    groups = {row["group_id"]: row for row in manifest["trajectory_groups"]}
    entries = []
    for task in manifest["tasks"]:
        group = groups[task["group_id"]]
        frozen = freeze["regions"][task["region"]][task["method"]]
        for field in ("method", "region", "rounds", "parameters", "epsilons", "worker_count"):
            if task[field] != group[field]:
                task_id = task["task_id"]
                raise ValueError(f"Task/group mismatch for {task_id}: {field}.")
        if task["parameters"] != frozen["parameters"] or task["rounds"] != frozen["rounds"]:
            task_id = task["task_id"]
            raise ValueError(f"Task is not frozen: {task_id}.")
        entries.append(
            {
                "task_id": task["task_id"],
                "group_id": task["group_id"],
                "method": task["method"],
                "region": task["region"],
                "formal_seed": int(task["formal_seed"]),
                "worker_count": int(task["worker_count"]),
                "rounds": int(task["rounds"]),
                "epsilons": [float(value) for value in task["epsilons"]],
                "parameters": dict(task["parameters"]),
            }
        )
    if len(entries) != 120 or len({row["task_id"] for row in entries}) != 120:
        raise ValueError("Formal schedule must contain 120 unique tasks.")
    workers_per_task = {row["worker_count"] for row in entries}
    if len(workers_per_task) != 1:
        raise ValueError("Formal tasks must use one fixed worker count.")
    worker_count = next(iter(workers_per_task))
    max_workers = int(cfg["cpu_process"]["max_total_worker_processes"])
    concurrency = max_workers // worker_count
    if concurrency < 1 or concurrency * worker_count > max_workers:
        raise ValueError("Invalid formal concurrency.")
    schedule = {
        "schema_version": FORMAL_RUN_SCHEMA_VERSION,
        "status": "prepared",
        # A regenerated schedule must retain the same identity across resumes.
        "created_at_utc": manifest["created_at_utc"],
        "launches_started": 0,
        "formal_manifest_sha256": manifest["manifest_sha256"],
        "freeze_sha256": freeze["freeze_sha256"],
        "task_count": len(entries),
        "max_concurrent_tasks": concurrency,
        "max_total_worker_processes": max_workers,
        "entries": entries,
    }
    schedule["schedule_sha256"] = object_sha256(schedule)
    return schedule


def _task_config(cfg: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    candidate = {
        "method": entry["method"],
        "parameters": entry["parameters"],
    }
    return candidate_config(cfg, candidate, int(entry["rounds"]))


def _run_entry(cfg: Dict[str, Any], root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    current = _task_config(cfg, entry)
    task = CpuFoTask(entry["method"], entry["formal_seed"], entry["worker_count"])
    group_root = root / "formal" / entry["group_id"]
    result = run_or_resume_task(
        current,
        task,
        group_root,
        process_config_from_experiment(current),
    )
    return {
        "task_id": entry["task_id"],
        "group_id": entry["group_id"],
        "method": entry["method"],
        "region": entry["region"],
        "formal_seed": entry["formal_seed"],
        "status": result.status,
        "task_key": result.task_key,
        "partial_path": str(result.partial_path.relative_to(root)),
        "manifest_path": str(result.manifest_path.relative_to(root)),
        "row_count": result.row_count,
    }


def _progress(schedule: Dict[str, Any], records: List[Dict[str, Any]], expected: int) -> Dict[str, Any]:
    completed = sum(row["status"] in {"completed", "resumed", "recovered"} for row in records)
    failed = sum(row["status"] == "failed" for row in records)
    return {
        "schema_version": FORMAL_RUN_SCHEMA_VERSION,
        "status": "complete" if completed == expected and failed == 0 else "running",
        "updated_at_utc": utc_now(),
        "schedule_sha256": schedule["schedule_sha256"],
        "expected_tasks": expected,
        "attempted_tasks": len(records),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "records": sorted(records, key=lambda row: row["task_id"]),
    }


def run_formal(
    cfg: Dict[str, Any],
    root: Path,
    schedule: Dict[str, Any],
    max_tasks: int | None = None,
) -> Dict[str, Any]:
    entries = schedule["entries"][:max_tasks] if max_tasks is not None else schedule["entries"]
    progress_name = "formal_smoke_progress.json" if max_tasks is not None else "formal_progress.json"
    completion_name = "formal_smoke_completion.json" if max_tasks is not None else "formal_completion.json"
    records: List[Dict[str, Any]] = []
    concurrency = int(schedule["max_concurrent_tasks"])
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="formal") as pool:
        futures: Dict[Future[Dict[str, Any]], Dict[str, Any]] = {
            pool.submit(_run_entry, cfg, root, entry): entry for entry in entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                record = future.result()
            except BaseException as error:
                record = {
                    "task_id": entry["task_id"],
                    "group_id": entry["group_id"],
                    "method": entry["method"],
                    "region": entry["region"],
                    "formal_seed": entry["formal_seed"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            records.append(record)
            progress = _progress(schedule, records, len(entries))
            atomic_write_json(root / progress_name, progress)
            task_id = record["task_id"]
            task_status = record["status"]
            completed_tasks = progress["completed_tasks"]
            failed_tasks = progress["failed_tasks"]
            print(
                f"formal={task_id} status={task_status} "
                f"progress={completed_tasks}/{len(entries)} failures={failed_tasks}",
                flush=True,
            )
    completion = _progress(schedule, records, len(entries))
    completion["status"] = "complete" if completion["completed_tasks"] == len(entries) and completion["failed_tasks"] == 0 else "incomplete"
    atomic_write_json(root / completion_name, completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    parser.add_argument("--phase", choices=["prepare", "run"], default="prepare")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    schedule = build_formal_schedule(cfg, root)
    atomic_write_json(root / "formal_schedule.json", schedule)
    if args.phase == "prepare":
        task_count = schedule["task_count"]
        concurrency = schedule["max_concurrent_tasks"]
        print(f"status=prepared tasks={task_count} concurrency={concurrency} launches_started=0")
        return
    completion = run_formal(cfg, root, schedule, args.max_tasks)
    completion_status = completion["status"]
    completed_tasks = completion["completed_tasks"]
    expected_tasks = completion["expected_tasks"]
    failed_tasks = completion["failed_tasks"]
    print(
        f"status={completion_status} tasks={completed_tasks}/"
        f"{expected_tasks} failures={failed_tasks}"
    )
    if completion["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
