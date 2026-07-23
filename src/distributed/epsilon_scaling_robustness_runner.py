"""Run hash-frozen worker-count robustness tasks in resource-bounded stages."""

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


RUNNER_SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    validate_scaling_protocol(cfg)
    manifest = _load(root / "robustness_manifest.json")
    unhashed = dict(manifest)
    expected = unhashed.pop("manifest_sha256", None)
    if expected != object_sha256(unhashed):
        raise ValueError("Robustness manifest hash mismatch.")
    if manifest.get("status") != "prepared" or manifest.get("launches_started") != 0:
        raise ValueError("Robustness manifest is not zero-launch prepared.")
    if len(manifest["tasks"]) != 240 or manifest["runnable_task_count"] != 180:
        raise ValueError("Robustness task counts are invalid.")
    return manifest


def stage_entries(
    manifest: Dict[str, Any], worker_count: int
) -> List[Dict[str, Any]]:
    entries = [
        row
        for row in manifest["tasks"]
        if row["source"] == "run" and int(row["worker_count"]) == int(worker_count)
    ]
    if worker_count not in {1, 2, 4} or len(entries) != 60:
        raise ValueError(f"Expected 60 runnable tasks for m={worker_count}.")
    if len({row["task_id"] for row in entries}) != len(entries):
        raise ValueError("Robustness stage task ids are not unique.")
    return entries


def _task_config(cfg: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    candidate = {"method": entry["method"], "parameters": entry["parameters"]}
    return candidate_config(cfg, candidate, int(entry["rounds"]))


def _run_entry(cfg: Dict[str, Any], root: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    current = _task_config(cfg, entry)
    task = CpuFoTask(entry["method"], entry["formal_seed"], entry["worker_count"])
    method = entry["method"]
    region = entry["region"]
    worker_count = entry["worker_count"]
    group_id = f"{method}__{region}__m{worker_count}"
    group_root = root / "robustness" / f"m{worker_count}" / group_id
    result = run_or_resume_task(
        current,
        task,
        group_root,
        process_config_from_experiment(current),
    )
    return {
        "task_id": entry["task_id"],
        "method": entry["method"],
        "region": entry["region"],
        "epsilon": entry["epsilon"],
        "formal_seed": entry["formal_seed"],
        "worker_count": entry["worker_count"],
        "status": result.status,
        "task_key": result.task_key,
        "partial_path": str(result.partial_path.relative_to(root)),
        "manifest_path": str(result.manifest_path.relative_to(root)),
        "row_count": result.row_count,
    }


def _status(records: List[Dict[str, Any]], expected: int, manifest_sha256: str) -> Dict[str, Any]:
    completed = sum(row["status"] in {"completed", "resumed", "recovered"} for row in records)
    failed = sum(row["status"] == "failed" for row in records)
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "complete" if completed == expected and failed == 0 else "running",
        "updated_at_utc": utc_now(),
        "robustness_manifest_sha256": manifest_sha256,
        "expected_tasks": expected,
        "attempted_tasks": len(records),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "records": sorted(records, key=lambda row: row["task_id"]),
    }


def run_stage(
    cfg: Dict[str, Any],
    root: Path,
    manifest: Dict[str, Any],
    worker_count: int,
    max_tasks: int | None = None,
) -> Dict[str, Any]:
    entries = stage_entries(manifest, worker_count)
    if max_tasks is not None:
        entries = entries[:max_tasks]
    concurrency = int(manifest["max_concurrent_tasks_by_worker"][str(worker_count)])
    if concurrency * worker_count > int(manifest["max_total_worker_processes"]):
        raise ValueError("Robustness stage exceeds worker-process limit.")
    prefix = f"robustness_smoke_m{worker_count}" if max_tasks is not None else f"robustness_m{worker_count}"
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=prefix) as pool:
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
                    "method": entry["method"],
                    "region": entry["region"],
                    "epsilon": entry["epsilon"],
                    "formal_seed": entry["formal_seed"],
                    "worker_count": entry["worker_count"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            records.append(record)
            progress = _status(records, len(entries), manifest["manifest_sha256"])
            atomic_write_json(root / f"{prefix}_progress.json", progress)
            task_id = record["task_id"]
            task_status = record["status"]
            completed = progress["completed_tasks"]
            failed = progress["failed_tasks"]
            print(
                f"robustness={task_id} status={task_status} "
                f"progress={completed}/{len(entries)} failures={failed}",
                flush=True,
            )
    completion = _status(records, len(entries), manifest["manifest_sha256"])
    if completion["completed_tasks"] != len(entries) or completion["failed_tasks"]:
        completion["status"] = "incomplete"
    atomic_write_json(root / f"{prefix}_completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    parser.add_argument("--phase", choices=["prepare", "run"], default="prepare")
    parser.add_argument("--worker", type=int, choices=[1, 2, 4], default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    manifest = validate_manifest(cfg, root)
    if args.phase == "prepare":
        runnable = manifest["runnable_task_count"]
        reused = manifest["reused_formal_task_count"]
        print(
            f"status=prepared runnable={runnable} "
            f"reused={reused} launches_started=0"
        )
        return
    if args.worker is None:
        raise ValueError("--worker is required for a run stage.")
    completion = run_stage(cfg, root, manifest, args.worker, args.max_tasks)
    status = completion["status"]
    completed = completion["completed_tasks"]
    expected = completion["expected_tasks"]
    failed = completion["failed_tasks"]
    print(f"status={status} tasks={completed}/{expected} failures={failed}")
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
