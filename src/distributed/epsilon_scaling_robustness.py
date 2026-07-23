"""Prepare and audit the frozen worker-count robustness experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, object_sha256, utc_now
from src.distributed.epsilon_scaling import epsilon_region, validate_scaling_protocol


ROBUSTNESS_SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_robustness(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    validate_scaling_protocol(cfg)
    freeze = _load(root / "frozen_region_configs.json")
    formal = _load(root / "formal_completion.json")
    formal_audit = _load(root / "formal_result_audit.json")
    if freeze.get("status") != "frozen" or formal.get("status") != "complete":
        raise ValueError("Frozen configs and completed formal results are required.")
    if formal_audit.get("status") != "passed" or formal_audit.get("passed_tasks") != 120:
        raise ValueError("Formal results must pass all audits before reuse.")
    formal_records = {row["task_id"]: row for row in formal["records"]}
    robustness = cfg["epsilon_scaling"]["robustness"]
    reference_worker = int(cfg["epsilon_scaling"]["reference_worker"])
    tasks: List[Dict[str, Any]] = []
    for method in cfg["methods"]["sfo"]:
        for epsilon in robustness["epsilons"]:
            epsilon = float(epsilon)
            region = epsilon_region(epsilon)
            selected = freeze["regions"][region][method]
            for worker in robustness["workers"]:
                worker = int(worker)
                for seed in robustness["seeds"]:
                    seed = int(seed)
                    task_id = f"{method}__{region}__m{worker}__seed{seed}"
                    row = {
                        "task_id": task_id,
                        "method": method,
                        "epsilon": epsilon,
                        "region": region,
                        "worker_count": worker,
                        "formal_seed": seed,
                        "rounds": int(selected["rounds"]),
                        "parameters": dict(selected["parameters"]),
                        "source": "run",
                    }
                    if worker == reference_worker:
                        formal_id = f"{method}__{region}__seed{seed}"
                        source = formal_records[formal_id]
                        partial = root / source["partial_path"]
                        task_manifest = root / source["manifest_path"]
                        row.update(
                            {
                                "source": "reuse-formal",
                                "source_task_id": formal_id,
                                "source_partial_path": source["partial_path"],
                                "source_partial_sha256": file_sha256(partial),
                                "source_manifest_path": source["manifest_path"],
                                "source_manifest_sha256": file_sha256(task_manifest),
                            }
                        )
                    tasks.append(row)
    if len(tasks) != 240 or len({row["task_id"] for row in tasks}) != 240:
        raise ValueError("Robustness manifest must contain 240 unique settings.")
    runnable = sum(row["source"] == "run" for row in tasks)
    reused = sum(row["source"] == "reuse-formal" for row in tasks)
    max_processes = int(cfg["cpu_process"]["max_total_worker_processes"])
    manifest = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "launches_started": 0,
        "freeze_sha256": freeze["freeze_sha256"],
        "formal_schedule_sha256": formal_audit["schedule_sha256"],
        "workers": [int(value) for value in robustness["workers"]],
        "epsilons": [float(value) for value in robustness["epsilons"]],
        "seeds": [int(value) for value in robustness["seeds"]],
        "logical_task_count": len(tasks),
        "runnable_task_count": runnable,
        "reused_formal_task_count": reused,
        "max_total_worker_processes": max_processes,
        "max_concurrent_tasks_by_worker": {
            str(worker): max_processes // int(worker) for worker in robustness["workers"]
        },
        "tasks": tasks,
        "notes": [
            "The m=8 subset reuses hash-audited formal trajectories for seeds 0-9.",
            "Runnable tasks are staged by worker count and never exceed 32 worker processes.",
            "Each epsilon uses the region-frozen parameters selected without formal seeds.",
        ],
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def audit_robustness_preflight(cfg: Dict[str, Any], root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    unhashed = dict(manifest)
    expected_hash = unhashed.pop("manifest_sha256", None)
    errors: List[str] = []
    if expected_hash != object_sha256(unhashed):
        errors.append("robustness manifest hash mismatch")
    if manifest.get("status") != "prepared" or manifest.get("launches_started") != 0:
        errors.append("robustness manifest is not zero-launch prepared")
    reused = [row for row in manifest["tasks"] if row["source"] == "reuse-formal"]
    runnable = [row for row in manifest["tasks"] if row["source"] == "run"]
    for row in reused:
        task_id = row["task_id"]
        if file_sha256(root / row["source_partial_path"]) != row["source_partial_sha256"]:
            errors.append(f"reused partial hash mismatch: {task_id}")
        if file_sha256(root / row["source_manifest_path"]) != row["source_manifest_sha256"]:
            errors.append(f"reused manifest hash mismatch: {task_id}")
    if len(reused) != 60 or len(runnable) != 180:
        errors.append("reuse/runnable task counts are invalid")
    max_processes = int(cfg["cpu_process"]["max_total_worker_processes"])
    if any(int(worker) * int(limit) > max_processes for worker, limit in manifest["max_concurrent_tasks_by_worker"].items()):
        errors.append("worker concurrency exceeds process limit")

    formal_preflight = _load(root / "formal_preflight_audit.json")
    m8_cpu_hours_for_ten_seeds = float(formal_preflight["estimated_cpu_hours"]) / 2.0
    estimated_new_cpu_hours = m8_cpu_hours_for_ten_seeds * 3.0
    estimated_new_output_mib = float(formal_preflight["estimated_output_mib"]) / 2.0 * 3.0
    free_gib = shutil.disk_usage(root).free / 1073741824.0
    report = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "errors": errors,
        "logical_tasks": len(manifest["tasks"]),
        "runnable_tasks": len(runnable),
        "reused_formal_tasks": len(reused),
        "estimated_new_cpu_hours": estimated_new_cpu_hours,
        "estimated_new_output_mib": estimated_new_output_mib,
        "estimated_wall_hours_lower_bound": estimated_new_cpu_hours / max_processes,
        "conservative_wall_hours": 4.5,
        "free_disk_gib": free_gib,
        "launch_gate_passed": not errors and estimated_new_output_mib / 1024.0 < free_gib,
        "required_runtime_timeout_seconds": 7200,
        "notes": [
            "CPU-hour estimate assumes approximately constant total compute across worker counts.",
            "The 4.5-hour wall estimate includes worker-count scaling and scheduling overhead.",
            "This preflight launches no robustness workers.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    manifest = prepare_robustness(cfg, root)
    audit = audit_robustness_preflight(cfg, root, manifest)
    atomic_write_json(root / "robustness_manifest.json", manifest)
    atomic_write_json(root / "robustness_preflight_audit.json", audit)
    status = audit["status"]
    runnable = audit["runnable_tasks"]
    reused = audit["reused_formal_tasks"]
    cpu_hours = audit["estimated_new_cpu_hours"]
    print(f"status={status} runnable={runnable} reused={reused} estimated_cpu_hours={cpu_hours:.1f}")
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
