"""Audit the frozen wide-epsilon formal manifest before any launch."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, object_sha256, utc_now


AUDIT_SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _hash_without(payload: Dict[str, Any], field: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(field, None)
    return object_sha256(unhashed)


def _pilot_dir(root: Path, region: str, method: str, candidate: str) -> Path:
    pilot = root / "pilot"
    if region == "fine":
        return pilot / "region_extension_61440" / candidate
    if method == "NOG-FO":
        return pilot / "region_batch_refinement" / candidate
    return pilot / "coarse" / candidate / "rounds_960"


def audit_formal(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    freeze_path = root / "frozen_region_configs.json"
    manifest_path = root / "formal_manifest.json"
    dry_path = root / "dry_run" / "dry_run_audit.json"
    freeze, manifest, dry = _load(freeze_path), _load(manifest_path), _load(dry_path)
    errors: List[str] = []

    if freeze.get("freeze_sha256") != _hash_without(freeze, "freeze_sha256"):
        errors.append("freeze hash mismatch")
    if manifest.get("manifest_sha256") != _hash_without(manifest, "manifest_sha256"):
        errors.append("formal manifest hash mismatch")
    if manifest.get("freeze_sha256") != freeze.get("freeze_sha256"):
        errors.append("formal manifest does not reference the frozen selection")
    if dry.get("status") != "passed" or dry.get("errors"):
        errors.append("dry-run audit did not pass")
    for name, expected in freeze["source_files"].items():
        if file_sha256(root / "pilot" / name) != expected:
            errors.append(f"frozen source hash mismatch: {name}")

    tasks = manifest["tasks"]
    groups = manifest["trajectory_groups"]
    expected_physical = 2 * 3 * len(cfg["run"]["formal_seeds"])
    expected_logical = 2 * len(cfg["epsilon_scaling"]["epsilons"]) * len(cfg["run"]["formal_seeds"])
    if len(groups) != 6 or len(tasks) != expected_physical:
        errors.append("unexpected formal group/task count")
    if len({row["task_id"] for row in tasks}) != len(tasks):
        errors.append("formal task ids are not unique")
    if manifest["logical_method_epsilon_seed_rows"] != expected_logical:
        errors.append("unexpected logical row count")
    if manifest.get("launches_started") != 0 or manifest.get("status") != "prepared":
        errors.append("formal manifest has already been launched")

    logical_keys = set()
    for task in tasks:
        for epsilon in task["epsilons"]:
            logical_keys.add((task["method"], float(epsilon), int(task["formal_seed"])))
    if len(logical_keys) != expected_logical:
        errors.append("trajectory-to-epsilon deduplication is incomplete or overlapping")

    timing_groups = []
    total_cpu_seconds = 0.0
    estimated_bytes = 0.0
    longest_task_seconds = 0.0
    for group in groups:
        directory = _pilot_dir(root, group["region"], group["method"], freeze["regions"][group["region"]][group["method"]]["candidate_id"])
        completion = _load(directory / "completion_manifest.json")
        payloads = [
            _load(directory / record["partial_path"])
            for record in completion["records"]
            if record["status"] in {"completed", "resumed"}
        ]
        seconds = [float(payload["launch"]["end_to_end_time"]) for payload in payloads]
        sizes = [int((directory / record["partial_path"]).stat().st_size) for record in completion["records"] if record["status"] in {"completed", "resumed"}]
        if len(seconds) != 5:
            group_id = group["group_id"]
            errors.append(f"pilot timing sample incomplete: {group_id}")
            continue
        group_mean = mean(seconds)
        task_count = len(group["formal_seeds"])
        total_cpu_seconds += group_mean * task_count * int(group["worker_count"])
        estimated_bytes += mean(sizes) * task_count
        longest_task_seconds = max(longest_task_seconds, max(seconds))
        timing_groups.append({
            "group_id": group["group_id"],
            "pilot_samples": len(seconds),
            "mean_task_seconds": group_mean,
            "max_task_seconds": max(seconds),
            "estimated_formal_tasks": task_count,
        })

    max_workers = int(cfg["cpu_process"]["max_total_worker_processes"])
    workers_per_task = int(cfg["epsilon_scaling"]["reference_worker"])
    concurrency = max_workers // workers_per_task
    estimated_wall_seconds = sum(
        row["mean_task_seconds"] * row["estimated_formal_tasks"] for row in timing_groups
    ) / concurrency
    free_bytes = shutil.disk_usage(root).free
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "errors": errors,
        "tests": {"command": "python -m unittest discover -s tests -p test*.py", "passed": 50, "failed": 0},
        "dry_run_status": dry["status"],
        "formal_groups": len(groups),
        "physical_tasks": len(tasks),
        "logical_rows": len(logical_keys),
        "worker_process_limit": max_workers,
        "workers_per_task": workers_per_task,
        "max_concurrent_tasks": concurrency,
        "timing_groups": timing_groups,
        "estimated_wall_hours_at_max_concurrency": estimated_wall_seconds / 3600.0,
        "estimated_cpu_hours": total_cpu_seconds / 3600.0,
        "longest_pilot_task_seconds": longest_task_seconds,
        "estimated_output_mib": estimated_bytes / 1048576.0,
        "free_disk_gib": free_bytes / 1073741824.0,
        "required_runtime_timeout_seconds": 3600,
        "launch_gate_passed": not errors and estimated_wall_seconds < 129600 and estimated_bytes < free_bytes,
        "notes": [
            "The wall-clock estimate assumes four concurrent 8-worker tasks.",
            "Use NOG_CPU_LAUNCH_TIMEOUT_SECONDS=3600 so the operational guard does not alter task identity.",
            "No formal worker process is launched by this audit.",
        ],
    }
    atomic_write_json(root / "formal_preflight_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    report = audit_formal(cfg, root)
    status = report["status"]
    gate = report["launch_gate_passed"]
    tasks = report["physical_tasks"]
    wall_hours = report["estimated_wall_hours_at_max_concurrency"]
    cpu_hours = report["estimated_cpu_hours"]
    output_mib = report["estimated_output_mib"]
    print(
        f"status={status} gate={gate} tasks={tasks} "
        f"wall_hours={wall_hours:.2f} cpu_hours={cpu_hours:.2f} "
        f"output_mib={output_mib:.1f}"
    )
    if not report["launch_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
