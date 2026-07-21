"""Short real-process profiles used to size later FO experiments.

The reported timings are diagnostics and never formal paper measurements.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.common import validate_experiment_config
from src.distributed.cpu_fo_correctness import (
    atomic_write_csv,
    load_config,
    process_config_from_experiment,
)
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    effective_task_config,
    run_task_set,
)


def cpu_resource_record() -> Dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    quota_us = None
    period_us = None
    cgroup_version = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota_text, period_text = cpu_max.read_text(encoding="utf-8").split()
        cgroup_version = 2
        if quota_text != "max":
            quota_us = int(quota_text)
            period_us = int(period_text)
    else:
        quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.exists() and period_path.exists():
            cgroup_version = 1
            observed_quota = int(quota_path.read_text(encoding="utf-8"))
            if observed_quota > 0:
                quota_us = observed_quota
                period_us = int(period_path.read_text(encoding="utf-8"))
    quota_cpus = (
        quota_us / period_us
        if quota_us is not None and period_us is not None
        else None
    )
    return {
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpu_count": len(affinity),
        "affinity_cpu_list": affinity,
        "cgroup_version": cgroup_version,
        "cgroup_quota_us": quota_us,
        "cgroup_period_us": period_us,
        "cgroup_quota_cpus": quota_cpus,
    }


def profile_tasks(cfg: Dict[str, Any]) -> List[CpuFoTask]:
    pcfg = cfg["profile"]
    return [
        CpuFoTask(method, int(seed), int(worker_count))
        for seed in pcfg["seeds"]
        for method in pcfg["methods"]
        for worker_count in pcfg["workers"]
    ]


def evaluation_checkpoint_count(
    cfg: Dict[str, Any],
    method: str,
    rounds: int,
) -> int:
    interval = int(cfg["train"]["eval_every"])
    stride = (
        int(cfg["nog"]["M"])
        if method == "NOG-FO"
        else int(cfg["me_dol"]["epoch_length"])
    )
    last_evaluation = 0
    count = 0
    for iteration in range(stride, rounds + 1, stride):
        first = iteration == stride
        final = iteration == rounds
        due = iteration - last_evaluation >= interval
        if first or final or due:
            count += 1
            last_evaluation = iteration
    return count


def _task_summary_row(
    cfg: Dict[str, Any],
    rounds: int,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    final = payload["rows"][-1]
    launch = payload["launch"]
    checkpoint_count = len(payload["rows"])
    training_time = float(final["training_time"])
    evaluation_time = float(final["evaluation_time"])
    end_to_end_time = float(launch["end_to_end_time"])
    non_training_overhead = max(
        0.0,
        end_to_end_time - training_time - evaluation_time,
    )
    expected_960_checkpoints = evaluation_checkpoint_count(
        cfg,
        payload["method"],
        960,
    )
    evaluation_per_checkpoint = evaluation_time / checkpoint_count
    estimated_960_end_to_end = (
        non_training_overhead
        + training_time / rounds * 960
        + evaluation_per_checkpoint * expected_960_checkpoints
    )
    return {
        "method": payload["method"],
        "formal_seed": int(payload["formal_seed"]),
        "rounds": rounds,
        "worker_count": int(payload["worker_count"]),
        "checkpoint_count": checkpoint_count,
        "depth": int(final["depth"]),
        "total_work": int(final["total_work"]),
        "per_worker_work": int(final["per_worker_work_max"]),
        "training_time": training_time,
        "communication_time": float(final["communication_time"]),
        "evaluation_time": evaluation_time,
        "end_to_end_time": end_to_end_time,
        "non_training_overhead": non_training_overhead,
        "training_seconds_per_round": training_time / rounds,
        "total_work_per_second": float(final["total_work"]) / training_time,
        "estimated_960_checkpoints": expected_960_checkpoints,
        "estimated_960_end_to_end": estimated_960_end_to_end,
        "unique_child_pids": len(set(launch["child_pids"])),
        "timing_invariants": (
            math.isfinite(training_time)
            and math.isfinite(evaluation_time)
            and math.isfinite(end_to_end_time)
            and training_time > 0.0
            and evaluation_time > 0.0
            and float(final["communication_time"]) <= training_time + 1e-12
            and end_to_end_time >= training_time
            and len(set(launch["child_pids"]))
            == int(payload["worker_count"])
        ),
    }


def _add_scaling_columns(rows: List[Dict[str, Any]]) -> None:
    lookup = {
        (row["method"], row["formal_seed"], row["rounds"], row["worker_count"]): row
        for row in rows
    }
    for row in rows:
        reference = lookup.get(
            (row["method"], row["formal_seed"], row["rounds"], 1)
        )
        if reference is None:
            row["training_speedup_vs_m1"] = None
            row["training_efficiency_vs_m1"] = None
            row["end_to_end_speedup_vs_m1"] = None
            continue
        speedup = reference["training_time"] / row["training_time"]
        row["training_speedup_vs_m1"] = speedup
        row["training_efficiency_vs_m1"] = speedup / row["worker_count"]
        row["end_to_end_speedup_vs_m1"] = (
            reference["end_to_end_time"] / row["end_to_end_time"]
        )


def _slowdown_audit(
    cfg: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    pcfg = cfg["profile"]
    threshold = float(pcfg.get("m64_slowdown_ratio", 1.5))
    budgets = [int(value) for value in pcfg.get("slowdown_rounds", [48, 96])]
    ratios = []
    by_key = {
        (row["method"], row["rounds"], row["worker_count"]): row
        for row in rows
    }
    for method in pcfg["methods"]:
        for rounds in budgets:
            row_32 = by_key.get((method, rounds, 32))
            row_64 = by_key.get((method, rounds, 64))
            ratio = (
                row_64["training_time"] / row_32["training_time"]
                if row_32 is not None and row_64 is not None
                else None
            )
            ratios.append(
                {
                    "method": method,
                    "rounds": rounds,
                    "m64_over_m32_training_time": ratio,
                    "severe": ratio is None or ratio > threshold,
                }
            )
    all_observed = all(item["m64_over_m32_training_time"] is not None for item in ratios)
    persistent_global_slowdown = (
        all_observed and all(item["severe"] for item in ratios)
    )
    return {
        "threshold": threshold,
        "ratios": ratios,
        "all_observed": all_observed,
        "persistent_global_slowdown": persistent_global_slowdown,
    }


def run_profile(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    validate_experiment_config(cfg)
    root = Path(output_root)
    process_config = process_config_from_experiment(cfg)
    tasks = profile_tasks(cfg)
    max_attempts = int(cfg["profile"].get("max_attempts", 3))
    all_rows: List[Dict[str, Any]] = []
    round_manifests = []
    wall_start = time.perf_counter()

    for rounds in [int(value) for value in cfg["profile"]["rounds"]]:
        round_cfg = copy.deepcopy(cfg)
        round_cfg["train"]["rounds"] = rounds
        if rounds % int(round_cfg["nog"]["M"]) != 0:
            raise ValueError(f"rounds={rounds} is not divisible by nog.M.")
        if rounds % int(round_cfg["me_dol"]["epoch_length"]) != 0:
            raise ValueError(
                f"rounds={rounds} is not divisible by me_dol.epoch_length."
            )
        round_root = root / f"rounds_{rounds}"
        completion = None
        for attempt in range(1, max_attempts + 1):
            completion = run_task_set(
                round_cfg,
                tasks,
                round_root,
                process_config,
                continue_on_error=True,
            )
            if completion["failed_tasks"] == 0:
                break
        assert completion is not None
        round_manifests.append(
            {
                "rounds": rounds,
                "attempts": attempt,
                "status": completion["status"],
                "completed_tasks": completion["completed_tasks"],
                "failed_tasks": completion["failed_tasks"],
                "manifest": str(
                    (round_root / "completion_manifest.json").relative_to(root)
                ),
                "records": completion["records"],
            }
        )
        for record in completion["records"]:
            if record["status"] == "failed":
                continue
            with open(
                round_root / record["partial_path"],
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
            all_rows.append(_task_summary_row(round_cfg, rounds, payload))

    _add_scaling_columns(all_rows)
    expected_task_count = len(tasks) * len(cfg["profile"]["rounds"])
    successful_keys = {
        (row["method"], row["formal_seed"], row["rounds"], row["worker_count"])
        for row in all_rows
    }
    expected_keys = {
        (task.method, task.formal_seed, int(rounds), task.worker_count)
        for task in tasks
        for rounds in cfg["profile"]["rounds"]
    }
    failed_keys = sorted(expected_keys - successful_keys)
    m64_keys = {key for key in expected_keys if key[3] == 64}
    lower_worker_failures = [key for key in failed_keys if key[3] < 64]
    m64_available = not (m64_keys - successful_keys)
    timing_invariants = all(row["timing_invariants"] for row in all_rows)
    slowdown = _slowdown_audit(cfg, all_rows)
    resources = cpu_resource_record()
    recommended_max_workers = (
        32
        if (not m64_available or slowdown["persistent_global_slowdown"])
        else 64
    )
    recommendation_reasons = []
    if not m64_available:
        recommendation_reasons.append("m64_not_reliably_available")
    if slowdown["persistent_global_slowdown"]:
        recommendation_reasons.append("m64_persistent_slowdown_vs_m32")
    quota_cpus = resources["cgroup_quota_cpus"]
    if quota_cpus is not None:
        quota_compatible_workers = [
            int(worker)
            for worker in cfg["profile"]["workers"]
            if int(worker) <= quota_cpus
        ]
        if quota_compatible_workers:
            quota_max_workers = max(quota_compatible_workers)
            if recommended_max_workers > quota_max_workers:
                recommended_max_workers = quota_max_workers
            if max(int(worker) for worker in cfg["profile"]["workers"]) > quota_cpus:
                recommendation_reasons.append(
                    f"cgroup_cpu_quota_{quota_cpus:g}"
                )
    status = (
        "failed"
        if lower_worker_failures or not timing_invariants
        else (
            "completed_with_m64_unavailable"
            if not m64_available
            else "passed"
        )
    )
    report = {
        "schema_version": 1,
        "status": status,
        "purpose": "runtime_profile_not_pilot_or_formal",
        "config": effective_task_config(cfg),
        "expected_tasks": expected_task_count,
        "completed_tasks": len(all_rows),
        "failed_keys": failed_keys,
        "m64_available": m64_available,
        "timing_invariants": timing_invariants,
        "slowdown_audit": slowdown,
        "cpu_resources": resources,
        "recommended_max_workers": recommended_max_workers,
        "recommendation_reasons": recommendation_reasons,
        "sequential_profile_wall_time": time.perf_counter() - wall_start,
        "sum_task_end_to_end_time": sum(
            row["end_to_end_time"] for row in all_rows
        ),
        "round_manifests": round_manifests,
        "rows": all_rows,
        "caveats": [
            "Profile times are diagnostics, not paper performance results.",
            "ME-DOL total work changes with worker_count, so its m1 speedup is not strong scaling.",
            "The 960-round estimate linearly extrapolates short runs and is provisional.",
            "Worker counts above the cgroup CPU quota are oversubscribed.",
        ],
    }
    atomic_write_json(root / "config_used.json", effective_task_config(cfg))
    atomic_write_json(root / "profile_report.json", report)
    atomic_write_csv(root / "profile.csv", all_rows)
    if status == "failed":
        raise RuntimeError(
            f"CPU FO profile failed below m=64 or violated timing invariants: "
            f"{lower_worker_failures}."
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    validate_experiment_config(cfg)
    tasks = profile_tasks(cfg)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    )
    total_tasks = len(tasks) * len(cfg["profile"]["rounds"])
    print(
        f"purpose=runtime_profile tasks={total_tasks} "
        f"rounds={cfg['profile']['rounds']} "
        f"workers={cfg['profile']['workers']}"
    )
    print(f"output_root={output_root}")
    if args.dry_run:
        print("dry_run=true; no processes launched")
        return
    report = run_profile(cfg, output_root)
    print(
        f"status={report['status']} "
        f"completed={report['completed_tasks']}/{report['expected_tasks']} "
        f"m64_available={report['m64_available']} "
        f"recommended_max_workers={report['recommended_max_workers']}"
    )
    print(f"report={output_root / 'profile_report.json'}")


if __name__ == "__main__":
    main()
