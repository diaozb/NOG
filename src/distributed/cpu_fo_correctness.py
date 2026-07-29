"""Correctness matrix and audits for the real CPU-process FO runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from src.distributed.algorithms import run_me_dol, run_nog
from src.distributed.common import (
    build_problem,
    make_seed_bundle,
    make_worker_shards,
    validate_experiment_config,
)
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    effective_task_config,
    run_task_set,
)
from src.distributed.cpu_process import CpuProcessConfig


TIMING_KEYS = {
    "time_sec",
    "training_time",
    "communication_time",
    "evaluation_time",
}


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Correctness config must be a YAML mapping.")
    return value


def process_config_from_experiment(cfg: Dict[str, Any]) -> CpuProcessConfig:
    pcfg = cfg.get("cpu_process", {})
    return CpuProcessConfig(
        backend="gloo",
        process_group_timeout_seconds=float(
            pcfg.get("process_group_timeout_seconds", 60.0)
        ),
        launch_timeout_seconds=float(
            pcfg.get("launch_timeout_seconds", 120.0)
        ),
        intraop_threads=int(pcfg.get("threads_per_rank", 1)),
    )


def correctness_tasks(cfg: Dict[str, Any]) -> List[CpuFoTask]:
    ccfg = cfg["correctness"]
    return [
        CpuFoTask(method, int(seed), int(worker_count))
        for seed in ccfg["seeds"]
        for method in ccfg["methods"]
        for worker_count in ccfg["workers"]
    ]


def simulator_rows(
    cfg: Dict[str, Any],
    task: CpuFoTask,
) -> List[Dict[str, Any]]:
    effective = effective_task_config(cfg)
    seed_bundle = make_seed_bundle(
        task.formal_seed,
        task.method,
        task.worker_count,
    )
    problem = build_problem(effective, "cpu", seed_bundle.problem_seed)
    shards = make_worker_shards(
        problem.n,
        task.worker_count,
        "cpu",
        seed_bundle.partition_seed,
        shuffle=bool(
            effective["distributed"].get("shuffle_partitions", True)
        ),
    )
    if task.method == "NOG-FO":
        return run_nog(
            problem,
            effective,
            shards,
            seed_bundle,
            "sfo",
            task.method,
        )
    return run_me_dol(
        problem,
        effective,
        shards,
        seed_bundle,
        "sfo",
        task.method,
    )


def compare_trajectories(
    expected: List[Dict[str, Any]],
    observed: List[Dict[str, Any]],
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, float, List[str]]:
    errors: List[str] = []
    max_abs_difference = 0.0
    if len(expected) != len(observed):
        return (
            False,
            math.inf,
            [f"checkpoint count {len(observed)} != {len(expected)}"],
        )

    for row_index, (expected_row, observed_row) in enumerate(
        zip(expected, observed)
    ):
        expected_keys = set(expected_row) - TIMING_KEYS
        observed_keys = set(observed_row) - TIMING_KEYS
        if expected_keys != observed_keys:
            errors.append(
                f"row {row_index} keys differ: "
                f"missing={sorted(expected_keys - observed_keys)}, "
                f"extra={sorted(observed_keys - expected_keys)}"
            )
            continue
        for key in expected_keys:
            expected_value = expected_row[key]
            observed_value = observed_row[key]
            if isinstance(expected_value, float):
                difference = abs(expected_value - float(observed_value))
                max_abs_difference = max(max_abs_difference, difference)
                if not math.isclose(
                    expected_value,
                    float(observed_value),
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    errors.append(
                        f"row {row_index} {key}: "
                        f"{observed_value} != {expected_value}"
                    )
            elif expected_value != observed_value:
                errors.append(
                    f"row {row_index} {key}: "
                    f"{observed_value!r} != {expected_value!r}"
                )
    return not errors, max_abs_difference, errors


def expected_final_accounting(
    cfg: Dict[str, Any],
    task: CpuFoTask,
    checkpoint_count: int,
) -> Dict[str, Any]:
    rounds = int(cfg["train"]["rounds"])
    eval_calls = (
        int(cfg["oracle"]["eval_smooth_B"])
        * int(cfg["oracle"]["eval_data_B"])
    )
    if task.method == "NOG-FO":
        per_aggregation_total = (
            int(cfg["oracle"]["smooth_B"])
            * int(cfg["oracle"]["data_B_total"])
        )
        total_work = (rounds + 2) * per_aggregation_total
        per_worker = total_work // task.worker_count
        depth = rounds + 2
    else:
        total_work = rounds * task.worker_count
        per_worker = rounds
        depth = rounds
    return {
        "total_work": total_work,
        "per_worker_work": [per_worker] * task.worker_count,
        "per_worker_work_max": per_worker,
        "communication_round": depth,
        "depth": depth,
        "eval_work": checkpoint_count * eval_calls,
    }


def _nondecreasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(
        current >= previous
        for previous, current in zip(sequence, sequence[1:])
    )


def audit_task_payload(
    cfg: Dict[str, Any],
    task: CpuFoTask,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    rows = payload["rows"]
    expected_rows = simulator_rows(cfg, task)
    ccfg = cfg["correctness"]
    trajectory_close, max_abs_difference, trajectory_errors = (
        compare_trajectories(
            expected_rows,
            rows,
            rel_tol=float(ccfg.get("relative_tolerance", 1e-5)),
            abs_tol=float(ccfg.get("absolute_tolerance", 1e-7)),
        )
    )

    seed_bundle = make_seed_bundle(
        task.formal_seed,
        task.method,
        task.worker_count,
    )
    expected_seed_fields = seed_bundle.as_dict()
    seed_ok = all(
        all(row.get(key) == value for key, value in expected_seed_fields.items())
        for row in rows
    )

    rank_metadata = payload["rank_metadata"]
    rank_pids = [int(item["pid"]) for item in rank_metadata]
    launch_pids = [int(pid) for pid in payload["launch"]["child_pids"]]
    pid_ok = (
        len(rank_pids) == task.worker_count
        and len(set(rank_pids)) == task.worker_count
        and set(rank_pids) == set(launch_pids)
    )
    shard_ok = (
        sum(int(item["shard_size"]) for item in rank_metadata)
        == int(cfg["problem"]["n_data"])
        and len({item["shard_sha256"] for item in rank_metadata})
        == task.worker_count
    )
    thread_ok = all(
        int(item["torch_threads"])
        == int(cfg["cpu_process"]["threads_per_rank"])
        for item in rank_metadata
    )

    expected_accounting = expected_final_accounting(
        cfg,
        task,
        len(rows),
    )
    final = rows[-1]
    accounting_ok = all(
        final.get(key) == value
        for key, value in expected_accounting.items()
    )
    work_depth_monotone = (
        _nondecreasing(row["total_work"] for row in rows)
        and _nondecreasing(row["per_worker_work_max"] for row in rows)
        and _nondecreasing(row["communication_round"] for row in rows)
        and _nondecreasing(row["eval_work"] for row in rows)
    )

    training_times = [float(row["training_time"]) for row in rows]
    communication_times = [float(row["communication_time"]) for row in rows]
    evaluation_times = [float(row["evaluation_time"]) for row in rows]
    time_alias_ok = all(
        math.isclose(
            float(row["time_sec"]),
            float(row["training_time"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in rows
    )
    timing_ok = (
        all(value > 0.0 for value in training_times)
        and all(value >= 0.0 for value in communication_times)
        and all(value > 0.0 for value in evaluation_times)
        and _nondecreasing(training_times)
        and _nondecreasing(communication_times)
        and _nondecreasing(evaluation_times)
        and all(
            communication <= training + 1e-12
            for communication, training in zip(
                communication_times,
                training_times,
            )
        )
        and float(payload["launch"]["end_to_end_time"])
        >= training_times[-1]
        and time_alias_ok
    )

    checks = {
        "trajectory_allclose": trajectory_close,
        "seed_mapping": seed_ok,
        "independent_pids": pid_ok,
        "shard_metadata": shard_ok,
        "one_thread_per_rank": thread_ok,
        "final_work_depth_eval": accounting_ok,
        "work_depth_time_monotone": work_depth_monotone,
        "timing_partition": timing_ok,
        "rng_mode": payload.get("rng_mode") == "rank_schedule",
        "task_identity": (
            payload.get("method") == task.method
            and payload.get("formal_seed") == task.formal_seed
            and payload.get("worker_count") == task.worker_count
        ),
    }
    return {
        "task": task.as_dict(),
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_count": len(rows),
        "max_abs_trajectory_difference": max_abs_difference,
        "trajectory_errors": trajectory_errors[:20],
        "expected_final_accounting": expected_accounting,
        "observed_final_accounting": {
            key: final.get(key)
            for key in expected_accounting
        },
        "final_timing": {
            "training_time": training_times[-1],
            "communication_time": communication_times[-1],
            "evaluation_time": evaluation_times[-1],
            "end_to_end_time": float(payload["launch"]["end_to_end_time"]),
        },
        "rank_pids": rank_pids,
        "rank_metadata": rank_metadata,
        "seed_bundle": expected_seed_fields,
    }


def atomic_write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        fieldnames = list(rows[0]) if rows else []
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_correctness_matrix(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    validate_experiment_config(cfg)
    tasks = correctness_tasks(cfg)
    process_config = process_config_from_experiment(cfg)
    root = Path(output_root)
    completion = run_task_set(
        cfg,
        tasks,
        root,
        process_config,
        continue_on_error=False,
    )
    audits = []
    record_by_task = {
        (
            record["task"]["method"],
            record["task"]["formal_seed"],
            record["task"]["worker_count"],
        ): record
        for record in completion["records"]
    }
    for task in tasks:
        record = record_by_task[
            (task.method, task.formal_seed, task.worker_count)
        ]
        partial_path = root / record["partial_path"]
        with open(partial_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        audits.append(audit_task_payload(cfg, task, payload))

    passed = all(audit["passed"] for audit in audits)
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "purpose": "correctness_only_not_pilot_profile_or_formal",
        "config": effective_task_config(cfg),
        "completion_status": completion["status"],
        "task_count": len(tasks),
        "passed_tasks": sum(audit["passed"] for audit in audits),
        "failed_tasks": sum(not audit["passed"] for audit in audits),
        "audits": audits,
    }
    atomic_write_json(root / "config_used.json", effective_task_config(cfg))
    atomic_write_json(root / "correctness_audit.json", report)
    summary_rows = [
        {
            "method": audit["task"]["method"],
            "formal_seed": audit["task"]["formal_seed"],
            "worker_count": audit["task"]["worker_count"],
            "passed": audit["passed"],
            "checkpoint_count": audit["checkpoint_count"],
            "max_abs_trajectory_difference": audit[
                "max_abs_trajectory_difference"
            ],
            "final_depth": audit["observed_final_accounting"]["depth"],
            "final_total_work": audit["observed_final_accounting"]["total_work"],
            "final_per_worker_work": audit[
                "observed_final_accounting"
            ]["per_worker_work_max"],
            "final_eval_work": audit["observed_final_accounting"]["eval_work"],
            **audit["final_timing"],
        }
        for audit in audits
    ]
    atomic_write_csv(root / "correctness_summary.csv", summary_rows)
    if not passed:
        failed = [
            audit["task"]
            for audit in audits
            if not audit["passed"]
        ]
        raise RuntimeError(f"CPU FO correctness matrix failed: {failed}.")
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
    tasks = correctness_tasks(cfg)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    )
    print(f"purpose=correctness_only tasks={len(tasks)}")
    print(f"output_root={output_root}")
    for task in tasks:
        print(
            f"  method={task.method} seed={task.formal_seed} "
            f"workers={task.worker_count}"
        )
    if args.dry_run:
        print("dry_run=true; no processes launched")
        return
    report = run_correctness_matrix(cfg, output_root)
    print(
        f"status={report['status']} "
        f"passed={report['passed_tasks']}/{report['task_count']}"
    )
    print(f"audit={output_root / 'correctness_audit.json'}")


if __name__ == "__main__":
    main()
