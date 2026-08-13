"""Step ZO-9A/B: audit real Gloo processes against the logical ZO simulator."""

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
from src.distributed.cpu_fo_correctness import compare_trajectories
from src.distributed.cpu_process import CpuProcessConfig
from src.distributed.cpu_zo_algorithms import (
    SUPPORTED_CPU_ZO_METHODS,
    run_cpu_zo_task,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/distributed_cpu_zo_equivalence.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/distributed_cpu_zo/equivalence"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Equivalence config must be a mapping.")
    validate_experiment_config(value)
    return value


def process_config(cfg: Dict[str, Any]) -> CpuProcessConfig:
    pcfg = cfg["cpu_process"]
    return CpuProcessConfig(
        backend="gloo",
        process_group_timeout_seconds=float(pcfg["process_group_timeout_seconds"]),
        launch_timeout_seconds=float(pcfg["launch_timeout_seconds"]),
        intraop_threads=int(pcfg["threads_per_rank"]),
    )


def task_matrix(cfg: Dict[str, Any]) -> List[tuple[str, int, int]]:
    ecfg = cfg["equivalence"]
    methods = [str(value) for value in ecfg["methods"]]
    unknown = set(methods) - SUPPORTED_CPU_ZO_METHODS
    if unknown:
        raise ValueError(f"Unsupported equivalence methods: {sorted(unknown)}.")
    return [
        (method, int(seed), int(workers))
        for seed in ecfg["seeds"]
        for method in methods
        for workers in ecfg["workers"]
    ]


def simulator_rows(
    cfg: Dict[str, Any], method: str, formal_seed: int, workers: int
) -> List[Dict[str, Any]]:
    bundle = make_seed_bundle(formal_seed, method, workers)
    problem = build_problem(cfg, "cpu", bundle.problem_seed)
    shards = make_worker_shards(
        problem.n,
        workers,
        "cpu",
        bundle.partition_seed,
        shuffle=bool(cfg["distributed"].get("shuffle_partitions", True)),
    )
    if method == "NOG-ZO":
        return run_nog(problem, cfg, shards, bundle, "szo", method)
    return run_me_dol(problem, cfg, shards, bundle, "szo", method)


def expected_final_accounting(
    cfg: Dict[str, Any], method: str, workers: int, checkpoints: int
) -> Dict[str, Any]:
    rounds = int(cfg["train"]["rounds"])
    eval_calls = int(cfg["oracle"]["eval_smooth_B"]) * int(
        cfg["oracle"]["eval_data_B"]
    )
    if method == "NOG-ZO":
        local_data = int(cfg["oracle"]["data_B_total"]) // workers
        per_layer_per_worker = (
            2 * int(cfg["oracle"]["smooth_B"]) * local_data
        )
        per_worker = (rounds + 2) * per_layer_per_worker
        depth = rounds + 2
    else:
        per_layer_per_worker = (
            2
            * int(cfg["me_dol"].get("smooth_B", 1))
            * int(cfg["me_dol"].get("data_B_per_worker", 1))
        )
        per_worker = rounds * per_layer_per_worker
        depth = rounds
    return {
        "total_work": per_worker * workers,
        "per_worker_work": [per_worker] * workers,
        "per_worker_work_max": per_worker,
        "communication_round": depth,
        "depth": depth,
        "eval_work": checkpoints * eval_calls,
    }


def _nondecreasing(values: Iterable[float]) -> bool:
    values = list(values)
    return all(right >= left for left, right in zip(values, values[1:]))


def audit_payload(
    cfg: Dict[str, Any], method: str, seed: int, workers: int, payload: Dict[str, Any]
) -> Dict[str, Any]:
    expected_rows = simulator_rows(cfg, method, seed, workers)
    observed_rows = payload["rows"]
    ecfg = cfg["equivalence"]
    close, max_difference, errors = compare_trajectories(
        expected_rows,
        observed_rows,
        rel_tol=float(ecfg["relative_tolerance"]),
        abs_tol=float(ecfg["absolute_tolerance"]),
    )
    final = observed_rows[-1]
    accounting = expected_final_accounting(
        cfg, method, workers, len(observed_rows)
    )
    metadata = payload["rank_metadata"]
    rank_pids = [int(item["pid"]) for item in metadata]
    child_pids = [int(value) for value in payload["launch"]["child_pids"]]
    checks = {
        "trajectory_allclose": close,
        "task_identity": (
            payload["method"] == method
            and int(payload["formal_seed"]) == seed
            and int(payload["worker_count"]) == workers
        ),
        "rank_schedule": payload.get("rng_mode") == "rank_schedule",
        "independent_processes": (
            len(rank_pids) == workers
            and len(set(rank_pids)) == workers
            and set(rank_pids) == set(child_pids)
        ),
        "one_thread_per_rank": all(
            int(item["torch_threads"]) == int(cfg["cpu_process"]["threads_per_rank"])
            for item in metadata
        ),
        "exhaustive_distinct_shards": (
            sum(int(item["shard_size"]) for item in metadata)
            == int(cfg["problem"]["n_data"])
            and len({item["shard_sha256"] for item in metadata}) == workers
        ),
        "final_accounting": all(final.get(key) == value for key, value in accounting.items()),
        "monotone_counters": all(
            _nondecreasing(row[key] for row in observed_rows)
            for key in ["total_work", "per_worker_work_max", "depth", "eval_work"]
        ),
        "timing_sanity": (
            float(payload["launch"]["end_to_end_time"]) > 0
            and all(float(row["training_time"]) > 0 for row in observed_rows)
            and all(float(row["communication_time"]) >= 0 for row in observed_rows)
            and all(float(row["evaluation_time"]) > 0 for row in observed_rows)
        ),
    }
    return {
        "method": method,
        "seed": seed,
        "workers": workers,
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoints": len(observed_rows),
        "max_abs_trajectory_difference": max_difference,
        "trajectory_errors": errors[:20],
        "expected_final_accounting": accounting,
        "observed_final_accounting": {
            key: final.get(key) for key in accounting
        },
        "end_to_end_time": float(payload["launch"]["end_to_end_time"]),
        "rank_pids": rank_pids,
    }


def run_matrix(cfg: Dict[str, Any], output: Path) -> Dict[str, Any]:
    tasks = task_matrix(cfg)
    partials = output / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    audits = []
    for index, (method, seed, workers) in enumerate(tasks, start=1):
        safe = method.replace("+", "plus")
        path = partials / f"{safe}__m{workers}__seed{seed}.json"
        print(
            f"[{index}/{len(tasks)}] method={method} seed={seed} workers={workers}",
            flush=True,
        )
        # The matrix is deliberately tiny. Re-run each probe instead of silently
        # trusting a partial created by a different source revision.
        run_cpu_zo_task(cfg, method, seed, workers, path, process_config(cfg))
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = audit_payload(cfg, method, seed, workers, payload)
        audits.append(audit)
        if not audit["passed"]:
            atomic_json(output / "equivalence_audit.json", {"audits": audits})
            raise RuntimeError(f"ZO process equivalence failed: {method}/m={workers}.")
    report = {
        "schema_version": 1,
        "status": "passed",
        "claim_boundary": (
            "numerical and accounting equivalence for NOG-ZO and ME-DOL-ZO; "
            "not a cluster speedup benchmark and not coverage of DGFM/DGFM+"
        ),
        "task_count": len(tasks),
        "passed_tasks": sum(audit["passed"] for audit in audits),
        "audits": audits,
    }
    atomic_json(output / "equivalence_audit.json", report)
    atomic_csv(
        output / "equivalence_summary.csv",
        [
            {
                "method": item["method"],
                "seed": item["seed"],
                "workers": item["workers"],
                "passed": item["passed"],
                "checkpoints": item["checkpoints"],
                "max_abs_trajectory_difference": item[
                    "max_abs_trajectory_difference"
                ],
                "final_depth": item["observed_final_accounting"]["depth"],
                "final_total_work": item["observed_final_accounting"]["total_work"],
                "final_per_worker_work": item["observed_final_accounting"][
                    "per_worker_work_max"
                ],
                "end_to_end_time": item["end_to_end_time"],
            }
            for item in audits
        ],
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    tasks = task_matrix(cfg)
    print(f"step=ZO-9A/B tasks={len(tasks)} output={args.output}")
    for method, seed, workers in tasks:
        print(f"  {method} seed={seed} workers={workers}")
    if args.dry_run:
        return
    report = run_matrix(cfg, Path(args.output))
    print(f"status={report['status']} passed={report['passed_tasks']}/{report['task_count']}")


if __name__ == "__main__":
    main()
