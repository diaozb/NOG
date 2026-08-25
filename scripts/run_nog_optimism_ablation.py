"""Run the fixed-parameter NOG optimistic/non-optimistic CPU ablation.

The two methods share the same problem, partition, stateless oracle streams,
initialization accounting, batch, M, eta, evaluation bank, and worker count.
Only the update rule differs.  This script is intentionally separate from the
paper formal runner and writes a new versioned result directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    environment_record,
    run_or_resume_task,
)
from src.distributed.cpu_fo_correctness import load_config, process_config_from_experiment


DEFAULT_CONFIG = ROOT / "configs/distributed_cpu_fo_nog_optimism_ablation_v4.yaml"
DEFAULT_OUTPUT = ROOT / "results/nog_optimism_ablation_20260825"
METHODS = ("NOG-FO", "NOG-FO-NONOPT")


def _seeds(cfg: dict, stage: str) -> list[int]:
    values = cfg["ablation"]["pilot_seeds" if stage == "pilot" else "formal_seeds"]
    return [int(value) for value in values]


def _write_protocol(cfg: dict, output: Path, stage: str, seeds: list[int], concurrency: int) -> None:
    payload = {
        "status": "running",
        "stage": stage,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "config_path": str(DEFAULT_CONFIG),
        "methods": list(METHODS),
        "seeds": seeds,
        "worker_count": int(cfg["ablation"]["worker_count"]),
        "concurrency": int(concurrency),
        "same_random_streams": True,
        "same_initialization_oracle_count": True,
        "fixed_parameters": copy.deepcopy(cfg["ablation"]["fixed_parameters"]),
        "problem": copy.deepcopy(cfg["problem"]),
        "train": copy.deepcopy(cfg["train"]),
        "oracle": copy.deepcopy(cfg["oracle"]),
        "formal_seed_reservation": cfg["ablation"]["formal_seeds"],
    }
    atomic_write_json(output / f"protocol_{stage}.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "formal"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seeds", default=None, help="comma-separated override for smoke tests")
    parser.add_argument("--rounds", type=int, default=None, help="override training rounds for smoke tests")
    parser.add_argument("--eval-every", type=int, default=None, help="override evaluation interval for smoke tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    cfg["run"]["device"] = "cpu"
    cfg["distributed"]["rng_mode"] = "rank_schedule"
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else _seeds(cfg, args.stage)
    )
    if args.rounds is not None:
        if args.rounds < 1 or args.rounds % int(cfg["nog"]["M"]) != 0:
            raise ValueError("--rounds must be positive and divisible by M.")
        cfg["train"]["rounds"] = int(args.rounds)
    if args.eval_every is not None:
        cfg["train"]["eval_every"] = int(args.eval_every)
    worker_count = int(cfg["ablation"]["worker_count"])
    if args.concurrency < 1 or args.concurrency * worker_count > int(cfg["cpu_process"]["max_total_worker_processes"]):
        raise ValueError("Concurrency exceeds the configured CPU worker-process budget.")

    output = args.output / args.stage
    output.mkdir(parents=True, exist_ok=True)
    _write_protocol(cfg, output, args.stage, seeds, args.concurrency)
    process = process_config_from_experiment(cfg)
    tasks = [CpuFoTask(method, seed, worker_count) for seed in seeds for method in METHODS]
    print(
        f"stage={args.stage} tasks={len(tasks)} methods={METHODS} "
        f"seeds={seeds} workers_per_task={worker_count} concurrency={args.concurrency}",
        flush=True,
    )
    started = time.monotonic()
    records = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="ablation") as pool:
        futures = {
            pool.submit(run_or_resume_task, cfg, task, output, process): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                record = {
                    "task": task.as_dict(),
                    "status": result.status,
                    "task_key": result.task_key,
                    "partial_path": str(result.partial_path.relative_to(output)),
                    "manifest_path": str(result.manifest_path.relative_to(output)),
                    "row_count": result.row_count,
                }
                records.append(record)
                print(f"completed {task.method} seed={task.formal_seed} status={result.status}", flush=True)
            except BaseException as error:  # retain failure metadata and continue all paired tasks
                failure = {
                    "task": task.as_dict(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                failures.append(failure)
                records.append(failure)
                print(f"FAILED {task.method} seed={task.formal_seed}: {error}", flush=True)

    records.sort(key=lambda row: (row["task"]["formal_seed"], row["task"]["method"]))
    completion = {
        "status": "complete" if not failures else "failed_tasks_present",
        "stage": args.stage,
        "elapsed_seconds": time.monotonic() - started,
        "task_count": len(tasks),
        "completed_count": len(tasks) - len(failures),
        "failure_count": len(failures),
        "records": records,
        "environment": environment_record(),
        "python_executable": "/root/miniconda3/envs/NOG/bin/python",
    }
    atomic_write_json(output / "completion.json", completion)
    print(json.dumps({k: completion[k] for k in ("status", "task_count", "completed_count", "failure_count", "elapsed_seconds")}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
