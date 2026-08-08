"""Step ZO-7A: resumable fixed-configuration dimension calibration."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.distributed.run_distributed_baselines import (
    environment_record,
    load_config,
    save_yaml,
)
from src.distributed.zo_formal import (
    atomic_csv,
    atomic_json,
    load_and_verify_freeze,
    sha256,
)
from src.distributed.zo_range_pilot import candidate_id
from src.distributed.zo_refine_pilot import run_one, summarize
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = ROOT / "zo_experiments/frozen_parameters.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/dimension"
    / "calibration_fixed_params_work983040"
)
DEFAULT_DIMENSIONS = [25, 50, 100, 200]
DEFAULT_SEED = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dimensions", default="25,50,100,200")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def acquire_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / "runner.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("Another Step ZO-7A runner is active.") from error
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def validate_partial(
    frame: pd.DataFrame,
    method: str,
    parameters: dict[str, Any],
    dimension: int,
    seed: int,
    target_work: int,
) -> None:
    if set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"Method mismatch for d={dimension}/{method}.")
    if set(frame["formal_seed"].astype(int)) != {seed}:
        raise ValueError(f"Seed mismatch for d={dimension}/{method}.")
    if set(frame["dimension"].astype(int)) != {dimension}:
        raise ValueError(f"Dimension mismatch for d={dimension}/{method}.")
    expected = json.dumps(parameters, sort_keys=True)
    if set(frame["candidate_parameters"].astype(str)) != {expected}:
        raise ValueError(f"Parameter mismatch for d={dimension}/{method}.")
    ordered = frame.sort_values("depth")
    if int(ordered["total_work"].iloc[-1]) != target_work:
        raise ValueError(f"Work mismatch for d={dimension}/{method}.")


def task_name(
    method: str, parameters: dict[str, Any], dimension: int, seed: int
) -> str:
    return (
        f"d-{dimension}__{candidate_id(method, parameters)}__seed-{seed}"
    )


def main() -> None:
    args = parse_args()
    dimensions = [
        int(value)
        for value in args.dimensions.split(",")
        if value.strip()
    ]
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("Dimensions must be positive integers.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Dimensions must be unique.")

    freeze_path = Path(args.freeze).resolve()
    freeze = load_and_verify_freeze(freeze_path)
    if args.seed in {
        int(value)
        for key in ["pilot_seeds", "formal_seeds"]
        for value in freeze[key]
    }:
        raise ValueError("Calibration seed overlaps pilot/formal seeds.")
    base_cfg = load_config(ROOT / freeze["base_config"])
    if args.seed in {
        int(value) for value in base_cfg["run"]["anomaly_seeds"]
    }:
        raise ValueError("Calibration seed overlaps anomaly seeds.")
    base_cfg["run"]["pilot_selection_complete"] = True
    base_cfg["pilot"]["refine"]["target_total_work"] = int(
        freeze["target_total_work"]
    )
    base_cfg["pilot"]["refine"]["eval_every"] = int(freeze["eval_every"])
    base_cfg["oracle"]["eval_smooth_B"] = int(freeze["eval_smooth_B"])
    base_cfg["oracle"]["eval_data_B"] = int(freeze["eval_data_B"])

    candidates = [
        (entry["method"], entry["parameters"])
        for entry in freeze["selected_candidates"]
    ]
    target_work = int(freeze["target_total_work"])
    worker_count = int(freeze["worker_count"])
    device = get_device(base_cfg["run"].get("device", "auto"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    tasks = [
        (dimension, method, parameters)
        for dimension in dimensions
        for method, parameters in candidates
    ]
    print(
        f"step=ZO-7A device={device} tasks={len(tasks)} "
        f"dimensions={dimensions} seed={args.seed} "
        f"target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for dimension, method, parameters in tasks:
            print(task_name(method, parameters, dimension, args.seed))
        return

    lock_handle = acquire_lock(output)
    try:
        partials.mkdir(parents=True, exist_ok=True)
        shutil.copy2(freeze_path, output / "frozen_parameters.json")
        save_yaml(base_cfg, output / "config_base.yaml")
        atomic_json(environment_record(device), output / "environment.json")
        atomic_json(
            {
                "step": "ZO-7A",
                "role": "dimension_runtime_and_hit_calibration",
                "paper_result": False,
                "merge_with_formal_results": False,
                "parameter_policy": (
                    "reuse_d100_frozen_candidates_without_retuning"
                ),
                "dimensions": dimensions,
                "calibration_seed": int(args.seed),
                "target_total_work": target_work,
                "worker_count": worker_count,
                "freeze_sha256": sha256(freeze_path),
                "purpose": (
                    "measure runtime, hit range, and cross-dimension "
                    "stability before formal dimension scaling"
                ),
            },
            output / "calibration_manifest.json",
        )

        completed: list[pd.DataFrame] = []
        runtime_records: list[dict[str, Any]] = []
        runtime_path = output / "task_runtimes.csv"
        if runtime_path.exists():
            runtime_records = pd.read_csv(runtime_path).to_dict(
                orient="records"
            )
        runtime_keys = {
            (int(row["dimension"]), str(row["method"]), int(row["seed"]))
            for row in runtime_records
        }
        for index, (dimension, method, parameters) in enumerate(
            tasks, start=1
        ):
            name = task_name(method, parameters, dimension, args.seed)
            path = partials / f"{name}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                validate_partial(
                    frame,
                    method,
                    parameters,
                    dimension,
                    args.seed,
                    target_work,
                )
                print(f"[{index}/{len(tasks)}] resume {name}", flush=True)
            else:
                print(f"[{index}/{len(tasks)}] run {name}", flush=True)
                cfg = copy.deepcopy(base_cfg)
                cfg["problem"]["d"] = int(dimension)
                start = time.perf_counter()
                frame = run_one(
                    cfg,
                    method,
                    parameters,
                    args.seed,
                    target_work,
                    worker_count,
                    device,
                )
                elapsed = float(time.perf_counter() - start)
                frame["dimension"] = int(dimension)
                frame["seed_role"] = "dimension_calibration"
                frame["calibration_seed"] = int(args.seed)
                validate_partial(
                    frame,
                    method,
                    parameters,
                    dimension,
                    args.seed,
                    target_work,
                )
                atomic_csv(frame, path)
                key = (dimension, method, args.seed)
                if key not in runtime_keys:
                    runtime_records.append(
                        {
                            "dimension": int(dimension),
                            "method": method,
                            "seed": int(args.seed),
                            "runtime_seconds": elapsed,
                            "rows": int(len(frame)),
                            "final_depth": int(
                                frame.sort_values("depth")["depth"].iloc[-1]
                            ),
                            "final_work": int(
                                frame.sort_values("depth")[
                                    "total_work"
                                ].iloc[-1]
                            ),
                        }
                    )
                    runtime_keys.add(key)
                    atomic_csv(
                        pd.DataFrame(runtime_records), runtime_path
                    )
            completed.append(frame)
            atomic_json(
                {
                    "step": "ZO-7A",
                    "status": "running",
                    "completed_tasks": index,
                    "total_tasks": len(tasks),
                    "last_dimension": int(dimension),
                    "last_method": method,
                    "calibration_seed": int(args.seed),
                },
                output / "progress.json",
            )

        results = pd.concat(completed, ignore_index=True)
        summaries: list[pd.DataFrame] = []
        for dimension, frame in results.groupby("dimension", sort=True):
            summary = summarize(
                frame, base_cfg["epsilon_scaling"]["epsilons"]
            )
            summary.insert(0, "dimension", int(dimension))
            summaries.append(summary)
        combined_summary = pd.concat(summaries, ignore_index=True)
        atomic_csv(results, output / "results.csv")
        atomic_csv(combined_summary, output / "summary.csv")
        atomic_json(
            {
                "step": "ZO-7A",
                "status": "complete",
                "completed_tasks": len(tasks),
                "total_tasks": len(tasks),
                "dimensions": dimensions,
                "methods": [method for method, _ in candidates],
                "calibration_seed": int(args.seed),
                "paper_result": False,
            },
            output / "progress.json",
        )
        print(combined_summary.to_string(index=False), flush=True)
        print(f"saved={output}", flush=True)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
