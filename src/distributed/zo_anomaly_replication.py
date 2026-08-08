"""Resumable Step ZO-6B replication on reserved anomaly seeds.

The runner uses the four frozen formal configurations at the original work
budget.  Its output is diagnostic and is never merged into the formal run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
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
    / "outputs/distributed_zo/zo_theory_validation/diagnostic"
    / "anomaly_seeds_fixed_work_983040"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_partial(
    frame: pd.DataFrame,
    method: str,
    parameters: dict[str, Any],
    seed: int,
    target_work: int,
) -> None:
    if set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"Partial method mismatch for {method}/seed-{seed}.")
    if set(frame["formal_seed"].astype(int)) != {seed}:
        raise ValueError(f"Partial seed mismatch for {method}/seed-{seed}.")
    expected_parameters = json.dumps(parameters, sort_keys=True)
    if set(frame["candidate_parameters"].astype(str)) != {
        expected_parameters
    }:
        raise ValueError(
            f"Partial parameter mismatch for {method}/seed-{seed}."
        )
    final_work = int(frame.sort_values("depth")["total_work"].iloc[-1])
    if final_work != target_work:
        raise ValueError(
            f"Partial work mismatch for {method}/seed-{seed}: "
            f"{final_work}!={target_work}."
        )


def acquire_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "runner.lock"
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"Another Step ZO-6B runner holds {lock_path}."
        ) from error
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def main() -> None:
    args = parse_args()
    freeze_path = Path(args.freeze).resolve()
    freeze = load_and_verify_freeze(freeze_path)
    cfg = load_config(ROOT / freeze["base_config"])
    cfg["run"]["pilot_selection_complete"] = True
    cfg["pilot"]["refine"]["target_total_work"] = int(
        freeze["target_total_work"]
    )
    cfg["pilot"]["refine"]["eval_every"] = int(freeze["eval_every"])
    cfg["oracle"]["eval_smooth_B"] = int(freeze["eval_smooth_B"])
    cfg["oracle"]["eval_data_B"] = int(freeze["eval_data_B"])

    reserved = [int(value) for value in cfg["run"]["anomaly_seeds"]]
    seeds = (
        reserved
        if args.seeds is None
        else [int(value) for value in args.seeds.split(",") if value.strip()]
    )
    if not set(seeds).issubset(reserved):
        raise ValueError("Seeds must be reserved anomaly seeds.")
    prior = {
        int(value)
        for key in ["pilot_seeds", "formal_seeds"]
        for value in freeze[key]
    }
    if set(seeds).intersection(prior):
        raise ValueError("Anomaly seeds overlap pilot or formal seeds.")

    candidates = [
        (entry["method"], entry["parameters"])
        for entry in freeze["selected_candidates"]
    ]
    target_work = int(freeze["target_total_work"])
    worker_count = int(freeze["worker_count"])
    device = get_device(cfg["run"].get("device", "auto"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    tasks = [
        (method, parameters, seed)
        for method, parameters in candidates
        for seed in seeds
    ]
    print(
        f"step=ZO-6B device={device} tasks={len(tasks)} "
        f"seeds={seeds} target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for method, parameters, seed in tasks:
            print(f"{candidate_id(method, parameters)} seed={seed}")
        return

    lock_handle = acquire_lock(output)
    try:
        partials.mkdir(parents=True, exist_ok=True)
        shutil.copy2(freeze_path, output / "frozen_parameters.json")
        save_yaml(cfg, output / "config_frozen.yaml")
        atomic_json(environment_record(device), output / "environment.json")
        atomic_json(
            {
                "step": "ZO-6B",
                "role": "diagnostic_replication",
                "merge_with_formal_results": False,
                "parameter_policy": "reuse_frozen_formal_candidates",
                "budget_policy": "reuse_formal_target_total_work",
                "anomaly_seeds": seeds,
                "pilot_seeds": freeze["pilot_seeds"],
                "formal_seeds": freeze["formal_seeds"],
                "sets_pairwise_disjoint": True,
                "target_total_work": target_work,
                "worker_count": worker_count,
                "freeze_sha256": sha256(freeze_path),
            },
            output / "diagnostic_manifest.json",
        )

        completed_frames: list[pd.DataFrame] = []
        for index, (method, parameters, seed) in enumerate(tasks, start=1):
            identifier = candidate_id(method, parameters)
            path = partials / f"{identifier}__seed-{seed}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                validate_partial(
                    frame, method, parameters, seed, target_work
                )
                print(
                    f"[{index}/{len(tasks)}] resume {identifier} "
                    f"seed={seed}",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(tasks)}] run {identifier} seed={seed}",
                    flush=True,
                )
                frame = run_one(
                    cfg,
                    method,
                    parameters,
                    seed,
                    target_work,
                    worker_count,
                    device,
                )
                frame["seed_role"] = "anomaly"
                frame["diagnostic_seed"] = int(seed)
                validate_partial(
                    frame, method, parameters, seed, target_work
                )
                atomic_csv(frame, path)
            completed_frames.append(frame)
            atomic_json(
                {
                    "step": "ZO-6B",
                    "status": "running",
                    "completed_tasks": index,
                    "total_tasks": len(tasks),
                    "last_method": method,
                    "last_seed": seed,
                },
                output / "progress.json",
            )

        results = pd.concat(completed_frames, ignore_index=True)
        summary = summarize(
            results, cfg["epsilon_scaling"]["epsilons"]
        )
        atomic_csv(results, output / "results.csv")
        atomic_csv(summary, output / "summary.csv")
        atomic_json(
            {
                "step": "ZO-6B",
                "status": "complete",
                "completed_tasks": len(tasks),
                "total_tasks": len(tasks),
                "methods": [method for method, _ in candidates],
                "anomaly_seeds": seeds,
                "merge_with_formal_results": False,
            },
            output / "progress.json",
        )
        print(summary.to_string(index=False), flush=True)
        print(f"saved={output}", flush=True)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
