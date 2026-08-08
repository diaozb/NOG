"""Step ZO-7B: resumable formal fixed-configuration dimension run."""

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

import numpy as np
import pandas as pd

from src.distributed.run_distributed_baselines import (
    environment_record,
    load_config,
    save_yaml,
)
from src.distributed.zo_dimension_calibration import task_name
from src.distributed.zo_formal import (
    atomic_csv,
    atomic_json,
    load_and_verify_freeze,
    sha256,
)
from src.distributed.zo_refine_pilot import run_one, summarize
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "zo_experiments/dimension_scaling_manifest.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/dimension"
    / "formal_fixed_params_eps003_005"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError("Dimension manifest is not frozen.")
    if not manifest.get(
        "selection_happened_before_dimension_formal_runs"
    ):
        raise ValueError("Dimension protocol was not frozen before running.")
    if manifest.get("exact_exponent_acceptance_test") is not False:
        raise ValueError("Fixed-configuration run cannot test exact powers.")
    if int(manifest["calibration_seed"]) in {
        int(value) for value in manifest["formal_seeds"]
    }:
        raise ValueError("Calibration and formal seeds overlap.")
    for relative, expected in manifest["input_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"Dimension input hash mismatch for {relative}: "
                f"{actual}!={expected}"
            )
    return manifest


def acquire_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / "runner.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("Another Step ZO-7B runner is active.") from error
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
        raise ValueError(f"Method mismatch for d={dimension}/{method}/{seed}.")
    if set(frame["formal_seed"].astype(int)) != {seed}:
        raise ValueError(f"Seed mismatch for d={dimension}/{method}/{seed}.")
    if set(frame["dimension"].astype(int)) != {dimension}:
        raise ValueError(
            f"Dimension mismatch for d={dimension}/{method}/{seed}."
        )
    expected = json.dumps(parameters, sort_keys=True)
    if set(frame["candidate_parameters"].astype(str)) != {expected}:
        raise ValueError(
            f"Parameter mismatch for d={dimension}/{method}/{seed}."
        )
    ordered = frame.sort_values("depth")
    depths = ordered["depth"].to_numpy(dtype=float)
    works = ordered["total_work"].to_numpy(dtype=float)
    if not np.all(np.diff(depths) > 0):
        raise ValueError(
            f"Non-increasing depth for d={dimension}/{method}/{seed}."
        )
    if not np.all(np.diff(works) > 0):
        raise ValueError(
            f"Non-increasing work for d={dimension}/{method}/{seed}."
        )
    if int(works[-1]) != target_work:
        raise ValueError(f"Work mismatch for d={dimension}/{method}/{seed}.")
    proxy = ordered["stat_proxy"].to_numpy(dtype=float)
    if not np.all(np.isfinite(proxy)) or np.any(proxy < 0):
        raise ValueError(f"Invalid proxy for d={dimension}/{method}/{seed}.")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    freeze_path = ROOT / manifest["frozen_candidates_source"]
    freeze = load_and_verify_freeze(freeze_path)
    base_cfg = load_config(ROOT / freeze["base_config"])
    base_cfg["run"]["pilot_selection_complete"] = True
    base_cfg["pilot"]["refine"]["target_total_work"] = int(
        manifest["target_total_work"]
    )
    base_cfg["pilot"]["refine"]["eval_every"] = int(manifest["eval_every"])
    base_cfg["oracle"]["eval_smooth_B"] = int(manifest["eval_smooth_B"])
    base_cfg["oracle"]["eval_data_B"] = int(manifest["eval_data_B"])

    dimensions = [int(value) for value in manifest["dimensions_to_run"]]
    seeds = [int(value) for value in manifest["formal_seeds"]]
    candidates = [
        (entry["method"], entry["parameters"])
        for entry in freeze["selected_candidates"]
    ]
    target_work = int(manifest["target_total_work"])
    worker_count = int(manifest["worker_count"])
    device = get_device(base_cfg["run"].get("device", "auto"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    tasks = [
        (dimension, method, parameters, seed)
        for dimension in dimensions
        for method, parameters in candidates
        for seed in seeds
    ]
    print(
        f"step=ZO-7B device={device} tasks={len(tasks)} "
        f"dimensions={dimensions} seeds={seeds[0]}..{seeds[-1]} "
        f"target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for dimension, method, parameters, seed in tasks:
            print(task_name(method, parameters, dimension, seed))
        return

    lock_handle = acquire_lock(output)
    try:
        partials.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, output / "dimension_manifest.json")
        shutil.copy2(freeze_path, output / "frozen_parameters.json")
        save_yaml(base_cfg, output / "config_base.yaml")
        atomic_json(environment_record(device), output / "environment.json")

        completed: list[pd.DataFrame] = []
        runtimes_path = output / "task_runtimes.csv"
        runtime_records: list[dict[str, Any]] = (
            pd.read_csv(runtimes_path).to_dict(orient="records")
            if runtimes_path.exists()
            else []
        )
        runtime_keys = {
            (
                int(row["dimension"]),
                str(row["method"]),
                int(row["seed"]),
            )
            for row in runtime_records
        }
        for index, (dimension, method, parameters, seed) in enumerate(
            tasks, start=1
        ):
            name = task_name(method, parameters, dimension, seed)
            path = partials / f"{name}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                validate_partial(
                    frame,
                    method,
                    parameters,
                    dimension,
                    seed,
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
                    seed,
                    target_work,
                    worker_count,
                    device,
                )
                elapsed = float(time.perf_counter() - start)
                frame["dimension"] = int(dimension)
                frame["seed_role"] = "dimension_formal"
                validate_partial(
                    frame,
                    method,
                    parameters,
                    dimension,
                    seed,
                    target_work,
                )
                atomic_csv(frame, path)
                key = (dimension, method, seed)
                if key not in runtime_keys:
                    runtime_records.append(
                        {
                            "dimension": int(dimension),
                            "method": method,
                            "seed": int(seed),
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
                        pd.DataFrame(runtime_records), runtimes_path
                    )
            completed.append(frame)
            atomic_json(
                {
                    "step": "ZO-7B",
                    "status": "running",
                    "completed_tasks": index,
                    "total_tasks": len(tasks),
                    "last_dimension": int(dimension),
                    "last_method": method,
                    "last_seed": int(seed),
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
                "step": "ZO-7B",
                "status": "complete",
                "completed_tasks": len(tasks),
                "total_tasks": len(tasks),
                "dimensions_run": dimensions,
                "dimension_100_reused": True,
                "methods": [method for method, _ in candidates],
                "formal_seeds": seeds,
                "paper_result_role": manifest["paper_result_role"],
            },
            output / "progress.json",
        )
        print(combined_summary.to_string(index=False), flush=True)
        print(f"saved={output}", flush=True)
    finally:
        lock_handle.close()


if __name__ == "__main__":
    main()
