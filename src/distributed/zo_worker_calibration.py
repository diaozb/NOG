"""Step ZO-8A: resumable fixed-configuration worker calibration."""

from __future__ import annotations

import argparse
import ast
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
from src.distributed.zo_formal import (
    atomic_csv,
    atomic_json,
    load_and_verify_freeze,
    sha256,
)
from src.distributed.zo_range_pilot import candidate_id
from src.distributed.zo_refine_pilot import (
    apply_candidate,
    run_one,
    summarize,
    training_work_for_rounds,
)
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "zo_experiments/worker_scaling_calibration_manifest.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/worker"
    / "calibration_fixed_params_work983040"
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
        raise ValueError("Worker calibration manifest is not frozen.")
    if not manifest.get("selection_happened_before_worker_calibration_runs"):
        raise ValueError("Worker calibration protocol was not pre-frozen.")
    if manifest.get("paper_result") is not False:
        raise ValueError("Calibration output must not be labeled as a paper result.")
    seed_roles = [
        set(int(value) for value in manifest[key])
        for key in ["pilot_seeds", "formal_seeds", "anomaly_seeds"]
    ]
    seed_roles.extend(
        [
            {int(manifest["dimension_calibration_seed"])},
            {int(manifest["calibration_seed"])},
        ]
    )
    if any(
        left.intersection(right)
        for index, left in enumerate(seed_roles)
        for right in seed_roles[index + 1 :]
    ):
        raise ValueError("Worker calibration seed roles overlap.")
    if not manifest.get("all_seed_roles_disjoint"):
        raise ValueError("Seed-role disjointness is not asserted.")
    for relative, expected in manifest["input_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"Worker calibration input hash mismatch for {relative}: "
                f"{actual}!={expected}"
            )
    return manifest


def evaluation_interval(
    manifest: dict[str, Any], method: str, worker_count: int
) -> int:
    policy = manifest["evaluation_interval_policy"]
    try:
        interval = int(policy[method][str(int(worker_count))])
    except KeyError as error:
        raise ValueError(
            f"No frozen evaluation interval for {method}/m={worker_count}."
        ) from error
    if interval < 1:
        raise ValueError("Evaluation intervals must be positive.")
    return interval


def acquire_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    handle = (output / "runner.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError("Another Step ZO-8A runner is active.") from error
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def task_name(
    method: str, parameters: dict[str, Any], worker_count: int, seed: int
) -> str:
    return (
        f"workers-{worker_count}__{candidate_id(method, parameters)}"
        f"__seed-{seed}"
    )


def _parse_worker_calls(value: Any) -> list[int]:
    parsed = value if isinstance(value, list) else ast.literal_eval(str(value))
    return [int(item) for item in parsed]


def validate_partial(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    method: str,
    parameters: dict[str, Any],
    worker_count: int,
    seed: int,
    target_work: int,
    eval_interval: int,
) -> None:
    label = f"{method}/m={worker_count}/seed={seed}"
    if frame.empty:
        raise ValueError(f"Empty partial for {label}.")
    if set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"Method mismatch for {label}.")
    if set(frame["formal_seed"].astype(int)) != {seed}:
        raise ValueError(f"Seed mismatch for {label}.")
    if set(frame["worker_count"].astype(int)) != {worker_count}:
        raise ValueError(f"Worker mismatch for {label}.")
    if set(frame["worker_scaling_count"].astype(int)) != {worker_count}:
        raise ValueError(f"Worker-scaling metadata mismatch for {label}.")
    if set(frame["evaluation_interval_rounds"].astype(int)) != {eval_interval}:
        raise ValueError(f"Evaluation interval mismatch for {label}.")
    expected_parameters = json.dumps(parameters, sort_keys=True)
    if set(frame["candidate_parameters"].astype(str)) != {expected_parameters}:
        raise ValueError(f"Parameter mismatch for {label}.")

    ordered = frame.sort_values("depth")
    depths = ordered["depth"].to_numpy(dtype=float)
    works = ordered["total_work"].to_numpy(dtype=float)
    if not np.all(np.diff(depths) > 0):
        raise ValueError(f"Non-increasing depth for {label}.")
    if not np.all(np.diff(works) > 0):
        raise ValueError(f"Non-increasing work for {label}.")
    proxy = ordered["stat_proxy"].to_numpy(dtype=float)
    if not np.all(np.isfinite(proxy)) or np.any(proxy < 0):
        raise ValueError(f"Invalid stationarity proxy for {label}.")

    rounds = int(ordered["candidate_rounds"].iloc[-1])
    candidate_cfg = copy.deepcopy(cfg)
    apply_candidate(candidate_cfg, method, parameters)
    expected_work = training_work_for_rounds(
        candidate_cfg, method, rounds, worker_count
    )
    final_work = int(works[-1])
    if final_work != expected_work or final_work > target_work:
        raise ValueError(
            f"Work mismatch for {label}: {final_work}!={expected_work} "
            f"or exceeds {target_work}."
        )

    for row in ordered.itertuples(index=False):
        per_worker = _parse_worker_calls(row.per_worker_work)
        if len(per_worker) != worker_count:
            raise ValueError(f"Per-worker vector length mismatch for {label}.")
        if sum(per_worker) != int(row.total_work):
            raise ValueError(f"Per-worker sum mismatch for {label}.")
        if max(per_worker) != int(row.per_worker_work_max):
            raise ValueError(f"Per-worker maximum mismatch for {label}.")


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
    base_cfg["oracle"]["eval_smooth_B"] = int(manifest["eval_smooth_B"])
    base_cfg["oracle"]["eval_data_B"] = int(manifest["eval_data_B"])
    base_cfg["problem"]["d"] = int(manifest["dimension"])

    workers = [int(value) for value in manifest["workers"]]
    seed = int(manifest["calibration_seed"])
    candidates = [
        (entry["method"], entry["parameters"])
        for entry in freeze["selected_candidates"]
    ]
    target_work = int(manifest["target_total_work"])
    device = get_device(base_cfg["run"].get("device", "auto"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    tasks = [
        (worker_count, method, parameters)
        for worker_count in workers
        for method, parameters in candidates
    ]
    print(
        f"step=ZO-8A device={device} tasks={len(tasks)} workers={workers} "
        f"seed={seed} target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for worker_count, method, parameters in tasks:
            interval = evaluation_interval(manifest, method, worker_count)
            print(
                f"{task_name(method, parameters, worker_count, seed)} "
                f"eval_every={interval}",
                flush=True,
            )
        return

    lock_handle = acquire_lock(output)
    try:
        partials.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, output / "worker_calibration_manifest.json")
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
            (int(row["worker_count"]), str(row["method"]), int(row["seed"]))
            for row in runtime_records
        }
        for index, (worker_count, method, parameters) in enumerate(
            tasks, start=1
        ):
            interval = evaluation_interval(manifest, method, worker_count)
            name = task_name(method, parameters, worker_count, seed)
            path = partials / f"{name}.csv"
            cfg = copy.deepcopy(base_cfg)
            cfg["pilot"]["refine"]["eval_every"] = interval
            if path.exists():
                frame = pd.read_csv(path)
                validate_partial(
                    frame,
                    cfg,
                    method,
                    parameters,
                    worker_count,
                    seed,
                    target_work,
                    interval,
                )
                print(f"[{index}/{len(tasks)}] resume {name}", flush=True)
            else:
                print(
                    f"[{index}/{len(tasks)}] run {name} "
                    f"eval_every={interval}",
                    flush=True,
                )
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
                frame["worker_scaling_count"] = int(worker_count)
                frame["seed_role"] = "worker_calibration"
                frame["calibration_seed"] = seed
                frame["evaluation_interval_rounds"] = interval
                validate_partial(
                    frame,
                    cfg,
                    method,
                    parameters,
                    worker_count,
                    seed,
                    target_work,
                    interval,
                )
                atomic_csv(frame, path)
                key = (worker_count, method, seed)
                if key not in runtime_keys:
                    ordered = frame.sort_values("depth")
                    runtime_records.append(
                        {
                            "worker_count": worker_count,
                            "method": method,
                            "seed": seed,
                            "evaluation_interval_rounds": interval,
                            "runtime_seconds": elapsed,
                            "rows": len(frame),
                            "final_depth": int(ordered["depth"].iloc[-1]),
                            "final_work": int(ordered["total_work"].iloc[-1]),
                            "final_per_worker_work_max": int(
                                ordered["per_worker_work_max"].iloc[-1]
                            ),
                        }
                    )
                    runtime_keys.add(key)
                    atomic_csv(pd.DataFrame(runtime_records), runtimes_path)
            completed.append(frame)
            atomic_json(
                {
                    "step": "ZO-8A",
                    "status": "running",
                    "completed_tasks": index,
                    "total_tasks": len(tasks),
                    "last_worker_count": worker_count,
                    "last_method": method,
                    "calibration_seed": seed,
                },
                output / "progress.json",
            )

        results = pd.concat(completed, ignore_index=True)
        summaries: list[pd.DataFrame] = []
        for worker_count, frame in results.groupby(
            "worker_scaling_count", sort=True
        ):
            summary = summarize(frame, base_cfg["epsilon_scaling"]["epsilons"])
            summary.insert(0, "worker_count", int(worker_count))
            summaries.append(summary)
        combined_summary = pd.concat(summaries, ignore_index=True)
        atomic_csv(results, output / "results.csv")
        atomic_csv(combined_summary, output / "summary.csv")
        atomic_json(
            {
                "step": "ZO-8A",
                "status": "complete",
                "completed_tasks": len(tasks),
                "total_tasks": len(tasks),
                "workers": workers,
                "methods": [method for method, _ in candidates],
                "calibration_seed": seed,
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
