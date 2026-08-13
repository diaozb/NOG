"""Step ZO-8B: resumable formal fixed-configuration worker scaling."""

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
from src.distributed.zo_refine_pilot import run_one, summarize
from src.distributed.zo_worker_calibration import (
    evaluation_interval,
    task_name,
    validate_partial,
)
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "zo_experiments/worker_scaling_manifest.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/worker"
    / "formal_fixed_params_eps005"
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
        raise ValueError("Worker formal manifest is not frozen.")
    if not manifest.get("selection_happened_before_worker_formal_runs"):
        raise ValueError("Worker formal protocol was not pre-frozen.")
    if int(manifest["calibration_seed"]) in {
        int(value) for value in manifest["formal_seeds"]
    }:
        raise ValueError("Worker calibration and formal seeds overlap.")
    if not manifest.get("formal_and_calibration_seeds_disjoint"):
        raise ValueError("Formal/calibration seed disjointness is not asserted.")
    if manifest.get("primary_epsilons") != [0.05]:
        raise ValueError("Unexpected worker-scaling primary endpoint.")
    for relative, expected in manifest["input_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"Worker formal input hash mismatch for {relative}: "
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
        raise RuntimeError("Another Step ZO-8B runner is active.") from error
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


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

    workers = [int(value) for value in manifest["workers_to_run"]]
    seeds = [int(value) for value in manifest["formal_seeds"]]
    candidates = [
        (entry["method"], entry["parameters"])
        for entry in freeze["selected_candidates"]
    ]
    target_work = int(manifest["target_total_work"])
    device = get_device(base_cfg["run"].get("device", "auto"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    tasks = [
        (worker_count, method, parameters, seed)
        for worker_count in workers
        for method, parameters in candidates
        for seed in seeds
    ]
    print(
        f"step=ZO-8B device={device} tasks={len(tasks)} workers={workers} "
        f"seeds={seeds[0]}..{seeds[-1]} target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for worker_count, method, parameters, seed in tasks:
            print(
                f"{task_name(method, parameters, worker_count, seed)} "
                f"eval_every={evaluation_interval(manifest, method, worker_count)}",
                flush=True,
            )
        return

    lock_handle = acquire_lock(output)
    try:
        partials.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, output / "worker_scaling_manifest.json")
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
        for index, (worker_count, method, parameters, seed) in enumerate(
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
                frame["seed_role"] = "worker_formal"
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
                    "step": "ZO-8B",
                    "status": "running",
                    "completed_tasks": index,
                    "total_tasks": len(tasks),
                    "last_worker_count": worker_count,
                    "last_method": method,
                    "last_seed": seed,
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
                "step": "ZO-8B",
                "status": "complete",
                "completed_tasks": len(tasks),
                "total_tasks": len(tasks),
                "workers_run": workers,
                "reference_worker_reused": True,
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
