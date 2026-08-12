"""Step ZO-9C3: resumable fixed-parameter formal LIBSVM experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.distributed.run_distributed_baselines import (
    environment_record,
    load_config,
    save_yaml,
)
from src.distributed.zo_real_data_smoke import dataset_config
from src.distributed.zo_refine_pilot import run_one
from src.distributed.zo_range_pilot import candidate_id
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = ROOT / "zo_experiments/real_data/frozen_parameters.json"
DEFAULT_OUTPUT = ROOT / (
    "outputs/distributed_zo/zo_theory_validation/real_data/"
    "formal_fixed_work983040"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def verify_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "frozen":
        raise ValueError("Real-data parameters are not frozen.")
    if not freeze.get("selection_happened_before_formal_runs"):
        raise ValueError("Freeze does not certify pre-formal selection.")
    if not freeze.get("seed_sets_disjoint"):
        raise ValueError("Freeze does not certify disjoint seed roles.")
    if set(freeze["pilot_seeds"]).intersection(freeze["formal_seeds"]):
        raise ValueError("Pilot and formal seeds overlap.")
    for relative, expected in freeze["input_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"Frozen input hash mismatch: {relative}.")
    return freeze


def validate_frame(
    frame: pd.DataFrame,
    dataset: str,
    method: str,
    parameters: dict[str, Any],
    seed: int,
    target_work: int,
) -> None:
    label = f"{dataset}/{method}/seed-{seed}"
    if frame.empty or set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"Invalid method result: {label}.")
    if set(frame["formal_seed"].astype(int)) != {seed}:
        raise ValueError(f"Invalid seed result: {label}.")
    if set(frame["candidate_parameters"].astype(str)) != {
        json.dumps(parameters, sort_keys=True)
    }:
        raise ValueError(f"Parameter mismatch: {label}.")
    if not np.isfinite(
        frame[["objective", "stat_proxy", "train_accuracy"]].to_numpy(float)
    ).all():
        raise ValueError(f"Non-finite formal metric: {label}.")
    ordered = frame.sort_values("iteration")
    if not (ordered["total_work"].diff().dropna() > 0).all():
        raise ValueError(f"Non-increasing work: {label}.")
    if int(ordered.iloc[-1]["total_work"]) > target_work:
        raise ValueError(f"Formal work budget exceeded: {label}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_path = Path(args.freeze).resolve()
    freeze = verify_freeze(freeze_path)
    cfg = load_config(str(ROOT / freeze["base_config"]))
    cfg["run"]["pilot_selection_complete"] = True
    cfg["run"]["device"] = "cpu"
    cfg["pilot"]["refine"]["target_total_work"] = int(
        freeze["formal_target_total_work"]
    )
    cfg["pilot"]["refine"]["eval_every"] = int(freeze["eval_every"])
    cfg["oracle"]["eval_smooth_B"] = int(freeze["eval_smooth_B"])
    cfg["oracle"]["eval_data_B"] = int(freeze["eval_data_B"])
    target_work = int(freeze["formal_target_total_work"])
    workers = int(freeze["worker_count"])
    allowed = [int(value) for value in freeze["formal_seeds"]]
    seeds = allowed if args.seeds is None else [
        int(value) for value in args.seeds.split(",") if value.strip()
    ]
    if not set(seeds).issubset(allowed):
        raise ValueError("Requested seeds are not a subset of frozen formal seeds.")
    entries = freeze["selected_candidates"]
    tasks = [
        (entry["dataset"], entry["method"], entry["parameters"], seed)
        for entry in entries
        for seed in seeds
    ]
    device = get_device(cfg["run"].get("device", "cpu"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    print(
        f"step=ZO-9C3 device={device} tasks={len(tasks)} seeds={seeds} "
        f"target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for dataset, method, parameters, seed in tasks:
            print(
                f"{dataset}__{candidate_id(method, parameters)}__seed-{seed}",
                flush=True,
            )
        return

    partials.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freeze_path, output / "frozen_parameters.json")
    save_yaml(cfg, output / "config_frozen.yaml")
    atomic_json(environment_record(device), output / "environment.json")
    frames = []
    for index, (dataset, method, parameters, seed) in enumerate(tasks, start=1):
        name = f"{dataset}__{candidate_id(method, parameters)}__seed-{seed}.csv"
        path = partials / name
        if path.exists():
            print(f"[{index}/{len(tasks)}] resume {path.stem}", flush=True)
            frame = pd.read_csv(path)
        else:
            print(f"[{index}/{len(tasks)}] run {path.stem}", flush=True)
            frame = run_one(
                dataset_config(cfg, dataset), method, parameters, seed,
                target_work, workers, device,
            )
            frame["dataset"] = dataset
            frame["seed_role"] = "real_data_formal"
            atomic_csv(frame, path)
        validate_frame(frame, dataset, method, parameters, seed, target_work)
        frames.append(frame)
        atomic_json(
            {
                "status": "running",
                "completed_tasks": index,
                "total_tasks": len(tasks),
                "last_dataset": dataset,
                "last_method": method,
                "last_seed": seed,
            },
            output / "progress.json",
        )
    results = pd.concat(frames, ignore_index=True)
    final = (
        results.sort_values("iteration")
        .groupby(["dataset", "method", "formal_seed"], as_index=False)
        .tail(1)
    )
    summary = (
        final.groupby(["dataset", "method"], as_index=False)
        .agg(
            objective_mean=("objective", "mean"),
            objective_std=("objective", "std"),
            stat_proxy_mean=("stat_proxy", "mean"),
            stat_proxy_std=("stat_proxy", "std"),
            accuracy_mean=("train_accuracy", "mean"),
            accuracy_std=("train_accuracy", "std"),
            depth_mean=("depth", "mean"),
            total_work_mean=("total_work", "mean"),
            seeds=("formal_seed", "nunique"),
        )
        .sort_values(["dataset", "method"])
    )
    atomic_csv(results, output / "results.csv")
    atomic_csv(summary, output / "summary.csv")
    atomic_json(
        {
            "status": "complete",
            "completed_tasks": len(tasks),
            "total_tasks": len(tasks),
            "paper_result": True,
            "formal_seeds": seeds,
        },
        output / "progress.json",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
