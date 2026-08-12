"""Step ZO-9C2: preregistered, resumable calibration on LIBSVM data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.distributed.real_data import DATASETS, download_dataset, sha256
from src.distributed.run_distributed_baselines import (
    environment_record,
    load_config,
    save_yaml,
)
from src.distributed.zo_real_data_smoke import dataset_config
from src.distributed.zo_refine_pilot import refinement_candidates, run_one
from src.distributed.zo_range_pilot import candidate_id
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/distributed_zo_real_data_calibration.yaml"
DEFAULT_OUTPUT = ROOT / (
    "outputs/distributed_zo/zo_theory_validation/real_data/"
    "calibration_work98304"
)


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def safe_name(
    dataset: str, method: str, parameters: dict[str, Any], seed: int
) -> str:
    return f"{dataset}__{candidate_id(method, parameters)}__seed-{seed}.csv"


def validate_task(frame: pd.DataFrame, dataset: str, method: str, seed: int) -> None:
    label = f"{dataset}/{method}/seed-{seed}"
    if frame.empty:
        raise ValueError(f"Empty calibration result: {label}.")
    if set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"Method mismatch: {label}.")
    if set(frame["formal_seed"].astype(int)) != {int(seed)}:
        raise ValueError(f"Seed mismatch: {label}.")
    metrics = frame[["objective", "stat_proxy", "train_accuracy"]].to_numpy(float)
    if not np.isfinite(metrics).all():
        raise ValueError(f"Non-finite calibration metric: {label}.")
    ordered = frame.sort_values("iteration")
    if not (ordered["total_work"].diff().dropna() > 0).all():
        raise ValueError(f"Non-increasing training work: {label}.")
    if int(ordered.iloc[-1]["total_work"]) > int(
        ordered.iloc[-1]["target_total_work"]
    ):
        raise ValueError(f"Training-work budget exceeded: {label}.")


def final_rows(frames: list[pd.DataFrame], dataset: str) -> pd.DataFrame:
    results = pd.concat(frames, ignore_index=True)
    final = (
        results.sort_values("iteration")
        .groupby(
            ["method", "candidate_id", "candidate_parameters", "formal_seed"],
            as_index=False,
        )
        .tail(1)
        .copy()
    )
    final["dataset"] = dataset
    return final


def candidate_summary(final: pd.DataFrame) -> pd.DataFrame:
    return (
        final.groupby(
            ["dataset", "method", "candidate_id", "candidate_parameters"],
            as_index=False,
        )
        .agg(
            final_objective_mean=("objective", "mean"),
            final_objective_std=("objective", "std"),
            final_stat_proxy_mean=("stat_proxy", "mean"),
            final_stat_proxy_std=("stat_proxy", "std"),
            final_accuracy_mean=("train_accuracy", "mean"),
            final_depth_mean=("depth", "mean"),
            final_total_work_mean=("total_work", "mean"),
            candidate_rounds=("candidate_rounds", "first"),
            seed_count=("formal_seed", "nunique"),
        )
        .sort_values(
            ["dataset", "method", "final_objective_mean", "final_stat_proxy_mean"]
        )
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--stage", choices=["broad", "replicate", "all"], default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output = Path(args.output)
    partials = output / "partials"
    datasets = ["a9a", "ijcnn1"]
    candidates = refinement_candidates(cfg)
    pilot_seeds = [int(value) for value in cfg["run"]["pilot_seeds"]]
    broad_seed = int(cfg["pilot"]["broad_seed"])
    if broad_seed != pilot_seeds[0] or len(pilot_seeds) < 3:
        raise ValueError("Expected broad seed first and at least three pilot seeds.")
    formal_seeds = {int(value) for value in cfg["run"]["formal_seeds"]}
    if set(pilot_seeds).intersection(formal_seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    target_work = int(cfg["pilot"]["refine"]["target_total_work"])
    workers = int(cfg["distributed"]["comparison_worker"])
    top_k = int(cfg["pilot"]["advance_per_dataset_method"])
    device = get_device(cfg["run"].get("device", "auto"))

    broad_tasks = [
        (dataset, method, parameters, broad_seed)
        for dataset in datasets
        for method, parameters in candidates
    ]
    print(
        f"step=ZO-9C2 stage={args.stage} device={device} "
        f"broad_tasks={len(broad_tasks)} target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for dataset, method, parameters, seed in broad_tasks:
            print(
                f"{safe_name(dataset, method, parameters, seed)} "
                f"max_work={target_work} workers={workers}",
                flush=True,
            )
        return

    partials.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, output / "config_preregistered.yaml")
    atomic_json(environment_record(device), output / "environment.json")
    manifest = {}
    for dataset in datasets:
        data_path = download_dataset(dataset, Path(cfg["problem"]["data_root"]))
        manifest[dataset] = {
            **DATASETS[dataset],
            "path": str(data_path),
            "observed_sha256": sha256(data_path),
        }
    atomic_json(manifest, output / "data_manifest.json")

    broad_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in datasets}
    for index, (dataset, method, parameters, seed) in enumerate(broad_tasks, start=1):
        path = partials / safe_name(dataset, method, parameters, seed)
        if path.exists():
            print(f"[broad {index}/{len(broad_tasks)}] resume {path.stem}", flush=True)
            frame = pd.read_csv(path)
        else:
            print(f"[broad {index}/{len(broad_tasks)}] run {path.stem}", flush=True)
            frame = run_one(
                dataset_config(cfg, dataset),
                method,
                parameters,
                seed,
                target_work,
                workers,
                device,
            )
            frame["dataset"] = dataset
            frame["seed_role"] = "real_data_calibration"
            atomic_csv(frame, path)
        validate_task(frame, dataset, method, seed)
        broad_frames[dataset].append(frame)
        atomic_json(
            {
                "status": "running_broad",
                "completed_broad_tasks": index,
                "total_broad_tasks": len(broad_tasks),
            },
            output / "progress.json",
        )

    broad_final = pd.concat(
        [final_rows(broad_frames[name], name) for name in datasets],
        ignore_index=True,
    )
    broad_summary = candidate_summary(broad_final)
    atomic_csv(broad_summary, output / "broad_summary.csv")
    advanced = (
        broad_summary.groupby(["dataset", "method"], group_keys=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    atomic_csv(advanced, output / "advanced_candidates.csv")
    if args.stage == "broad":
        atomic_json(
            {"status": "broad_complete", "advanced_candidates": len(advanced)},
            output / "progress.json",
        )
        print(advanced.to_string(index=False), flush=True)
        return

    replicate_tasks = []
    for row in advanced.itertuples(index=False):
        parameters = json.loads(row.candidate_parameters)
        for seed in pilot_seeds[1:]:
            replicate_tasks.append((row.dataset, row.method, parameters, seed))
    combined_frames = [frame for group in broad_frames.values() for frame in group]
    for index, (dataset, method, parameters, seed) in enumerate(
        replicate_tasks, start=1
    ):
        path = partials / safe_name(dataset, method, parameters, seed)
        if path.exists():
            print(
                f"[replicate {index}/{len(replicate_tasks)}] resume {path.stem}",
                flush=True,
            )
            frame = pd.read_csv(path)
        else:
            print(
                f"[replicate {index}/{len(replicate_tasks)}] run {path.stem}",
                flush=True,
            )
            frame = run_one(
                dataset_config(cfg, dataset),
                method,
                parameters,
                seed,
                target_work,
                workers,
                device,
            )
            frame["dataset"] = dataset
            frame["seed_role"] = "real_data_calibration"
            atomic_csv(frame, path)
        validate_task(frame, dataset, method, seed)
        combined_frames.append(frame)
        atomic_json(
            {
                "status": "running_replicates",
                "completed_replicate_tasks": index,
                "total_replicate_tasks": len(replicate_tasks),
            },
            output / "progress.json",
        )

    results = pd.concat(combined_frames, ignore_index=True)
    advanced_keys = set(
        zip(advanced["dataset"].astype(str), advanced["candidate_id"].astype(str))
    )
    keep = [
        (str(dataset), str(identifier)) in advanced_keys
        for dataset, identifier in zip(results["dataset"], results["candidate_id"])
    ]
    advanced_results = results[np.asarray(keep, dtype=bool)]
    final = (
        advanced_results.sort_values("iteration")
        .groupby(
            [
                "dataset",
                "method",
                "candidate_id",
                "candidate_parameters",
                "formal_seed",
            ],
            as_index=False,
        )
        .tail(1)
    )
    summary = candidate_summary(final)
    selected = summary.groupby(["dataset", "method"], as_index=False).head(1)
    if not (selected["seed_count"] == len(pilot_seeds)).all():
        raise ValueError("Selected candidates do not contain every pilot seed.")
    atomic_csv(results, output / "results.csv")
    atomic_csv(summary, output / "replicated_summary.csv")
    atomic_csv(selected, output / "selected_parameters.csv")
    atomic_json(
        {
            "status": "complete",
            "paper_result": False,
            "selection_before_formal_runs": True,
            "target_total_work": target_work,
            "worker_count": workers,
            "pilot_seeds": pilot_seeds,
            "formal_seeds": sorted(formal_seeds),
            "selected_count": len(selected),
        },
        output / "progress.json",
    )
    print("selected_parameters:", flush=True)
    print(selected.to_string(index=False), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
