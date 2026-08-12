"""Step ZO-9C1: resumable four-method smoke on official LIBSVM data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.distributed.common import validate_experiment_config
from src.distributed.real_data import DATASETS, download_dataset, sha256
from src.distributed.run_distributed_baselines import (
    environment_record,
    load_config,
    run_selected,
    save_yaml,
)
from src.synthetic.run_synthetic import get_device


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/distributed_zo_real_data_smoke.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/real_data/smoke_fixed_params"
)


def dataset_config(base: dict[str, Any], name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    spec = DATASETS[name]
    cfg["problem"]["dataset"] = name
    cfg["problem"]["d"] = int(spec["d"])
    cfg["problem"]["n_data"] = int(spec["n"])
    cfg["problem"]["lam"] = 1.0e-5 / int(spec["n"])
    return cfg


def validate_frame(frame: pd.DataFrame, method: str, dataset: str) -> None:
    label = f"{dataset}/{method}"
    if frame.empty:
        raise ValueError(f"Empty smoke result: {label}.")
    if set(frame["method"].astype(str)) != {method}:
        raise ValueError(f"Method mismatch: {label}.")
    if not np.isfinite(
        frame[["objective", "stat_proxy", "train_accuracy"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Non-finite metric: {label}.")
    if not frame["train_accuracy"].between(0, 1).all():
        raise ValueError(f"Invalid accuracy: {label}.")
    ordered = frame.sort_values("depth")
    if not (ordered["depth"].diff().dropna() > 0).all():
        raise ValueError(f"Non-increasing depth: {label}.")
    if not (ordered["total_work"].diff().dropna() > 0).all():
        raise ValueError(f"Non-increasing work: {label}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    output = Path(args.output)
    methods = [str(value) for value in base["methods"]["szo"]]
    datasets = ["a9a", "ijcnn1"]
    seed = int(base["run"]["pilot_seeds"][0])
    workers = int(base["distributed"]["comparison_worker"])
    device = get_device(base["run"].get("device", "auto"))
    tasks = [(dataset, method) for dataset in datasets for method in methods]
    print(
        f"step=ZO-9C1 device={device} tasks={len(tasks)} seed={seed} "
        f"workers={workers}",
        flush=True,
    )
    if args.dry_run:
        for dataset, method in tasks:
            print(f"  dataset={dataset} method={method}", flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    partials = output / "partials"
    partials.mkdir(exist_ok=True)
    save_yaml(base, output / "config_base.yaml")
    (output / "environment.json").write_text(
        json.dumps(environment_record(device), indent=2), encoding="utf-8"
    )
    data_manifest = {}
    for dataset in datasets:
        path = download_dataset(dataset, Path(base["problem"]["data_root"]))
        data_manifest[dataset] = {
            **DATASETS[dataset],
            "path": str(path),
            "observed_sha256": sha256(path),
        }
    (output / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2), encoding="utf-8"
    )

    frames = []
    for index, (dataset, method) in enumerate(tasks, start=1):
        path = partials / f"{dataset}__{method.replace('+', 'plus')}__seed{seed}.csv"
        if path.exists():
            print(f"[{index}/{len(tasks)}] resume {dataset}/{method}", flush=True)
            frame = pd.read_csv(path)
        else:
            print(f"[{index}/{len(tasks)}] run {dataset}/{method}", flush=True)
            cfg = dataset_config(base, dataset)
            validate_experiment_config(cfg)
            frame = run_selected(cfg, [method], [seed], [workers], device)
            frame["dataset"] = dataset
            frame["seed_role"] = "real_data_smoke"
            frame.to_csv(path, index=False)
        validate_frame(frame, method, dataset)
        frames.append(frame)
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_tasks": index,
                    "total_tasks": len(tasks),
                    "last_dataset": dataset,
                    "last_method": method,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output / "results.csv", index=False)
    summary = (
        results.sort_values("depth")
        .groupby(["dataset", "method"], as_index=False)
        .tail(1)[
            [
                "dataset",
                "method",
                "objective",
                "stat_proxy",
                "train_accuracy",
                "depth",
                "total_work",
                "per_worker_work_max",
            ]
        ]
        .sort_values(["dataset", "method"])
    )
    summary.to_csv(output / "summary.csv", index=False)
    (output / "progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_tasks": len(tasks),
                "total_tasks": len(tasks),
                "paper_result": False,
                "purpose": "compatibility_and_numerical_stability_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
