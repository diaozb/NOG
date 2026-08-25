#!/usr/bin/env python3
"""Run CPU NOG-ZO optimistic/non-optimistic ablations on frozen settings."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distributed.run_distributed_baselines import environment_record, load_config
from src.distributed.zo_refine_pilot import run_one


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TARGET_WORK = 983_040
PILOT_SEEDS = list(range(700, 705))
FORMAL_SEEDS = list(range(720, 740))
METHODS = ["NOG-ZO", "NOG-ZO-NONOPT"]
DATASETS = ["synthetic_maxsinl1", "a9a", "ijcnn1"]
_THREADS_CONFIGURED = False

CONFIGS = {
    "synthetic_maxsinl1": ROOT / "zo_experiments/pilot_inputs/config_base.yaml",
    "a9a": ROOT / "configs/distributed_zo_batch_svm_a9a.yaml",
    "ijcnn1": ROOT / "configs/distributed_zo_batch_svm_ijcnn1.yaml",
}
PARAMETERS = {
    "synthetic_maxsinl1": {"M": 2, "eta": 0.01, "smooth_B": 8},
    "a9a": {"M": 1, "eta": 9e-5, "smooth_B": 1, "data_B_total": 64},
    "ijcnn1": {"M": 1, "eta": 1e-4, "smooth_B": 1, "data_B_total": 64},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_label(dataset: str) -> str:
    return "SyntheticMaxSinL1" if dataset == "synthetic_maxsinl1" else dataset


def make_config(dataset: str) -> dict[str, Any]:
    cfg = load_config(str(CONFIGS[dataset]))
    cfg["run"]["device"] = "cpu"
    cfg["run"]["pilot_selection_complete"] = True
    cfg["pilot"]["refine"]["target_total_work"] = TARGET_WORK
    # The frozen config's dense-evaluation interval is already the latest
    # protocol value: 4 for SyntheticMaxSinL1 and 96 for SVM.
    return cfg


def task_path(stage: str, dataset: str, method: str, seed: int) -> Path:
    safe = method.replace("+", "plus")
    return OUT / "raw" / stage / dataset / f"{safe}__seed-{seed}.csv"


def run_task(task: tuple[str, str, int, str]) -> dict[str, Any]:
    stage, dataset, seed, device = task
    path = task_path(stage, dataset, "NOG-ZO", seed)
    # Method is encoded separately in the task tuple below; this placeholder
    # is replaced by run_task_with_method for clarity under ProcessPool.
    raise RuntimeError("internal task dispatch error")


def run_task_with_method(task: tuple[str, str, str, int, str]) -> dict[str, Any]:
    stage, dataset, method, seed, device = task
    path = task_path(stage, dataset, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if path.exists():
        frame = pd.read_csv(path)
        return {"stage": stage, "dataset": dataset, "method": method, "seed": seed, "status": "resumed", "path": str(path), "rows": len(frame), "seconds": 0.0}
    try:
        global _THREADS_CONFIGURED
        if not _THREADS_CONFIGURED:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # A worker may have initialized the inter-op pool while
                # importing torch; its existing setting is still safe.
                pass
            _THREADS_CONFIGURED = True
        cfg = make_config(dataset)
        frame = run_one(
            cfg,
            method,
            PARAMETERS[dataset],
            seed,
            TARGET_WORK,
            int(cfg["distributed"]["comparison_worker"]),
            device,
        )
        frame["dataset"] = dataset_label(dataset)
        frame["ablation_stage"] = stage
        frame["ablation_method"] = method
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        return {"stage": stage, "dataset": dataset, "method": method, "seed": seed, "status": "completed", "path": str(path), "rows": len(frame), "seconds": time.time() - started}
    except Exception as exc:  # pragma: no cover - captured into completion audit
        return {"stage": stage, "dataset": dataset, "method": method, "seed": seed, "status": "failed", "path": str(path), "rows": 0, "seconds": time.time() - started, "error": repr(exc)}


def run_stage(stage: str, seeds: list[int], concurrency: int) -> list[dict[str, Any]]:
    tasks = [(stage, dataset, method, seed, "cpu") for dataset in DATASETS for method in METHODS for seed in seeds]
    print(f"stage={stage} tasks={len(tasks)} concurrency={concurrency} device=cpu", flush=True)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_task_with_method, task) for task in tasks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(tasks)}] {record['status']} {record['dataset']} {record['method']} seed={record['seed']} seconds={record['seconds']:.1f}", flush=True)
    records.sort(key=lambda r: (r["dataset"], r["method"], r["seed"]))
    (OUT / f"completion_{stage}.json").write_text(json.dumps({"status": "complete" if not any(r["status"] == "failed" for r in records) else "failed", "stage": stage, "records": records}, indent=2) + "\n", encoding="utf-8")
    return records


def write_protocol() -> None:
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_runs",
        "device": "cpu",
        "target_total_work": TARGET_WORK,
        "worker_count": 8,
        "pilot_seeds": PILOT_SEEDS,
        "formal_seeds": FORMAL_SEEDS,
        "seed_sets_disjoint": True,
        "methods": METHODS,
        "datasets": DATASETS,
        "parameters_frozen_from_latest_results": PARAMETERS,
        "source_configs": {k: str(v.relative_to(ROOT)) for k, v in CONFIGS.items()},
        "source_config_sha256": {k: sha256(v) for k, v in CONFIGS.items()},
        "algorithm_change": "NOG-ZO-NONOPT replaces update-2*eta*g_tm1+eta*g_tm2 with update-eta*g_tm1; all oracle streams, initial calls, work/depth accounting and evaluation settings are otherwise unchanged.",
        "latest_baseline_sources": {
            "synthetic_maxsinl1": "outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040",
            "a9a_ijcnn1": "results/advisor_cpu_batch_retuned_svm/merged/formal_trajectories.csv",
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    for dataset, config in CONFIGS.items():
        (OUT / "configs").mkdir(exist_ok=True)
        shutil.copy2(config, OUT / "configs" / f"{dataset}.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "formal", "both"], default="both")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    write_protocol()
    if args.smoke:
        records = run_stage("smoke", [700], min(args.concurrency, 3))
        print(json.dumps(records, indent=2), flush=True)
        return
    if args.stage in {"pilot", "both"}:
        run_stage("pilot", PILOT_SEEDS, args.concurrency)
    if args.stage in {"formal", "both"}:
        run_stage("formal", FORMAL_SEEDS, args.concurrency)
    (OUT / "environment.json").write_text(json.dumps(environment_record("cpu"), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
