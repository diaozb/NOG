#!/usr/bin/env python3
"""Run frozen retuned NOG-ZO formal seeds on CPU with an exact x=0 row."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distributed.common import (
    build_problem,
    evaluate_point,
    evaluation_seed,
    make_seed_bundle,
)
from src.distributed.run_distributed_baselines import environment_record, load_config, save_yaml
from src.distributed.zo_refine_pilot import apply_candidate, run_one
from src.distributed.zo_range_pilot import candidate_id

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", default=None)
    p.add_argument("--cpu-threads", type=int, default=2)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("device") != "cpu":
        raise ValueError("Freeze must be CPU and status=frozen")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seeds = [int(x) for x in (args.seeds.split(",") if args.seeds else freeze["formal_seeds"]) if str(x).strip()]
    if not set(seeds).issubset(set(freeze["formal_seeds"])):
        raise ValueError("Requested seeds are not frozen formal seeds")
    output = args.output.resolve()
    partials = output / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_parameters.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    environment = environment_record("cpu")
    environment.update({"cpu_threads": args.cpu_threads, "torch_interop_threads": torch.get_num_interop_threads(), "formal_runner": "run_retuned_nog_formal.py"})
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    frames: list[pd.DataFrame] = []
    tasks = [(entry["dataset"], entry["parameters"], seed) for entry in freeze["selected_candidates"] for seed in seeds]
    print(f"device=cpu cpu_threads={args.cpu_threads} tasks={len(tasks)} target_work={freeze['formal_target_total_work']}", flush=True)
    for index, (dataset, parameters, seed) in enumerate(tasks, 1):
        cfg = load_config(str(ROOT / freeze["config_by_dataset"][dataset]))
        cfg["run"]["pilot_selection_complete"] = True
        cfg["run"]["device"] = "cpu"
        cfg["problem"]["load_test"] = True
        cfg["pilot"]["refine"]["target_total_work"] = int(freeze["formal_target_total_work"])
        cfg["pilot"]["refine"]["eval_every"] = int(freeze["eval_every"])
        cfg["oracle"]["eval_smooth_B"] = int(freeze["eval_smooth_B"])
        cfg["oracle"]["eval_data_B"] = int(freeze["eval_data_B"])
        cid = candidate_id("NOG-ZO", parameters)
        path = partials / f"{dataset}__{cid}__seed-{seed}.csv"
        if path.exists():
            print(f"[{index}/{len(tasks)}] resume {path.stem}", flush=True)
            frame = pd.read_csv(path)
        else:
            print(f"[{index}/{len(tasks)}] run {path.stem}", flush=True)
            frame = run_one(cfg, "NOG-ZO", parameters, seed, int(freeze["formal_target_total_work"]), int(freeze["worker_count"]), "cpu")
            frame["dataset"] = dataset
            # Exact initialization evaluation at x=0, using the same fixed evaluation bank.
            problem = build_problem(cfg, "cpu", 100000 + seed)
            bundle = make_seed_bundle(seed, "NOG-ZO", int(freeze["worker_count"]))
            metrics, eval_calls = evaluate_point(problem, torch.zeros(problem.d, device="cpu"), cfg, evaluation_seed(bundle, 0, "fixed_bank"))
            init = {col: float("nan") for col in frame.columns}
            init.update({
                "dataset": dataset,
                "method": "NOG-ZO", "base_method": "NOG", "iteration": 0, "round": 0, "block_id": 0,
                "worker_count": int(freeze["worker_count"]), "eval_point": "x0_exact", "delta": float(cfg["oracle"]["target_delta"]),
                "target_delta": float(cfg["oracle"]["target_delta"]), "smoothing_delta": float(cfg["nog"]["smoothing_delta"]),
                "M": int(parameters["M"]), "lr_or_eta": float(parameters["eta"]), "block_oracle_norm": float("nan"),
                "time_sec": 0.0, **bundle.as_dict(), "oracle_type": "szo", "work": 0, "total_work": 0,
                "per_worker_work_max": 0, "per_worker_work": json.dumps([0] * int(freeze["worker_count"])), "eval_work": eval_calls,
                "communication_round": 0, "depth": 0, **metrics, "evaluation_delta": float(cfg["oracle"]["evaluation_delta"]),
                "candidate_id": cid, "candidate_parameters": json.dumps(parameters, sort_keys=True),
                "candidate_rounds": int(frame["candidate_rounds"].iloc[0]), "target_total_work": int(freeze["formal_target_total_work"]),
            })
            frame = pd.concat([pd.DataFrame([init]), frame], ignore_index=True)
            frame.to_csv(path, index=False)
        if "dataset" not in frame:
            frame["dataset"] = dataset
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output / "formal_trajectories.csv", index=False)
    final = results.sort_values("iteration").groupby(["dataset", "method", "formal_seed"], as_index=False).tail(1) if "dataset" in results else results.sort_values("iteration").groupby(["method", "formal_seed"], as_index=False).tail(1)
    # Dataset is attached after each task to keep the native runner output intact.
    if "dataset" not in results:
        raise RuntimeError("Internal dataset column missing")
    final = results.sort_values("iteration").groupby(["dataset", "method", "formal_seed"], as_index=False).tail(1)
    summary = final.groupby(["dataset", "method"], as_index=False).agg(
        objective_mean=("objective", "mean"), objective_std=("objective", "std"),
        stat_proxy_mean=("stat_proxy", "mean"), stat_proxy_std=("stat_proxy", "std"),
        train_accuracy_mean=("train_accuracy", "mean"), train_accuracy_std=("train_accuracy", "std"),
        depth_mean=("depth", "mean"), total_work_mean=("total_work", "mean"), seeds=("formal_seed", "nunique"))
    summary.to_csv(output / "formal_summary.csv", index=False)
    (output / "progress.json").write_text(json.dumps({"status": "complete", "tasks": len(tasks), "seeds": seeds, "formal_target_total_work": freeze["formal_target_total_work"]}, indent=2) + "\n", encoding="utf-8")
    save_yaml({"freeze": str(args.freeze), "formal_target_total_work": freeze["formal_target_total_work"]}, output / "run_metadata.yaml")
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
