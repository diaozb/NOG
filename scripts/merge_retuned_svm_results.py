#!/usr/bin/env python3
"""Merge retuned NOG formal shards with unchanged baseline formal shards."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.distributed.common import build_problem, evaluate_point, evaluation_seed, make_seed_bundle  # noqa: E402
from src.distributed.run_distributed_baselines import load_config  # noqa: E402
from src.distributed.zo_real_data_smoke import dataset_config  # noqa: E402


METHODS = ("NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+")
BASELINES = METHODS[1:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--retuned-root", type=Path, required=True)
    p.add_argument("--retuned-shards", type=Path, default=None)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def x0_row(frame: pd.DataFrame, dataset: str, method: str, seed: int, cfg: dict) -> dict:
    row = {col: np.nan for col in frame.columns}
    problem = build_problem(cfg, "cpu", 100000 + seed)
    workers = int(frame["worker_count"].iloc[0])
    bundle = make_seed_bundle(seed, method, workers)
    metrics, eval_calls = evaluate_point(problem, torch.zeros(problem.d, device="cpu"), cfg, evaluation_seed(bundle, 0, "fixed_bank"))
    row.update({"dataset": dataset, "method": method, "base_method": method.replace("-ZO", "").replace("+", "+"), "iteration": 0, "round": 0, "worker_count": workers, "eval_point": "x0_exact", "time_sec": 0.0, **bundle.as_dict(), "oracle_type": "szo", "work": 0, "total_work": 0, "per_worker_work_max": 0, "per_worker_work": json.dumps([0] * workers), "eval_work": eval_calls, "communication_round": 0, "depth": 0, **metrics, "candidate_parameters": frame["candidate_parameters"].iloc[0], "candidate_id": frame["candidate_id"].iloc[0], "candidate_rounds": frame["candidate_rounds"].iloc[0], "target_total_work": frame["target_total_work"].iloc[0], "seed_role": "real_data_supplement_formal"})
    return row


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    shard_root = args.retuned_shards if args.retuned_shards is not None else args.retuned_root / "formal_shards"
    new_paths = sorted(shard_root.glob("seed-*/formal_trajectories.csv"))
    if len(new_paths) != 20:
        raise RuntimeError(f"Expected 20 retuned seed shards, found {len(new_paths)}")
    nog = pd.concat([pd.read_csv(p) for p in new_paths], ignore_index=True)
    if set(nog["method"].unique()) != {"NOG-ZO"}:
        raise RuntimeError("Retuned shards contain non-NOG methods")
    baseline_paths = sorted((args.baseline_root).glob("seed_*/partials/*.csv"))
    baseline_frames = []
    for path in baseline_paths:
        frame = pd.read_csv(path)
        if str(frame["method"].iloc[0]) in BASELINES:
            baseline_frames.append(frame)
    if len(baseline_frames) != 120:
        raise RuntimeError(f"Expected 120 baseline partials, found {len(baseline_frames)}")
    baselines = pd.concat(baseline_frames, ignore_index=True)
    # Add exact x=0 rows to unchanged baselines using the same fixed evaluation bank.
    cfg_base = load_config(str(ROOT / "configs/distributed_zo_real_data_calibration_final_boundary.yaml"))
    x0 = []
    for (dataset, method, seed), frame in baselines.groupby(["dataset", "method", "formal_seed"], sort=False):
        cfg = dataset_config(cfg_base, str(dataset))
        cfg["problem"]["load_test"] = True
        x0.append(x0_row(frame, str(dataset), str(method), int(seed), cfg))
    baselines = pd.concat([pd.DataFrame(x0), baselines], ignore_index=True, sort=False)
    merged = pd.concat([nog, baselines], ignore_index=True, sort=False)
    merged = merged.sort_values(["dataset", "method", "formal_seed", "iteration"]).reset_index(drop=True)
    expected = {(d, m, s) for d in ("a9a", "ijcnn1") for m in METHODS for s in range(20)}
    observed = {(str(d), str(m), int(s)) for d, m, s in merged[["dataset", "method", "formal_seed"]].drop_duplicates().itertuples(index=False, name=None)}
    if observed != expected:
        raise RuntimeError(f"Formal identities mismatch; missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    final = merged.groupby(["dataset", "method", "formal_seed"], as_index=False).tail(1)
    summary = final.groupby(["dataset", "method"], as_index=False).agg(objective_mean=("objective", "mean"), objective_std=("objective", "std"), stat_proxy_mean=("stat_proxy", "mean"), stat_proxy_std=("stat_proxy", "std"), train_accuracy_mean=("train_accuracy", "mean"), train_accuracy_std=("train_accuracy", "std"), test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"), depth_mean=("depth", "mean"), total_work_mean=("total_work", "mean"), seed_count=("formal_seed", "nunique"))
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "formal_trajectories.csv", index=False)
    summary.to_csv(args.output / "formal_summary.csv", index=False)
    audit = pd.DataFrame([{"dataset": d, "method": m, "seed_count": int(merged[(merged.dataset == d) & (merged.method == m)]["formal_seed"].nunique()), "source": "retuned formal shards" if m == "NOG-ZO" else "supplement_formal_cpu_v2_seed_shards", "baseline_reuse": m != "NOG-ZO"} for d in ("a9a", "ijcnn1") for m in METHODS])
    audit.to_csv(args.output / "baseline_reuse_audit.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
