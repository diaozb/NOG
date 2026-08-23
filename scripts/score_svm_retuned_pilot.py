#!/usr/bin/env python3
"""Score preregistered NOG SVM pilot trajectories and select validation configs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--datasets", default="a9a,ijcnn1")
    p.add_argument("--seeds", default="100,101,102,103,104")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--top-per-dataset", type=int, default=3)
    p.add_argument("--stage-tag", default="search")
    p.add_argument("--expected-candidates", type=int, default=15)
    p.add_argument("--run-prefix", default="retuned_svm")
    return p.parse_args()


def candidate_from_rows(frame: pd.DataFrame) -> tuple[str, str]:
    return str(frame.iloc[0]["candidate_id"]), str(frame.iloc[0]["candidate_parameters"])


def main() -> None:
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    records: list[dict[str, object]] = []
    for dataset in datasets:
        base = args.root / f"{args.run_prefix}_{dataset}" / "pilot"
        for seed in seeds:
            paths = sorted(base.glob(f"refine_work_983040_{args.stage_tag}_seed{seed}/partials/*.csv"))
            if len(paths) != int(args.expected_candidates):
                raise RuntimeError(f"{dataset} seed {seed}: expected {args.expected_candidates} partials, found {len(paths)}")
            for path in paths:
                frame = pd.read_csv(path).sort_values("total_work")
                if frame.empty or not np.isfinite(frame["stat_proxy"].to_numpy(float)).all():
                    raise RuntimeError(f"Non-finite or empty candidate: {path}")
                candidate_id, params = candidate_from_rows(frame)
                values: list[float] = []
                actual_work: list[int] = []
                for fraction in FRACTIONS:
                    target = 983040.0 * fraction
                    eligible = frame[frame["total_work"] <= target]
                    if eligible.empty:
                        raise RuntimeError(f"No checkpoint below target {target}: {path}")
                    row = eligible.iloc[(eligible["total_work"] - target).abs().argmin()]
                    epsilon = float(row["stat_proxy"])
                    if not np.isfinite(epsilon) or epsilon <= 0:
                        raise RuntimeError(f"Invalid epsilon {epsilon}: {path}")
                    values.append(epsilon)
                    actual_work.append(int(row["total_work"]))
                final = frame.iloc[-1]
                records.append({
                    "dataset": dataset,
                    "search_seed": seed,
                    "candidate_id": candidate_id,
                    "candidate_parameters": params,
                    "epsilon_25": values[0],
                    "epsilon_50": values[1],
                    "epsilon_75": values[2],
                    "epsilon_100": values[3],
                    "actual_work_25": actual_work[0],
                    "actual_work_50": actual_work[1],
                    "actual_work_75": actual_work[2],
                    "actual_work_100": actual_work[3],
                    "final_epsilon": float(final["stat_proxy"]),
                    "final_depth": int(final["communication_round"]),
                    "final_work": int(final["total_work"]),
                })
    per_seed = pd.DataFrame(records)
    eps_cols = [f"epsilon_{int(100*f)}" for f in FRACTIONS]
    per_seed["log_epsilon_score"] = np.log(per_seed[eps_cols].to_numpy(float)).mean(axis=1)
    summary = (
        per_seed.groupby(["dataset", "candidate_id", "candidate_parameters"], as_index=False)
        .agg(
            log_epsilon_score=("log_epsilon_score", "mean"),
            log_epsilon_score_std=("log_epsilon_score", "std"),
            epsilon_25_mean=("epsilon_25", "mean"),
            epsilon_50_mean=("epsilon_50", "mean"),
            epsilon_75_mean=("epsilon_75", "mean"),
            epsilon_100_mean=("epsilon_100", "mean"),
            epsilon_100_std=("epsilon_100", "std"),
            final_depth_mean=("final_depth", "mean"),
            final_work_mean=("final_work", "mean"),
            seed_count=("search_seed", "nunique"),
        )
        .sort_values(["dataset", "log_epsilon_score", "epsilon_100_mean", "epsilon_100_std", "final_depth_mean"])
    )
    summary["rank"] = summary.groupby("dataset")["log_epsilon_score"].rank(method="first").astype(int)
    selected = summary[summary["rank"] <= int(args.top_per_dataset)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output.with_name("pilot_grid_per_seed.csv"), index=False)
    summary.to_csv(args.output, index=False)
    selected.to_csv(args.output.with_name("pilot_selected_for_validation.csv"), index=False)
    print(summary.to_string(index=False))
    print("SELECTED")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
