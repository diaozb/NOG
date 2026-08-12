"""Aggregate and plot completed sharded Step ZO-9C3 trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / (
    "outputs/distributed_zo/zo_theory_validation/real_data/"
    "formal_fixed_work983040_shards"
)
DEFAULT_OUTPUT = ROOT / "zo_experiments/real_data/formal"
METHOD_ORDER = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def collect(source: Path) -> pd.DataFrame:
    frames = []
    tasks = []
    for shard in range(4):
        root = source / f"shard_{shard}"
        progress = json.loads((root / "progress.json").read_text())
        if progress.get("status") != "complete" or progress.get("completed_tasks") != 40:
            raise ValueError(f"Shard {shard} is not complete.")
        for path in sorted((root / "partials").glob("*.csv")):
            frame = pd.read_csv(path)
            if frame.empty:
                raise ValueError(f"Empty formal partial: {path}.")
            frames.append(frame)
            tasks.append(
                (
                    str(frame.iloc[0]["dataset"]),
                    str(frame.iloc[0]["method"]),
                    int(frame.iloc[0]["formal_seed"]),
                )
            )
    expected = {
        (dataset, method, seed)
        for dataset in ["a9a", "ijcnn1"]
        for method in METHOD_ORDER
        for seed in range(20)
    }
    if len(tasks) != 160 or set(tasks) != expected or len(set(tasks)) != 160:
        raise ValueError("Formal task identities are not exactly the 160 preregistered tasks.")
    results = pd.concat(frames, ignore_index=True)
    metrics = results[["objective", "stat_proxy", "train_accuracy"]].to_numpy(float)
    if not np.isfinite(metrics).all():
        raise ValueError("Formal results contain non-finite metrics.")
    return results


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = (
        results.sort_values("iteration")
        .groupby(["dataset", "method", "formal_seed"], as_index=False)
        .tail(1)
    )
    if not (final["total_work"].astype(int) <= 983040).all():
        raise ValueError("A formal task exceeded its training-work budget.")
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
            depth_std=("depth", "std"),
            total_work_mean=("total_work", "mean"),
            total_work_std=("total_work", "std"),
            seed_count=("formal_seed", "nunique"),
        )
        .sort_values(["dataset", "method"])
    )
    if not (summary["seed_count"].astype(int) == 20).all():
        raise ValueError("A dataset/method group is missing a formal seed.")
    return final, summary


def plot_curves(results: pd.DataFrame, output: Path) -> None:
    for dataset in ["a9a", "ijcnn1"]:
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        subset = results[results["dataset"] == dataset].copy()
        for method in METHOD_ORDER:
            frame = subset[subset["method"] == method]
            grouped = frame.groupby("total_work")["objective"].agg(["mean", "std"])
            x = grouped.index.to_numpy(dtype=float)
            y = grouped["mean"].to_numpy(dtype=float)
            s = grouped["std"].fillna(0).to_numpy(dtype=float)
            axes[0].plot(x, y, label=method)
            axes[0].fill_between(x, y - s, y + s, alpha=0.13)
            grouped = frame.groupby("depth")["stat_proxy"].agg(["mean", "std"])
            x = grouped.index.to_numpy(dtype=float)
            y = grouped["mean"].to_numpy(dtype=float)
            s = grouped["std"].fillna(0).to_numpy(dtype=float)
            axes[1].plot(x, y, label=method)
            axes[1].fill_between(x, y - s, y + s, alpha=0.13)
        axes[0].set_xlabel("training SZO calls")
        axes[0].set_ylabel("objective")
        axes[1].set_xlabel("communication depth")
        axes[1].set_ylabel("stationarity proxy")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle(f"Capped-l1 SVM on {dataset} (20 formal seeds)")
        figure.tight_layout()
        figure.savefig(output / f"{dataset}_formal_curves.png", dpi=220)
        figure.savefig(output / f"{dataset}_formal_curves.pdf")
        plt.close(figure)


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = collect(source)
    final, summary = summarize(results)
    results.to_csv(output / "results.csv", index=False)
    final.to_csv(output / "final_per_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    plot_curves(results, output)
    shutil.copy2(source / "shard_0/frozen_parameters.json", output / "frozen_parameters.json")
    payload = {
        "status": "complete",
        "task_count": 160,
        "dataset_method_groups": 8,
        "formal_seed_count_per_group": 20,
        "claim": "paper-result real-data experiment; interpret independently of synthetic epsilon scaling",
    }
    (output / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
