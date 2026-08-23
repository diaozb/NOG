#!/usr/bin/env python3
"""Make full and strict 0--0.05 equal-budget SVM curves plus threshold tables."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+")
DATASETS = ("a9a", "ijcnn1")
COLORS = {"NOG-ZO": "#0072B2", "ME-DOL-ZO": "#D55E00", "DGFM": "#009E73", "DGFM+": "#CC79A7"}
DEPTH_TARGETS = [0.0, 768.0, 1536.0, 2304.0, 3072.0, 3840.0]
WORK_TARGETS = [0.0] + [float(98304 * i) for i in range(1, 11)]
THRESHOLDS = [0.05, 0.03, 0.02, 0.015]
T95 = 2.093024054


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def nearest(frame: pd.DataFrame, column: str, target: float) -> pd.Series:
    distances = (frame[column].astype(float) - target).abs()
    order = pd.DataFrame({"distance": distances, "above": frame[column].astype(float) > target, "iteration": frame["iteration"].astype(int)})
    idx = order.sort_values(["distance", "above", "iteration"]).index[0]
    return frame.loc[idx]


def summarize_budget(df: pd.DataFrame, view: str, targets: list[float]) -> pd.DataFrame:
    column = "depth" if view == "communication_depth" else "total_work"
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for index, target in enumerate(targets):
                vals = []
                actual = []
                for seed in range(20):
                    frame = df[(df.dataset == dataset) & (df.method == method) & (df.formal_seed == seed)]
                    if frame.empty:
                        raise RuntimeError(f"Missing {dataset}/{method}/seed-{seed}")
                    row = nearest(frame, column, target)
                    vals.append(float(row.stat_proxy))
                    actual.append(float(row[column]))
                arr = np.asarray(vals, float)
                mean = float(arr.mean())
                se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
                rows.append({"dataset": dataset, "method": method, "budget_view": view, "target_index": index, "target_budget": target, "epsilon_mean": mean, "epsilon_std": float(arr.std(ddof=1)), "epsilon_ci95_low": mean - T95 * se, "epsilon_ci95_high": mean + T95 * se, "actual_budget_min": min(actual), "actual_budget_max": max(actual), "seed_count": len(arr), "clipped_above_0p05": bool(mean > 0.05)})
    return pd.DataFrame(rows)


def plot_grid(data: pd.DataFrame, output: Path, zoom: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    for row, dataset in enumerate(DATASETS):
        for col, view in enumerate(("communication_depth", "training_work")):
            ax = axes[row, col]
            panel = data[(data.dataset == dataset) & (data.budget_view == view)]
            xcol = "target_budget"
            for method in METHODS:
                part = panel[panel.method == method].sort_values("target_index")
                ax.plot(part[xcol], part.epsilon_mean, marker="o", markersize=3.3, linewidth=2.8 if method == "NOG-ZO" else 2.0, color=COLORS[method], label=method)
                ax.fill_between(part[xcol].to_numpy(float), part.epsilon_ci95_low.to_numpy(float), part.epsilon_ci95_high.to_numpy(float), color=COLORS[method], alpha=0.13, linewidth=0)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"{dataset} — {'same communication depth' if view == 'communication_depth' else 'same training work'}")
            ax.set_xlabel("communication depth" if view == "communication_depth" else "training work")
            ax.set_ylabel("epsilon" if col == 0 else "")
            if view == "communication_depth":
                ax.set_xticks(DEPTH_TARGETS)
            else:
                ax.set_xticks(WORK_TARGETS[::2])
                ax.ticklabel_format(style="plain", axis="x")
            if zoom:
                ax.set_ylim(0.0, 0.05)
                ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.text(0.02, 0.96, "Early epsilon>0.05 points clipped;\nfirst shown point is native checkpoint", transform=ax.transAxes, fontsize=8, va="top", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
            else:
                ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Retuned capped-l1 SVM: four-algorithm epsilon curves\nMean over 20 formal seeds; shaded bands are 95% CI; native checkpoints only", y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def threshold_table(df: pd.DataFrame, output: Path) -> None:
    per_seed: list[dict[str, object]] = []
    for dataset in DATASETS:
        for method in METHODS:
            for seed in range(20):
                frame = df[(df.dataset == dataset) & (df.method == method) & (df.formal_seed == seed)].sort_values("iteration")
                for threshold in THRESHOLDS:
                    hits = frame[frame.stat_proxy <= threshold]
                    if hits.empty:
                        per_seed.append({"dataset": dataset, "method": method, "formal_seed": seed, "threshold": threshold, "depth_first_hit": np.nan, "work_first_hit": np.nan, "hit": False})
                    else:
                        hit = hits.iloc[0]
                        per_seed.append({"dataset": dataset, "method": method, "formal_seed": seed, "threshold": threshold, "depth_first_hit": float(hit.depth), "work_first_hit": float(hit.total_work), "hit": True})
    per = pd.DataFrame(per_seed)
    summary = per.groupby(["dataset", "method", "threshold"], as_index=False).agg(hit_count=("hit", "sum"), seed_count=("formal_seed", "nunique"), depth_first_hit_mean=("depth_first_hit", "mean"), depth_first_hit_std=("depth_first_hit", "std"), work_first_hit_mean=("work_first_hit", "mean"), work_first_hit_std=("work_first_hit", "std"))
    per.to_csv(output.with_name("threshold_first_hit_per_seed.csv"), index=False)
    summary.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.source)
    required = {"dataset", "method", "formal_seed", "iteration", "depth", "total_work", "stat_proxy"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    full = pd.concat([summarize_budget(df, "communication_depth", DEPTH_TARGETS), summarize_budget(df, "training_work", WORK_TARGETS)], ignore_index=True)
    full.to_csv(args.output / "plot_data_full.csv", index=False)
    zoom = full.copy()
    zoom.to_csv(args.output / "plot_data_zoom_0_005.csv", index=False)
    plot_grid(full, args.output / "svm_equal_budget_epsilon_full", zoom=False)
    plot_grid(zoom, args.output / "svm_equal_budget_epsilon_zoom_0_005", zoom=True)
    threshold_table(df, args.output / "threshold_first_hit.csv")
    print(full.to_string(index=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
