#!/usr/bin/env python3
"""Plot latest SVM NOG-ZO against the paired optimistic ablation.

The layout and budget grids intentionally match the existing four-algorithm
SVM figure: 2x2 panels, nearest native checkpoints, 20-seed means and 95% t
confidence bands.  No interpolation is performed.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = OUT / "formal_trajectories_svm_and_ablation.csv"
METHODS = ("latest_NOG-ZO", "current_NOG-opt", "current_NOG-nonopt")
LABELS = {
    "latest_NOG-ZO": "latest NOG-ZO",
    "current_NOG-opt": "NOG-opt",
    "current_NOG-nonopt": "NOG-non-opt",
}
COLORS = {
    "latest_NOG-ZO": "#0072B2",
    "current_NOG-opt": "#D62728",
    "current_NOG-nonopt": "#2CA02C",
}
DEPTH_TARGETS = [float(768 * i) for i in range(1, 6)]
WORK_TARGETS = [float(98304 * i) for i in range(1, 11)]
T95 = 2.093024054


def nearest(group: pd.DataFrame, column: str, target: float) -> pd.Series:
    return group.iloc[
        (group[column].sub(target).abs()
         + (group[column].gt(target).astype(float) * 1e-9)
         + (group["iteration"] * 1e-15)).argmin()
    ]


def ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = T95 * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def load() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    frame = frame[frame["comparison_method"].isin(METHODS)].copy()
    frame["formal_seed"] = pd.to_numeric(frame["formal_seed"])
    frame["iteration"] = pd.to_numeric(frame["iteration"])
    frame["depth"] = pd.to_numeric(frame["depth"])
    frame["total_work"] = pd.to_numeric(frame["total_work"])
    frame["stat_proxy"] = pd.to_numeric(frame["stat_proxy"])
    if frame.stat_proxy.isna().any():
        raise ValueError("non-finite stat_proxy in SVM ablation input")
    counts = frame.groupby(["dataset", "comparison_method"])["formal_seed"].nunique()
    if (counts != 20).any():
        raise ValueError(f"Expected 20 seeds per group, got:\n{counts}")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for dataset in ("a9a", "ijcnn1"):
        # Match the historical figure's common x=0 reference: the latest NOG
        # formal seed set's first native checkpoint, shown as a black star.
        initial = frame[(frame.dataset == dataset) & (frame.comparison_method == "latest_NOG-ZO")]
        initial = initial.sort_values("iteration").groupby("formal_seed", as_index=False).head(1)
        mean, low, high = ci(initial.stat_proxy.to_numpy(float))
        for view, targets, column in (
            ("communication_depth", DEPTH_TARGETS, "depth"),
            ("training_work", WORK_TARGETS, "total_work"),
        ):
            for method in METHODS:
                rows.append({"dataset": dataset, "budget_view": view, "target_index": 0,
                             "target_budget": 0.0, "method": method,
                             "point_role": "common_initial_reference",
                             "epsilon_mean": mean, "epsilon_ci95_low": low,
                             "epsilon_ci95_high": high, "actual_budget_min": 0.0,
                             "actual_budget_max": 0.0, "seed_count": 20})
            for index, target in enumerate(targets, 1):
                for method in METHODS:
                    selected = []
                    for seed, group in frame[(frame.dataset == dataset) & (frame.comparison_method == method)].groupby("formal_seed"):
                        selected.append(nearest(group, column, target))
                    values = np.asarray([r.stat_proxy for r in selected], dtype=float)
                    m, l, h = ci(values)
                    actual = np.asarray([r[column] for r in selected], dtype=float)
                    rows.append({"dataset": dataset, "budget_view": view, "target_index": index,
                                 "target_budget": target, "method": method,
                                 "point_role": "matched_budget_checkpoint",
                                 "epsilon_mean": m, "epsilon_ci95_low": l,
                                 "epsilon_ci95_high": h,
                                 "actual_budget_min": float(actual.min()),
                                 "actual_budget_max": float(actual.max()), "seed_count": len(values)})
    return pd.DataFrame(rows)


def plot(records: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10.5,
                         "axes.titlesize": 11, "legend.fontsize": 9.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharey=True)
    labels = (("(a)", "(b)"), ("(c)", "(d)"))
    max_depth_mismatch = 0.0
    max_work_mismatch = 0.0
    for ri, dataset in enumerate(("a9a", "ijcnn1")):
        for ci_index, view in enumerate(("communication_depth", "training_work")):
            ax = axes[ri, ci_index]
            part = records[(records.dataset == dataset) & (records.budget_view == view)]
            for method in METHODS:
                q = part[part.method == method].sort_values("target_index")
                x = q.target_budget.to_numpy(float)
                y = q.epsilon_mean.to_numpy(float)
                lo = q.epsilon_ci95_low.to_numpy(float)
                hi = q.epsilon_ci95_high.to_numpy(float)
                ax.fill_between(x, lo, hi, color=COLORS[method], alpha=.12, linewidth=0, zorder=1)
                draw_order = {"latest_NOG-ZO": 2, "current_NOG-opt": 3, "current_NOG-nonopt": 4}[method]
                ax.plot(x, y, color=COLORS[method], lw=2.8 if method == "latest_NOG-ZO" else 2.2,
                        marker="o", ms=4.8, markeredgecolor="white", markeredgewidth=.6,
                        label=LABELS[method], zorder=draw_order)
                if view == "communication_depth":
                    max_depth_mismatch = max(max_depth_mismatch, float(np.maximum(abs(q.actual_budget_min-x), abs(q.actual_budget_max-x)).max()))
                else:
                    max_work_mismatch = max(max_work_mismatch, float(np.maximum(abs(q.actual_budget_min-x), abs(q.actual_budget_max-x)).max()))
            init = part[(part.method == "latest_NOG-ZO") & (part.point_role == "common_initial_reference")].iloc[0]
            ax.scatter([0], [init.epsilon_mean], marker="*", s=95, color="#222222", edgecolor="white", linewidth=.7, zorder=6)
            ax.set_ylim(0, .405)
            ax.grid(True, color="#B8B8B8", alpha=.45)
            ax.spines[["top", "right"]].set_visible(False)
            if view == "communication_depth":
                ax.set_xlim(-120, 4100); ax.set_xlabel("Common communication-depth target")
                subtitle = "same depth checkpoints"
            else:
                ax.set_xlim(-30000, 1050000); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
                ax.set_xlabel("Common training-work target"); subtitle = "same work checkpoints"
            ax.set_title(f"{labels[ri][ci_index]} {dataset}: {subtitle}")
            if ci_index == 0: ax.set_ylabel("Achieved epsilon (lower is better)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.suptitle("SVM NOG optimistic ablation: epsilon curves at matched budget checkpoints", fontsize=14, y=.992)
    fig.text(.5, .012, "Lines: 20-seed means; shaded bands: 95% CI; nearest native checkpoints only; no interpolation. "
             f"Maximum native mismatch: {max_depth_mismatch:.0f} depth, {max_work_mismatch:.0f} work. "
             "The black star is the common initialization reference from latest NOG-ZO.", ha="center", fontsize=8.8, color="#444444")
    fig.subplots_adjust(left=.08, right=.985, bottom=.10, top=.88, hspace=.31, wspace=.11)
    stem = OUT / "svm_nog_opt_nonopt_equal_budget_epsilon"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = summarize(load())
    records.to_csv(OUT / "svm_nog_opt_nonopt_equal_budget_epsilon_data.csv", index=False, float_format="%.10g")
    plot(records)
    print("saved SVM NOG ablation equal-budget plot and CSV")


if __name__ == "__main__":
    main()
