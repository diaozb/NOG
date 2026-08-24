#!/usr/bin/env python3
"""Generate the two main-paper first-hit figures from audited formal summaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = Path(__file__).resolve().parents[1] / "figures"
FO_SOURCE = ROOT / "results/theory_validation_v4/analysis/formal_summary.csv"
ZO_SOURCE = ROOT / "zo_experiments/formal/formal_summary.csv"
T95_19 = 2.093024054

COLORS = {
    "NOG-FO": "#0072B2",
    "ME-DOL-FO": "#D55E00",
    "NOG-ZO": "#0072B2",
    "ME-DOL-ZO": "#D55E00",
    "DGFM": "#009E73",
    "DGFM+": "#CC79A7",
}
MARKERS = {
    "NOG-FO": "o",
    "ME-DOL-FO": "s",
    "NOG-ZO": "o",
    "ME-DOL-ZO": "s",
    "DGFM": "^",
    "DGFM+": "D",
}


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"target $\epsilon$ (smaller $\rightarrow$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.7)


def _draw(
    ax: plt.Axes,
    frame: pd.DataFrame,
    methods: tuple[str, ...],
    mean_column: str,
    sd_column: str,
    count_column: str,
) -> None:
    for method in methods:
        part = frame[frame["method"] == method].sort_values("epsilon", ascending=False)
        epsilon = part["epsilon"].to_numpy(float)
        mean = part[mean_column].to_numpy(float)
        count = part[count_column].to_numpy(float)
        half_width = T95_19 * part[sd_column].to_numpy(float) / np.sqrt(count)
        lower = np.maximum(mean - half_width, np.finfo(float).tiny)
        upper = mean + half_width
        ax.plot(
            epsilon,
            mean,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.2,
            linewidth=2.1 if method.startswith("NOG") else 1.8,
            label=method,
        )
        ax.fill_between(epsilon, lower, upper, color=COLORS[method], alpha=0.14, linewidth=0)


def make_fo() -> None:
    frame = pd.read_csv(FO_SOURCE)
    frame = frame[
        (frame["scope"] == "primary")
        & (frame["epsilon"] >= 0.0105)
        & (frame["epsilon"] <= 0.2)
        & (frame["hit_count"] == 20)
    ].copy()
    methods = ("NOG-FO", "ME-DOL-FO")
    if set(frame["method"]) != set(methods) or len(frame) != 50:
        raise RuntimeError("Unexpected FO formal-summary support.")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
    _draw(axes[0], frame, methods, "depth_mean", "depth_sd", "num_seeds")
    _draw(axes[1], frame, methods, "total_work_mean", "total_work_sd", "num_seeds")
    _style_axis(axes[0], "first-hit communication depth")
    _style_axis(axes[1], "first-hit training SFO work")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "fo_first_hit_depth_work.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fo_first_hit_depth_work.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_zo() -> None:
    frame = pd.read_csv(ZO_SOURCE)
    frame = frame[
        (frame["epsilon"] >= 0.03)
        & (frame["epsilon"] <= 0.2)
        & (frame["hits"] == 20)
    ].copy()
    methods = ("NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+")
    expected = 13 * len(methods)
    if set(frame["method"]) != set(methods) or len(frame) != expected:
        raise RuntimeError("Unexpected ZO complete-pair support.")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
    _draw(axes[0], frame, methods, "mean_first_hit_depth", "sd_first_hit_depth", "seeds")
    _draw(axes[1], frame, methods, "mean_first_hit_work", "sd_first_hit_work", "seeds")
    _style_axis(axes[0], "first-hit communication depth")
    _style_axis(axes[1], "first-hit training SZO work")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.035))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "zo_first_hit_depth_work.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "zo_first_hit_depth_work.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fo()
    make_zo()
