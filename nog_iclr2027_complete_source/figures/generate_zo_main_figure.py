#!/usr/bin/env python3
"""Generate the paper's ZO paired depth/work-ratio figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent.parent / "zo_experiments" / "formal" / "formal_ratios.csv"
BASELINES = ("ME-DOL-ZO", "DGFM", "DGFM+")
COLORS = {"ME-DOL-ZO": "#D55E00", "DGFM": "#0072B2", "DGFM+": "#009E73"}
MARKERS = {"ME-DOL-ZO": "o", "DGFM": "s", "DGFM+": "^"}


with CSV_PATH.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

# The main-paper claim is preregistered on the complete-pair region only.
rows = [row for row in rows if row["complete_pairing"] == "True"]

plt.rcParams.update(
    {
        "font.size": 8.1,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.7,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

figure, axes = plt.subplots(1, 2, figsize=(6.75, 2.25), constrained_layout=True)
metrics = (
    ("mean_depth_ratio", "depth_ratio_ci_low", "depth_ratio_ci_high", "Baseline / NOG-ZO depth"),
    ("mean_work_ratio", "work_ratio_ci_low", "work_ratio_ci_high", "Baseline / NOG-ZO work"),
)

for axis, (metric, low_key, high_key, ylabel) in zip(axes, metrics):
    for baseline in BASELINES:
        frame = sorted(
            (row for row in rows if row["baseline"] == baseline),
            key=lambda row: float(row["epsilon"]),
            reverse=True,
        )
        epsilon = np.asarray([float(row["epsilon"]) for row in frame])
        mean = np.asarray([float(row[metric]) for row in frame])
        low = np.asarray([float(row[low_key]) for row in frame])
        high = np.asarray([float(row[high_key]) for row in frame])
        axis.plot(
            epsilon,
            mean,
            color=COLORS[baseline],
            marker=MARKERS[baseline],
            markersize=3.0,
            linewidth=1.45,
            label=baseline,
        )
        axis.fill_between(epsilon, low, high, color=COLORS[baseline], alpha=0.14, linewidth=0)

    axis.axhline(1.0, color="0.30", linestyle="--", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xticks([0.2, 0.1, 0.05, 0.03, 0.018], [".2", ".1", ".05", ".03", ".018"])
    axis.grid(True, which="major", color="0.88", linewidth=0.5)
    axis.set_xlabel(r"Target accuracy $\epsilon$ (smaller $\rightarrow$)")
    axis.set_ylabel(ylabel)

axes[0].set_title("(a) Communication depth")
axes[1].set_title("(b) Total SZO work")
axes[0].legend(frameon=False, loc="upper left")

for suffix in ("pdf", "png"):
    figure.savefig(HERE / f"zo_depth_work_ratios.{suffix}", dpi=300, bbox_inches="tight")
plt.close(figure)
