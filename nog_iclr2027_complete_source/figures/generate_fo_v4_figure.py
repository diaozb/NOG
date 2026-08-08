#!/usr/bin/env python3
"""Generate the fixed-batch FO figure used by the ICLR manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "results" / "theory_validation_v4" / "analysis"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


summary = [
    row
    for row in read_csv("formal_summary.csv")
    if row["scope"] == "primary" and float(row["epsilon"]) >= 0.0105
]
ratios = [
    row
    for row in read_csv("formal_ratios.csv")
    if int(row["data_B_total"]) == 8
]

by_method = {
    method: sorted(
        (row for row in summary if row["method"] == method),
        key=lambda row: float(row["epsilon"]),
    )
    for method in ("NOG-FO", "ME-DOL-FO")
}
ratios.sort(key=lambda row: float(row["epsilon"]))

plt.rcParams.update(
    {
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.7,
        "legend.fontsize": 7.7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.35), constrained_layout=True)
colors = {"NOG-FO": "#0072B2", "ME-DOL-FO": "#D55E00"}
markers = {"NOG-FO": "o", "ME-DOL-FO": "s"}

for method, rows in by_method.items():
    eps = np.asarray([float(row["epsilon"]) for row in rows])
    mean = np.asarray([float(row["depth_mean"]) for row in rows])
    sd = np.asarray([float(row["depth_sd"]) for row in rows])
    axes[0].plot(
        eps,
        mean,
        marker=markers[method],
        markersize=3.3,
        linewidth=1.5,
        color=colors[method],
        label=method,
    )
    axes[0].fill_between(
        eps,
        np.maximum(mean - sd, 1e-12),
        mean + sd,
        color=colors[method],
        alpha=0.14,
        linewidth=0,
    )

eps = np.asarray([float(row["epsilon"]) for row in ratios])
depth = np.asarray([float(row["depth_ratio_mean"]) for row in ratios])
depth_lo = np.asarray([float(row["depth_ratio_ci_low"]) for row in ratios])
depth_hi = np.asarray([float(row["depth_ratio_ci_high"]) for row in ratios])
work = np.asarray([float(row["work_ratio_mean"]) for row in ratios])
work_lo = np.asarray([float(row["work_ratio_ci_low"]) for row in ratios])
work_hi = np.asarray([float(row["work_ratio_ci_high"]) for row in ratios])

axes[1].plot(eps, depth, color="#009E73", marker="o", markersize=3.3,
             linewidth=1.5, label=r"Depth: ME-DOL/NOG")
axes[1].fill_between(eps, depth_lo, depth_hi, color="#009E73", alpha=0.14,
                     linewidth=0)
axes[1].plot(eps, work, color="#CC79A7", marker="s", markersize=3.3,
             linewidth=1.5, label=r"Work: NOG/ME-DOL")
axes[1].fill_between(eps, work_lo, work_hi, color="#CC79A7", alpha=0.14,
                     linewidth=0)
axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=0.9)

ticks = [0.2, 0.1, 0.05, 0.02, 0.0105]
tick_labels = ["0.2", "0.1", "0.05", "0.02", "0.0105"]
for ax in axes:
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(ticks, tick_labels)
    ax.grid(True, which="major", color="0.88", linewidth=0.55)
    ax.set_xlabel(r"Target accuracy $\epsilon$ (smaller $\rightarrow$)")

axes[0].set_yscale("log")
axes[0].set_ylabel("Confirmed first-hit depth")
axes[0].set_title("(a) Absolute communication depth")
axes[0].legend(frameon=False, loc="upper left")

axes[1].set_yscale("log")
axes[1].set_yticks([0.5, 1.0, 2.0], ["0.5", "1", "2"])
axes[1].set_ylim(0.43, 2.3)
axes[1].set_ylabel("Paired ratio")
axes[1].set_title("(b) Depth and work ratios")
axes[1].legend(frameon=False, loc="center left")

for suffix in ("pdf", "png"):
    fig.savefig(HERE / f"fo_v4_fixed_batch.{suffix}", dpi=300, bbox_inches="tight")
plt.close(fig)
