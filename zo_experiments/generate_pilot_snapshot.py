#!/usr/bin/env python3
"""Generate the preliminary four-method ZO pilot snapshot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "zo_experiments"
BASELINE = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/pilot"
    / "refine_work_983040_baselines_dense_eval4/results.csv"
)
NOG = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/pilot"
    / "refine_work_983040_nog_dense_eval4/results.csv"
)

SELECTED = {
    "NOG-ZO": "NOG-ZO__M-2__eta-0p01__smooth_B-8",
    "ME-DOL-ZO": "ME-DOL-ZO__epoch_length-12__theory_multiplier-60p0",
    "DGFM": "DGFM__eta-0p05",
    "DGFM+": "DGFMplus__eta-0p05",
}
EPSILONS = [
    0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06,
    0.05, 0.04, 0.03, 0.025, 0.02, 0.018, 0.016, 0.015, 0.014,
    0.013, 0.012, 0.0115, 0.011, 0.01075, 0.0105,
]


def first_confirmed_hit(frame: pd.DataFrame, epsilon: float):
    frame = frame.sort_values("iteration").reset_index(drop=True)
    below = frame["stat_proxy"].le(epsilon)
    confirmed = below & below.shift(-1, fill_value=False)
    return None if not confirmed.any() else frame.loc[confirmed.idxmax()]


raw = pd.concat([pd.read_csv(BASELINE), pd.read_csv(NOG)], ignore_index=True)
raw = raw[
    raw.apply(
        lambda row: SELECTED.get(row["method"]) == row["candidate_id"], axis=1
    )
]

records = []
for method in SELECTED:
    method_frame = raw[raw["method"].eq(method)]
    for epsilon in EPSILONS:
        hits = []
        for _, seed_frame in method_frame.groupby("formal_seed"):
            hit = first_confirmed_hit(seed_frame, epsilon)
            if hit is not None:
                hits.append(hit)
        records.append(
            {
                "method": method,
                "epsilon": epsilon,
                "hit_count": len(hits),
                "seed_count": 5,
                "hit_rate": len(hits) / 5,
                "depth_mean_conditional": (
                    float(np.mean([row["depth"] for row in hits])) if hits else np.nan
                ),
                "work_mean_conditional": (
                    float(np.mean([row["total_work"] for row in hits]))
                    if hits
                    else np.nan
                ),
            }
        )

summary = pd.DataFrame(records)
summary.to_csv(OUT / "pilot_snapshot.csv", index=False)

plt.rcParams.update(
    {
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.7,
        "legend.fontsize": 7.3,
        "pdf.fonttype": 42,
    }
)
colors = {
    "NOG-ZO": "#0072B2",
    "ME-DOL-ZO": "#D55E00",
    "DGFM": "#009E73",
    "DGFM+": "#CC79A7",
}
markers = {"NOG-ZO": "o", "ME-DOL-ZO": "s", "DGFM": "^", "DGFM+": "D"}
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.75), constrained_layout=True)

for method in SELECTED:
    part = summary[summary["method"].eq(method)].sort_values("epsilon")
    kwargs = {
        "label": method,
        "color": colors[method],
        "marker": markers[method],
        "markersize": 3.2,
        "linewidth": 1.4,
    }
    axes[0].plot(part["epsilon"], part["hit_rate"], **kwargs)
    axes[1].plot(part["epsilon"], part["depth_mean_conditional"], **kwargs)
    axes[2].plot(part["epsilon"], part["work_mean_conditional"], **kwargs)

for axis in axes:
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.grid(True, which="major", color="0.88", linewidth=0.55)
    axis.set_xlabel(r"Target $\epsilon$ (smaller $\rightarrow$)")
axes[0].set_ylim(-0.05, 1.05)
axes[0].set_ylabel("Confirmed-hit rate")
axes[0].set_title("(a) Hit rate over five pilot seeds")
axes[1].set_yscale("log")
axes[1].set_ylabel("Mean first-hit depth")
axes[1].set_title("(b) Depth, conditional on hit")
axes[2].set_yscale("log")
axes[2].set_ylabel("Mean first-hit SZO work")
axes[2].set_title("(c) Work, conditional on hit")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Preliminary fixed-work ZO pilot (not formal results)", fontsize=9.5)

for suffix in ("png", "pdf"):
    fig.savefig(
        OUT / "figures" / f"zo_pilot_snapshot.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
plt.close(fig)
