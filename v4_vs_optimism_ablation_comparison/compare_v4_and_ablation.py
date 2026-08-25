#!/usr/bin/env python3
"""Compare the fine epsilon grid of FO v4 with the optimism ablation.

The v4 summary already contains first-hit statistics on its fine epsilon grid.
For the ablation, first hits are reconstructed from the raw checkpoint table;
two consecutive checkpoints at or below a threshold are required.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
V4_SUMMARY = ROOT / "results/theory_validation_v4/analysis/formal_summary.csv"
V4_FROZEN = ROOT / "results/theory_validation_v4/frozen_parameters.json"
ABL_TRAJ = ROOT / "results/nog_optimism_ablation_20260825/analysis/formal_trajectories.csv"


def reconstruct_ablation(df: pd.DataFrame, epsilons: list[float]) -> pd.DataFrame:
    """Return per-method threshold means from raw ablation checkpoints."""
    # seed 631 has an identical temporary and final record.  De-duplicate it
    # before computing seed means; no trajectory values are discarded.
    df = df.sort_values(["method", "formal_seed", "round", "source_partial"])
    df = df.drop_duplicates(["method", "formal_seed", "round"], keep="last")
    rows = []
    for method, mdf in df.groupby("method", sort=True):
        for seed, sdf in mdf.groupby("formal_seed", sort=True):
            sdf = sdf.sort_values("round").reset_index(drop=True)
            for eps in epsilons:
                hit_idx = None
                for i in range(1, len(sdf)):
                    if sdf.loc[i - 1, "stat_proxy"] <= eps and sdf.loc[i, "stat_proxy"] <= eps:
                        hit_idx = i
                        break
                if hit_idx is None:
                    hit = False
                    hit_depth = float("nan")
                    hit_work = float("nan")
                else:
                    hit = True
                    # Report the first checkpoint in the confirmed pair,
                    # matching the v4 threshold convention.
                    first_idx = hit_idx - 1
                    hit_depth = float(sdf.loc[first_idx, "depth"])
                    hit_work = float(sdf.loc[first_idx, "work"])
                rows.append(
                    {
                        "method": method,
                        "formal_seed": int(seed),
                        "epsilon": eps,
                        "hit": hit,
                        "first_hit_depth": hit_depth,
                        "first_hit_work": hit_work,
                        "capped_depth": float(sdf["depth"].max()),
                        "capped_work": float(sdf["work"].max()),
                    }
                )
    per_seed = pd.DataFrame(rows)
    out = []
    for (method, eps), g in per_seed.groupby(["method", "epsilon"], sort=False):
        hits = g[g["hit"]]
        out.append(
            {
                "method": {"NOG-FO": "current_NOG-opt", "NOG-FO-NONOPT": "current_NOG-nonopt"}.get(method, method),
                "epsilon": eps,
                "seed_count": len(g),
                "hit_count": int(g["hit"].sum()),
                "hit_rate": float(g["hit"].mean()),
                "first_hit_depth_mean": float(hits["first_hit_depth"].mean()) if len(hits) else float("nan"),
                "first_hit_depth_sd": float(hits["first_hit_depth"].std(ddof=1)) if len(hits) > 1 else float("nan"),
                "first_hit_work_mean": float(hits["first_hit_work"].mean()) if len(hits) else float("nan"),
                "first_hit_work_sd": float(hits["first_hit_work"].std(ddof=1)) if len(hits) > 1 else float("nan"),
                "capped_depth_mean": float(g["capped_depth"].mean()),
                "capped_depth_sd": float(g["capped_depth"].std(ddof=1)) if len(g) > 1 else float("nan"),
                "capped_work_mean": float(g["capped_work"].mean()),
                "capped_work_sd": float(g["capped_work"].std(ddof=1)) if len(g) > 1 else float("nan"),
                "source": "current_ablation_raw_trajectories",
            }
        )
    return pd.DataFrame(out)


def load_v4(epsilons: list[float]) -> pd.DataFrame:
    v4 = pd.read_csv(V4_SUMMARY)
    v4 = v4[(v4["method"] == "NOG-FO") & (v4["epsilon"].isin(epsilons))].copy()
    # Keep the original scope label: primary points were within the frozen
    # protocol and exploratory points were reported as censored extensions.
    v4["method"] = "v4_NOG-FO"
    v4 = v4.rename(
        columns={
            "num_seeds": "seed_count",
            "hit_count": "hit_count",
            "hit_rate": "hit_rate",
            "depth_mean": "first_hit_depth_mean",
            "depth_sd": "first_hit_depth_sd",
            "total_work_mean": "first_hit_work_mean",
            "total_work_sd": "first_hit_work_sd",
            "capped_depth_mean": "capped_depth_mean",
            "capped_total_work_mean": "capped_work_mean",
        }
    )
    for c in ["capped_depth_sd", "capped_work_sd"]:
        v4[c] = float("nan")
    v4["source"] = "theory_validation_v4_formal_summary"
    cols = [
        "method", "epsilon", "seed_count", "hit_count", "hit_rate",
        "first_hit_depth_mean", "first_hit_depth_sd", "first_hit_work_mean", "first_hit_work_sd",
        "capped_depth_mean", "capped_depth_sd", "capped_work_mean", "capped_work_sd", "source", "scope",
    ]
    return v4[cols]


def make_markdown_table(table: pd.DataFrame) -> None:
    cols = [
        "epsilon", "v4_NOG-FO_hit", "v4_NOG-FO_depth_first_hit", "v4_NOG-FO_work_first_hit",
        "current_NOG-opt_hit", "current_NOG-opt_depth_first_hit", "current_NOG-opt_work_first_hit",
        "current_NOG-nonopt_hit", "current_NOG-nonopt_depth_first_hit", "current_NOG-nonopt_work_first_hit",
    ]
    lines = [
        "| epsilon | v4 NOG hit | v4 NOG depth (first hit) | v4 NOG work (first hit) | current opt hit | current opt depth (first hit) | current opt work (first hit) | current non-opt hit | current non-opt depth (first hit) | current non-opt work (first hit) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        vals = [
            f"{r[c]:.5g}" if c == "epsilon" else (f"{r[c]:.1f}" if "depth" in c or "work" in c else f"{int(r[c])}/{int(r[c.replace('_hit', '_seeds')])}" if c.endswith("_hit") and pd.notna(r[c]) else "—")
            for c in cols
        ]
        lines.append("| " + " | ".join(vals) + " |")
    (OUT / "epsilon_threshold_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(table: pd.DataFrame) -> None:
    colors = {"v4_NOG-FO": "#1f77b4", "current_NOG-opt": "#d62728", "current_NOG-nonopt": "#2ca02c"}
    labels = {"v4_NOG-FO": "v4 NOG-FO", "current_NOG-opt": "current NOG-opt", "current_NOG-nonopt": "current NOG-non-opt"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, y, ylabel in zip(axes, ["first_hit_depth_mean", "first_hit_work_mean"], ["communication depth (first-hit mean)", "training work (first-hit mean)"]):
        for method in colors:
            g = table[table["method"] == method].sort_values("epsilon", ascending=False)
            ax.plot(g["epsilon"], g[y], marker="o", ms=3.8, lw=1.7, color=colors[method], label=labels[method])
            cens = g[(g["hit_rate"] < 1.0) & g[y].notna()]
            if len(cens):
                ax.scatter(cens["epsilon"], cens[y], marker="o", s=34, facecolors="none", edgecolors=colors[method], linewidths=1.2, zorder=4)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"target $\epsilon$ (v4 fine grid)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("SyntheticMaxSinL1: FO v4 vs current optimistic/non-optimistic ablation\nFO v4 eval_every=2; current ablation eval_every=24; open markers indicate hit rate < 100%")
    fig.savefig(OUT / "v4_vs_current_epsilon_depth_work.png", dpi=220)
    fig.savefig(OUT / "v4_vs_current_epsilon_depth_work.pdf")
    plt.close(fig)


def main() -> None:
    v4 = pd.read_csv(V4_SUMMARY)
    epsilons = sorted(v4.loc[v4["method"] == "NOG-FO", "epsilon"].dropna().unique(), reverse=True)
    v4_table = load_v4(epsilons)
    current = reconstruct_ablation(pd.read_csv(ABL_TRAJ), epsilons)
    all_rows = pd.concat([v4_table, current], ignore_index=True, sort=False)
    all_rows = all_rows.sort_values(["epsilon", "method"], ascending=[False, True])
    all_rows.to_csv(OUT / "epsilon_threshold_long.csv", index=False, float_format="%.10g")

    wide = pd.DataFrame({"epsilon": epsilons})
    for method, prefix in [("v4_NOG-FO", "v4_NOG-FO"), ("current_NOG-opt", "current_NOG-opt"), ("current_NOG-nonopt", "current_NOG-nonopt")]:
        g = all_rows[all_rows["method"] == method].set_index("epsilon")
        wide[f"{prefix}_seeds"] = g["seed_count"].reindex(epsilons).to_numpy()
        wide[f"{prefix}_hit"] = g["hit_count"].reindex(epsilons).to_numpy()
        wide[f"{prefix}_hit_rate"] = g["hit_rate"].reindex(epsilons).to_numpy()
        wide[f"{prefix}_depth_first_hit"] = g["first_hit_depth_mean"].reindex(epsilons).to_numpy()
        wide[f"{prefix}_depth_capped"] = g["capped_depth_mean"].reindex(epsilons).to_numpy()
        wide[f"{prefix}_work_first_hit"] = g["first_hit_work_mean"].reindex(epsilons).to_numpy()
        wide[f"{prefix}_work_capped"] = g["capped_work_mean"].reindex(epsilons).to_numpy()
    schedule = {float(x["epsilon"]): int(x["data_B_total"]) for x in json.loads(V4_FROZEN.read_text())["selected_schedule"]}
    v4_rows = all_rows[all_rows["method"] == "v4_NOG-FO"].set_index("epsilon")
    v4_batch = []
    for e in epsilons:
        if float(e) in schedule:
            v4_batch.append(schedule[float(e)])
        else:
            # Exploratory v4 points are not in the frozen schedule JSON; the
            # work/depth ledger identifies their fixed batch unambiguously.
            row = v4_rows.loc[float(e)]
            v4_batch.append(int(round(row["capped_work_mean"] / row["capped_depth_mean"])))
    wide["v4_data_B_total"] = v4_batch
    wide["current_data_B_total"] = 8
    wide.to_csv(OUT / "epsilon_threshold_comparison.csv", index=False, float_format="%.10g")
    make_markdown_table(wide)
    plot(all_rows)


if __name__ == "__main__":
    main()
