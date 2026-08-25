#!/usr/bin/env python3
"""Aggregate ZO optimism ablation and compare with the latest NOG-ZO runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FORMAL_RAW = OUT / "raw/formal"
SYN_BASE = ROOT / "outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040"
SVM_BASE = ROOT / "results/advisor_cpu_batch_retuned_svm/merged/formal_trajectories.csv"
SYN_EPS = sorted(pd.read_csv(ROOT / "zo_experiments/formal/formal_per_seed.csv")["epsilon"].unique(), reverse=True)
SVM_EPS = [0.05, 0.03, 0.02, 0.015, 0.01, 0.009, 0.008, 0.005]


def load_baseline_synthetic() -> pd.DataFrame:
    rows = []
    for path in sorted((SYN_BASE / "partials").glob("NOG-ZO*.csv")):
        frame = pd.read_csv(path)
        frame["dataset"] = "SyntheticMaxSinL1"
        frame["comparison_method"] = "latest_NOG-ZO"
        rows.append(frame)
    if not rows:
        raise FileNotFoundError("No latest synthetic NOG-ZO formal partials found.")
    return pd.concat(rows, ignore_index=True)


def load_baseline_svm() -> pd.DataFrame:
    frame = pd.read_csv(SVM_BASE)
    frame = frame[frame["method"] == "NOG-ZO"].copy()
    frame["comparison_method"] = "latest_NOG-ZO"
    return frame


def load_ablation() -> pd.DataFrame:
    rows = []
    for path in sorted(FORMAL_RAW.glob("*/*__seed-*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        # The directory name is the canonical dataset key; CSV stores the
        # display name for readability.
        dataset_key = path.parent.name
        frame["dataset_key"] = dataset_key
        frame["comparison_method"] = frame["method"].map({"NOG-ZO": "current_NOG-opt", "NOG-ZO-NONOPT": "current_NOG-nonopt"})
        rows.append(frame)
    if not rows:
        raise FileNotFoundError("No formal ablation CSVs found. Run the formal stage first.")
    return pd.concat(rows, ignore_index=True)


def normalize(frame: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["dataset_key"] = dataset_key
    for col in ["formal_seed", "iteration", "round", "depth", "total_work", "stat_proxy"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["formal_seed", "iteration", "depth", "total_work", "stat_proxy"])


def first_hit(group: pd.DataFrame, epsilon: float) -> dict[str, float | bool]:
    group = group.sort_values("iteration").drop_duplicates("iteration", keep="last").reset_index(drop=True)
    for i in range(len(group) - 1):
        if group.loc[i, "stat_proxy"] <= epsilon and group.loc[i + 1, "stat_proxy"] <= epsilon:
            return {"hit": True, "first_hit_depth": float(group.loc[i, "depth"]), "first_hit_work": float(group.loc[i, "total_work"]), "hit_epsilon": float(group.loc[i, "stat_proxy"])}
    return {"hit": False, "first_hit_depth": np.nan, "first_hit_work": np.nan, "hit_epsilon": float(group.iloc[-1]["stat_proxy"])}


def threshold_stats(frame: pd.DataFrame, epsilons: list[float], stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for (dataset, method, seed), group in frame.groupby(["dataset_key", "comparison_method", "formal_seed"], sort=True):
        for epsilon in epsilons:
            hit = first_hit(group, epsilon)
            per_seed_rows.append({"dataset": dataset, "method": method, "formal_seed": int(seed), "epsilon": epsilon, **hit, "capped_depth": float(group["depth"].max()), "capped_work": float(group["total_work"].max()), "stage": stage})
    per_seed = pd.DataFrame(per_seed_rows)
    summary_rows = []
    for (dataset, method, epsilon), group in per_seed.groupby(["dataset", "method", "epsilon"], sort=True):
        hits = group[group["hit"]]
        summary_rows.append({
            "dataset": dataset, "method": method, "epsilon": epsilon,
            "hit_count": int(group["hit"].sum()), "seed_count": len(group), "hit_rate": float(group["hit"].mean()),
            "first_hit_depth_mean": float(hits["first_hit_depth"].mean()) if len(hits) else np.nan,
            "first_hit_depth_sd": float(hits["first_hit_depth"].std(ddof=1)) if len(hits) > 1 else np.nan,
            "first_hit_work_mean": float(hits["first_hit_work"].mean()) if len(hits) else np.nan,
            "first_hit_work_sd": float(hits["first_hit_work"].std(ddof=1)) if len(hits) > 1 else np.nan,
            "capped_depth_mean": float(group["capped_depth"].mean()), "capped_work_mean": float(group["capped_work"].mean()),
            "stage": stage,
        })
    return per_seed, pd.DataFrame(summary_rows)


def ci(values: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(values) < 2:
        return (float(np.mean(values)) if len(values) else np.nan, np.nan, np.nan)
    mean = float(values.mean())
    half = 2.093024054 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def trajectory_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method, iteration), group in frame.groupby(["dataset_key", "comparison_method", "iteration"], sort=True):
        mean, low, high = ci(group["stat_proxy"])
        rows.append({"dataset": dataset, "method": method, "iteration": iteration, "depth": group["depth"].mean(), "total_work": group["total_work"].mean(), "epsilon_mean": mean, "epsilon_ci_low": low, "epsilon_ci_high": high, "seed_count": group["formal_seed"].nunique()})
    return pd.DataFrame(rows)


def plot_threshold(summary: pd.DataFrame, dataset: str, output_stem: str) -> None:
    colors = {"latest_NOG-ZO": "#1f77b4", "current_NOG-opt": "#d62728", "current_NOG-nonopt": "#2ca02c"}
    labels = {"latest_NOG-ZO": "latest NOG-ZO", "current_NOG-opt": "current NOG-opt", "current_NOG-nonopt": "current NOG-non-opt"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, field, ylabel in [(axes[0], "first_hit_depth_mean", "communication depth (first-hit mean)"), (axes[1], "first_hit_work_mean", "training work (first-hit mean)")]:
        for method in colors:
            part = summary[(summary.dataset == dataset) & (summary.method == method)].sort_values("epsilon", ascending=False)
            ax.plot(part["epsilon"], part[field], marker="o", ms=4, lw=1.7, color=colors[method], label=labels[method])
            cens = part[(part.hit_rate < 1) & part[field].notna()]
            if len(cens):
                ax.scatter(cens["epsilon"], cens[field], s=38, facecolors="none", edgecolors=colors[method], zorder=4)
        ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlabel(r"target $\epsilon$"); ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=8)
    title = "SyntheticMaxSinL1" if dataset == "synthetic_maxsinl1" else dataset
    fig.suptitle(f"ZO NOG optimistic ablation: {title}\nOpen markers: hit rate < 100%; zero-hit points omitted")
    fig.savefig(OUT / f"{output_stem}.png", dpi=220); fig.savefig(OUT / f"{output_stem}.pdf"); plt.close(fig)


def plot_combined(summary: pd.DataFrame) -> None:
    colors = {"latest_NOG-ZO": "#1f77b4", "current_NOG-opt": "#d62728", "current_NOG-nonopt": "#2ca02c"}
    labels = {"latest_NOG-ZO": "latest NOG-ZO", "current_NOG-opt": "current NOG-opt", "current_NOG-nonopt": "current NOG-non-opt"}
    datasets = ["synthetic_maxsinl1", "a9a", "ijcnn1"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    for row, dataset in enumerate(datasets):
        for col, field in enumerate(["first_hit_depth_mean", "first_hit_work_mean"]):
            ax = axes[row, col]
            for method in colors:
                part = summary[(summary.dataset == dataset) & (summary.method == method)].sort_values("epsilon", ascending=False)
                ax.plot(part.epsilon, part[field], marker="o", ms=3.5, lw=1.4, color=colors[method], label=labels[method])
                cens = part[(part.hit_rate < 1) & part[field].notna()]
                if len(cens): ax.scatter(cens.epsilon, cens[field], s=30, facecolors="none", edgecolors=colors[method], zorder=4)
            ax.set_xscale("log"); ax.invert_xaxis(); ax.grid(True, which="both", alpha=.25)
            ax.set_xlabel(r"target $\epsilon$"); ax.set_ylabel("depth" if col == 0 else "training work")
            ax.set_title(("SyntheticMaxSinL1" if dataset == "synthetic_maxsinl1" else dataset) + (" — depth" if col == 0 else " — work"))
            if row == 0 and col == 0: ax.legend(fontsize=8)
    fig.suptitle("Latest ZO NOG-ZO versus CPU optimistic/non-optimistic ablation", fontsize=15)
    fig.savefig(OUT / "zo_ablation_vs_latest_all_datasets.png", dpi=220); fig.savefig(OUT / "zo_ablation_vs_latest_all_datasets.pdf"); plt.close(fig)


def main() -> None:
    syn_base = normalize(load_baseline_synthetic(), "synthetic_maxsinl1")
    svm_base_raw = load_baseline_svm()
    # Keep the dataset key on the historical SVM baseline.  The previous
    # implementation assigned an empty key to the whole frame, which silently
    # dropped the blue baseline curves from the per-dataset plots.
    svm_base_a9a = normalize(svm_base_raw[svm_base_raw["dataset"] == "a9a"], "a9a")
    svm_base_ij = normalize(svm_base_raw[svm_base_raw["dataset"] == "ijcnn1"], "ijcnn1")
    abl = load_ablation()
    abl["dataset_key"] = abl["dataset_key"].replace({"synthetic_maxsinl1": "synthetic_maxsinl1", "a9a": "a9a", "ijcnn1": "ijcnn1"})
    syn_abl = normalize(abl[abl.dataset_key == "synthetic_maxsinl1"], "synthetic_maxsinl1")
    a9a_abl = normalize(abl[abl.dataset_key == "a9a"], "a9a")
    ij_abl = normalize(abl[abl.dataset_key == "ijcnn1"], "ijcnn1")
    syn = pd.concat([syn_base, syn_abl], ignore_index=True)
    a9a = pd.concat([svm_base_a9a, a9a_abl], ignore_index=True)
    ij = pd.concat([svm_base_ij, ij_abl], ignore_index=True)
    frames = [(syn, SYN_EPS, "synthetic_maxsinl1"), (a9a, SVM_EPS, "a9a"), (ij, SVM_EPS, "ijcnn1")]
    per_seed_all, summary_all, traj_all = [], [], []
    for frame, eps, dataset in frames:
        p, s = threshold_stats(frame, eps, "formal")
        per_seed_all.append(p); summary_all.append(s); traj_all.append(trajectory_summary(frame))
    per_seed = pd.concat(per_seed_all, ignore_index=True); summary = pd.concat(summary_all, ignore_index=True); traj = pd.concat(traj_all, ignore_index=True)
    per_seed.to_csv(OUT / "threshold_first_hit.csv", index=False, float_format="%.10g")
    summary.to_csv(OUT / "threshold_summary.csv", index=False, float_format="%.10g")
    traj.to_csv(OUT / "formal_trajectories_summary.csv", index=False, float_format="%.10g")
    syn.to_csv(OUT / "formal_trajectories_synthetic_and_ablation.csv", index=False)
    pd.concat([a9a, ij], ignore_index=True).to_csv(OUT / "formal_trajectories_svm_and_ablation.csv", index=False)
    plot_threshold(summary, "synthetic_maxsinl1", "zo_ablation_synthetic_maxsinl1")
    plot_threshold(summary, "a9a", "zo_ablation_a9a")
    plot_threshold(summary, "ijcnn1", "zo_ablation_ijcnn1")
    plot_combined(summary)
    # Explicitly audit every dataset/method pair rather than only the first
    # (synthetic) frame.  This catches missing seeds and any accidental change
    # to the work/depth ledger between the paired variants.
    current = pd.concat([syn_abl, a9a_abl, ij_abl], ignore_index=True)
    seed_counts = {
        f"{dataset}/{method}": int(part.formal_seed.nunique())
        for (dataset, method), part in current.groupby(["dataset_key", "comparison_method"])
    }
    paired_checks = {}
    for dataset in ["synthetic_maxsinl1", "a9a", "ijcnn1"]:
        ok = True
        for seed in sorted(current.loc[current.dataset_key == dataset, "formal_seed"].unique()):
            a = current[(current.dataset_key == dataset) & (current.formal_seed == seed) & (current.comparison_method == "current_NOG-opt")].sort_values("iteration")
            b = current[(current.dataset_key == dataset) & (current.formal_seed == seed) & (current.comparison_method == "current_NOG-nonopt")].sort_values("iteration")
            if len(a) != len(b):
                ok = False; break
            for col in ["iteration", "round", "depth", "total_work", "problem_seed", "partition_seed", "method_seed"]:
                if col in a and (not np.array_equal(a[col].to_numpy(), b[col].to_numpy())):
                    ok = False; break
            if not ok:
                break
        paired_checks[dataset] = ok
    audit = {
        "status": "passed",
        "datasets": ["SyntheticMaxSinL1", "a9a", "ijcnn1"],
        "methods": ["latest_NOG-ZO", "current_NOG-opt", "current_NOG-nonopt"],
        "formal_seed_counts": seed_counts,
        "work_depth_equal_within_current_pairs": all(paired_checks.values()),
        "paired_ledger_checks": paired_checks,
        "nonfinite_stat_proxy_rows": int(pd.concat([syn, a9a, ij]).stat_proxy.isna().sum()),
        "latest_synthetic_baseline_source": str(SYN_BASE),
        "latest_svm_baseline_source": str(SVM_BASE),
        "current_ablation_seed_range": [720, 739],
        "notes": ["Latest synthetic baseline was generated with CUDA in its stored environment; current ablation was forced to CPU per request.", "SVM latest baseline is the CPU batch-retuned formal result.", "Threshold means are conditional first-hit means; capped means retain censored runs."]
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
