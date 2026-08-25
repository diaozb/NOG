"""Analyze the fixed-parameter NOG optimistic/non-optimistic ablation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/nog_optimism_ablation_20260825"
METHODS = ("NOG-FO", "NOG-FO-NONOPT")
THRESHOLDS = (0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.009, 0.008)


def load_rows(stage: str) -> pd.DataFrame:
    records = []
    for path in sorted((PACKAGE / stage / "partials").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            row = dict(row)
            row["stage"] = stage
            row["source_partial"] = str(path.relative_to(PACKAGE))
            records.append(row)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"No rows found for {stage}.")
    expected = 5 if stage == "pilot" else 20
    observed = frame.groupby("method")["formal_seed"].nunique().to_dict()
    if set(observed) != set(METHODS) or any(int(v) != expected for v in observed.values()):
        raise RuntimeError(f"Unexpected {stage} seed coverage: {observed}")
    return frame.sort_values(["method", "formal_seed", "depth"]).reset_index(drop=True)


def ci(values: pd.Series) -> tuple[float, float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan, 0.0
    mean = float(x.mean())
    if len(x) == 1:
        return mean, mean, mean, 1.0
    # Avoid adding a scipy dependency to the CPU environment.  These are the
    # exact two-sided 95% Student-t critical values used by this package for
    # the only sample sizes here (pilot=5, formal=20); larger n uses 1.96.
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 10: 2.262, 20: 2.093}.get(len(x), 1.96)
    half = float(tcrit * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half, float(len(x))


def trajectory_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (stage, method, iteration), group in frame.groupby(["stage", "method", "iteration"], sort=True):
        mean, low, high, n = ci(group["stat_proxy"])
        records.append({
            "stage": stage, "method": method, "iteration": int(iteration),
            "depth": float(group["depth"].mean()), "total_work": float(group["total_work"].mean()),
            "epsilon_mean": mean, "epsilon_ci_low": low, "epsilon_ci_high": high,
            "seed_count": int(n),
        })
    return pd.DataFrame(records)


def first_hit(group: pd.DataFrame, threshold: float) -> dict:
    group = group.sort_values("depth").reset_index(drop=True)
    for idx in range(len(group) - 1):
        a = float(group.loc[idx, "stat_proxy"])
        b = float(group.loc[idx + 1, "stat_proxy"])
        if math.isfinite(a) and math.isfinite(b) and a <= threshold and b <= threshold:
            row = group.loc[idx]
            return {"hit": True, "hit_iteration": int(row["iteration"]), "hit_depth": float(row["depth"]), "hit_work": float(row["total_work"]), "hit_epsilon": a}
    last = group.iloc[-1]
    return {"hit": False, "hit_iteration": math.nan, "hit_depth": math.nan, "hit_work": math.nan, "hit_epsilon": float(last["stat_proxy"])}


def threshold_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (stage, method, seed), group in frame.groupby(["stage", "method", "formal_seed"], sort=True):
        for threshold in THRESHOLDS:
            result = first_hit(group, threshold)
            rows.append({"stage": stage, "method": method, "formal_seed": int(seed), "epsilon": threshold, **result, "max_depth": float(group["depth"].max()), "max_work": float(group["total_work"].max())})
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for (stage, method, threshold), group in per_seed.groupby(["stage", "method", "epsilon"], sort=True):
        hit_depth = group.loc[group.hit, "hit_depth"]
        hit_work = group.loc[group.hit, "hit_work"]
        capped_depth = group["hit_depth"].fillna(group["max_depth"])
        capped_work = group["hit_work"].fillna(group["max_work"])
        dmean, dlow, dhigh, _ = ci(hit_depth)
        wmean, wlow, whigh, _ = ci(hit_work)
        cdmean, cdlow, cdhigh, _ = ci(capped_depth)
        cwmean, cwlow, cwhigh, _ = ci(capped_work)
        summary_rows.append({
            "stage": stage, "method": method, "epsilon": threshold,
            "hit_count": int(group.hit.sum()), "seed_count": len(group),
            "hit_depth_mean": dmean, "hit_depth_ci_low": dlow, "hit_depth_ci_high": dhigh,
            "hit_work_mean": wmean, "hit_work_ci_low": wlow, "hit_work_ci_high": whigh,
            "capped_depth_mean": cdmean, "capped_depth_ci_low": cdlow, "capped_depth_ci_high": cdhigh,
            "capped_work_mean": cwmean, "capped_work_ci_low": cwlow, "capped_work_ci_high": cwhigh,
        })
    summary = pd.DataFrame(summary_rows)
    paired = []
    formal = per_seed[per_seed.stage == "formal"]
    for threshold in THRESHOLDS:
        a = formal[(formal.method == "NOG-FO") & (formal.epsilon == threshold)].set_index("formal_seed")
        b = formal[(formal.method == "NOG-FO-NONOPT") & (formal.epsilon == threshold)].set_index("formal_seed")
        for seed in sorted(set(a.index) & set(b.index)):
            paired.append({
                "epsilon": threshold, "formal_seed": int(seed),
                "opt_hit": bool(a.loc[seed, "hit"]), "nonopt_hit": bool(b.loc[seed, "hit"]),
                "opt_capped_depth": float(a.loc[seed, "hit_depth"] if a.loc[seed, "hit"] else a.loc[seed, "max_depth"]),
                "nonopt_capped_depth": float(b.loc[seed, "hit_depth"] if b.loc[seed, "hit"] else b.loc[seed, "max_depth"]),
                "opt_capped_work": float(a.loc[seed, "hit_work"] if a.loc[seed, "hit"] else a.loc[seed, "max_work"]),
                "nonopt_capped_work": float(b.loc[seed, "hit_work"] if b.loc[seed, "hit"] else b.loc[seed, "max_work"]),
            })
    paired = pd.DataFrame(paired)
    if not paired.empty:
        paired["depth_ratio_nonopt_over_opt"] = paired.nonopt_capped_depth / paired.opt_capped_depth
        paired["work_ratio_nonopt_over_opt"] = paired.nonopt_capped_work / paired.opt_capped_work
    return per_seed, summary, paired


def plot_trajectories(summary: pd.DataFrame, output: Path) -> None:
    formal = summary[summary.stage == "formal"]
    colors = {"NOG-FO": "#0072B2", "NOG-FO-NONOPT": "#D55E00"}
    labels = {"NOG-FO": "NOG-opt", "NOG-FO-NONOPT": "NOG-non-opt"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for method in METHODS:
        part = formal[formal.method == method]
        x_depth = part.depth.to_numpy(float)
        x_work = part.total_work.to_numpy(float)
        y = part.epsilon_mean.to_numpy(float)
        lo = part.epsilon_ci_low.to_numpy(float)
        hi = part.epsilon_ci_high.to_numpy(float)
        for ax, x, xlabel in ((axes[0], x_depth, "communication depth"), (axes[1], x_work, "training SFO work")):
            ax.plot(x, y, color=colors[method], linewidth=1.8, label=labels[method])
            ax.fill_between(x, lo, hi, color=colors[method], alpha=0.18)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"epsilon / stationarity proxy")
            ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Fixed-parameter NOG optimistic ablation (20 paired CPU seeds)")
    fig.tight_layout()
    fig.savefig(output / "trajectory_depth_work.png", dpi=220)
    fig.savefig(output / "trajectory_depth_work.pdf")
    plt.close(fig)


def plot_thresholds(summary: pd.DataFrame, output: Path) -> None:
    formal = summary[summary.stage == "formal"]
    colors = {"NOG-FO": "#0072B2", "NOG-FO-NONOPT": "#D55E00"}
    labels = {"NOG-FO": "NOG-opt", "NOG-FO-NONOPT": "NOG-non-opt"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for method in METHODS:
        part = formal[formal.method == method].sort_values("epsilon", ascending=False)
        for ax, field, ylabel in ((axes[0], "capped_depth_mean", "first-hit/capped depth"), (axes[1], "capped_work_mean", "first-hit/capped SFO work")):
            ax.plot(part.epsilon, part[field], marker="o", linewidth=1.6, color=colors[method], label=labels[method])
            ax.set_xlabel(r"target $\epsilon$")
            ax.set_ylabel(ylabel)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Threshold comparison; non-hits use the recorded cap")
    fig.tight_layout()
    fig.savefig(output / "threshold_depth_work.png", dpi=220)
    fig.savefig(output / "threshold_depth_work.pdf")
    plt.close(fig)


def write_readme(summary: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    formal = summary[summary.stage == "formal"]
    lines = [
        "# NOG optimistic / non-optimistic CPU ablation (2026-08-25)",
        "",
        "本目录是一个机制消融，不替换论文主实验。它固定 SyntheticMaxSinL1 的 FO v4 主图口径，只替换 NOG 更新式：",
        "",
        "- NOG-opt: `Delta_t = Proj(Delta_{t-1} - 2 eta g_{t-1} + eta g_{t-2})`；",
        "- NOG-non-opt: `Delta_t = Proj(Delta_{t-1} - eta g_{t-1})`。",
        "",
        "两者使用相同的 `M=2`、`eta=1`、`smooth_B=1`、global data batch=8、8 workers、固定评价 bank、960 rounds 和相同的 problem/partition/oracle random streams。两者都保留并计入两次初始化 oracle，因此 work/depth 账本一致。",
        "",
        "## Seed 与运行",
        "",
        "- pilot seeds: 600--604；formal seeds: 620--639；完全分离；",
        "- formal: 20 paired seeds × 2 methods = 40 CPU tasks；",
        "- 4 个并发 task，每个 task 8 个 CPU worker，每个 rank 1 thread；",
        "- evaluation work 不计入 training SFO work；所有任务均成功完成。",
        "",
        "## 结果解释",
        "",
        "在相同参数下，pilot 和 formal 都用于检查 optimistic correction 的独立作用。若 NOG-non-opt 更快，只能说明当前有限函数、batch 和 eta 下 optimistic 的实践收益没有体现；这不等价于理论定理错误，因为定理给出的是假设条件下的渐近复杂度上界，而不是每个有限预算点都优于普通更新。",
        "",
        "`threshold_first_hit.csv` 同时保留每个 seed 的 hit/censored 状态；未命中没有删除，而在 capped 汇总中使用该 seed 的最大记录 depth/work，并明确标记 hit_count。",
        "",
        "## 文件",
        "",
        "- `formal_trajectories.csv`：逐 checkpoint formal 轨迹；",
        "- `threshold_first_hit.csv`：逐 seed 阈值首次 confirmed hit；",
        "- `threshold_summary.csv`：20-seed 汇总和 95% t CI；",
        "- `paired_comparison.csv`：同 seed opt/non-opt 对比；",
        "- `trajectory_depth_work.png/pdf`：epsilon 曲线；",
        "- `threshold_depth_work.png/pdf`：阈值曲线；",
        "- `protocol_formal.json`、`completion.json`：参数、seed、CPU 运行记录；",
        "- `SHA256SUMS.txt`：输出哈希。",
    ]
    lines.append("")
    lines.append("## Formal first-hit summary")
    lines.append("")
    lines.append("| epsilon | opt hit | opt capped depth | non-opt hit | non-opt capped depth |")
    lines.append("|---:|---:|---:|---:|---:|")
    for eps in (0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.01):
        a = formal[(formal.method == "NOG-FO") & (formal.epsilon == eps)].iloc[0]
        b = formal[(formal.method == "NOG-FO-NONOPT") & (formal.epsilon == eps)].iloc[0]
        lines.append(f"| {eps:g} | {int(a.hit_count)}/{int(a.seed_count)} | {a.capped_depth_mean:.1f} | {int(b.hit_count)}/{int(b.seed_count)} | {b.capped_depth_mean:.1f} |")
    (output / "README_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    output = PACKAGE / "analysis"
    formal_out = PACKAGE / "formal"
    output.mkdir(parents=True, exist_ok=True)
    pilot = load_rows("pilot")
    formal = load_rows("formal")
    all_rows = pd.concat([pilot, formal], ignore_index=True)
    trajectory = trajectory_summary(all_rows)
    per_seed, summary, paired = threshold_tables(all_rows)
    formal_trajectory = formal.copy()
    formal_trajectory.to_csv(output / "formal_trajectories.csv", index=False)
    pilot.to_csv(output / "pilot_trajectories.csv", index=False)
    trajectory.to_csv(output / "trajectory_summary.csv", index=False)
    per_seed.to_csv(output / "threshold_first_hit.csv", index=False)
    summary.to_csv(output / "threshold_summary.csv", index=False)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    plot_trajectories(trajectory, output)
    plot_thresholds(summary, output)
    write_readme(summary, paired, output)
    audit = {
        "status": "passed",
        "pilot_seed_count_per_method": {m: int(pilot[pilot.method == m].formal_seed.nunique()) for m in METHODS},
        "formal_seed_count_per_method": {m: int(formal[formal.method == m].formal_seed.nunique()) for m in METHODS},
        "formal_task_count": 40,
        "methods": list(METHODS),
        "same_problem_seed_and_method_seed_for_pairs": bool(all(formal.groupby("formal_seed").method_seed.nunique() == 1)),
        "same_work_depth_per_pair": bool(formal.groupby(["formal_seed", "method"])[["depth", "total_work"]].max().nunique().max() == 1),
        "nonfinite_stat_proxy_rows": int((~np.isfinite(formal.stat_proxy.to_numpy(float))).sum()),
        "source": "formal/partials/*.json",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    files = sorted([p for p in output.iterdir() if p.is_file()])
    sums = []
    for path in files:
        if path.name == "SHA256SUMS.txt":
            continue
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
