"""Render wide-epsilon figures and a censoring-aware final report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


def _csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _save(fig: Any, root: Path, name: str) -> List[Path]:
    paths = [root / f"{name}.png", root / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def render_figures(root: Path) -> List[Path]:
    analysis = root / "analysis"
    figure_root = analysis / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    summaries = _csv(analysis / "formal_summary.csv")
    ratios = _csv(analysis / "formal_ratios.csv")
    robust = _csv(analysis / "robustness_summary.csv")
    robust_ratios = _csv(analysis / "robustness_ratios.csv")
    colors = {"NOG-FO": "#1769aa", "ME-DOL-FO": "#d1495b"}
    outputs: List[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method in ("NOG-FO", "ME-DOL-FO"):
        rows = sorted(
            [row for row in summaries if row["method"] == method],
            key=lambda row: _number(row, "epsilon"),
        )
        x = [_number(row, "epsilon") for row in rows]
        y = [_number(row, "hit_rate") for row in rows]
        low = [_number(row, "hit_rate_ci_low") for row in rows]
        high = [_number(row, "hit_rate_ci_high") for row in rows]
        ax.plot(x, y, marker="o", linewidth=2, label=method, color=colors[method])
        ax.fill_between(x, low, high, alpha=0.16, color=colors[method])
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel(r"Stationarity threshold $\epsilon$ (decreasing $\rightarrow$)")
    ax.set_ylabel("Confirmed-hit rate (20 seeds)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    outputs.extend(_save(fig, figure_root, "formal_hit_rate_vs_epsilon"))

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True)
    x = [_number(row, "epsilon") for row in ratios]
    panels = (
        (axes[0], "depth_me_over_nog_ratio_of_capped_means", "ME-DOL / NOG depth (capped)", "depth_me_over_nog_paired_hit_count"),
        (axes[1], "work_nog_over_me_ratio_of_capped_means", "NOG / ME-DOL total work (capped)", "work_nog_over_me_paired_hit_count"),
    )
    for axis, field, label, count_field in panels:
        y = [_number(row, field) for row in ratios]
        full = [(left, right) for left, right, row in zip(x, y, ratios) if int(row[count_field]) == 20]
        censored = [(left, right) for left, right, row in zip(x, y, ratios) if int(row[count_field]) < 20]
        axis.plot(x, y, color="#444444", linewidth=1.5, alpha=0.65)
        axis.scatter([row[0] for row in full], [row[1] for row in full], color="#1769aa", label="20/20 paired hits")
        axis.scatter([row[0] for row in censored], [row[1] for row in censored], facecolors="none", edgecolors="#d1495b", s=55, label="censored")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"Stationarity threshold $\epsilon$ (decreasing $\rightarrow$)")
    outputs.extend(_save(fig, figure_root, "formal_depth_work_ratios"))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method in ("NOG-FO", "ME-DOL-FO"):
        rows = sorted(
            [row for row in summaries if row["method"] == method],
            key=lambda row: _number(row, "epsilon"),
        )
        x = [_number(row, "epsilon") for row in rows]
        for axis, field, label in (
            (axes[0], "depth_capped_mean", "Capped first-hit depth"),
            (axes[1], "total_work_capped_mean", "Capped first-hit total work"),
        ):
            axis.plot(x, [_number(row, field) for row in rows], marker="o", label=method, color=colors[method])
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.invert_xaxis()
            axis.set_xlabel(r"$\epsilon$")
            axis.set_ylabel(label)
            axis.grid(True, which="both", alpha=0.25)
    axes[0].legend()
    outputs.extend(_save(fig, figure_root, "formal_capped_depth_work"))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for axis, epsilon in zip(axes, (0.1, 0.01, 0.005)):
        for method in ("NOG-FO", "ME-DOL-FO"):
            rows = sorted(
                [row for row in robust if row["method"] == method and _number(row, "epsilon") == epsilon],
                key=lambda row: int(row["worker_count"]),
            )
            axis.plot(
                [int(row["worker_count"]) for row in rows],
                [_number(row, "hit_rate") for row in rows],
                marker="o",
                linewidth=2,
                label=method,
                color=colors[method],
            )
        axis.set_title(rf"$\epsilon={epsilon:g}$")
        axis.set_xlabel("workers m")
        axis.set_xticks([1, 2, 4, 8])
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Confirmed-hit rate (10 seeds)")
    axes[0].legend(fontsize=8)
    outputs.extend(_save(fig, figure_root, "robustness_hit_rate_by_workers"))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for epsilon, marker in ((0.1, "o"), (0.01, "s"), (0.005, "^")):
        rows = sorted(
            [row for row in robust_ratios if _number(row, "epsilon") == epsilon],
            key=lambda row: int(row["worker_count"]),
        )
        ax.plot(
            [int(row["worker_count"]) for row in rows],
            [_number(row, "depth_me_over_nog_ratio_of_capped_means") for row in rows],
            marker=marker,
            linewidth=2,
            label=rf"$\epsilon={epsilon:g}$",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("workers m")
    ax.set_ylabel("ME-DOL / NOG depth (capped)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    outputs.extend(_save(fig, figure_root, "robustness_depth_ratio_by_workers"))
    return outputs


def render_report(root: Path, figures: List[Path]) -> Dict[str, Any]:
    analysis = root / "analysis"
    summaries = _csv(analysis / "formal_summary.csv")
    ratios = _csv(analysis / "formal_ratios.csv")
    summary_lookup = {(row["method"], _number(row, "epsilon")): row for row in summaries}
    ratio_lookup = {_number(row, "epsilon"): row for row in ratios}
    lines = [
        "# Wide-epsilon NOG-FO vs ME-DOL-FO report",
        "",
        "## Protocol",
        "",
        "- epsilon: 0.2 down to 0.002 (17 values; log-scale figures)",
        "- primary: m=8, 20 formal seeds; pilot seeds 100-104 are disjoint",
        "- non-hits remain right-censored; capped mean, KM restricted mean, and lower bounds are always reported",
        "- robustness: m in {1,2,4,8}, epsilon in {0.1,0.01,0.005}, 10 seeds",
        "",
        "## Selected primary results",
        "",
        "| epsilon | NOG hits | ME-DOL hits | capped depth ME/NOG | status |",
        "|---:|---:|---:|---:|---|",
    ]
    for epsilon in (0.1, 0.03, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.002):
        nog = summary_lookup[("NOG-FO", epsilon)]
        me = summary_lookup[("ME-DOL-FO", epsilon)]
        ratio = ratio_lookup[epsilon]
        paired = int(ratio["depth_me_over_nog_paired_hit_count"])
        paired_text = "full" if paired == 20 else f"censored ({paired}/20 paired)"
        nog_hits = nog["hit_count"]
        me_hits = me["hit_count"]
        depth_ratio = float(ratio["depth_me_over_nog_ratio_of_capped_means"])
        lines.append(f"| {epsilon:g} | {nog_hits}/20 | {me_hits}/20 | {depth_ratio:.3g} | {paired_text} |")
    lines.extend(
        [
            "",
            "## Hypothesis decisions",
            "",
            "| Claim | Decision | Evidence |",
            "|---|---|---|",
            "| NOG depth advantage grows for demanding epsilon | **Partially supported** | At epsilon=0.009 and 0.008 both methods hit 20/20 and depth ME/NOG is about 12.4 and 15.3. The full trend is not monotone because frozen region changes and censoring create discontinuities. |",
            "| NOG/ME-DOL work ratio remains approximately constant | **Not supported** | The all-epsilon capped work-ratio coefficient of variation is about 1.11. |",
            "| ME-DOL non-hits are reported without blanks | **Supported** | Every non-hit contributes a censoring limit; capped and KM summaries are always present. |",
            "| Depth advantage is robust to worker count | **Supported at epsilon=0.01** | NOG hits 10/10 for every m; ME-DOL hits 0/10, 4/10, 7/10, 8/10. Capped depth ME/NOG is 6.84, 7.26, 5.39, 4.26. |",
            "| epsilon=0.005 has finite time-to-hit ratios | **Not supported** | Neither method hits for any worker count; only censoring lower bounds are valid. |",
            "",
            "## Recommended wording",
            "",
            "> Across 20 formal seeds, NOG-FO shows a substantial empirical communication-depth advantage at the fully observed thresholds epsilon=0.009 and 0.008, while maintaining higher hit rates at several more demanding thresholds. Worker-count robustness supports the advantage at epsilon=0.01. Results below the observed hitting range are reported as right-censored lower bounds rather than omitted or treated as hits.",
            "",
            "Do not claim constant work ratio, monotone growth across all 17 epsilon values, or hits at epsilon<=0.005.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in figures:
        if path.suffix == ".png":
            lines.append(f"- [{path.name}](figures/{path.name})")
    lines.extend(
        [
            "",
            "## 中文结论",
            "",
            "结果支持 NOG 在若干可观测 epsilon 上具有明显通信深度优势，且 epsilon=0.01 的优势对 m=1,2,4,8 稳健。结果不支持 work 比例在完整 epsilon 范围内基本不变；epsilon<=0.005 时两种方法均未命中，必须按删失下界报告。",
        ]
    )
    report_path = analysis / "epsilon_scaling_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "report": str(report_path.relative_to(root)),
        "figures": [
            {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
            for path in figures
        ],
        "hypotheses": {
            "depth_advantage": "partially-supported",
            "work_ratio_stability": "not-supported",
            "nonhit_reporting": "supported",
            "worker_robustness_at_0.01": "supported",
            "finite_comparison_at_0.005": "not-supported",
        },
    }
    atomic_write_json(analysis / "figure_report_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    figures = render_figures(root)
    manifest = render_report(root, figures)
    status = manifest["status"]
    report = manifest["report"]
    print(f"status={status} figures={len(figures)} report={report}")


if __name__ == "__main__":
    main()
