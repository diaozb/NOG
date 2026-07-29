"""Render the Chinese report and all-epsilon hit-rate figure from formal statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


def _json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: str | float | None, digits: int = 2) -> str:
    if value in (None, "", "None"):
        return "—"
    return f"{float(value):.{digits}f}"


def _hit_figure(summary: List[Dict[str, str]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for method, color in [("NOG-FO", "#1f77b4"), ("ME-DOL-FO", "#d62728")]:
        rows = [row for row in summary if row["method"] == method]
        axis.plot(
            [float(row["epsilon"]) for row in rows],
            [float(row["hit_rate"]) for row in rows],
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=method,
            color=color,
        )
    axis.axvline(0.01, linestyle="--", linewidth=0.9, color="black", label="primary boundary")
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)")
    axis.set_ylabel("confirmed-hit rate")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_report(analysis_root: Path, freeze_path: Path) -> Dict[str, Any]:
    freeze = _json(freeze_path)
    trends = _json(analysis_root / "formal_trends.json")
    summary = _csv(analysis_root / "formal_summary.csv")
    ratios = _csv(analysis_root / "formal_ratios.csv")
    ratio_by_epsilon = {float(row["epsilon"]): row for row in ratios}
    summary_by_key = {
        (row["method"], float(row["epsilon"])): row for row in summary
    }
    hit_path = analysis_root / "figures" / "hit_rate_vs_epsilon.png"
    _hit_figure(summary, hit_path)

    primary_lines = [
        "| epsilon | NOG batch | NOG hit | ME-DOL hit | ME/NOG depth | NOG/ME work |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for epsilon in freeze["primary_epsilons"]:
        epsilon = float(epsilon)
        ratio = ratio_by_epsilon[epsilon]
        nog = summary_by_key[("NOG-FO", epsilon)]
        me = summary_by_key[("ME-DOL-FO", epsilon)]
        primary_lines.append(
            "| {eps:g} | {batch} | {nh}/20 | {mh}/20 | {depth}x | {work}x |".format(
                eps=epsilon,
                batch=ratio["data_B_total"],
                nh=nog["hit_count"],
                mh=me["hit_count"],
                depth=_fmt(ratio["depth_ratio_mean"]),
                work=_fmt(ratio["work_ratio_mean"]),
            )
        )

    exploratory_lines = [
        "| epsilon | NOG hit | ME-DOL hit | NOG capped depth | ME capped depth |",
        "|---:|---:|---:|---:|---:|",
    ]
    exploratory_eps = sorted(
        {
            float(row["epsilon"])
            for row in summary
            if row["scope"] == "exploratory_censored"
        },
        reverse=True,
    )
    for epsilon in exploratory_eps:
        nog = summary_by_key[("NOG-FO", epsilon)]
        me = summary_by_key[("ME-DOL-FO", epsilon)]
        exploratory_lines.append(
            "| {eps:g} | {nh}/20 | {mh}/20 | {nd} | {md} |".format(
                eps=epsilon,
                nh=nog["hit_count"],
                mh=me["hit_count"],
                nd=_fmt(nog["capped_depth_mean"], 1),
                md=_fmt(me["capped_depth_mean"], 1),
            )
        )

    verdict = trends["verdict"]
    observed = trends["observed_exponents"]
    theory = trends["theory_reference_exponents"]
    report = [
        "# NOG 宽 epsilon 理论验证实验（v4）",
        "",
        "## 结论",
        "",
        (
            f"在 27 个 primary epsilon（0.2 到 0.01）和每点 20 个独立 formal seeds 上，"
            f"ME-DOL/NOG depth 比例的 Spearman rho 为 "
            f"{trends['depth_ratio']['spearman_rho']:.3f}。"
        ),
        (
            f"NOG/ME-DOL work 比例均值为 {trends['work_ratio_mean']:.2f}x，"
            f"范围 {trends['work_ratio_min']:.2f}--{trends['work_ratio_max']:.2f}x，"
            f"CV={trends['work_ratio_coefficient_of_variation']:.3f}。"
        ),
        (
            "预先冻结判据的总 verdict："
            f"**{'supported' if verdict['primary_claim_supported'] else 'not fully supported'}**。"
        ),
        "",
        "这里的正确解释是有限区间中的定性 scaling evidence。固定 d、delta 和单一 problem family，不能据此声称已经精确验证渐进指数。",
        "",
        "## Primary 结果（完整保留 27 个点）",
        "",
        *primary_lines,
        "",
        "比例是对 paired formal seeds 计算的均值；若任一方法未命中则使用预注册上限的 capped 值，并在 formal_ratios.csv 中标记。",
        "",
        "![Depth/work ratios](figures/depth_work_ratios.png)",
        "",
        "![Depth/work versus epsilon](figures/depth_work_vs_epsilon.png)",
        "",
        "## 0.01 以下 exploratory/censored 结果",
        "",
        "这些点不参与 primary 参数选择或主趋势 verdict。未命中不会留空，而是保留 hit rate 和最大预算下界。",
        "",
        *exploratory_lines,
        "",
        "![Hit rate](figures/hit_rate_vs_epsilon.png)",
        "",
        "## 理论参照与观测斜率",
        "",
        "| metric | theory exponent | observed log-log slope |",
        "|---|---:|---:|",
        f"| NOG depth | {theory['NOG_depth']:.3f} | {observed['NOG_depth']:.3f} |",
        f"| ME-DOL depth | {theory['ME_DOL_depth']:.3f} | {observed['ME_DOL_depth']:.3f} |",
        f"| NOG work | {theory['NOG_work']:.3f} | {observed['NOG_work']:.3f} |",
        f"| ME-DOL work | {theory['ME_DOL_work']:.3f} | {observed['ME_DOL_work']:.3f} |",
        "",
        "论文的 first-order 结论是 NOG depth 为 O(epsilon^-5/3)、ME-DOL depth 为 O(epsilon^-3)，两者 work 同为 O(epsilon^-3)。本实验采用 pilot-calibrated matched-work batch schedule 来检验有限区间中的方向性，而不是把理论常数当成已知。",
        "",
        "## 可复现性与防止事后挑结果",
        "",
        "- Pilot seeds 100--104；formal seeds 0--19，严格不重叠。",
        "- NOG batch grid 为 8,16,...,64；只允许 epsilon 变小时 batch 不下降。",
        "- 参数冻结发生在任何 formal partial 生成之前。",
        "- 每个 hit 要求连续两个 high-precision checkpoints 达标。",
        "- 所有 raw pilot/formal 输入的 SHA256 均记录在 frozen_parameters.json 和 analysis_manifest.json。",
        "- 并发上限是 4 个任务 × 8 workers = 32 个 worker processes。",
    ]
    report_path = analysis_root / "theory_validation_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    inputs = [
        analysis_root / "formal_summary.csv",
        analysis_root / "formal_ratios.csv",
        analysis_root / "formal_trends.json",
        freeze_path,
    ]
    manifest = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "primary_rows": len(ratios),
        "exploratory_epsilon_count": len(exploratory_eps),
        "verdict": verdict,
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in inputs],
        "outputs": [
            {"path": str(report_path), "sha256": file_sha256(report_path)},
            {"path": str(hit_path), "sha256": file_sha256(hit_path)},
            {"path": str(hit_path.with_suffix('.pdf')), "sha256": file_sha256(hit_path.with_suffix('.pdf'))},
        ],
    }
    atomic_write_json(analysis_root / "report_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/analysis",
    )
    parser.add_argument(
        "--freeze",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/frozen_parameters.json",
    )
    args = parser.parse_args()
    result = build_report(Path(args.analysis_root), Path(args.freeze))
    print(
        f"status={result['status']} primary_rows={result['primary_rows']} "
        f"exploratory={result['exploratory_epsilon_count']}"
    )


if __name__ == "__main__":
    main()
