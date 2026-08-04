"""Render the auditable Chinese report for the symmetric low-epsilon v5 study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


def _json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


def _int(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):,.0f}"


def render_report(
    freeze_path: Path, analysis_root: Path, output_path: Path
) -> Dict[str, Any]:
    freeze = _json(freeze_path)
    trends_path = analysis_root / "formal_trends.json"
    ratios_path = analysis_root / "formal_ratios.csv"
    summary_path = analysis_root / "formal_summary.csv"
    fixed_path = analysis_root / "fixed_batch_diagnostics.csv"
    manifest_path = analysis_root / "analysis_manifest.json"
    trends = _json(trends_path)
    ratios = _csv(ratios_path)
    summary = _csv(summary_path)
    fixed = _csv(fixed_path)
    analysis_manifest = _json(manifest_path)
    if analysis_manifest.get("status") != "complete":
        raise ValueError("Low-epsilon analysis has not completed.")

    summary_lookup = {
        (row["method"], float(row["epsilon"]), row["scope"]): row
        for row in summary
    }
    primary = [row for row in ratios if row["scope"] == "primary"]
    diagnostics = [row for row in ratios if row["scope"] == "diagnostic_midpoint"]
    verdict = trends["verdict"]
    status = "supported" if verdict["low_epsilon_claim_supported"] else "not fully supported"
    nog = freeze["selected_algorithms"]["NOG-FO"]
    me = freeze["selected_algorithms"]["ME-DOL-FO"]

    lines = [
        "# NOG 低 epsilon 对称验证实验（v5）",
        "",
        "## 结论",
        "",
        (
            f"本轮在 {len(primary)} 个预注册阈值（0.01000 到 0.00400）上运行；"
            f"异常确认后每点使用 {int(trends['formal_seed_count'])} 个独立 formal seeds。"
        ),
        (
            f"预注册 schedule 下，`ME-DOL/NOG depth` 从 "
            f"{float(trends['depth_ratio_start']):.3f}x 变化到 "
            f"{float(trends['depth_ratio_end']):.3f}x，Spearman rho="
            f"{float(trends['depth_ratio']['spearman_rho']):.3f}。"
        ),
        (
            f"`NOG/ME-DOL work` 均值 {float(trends['work_ratio_mean']):.3f}x，"
            f"范围 {float(trends['work_ratio_min']):.3f}--"
            f"{float(trends['work_ratio_max']):.3f}x，CV="
            f"{float(trends['work_ratio_coefficient_of_variation']):.3f}。"
        ),
        f"按 formal 运行前冻结的全部门槛，总 verdict 为 **{status}**。",
        "",
        "这一 verdict 不因结果方向而调整。固定 batch 诊断用于解释 schedule 切换造成的跳变，"
        "不能替代预注册主结果。",
        "",
        "## 预注册主结果",
        "",
        "比例方向：`ME/NOG depth > 1` 表示 NOG 通信 depth 更低；"
        "`NOG/ME work` 越接近 1 表示训练 oracle work 越匹配。",
        "",
        "| ε | N batch | ME batch | N hit | ME hit | N depth | ME depth | ME/N depth | N work | ME work | N/ME work |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        epsilon = float(row["epsilon"])
        scope = row["scope"]
        n = summary_lookup[("NOG-FO", epsilon, scope)]
        m = summary_lookup[("ME-DOL-FO", epsilon, scope)]
        marker = "†" if row["ratios_use_capped_values"] == "True" else ""
        lines.append(
            f"| {epsilon:.5f} | {int(row['nog_batch_total'])} | {int(row['me_batch_total'])} | "
            f"{int(n['hit_count'])}/{int(n['num_seeds'])} | {int(m['hit_count'])}/{int(m['num_seeds'])} | "
            f"{_fmt(n['depth_mean_hits'], 1)} ± {_fmt(n['depth_sd_hits'], 1)} | "
            f"{_fmt(m['depth_mean_hits'], 1)} ± {_fmt(m['depth_sd_hits'], 1)} | "
            f"{_fmt(row['depth_ratio_mean'], 3)}x{marker} | "
            f"{_int(n['work_mean_hits'])} ± {_int(n['work_sd_hits'])} | "
            f"{_int(m['work_mean_hits'])} ± {_int(m['work_sd_hits'])} | "
            f"{_fmt(row['work_ratio_mean'], 3)}x{marker} |"
        )
    lines.extend([
        "",
        "† 表示存在 non-hit，比例使用预注册最大预算的 capped 值；所有 non-hit 均保留。",
        "",
        "![低 epsilon 主比例](figures/low_epsilon_ratios.png)",
        "",
        "![低 epsilon hit rate](figures/low_epsilon_hit_rates.png)",
        "",
        "## 参数如何冻结",
        "",
        f"- NOG：`M={int(nog['M'])}, eta={float(nog['eta']):g}, smooth_B=1`。",
        f"- ME-DOL：`H={int(me['epoch_length'])}, multiplier={float(me['theory_multiplier']):g}, smooth_B=1`。",
        "- 双方各 6 个算法候选，使用相同 pilot seeds 110--114、相同总 batch 128、"
        "相同 15360 training rounds；先最大化连续 5/5 hit 范围，再按总 hits 和 first-hit work 选择。",
        "- 第二阶段双方使用相同总 batch 网格 32/64/96/128/192/256，分别选择 5/5 hit 后"
        " first-hit work 最小、且 ε 变小时不下降的 schedule。",
        "- 统一每 24 iterations 评估。NOG 两次初始化 oracle/all-reduce 计入通信 depth，"
        "所以最大记录 depth 为 15362；ME-DOL 为 15360。",
        "",
        "## 异常确认",
        "",
        f"初始 20 seeds 发现 {len(trends['initial_adjacent_depth_drop_anomalies'])} 个超过 20% 的相邻下降；"
        f"因此按冻结协议追加 seeds 40--49：{'是' if trends['extra_confirmation_used'] else '否'}。",
        f"30 seeds 复核后仍存在 {len(trends['adjacent_depth_drop_anomalies'])} 个超过 20% 的相邻下降。",
        "参数和 batch schedule 在确认阶段均未改变。",
        "",
    ])
    if diagnostics:
        lines.extend([
            "### 自动加入的诊断中点",
            "",
            "| ε | N batch | ME batch | paired hit | ME/N depth | N/ME work |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for row in diagnostics:
            lines.append(
                f"| {float(row['epsilon']):.6f} | {int(row['nog_batch_total'])} | "
                f"{int(row['me_batch_total'])} | {int(row['paired_hit_count'])}/{int(row['num_seeds'])} | "
                f"{_fmt(row['depth_ratio_mean'], 3)}x | {_fmt(row['work_ratio_mean'], 3)}x |"
            )
        lines.append("")

    full_fixed = [row for row in fixed if int(row["paired_hit_count"]) == int(row["num_seeds"])]
    lines.extend([
        "## 固定共同 batch 诊断",
        "",
        "下图仅使用双方 formal 中都已运行的共同总 batch 32/64/96/128。"
        "每条线内部不切换 batch，因此可以判断主曲线跳变是否来自 schedule 边界。"
        "这是预先冻结数据的补充切片，不重新选择参数，也不替代主 verdict。",
        "",
        f"共有 {len(full_fixed)} 个 batch/ε 组合达到双方全部 seeds paired hit。",
        "",
        "![固定 batch 诊断](figures/fixed_batch_diagnostics.png)",
        "",
        "## 与 v4 的关系",
        "",
        "v4 使用 NOG `M=2, eta=1`、ME-DOL `H=6, multiplier=100`，并按 matched-work 目标"
        "单独校准 NOG batch。v5 改用完全对称的两阶段调参、新 pilot/formal seeds、共同评估网格"
        "和更高预算，因此数值不能与 v4 当成同一参数曲线直接拼接。v4 保留为历史基线，v5 是独立验证。",
        "",
        "## 可复现性与限制",
        "",
        "- Pilot seeds 110--114；formal seeds 20--39；异常确认 seeds 40--49，集合严格不相交。",
        "- 参数冻结发生在 v5 formal partial 生成之前；任务支持 SHA/fingerprint 验证和断点恢复。",
        "- 每次 hit 要求连续两个 high-precision checkpoints 达标。",
        "- 最多 4 个任务并行，每任务 8 workers，总计不超过 32 worker processes。",
        "- evaluation work 不计入训练 SFO work，但对双方使用同一固定 high-precision bank。",
        "- 结果只支持当前 synthetic problem family 和冻结协议，不能推出所有问题上的普适优越性。",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    inputs = [freeze_path, trends_path, ratios_path, summary_path, fixed_path, manifest_path]
    report_manifest = {
        "schema_version": 2,
        "status": "complete",
        "created_at_utc": utc_now(),
        "verdict": verdict,
        "report_path": str(output_path),
        "report_sha256": file_sha256(output_path),
        "input_artifacts": [{"path": str(path), "sha256": file_sha256(path)} for path in inputs],
    }
    atomic_write_json(analysis_root / "report_manifest.json", report_manifest)
    return report_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric")
    parser.add_argument("--freeze", default=str(root / "frozen_parameters.json"))
    parser.add_argument("--analysis-root", default=str(root / "analysis"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    analysis_root = Path(args.analysis_root)
    result = render_report(Path(args.freeze), analysis_root, Path(args.output or analysis_root / "low_epsilon_report.md"))
    print(f"status={result['status']} verdict={result['verdict']}")


if __name__ == "__main__":
    main()
