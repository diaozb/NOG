"""Generate the root-level, audit-backed report for all completed ZO experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ZO_ALL_EXPERIMENT_RESULTS.md"
METHODS = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]
BASELINES = ["ME-DOL-ZO", "DGFM", "DGFM+"]


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def number(value: str | float | int | None, digits: int = 1) -> str:
    if value in (None, ""):
        return "—"
    numeric = float(value)
    return f"{numeric:,.{digits}f}"


def ratio(value: str | float | int | None) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.3f}"


def epsilon(value: str | float) -> str:
    numeric = float(value)
    if numeric >= 0.1:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}"


def ci(row: dict[str, str], stem: str) -> str:
    mean = row.get(f"mean_{stem}", "")
    low = row.get(f"{stem}_ci_low", "")
    high = row.get(f"{stem}_ci_high", "")
    if not mean:
        return "—"
    return f"{ratio(mean)} [{ratio(low)}, {ratio(high)}]"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join(["---"] * len(headers)) + "|")
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def hit_cell(row: dict[str, str], *, include_per_worker: bool = False) -> str:
    hits = int(float(row["hits"]))
    seeds = int(float(row["seeds"]))
    marker = "†" if hits < seeds else ""
    if hits == 0:
        if include_per_worker:
            return f"{hits}/{seeds}{marker}; —; —; —"
        return f"{hits}/{seeds}{marker}; —; —"
    cell = (
        f"{hits}/{seeds}{marker}; "
        f"{number(row['mean_first_hit_depth'])}; "
        f"{number(row['mean_first_hit_work'])}"
    )
    if include_per_worker:
        cell += f"; {number(row['mean_first_hit_per_worker_work'])}"
    return cell


def link(path: str, label: str | None = None) -> str:
    return f"[{label or Path(path).name}]({path})"


def generate() -> str:
    formal_summary = read_csv("zo_experiments/formal/formal_summary.csv")
    formal_ratios = read_csv("zo_experiments/formal/formal_ratios.csv")
    theory = read_json("zo_experiments/formal/theory_comparison.json")
    anomaly = read_csv("zo_experiments/formal/anomaly_method_summary.csv")
    replication = read_csv("zo_experiments/formal/anomaly_replication_comparison.csv")
    dimension_summary = read_csv("zo_experiments/dimension/dimension_summary.csv")
    dimension_ratios = read_csv("zo_experiments/dimension/dimension_ratios.csv")
    dimension_trends = read_json("zo_experiments/dimension/dimension_trends.json")
    worker_summary = read_csv("zo_experiments/worker/worker_summary.csv")
    worker_ratios = read_csv("zo_experiments/worker/worker_relative_to_m1.csv")
    worker_trends = read_json("zo_experiments/worker/worker_trends.json")
    equivalence = read_csv("zo_experiments/process_equivalence/equivalence_summary.csv")
    real_summary = read_csv("zo_experiments/real_data/formal/summary.csv")

    lines: list[str] = []
    add = lines.append
    extend = lines.extend

    add("# ZO 全部实验设置、方法与结果汇总")
    add("")
    add("> 更新时间：2026-08-13。本文件由已经审计的 CSV/JSON 自动生成，集中展示当前仓库中所有已完成的 ZO 实验。")
    add("> `depth` 表示通信依赖深度，不是 wall-clock 时间；`work` 表示训练阶段的 SZO function calls，评估 work 单独计数。")
    add("> 带 † 的均值只在命中的 seeds 上计算，属于右删失条件均值，不能与完整命中结果等价解释。")
    add("")
    add("完整生成命令：")
    add("")
    add("```bash")
    add("conda run -n NOG python scripts/generate_zo_all_results.py")
    add("```")
    add("")

    add("## 1. 实验目标与理论参照")
    add("")
    add("比较 NOG-ZO、ME-DOL-ZO、DGFM 和 DGFM+。比例方向统一为 `baseline/NOG-ZO`；比例大于 1 表示 baseline 消耗更多 depth 或 work。理论表只给出论文中的主阶，实验用于佐证有限样本趋势，不用于从数据反推定理。")
    add("")
    extend(table(
        ["方法", "communication depth", "total SZO work"],
        [
            ["NOG-ZO", r"$\mathcal{O}(d^{1/3}\delta^{-1}\epsilon^{-5/3})$", r"$\mathcal{O}(d\delta^{-1}\epsilon^{-3})$"],
            ["ME-DOL-ZO", r"$\mathcal{O}(d\delta^{-1}\epsilon^{-3})$", r"$\mathcal{O}(d\delta^{-1}\epsilon^{-3})$"],
            ["DGFM", r"$\mathcal{O}(d^{3/2}\delta^{-1}\epsilon^{-4})$", r"$\mathcal{O}(d^{3/2}\delta^{-1}\epsilon^{-4})$"],
            ["DGFM+", r"$\mathcal{O}(d^{3/2}\delta^{-1}\epsilon^{-3})$", r"$\mathcal{O}(d^{3/2}\delta^{-1}\epsilon^{-3})$"],
        ],
    ))
    add("")
    extend(table(
        ["比例", "理论 depth ratio", "理论 work ratio"],
        [
            ["ME-DOL-ZO/NOG-ZO", r"$d^{2/3}\epsilon^{-4/3}$", r"$\epsilon^0$（常数阶）"],
            ["DGFM/NOG-ZO", r"$d^{7/6}\epsilon^{-7/3}$", r"$d^{1/2}\epsilon^{-1}$"],
            ["DGFM+/NOG-ZO", r"$d^{7/6}\epsilon^{-4/3}$", r"$d^{1/2}\epsilon^0$（对 $\epsilon$ 为常数阶）"],
        ],
    ))
    add("")

    add("## 2. 通用实现、计数与评估方法")
    add("")
    extend(table(
        ["项目", "设置"],
        [
            ["SZO estimator", "统一 two-point estimator；一个方向/数据样本对计 2 次 function calls"],
            ["DGFM+ difference", "普通 variance-reduced difference 同时评估当前位置和上一位置，相应 work 被完整计入"],
            ["training/evaluation", "training SZO work 与 evaluation work 分开记录"],
            ["主合成问题", r"维度 $d=100$，8 个逻辑 workers，complete mixing graph"],
            ["formal seeds", "0--19，共 20 个；四方法同 seed 配对"],
            ["formal work cap", "每个 method/seed 983,040 次训练 SZO calls"],
            ["评估 bank", "256 个平滑样本 × 512 个数据样本"],
            ["评估间隔", "每 4 个底层 depth 单位，并遵守方法自身有效 checkpoint 结构"],
            ["confirmed first hit", r"连续两个 checkpoint 的方法无关 stationarity proxy 均不超过 $\epsilon$"],
            ["主要终点", "同 seed 的 baseline/NOG-ZO first-hit depth ratio"],
            ["不确定性", "跨 20 个 paired formal seeds 的 bootstrap 95% CI"],
        ],
    ))
    add("")
    add("这里的 stationarity proxy 是平滑目标的 Monte Carlo 梯度范数估计，不是精确 Goldstein subdifferential distance。")
    add("")

    add("## 3. 参数选择与冻结")
    add("")
    add("先使用与 formal seeds 不相交的 pilot/calibration seeds 搜索有限候选网格，再冻结参数、配置和输入哈希。formal seeds 只用于最终评估，不参与重新调参。")
    add("")
    extend(table(
        ["方法", "主合成实验冻结参数", "最终 depth 上限", "最终 work 上限"],
        [
            ["NOG-ZO", r"$M=2$, $\eta=0.01$, `smooth_B=8`", "960", "983,040"],
            ["ME-DOL-ZO", "`epoch_length=12`, `theory_multiplier=60`", "3,840", "983,040"],
            ["DGFM", r"$\eta=0.05$", "7,680", "983,040"],
            ["DGFM+", r"$\eta=0.05$", "7,224", "983,040"],
        ],
    ))
    add("")
    add("![Pilot 结果快照](zo_experiments/figures/zo_pilot_snapshot.png)")
    add("")

    add(r"## 4. 主实验：$\epsilon$-scaling（Step ZO-5）")
    add("")
    eps_order = list(dict.fromkeys(float(row["epsilon"]) for row in formal_summary))
    add(r"完整 $\epsilon$ 网格：`" + ", ".join(epsilon(v) for v in eps_order) + r"`。同一条 anytime trajectory 提取全部阈值，因此增加 $\epsilon$ 点本身不需要重新训练。")
    add("")
    add("### 4.1 全部绝对 first-hit depth/work")
    add("")
    add("单元格格式：`hits/20; mean depth; mean work`。部分命中或未命中用 † 标记。")
    add("")
    formal_lookup = {(row["method"], float(row["epsilon"])): row for row in formal_summary}
    absolute_rows = []
    for eps in eps_order:
        absolute_rows.append([epsilon(eps)] + [hit_cell(formal_lookup[(method, eps)]) for method in METHODS])
    extend(table([r"$\epsilon$"] + METHODS, absolute_rows))
    add("")
    add("![Formal hit rate、absolute depth 和 work](zo_experiments/formal/figures/formal_hit_depth_work.png)")
    add("")

    add("### 4.2 全部 baseline/NOG-ZO depth/work ratios")
    add("")
    add("ratio 先在相同 seed 内计算，再对 paired seeds 求均值。只有 `complete` 行可用于无删失趋势拟合。")
    add("")
    for baseline in BASELINES:
        add(f"#### {baseline}/NOG-ZO")
        add("")
        rows = []
        for row in formal_ratios:
            if row["baseline"] != baseline:
                continue
            status = "complete" if row["complete_pairing"] == "True" else "censored†"
            rows.append([
                epsilon(row["epsilon"]),
                f"{row['paired_hits']}/{row['total_seeds']}",
                ci(row, "depth_ratio"),
                ci(row, "work_ratio"),
                status,
            ])
        extend(table([r"$\epsilon$", "paired hits", "depth ratio [95% CI]", "work ratio [95% CI]", "状态"], rows))
        add("")
    add("![Formal paired depth/work ratios](zo_experiments/formal/figures/formal_ratios.png)")
    add("")

    add("### 4.3 完整配对区间的趋势与理论对照")
    add("")
    trend_rows = []
    for row in theory["comparison"]:
        trend_rows.append([
            row["baseline"],
            f"{epsilon(row['epsilon_high'])}→{epsilon(row['epsilon_low'])}",
            str(row["complete_pairing_points"]),
            f"{ratio(row['depth_ratio_high_epsilon'])}→{ratio(row['depth_ratio_low_epsilon'])}",
            f"{ratio(row['observed_depth_ratio_slope'])} [{ratio(row['depth_slope_ci_low'])}, {ratio(row['depth_slope_ci_high'])}]",
            ratio(row["depth_spearman"]),
            f"{ratio(row['work_ratio_high_epsilon'])}→{ratio(row['work_ratio_low_epsilon'])}",
            f"{ratio(row['observed_work_ratio_slope'])} [{ratio(row['work_slope_ci_low'])}, {ratio(row['work_slope_ci_high'])}]",
        ])
    extend(table(
        ["baseline", r"完整 $\epsilon$ 区间", "点数", "depth ratio 首→尾", "depth slope [95% CI]", "Spearman", "work ratio 首→尾", "work slope [95% CI]"],
        trend_rows,
    ))
    add("")
    add(r"结论：三条 depth ratio 曲线在完整配对区间内均随 $\epsilon$ 减小单调增加，定性支持 NOG-ZO 更优的 $\epsilon$-depth 依赖；有限样本斜率没有恢复理论精确指数。ME-DOL-ZO 和 DGFM+ 的实测 work ratio 也增加，因此不能声称其已恢复理论常数阶 work ratio。")
    add("")

    add(r"## 5. 低 $\epsilon$、删失与反弹诊断（Step ZO-6）")
    add("")
    anomaly_rows = []
    for row in anomaly:
        anomaly_rows.append([
            row["method"],
            f"{float(row['mean_raw_floor']):.5f}",
            f"{float(row['mean_confirmed_floor']):.5f}",
            f"{100*float(row['median_confirmed_floor_work_fraction']):.1f}%",
            ratio(row["mean_final_over_confirmed_floor"]),
            row["dominant_extension_outlook"],
        ])
    extend(table(
        ["方法", "mean raw floor", "mean confirmed floor", "median floor work 位置", "final/floor", "预算扩展判断"],
        anomaly_rows,
    ))
    add("")
    add(r"NOG-ZO 的 confirmed floor 通常在约四分之一预算处出现，随后 proxy 反弹；因此继续增加同一冻结配置的预算不太可能解决 $\epsilon$ 更小处的 non-hit。reserved anomaly seeds 200--204 独立复现了这一模式。")
    add("")
    replication_rows = []
    for row in replication:
        replication_rows.append([
            row["method"],
            f"{float(row['formal_mean_confirmed_floor']):.5f}",
            f"{float(row['anomaly_mean_confirmed_floor']):.5f}",
            f"{100*float(row['formal_early_rebound_rate']):.0f}% / {100*float(row['anomaly_early_rebound_rate']):.0f}%",
            row["trajectory_pattern_replicated"],
            row["budget_extension_decision"],
        ])
    extend(table(
        ["方法", "formal floor", "reserved-seed floor", "early rebound formal/anomaly", "模式复现", "决策"],
        replication_rows,
    ))
    add("")
    add("![完整 proxy trajectories](zo_experiments/formal/figures/anomaly_proxy_trajectories.png)")
    add("")
    add("![单 checkpoint 与 confirmed-hit 边界](zo_experiments/formal/figures/boundary_checkpoint_sensitivity.png)")
    add("")
    add("![Formal 与 reserved anomaly seeds](zo_experiments/formal/figures/anomaly_replication_comparison.png)")
    add("")

    add("## 6. Dimension sensitivity（Step ZO-7）")
    add("")
    add(r"固定 $d=100$ 时选出的参数，不针对每个维度重新调参。测试 $d\in\{25,50,100,200\}$，四方法 × 20 seeds，共 320/320 tasks；每个任务 work cap 983,040，workers=8。")
    add("")
    add("### 6.1 全部 dimension absolute results")
    add("")
    add("单元格格式：`hits/20; conditional mean depth; conditional mean work`。")
    add("")
    dim_lookup = {(int(row["dimension"]), row["method"], float(row["epsilon"])): row for row in dimension_summary}
    dim_eps = list(dict.fromkeys(float(row["epsilon"]) for row in dimension_summary))
    dim_rows: list[list[str]] = []
    for dim in sorted({int(row["dimension"]) for row in dimension_summary}):
        for eps in dim_eps:
            sample = dim_lookup[(dim, "NOG-ZO", eps)]
            dim_rows.append([str(dim), epsilon(eps), sample["scope"]] + [hit_cell(dim_lookup[(dim, method, eps)]) for method in METHODS])
    extend(table([r"$d$", r"$\epsilon$", "scope"] + METHODS, dim_rows))
    add("")
    add("![Dimension absolute metrics](zo_experiments/dimension/figures/dimension_hit_depth_work.png)")
    add("")
    add("### 6.2 Primary dimension paired ratios")
    add("")
    primary_dim_rows = []
    for row in dimension_ratios:
        if row["scope"] != "primary":
            continue
        primary_dim_rows.append([
            row["dimension"], epsilon(row["epsilon"]), row["baseline"],
            f"{row['paired_hits']}/{row['total_seeds']}", ci(row, "depth_ratio"), ci(row, "work_ratio"),
        ])
    extend(table([r"$d$", r"$\epsilon$", "baseline", "paired hits", "depth ratio [95% CI]", "work ratio [95% CI]"], primary_dim_rows))
    add("")
    add("![Dimension paired ratios](zo_experiments/dimension/figures/dimension_ratios.png)")
    add("")
    add("### 6.3 Dimension slope")
    add("")
    dim_trend_rows = []
    for row in dimension_trends:
        dim_trend_rows.append([
            row["baseline"], epsilon(row["epsilon"]),
            f"{ratio(row['observed_depth_ratio_slope'])} [{ratio(row['depth_slope_ci_low'])}, {ratio(row['depth_slope_ci_high'])}]",
            ratio(row["theory_depth_ratio_power"]),
            f"{ratio(row['observed_work_ratio_slope'])} [{ratio(row['work_slope_ci_low'])}, {ratio(row['work_slope_ci_high'])}]",
            ratio(row["theory_work_ratio_power"]),
        ])
    extend(table(["baseline", r"$\epsilon$", "depth slope [95% CI]", "理论", "work slope [95% CI]", "理论"], dim_trend_rows))
    add("")
    add(r"结论：两个 primary $\epsilon$ 在所有维度和方法上均 20/20 hits，且 NOG-ZO 始终具有更低 first-hit depth；但 6 个 depth slope 中只有 2 个置信区间严格大于 0，没有一个包含对应理论参考幂次。因此只能报告固定配置 dimension sensitivity，不能声称验证精确维度指数。")
    add("")
    add("### 6.4 全部 dimension paired ratios（含 descriptive/censored）")
    add("")
    add("<details>")
    add("<summary>展开 84 行完整 dimension ratio 表</summary>")
    add("")
    all_dim_ratio_rows = []
    for row in dimension_ratios:
        status = "complete" if row["complete_pairing"] == "True" else "censored†"
        all_dim_ratio_rows.append([
            row["dimension"], epsilon(row["epsilon"]), row["scope"], row["baseline"],
            f"{row['paired_hits']}/{row['total_seeds']}", ci(row, "depth_ratio"), ci(row, "work_ratio"), status,
        ])
    extend(table(
        [r"$d$", r"$\epsilon$", "scope", "baseline", "paired hits", "depth ratio [95% CI]", "work ratio [95% CI]", "状态"],
        all_dim_ratio_rows,
    ))
    add("")
    add("</details>")
    add("")

    add("## 7. Logical-worker sensitivity（Step ZO-8）")
    add("")
    add(r"固定 $d=100,m=8$ 时冻结的算法参数，不进行 worker-wise retuning。测试 $m\in\{1,2,4,8\}$，四方法 × 20 seeds，共 320/320 tasks。该实验是单进程逻辑依赖模拟，不是 wall-clock speedup benchmark。")
    add("")
    add("### 7.1 全部 worker absolute results")
    add("")
    add("单元格格式：`hits/20; conditional mean depth; total work; per-worker work`。")
    add("")
    worker_lookup = {(int(row["worker_count"]), row["method"], float(row["epsilon"])): row for row in worker_summary}
    worker_eps = list(dict.fromkeys(float(row["epsilon"]) for row in worker_summary))
    worker_rows: list[list[str]] = []
    for workers in sorted({int(row["worker_count"]) for row in worker_summary}):
        for eps in worker_eps:
            sample = worker_lookup[(workers, "NOG-ZO", eps)]
            worker_rows.append([str(workers), epsilon(eps), sample["scope"]] + [hit_cell(worker_lookup[(workers, method, eps)], include_per_worker=True) for method in METHODS])
    extend(table([r"$m$", r"$\epsilon$", "scope"] + METHODS, worker_rows))
    add("")
    add("![Worker absolute metrics](zo_experiments/worker/figures/worker_hit_depth_work.png)")
    add("")
    add(r"### 7.2 $\epsilon=0.05$ 相对 $m=1$ 的同方法 ratios")
    add("")
    primary_worker_rows = []
    for row in worker_ratios:
        if float(row["epsilon"]) != 0.05:
            continue
        primary_worker_rows.append([
            row["method"], row["worker_count"], f"{row['paired_hits']}/{row['total_seeds']}",
            ci(row, "depth_ratio"), ci(row, "work_ratio"), ci(row, "per_worker_work_ratio"),
        ])
    extend(table(
        ["方法", r"$m$", "paired hits", "depth ratio [95% CI]", "total-work ratio [95% CI]", "per-worker ratio [95% CI]"],
        primary_worker_rows,
    ))
    add("")
    add("![Worker ratios relative to m=1](zo_experiments/worker/figures/worker_relative_to_m1.png)")
    add("")
    add("### 7.3 Worker log-log slopes")
    add("")
    worker_trend_rows = []
    for row in worker_trends:
        worker_trend_rows.append([
            row["method"], str(row["common_complete_seeds"]),
            f"{ratio(row['depth_slope'])} [{ratio(row['depth_slope_ci_low'])}, {ratio(row['depth_slope_ci_high'])}]",
            f"{ratio(row['work_slope'])} [{ratio(row['work_slope_ci_low'])}, {ratio(row['work_slope_ci_high'])}]",
            f"{ratio(row['per_worker_work_slope'])} [{ratio(row['per_worker_work_slope_ci_low'])}, {ratio(row['per_worker_work_slope_ci_high'])}]",
        ])
    extend(table(["方法", "共同完整 seeds", "depth slope", "total-work slope", "per-worker slope"], worker_trend_rows))
    add("")
    add(r"NOG-ZO 在 $m=1\to8$ 时 mean depth 为 214.2→214.4、total work 为 219,340.8→219,545.6，而 per-worker work 为 219,340.8→27,443.2，约为 $1/8$。这个结果支持固定总工作量下的逻辑 work decomposition，但不等于真实集群线性加速。")
    add("")
    add("![Worker terminal accounting](zo_experiments/worker/figures/worker_terminal_accounting.png)")
    add("")
    add("### 7.4 全部 worker ratios（含 descriptive/censored）")
    add("")
    add("<details>")
    add("<summary>展开 112 行完整 worker ratio 表</summary>")
    add("")
    all_worker_ratio_rows = []
    for row in worker_ratios:
        status = "complete" if row["complete_pairing"] == "True" else "censored†"
        all_worker_ratio_rows.append([
            row["method"], epsilon(row["epsilon"]), row["worker_count"],
            f"{row['paired_hits']}/{row['total_seeds']}", ci(row, "depth_ratio"),
            ci(row, "work_ratio"), ci(row, "per_worker_work_ratio"), status,
        ])
    extend(table(
        ["方法", r"$\epsilon$", r"$m$", "paired hits", "depth ratio [95% CI]", "total-work ratio [95% CI]", "per-worker ratio [95% CI]", "状态"],
        all_worker_ratio_rows,
    ))
    add("")
    add("</details>")
    add("")

    add("## 8. 真实 Gloo 多进程等价审计（Step ZO-9B）")
    add("")
    add("比较单进程 dependency simulator 与独立 Gloo CPU processes。当前覆盖 NOG-ZO 和 ME-DOL-ZO 的 1、2、8 workers；不是 DGFM/DGFM+ 的实现等价证明，也不是集群加速测试。")
    add("")
    equivalence_rows = []
    for row in equivalence:
        equivalence_rows.append([
            row["method"], row["workers"], row["passed"], row["checkpoints"],
            f"{float(row['max_abs_trajectory_difference']):.3e}",
            number(row["final_depth"], 0), number(row["final_total_work"], 0),
            number(row["final_per_worker_work"], 0), f"{float(row['end_to_end_time']):.2f}s",
        ])
    extend(table(
        ["方法", "workers", "通过", "checkpoints", "最大轨迹误差", "final depth", "total work", "per-worker work", "运行时间"],
        equivalence_rows,
    ))
    add("")
    add("结果为 6/6 通过，最大 checkpoint-wise absolute trajectory difference 为 5.960e-08；trajectory、任务身份、rank seed、独立 PID、work/depth accounting 和单调计数审计全部通过。")
    add("")

    add("## 9. a9a/ijcnn1 真实数据实验（Step ZO-9C）")
    add("")
    add("### 9.1 问题与统一协议")
    add("")
    add("使用官方 LIBSVM training sets，在 nonsmooth/nonconvex capped-l1 SVM 上比较四种方法。每行 feature 归一化到单位 l2 范数，下载文件通过 SHA256 验证。")
    add("")
    extend(table(
        ["项目", "设置"],
        [
            ["datasets", "a9a：32,561×123；ijcnn1：49,990×22"],
            ["workers/topology", "8 个逻辑 workers，complete mixing graph"],
            ["formal seeds", "0--19，每个 dataset/method 20 seeds"],
            ["work cap", "每个 dataset/method/seed 983,040 training SZO calls"],
            ["internal/eval radius", "0.001 / 0.002"],
            ["evaluation bank", "32 smoothing samples × 256 data samples"],
            ["校准 seeds", "303--311，与 formal seeds 不相交"],
            ["校准排序", "final objective 优先，然后 stationarity proxy"],
        ],
    ))
    add("")
    extend(table(
        ["dataset", "NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"],
        [
            ["a9a", r"$M=1$, $\eta=3\times10^{-5}$, `smooth_B=1`", "`epoch=6`, `multiplier=30,000`", r"$\eta=0.5$", r"$\eta=0.2$"],
            ["ijcnn1", r"$M=1$, $\eta=10^{-4}$, `smooth_B=1`", "`epoch=6`, `multiplier=30,000`", r"$\eta=2.0$", r"$\eta=0.1$"],
        ],
    ))
    add("")
    add("### 9.2 全部正式结果")
    add("")
    add("数值为 20 seeds 的 mean ± sample standard deviation。四种方法的 work 相同，因此这里不再计算无意义的 work ratio；depth ratio 可直接由固定终点 depth 得到。")
    add("")
    real_lookup = {(row["dataset"], row["method"]): row for row in real_summary}
    real_rows = []
    for dataset in ["a9a", "ijcnn1"]:
        nog = real_lookup[(dataset, "NOG-ZO")]
        for method in METHODS:
            row = real_lookup[(dataset, method)]
            real_rows.append([
                dataset, method,
                f"{float(row['objective_mean']):.4f} ± {float(row['objective_std']):.4f}",
                f"{float(row['stat_proxy_mean']):.4f} ± {float(row['stat_proxy_std']):.4f}",
                f"{float(row['accuracy_mean']):.4f} ± {float(row['accuracy_std']):.4f}",
                number(row["depth_mean"], 0),
                ratio(float(row["depth_mean"]) / float(nog["depth_mean"])),
                number(row["total_work_mean"], 0),
                ratio(float(row["total_work_mean"]) / float(nog["total_work_mean"])),
            ])
    extend(table(
        ["dataset", "方法", "objective", "stat proxy", "accuracy", "depth", "depth/NOG", "work", "work/NOG"],
        real_rows,
    ))
    add("")
    add("![a9a formal curves](zo_experiments/real_data/formal/a9a_formal_curves.png)")
    add("")
    add("![ijcnn1 formal curves](zo_experiments/real_data/formal/ijcnn1_formal_curves.png)")
    add("")
    add("结果边界：a9a 上 NOG-ZO 的 objective 和 proxy 优于 ME-DOL-ZO，但 DGFM 最好；ijcnn1 上 NOG-ZO 的 proxy 最低，而 ME-DOL-ZO 的 objective 更低、DGFM 的 accuracy 更高。NOG-ZO depth=7,680，ME-DOL-ZO depth=3,840，因此真实数据实验没有复现合成问题上的 NOG-ZO 通信优势，只能支持其优化质量具有竞争力。")
    add("")

    add("## 10. 完整性、审计状态和论文可用结论")
    add("")
    extend(table(
        ["实验", "任务数", "状态", "可安全使用的结论"],
        [
            [r"主 $\epsilon$-scaling", "80/80", "pass", r"完整配对区间内 baseline/NOG-ZO depth ratios 随 $\epsilon$ 减小而增加"],
            ["reserved-seed anomaly", "20/20", "complete", "NOG-ZO late-run rebound 在独立 seeds 上复现"],
            ["dimension sensitivity", "320/320", "pass", "NOG-ZO 保持较低 depth；不支持精确维度幂次"],
            ["worker sensitivity", "320/320", "pass", r"NOG-ZO total work/depth 近似不变，per-worker work 约按 $1/m$ 缩放"],
            ["Gloo process equivalence", "6/6", "pass", "NOG-ZO/ME-DOL-ZO simulator 与真实 CPU process 数值和计数等价"],
            ["a9a/ijcnn1", "160/160", "complete", "优化质量有竞争力，但没有复现通信 depth 优势"],
        ],
    ))
    add("")
    add(r"建议论文正文以主 $\epsilon$-scaling 的完整命中区间作为主要实验证据；worker 结果可作为补充。dimension、低 $\epsilon$ 删失和真实数据结果应放入附录或 limitation，并保留当前报告中的结论边界。")
    add("")

    add("## 11. 原始汇总、审计和复现入口")
    add("")
    file_rows = [
        ["ZO 总说明", link("ZO-README.md")],
        ["完整实验计划", link("ZO_plan.md")],
        ["主实验绝对结果", link("zo_experiments/formal/formal_summary.csv")],
        ["主实验 paired ratios", link("zo_experiments/formal/formal_ratios.csv")],
        ["主实验审计", link("zo_experiments/formal/audit.json")],
        ["理论比较", link("zo_experiments/formal/theory_comparison.json")],
        ["删失与异常报告", link("zo_experiments/formal/STEP_ZO_6A_ANOMALY_AUDIT.md")],
        ["reserved-seed 决策", link("zo_experiments/formal/STEP_ZO_6C_REPLICATION_DECISION.md")],
        ["dimension 完整报告", link("zo_experiments/dimension/README.md")],
        ["dimension summary", link("zo_experiments/dimension/dimension_summary.csv")],
        ["dimension ratios", link("zo_experiments/dimension/dimension_ratios.csv")],
        ["worker 完整报告", link("zo_experiments/worker/README.md")],
        ["worker summary", link("zo_experiments/worker/worker_summary.csv")],
        ["worker ratios", link("zo_experiments/worker/worker_relative_to_m1.csv")],
        ["Gloo 等价审计", link("zo_experiments/process_equivalence/equivalence_audit.json")],
        ["真实数据完整报告", link("zo_experiments/real_data/README.md")],
        ["真实数据 summary", link("zo_experiments/real_data/formal/summary.csv")],
        ["真实数据冻结参数", link("zo_experiments/real_data/frozen_parameters.json")],
    ]
    extend(table(["内容", "文件"], file_rows))
    add("")
    add("复现入口和分步命令详见 `ZO-README.md` 的运行方法部分。所有正式分析器只读取冻结后的 raw trajectories，不使用 formal 结果重新选择参数。")
    add("")
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT.write_text(generate(), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
