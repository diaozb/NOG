"""Build the Step 7E advisor/paper package from audited Step 7 outputs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.distributed.cpu_fo_formal_figures import load_verified_inputs
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


REPORT_SCHEMA_VERSION = 1
EXPECTED_FIGURES = (
    "depth_vs_epsilon.png",
    "depth_vs_epsilon.pdf",
    "work_vs_epsilon.png",
    "work_vs_epsilon.pdf",
    "stat_proxy_vs_depth.png",
    "stat_proxy_vs_depth.pdf",
    "stat_proxy_vs_work.png",
    "stat_proxy_vs_work.pdf",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_figure_package(
    figure_root: str | Path,
    analysis_completion: Dict[str, Any],
) -> Dict[str, Any]:
    root = Path(figure_root)
    manifest = _load_json(root / "figure_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Step 7D figure manifest is not complete.")
    for key in ("formal_manifest_sha256", "frozen_config_sha256"):
        if manifest.get(key) != analysis_completion.get(key):
            raise ValueError(f"Step 7C/7D {key} mismatch.")
    if set(manifest.get("figures", {})) != set(EXPECTED_FIGURES):
        raise ValueError("Step 7D figure coverage mismatch.")
    for name in EXPECTED_FIGURES:
        if file_sha256(root / name) != manifest["figures"][name]:
            raise ValueError(f"Step 7D figure SHA256 mismatch: {name}.")
    return manifest


def _number(value: Any, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}"


def _integerish(value: Any) -> str:
    return f"{float(value):.0f}"


def _summary_lookup(summaries: pd.DataFrame) -> Dict[tuple[str, float], pd.Series]:
    return {
        (str(row["method"]), float(row["epsilon"])): row
        for _, row in summaries.iterrows()
    }


def render_markdown_report(
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    lookup = _summary_lookup(summaries)
    lines = [
        "# Step 7：NOG-FO vs ME-DOL-FO Formal Accuracy 最终结果",
        "",
        "## 一句话结论",
        "",
        "在双方均 `5/5` formal seeds confirmed hit 的 `epsilon = 0.011, 0.010, 0.009` 上，NOG-FO 达到相同 stationarity threshold 所需的 mean communication depth 比 ME-DOL-FO 小 `7.15x, 6.14x, 10.71x`，与论文 Section 5 所预测的 communication advantage 定性一致；但 NOG-FO 的 finite total SFO work 高 `4.47x, 10.43x, 2.99x`。因此当前证据支持 communication-depth advantage，并显示 first-hit training time 更短；它不支持 finite-work advantage，也不能替代 Step 8 的受控 runtime-scaling 结论。",
        "",
        "## 实验协议",
        "",
        "- Target：`(2 delta, epsilon)`-Goldstein stationarity，其中 `delta=0.1`，即固定 neighborhood 为 `2 delta=0.2`；",
        "- Methods：`NOG-FO` 与 `ME-DOL-FO`；每个 logical worker 是一个真实 CPU process，`m=8`，Gloo exact all-reduce；",
        "- Problem：`SyntheticMaxSinL1`，`d=100, n_data=4096, R=4, lambda=0.001`；",
        "- Formal seeds：`[0,1,2,3,4]`；pilot seeds `[100,101,102]` 严格隔离；",
        "- 每个 `method x epsilon` 的 hyperparameters 只用 pilot seeds 选择并在 formal run 前冻结；",
        "- 使用共同的 fixed high-precision evaluation sample bank；连续两个 checkpoints 满足 `stat_proxy <= epsilon` 才记为 confirmed hit；",
        "- Work 指 training SFO calls，不包含单列审计的 evaluation work。",
        "",
        "## 与 Section 5 理论结果的关系",
        "",
        "本结果包采用论文当前的 `(2 delta, epsilon)` 定义及以下 SFO 复杂度口径：",
        "",
        "| Method | Communication | Work |",
        "|---|---:|---:|",
        "| ME-DOL (SFO) | `O(delta^{-1} epsilon^{-3})` | `O(delta^{-1} epsilon^{-3})` |",
        "| NOG (SFO) | `O(d^{1/3} delta^{-1} epsilon^{-5/3})` | `O(delta^{-1} epsilon^{-3})` |",
        "",
        "理论预测 NOG 改善 communication exponent，同时与 ME-DOL 保持相同 asymptotic work order。本实验固定了 `d` 和 `delta`，且只有三个双方 full-hit thresholds，因此只能检验 qualitative depth advantage，不能可靠拟合或宣称验证 `epsilon^{-5/3}` 与 `epsilon^{-3}` 的 asymptotic slopes。实验中的 constant、batch 和 tuned hyperparameters 会影响 finite work；NOG work 高 3–10 倍不否定同阶理论，但也不能被表述为验证了 work 同阶。",
        "",
        "## Formal results",
        "",
        "下表 mean ± sample std 只对成功 seeds 计算；带 `†` 的行存在 right censoring，不能使用 hit-only ratio 做无条件比较。",
        "",
        "| epsilon | NOG hit | ME hit | NOG depth | ME depth | Depth gain | NOG total work | ME total work | NOG/ME work |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons.sort_values("epsilon", ascending=False).itertuples():
        epsilon = float(row.epsilon)
        nog = lookup[("NOG-FO", epsilon)]
        me = lookup[("ME-DOL-FO", epsilon)]
        full = bool(row.full_hit_comparison)
        nog_dagger = "" if bool(nog["full_hit"]) else "†"
        me_dagger = "" if bool(me["full_hit"]) else "†"
        gain = f"{float(row.depth_improvement_me_over_nog):.2f}x" if full else "—"
        work_ratio = f"{float(row.total_work_ratio_nog_over_me):.2f}x" if full else "—"
        lines.append(
            "| {eps:g} | {nh}/5{ndg} | {mh}/5{mdg} | {nd} ± {nds}{ndg} | "
            "{md} ± {mds}{mdg} | {gain} | {nw} ± {nws}{ndg} | "
            "{mw} ± {mws}{mdg} | {wr} |".format(
                eps=epsilon,
                nh=int(nog["hit_count"]),
                mh=int(me["hit_count"]),
                ndg=nog_dagger,
                mdg=me_dagger,
                nd=_number(nog["first_hit_depth_mean"]),
                nds=_number(nog["first_hit_depth_std"]),
                md=_number(me["first_hit_depth_mean"]),
                mds=_number(me["first_hit_depth_std"]),
                gain=gain,
                nw=_integerish(nog["first_hit_total_work_mean"]),
                nws=_integerish(nog["first_hit_total_work_std"]),
                mw=_integerish(me["first_hit_total_work_mean"]),
                mws=_integerish(me["first_hit_total_work_std"]),
                wr=work_ratio,
            )
        )
    lines.extend(
        [
            "",
            "`†` `epsilon=0.008`：NOG `5/5`、ME-DOL `1/5`；`epsilon=0.0075`：NOG `4/5`、ME-DOL `3/5`。这些条件均值存在 censoring bias。",
            "",
            "## 如何解释结果",
            "",
            "1. **Communication depth**：三个 full-hit thresholds 上均有稳定的大幅优势，且在更严格 epsilon 上优势没有消失；这是当前最强、最适合写入论文的证据。",
            "2. **Finite SFO work**：三个可公平比较的 thresholds 上，NOG 分别多使用约 `4.47x, 10.43x, 2.99x` total/per-worker SFO work。由于两种方法都在 `m=8`，total-work ratio 与 per-worker-work ratio 相同。",
            "3. **Training time**：对应三个 full-hit thresholds，formal checkpoint training time 的 ME-DOL/NOG mean ratios 为 `4.81x, 5.51x, 7.96x`。这说明真实 CPU-process 实现中低 depth 已转化为较短训练时间，但它不是受控 systems benchmark；worker scaling、warm-up 和重复计时属于 Step 8。",
            "4. **严格 thresholds**：`epsilon=0.008` 明显有利于 NOG 的 hit rate，但 ME-DOL 只有一个成功 seed；`epsilon=0.0075` 两者都 censored。因此可报告 hit rates，不能用这些行的成功-seed ratios 证明 speedup。",
            "5. **Same stationary point**：这里指相同 problem、`delta` 和 empirical stationarity threshold，不表示两个算法到达完全相同的 parameter vector。",
            "",
            "## Frozen configurations",
            "",
            "| epsilon | NOG-FO | ME-DOL-FO |",
            "|---:|---|---|",
        ]
    )
    for epsilon in sorted(summaries["epsilon"].unique(), reverse=True):
        nog = lookup[("NOG-FO", float(epsilon))]
        me = lookup[("ME-DOL-FO", float(epsilon))]
        nog_parameters = json.loads(str(nog["parameters"]))
        me_parameters = json.loads(str(me["parameters"]))
        lines.append(
            "| {eps:g} | `M={M}, eta={eta:g}, smooth_B={smooth}, data_B_total={data}`; "
            "rounds `{nr}` | `epoch_length={epoch}, theory_multiplier={mult:g}`; rounds `{mr}` |".format(
                eps=float(epsilon),
                M=int(nog_parameters["M"]),
                eta=float(nog_parameters["eta"]),
                smooth=int(nog_parameters["smooth_B"]),
                data=int(nog_parameters["data_B_total"]),
                nr=int(nog["formal_config_id"].split("__rounds-")[1].split("__")[0]),
                epoch=int(me_parameters["epoch_length"]),
                mult=float(me_parameters["theory_multiplier"]),
                mr=int(me["formal_config_id"].split("__rounds-")[1].split("__")[0]),
            )
        )
    lines.extend(
        [
            "",
            "配置随 epsilon 变化是预注册设计：每个 threshold 在独立 pilot seeds 上选择配置；formal seeds 没有参与调参。图中的 trajectory panels 也分别使用对应 frozen config，不能解释为一组 universal hyperparameters 的 epsilon scaling。",
            "",
            "## Figures",
            "",
            "- [`depth_vs_epsilon.pdf`](../figures/depth_vs_epsilon.pdf)：论文主图候选，最直接展示 communication advantage；",
            "- [`work_vs_epsilon.pdf`](../figures/work_vs_epsilon.pdf)：与主图配套，揭示 finite-work tradeoff；",
            "- [`stat_proxy_vs_depth.pdf`](../figures/stat_proxy_vs_depth.pdf)：五个 epsilon-specific panels 的完整收敛轨迹；",
            "- [`stat_proxy_vs_work.pdf`](../figures/stat_proxy_vs_work.pdf)：相同轨迹按 training SFO work 展示。",
            "",
            "Threshold plots 中 filled marker 表示 `5/5` hit，hollow marker 表示 censored conditional mean；所有 error bars/bands 均为 mean ± one sample std。",
            "",
            "## 建议写入论文的英文表述",
            "",
            "> On the three stationarity tolerances for which both methods achieved a 5/5 confirmed-hit rate, NOG-FO required 6.14--10.71 times fewer communication rounds than ME-DOL-FO. This qualitative advantage is consistent with the improved communication dependence predicted by our theory. NOG-FO used 2.99--10.43 times more finite-sample SFO work, highlighting a practical constant-factor tradeoff that is not captured by the shared asymptotic work order. At the two strictest tolerances, at least one method was right-censored; we therefore report hit rates and omit unconditional speedup ratios for those settings.",
            "",
            "这里建议使用 `consistent with`，不要使用 `verifies the epsilon^{-5/3} rate`、`same work in practice` 或在 censored thresholds 上写无条件 `speedup`。",
            "",
            "## 当前限制与后续动作",
            "",
            "- 只有三个双方 full-hit epsilon，epsilon 范围也较窄，不足以拟合 asymptotic exponent；",
            "- `d` 与 `delta` 固定，当前实验不验证 `d^{1/3}` 或 `delta^{-1}` dependence；",
            "- epsilon-specific tuning 提高了每个 threshold 的实际表现，但不适合用单一 log-log slope 解释理论；",
            "- Synthetic problem 不能替代 real-data generalization evidence；",
            "- Step 8 仍需做预注册的 CPU runtime scaling，才能正式讨论 parallel speedup/efficiency；",
            "- 是否扩展更多 epsilon、delta、dimension 或 ZO/DGFM/DGFM+，建议先把本结果包发给学长确认，不在当前 formal results 上事后追加有利配置。",
            "",
            "## 复现",
            "",
            "```bash",
            "/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_formal_analysis \\",
            "  --config configs/distributed_cpu_fo_pilot.yaml",
            "/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_formal_figures",
            "/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_formal_report",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_table(
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    lookup = _summary_lookup(summaries)
    rows = []
    for comparison in comparisons.sort_values("epsilon", ascending=False).itertuples():
        epsilon = float(comparison.epsilon)
        nog = lookup[("NOG-FO", epsilon)]
        me = lookup[("ME-DOL-FO", epsilon)]
        full = bool(comparison.full_hit_comparison)
        nog_marker = "" if bool(nog["full_hit"]) else r"$^{\dagger}$"
        me_marker = "" if bool(me["full_hit"]) else r"$^{\dagger}$"
        gain = f"{float(comparison.depth_improvement_me_over_nog):.2f}$\\times$" if full else "--"
        ratio = f"{float(comparison.total_work_ratio_nog_over_me):.2f}$\\times$" if full else "--"
        rows.append(
            f"{epsilon:g} & {int(nog['hit_count'])}/5{nog_marker} & {int(me['hit_count'])}/5{me_marker} & "
            f"{float(nog['first_hit_depth_mean']):.1f} $\\pm$ {float(nog['first_hit_depth_std']):.1f}{nog_marker} & "
            f"{float(me['first_hit_depth_mean']):.1f} $\\pm$ {float(me['first_hit_depth_std']):.1f}{me_marker} & "
            f"{gain} & {float(nog['first_hit_total_work_mean']):.0f}{nog_marker} & "
            f"{float(me['first_hit_total_work_mean']):.0f}{me_marker} & {ratio} \\\\"
        )
    body = "\n".join(rows)
    return rf"""% Requires \usepackage{{booktabs,graphicx}}.
\begin{{table}}[t]
\centering
\caption{{Formal CPU-process comparison at $m=8$ and $\delta=0.1$. A hit requires two consecutive evaluation checkpoints below $\epsilon$. Depth/work statistics are means over successful formal seeds; $\dagger$ denotes right-censored settings, for which unconditional ratios are omitted. Training SFO work excludes evaluation work.}}
\label{{tab:formal-cpu-fo}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{c cc cc c cc c}}
\toprule
$\epsilon$ & \multicolumn{{2}}{{c}}{{Hit rate}} & \multicolumn{{2}}{{c}}{{Depth (mean $\pm$ std)}} & Depth gain & \multicolumn{{2}}{{c}}{{Total SFO work (mean)}} & Work ratio \\
& NOG & ME-DOL & NOG & ME-DOL & ME/NOG & NOG & ME-DOL & NOG/ME \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def render_captions() -> str:
    return r"""# Paper figure captions

## `depth_vs_epsilon.pdf`

**Communication depth to a confirmed stationarity threshold.** Mean first-hit communication depth and one sample standard deviation over five formal seeds for NOG-FO and ME-DOL-FO at $m=8$ and $\delta=0.1$. A confirmed hit requires two consecutive evaluation checkpoints with stationarity proxy at most $\epsilon$. Filled markers indicate a 5/5 hit rate; hollow markers are conditional means over successful seeds in right-censored settings, with hit counts shown next to the marker. Configurations were selected independently for each method and tolerance using disjoint pilot seeds.

## `work_vs_epsilon.pdf`

**Training SFO work to a confirmed stationarity threshold.** Mean first-hit total training SFO calls and one sample standard deviation under the same protocol as the communication-depth figure. Evaluation calls from the shared high-precision sample bank are audited separately and excluded. Hollow markers denote right-censored conditional means and should not be interpreted as unconditional method comparisons.

## `stat_proxy_vs_depth.pdf`

**Stationarity trajectories versus communication depth.** Lines and shaded bands show the mean and one sample standard deviation across five formal seeds. Each panel compares the two pilot-frozen configurations selected for the displayed tolerance; the dashed line is the target $\epsilon$. Panel titles report confirmed-hit counts. Because configurations vary across tolerances, the panels are separate empirical operating points rather than a single-hyperparameter scaling curve.

## `stat_proxy_vs_work.pdf`

**Stationarity trajectories versus total training SFO work.** The trajectories and aggregation protocol match the communication-depth panels, with the horizontal axis replaced by cumulative training SFO calls. Evaluation work is excluded. This view exposes the finite-work tradeoff accompanying NOG-FO's lower communication depth.
"""


def render_advisor_message() -> str:
    return """学长好，FO 的真实 CPU-process formal experiment 已经完成并审计过了。我们固定 delta=0.1、m=8，用 5 个 formal seeds，对每个 epsilon 使用独立 pilot seeds 选出的 frozen config；命中要求连续两个 evaluation checkpoints 都低于 epsilon。

在 epsilon=0.011/0.010/0.009 上，两种方法都是 5/5 命中。NOG-FO 相比 ME-DOL-FO 的平均 communication depth 分别减少约 7.15x、6.14x、10.71x；对应实际 training time 快约 4.81x、5.51x、7.96x。这个结果定性支持论文里 NOG communication complexity 更优的结论。

不过 NOG 的 finite total SFO work 在这三组上分别高约 4.47x、10.43x、2.99x，所以目前不能说实际 work 区别不大，只能说理论上 asymptotic order 相同、实验中 constant-factor gap 较明显。epsilon=0.008 时 NOG 5/5、ME-DOL 1/5；epsilon=0.0075 时 NOG 4/5、ME-DOL 3/5，这两组有 censoring，我没有用成功 seeds 的 ratio 做无条件 speedup 结论。

表格、四组 PNG/PDF 图、可复现配置和限制说明都已整理好。建议论文里写“consistent with the predicted communication advantage”，暂时不要写“verified epsilon^{-5/3} scaling”。如果您认可这个方向，下一步可以继续做 CPU worker scaling/runtime benchmark，或者再决定是否补更多 epsilon/delta/dimension。"""


def build_report_package(
    repo_root: str | Path,
    formal_root: str | Path,
    figure_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    repo = Path(repo_root)
    formal = Path(formal_root)
    figures = Path(figure_root)
    output = Path(output_root)
    _, summaries, _, completion = load_verified_inputs(formal)
    figure_manifest = validate_figure_package(figures, completion)
    comparisons = pd.read_csv(formal / "method_comparison.csv")
    comparisons["full_hit_comparison"] = (
        comparisons["full_hit_comparison"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    files = {
        "FINAL_RESULTS.md": render_markdown_report(summaries, comparisons),
        "paper_table.tex": render_latex_table(summaries, comparisons),
        "figure_captions.md": render_captions(),
        "advisor_message.md": render_advisor_message(),
    }
    for name, content in files.items():
        _atomic_write_text(output / name, content)

    paper_path = repo / "NeurIPS_NOG.pdf"
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_manifest_sha256": completion["formal_manifest_sha256"],
        "frozen_config_sha256": completion["frozen_config_sha256"],
        "paper_pdf_sha256": file_sha256(paper_path),
        "figure_manifest_sha256": file_sha256(figures / "figure_manifest.json"),
        "verified_figure_count": int(figure_manifest["figure_count"]),
        "formal_full_hit_epsilons": [0.011, 0.010, 0.009],
        "right_censored_epsilons": [0.008, 0.0075],
        "deliverables": {
            name: file_sha256(output / name) for name in files
        },
        "claim_boundary": {
            "supported": "qualitative communication-depth advantage on three mutual-full-hit thresholds",
            "not_supported": [
                "empirical verification of epsilon asymptotic exponents",
                "finite-work parity or advantage for NOG-FO",
                "unconditional ratios at right-censored thresholds",
                "parallel scaling claims before Step 8 runtime study",
            ],
        },
    }
    atomic_write_json(output / "step7_completion.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--formal-root", default="outputs/distributed_cpu_fo/formal_accuracy"
    )
    parser.add_argument(
        "--figure-root", default="outputs/distributed_cpu_fo/figures"
    )
    parser.add_argument(
        "--output-root", default="outputs/distributed_cpu_fo/step7_final"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_report_package(
        args.repo_root, args.formal_root, args.figure_root, args.output_root
    )
    print(
        f"phase=step7-final status={manifest['status']} "
        f"deliverables={len(manifest['deliverables'])} "
        f"figures={manifest['verified_figure_count']}"
    )


if __name__ == "__main__":
    main()
