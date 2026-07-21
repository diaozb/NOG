"""Build the Step 8E joint accuracy/runtime result package."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.distributed.cpu_fo_formal_figures import (
    load_verified_inputs as load_verified_formal_inputs,
)
from src.distributed.cpu_fo_runtime_figures import (
    load_verified_inputs as load_verified_runtime_inputs,
)
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


REPORT_SCHEMA_VERSION = 1
EXPECTED_RUNTIME_FIGURES = (
    "runtime_vs_workers.png",
    "runtime_vs_workers.pdf",
    "nog_strong_scaling_speedup.png",
    "nog_strong_scaling_speedup.pdf",
    "communication_fraction_vs_workers.png",
    "communication_fraction_vs_workers.pdf",
    "full_budget_method_comparison.png",
    "full_budget_method_comparison.pdf",
)
EXPECTED_STEP7_DELIVERABLES = (
    "FINAL_RESULTS.md",
    "paper_table.tex",
    "figure_captions.md",
    "advisor_message.md",
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


def validate_runtime_figure_package(
    figure_root: str | Path,
    runtime_completion: Dict[str, Any],
) -> Dict[str, Any]:
    root = Path(figure_root)
    manifest_path = root / "runtime_figure_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Step 8D figure manifest is not complete.")
    if manifest.get("runtime_manifest_sha256") != runtime_completion.get(
        "runtime_manifest_sha256"
    ):
        raise ValueError("Step 8C/8D runtime manifest hash mismatch.")
    if set(manifest.get("figures", {})) != set(EXPECTED_RUNTIME_FIGURES):
        raise ValueError("Step 8D figure coverage mismatch.")
    for name in EXPECTED_RUNTIME_FIGURES:
        if file_sha256(root / name) != manifest["figures"][name]:
            raise ValueError(f"Step 8D figure SHA256 mismatch: {name}.")
    for name, expected in manifest.get("source_sha256", {}).items():
        if runtime_completion.get("output_sha256", {}).get(name) != expected:
            raise ValueError(f"Step 8C/8D source SHA256 mismatch: {name}.")
    notes = root / "runtime_figure_notes.md"
    if file_sha256(notes) != manifest.get("notes_sha256"):
        raise ValueError("Step 8D figure-notes SHA256 mismatch.")
    return manifest


def validate_step7_package(
    step7_root: str | Path,
    formal_completion: Dict[str, Any],
) -> Dict[str, Any]:
    root = Path(step7_root)
    manifest = _load_json(root / "step7_completion.json")
    if manifest.get("status") != "complete":
        raise ValueError("Step 7E package is not complete.")
    for key in ("formal_manifest_sha256", "frozen_config_sha256"):
        if manifest.get(key) != formal_completion.get(key):
            raise ValueError(f"Step 7C/7E {key} mismatch.")
    if set(manifest.get("deliverables", {})) != set(EXPECTED_STEP7_DELIVERABLES):
        raise ValueError("Step 7E deliverable coverage mismatch.")
    for name in EXPECTED_STEP7_DELIVERABLES:
        if file_sha256(root / name) != manifest["deliverables"][name]:
            raise ValueError(f"Step 7E deliverable SHA256 mismatch: {name}.")
    return manifest


def _formal_lookup(summaries: pd.DataFrame) -> Dict[tuple[str, float], pd.Series]:
    return {
        (str(row["method"]), float(row["epsilon"])): row
        for _, row in summaries.iterrows()
    }


def _runtime_ranges(
    summaries: pd.DataFrame,
    speedups: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> Dict[str, float]:
    nog_m32 = speedups[
        (speedups["method"] == "NOG-FO") & (speedups["worker_count"] == 32)
    ]
    m1 = summaries[summaries["worker_count"] == 1]
    m32 = summaries[summaries["worker_count"] == 32]
    return {
        "training_ratio_min": float(comparisons["training_time_ratio_nog_over_me"].min()),
        "training_ratio_max": float(comparisons["training_time_ratio_nog_over_me"].max()),
        "end_to_end_ratio_min": float(comparisons["end_to_end_time_ratio_nog_over_me"].min()),
        "end_to_end_ratio_max": float(comparisons["end_to_end_time_ratio_nog_over_me"].max()),
        "work_ratio_min": float(comparisons["work_ratio_nog_over_me"].min()),
        "work_ratio_max": float(comparisons["work_ratio_nog_over_me"].max()),
        "nog_m32_speedup_min": float(nog_m32["training_speedup_vs_m1"].min()),
        "nog_m32_speedup_max": float(nog_m32["training_speedup_vs_m1"].max()),
        "comm_m1_min": 100.0 * float(m1["communication_fraction_median"].min()),
        "comm_m1_max": 100.0 * float(m1["communication_fraction_median"].max()),
        "comm_m32_min": 100.0 * float(m32["communication_fraction_median"].min()),
        "comm_m32_max": 100.0 * float(m32["communication_fraction_median"].max()),
    }


def render_markdown_report(
    formal_summaries: pd.DataFrame,
    runtime_summaries: pd.DataFrame,
    speedups: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> str:
    ranges = _runtime_ranges(runtime_summaries, speedups, comparisons)
    formal = _formal_lookup(formal_summaries)
    lines = [
        "# Step 8：NOG-FO vs ME-DOL-FO CPU Runtime 最终结果",
        "",
        "## 一句话结论",
        "",
        "Step 7 的 formal accuracy experiment 显示，在双方均 `5/5` confirmed hit 的 `epsilon=0.011,0.010,0.009` 上，NOG-FO 达到相同 empirical stationarity threshold 所需 mean communication depth 少 `6.14--10.71x`；Step 8 的真实 CPU-process repeated benchmark 则显示，NOG-FO 在当前单机 CPU/Gloo 环境下没有获得 positive strong scaling，增加 workers 反而变慢。NOG 的 frozen full-budget median training time 是 ME-DOL 的 `0.539--0.821x`，但两者没有 empirical-work matching，不能把该 wall-clock ratio 解释为 finite-work parity、time-to-epsilon speedup 或 Section 5 asymptotic Work complexity 的验证。",
        "",
        "## Step 8 protocol",
        "",
        "- Methods：`NOG-FO`、`ME-DOL-FO`；真实 CPU processes + Gloo exact all-reduce；",
        "- Tolerances：`epsilon=[0.010,0.009,0.008]`；worker counts：`m=[1,2,4,8,16,32]`；benchmark seed：`0`；",
        "- 每个 unique config × worker setting 先运行 one-update-unit warm-up，再运行 3 个 measured repeats；",
        "- 24 个 warm-ups 不进入统计；72 个 physical measured runs 展开为 108 个 method-epsilon-worker-repeat rows；",
        "- NOG 每个 epsilon 使用 Step 6 pilot-frozen config、960 rounds；ME-DOL 三个 epsilon 映射到同一个 frozen config、1920 rounds；",
        "- `training_time` 排除 process startup、evaluation 和 serialization；`end_to_end_time` 包含这些开销；",
        "- 每个点报告 3 repeats 的 median 和完整 `[min,max]`，不删除 outliers；",
        "- 该 benchmark 运行完整 frozen budget，不是 first-hit time-to-epsilon。",
        "",
        "## Runtime results at m=8",
        "",
        "选择 `m=8` 单列，是为了与 Step 7 formal accuracy 的 worker count 对齐；仍然是 full-budget comparison。",
        "",
        "| epsilon | Accuracy status | NOG training (s) | ME training (s) | NOG/ME time | NOG/ME end-to-end | NOG/ME work |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons[comparisons["worker_count"] == 8].sort_values(
        "epsilon", ascending=False
    ).itertuples():
        epsilon = float(row.epsilon)
        nog = formal[("NOG-FO", epsilon)]
        me = formal[("ME-DOL-FO", epsilon)]
        if bool(nog["full_hit"]) and bool(me["full_hit"]):
            status = "both 5/5 confirmed hit"
        else:
            status = f"NOG {int(nog['hit_count'])}/5; ME {int(me['hit_count'])}/5†"
        lines.append(
            "| {eps:g} | {status} | {nt:.2f} | {mt:.2f} | {tr:.3f} | {er:.3f} | {wr:.3f} |".format(
                eps=epsilon,
                status=status,
                nt=float(row.nog_training_time_median),
                mt=float(row.me_dol_training_time_median),
                tr=float(row.training_time_ratio_nog_over_me),
                er=float(row.end_to_end_time_ratio_nog_over_me),
                wr=float(row.work_ratio_nog_over_me),
            )
        )
    lines.extend(
        [
            "",
            "`†` `epsilon=0.008` 的 ME-DOL formal accuracy 为 `1/5`，所以该行只能说明完整预算的执行时间，不能说明达到 epsilon 的 runtime。",
            "",
            "## Scaling result",
            "",
            "- 两个 methods、三个 epsilon 的最低 median training time 全部出现在 `m=1`；",
            f"- NOG 是 fixed-total-work workload，但 `m=32` 相对 `m=1` 的 training speedup 仅为 `{ranges['nog_m32_speedup_min']:.3f}--{ranges['nog_m32_speedup_max']:.3f}x`，即不是加速而是约 `7.2--8.5x` slowdown；",
            "- ME-DOL 每个 worker 每轮各执行一个 SFO call，total work 随 m 线性增长，因此 ME-DOL 的 `T1/Tm` 不能解释为 strong-scaling efficiency；",
            f"- median communication/training-time fraction 从 `m=1` 的约 `{ranges['comm_m1_min']:.1f}--{ranges['comm_m1_max']:.1f}%` 增至 `m=32` 的约 `{ranges['comm_m32_min']:.1f}--{ranges['comm_m32_max']:.1f}%`；",
            "- 结果只说明当前 single-host CPU process/Gloo implementation；不能外推为多机、真实网络或 GPU collective 的 scaling behavior。",
            "",
            "## Full-budget method comparison",
            "",
            f"跨全部 epsilon 和 worker settings，NOG/ME-DOL median training-time ratio 为 `{ranges['training_ratio_min']:.3f}--{ranges['training_ratio_max']:.3f}`，end-to-end ratio 为 `{ranges['end_to_end_ratio_min']:.3f}--{ranges['end_to_end_ratio_max']:.3f}`。因此 NOG 确实更快完成了各自的 frozen full budget，但对应 NOG/ME-DOL total SFO-work ratio 为 `{ranges['work_ratio_min']:.3f}--{ranges['work_ratio_max']:.3f}`。",
            "",
            "这个组合结果并不矛盾：wall-clock cost 取决于 update structure、communication depth、batch vectorization、process synchronization 和 Python/PyTorch implementation，而 SFO work 只计 oracle samples。当前结果支持的是 implementation-level full-budget runtime difference；它不支持相同 finite work 下 NOG 更快。",
            "",
            "## 与 Step 7 和 Section 5 的联合解释",
            "",
            "| Question | Evidence | Conclusion |",
            "|---|---|---|",
            "| 达到相同 empirical threshold 的 depth 是否更小？ | Step 7，三个 mutual-full-hit thresholds | 是，NOG mean depth 少 `6.14--10.71x`；与 predicted communication advantage 定性一致 |",
            "| finite SFO work 是否相近或更小？ | Step 7 first-hit work；Step 8 full-budget work | 否，当前 tuned configs 下 NOG 使用更多 finite SFO work |",
            "| 单机增加 CPU workers 是否带来 acceleration？ | Step 8 repeated scaling benchmark | 否，两种方法均在 `m=1` 最快 |",
            "| NOG 是否更快完成 frozen full budget？ | Step 8 repeated timing | 是，但 workload 未 work matched，且不是 time-to-epsilon |",
            "| 是否验证了 `epsilon^{-5/3}`、`epsilon^{-3}`、`d^{1/3}` 或 `delta^{-1}`？ | epsilon/d/delta 覆盖不足 | 否，只能使用 `consistent with` 的 qualitative wording |",
            "",
            "## Figures 与使用建议",
            "",
            "### 推荐作为论文主结果",
            "",
            "- [`depth_vs_epsilon.pdf`](../figures/depth_vs_epsilon.pdf)：Step 7 communication-depth 主图；",
            "- [`work_vs_epsilon.pdf`](../figures/work_vs_epsilon.pdf)：必须与 depth evidence 配套，呈现 finite-work tradeoff。",
            "",
            "### 可放 supplement 或发给学长",
            "",
            "- [`runtime_vs_workers.pdf`](../runtime/figures/runtime_vs_workers.pdf)：完整 timing、range 和 long-tail behavior；",
            "- [`full_budget_method_comparison.pdf`](../runtime/figures/full_budget_method_comparison.pdf)：runtime ratio 与 work mismatch 必须成对展示；",
            "- [`communication_fraction_vs_workers.pdf`](../runtime/figures/communication_fraction_vs_workers.pdf)：systems diagnostic。",
            "",
            "### 仅作为 negative-scaling diagnostic",
            "",
            "- [`nog_strong_scaling_speedup.pdf`](../runtime/figures/nog_strong_scaling_speedup.pdf)：不能作为 positive scaling evidence；如果篇幅有限，不建议放正文。",
            "",
            "## 建议写入论文的英文表述",
            "",
            "> On the three tolerances for which both methods achieved a 5/5 confirmed-hit rate, NOG-FO required 6.14--10.71 times fewer communication rounds than ME-DOL-FO, a qualitative advantage consistent with the improved communication dependence predicted by our theory. In a separate single-host CPU/Gloo benchmark, NOG-FO completed its pilot-frozen full budget in 0.539--0.821 times the median training time of ME-DOL-FO. This benchmark did not exhibit positive process-level strong scaling, and the frozen workloads were not matched by empirical SFO work; accordingly, we do not interpret the timing ratio as a work-matched time-to-stationarity speedup or as verification of the asymptotic work complexity.",
            "",
            "## 不应使用的 claims",
            "",
            "- `NOG scales linearly with the number of workers`；",
            "- `NOG reaches epsilon 1.2--1.9x faster in the runtime benchmark`；",
            "- `NOG and ME-DOL use the same work in practice`；",
            "- `The experiments verify the epsilon^{-5/3} communication rate`；",
            "- 把 `epsilon=0.008` 的 ME-DOL full-budget timing 写成 uncensored time-to-epsilon；",
            "- 只展示 runtime ratio 而隐藏相邻的 work-ratio panel。",
            "",
            "## Reproduction",
            "",
            "```bash",
            "/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_runtime_analysis",
            "/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_runtime_figures",
            "/root/miniconda3/envs/NOG/bin/python -m src.distributed.cpu_fo_runtime_report",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_captions() -> str:
    return r"""# Step 8 paper-candidate figure captions

## `runtime_vs_workers.pdf`

**Single-host CPU-process runtime under pilot-frozen full budgets.** Training and end-to-end times are shown for NOG-FO and ME-DOL-FO with $m\in\{1,2,4,8,16,32\}$ worker processes. Each point is the median of three measured repeats, and error bars span the observed minimum and maximum without outlier removal. Training time excludes process startup, fixed-bank evaluation, and serialization, whereas end-to-end time includes these costs. All runs use complete pilot-frozen budgets rather than first-hit stopping, and the ME-DOL configuration is shared across the three displayed tolerance labels.

## `nog_strong_scaling_speedup.pdf`

**Fixed-total-work strong-scaling diagnostic for NOG-FO.** Speedup is the ratio between the median training time at one worker and at $m$ workers. NOG's total SFO work is fixed across worker counts. Values below one show that, on this single-host CPU/Gloo implementation, process and collective overhead exceeds the available local parallelism benefit. The dashed line denotes ideal linear scaling. This diagnostic should not be generalized to multi-node or GPU collectives.

## `communication_fraction_vs_workers.pdf`

**Measured communication fraction within training time.** The plotted fraction is communication time divided by training time for the complete frozen budget. It excludes startup, evaluation, and serialization, and therefore is not a fraction of end-to-end time. The increase with worker count is consistent with the observed absence of positive single-host process scaling.

## `full_budget_method_comparison.pdf`

**Runtime and SFO-work ratios under unmatched frozen budgets.** The left panel reports the NOG-FO/ME-DOL-FO median training-time ratio, where values below one indicate that NOG completed its frozen budget faster. The right panel reports the corresponding total training SFO-work ratio. Because the accuracy-selected configurations are not work matched, the timing ratio is an implementation-level full-budget comparison rather than a work-matched time-to-stationarity speedup. ME-DOL is right-censored at $\epsilon=0.008$ in the formal accuracy experiment.
"""


def render_advisor_message() -> str:
    return """学长好，FO 的真实 CPU-process runtime/scaling benchmark 也完成并审计好了。我们在 epsilon=0.010/0.009/0.008、m=1/2/4/8/16/32 上，每个 setting 做了 warm-up 和 3 次正式重复，统计 median[min,max]，没有删除 outlier。

这台单机 CPU + Gloo 环境下没有观察到 positive scaling：NOG 和 ME-DOL 都是 m=1 最快，NOG 到 m=32 时相对 m=1 只有 0.118--0.139x speedup（实际是变慢约 7--8.5 倍）。所以不能说增加 CPU workers 后 NOG 跑得更快，也不能把这个结果外推到多机/GPU。

另一方面，在相同 m 下，NOG 完成各自 frozen full budget 的 median training time 是 ME-DOL 的 0.539--0.821x，确实更短。但两种方法没有按 empirical SFO work 匹配，NOG/ME-DOL full-budget work ratio 是 4.008--256.533x；epsilon=0.008 的 ME-DOL accuracy 还是 censored。因此这部分只能写 implementation-level full-budget runtime difference，不能写成 work-matched time-to-epsilon speedup。

和 Step 7 合起来，最稳妥的论文结论仍然是：在双方 5/5 命中的三个 epsilon 上，NOG 达到相同 threshold 的 communication depth 少 6.14--10.71x，与理论 communication advantage 定性一致；finite SFO work 没有显示优势。建议正文优先放 depth 图并配套 work 图，runtime/scaling 图放 supplement 或先作为内部 diagnostic。"""


def render_asset_recommendations() -> str:
    return """# Step 7 + Step 8 Asset Recommendations

## Main paper

1. `../figures/depth_vs_epsilon.pdf` — primary evidence for the qualitative communication-depth advantage.
2. `../figures/work_vs_epsilon.pdf` — required companion showing the finite-SFO-work tradeoff.
3. The Step 7 formal table in `../step7_final/paper_table.tex`.

## Supplement or advisor discussion

1. `../runtime/figures/runtime_vs_workers.pdf` — complete median/min/max timing evidence.
2. `../runtime/figures/full_budget_method_comparison.pdf` — use both panels together; never crop away the work ratio.
3. `../runtime/figures/communication_fraction_vs_workers.pdf` — systems-overhead diagnostic.

## Diagnostic only

- `../runtime/figures/nog_strong_scaling_speedup.pdf` — transparent negative result, not evidence of positive scaling.

## Required wording boundaries

- Use “consistent with the predicted communication advantage.”
- State that runtime uses complete frozen budgets and is not first-hit time-to-epsilon.
- State that frozen configurations are not empirical-work matched.
- State that the single-host CPU/Gloo benchmark does not show positive strong scaling.
- Preserve the right-censoring warning for ME-DOL at epsilon 0.008.
"""


def build_report_package(
    repo_root: str | Path,
    formal_root: str | Path,
    step7_root: str | Path,
    runtime_root: str | Path,
    runtime_figure_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    repo = Path(repo_root)
    formal_path = Path(formal_root)
    step7_path = Path(step7_root)
    runtime_path = Path(runtime_root)
    figure_path = Path(runtime_figure_root)
    output_path = Path(output_root)

    _, formal_summaries, _, formal_completion = load_verified_formal_inputs(
        formal_path
    )
    _, runtime_summaries, speedups, comparisons, runtime_completion = (
        load_verified_runtime_inputs(runtime_path)
    )
    step7_manifest = validate_step7_package(step7_path, formal_completion)
    figure_manifest = validate_runtime_figure_package(
        figure_path, runtime_completion
    )

    files = {
        "FINAL_RUNTIME_RESULTS.md": render_markdown_report(
            formal_summaries, runtime_summaries, speedups, comparisons
        ),
        "runtime_figure_captions.md": render_captions(),
        "advisor_message.md": render_advisor_message(),
        "asset_recommendations.md": render_asset_recommendations(),
    }
    for name, content in files.items():
        _atomic_write_text(output_path / name, content)

    paper_path = repo / "NeurIPS_NOG.pdf"
    if file_sha256(paper_path) != step7_manifest.get("paper_pdf_sha256"):
        raise ValueError("Paper PDF changed after the verified Step 7 package.")
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_manifest_sha256": formal_completion["formal_manifest_sha256"],
        "frozen_config_sha256": formal_completion["frozen_config_sha256"],
        "runtime_manifest_sha256": runtime_completion["runtime_manifest_sha256"],
        "paper_pdf_sha256": file_sha256(paper_path),
        "step7_completion_sha256": file_sha256(
            step7_path / "step7_completion.json"
        ),
        "runtime_analysis_completion_sha256": file_sha256(
            runtime_path / "runtime_analysis_completion.json"
        ),
        "runtime_figure_manifest_sha256": file_sha256(
            figure_path / "runtime_figure_manifest.json"
        ),
        "verified_runtime_figure_count": int(figure_manifest["figure_count"]),
        "deliverables": {
            name: file_sha256(output_path / name) for name in files
        },
        "joint_claim_boundary": {
            "supported": [
                "qualitative NOG communication-depth advantage on three mutual-full-hit thresholds",
                "NOG completed its unmatched frozen full budget faster in this single-host CPU/Gloo implementation",
                "no positive single-host CPU-process strong scaling was observed",
            ],
            "not_supported": [
                "empirical verification of epsilon, dimension, or delta asymptotic exponents",
                "finite-work parity or a finite-work advantage for NOG-FO",
                "work-matched time-to-epsilon runtime speedup",
                "positive parallel scaling or extrapolation to multi-node/GPU systems",
                "unconditional runtime-to-epsilon claims at right-censored thresholds",
            ],
        },
        "step8_complete": True,
    }
    atomic_write_json(output_path / "step8_completion.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--formal-root", default="outputs/distributed_cpu_fo/formal_accuracy"
    )
    parser.add_argument(
        "--step7-root", default="outputs/distributed_cpu_fo/step7_final"
    )
    parser.add_argument(
        "--runtime-root", default="outputs/distributed_cpu_fo/runtime"
    )
    parser.add_argument(
        "--runtime-figure-root",
        default="outputs/distributed_cpu_fo/runtime/figures",
    )
    parser.add_argument(
        "--output-root", default="outputs/distributed_cpu_fo/step8_final"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_report_package(
        args.repo_root,
        args.formal_root,
        args.step7_root,
        args.runtime_root,
        args.runtime_figure_root,
        args.output_root,
    )
    print(
        f"phase=runtime-report status={manifest['status']} "
        f"deliverables={len(manifest['deliverables'])} "
        f"step8_complete={manifest['step8_complete']}"
    )


if __name__ == "__main__":
    main()
