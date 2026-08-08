"""Audit and analyse the frozen 20-seed formal ZO experiment.

This module is deliberately separate from the formal runner.  It never tunes
parameters and never modifies the raw formal trajectories.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
FORMAL = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040"
)
RESULTS = FORMAL / "results.csv"
FREEZE = FORMAL / "frozen_parameters.json"
OUTPUT = ROOT / "zo_experiments/formal"
FIGURES = OUTPUT / "figures"

METHODS = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]
BASELINES = ["ME-DOL-ZO", "DGFM", "DGFM+"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_sd(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def first_confirmed_hit(
    ordered: pd.DataFrame, epsilon: float, consecutive: int
) -> pd.Series | None:
    below = ordered["stat_proxy"].to_numpy(dtype=float) <= epsilon
    if consecutive <= 1:
        starts = np.flatnonzero(below)
    else:
        window = np.convolve(
            below.astype(np.int8),
            np.ones(consecutive, dtype=np.int8),
            mode="valid",
        )
        starts = np.flatnonzero(window == consecutive)
    if starts.size == 0:
        return None
    return ordered.iloc[int(starts[0])]


def bootstrap_ci(
    values: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def canonical_parameters(value: str) -> str:
    return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))


def audit(
    results: pd.DataFrame,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    target_work = int(freeze["target_total_work"])
    expected_seeds = {int(value) for value in freeze["formal_seeds"]}
    selected = {
        entry["method"]: json.dumps(
            entry["parameters"], sort_keys=True, separators=(",", ":")
        )
        for entry in freeze["selected_candidates"]
    }
    observed_pairs = {
        (str(method), int(seed))
        for method, seed in results[["method", "formal_seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    expected_pairs = {
        (method, seed) for method in selected for seed in expected_seeds
    }
    if observed_pairs != expected_pairs:
        errors.append(
            "method/seed task mismatch: "
            f"missing={sorted(expected_pairs - observed_pairs)}, "
            f"extra={sorted(observed_pairs - expected_pairs)}"
        )

    task_rows: list[dict[str, Any]] = []
    for (method, seed), frame in results.groupby(
        ["method", "formal_seed"], sort=True
    ):
        ordered = frame.sort_values("depth").reset_index(drop=True)
        parameters = {
            canonical_parameters(value)
            for value in ordered["candidate_parameters"].dropna().astype(str)
        }
        if parameters != {selected[str(method)]}:
            errors.append(
                f"{method}/seed-{seed}: parameter mismatch {parameters}"
            )
        depths = ordered["depth"].to_numpy(dtype=float)
        works = ordered["total_work"].to_numpy(dtype=float)
        if not np.all(np.diff(depths) > 0):
            errors.append(f"{method}/seed-{seed}: depth is not strictly increasing")
        if not np.all(np.diff(works) > 0):
            errors.append(f"{method}/seed-{seed}: work is not strictly increasing")
        if int(works[-1]) != target_work:
            errors.append(
                f"{method}/seed-{seed}: final work {works[-1]} != {target_work}"
            )
        proxy = ordered["stat_proxy"].to_numpy(dtype=float)
        if not np.all(np.isfinite(proxy)) or np.any(proxy < 0):
            errors.append(f"{method}/seed-{seed}: invalid stat_proxy")
        task_rows.append(
            {
                "method": str(method),
                "formal_seed": int(seed),
                "checkpoints": int(len(ordered)),
                "final_depth": int(depths[-1]),
                "final_work": int(works[-1]),
                "final_stat_proxy": float(proxy[-1]),
                "min_stat_proxy": float(proxy.min()),
                "parameters": selected[str(method)],
            }
        )

    task_frame = pd.DataFrame(task_rows)
    return {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "raw_results": str(RESULTS.relative_to(ROOT)),
        "raw_results_sha256": sha256(RESULTS),
        "freeze_sha256": sha256(FREEZE),
        "rows": int(len(results)),
        "tasks_expected": int(len(expected_pairs)),
        "tasks_observed": int(len(observed_pairs)),
        "methods": sorted(selected),
        "formal_seeds": sorted(expected_seeds),
        "target_total_work": target_work,
        "task_checkpoint_counts": {
            method: sorted(
                task_frame.loc[
                    task_frame["method"] == method, "checkpoints"
                ].astype(int).unique().tolist()
            )
            for method in METHODS
        },
        "task_final_depths": {
            method: sorted(
                task_frame.loc[
                    task_frame["method"] == method, "final_depth"
                ].astype(int).unique().tolist()
            )
            for method in METHODS
        },
        "selected_parameters": selected,
    }


def build_per_seed(
    results: pd.DataFrame,
    epsilons: list[float],
    consecutive: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, seed), frame in results.groupby(
        ["method", "formal_seed"], sort=False
    ):
        ordered = frame.sort_values("depth").reset_index(drop=True)
        censor_depth = float(ordered["depth"].iloc[-1])
        censor_work = float(ordered["total_work"].iloc[-1])
        for epsilon in epsilons:
            hit = first_confirmed_hit(ordered, epsilon, consecutive)
            records.append(
                {
                    "method": str(method),
                    "formal_seed": int(seed),
                    "epsilon": float(epsilon),
                    "hit": hit is not None,
                    "first_hit_depth": (
                        float(hit["depth"]) if hit is not None else np.nan
                    ),
                    "first_hit_work": (
                        float(hit["total_work"]) if hit is not None else np.nan
                    ),
                    "first_hit_stat_proxy": (
                        float(hit["stat_proxy"]) if hit is not None else np.nan
                    ),
                    "censor_depth": censor_depth,
                    "censor_work": censor_work,
                    "capped_depth": (
                        float(hit["depth"]) if hit is not None else censor_depth
                    ),
                    "capped_work": (
                        float(hit["total_work"]) if hit is not None else censor_work
                    ),
                }
            )
    return pd.DataFrame(records)


def build_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, epsilon), frame in per_seed.groupby(
        ["method", "epsilon"], sort=False
    ):
        hits = frame.loc[frame["hit"]]
        records.append(
            {
                "method": str(method),
                "epsilon": float(epsilon),
                "seeds": int(len(frame)),
                "hits": int(frame["hit"].sum()),
                "hit_rate": float(frame["hit"].mean()),
                "mean_first_hit_depth": (
                    float(hits["first_hit_depth"].mean())
                    if len(hits)
                    else np.nan
                ),
                "sd_first_hit_depth": (
                    sample_sd(hits["first_hit_depth"]) if len(hits) else np.nan
                ),
                "mean_first_hit_work": (
                    float(hits["first_hit_work"].mean())
                    if len(hits)
                    else np.nan
                ),
                "sd_first_hit_work": (
                    sample_sd(hits["first_hit_work"]) if len(hits) else np.nan
                ),
                "mean_capped_depth": float(frame["capped_depth"].mean()),
                "sd_capped_depth": sample_sd(frame["capped_depth"]),
                "mean_capped_work": float(frame["capped_work"].mean()),
                "sd_capped_work": sample_sd(frame["capped_work"]),
            }
        )
    return pd.DataFrame(records)


def build_ratios(
    per_seed: pd.DataFrame,
    epsilons: list[float],
    repetitions: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    nog = per_seed.loc[per_seed["method"] == "NOG-ZO"].set_index(
        ["formal_seed", "epsilon"]
    )
    for baseline_index, baseline in enumerate(BASELINES):
        base = per_seed.loc[per_seed["method"] == baseline].set_index(
            ["formal_seed", "epsilon"]
        )
        for epsilon_index, epsilon in enumerate(epsilons):
            depth_ratios: list[float] = []
            work_ratios: list[float] = []
            paired_seeds: list[int] = []
            for seed in sorted(
                set(nog.index.get_level_values(0))
                & set(base.index.get_level_values(0))
            ):
                nrow = nog.loc[(seed, epsilon)]
                brow = base.loc[(seed, epsilon)]
                if bool(nrow["hit"]) and bool(brow["hit"]):
                    paired_seeds.append(int(seed))
                    depth_ratios.append(
                        float(brow["first_hit_depth"])
                        / float(nrow["first_hit_depth"])
                    )
                    work_ratios.append(
                        float(brow["first_hit_work"])
                        / float(nrow["first_hit_work"])
                    )
            dvalues = np.asarray(depth_ratios, dtype=float)
            wvalues = np.asarray(work_ratios, dtype=float)
            seed_offset = baseline_index * 10000 + epsilon_index * 10
            dlow, dhigh = bootstrap_ci(
                dvalues, repetitions, bootstrap_seed + seed_offset
            )
            wlow, whigh = bootstrap_ci(
                wvalues, repetitions, bootstrap_seed + seed_offset + 1
            )
            records.append(
                {
                    "baseline": baseline,
                    "ratio_direction": f"{baseline}/NOG-ZO",
                    "epsilon": float(epsilon),
                    "paired_hits": int(len(dvalues)),
                    "total_seeds": int(
                        len(set(nog.index.get_level_values(0)))
                    ),
                    "complete_pairing": bool(
                        len(dvalues) == len(set(nog.index.get_level_values(0)))
                    ),
                    "paired_seeds": ",".join(map(str, paired_seeds)),
                    "mean_depth_ratio": (
                        float(dvalues.mean()) if dvalues.size else np.nan
                    ),
                    "sd_depth_ratio": (
                        sample_sd(dvalues) if dvalues.size else np.nan
                    ),
                    "depth_ratio_ci_low": dlow,
                    "depth_ratio_ci_high": dhigh,
                    "mean_work_ratio": (
                        float(wvalues.mean()) if wvalues.size else np.nan
                    ),
                    "sd_work_ratio": (
                        sample_sd(wvalues) if wvalues.size else np.nan
                    ),
                    "work_ratio_ci_low": wlow,
                    "work_ratio_ci_high": whigh,
                }
            )
    return pd.DataFrame(records)


def trend_summary(ratios: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for baseline in BASELINES:
        frame = ratios.loc[
            (ratios["baseline"] == baseline)
            & ratios["complete_pairing"]
            & ratios["mean_depth_ratio"].notna()
        ].sort_values("epsilon", ascending=False)
        entry: dict[str, Any] = {
            "complete_pairing_epsilons": frame["epsilon"].tolist(),
            "complete_pairing_points": int(len(frame)),
        }
        for metric in ["mean_depth_ratio", "mean_work_ratio"]:
            finite = frame.loc[(frame[metric] > 0) & frame[metric].notna()]
            if len(finite) >= 2:
                x = np.log(1.0 / finite["epsilon"].to_numpy(dtype=float))
                y = np.log(finite[metric].to_numpy(dtype=float))
                slope = float(np.polyfit(x, y, 1)[0])
                spearman = float(
                    pd.Series(x).rank().corr(pd.Series(y).rank())
                )
            else:
                slope = float("nan")
                spearman = float("nan")
            entry[metric] = {
                "log_log_slope_vs_inverse_epsilon": slope,
                "spearman_vs_inverse_epsilon": spearman,
            }
        output[baseline] = entry
    return output


def configure_epsilon_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.grid(True, which="both", alpha=0.25)
    axis.set_xlabel("epsilon (smaller to the right)")


def make_figures(summary: pd.DataFrame, ratios: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "NOG-ZO": "#1f77b4",
        "ME-DOL-ZO": "#d62728",
        "DGFM": "#2ca02c",
        "DGFM+": "#9467bd",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for method in METHODS:
        frame = summary.loc[summary["method"] == method].sort_values(
            "epsilon", ascending=False
        )
        axes[0].plot(
            frame["epsilon"],
            frame["hit_rate"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=method,
            color=colors[method],
        )
        complete = frame.loc[frame["hits"] == frame["seeds"]]
        partial = frame.loc[
            (frame["hits"] > 0) & (frame["hits"] < frame["seeds"])
        ]
        axes[1].plot(
            complete["epsilon"],
            complete["mean_first_hit_depth"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=method,
            color=colors[method],
        )
        axes[1].scatter(
            partial["epsilon"],
            partial["mean_first_hit_depth"],
            marker="x",
            s=40,
            color=colors[method],
        )
        axes[2].plot(
            complete["epsilon"],
            complete["mean_first_hit_work"],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=method,
            color=colors[method],
        )
        axes[2].scatter(
            partial["epsilon"],
            partial["mean_first_hit_work"],
            marker="x",
            s=40,
            color=colors[method],
        )
    axes[0].set_ylabel("confirmed-hit rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("mean first-hit depth")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("mean first-hit work")
    axes[2].set_yscale("log")
    for axis in axes:
        configure_epsilon_axis(axis)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Formal ZO experiment: 20 seeds, fixed total work 983,040"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "formal_hit_depth_work.png", dpi=220)
    figure.savefig(FIGURES / "formal_hit_depth_work.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for baseline in BASELINES:
        frame = ratios.loc[ratios["baseline"] == baseline].sort_values(
            "epsilon", ascending=False
        )
        complete = frame.loc[frame["complete_pairing"]]
        partial = frame.loc[
            (~frame["complete_pairing"]) & (frame["paired_hits"] > 0)
        ]
        color = colors[baseline]
        for axis, metric, low, high in [
            (
                axes[0],
                "mean_depth_ratio",
                "depth_ratio_ci_low",
                "depth_ratio_ci_high",
            ),
            (
                axes[1],
                "mean_work_ratio",
                "work_ratio_ci_low",
                "work_ratio_ci_high",
            ),
        ]:
            axis.plot(
                complete["epsilon"],
                complete[metric],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=baseline,
                color=color,
            )
            axis.fill_between(
                complete["epsilon"].to_numpy(dtype=float),
                complete[low].to_numpy(dtype=float),
                complete[high].to_numpy(dtype=float),
                alpha=0.15,
                color=color,
            )
            axis.scatter(
                partial["epsilon"],
                partial[metric],
                marker="x",
                s=45,
                color=color,
            )
    axes[0].set_ylabel("baseline / NOG-ZO depth ratio")
    axes[1].set_ylabel("baseline / NOG-ZO work ratio")
    for axis in axes:
        configure_epsilon_axis(axis)
        axis.set_yscale("log")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Paired formal ratios; solid lines use all 20 paired hits"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "formal_ratios.png", dpi=220)
    figure.savefig(FIGURES / "formal_ratios.pdf")
    plt.close(figure)


def format_value(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def write_report(
    freeze: dict[str, Any],
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    trends: dict[str, Any],
) -> None:
    representatives = [
        float(value)
        for value in [0.2, 0.1, 0.05, 0.02, 0.018, 0.016, 0.015, 0.014, 0.013, 0.01, 0.009, 0.008, 0.005, 0.002]
    ]
    lines = [
        "# Formal ZO experiment (Step ZO-5B)",
        "",
        "This report is generated from the frozen 20-seed formal run. "
        "No formal seed was used for parameter selection.",
        "",
        "## Protocol",
        "",
        f"- Methods: {', '.join(METHODS)}.",
        f"- Formal seeds: {len(freeze['formal_seeds'])} "
        f"({min(freeze['formal_seeds'])} through {max(freeze['formal_seeds'])}).",
        f"- Equal total-work budget per method and seed: "
        f"{int(freeze['target_total_work']):,}.",
        f"- Workers: {int(freeze['worker_count'])}; evaluation every "
        f"{int(freeze['eval_every'])} communication-depth units.",
        f"- Evaluation bank: smooth B={int(freeze['eval_smooth_B'])}, "
        f"data B={int(freeze['eval_data_B'])}.",
        f"- A hit requires {int(freeze['confirmed_hit_consecutive'])} "
        "consecutive checkpoints at or below epsilon.",
        "- Primary endpoint: paired communication-depth ratio, baseline / NOG-ZO.",
        "- Work ratios are secondary descriptive endpoints.",
        "",
        "Frozen candidates:",
        "",
    ]
    for entry in freeze["selected_candidates"]:
        lines.append(
            f"- {entry['method']}: "
            f"{json.dumps(entry['parameters'], sort_keys=True)}"
        )
    lines.extend(
        [
            "",
            "## Representative first-hit results",
            "",
            "Each cell reports **hits/20; conditional mean depth; "
            "conditional mean work**. Conditional means exclude non-hits and "
            "must not be interpreted as complete-sample means.",
            "",
            "| epsilon | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for epsilon in representatives:
        cells = []
        for method in METHODS:
            row = summary.loc[
                (summary["method"] == method)
                & np.isclose(summary["epsilon"], epsilon)
            ].iloc[0]
            cells.append(
                f"{int(row['hits'])}/20; "
                f"{format_value(row['mean_first_hit_depth'], 1)}; "
                f"{format_value(row['mean_first_hit_work'], 1)}"
            )
        lines.append(f"| {epsilon:.4f} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Representative paired ratios",
            "",
            "Ratios are computed within the same formal seed and then averaged. "
            "Only rows with all 20 paired hits support the complete-pair trend; "
            "partial rows are explicitly labelled.",
            "",
            "| baseline | epsilon | paired hits | depth ratio (95% CI) | "
            "work ratio (95% CI) | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for baseline in BASELINES:
        for epsilon in representatives:
            row = ratios.loc[
                (ratios["baseline"] == baseline)
                & np.isclose(ratios["epsilon"], epsilon)
            ].iloc[0]
            status = "complete" if bool(row["complete_pairing"]) else "censored"
            lines.append(
                f"| {baseline} | {epsilon:.4f} | "
                f"{int(row['paired_hits'])}/20 | "
                f"{format_value(row['mean_depth_ratio'])} "
                f"[{format_value(row['depth_ratio_ci_low'])}, "
                f"{format_value(row['depth_ratio_ci_high'])}] | "
                f"{format_value(row['mean_work_ratio'])} "
                f"[{format_value(row['work_ratio_ci_low'])}, "
                f"{format_value(row['work_ratio_ci_high'])}] | {status} |"
            )

    lines.extend(
        [
            "",
            "## Trend diagnostics",
            "",
            "The diagnostics below use only epsilon points at which all 20 "
            "same-seed pairs hit. A positive log-log slope or positive "
            "Spearman coefficient means the ratio tends to increase as "
            "epsilon decreases.",
            "",
            "| baseline | complete points | depth slope | depth Spearman | "
            "work slope | work Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in BASELINES:
        entry = trends[baseline]
        depth = entry["mean_depth_ratio"]
        work = entry["mean_work_ratio"]
        lines.append(
            f"| {baseline} | {entry['complete_pairing_points']} | "
            f"{format_value(depth['log_log_slope_vs_inverse_epsilon'], 3)} | "
            f"{format_value(depth['spearman_vs_inverse_epsilon'], 3)} | "
            f"{format_value(work['log_log_slope_vs_inverse_epsilon'], 3)} | "
            f"{format_value(work['spearman_vs_inverse_epsilon'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Formal hit rate, depth and work](figures/formal_hit_depth_work.png)",
            "",
            "![Formal paired ratios](figures/formal_ratios.png)",
            "",
            "In both figures, solid ratio curves require all 20 paired hits. "
            "Crosses denote partially censored points and are not used as "
            "complete-sample scaling evidence.",
            "",
            "## Reproducibility artifacts",
            "",
            "- formal_per_seed.csv: first confirmed hit or censor limit for "
            "every method, seed and epsilon.",
            "- formal_summary.csv: hit counts and conditional/capped summaries.",
            "- formal_ratios.csv: same-seed paired ratios with bootstrap 95% CIs.",
            "- formal_trends.json: diagnostics on the complete-pair interval.",
            "- audit.json: identity, parameter, monotonicity and budget checks.",
            "- analysis_manifest.json: hashes of analysis inputs and outputs.",
            "",
            "Regenerate from the repository root with:",
            "",
            "    conda run -n NOG python -m src.distributed.zo_formal_analysis",
            "",
            "## Interpretation boundary",
            "",
            "This finite-budget simulation can support an empirical depth "
            "trend, but it does not prove the asymptotic exponents. At smaller "
            "epsilon values, non-hits are right-censored by the fixed budget. "
            "Conditional first-hit means in that region are susceptible to "
            "survivor bias; capped summaries are supplied only as budget-bound "
            "descriptions, not as true first-hit ratios.",
            "",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (FORMAL / "config_frozen.yaml").read_text(encoding="utf-8")
    )
    epsilons = [
        float(value) for value in config["epsilon_scaling"]["epsilons"]
    ]
    repetitions = int(
        config["epsilon_scaling"]["statistics"]["bootstrap_repetitions"]
    )
    bootstrap_seed = int(
        config["epsilon_scaling"]["statistics"]["bootstrap_seed"]
    )
    results = pd.read_csv(RESULTS)
    audit_record = audit(results, freeze)
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit_record, indent=2) + "\n", encoding="utf-8"
    )
    if audit_record["status"] != "pass":
        raise RuntimeError(
            "Formal audit failed: " + "; ".join(audit_record["errors"])
        )

    per_seed = build_per_seed(
        results,
        epsilons,
        int(freeze["confirmed_hit_consecutive"]),
    )
    summary = build_summary(per_seed)
    ratios = build_ratios(
        per_seed,
        epsilons,
        repetitions,
        bootstrap_seed,
    )
    trends = trend_summary(ratios)

    per_seed.to_csv(OUTPUT / "formal_per_seed.csv", index=False)
    summary.to_csv(OUTPUT / "formal_summary.csv", index=False)
    ratios.to_csv(OUTPUT / "formal_ratios.csv", index=False)
    (OUTPUT / "formal_trends.json").write_text(
        json.dumps(trends, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(summary, ratios)
    write_report(freeze, summary, ratios, trends)

    generated = [
        OUTPUT / "audit.json",
        OUTPUT / "formal_per_seed.csv",
        OUTPUT / "formal_summary.csv",
        OUTPUT / "formal_ratios.csv",
        OUTPUT / "formal_trends.json",
        OUTPUT / "README.md",
        FIGURES / "formal_hit_depth_work.png",
        FIGURES / "formal_hit_depth_work.pdf",
        FIGURES / "formal_ratios.png",
        FIGURES / "formal_ratios.pdf",
    ]
    manifest = {
        "analysis": "Step ZO-5B",
        "raw_results": str(RESULTS.relative_to(ROOT)),
        "raw_results_sha256": sha256(RESULTS),
        "frozen_parameters": str(FREEZE.relative_to(ROOT)),
        "frozen_parameters_sha256": sha256(FREEZE),
        "confirmed_hit_consecutive": int(
            freeze["confirmed_hit_consecutive"]
        ),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "generated_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in generated
        },
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"audit=pass tasks={audit_record['tasks_observed']} "
        f"raw_rows={audit_record['rows']} per_seed_rows={len(per_seed)} "
        f"summary_rows={len(summary)} ratio_rows={len(ratios)}"
    )
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
