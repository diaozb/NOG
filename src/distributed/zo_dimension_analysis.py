"""Step ZO-7C: audited fixed-configuration dimension analysis.

This module merges the completed Step ZO-7B trajectories at dimensions 25,
50, and 200 with the previously audited dimension-100 formal trajectories.
It never tunes parameters and never modifies either raw input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.distributed.zo_formal_analysis import (
    bootstrap_ci,
    canonical_parameters,
    first_confirmed_hit,
    sample_sd,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "zo_experiments/dimension_scaling_manifest.json"
FREEZE = ROOT / "zo_experiments/frozen_parameters.json"
NEW_RESULTS = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/dimension"
    / "formal_fixed_params_eps003_005/results.csv"
)
D100_RESULTS = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal"
    / "fixed_work_983040/results.csv"
)
RUN_PROGRESS = NEW_RESULTS.parent / "progress.json"
FORMAL_AUDIT = ROOT / "zo_experiments/formal/audit.json"
OUTPUT = ROOT / "zo_experiments/dimension"
FIGURES = OUTPUT / "figures"

METHODS = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]
BASELINES = ["ME-DOL-ZO", "DGFM", "DGFM+"]
COLORS = {
    "NOG-ZO": "#1f77b4",
    "ME-DOL-ZO": "#d62728",
    "DGFM": "#2ca02c",
    "DGFM+": "#9467bd",
}
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260809


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_parameters(freeze: dict[str, Any]) -> dict[str, str]:
    return {
        str(entry["method"]): json.dumps(
            entry["parameters"], sort_keys=True, separators=(",", ":")
        )
        for entry in freeze["selected_candidates"]
    }


def audit_and_merge(
    manifest: dict[str, Any], freeze: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify all frozen inputs and return an in-memory four-dimension frame."""

    errors: list[str] = []
    expected_hashes = manifest["input_sha256"]
    for relative, expected in expected_hashes.items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing frozen input: {relative}")
        elif sha256(path) != expected:
            errors.append(f"frozen input hash mismatch: {relative}")

    progress = read_json(RUN_PROGRESS)
    if progress.get("status") != "complete":
        errors.append("Step ZO-7B progress is not complete")
    if int(progress.get("completed_tasks", -1)) != 240:
        errors.append("Step ZO-7B did not record 240 completed tasks")

    formal_audit = read_json(FORMAL_AUDIT)
    if formal_audit.get("status") != "pass":
        errors.append("dimension-100 source formal audit did not pass")
    if formal_audit.get("raw_results_sha256") != sha256(D100_RESULTS):
        errors.append("dimension-100 source hash differs from formal audit")

    new = pd.read_csv(NEW_RESULTS)
    reused = pd.read_csv(D100_RESULTS)
    if "dimension" in reused.columns:
        errors.append("dimension-100 source unexpectedly contains dimension")
    reused["dimension"] = 100
    reused["seed_role"] = "formal_reused_dimension_100"
    combined = pd.concat([new, reused], ignore_index=True, sort=False)

    dimensions = [int(value) for value in manifest["dimensions"]]
    run_dimensions = [int(value) for value in manifest["dimensions_to_run"]]
    seeds = [int(value) for value in manifest["formal_seeds"]]
    parameters = selected_parameters(freeze)
    expected_tasks = {
        (dimension, method, seed)
        for dimension in dimensions
        for method in METHODS
        for seed in seeds
    }
    observed_tasks = {
        (int(dimension), str(method), int(seed))
        for dimension, method, seed in combined[
            ["dimension", "method", "formal_seed"]
        ]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if observed_tasks != expected_tasks:
        errors.append(
            "dimension/method/seed mismatch: "
            f"missing={sorted(expected_tasks - observed_tasks)}, "
            f"extra={sorted(observed_tasks - expected_tasks)}"
        )

    observed_new_dimensions = sorted(new["dimension"].astype(int).unique())
    if observed_new_dimensions != sorted(run_dimensions):
        errors.append(
            f"ZO-7B dimensions {observed_new_dimensions} != {run_dimensions}"
        )

    target_work = int(manifest["target_total_work"])
    worker_count = int(manifest["worker_count"])
    task_records: list[dict[str, Any]] = []
    for (dimension, method, seed), frame in combined.groupby(
        ["dimension", "method", "formal_seed"], sort=True
    ):
        ordered = frame.sort_values("depth").reset_index(drop=True)
        observed_parameters = {
            canonical_parameters(value)
            for value in ordered["candidate_parameters"].dropna().astype(str)
        }
        if observed_parameters != {parameters[str(method)]}:
            errors.append(
                f"d={dimension}/{method}/seed-{seed}: parameter mismatch"
            )
        depths = ordered["depth"].to_numpy(dtype=float)
        works = ordered["total_work"].to_numpy(dtype=float)
        proxy = ordered["stat_proxy"].to_numpy(dtype=float)
        if not np.all(np.diff(depths) > 0):
            errors.append(
                f"d={dimension}/{method}/seed-{seed}: non-increasing depth"
            )
        if not np.all(np.diff(works) > 0):
            errors.append(
                f"d={dimension}/{method}/seed-{seed}: non-increasing work"
            )
        if int(works[-1]) != target_work:
            errors.append(
                f"d={dimension}/{method}/seed-{seed}: final work mismatch"
            )
        if set(ordered["worker_count"].astype(int)) != {worker_count}:
            errors.append(
                f"d={dimension}/{method}/seed-{seed}: worker mismatch"
            )
        if not np.all(np.isfinite(proxy)) or np.any(proxy < 0):
            errors.append(
                f"d={dimension}/{method}/seed-{seed}: invalid stat_proxy"
            )
        task_records.append(
            {
                "dimension": int(dimension),
                "method": str(method),
                "formal_seed": int(seed),
                "checkpoints": int(len(ordered)),
                "final_depth": int(depths[-1]),
                "final_work": int(works[-1]),
                "min_stat_proxy": float(proxy.min()),
            }
        )

    tasks = pd.DataFrame(task_records)
    audit = {
        "step": "ZO-7C",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "rows": int(len(combined)),
        "tasks_expected": int(len(expected_tasks)),
        "tasks_observed": int(len(observed_tasks)),
        "dimensions": dimensions,
        "methods": METHODS,
        "formal_seeds": seeds,
        "target_total_work": target_work,
        "worker_count": worker_count,
        "new_results": str(NEW_RESULTS.relative_to(ROOT)),
        "new_results_sha256": sha256(NEW_RESULTS),
        "dimension_100_results": str(D100_RESULTS.relative_to(ROOT)),
        "dimension_100_results_sha256": sha256(D100_RESULTS),
        "manifest_sha256": sha256(MANIFEST),
        "freeze_sha256": sha256(FREEZE),
        "task_checkpoint_counts": {
            f"d={dimension}/{method}": sorted(
                tasks.loc[
                    (tasks["dimension"] == dimension)
                    & (tasks["method"] == method),
                    "checkpoints",
                ]
                .astype(int)
                .unique()
                .tolist()
            )
            for dimension in dimensions
            for method in METHODS
        },
        "task_final_depths": {
            f"d={dimension}/{method}": sorted(
                tasks.loc[
                    (tasks["dimension"] == dimension)
                    & (tasks["method"] == method),
                    "final_depth",
                ]
                .astype(int)
                .unique()
                .tolist()
            )
            for dimension in dimensions
            for method in METHODS
        },
        "selected_parameters": parameters,
    }
    return combined, audit


def build_per_seed(
    results: pd.DataFrame,
    epsilons: list[float],
    consecutive: int,
    primary_epsilons: list[float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    primary = set(primary_epsilons)
    for (dimension, method, seed), frame in results.groupby(
        ["dimension", "method", "formal_seed"], sort=False
    ):
        ordered = frame.sort_values("depth").reset_index(drop=True)
        censor_depth = float(ordered["depth"].iloc[-1])
        censor_work = float(ordered["total_work"].iloc[-1])
        for epsilon in epsilons:
            hit = first_confirmed_hit(ordered, epsilon, consecutive)
            records.append(
                {
                    "dimension": int(dimension),
                    "method": str(method),
                    "formal_seed": int(seed),
                    "epsilon": float(epsilon),
                    "scope": (
                        "primary" if epsilon in primary else "descriptive"
                    ),
                    "hit": hit is not None,
                    "first_hit_depth": (
                        float(hit["depth"]) if hit is not None else np.nan
                    ),
                    "first_hit_work": (
                        float(hit["total_work"])
                        if hit is not None
                        else np.nan
                    ),
                    "first_hit_stat_proxy": (
                        float(hit["stat_proxy"])
                        if hit is not None
                        else np.nan
                    ),
                    "censor_depth": censor_depth,
                    "censor_work": censor_work,
                    "capped_depth": (
                        float(hit["depth"])
                        if hit is not None
                        else censor_depth
                    ),
                    "capped_work": (
                        float(hit["total_work"])
                        if hit is not None
                        else censor_work
                    ),
                }
            )
    return pd.DataFrame(records)


def build_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dimension, method, epsilon, scope), frame in per_seed.groupby(
        ["dimension", "method", "epsilon", "scope"], sort=False
    ):
        hits = frame.loc[frame["hit"]]
        records.append(
            {
                "dimension": int(dimension),
                "method": str(method),
                "epsilon": float(epsilon),
                "scope": str(scope),
                "seeds": int(len(frame)),
                "hits": int(frame["hit"].sum()),
                "hit_rate": float(frame["hit"].mean()),
                "mean_first_hit_depth": (
                    float(hits["first_hit_depth"].mean())
                    if len(hits)
                    else np.nan
                ),
                "sd_first_hit_depth": (
                    sample_sd(hits["first_hit_depth"])
                    if len(hits)
                    else np.nan
                ),
                "mean_first_hit_work": (
                    float(hits["first_hit_work"].mean())
                    if len(hits)
                    else np.nan
                ),
                "sd_first_hit_work": (
                    sample_sd(hits["first_hit_work"])
                    if len(hits)
                    else np.nan
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
    dimensions = sorted(per_seed["dimension"].astype(int).unique())
    seed_count = int(per_seed["formal_seed"].nunique())
    available_baselines = [
        baseline
        for baseline in BASELINES
        if baseline in set(per_seed["method"].astype(str))
    ]
    for dimension_index, dimension in enumerate(dimensions):
        dimension_frame = per_seed.loc[per_seed["dimension"] == dimension]
        nog = dimension_frame.loc[
            dimension_frame["method"] == "NOG-ZO"
        ].set_index(["formal_seed", "epsilon"])
        for baseline_index, baseline in enumerate(available_baselines):
            base = dimension_frame.loc[
                dimension_frame["method"] == baseline
            ].set_index(["formal_seed", "epsilon"])
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
                offset = (
                    dimension_index * 100000
                    + baseline_index * 10000
                    + epsilon_index * 10
                )
                dlow, dhigh = bootstrap_ci(
                    dvalues, repetitions, bootstrap_seed + offset
                )
                wlow, whigh = bootstrap_ci(
                    wvalues, repetitions, bootstrap_seed + offset + 1
                )
                records.append(
                    {
                        "dimension": int(dimension),
                        "baseline": baseline,
                        "ratio_direction": f"{baseline}/NOG-ZO",
                        "epsilon": float(epsilon),
                        "scope": str(
                            dimension_frame.loc[
                                dimension_frame["epsilon"].eq(epsilon),
                                "scope",
                            ].iloc[0]
                        ),
                        "paired_hits": int(len(dvalues)),
                        "total_seeds": seed_count,
                        "complete_pairing": bool(len(dvalues) == seed_count),
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


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(y, y[0]):
        return 0.0
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def build_dimension_trends(
    per_seed: pd.DataFrame,
    primary_epsilons: list[float],
    theory_powers: dict[str, dict[str, float]],
    repetitions: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    dimensions = np.asarray(
        sorted(per_seed["dimension"].astype(int).unique()), dtype=float
    )
    seeds = np.asarray(
        sorted(per_seed["formal_seed"].astype(int).unique()), dtype=int
    )
    available_baselines = [
        baseline
        for baseline in BASELINES
        if baseline in set(per_seed["method"].astype(str))
    ]
    for baseline_index, baseline in enumerate(available_baselines):
        for epsilon_index, epsilon in enumerate(primary_epsilons):
            frame = per_seed.loc[
                per_seed["epsilon"].eq(epsilon)
                & per_seed["method"].isin(["NOG-ZO", baseline])
            ]
            nog = frame.loc[frame["method"] == "NOG-ZO"].set_index(
                ["dimension", "formal_seed"]
            )
            base = frame.loc[frame["method"] == baseline].set_index(
                ["dimension", "formal_seed"]
            )
            depth = np.empty((len(seeds), len(dimensions)), dtype=float)
            work = np.empty_like(depth)
            complete = True
            for seed_index, seed in enumerate(seeds):
                for dimension_index, dimension in enumerate(dimensions):
                    nrow = nog.loc[(int(dimension), int(seed))]
                    brow = base.loc[(int(dimension), int(seed))]
                    if not (bool(nrow["hit"]) and bool(brow["hit"])):
                        complete = False
                        depth[seed_index, dimension_index] = np.nan
                        work[seed_index, dimension_index] = np.nan
                    else:
                        depth[seed_index, dimension_index] = (
                            float(brow["first_hit_depth"])
                            / float(nrow["first_hit_depth"])
                        )
                        work[seed_index, dimension_index] = (
                            float(brow["first_hit_work"])
                            / float(nrow["first_hit_work"])
                        )
            if not complete:
                raise ValueError(
                    f"Primary dimension trend is censored: {baseline}/{epsilon}"
                )
            rng = np.random.default_rng(
                bootstrap_seed + baseline_index * 10000 + epsilon_index * 100
            )
            row: dict[str, Any] = {
                "baseline": baseline,
                "ratio_direction": f"{baseline}/NOG-ZO",
                "epsilon": float(epsilon),
                "dimensions": ",".join(str(int(value)) for value in dimensions),
                "formal_seeds": int(len(seeds)),
            }
            for metric, values in [("depth", depth), ("work", work)]:
                mean_values = values.mean(axis=0)
                observed = log_slope(dimensions, mean_values)
                slopes = np.empty(repetitions, dtype=float)
                for index in range(repetitions):
                    sampled = rng.integers(0, len(seeds), size=len(seeds))
                    slopes[index] = log_slope(
                        dimensions, values[sampled].mean(axis=0)
                    )
                low, high = np.quantile(slopes, [0.025, 0.975])
                theory = float(
                    theory_powers[f"{baseline}/NOG-ZO"][metric]
                )
                row.update(
                    {
                        f"observed_{metric}_ratio_slope": observed,
                        f"{metric}_slope_ci_low": float(low),
                        f"{metric}_slope_ci_high": float(high),
                        f"{metric}_spearman": spearman(
                            dimensions, mean_values
                        ),
                        f"{metric}_ratio_d25": float(mean_values[0]),
                        f"{metric}_ratio_d200": float(mean_values[-1]),
                        f"theory_{metric}_ratio_power": theory,
                        f"{metric}_positive_direction_supported": bool(
                            low > 0
                        ),
                        f"exact_{metric}_power_inside_ci": bool(
                            low - 1e-12 <= theory <= high + 1e-12
                        ),
                    }
                )
            records.append(row)
    return pd.DataFrame(records)


def configure_dimension_axis(axis: plt.Axes, dimensions: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(dimensions, labels=[str(value) for value in dimensions])
    axis.grid(True, which="both", alpha=0.25)
    axis.set_xlabel("dimension d")


def make_figures(
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    primary_epsilons: list[float],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    dimensions = sorted(summary["dimension"].astype(int).unique())
    figure, axes = plt.subplots(
        len(primary_epsilons), 3, figsize=(15.5, 8.0), squeeze=False
    )
    for row_index, epsilon in enumerate(primary_epsilons):
        epsilon_frame = summary.loc[summary["epsilon"].eq(epsilon)]
        for method in METHODS:
            frame = epsilon_frame.loc[
                epsilon_frame["method"] == method
            ].sort_values("dimension")
            axes[row_index, 0].plot(
                frame["dimension"], frame["hit_rate"], marker="o",
                linewidth=1.7, label=method, color=COLORS[method]
            )
            axes[row_index, 1].plot(
                frame["dimension"], frame["mean_first_hit_depth"],
                marker="o", linewidth=1.7, label=method,
                color=COLORS[method]
            )
            axes[row_index, 2].plot(
                frame["dimension"], frame["mean_first_hit_work"],
                marker="o", linewidth=1.7, label=method,
                color=COLORS[method]
            )
        axes[row_index, 0].set_ylabel(
            f"epsilon={epsilon:g}\nconfirmed-hit rate"
        )
        axes[row_index, 0].set_ylim(-0.03, 1.03)
        axes[row_index, 1].set_ylabel("mean first-hit depth")
        axes[row_index, 2].set_ylabel("mean first-hit work")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 2].set_yscale("log")
        for axis in axes[row_index]:
            configure_dimension_axis(axis, dimensions)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "ZO fixed-configuration dimension sensitivity: absolute metrics"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "dimension_hit_depth_work.png", dpi=220)
    figure.savefig(
        FIGURES / "dimension_hit_depth_work.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        len(primary_epsilons), 2, figsize=(11.8, 8.0), squeeze=False
    )
    for row_index, epsilon in enumerate(primary_epsilons):
        epsilon_frame = ratios.loc[ratios["epsilon"].eq(epsilon)]
        for baseline in BASELINES:
            frame = epsilon_frame.loc[
                epsilon_frame["baseline"] == baseline
            ].sort_values("dimension")
            color = COLORS[baseline]
            for axis, metric, low, high in [
                (
                    axes[row_index, 0], "mean_depth_ratio",
                    "depth_ratio_ci_low", "depth_ratio_ci_high"
                ),
                (
                    axes[row_index, 1], "mean_work_ratio",
                    "work_ratio_ci_low", "work_ratio_ci_high"
                ),
            ]:
                axis.plot(
                    frame["dimension"], frame[metric], marker="o",
                    linewidth=1.8, label=baseline, color=color
                )
                axis.fill_between(
                    frame["dimension"].to_numpy(dtype=float),
                    frame[low].to_numpy(dtype=float),
                    frame[high].to_numpy(dtype=float),
                    alpha=0.15, color=color
                )
        axes[row_index, 0].set_ylabel(
            f"epsilon={epsilon:g}\nbaseline/NOG depth"
        )
        axes[row_index, 1].set_ylabel("baseline/NOG work")
        for axis in axes[row_index]:
            configure_dimension_axis(axis, dimensions)
            axis.set_yscale("log")
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Same-seed paired ratios; all primary points have 20 pairs"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "dimension_ratios.png", dpi=220)
    figure.savefig(
        FIGURES / "dimension_ratios.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def fmt(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):,.{digits}f}"


def write_report(
    manifest: dict[str, Any],
    audit: dict[str, Any],
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    trends: pd.DataFrame,
) -> None:
    primary = [float(value) for value in manifest["primary_epsilons"]]
    lines = [
        "# Step ZO-7C: fixed-configuration dimension sensitivity",
        "",
        "## 1. Scope and audit",
        "",
        "This report merges the completed d=25,50,200 Step ZO-7B runs with "
        "the hash-verified d=100 formal trajectories. It does not retune any "
        "method. The result is a qualitative fixed-configuration sensitivity "
        "check, not an exact dimension-exponent experiment.",
        "",
        f"- Audit: **{audit['status']}**; {audit['tasks_observed']}/"
        f"{audit['tasks_expected']} dimension-method-seed tasks.",
        f"- Dimensions: {', '.join(map(str, audit['dimensions']))}.",
        f"- Methods: {', '.join(METHODS)}; formal seeds: 0--19.",
        f"- Maximum work per task: {audit['target_total_work']:,} two-point "
        "SZO calls; eight logical workers.",
        "- A hit requires two consecutive method-independent evaluation "
        "checkpoints at or below epsilon.",
        "- Primary epsilons: 0.05 and 0.03; both have complete 20/20 "
        "coverage for every method and dimension.",
        "",
        "![Absolute dimension metrics](figures/dimension_hit_depth_work.png)",
        "",
        "## 2. Absolute confirmed first-hit results",
        "",
        "Each cell is `hits/20; mean depth; mean work`. All primary means are "
        "uncensored because all 20 seeds hit.",
        "",
    ]
    for epsilon in primary:
        lines.extend(
            [
                f"### epsilon = {epsilon:g}",
                "",
                "| dimension | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for dimension in audit["dimensions"]:
            cells: list[str] = []
            for method in METHODS:
                row = summary.loc[
                    summary["dimension"].eq(dimension)
                    & summary["method"].eq(method)
                    & summary["epsilon"].eq(epsilon)
                ].iloc[0]
                cells.append(
                    f"{int(row['hits'])}/20; "
                    f"{fmt(row['mean_first_hit_depth'], 1)}; "
                    f"{fmt(row['mean_first_hit_work'], 1)}"
                )
            lines.append(f"| {dimension} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## 3. Same-seed baseline/NOG-ZO ratios",
            "",
            "Ratios are means of the 20 within-seed ratios, not ratios of "
            "method means. Parentheses give bootstrap 95% confidence "
            "intervals.",
            "",
            "![Paired dimension ratios](figures/dimension_ratios.png)",
            "",
        ]
    )
    for epsilon in primary:
        lines.extend(
            [
                f"### epsilon = {epsilon:g}",
                "",
                "| dimension | baseline | depth ratio (95% CI) | "
                "work ratio (95% CI) |",
                "|---:|---|---:|---:|",
            ]
        )
        frame = ratios.loc[ratios["epsilon"].eq(epsilon)].sort_values(
            ["dimension", "baseline"]
        )
        for _, row in frame.iterrows():
            lines.append(
                f"| {int(row['dimension'])} | {row['baseline']} | "
                f"{fmt(row['mean_depth_ratio'])} "
                f"[{fmt(row['depth_ratio_ci_low'])}, "
                f"{fmt(row['depth_ratio_ci_high'])}] | "
                f"{fmt(row['mean_work_ratio'])} "
                f"[{fmt(row['work_ratio_ci_low'])}, "
                f"{fmt(row['work_ratio_ci_high'])}] |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. Dimension slopes and theory boundary",
            "",
            "The empirical slope fits log(mean paired ratio) against log(d). "
            "Confidence intervals jointly resample the same 20 seed IDs "
            "across all four dimensions. The theory column is a reference "
            "power only; exact exponent recovery was not preregistered.",
            "",
            "| baseline | epsilon | depth slope (95% CI) | theory | "
            "work slope (95% CI) | theory |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in trends.iterrows():
        lines.append(
            f"| {row['baseline']} | {row['epsilon']:g} | "
            f"{fmt(row['observed_depth_ratio_slope'], 3)} "
            f"[{fmt(row['depth_slope_ci_low'], 3)}, "
            f"{fmt(row['depth_slope_ci_high'], 3)}] | "
            f"{fmt(row['theory_depth_ratio_power'], 3)} | "
            f"{fmt(row['observed_work_ratio_slope'], 3)} "
            f"[{fmt(row['work_slope_ci_low'], 3)}, "
            f"{fmt(row['work_slope_ci_high'], 3)}] | "
            f"{fmt(row['theory_work_ratio_power'], 3)} |"
        )

    supported = int(trends["depth_positive_direction_supported"].sum())
    exact = int(trends["exact_depth_power_inside_ci"].sum())
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            f"Only {supported}/{len(trends)} primary baseline/epsilon depth "
            "slopes have a bootstrap interval strictly above zero, and "
            f"{exact}/{len(trends)} intervals contain the corresponding "
            "worst-case reference power.",
            "",
            "The absolute first-hit depth generally rises with dimension, "
            "but the relative baseline/NOG-ZO curves are mostly flat or "
            "non-monotone. Therefore this fixed-configuration experiment "
            "does **not** empirically recover the theoretical dimension "
            "powers. It remains useful as a reproducible sensitivity check "
            "showing that the epsilon-scale communication advantage is not "
            "created by a single d=100 run.",
            "",
            "Likely reasons include frozen d=100 hyperparameters, finite "
            "dimension-independent work budgets, non-tight worst-case bounds, "
            "and a synthetic data generator whose instance geometry changes "
            "with d. A true exponent experiment would need dimension-aware "
            "theory-prescribed batch/step scaling and a separately frozen "
            "protocol.",
            "",
            "## 6. Censoring below the primary range",
            "",
            "At epsilon=0.02, NOG-ZO, ME-DOL-ZO, and DGFM retain 20/20 hits "
            "at all dimensions, while DGFM+ has 8/20 hits at d=25 and 0/20 "
            "at d=50,100,200. Those conditional DGFM+ values are censored "
            "and are excluded from dimension-slope fitting. Full descriptive "
            "hit rates and capped values are in dimension_summary.csv.",
            "",
            "## 7. Paper-safe claim",
            "",
            "> Under one frozen configuration, the primary thresholds remain "
            "fully attainable across d=25--200, and NOG-ZO retains lower "
            "communication depth than the three baselines. The relative "
            "dimension slopes do not recover the worst-case theoretical "
            "powers, so the study is reported only as fixed-configuration "
            "dimension sensitivity.",
            "",
            "Do not state that these results verify exact dimension "
            "exponents or that parameter selection is dimension-wise optimal.",
            "",
            "## 8. Reproduction",
            "",
            "~~~bash",
            "conda run -n NOG python -m src.distributed.zo_dimension_analysis",
            "~~~",
            "",
            "Raw trajectories remain under outputs/distributed_zo and are "
            "not modified by the analysis.",
        ]
    )
    (OUTPUT / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = read_json(MANIFEST)
    freeze = read_json(FREEZE)
    results, audit = audit_and_merge(manifest, freeze)
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    if audit["status"] != "pass":
        raise RuntimeError("Dimension audit failed: " + "; ".join(audit["errors"]))

    primary = [float(value) for value in manifest["primary_epsilons"]]
    descriptive = [
        float(value) for value in manifest["descriptive_censored_epsilons"]
    ]
    epsilons = primary + [
        value for value in descriptive if value not in set(primary)
    ]
    consecutive = int(manifest["confirmed_hit_consecutive"])
    per_seed = build_per_seed(results, epsilons, consecutive, primary)
    summary = build_summary(per_seed)
    ratios = build_ratios(
        per_seed, epsilons, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED
    )
    trends = build_dimension_trends(
        per_seed,
        primary,
        manifest["theory_reference_ratio_powers"],
        BOOTSTRAP_REPETITIONS,
        BOOTSTRAP_SEED,
    )

    per_seed.to_csv(OUTPUT / "dimension_per_seed.csv", index=False)
    summary.to_csv(OUTPUT / "dimension_summary.csv", index=False)
    ratios.to_csv(OUTPUT / "dimension_ratios.csv", index=False)
    trends.to_csv(OUTPUT / "dimension_trends.csv", index=False)
    (OUTPUT / "dimension_trends.json").write_text(
        json.dumps(trends.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )
    make_figures(summary, ratios, primary)
    write_report(manifest, audit, summary, ratios, trends)

    generated = [
        OUTPUT / "audit.json",
        OUTPUT / "dimension_per_seed.csv",
        OUTPUT / "dimension_summary.csv",
        OUTPUT / "dimension_ratios.csv",
        OUTPUT / "dimension_trends.csv",
        OUTPUT / "dimension_trends.json",
        OUTPUT / "README.md",
        FIGURES / "dimension_hit_depth_work.png",
        FIGURES / "dimension_hit_depth_work.pdf",
        FIGURES / "dimension_ratios.png",
        FIGURES / "dimension_ratios.pdf",
    ]
    analysis_manifest = {
        "analysis": "Step ZO-7C",
        "paper_result_role": manifest["paper_result_role"],
        "claim_boundary": manifest["claim_boundary"],
        "input_sha256": {
            str(NEW_RESULTS.relative_to(ROOT)): sha256(NEW_RESULTS),
            str(D100_RESULTS.relative_to(ROOT)): sha256(D100_RESULTS),
            str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
            str(FREEZE.relative_to(ROOT)): sha256(FREEZE),
        },
        "confirmed_hit_consecutive": consecutive,
        "primary_epsilons": primary,
        "descriptive_epsilons": descriptive,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "generated_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in generated
        },
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"audit=pass tasks={audit['tasks_observed']} rows={audit['rows']} "
        f"per_seed_rows={len(per_seed)} summary_rows={len(summary)} "
        f"ratio_rows={len(ratios)} trend_rows={len(trends)}"
    )
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
