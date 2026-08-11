"""Step ZO-8C: audited fixed-configuration logical-worker analysis.

The analysis merges the completed m=1,2,4 trajectories with the previously
audited m=8 formal trajectories. It never modifies raw data or retunes a
method, and it does not interpret single-process timings as cluster speedup.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.distributed.run_distributed_baselines import load_config
from src.distributed.zo_formal_analysis import (
    bootstrap_ci,
    canonical_parameters,
    first_confirmed_hit,
    sample_sd,
    sha256,
)
from src.distributed.zo_refine_pilot import (
    apply_candidate,
    training_work_for_rounds,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "zo_experiments/worker_scaling_manifest.json"
FREEZE = ROOT / "zo_experiments/frozen_parameters.json"
NEW_RESULTS = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/worker"
    / "formal_fixed_params_eps005/results.csv"
)
REFERENCE_RESULTS = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal"
    / "fixed_work_983040/results.csv"
)
RUN_PROGRESS = NEW_RESULTS.parent / "progress.json"
FORMAL_AUDIT = ROOT / "zo_experiments/formal/audit.json"
OUTPUT = ROOT / "zo_experiments/worker"
FIGURES = OUTPUT / "figures"

METHODS = ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]
COLORS = {
    "NOG-ZO": "#1f77b4",
    "ME-DOL-ZO": "#d62728",
    "DGFM": "#2ca02c",
    "DGFM+": "#9467bd",
}
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260811


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
    errors: list[str] = []
    for relative, expected in manifest["input_sha256"].items():
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing frozen input: {relative}")
        elif sha256(path) != expected:
            errors.append(f"frozen input hash mismatch: {relative}")

    progress = read_json(RUN_PROGRESS)
    if progress.get("status") != "complete":
        errors.append("Step ZO-8B progress is not complete")
    if int(progress.get("completed_tasks", -1)) != 240:
        errors.append("Step ZO-8B did not record 240 completed tasks")

    formal_audit = read_json(FORMAL_AUDIT)
    if formal_audit.get("status") != "pass":
        errors.append("reference-worker formal audit did not pass")
    if formal_audit.get("raw_results_sha256") != sha256(REFERENCE_RESULTS):
        errors.append("reference-worker hash differs from formal audit")

    new = pd.read_csv(NEW_RESULTS)
    reused = pd.read_csv(REFERENCE_RESULTS)
    reused = reused.loc[reused["worker_count"].astype(int).eq(8)].copy()
    reused["worker_scaling_count"] = 8
    reused["seed_role"] = "formal_reused_worker_8"
    reused["evaluation_interval_rounds"] = reused["method"].map(
        {method: int(manifest["evaluation_interval_policy"][method]["8"])
         for method in METHODS}
    )
    combined = pd.concat([new, reused], ignore_index=True, sort=False)

    workers = [int(value) for value in manifest["workers"]]
    workers_run = [int(value) for value in manifest["workers_to_run"]]
    seeds = [int(value) for value in manifest["formal_seeds"]]
    parameters = selected_parameters(freeze)
    expected_tasks = {
        (worker, method, seed)
        for worker in workers
        for method in METHODS
        for seed in seeds
    }
    observed_tasks = {
        (int(worker), str(method), int(seed))
        for worker, method, seed in combined[
            ["worker_count", "method", "formal_seed"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    if observed_tasks != expected_tasks:
        errors.append(
            "worker/method/seed mismatch: "
            f"missing={sorted(expected_tasks - observed_tasks)}, "
            f"extra={sorted(observed_tasks - expected_tasks)}"
        )
    if sorted(new["worker_count"].astype(int).unique()) != sorted(workers_run):
        errors.append("new formal output has unexpected worker counts")

    base_cfg = load_config(ROOT / freeze["base_config"])
    target_work = int(manifest["target_total_work"])
    task_records: list[dict[str, Any]] = []
    for (worker, method, seed), frame in combined.groupby(
        ["worker_count", "method", "formal_seed"], sort=True
    ):
        worker = int(worker)
        method = str(method)
        seed = int(seed)
        label = f"m={worker}/{method}/seed-{seed}"
        ordered = frame.sort_values("depth").reset_index(drop=True)
        observed_parameters = {
            canonical_parameters(value)
            for value in ordered["candidate_parameters"].dropna().astype(str)
        }
        if observed_parameters != {parameters[method]}:
            errors.append(f"{label}: parameter mismatch")
        if set(ordered["worker_scaling_count"].astype(int)) != {worker}:
            errors.append(f"{label}: worker-scaling metadata mismatch")
        expected_interval = int(
            manifest["evaluation_interval_policy"][method][str(worker)]
        )
        if set(ordered["evaluation_interval_rounds"].astype(int)) != {
            expected_interval
        }:
            errors.append(f"{label}: evaluation interval mismatch")

        depths = ordered["depth"].to_numpy(dtype=float)
        works = ordered["total_work"].to_numpy(dtype=float)
        proxy = ordered["stat_proxy"].to_numpy(dtype=float)
        if not np.all(np.diff(depths) > 0):
            errors.append(f"{label}: non-increasing depth")
        if not np.all(np.diff(works) > 0):
            errors.append(f"{label}: non-increasing work")
        if not np.all(np.isfinite(proxy)) or np.any(proxy < 0):
            errors.append(f"{label}: invalid stat_proxy")

        rounds_values = ordered["candidate_rounds"].astype(int).unique()
        if len(rounds_values) != 1:
            errors.append(f"{label}: inconsistent candidate rounds")
            rounds = int(rounds_values[-1])
        else:
            rounds = int(rounds_values[0])
        method_cfg = copy.deepcopy(base_cfg)
        apply_candidate(method_cfg, method, json.loads(parameters[method]))
        expected_work = training_work_for_rounds(
            method_cfg, method, rounds, worker
        )
        final = ordered.iloc[-1]
        final_work = int(final["total_work"])
        if final_work != expected_work or final_work > target_work:
            errors.append(
                f"{label}: final work {final_work} != {expected_work}"
            )
        per_worker = [int(value) for value in ast.literal_eval(
            str(final["per_worker_work"])
        )]
        if len(per_worker) != worker:
            errors.append(f"{label}: final per-worker vector length mismatch")
        if sum(per_worker) != final_work:
            errors.append(f"{label}: final per-worker sum mismatch")
        if max(per_worker) != int(final["per_worker_work_max"]):
            errors.append(f"{label}: final per-worker maximum mismatch")
        task_records.append(
            {
                "worker_count": worker,
                "method": method,
                "formal_seed": seed,
                "checkpoints": int(len(ordered)),
                "final_depth": int(depths[-1]),
                "final_work": final_work,
                "final_per_worker_work_max": int(final["per_worker_work_max"]),
                "final_stat_proxy": float(proxy[-1]),
                "min_stat_proxy": float(proxy.min()),
            }
        )

    tasks = pd.DataFrame(task_records)
    audit = {
        "step": "ZO-8C",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "rows": int(len(combined)),
        "tasks_expected": int(len(expected_tasks)),
        "tasks_observed": int(len(observed_tasks)),
        "workers": workers,
        "methods": METHODS,
        "formal_seeds": seeds,
        "target_total_work": target_work,
        "new_results": str(NEW_RESULTS.relative_to(ROOT)),
        "new_results_sha256": sha256(NEW_RESULTS),
        "reference_results": str(REFERENCE_RESULTS.relative_to(ROOT)),
        "reference_results_sha256": sha256(REFERENCE_RESULTS),
        "manifest_sha256": sha256(MANIFEST),
        "freeze_sha256": sha256(FREEZE),
        "task_checkpoint_counts": {
            f"m={worker}/{method}": sorted(
                tasks.loc[
                    tasks["worker_count"].eq(worker)
                    & tasks["method"].eq(method),
                    "checkpoints",
                ].astype(int).unique().tolist()
            )
            for worker in workers for method in METHODS
        },
        "task_final_work": {
            f"m={worker}/{method}": sorted(
                tasks.loc[
                    tasks["worker_count"].eq(worker)
                    & tasks["method"].eq(method),
                    "final_work",
                ].astype(int).unique().tolist()
            )
            for worker in workers for method in METHODS
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
    for (worker, method, seed), frame in results.groupby(
        ["worker_count", "method", "formal_seed"], sort=False
    ):
        ordered = frame.sort_values("depth").reset_index(drop=True)
        final = ordered.iloc[-1]
        for epsilon in epsilons:
            hit = first_confirmed_hit(ordered, epsilon, consecutive)
            records.append(
                {
                    "worker_count": int(worker),
                    "method": str(method),
                    "formal_seed": int(seed),
                    "epsilon": float(epsilon),
                    "scope": "primary" if epsilon in primary else "descriptive",
                    "hit": hit is not None,
                    "first_hit_depth": np.nan if hit is None else float(hit["depth"]),
                    "first_hit_work": np.nan if hit is None else float(hit["total_work"]),
                    "first_hit_per_worker_work": (
                        np.nan if hit is None else float(hit["per_worker_work_max"])
                    ),
                    "first_hit_stat_proxy": (
                        np.nan if hit is None else float(hit["stat_proxy"])
                    ),
                    "censor_depth": float(final["depth"]),
                    "censor_work": float(final["total_work"]),
                    "censor_per_worker_work": float(final["per_worker_work_max"]),
                    "capped_depth": (
                        float(final["depth"]) if hit is None else float(hit["depth"])
                    ),
                    "capped_work": (
                        float(final["total_work"])
                        if hit is None else float(hit["total_work"])
                    ),
                    "capped_per_worker_work": (
                        float(final["per_worker_work_max"])
                        if hit is None else float(hit["per_worker_work_max"])
                    ),
                }
            )
    return pd.DataFrame(records)


def build_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (worker, method, epsilon, scope), frame in per_seed.groupby(
        ["worker_count", "method", "epsilon", "scope"], sort=False
    ):
        hits = frame.loc[frame["hit"]]
        records.append(
            {
                "worker_count": int(worker),
                "method": str(method),
                "epsilon": float(epsilon),
                "scope": str(scope),
                "seeds": int(len(frame)),
                "hits": int(frame["hit"].sum()),
                "hit_rate": float(frame["hit"].mean()),
                "mean_first_hit_depth": float(hits["first_hit_depth"].mean()) if len(hits) else np.nan,
                "sd_first_hit_depth": sample_sd(hits["first_hit_depth"]) if len(hits) else np.nan,
                "mean_first_hit_work": float(hits["first_hit_work"].mean()) if len(hits) else np.nan,
                "sd_first_hit_work": sample_sd(hits["first_hit_work"]) if len(hits) else np.nan,
                "mean_first_hit_per_worker_work": (
                    float(hits["first_hit_per_worker_work"].mean())
                    if len(hits) else np.nan
                ),
                "sd_first_hit_per_worker_work": (
                    sample_sd(hits["first_hit_per_worker_work"])
                    if len(hits) else np.nan
                ),
                "mean_capped_depth": float(frame["capped_depth"].mean()),
                "mean_capped_work": float(frame["capped_work"].mean()),
                "mean_capped_per_worker_work": float(
                    frame["capped_per_worker_work"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def build_relative_to_m1(
    per_seed: pd.DataFrame, epsilons: list[float], repetitions: int, seed: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    workers = sorted(per_seed["worker_count"].astype(int).unique())
    metrics = [
        "first_hit_depth",
        "first_hit_work",
        "first_hit_per_worker_work",
    ]
    available_methods = [
        method for method in METHODS
        if method in set(per_seed["method"].astype(str))
    ]
    for method_index, method in enumerate(available_methods):
        method_frame = per_seed.loc[per_seed["method"].eq(method)]
        reference = method_frame.loc[method_frame["worker_count"].eq(1)].set_index(
            ["formal_seed", "epsilon"]
        )
        for epsilon_index, epsilon in enumerate(epsilons):
            for worker_index, worker in enumerate(workers):
                current = method_frame.loc[
                    method_frame["worker_count"].eq(worker)
                ].set_index(["formal_seed", "epsilon"])
                paired: list[int] = []
                values = {metric: [] for metric in metrics}
                for formal_seed in sorted(per_seed["formal_seed"].astype(int).unique()):
                    rrow = reference.loc[(formal_seed, epsilon)]
                    crow = current.loc[(formal_seed, epsilon)]
                    if bool(rrow["hit"]) and bool(crow["hit"]):
                        paired.append(formal_seed)
                        for metric in metrics:
                            values[metric].append(float(crow[metric]) / float(rrow[metric]))
                row: dict[str, Any] = {
                    "method": method,
                    "epsilon": float(epsilon),
                    "worker_count": int(worker),
                    "ratio_direction": "m/current divided by m=1 within method",
                    "paired_hits": len(paired),
                    "total_seeds": int(per_seed["formal_seed"].nunique()),
                    "complete_pairing": len(paired) == int(per_seed["formal_seed"].nunique()),
                    "paired_seeds": ",".join(map(str, paired)),
                }
                offset = method_index * 100000 + epsilon_index * 1000 + worker_index * 10
                for metric_index, metric in enumerate(metrics):
                    array = np.asarray(values[metric], dtype=float)
                    low, high = bootstrap_ci(
                        array, repetitions, seed + offset + metric_index
                    )
                    label = metric.removeprefix("first_hit_")
                    row[f"mean_{label}_ratio"] = float(array.mean()) if array.size else np.nan
                    row[f"sd_{label}_ratio"] = sample_sd(array) if array.size else np.nan
                    row[f"{label}_ratio_ci_low"] = low
                    row[f"{label}_ratio_ci_high"] = high
                records.append(row)
    return pd.DataFrame(records)


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def build_trends(
    per_seed: pd.DataFrame,
    primary_epsilons: list[float],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    workers = np.asarray(
        sorted(per_seed["worker_count"].astype(int).unique()), dtype=float
    )
    metrics = [
        "first_hit_depth",
        "first_hit_work",
        "first_hit_per_worker_work",
    ]
    records: list[dict[str, Any]] = []
    available_methods = [
        method for method in METHODS
        if method in set(per_seed["method"].astype(str))
    ]
    for method_index, method in enumerate(available_methods):
        for epsilon_index, epsilon in enumerate(primary_epsilons):
            frame = per_seed.loc[
                per_seed["method"].eq(method)
                & per_seed["epsilon"].eq(epsilon)
            ]
            common_seeds = [
                int(formal_seed)
                for formal_seed, seed_frame in frame.groupby("formal_seed")
                if len(seed_frame) == len(workers) and seed_frame["hit"].all()
            ]
            row: dict[str, Any] = {
                "method": method,
                "epsilon": float(epsilon),
                "workers": ",".join(str(int(value)) for value in workers),
                "common_complete_seeds": len(common_seeds),
                "common_seed_ids": ",".join(map(str, common_seeds)),
            }
            for metric_index, metric in enumerate(metrics):
                values = np.asarray(
                    [
                        [
                            float(frame.loc[
                                frame["formal_seed"].eq(formal_seed)
                                & frame["worker_count"].eq(int(worker)), metric
                            ].iloc[0])
                            for worker in workers
                        ]
                        for formal_seed in common_seeds
                    ],
                    dtype=float,
                )
                mean_values = values.mean(axis=0)
                observed = log_slope(workers, mean_values)
                rng = np.random.default_rng(
                    seed + method_index * 10000 + epsilon_index * 100 + metric_index
                )
                slopes = np.empty(repetitions, dtype=float)
                for index in range(repetitions):
                    sampled = rng.integers(0, len(values), size=len(values))
                    slopes[index] = log_slope(workers, values[sampled].mean(axis=0))
                low, high = np.quantile(slopes, [0.025, 0.975])
                label = metric.removeprefix("first_hit_")
                row[f"{label}_slope"] = observed
                row[f"{label}_slope_ci_low"] = float(low)
                row[f"{label}_slope_ci_high"] = float(high)
                row[f"{label}_mean_m1"] = float(mean_values[0])
                row[f"{label}_mean_m8"] = float(mean_values[-1])
            records.append(row)
    return pd.DataFrame(records)


def build_terminal_summary(results: pd.DataFrame) -> pd.DataFrame:
    final = results.sort_values("depth").groupby(
        ["worker_count", "method", "formal_seed"], as_index=False
    ).tail(1)
    return final.groupby(["worker_count", "method"], as_index=False).agg(
        seeds=("formal_seed", "nunique"),
        final_depth_mean=("depth", "mean"),
        final_total_work_mean=("total_work", "mean"),
        final_per_worker_work_mean=("per_worker_work_max", "mean"),
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_sd=("stat_proxy", sample_sd),
    )


def configure_worker_axis(axis: plt.Axes, workers: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(workers, labels=[str(value) for value in workers])
    axis.set_xlabel("logical workers m")
    axis.grid(True, which="both", alpha=0.25)


def make_figures(
    summary: pd.DataFrame,
    relative: pd.DataFrame,
    terminal: pd.DataFrame,
    primary_epsilon: float,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    workers = sorted(summary["worker_count"].astype(int).unique())
    primary = summary.loc[summary["epsilon"].eq(primary_epsilon)]
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    specifications = [
        (axes[0, 0], "hit_rate", "confirmed-hit rate", False),
        (axes[0, 1], "mean_first_hit_depth", "conditional mean depth", True),
        (axes[1, 0], "mean_first_hit_work", "conditional mean total work", True),
        (axes[1, 1], "mean_first_hit_per_worker_work", "conditional mean per-worker work", True),
    ]
    for method in METHODS:
        frame = primary.loc[primary["method"].eq(method)].sort_values("worker_count")
        for axis, column, ylabel, log_y in specifications:
            axis.plot(
                frame["worker_count"], frame[column], marker="o", linewidth=1.8,
                label=method, color=COLORS[method]
            )
            axis.set_ylabel(ylabel)
            if log_y:
                axis.set_yscale("log")
            configure_worker_axis(axis, workers)
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        f"ZO fixed-configuration logical-worker sensitivity (epsilon={primary_epsilon:g})"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "worker_hit_depth_work.png", dpi=220)
    figure.savefig(
        FIGURES / "worker_hit_depth_work.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)

    frame = relative.loc[relative["epsilon"].eq(primary_epsilon)]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for method in METHODS:
        method_frame = frame.loc[frame["method"].eq(method)].sort_values("worker_count")
        for axis, metric, ylabel in [
            (axes[0], "depth", "depth / m=1 depth"),
            (axes[1], "work", "total work / m=1 work"),
            (axes[2], "per_worker_work", "per-worker work / m=1"),
        ]:
            axis.plot(
                method_frame["worker_count"], method_frame[f"mean_{metric}_ratio"],
                marker="o", linewidth=1.8, label=method, color=COLORS[method]
            )
            axis.fill_between(
                method_frame["worker_count"].to_numpy(dtype=float),
                method_frame[f"{metric}_ratio_ci_low"].to_numpy(dtype=float),
                method_frame[f"{metric}_ratio_ci_high"].to_numpy(dtype=float),
                alpha=0.15, color=COLORS[method]
            )
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel(ylabel)
            axis.set_yscale("log")
            configure_worker_axis(axis, workers)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Same-seed within-method ratios relative to one worker")
    figure.tight_layout()
    figure.savefig(FIGURES / "worker_relative_to_m1.png", dpi=220)
    figure.savefig(
        FIGURES / "worker_relative_to_m1.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for method in METHODS:
        method_frame = terminal.loc[terminal["method"].eq(method)].sort_values("worker_count")
        for axis, column, ylabel in [
            (axes[0], "final_total_work_mean", "terminal total work"),
            (axes[1], "final_per_worker_work_mean", "terminal per-worker work"),
            (axes[2], "final_stat_proxy_mean", "terminal stationarity proxy"),
        ]:
            axis.plot(
                method_frame["worker_count"], method_frame[column], marker="o",
                linewidth=1.8, label=method, color=COLORS[method]
            )
            configure_worker_axis(axis, workers)
            axis.set_ylabel(ylabel)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Terminal fixed-budget accounting and numerical endpoint")
    figure.tight_layout()
    figure.savefig(FIGURES / "worker_terminal_accounting.png", dpi=220)
    figure.savefig(
        FIGURES / "worker_terminal_accounting.pdf",
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
    relative: pd.DataFrame,
    trends: pd.DataFrame,
    terminal: pd.DataFrame,
) -> None:
    epsilon = float(manifest["primary_epsilons"][0])
    primary = summary.loc[summary["epsilon"].eq(epsilon)]
    lines = [
        "# Step ZO-8C: fixed-configuration logical-worker sensitivity",
        "",
        "## 1. Scope and audit",
        "",
        "This report merges the completed m=1,2,4 Step ZO-8B trajectories "
        "with the hash-verified m=8 formal trajectories. All four methods "
        "reuse their frozen d=100,m=8 parameters without worker-specific "
        "retuning. It is a logical single-process sensitivity experiment, not "
        "a wall-clock or multi-machine speedup benchmark.",
        "",
        f"- Audit: **{audit['status']}**; {audit['tasks_observed']}/"
        f"{audit['tasks_expected']} worker-method-seed tasks and "
        f"{audit['rows']:,} checkpoints.",
        "- Logical workers: 1, 2, 4, 8; methods: NOG-ZO, ME-DOL-ZO, DGFM, DGFM+.",
        "- Formal seeds: 0--19; maximum training work: 983,040 two-point SZO calls.",
        "- A confirmed hit requires two consecutive method-independent evaluation checkpoints.",
        "- Primary epsilon: 0.05. Epsilon 0.03 and below are descriptive/censored.",
        "- Checkpoint intervals were work-aligned within method; ME-DOL remains restricted to complete epochs.",
        "",
        "![Worker hit, depth, and work](figures/worker_hit_depth_work.png)",
        "",
        "## 2. Primary confirmed first-hit results",
        "",
        "Each cell is `hits/20; conditional mean depth; conditional mean total work; conditional mean per-worker work`. Conditional means must be read with hit counts.",
        "",
        "| workers | NOG-ZO | ME-DOL-ZO | DGFM | DGFM+ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for worker in audit["workers"]:
        cells: list[str] = []
        for method in METHODS:
            row = primary.loc[
                primary["worker_count"].eq(worker)
                & primary["method"].eq(method)
            ].iloc[0]
            cells.append(
                f"{int(row['hits'])}/20; {fmt(row['mean_first_hit_depth'], 1)}; "
                f"{fmt(row['mean_first_hit_work'], 1)}; "
                f"{fmt(row['mean_first_hit_per_worker_work'], 1)}"
            )
        lines.append(f"| {worker} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "DGFM+ at m=2 has 18/20 hits. Its conditional first-hit means and ratios are therefore censored and may be optimistic; the two non-hits remain in the per-seed and capped summaries.",
        "",
        "## 3. Same-seed ratios relative to one worker",
        "",
        "![Within-method worker ratios](figures/worker_relative_to_m1.png)",
        "",
        "Ratios compare each worker count with m=1 for the same method and seed. Confidence intervals are 2,000-repetition seed bootstrap intervals.",
        "",
        "| method | workers | pairs | depth ratio (95% CI) | total-work ratio (95% CI) | per-worker ratio (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    frame = relative.loc[relative["epsilon"].eq(epsilon)].sort_values(
        ["method", "worker_count"]
    )
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['worker_count'])} | "
            f"{int(row['paired_hits'])}/20 | "
            f"{fmt(row['mean_depth_ratio'])} [{fmt(row['depth_ratio_ci_low'])}, {fmt(row['depth_ratio_ci_high'])}] | "
            f"{fmt(row['mean_work_ratio'])} [{fmt(row['work_ratio_ci_low'])}, {fmt(row['work_ratio_ci_high'])}] | "
            f"{fmt(row['mean_per_worker_work_ratio'])} [{fmt(row['per_worker_work_ratio_ci_low'])}, {fmt(row['per_worker_work_ratio_ci_high'])}] |"
        )

    lines.extend([
        "",
        "## 4. Log-log sensitivity slopes",
        "",
        "Slopes fit log(mean first-hit metric) against log(m). The bootstrap jointly resamples seeds across all four worker counts. These are empirical fixed-configuration sensitivities, not preregistered exact worker exponents.",
        "",
        "| method | common complete seeds | depth slope (95% CI) | total-work slope (95% CI) | per-worker slope (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ])
    for _, row in trends.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['common_complete_seeds'])}/20 | "
            f"{fmt(row['depth_slope'], 3)} [{fmt(row['depth_slope_ci_low'], 3)}, {fmt(row['depth_slope_ci_high'], 3)}] | "
            f"{fmt(row['work_slope'], 3)} [{fmt(row['work_slope_ci_low'], 3)}, {fmt(row['work_slope_ci_high'], 3)}] | "
            f"{fmt(row['per_worker_work_slope'], 3)} [{fmt(row['per_worker_work_slope_ci_low'], 3)}, {fmt(row['per_worker_work_slope_ci_high'], 3)}] |"
        )

    nog = primary.loc[primary["method"].eq("NOG-ZO")].sort_values("worker_count")
    nog_depth_span = float(nog["mean_first_hit_depth"].max() / nog["mean_first_hit_depth"].min())
    nog_work_span = float(nog["mean_first_hit_work"].max() / nog["mean_first_hit_work"].min())
    nog_per_worker_ratio = float(
        nog.iloc[-1]["mean_first_hit_per_worker_work"]
        / nog.iloc[0]["mean_first_hit_per_worker_work"]
    )
    lines.extend([
        "",
        "## 5. Terminal accounting",
        "",
        "![Terminal accounting](figures/worker_terminal_accounting.png)",
        "",
        "Terminal total training work is 983,040 for NOG-ZO, ME-DOL-ZO, and DGFM. DGFM+ stops at the largest valid restart-aligned round and is within 576 calls of the same cap. Terminal per-worker work follows the expected near-1/m decomposition by construction. Evaluation work remains separate.",
        "",
        "## 6. Interpretation and theory boundary",
        "",
        f"For NOG-ZO at epsilon=0.05, mean first-hit depth varies by only {nog_depth_span:.3f}x and total work by {nog_work_span:.3f}x across m=1--8, while mean per-worker first-hit work falls to {nog_per_worker_ratio:.3f} of the m=1 value (approximately 1/8). This is the cleanest empirical result of the worker study.",
        "",
        "The other methods use fixed per-worker training batches, whereas NOG-ZO splits a fixed global batch. Consequently their total first-hit work need not stay constant as m changes. This implementation distinction is part of the frozen protocol and must be disclosed; the experiment does not establish a universally fair wall-clock comparison across worker counts.",
        "",
        "ME-DOL-ZO depth is mildly non-monotone at small m and rises at m=8. DGFM depth is nearly flat. DGFM+ depth falls strongly with m, but its m=2 estimate is censored. These observations are method sensitivity results, not evidence for exact asymptotic m powers.",
        "",
        "## 7. Descriptive epsilon=0.03 boundary",
        "",
        "At epsilon=0.03, ME-DOL-ZO/m=1 and DGFM+/m=2,4 have 0/20 confirmed hits; all remaining method-worker combinations have 20/20. No finite ratios or slopes are reported across that incomplete grid. Full capped and conditional values remain in worker_summary.csv.",
        "",
        "## 8. Paper-safe claim",
        "",
        "> Under one frozen configuration and fixed total training-work cap, NOG-ZO maintains essentially unchanged confirmed-first-hit communication depth and total work from one to eight logical workers, while its accounted per-worker work decreases approximately as 1/m. This is a logical work-decomposition result and not a measured multi-process speedup.",
        "",
        "Do not claim real cluster speedup, worker-wise optimal tuning, exact worker exponents, or complete primary coverage without mentioning the two DGFM+/m=2 non-hits.",
        "",
        "## 9. Reproduction",
        "",
        "~~~bash",
        "conda run -n NOG python -m src.distributed.zo_worker_analysis",
        "~~~",
        "",
        "Raw trajectories under outputs/distributed_zo are hash-audited and are not modified by this analysis.",
    ])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        raise RuntimeError("Worker audit failed: " + "; ".join(audit["errors"]))

    primary = [float(value) for value in manifest["primary_epsilons"]]
    descriptive = [
        float(value) for value in manifest["descriptive_censored_epsilons"]
    ]
    epsilons = primary + [value for value in descriptive if value not in set(primary)]
    per_seed = build_per_seed(
        results, epsilons, int(manifest["confirmed_hit_consecutive"]), primary
    )
    summary = build_summary(per_seed)
    relative = build_relative_to_m1(
        per_seed, epsilons, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED
    )
    trends = build_trends(
        per_seed, primary, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED
    )
    terminal = build_terminal_summary(results)

    per_seed.to_csv(OUTPUT / "worker_per_seed.csv", index=False)
    summary.to_csv(OUTPUT / "worker_summary.csv", index=False)
    relative.to_csv(OUTPUT / "worker_relative_to_m1.csv", index=False)
    trends.to_csv(OUTPUT / "worker_trends.csv", index=False)
    terminal.to_csv(OUTPUT / "worker_terminal_summary.csv", index=False)
    (OUTPUT / "worker_trends.json").write_text(
        json.dumps(trends.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )
    make_figures(summary, relative, terminal, primary[0])
    write_report(manifest, audit, summary, relative, trends, terminal)

    generated = [
        OUTPUT / "audit.json",
        OUTPUT / "worker_per_seed.csv",
        OUTPUT / "worker_summary.csv",
        OUTPUT / "worker_relative_to_m1.csv",
        OUTPUT / "worker_trends.csv",
        OUTPUT / "worker_trends.json",
        OUTPUT / "worker_terminal_summary.csv",
        OUTPUT / "README.md",
        FIGURES / "worker_hit_depth_work.png",
        FIGURES / "worker_hit_depth_work.pdf",
        FIGURES / "worker_relative_to_m1.png",
        FIGURES / "worker_relative_to_m1.pdf",
        FIGURES / "worker_terminal_accounting.png",
        FIGURES / "worker_terminal_accounting.pdf",
    ]
    analysis_manifest = {
        "analysis": "Step ZO-8C",
        "paper_result_role": manifest["paper_result_role"],
        "claim_boundary": manifest["claim_boundary"],
        "input_sha256": {
            str(NEW_RESULTS.relative_to(ROOT)): sha256(NEW_RESULTS),
            str(REFERENCE_RESULTS.relative_to(ROOT)): sha256(REFERENCE_RESULTS),
            str(MANIFEST.relative_to(ROOT)): sha256(MANIFEST),
            str(FREEZE.relative_to(ROOT)): sha256(FREEZE),
        },
        "confirmed_hit_consecutive": int(manifest["confirmed_hit_consecutive"]),
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
        f"relative_rows={len(relative)} trend_rows={len(trends)}"
    )
    print(f"saved={OUTPUT}")


if __name__ == "__main__":
    main()
