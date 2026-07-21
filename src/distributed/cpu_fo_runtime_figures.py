"""Generate Step 8D figures from hash-verified Step 8C runtime summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


FIGURE_SCHEMA_VERSION = 1
METHOD_ORDER = ("NOG-FO", "ME-DOL-FO")
EPSILON_ORDER = (0.010, 0.009, 0.008)
WORKER_ORDER = (1, 2, 4, 8, 16, 32)
METHOD_STYLE = {
    "NOG-FO": {"color": "#0072B2", "marker": "o"},
    "ME-DOL-FO": {"color": "#D55E00", "marker": "s"},
}
EPSILON_STYLE = {
    0.010: {"color": "#009E73", "marker": "o"},
    0.009: {"color": "#CC79A7", "marker": "s"},
    0.008: {"color": "#E69F00", "marker": "^"},
}
SOURCE_FILES = (
    "raw_repeats.csv",
    "runtime_summary.csv",
    "speedup_summary.csv",
    "method_runtime_comparison.csv",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    converted = series.astype(str).str.lower().map({"true": True, "false": False})
    if converted.isna().any():
        raise ValueError("Boolean runtime column contains an invalid value.")
    return converted


def load_verified_inputs(
    runtime_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    root = Path(runtime_root)
    completion = _load_json(root / "runtime_analysis_completion.json")
    audit = _load_json(root / "runtime_audit_report.json")
    if completion.get("status") != "complete":
        raise ValueError("Runtime analysis is not complete.")
    if audit.get("status") != "passed":
        raise ValueError("Runtime audit did not pass.")
    if completion.get("runtime_manifest_sha256") != audit.get(
        "runtime_manifest_sha256"
    ):
        raise ValueError("Runtime analysis and audit manifest hashes disagree.")
    for name in SOURCE_FILES:
        expected = completion.get("output_sha256", {}).get(name)
        if expected is None:
            raise ValueError(f"Runtime analysis did not record SHA256 for {name}.")
        if file_sha256(root / name) != expected:
            raise ValueError(f"Runtime figure input SHA256 mismatch: {name}.")

    repeats = pd.read_csv(root / "raw_repeats.csv")
    summaries = pd.read_csv(root / "runtime_summary.csv")
    speedups = pd.read_csv(root / "speedup_summary.csv")
    comparisons = pd.read_csv(root / "method_runtime_comparison.csv")
    speedups["is_strong_scaling_workload"] = _as_bool(
        speedups["is_strong_scaling_workload"]
    )

    expected_coverage = {
        (method, epsilon, worker)
        for method in METHOD_ORDER
        for epsilon in EPSILON_ORDER
        for worker in WORKER_ORDER
    }
    summary_coverage = {
        (str(row.method), float(row.epsilon), int(row.worker_count))
        for row in summaries.itertuples()
    }
    speedup_coverage = {
        (str(row.method), float(row.epsilon), int(row.worker_count))
        for row in speedups.itertuples()
    }
    comparison_coverage = {
        (float(row.epsilon), int(row.worker_count))
        for row in comparisons.itertuples()
    }
    if summary_coverage != expected_coverage or speedup_coverage != expected_coverage:
        raise ValueError("Runtime summary method/epsilon/worker coverage mismatch.")
    if comparison_coverage != {
        (epsilon, worker)
        for epsilon in EPSILON_ORDER
        for worker in WORKER_ORDER
    }:
        raise ValueError("Runtime comparison epsilon/worker coverage mismatch.")
    if (len(repeats), len(summaries), len(speedups), len(comparisons)) != (
        108,
        36,
        36,
        18,
    ):
        raise ValueError("Runtime figure input row counts do not match Step 8C.")
    numeric_columns = [
        "training_time_median",
        "training_time_min",
        "training_time_max",
        "end_to_end_time_median",
        "end_to_end_time_min",
        "end_to_end_time_max",
        "communication_fraction_median",
    ]
    if not np.isfinite(summaries[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Runtime summaries contain non-finite plot metrics.")
    for prefix in ("training_time", "end_to_end_time"):
        if not (
            (summaries[f"{prefix}_min"] <= summaries[f"{prefix}_median"])
            & (summaries[f"{prefix}_median"] <= summaries[f"{prefix}_max"])
        ).all():
            raise ValueError(f"Invalid min/median/max ordering for {prefix}.")
    strong_methods = set(
        speedups.loc[speedups["is_strong_scaling_workload"], "method"]
    )
    if strong_methods != {"NOG-FO"}:
        raise ValueError("Only NOG-FO may be labeled a fixed-work strong-scaling load.")
    if set(comparisons["comparison_scope"]) != {
        "full-frozen-budget-not-first-hit"
    }:
        raise ValueError("Runtime comparison scope is not the frozen full budget.")
    return repeats, summaries, speedups, comparisons, completion


def _save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    outputs = []
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {})):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _set_worker_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(WORKER_ORDER)
    axis.set_xticklabels([str(worker) for worker in WORKER_ORDER])
    axis.set_xlabel("CPU worker processes $m$")
    axis.grid(True, which="both", alpha=0.24, linewidth=0.7)


def _asymmetric_range(subset: pd.DataFrame, prefix: str) -> np.ndarray:
    median = subset[f"{prefix}_median"].to_numpy(dtype=float)
    return np.vstack(
        [
            median - subset[f"{prefix}_min"].to_numpy(dtype=float),
            subset[f"{prefix}_max"].to_numpy(dtype=float) - median,
        ]
    )


def plot_runtime_vs_workers(
    summaries: pd.DataFrame, output_stem: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.2), sharex=True)
    rows = (
        ("training_time", "Training time (s)"),
        ("end_to_end_time", "End-to-end time (s)"),
    )
    for column, epsilon in enumerate(EPSILON_ORDER):
        for row_index, (prefix, ylabel) in enumerate(rows):
            axis = axes[row_index, column]
            for method in METHOD_ORDER:
                subset = summaries[
                    (summaries["method"] == method)
                    & np.isclose(summaries["epsilon"], epsilon)
                ].sort_values("worker_count")
                style = METHOD_STYLE[method]
                axis.errorbar(
                    subset["worker_count"],
                    subset[f"{prefix}_median"],
                    yerr=_asymmetric_range(subset, prefix),
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=5.5,
                    capsize=3,
                    label=method,
                )
            axis.set_yscale("log")
            _set_worker_axis(axis)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 0:
                axis.set_title(rf"$\epsilon={epsilon:g}$")
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            marker=METHOD_STYLE[method]["marker"],
            lw=1.8,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Frozen full-budget runtime: median and [min, max] over 3 repeats",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return _save_figure(figure, output_stem)


def plot_nog_strong_scaling_speedup(
    speedups: pd.DataFrame, output_stem: Path
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(6.7, 4.7))
    subset = speedups[
        (speedups["method"] == "NOG-FO")
        & speedups["is_strong_scaling_workload"]
    ]
    for epsilon in EPSILON_ORDER:
        points = subset[np.isclose(subset["epsilon"], epsilon)].sort_values(
            "worker_count"
        )
        style = EPSILON_STYLE[epsilon]
        axis.plot(
            points["worker_count"],
            points["training_speedup_vs_m1"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=rf"$\epsilon={epsilon:g}$",
        )
    workers = np.asarray(WORKER_ORDER, dtype=float)
    axis.plot(workers, workers, color="#555555", linestyle="--", lw=1.2, label="Ideal")
    axis.axhline(1.0, color="#777777", linestyle=":", lw=1.0)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xticks(WORKER_ORDER)
    axis.set_xticklabels([str(worker) for worker in WORKER_ORDER])
    axis.set_yticks((0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32))
    axis.set_yticklabels(("0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32"))
    axis.set_xlabel("CPU worker processes $m$")
    axis.set_ylabel(r"Training speedup $T_1/T_m$")
    axis.set_title("NOG-FO fixed-total-work strong-scaling diagnostic")
    axis.grid(True, which="both", alpha=0.24, linewidth=0.7)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def plot_communication_fraction(
    summaries: pd.DataFrame, output_stem: Path
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 3.9), sharey=True)
    for axis, epsilon in zip(axes, EPSILON_ORDER):
        for method in METHOD_ORDER:
            subset = summaries[
                (summaries["method"] == method)
                & np.isclose(summaries["epsilon"], epsilon)
            ].sort_values("worker_count")
            style = METHOD_STYLE[method]
            axis.plot(
                subset["worker_count"],
                100.0 * subset["communication_fraction_median"],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5.5,
                label=method,
            )
        _set_worker_axis(axis)
        axis.set_title(rf"$\epsilon={epsilon:g}$")
    axes[0].set_ylabel("Communication / training time (%)")
    axes[-1].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Measured communication fraction within training time")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return _save_figure(figure, output_stem)


def plot_full_budget_comparison(
    comparisons: pd.DataFrame, output_stem: Path
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for epsilon in EPSILON_ORDER:
        subset = comparisons[np.isclose(comparisons["epsilon"], epsilon)].sort_values(
            "worker_count"
        )
        style = EPSILON_STYLE[epsilon]
        label = rf"$\epsilon={epsilon:g}$"
        axes[0].plot(
            subset["worker_count"],
            subset["training_time_ratio_nog_over_me"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=label,
        )
        axes[1].plot(
            subset["worker_count"],
            subset["work_ratio_nog_over_me"],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            label=label,
        )
    for axis in axes:
        _set_worker_axis(axis)
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("NOG / ME-DOL median training time")
    axes[0].set_title("Full-budget runtime ratio")
    axes[1].set_yscale("log", base=2)
    axes[1].set_ylabel("NOG / ME-DOL total SFO work")
    axes[1].set_title("Unmatched empirical work ratio")
    axes[1].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Frozen full-budget comparison (not first-hit time-to-epsilon)")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _save_figure(figure, output_stem)


def _figure_notes() -> str:
    return """# Step 8D Runtime Figure Notes

## `runtime_vs_workers`

Each point is the median of three measured real-CPU-process repeats; error bars span the observed `[min, max]` without outlier removal. Training time excludes process startup, evaluation, and serialization, while end-to-end time includes them. Every curve uses the complete pilot-frozen budget, so this is not first-hit time-to-epsilon. The ME-DOL configuration is identical across the three epsilon labels and was physically measured once per worker/repeat before deterministic expansion.

## `nog_strong_scaling_speedup`

Speedup is `median training time at m=1 / median training time at m`. Only NOG is plotted because its total SFO work is fixed across worker counts. Values below one mean that the single-machine CPU/Gloo process and communication overhead exceeds the available local parallelism benefit; the plot is a process-level scaling diagnostic, not evidence about a multi-node cluster.

## `communication_fraction_vs_workers`

Communication fraction is measured communication time divided by training time. It excludes startup, evaluation, and serialization and therefore must not be interpreted as a fraction of end-to-end time.

## `full_budget_method_comparison`

A runtime ratio below one means that NOG completed its frozen full budget faster in this implementation. The adjacent work-ratio panel is essential: the accuracy-selected configurations are not work matched, and NOG uses substantially more empirical SFO calls. Therefore the runtime ratio does not establish finite-work parity or directly validate the asymptotic Work complexity in Section 5. The epsilon=0.008 ME-DOL accuracy result remains right-censored, so neither panel is an unconditional time-to-epsilon comparison.
"""


def generate_figures(
    runtime_root: str | Path, output_root: str | Path
) -> Dict[str, Any]:
    runtime_path = Path(runtime_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    _, summaries, speedups, comparisons, completion = load_verified_inputs(
        runtime_path
    )

    generated: list[Path] = []
    generated.extend(
        plot_runtime_vs_workers(summaries, output_path / "runtime_vs_workers")
    )
    generated.extend(
        plot_nog_strong_scaling_speedup(
            speedups, output_path / "nog_strong_scaling_speedup"
        )
    )
    generated.extend(
        plot_communication_fraction(
            summaries, output_path / "communication_fraction_vs_workers"
        )
    )
    generated.extend(
        plot_full_budget_comparison(
            comparisons, output_path / "full_budget_method_comparison"
        )
    )
    notes_path = output_path / "runtime_figure_notes.md"
    notes_path.write_text(_figure_notes(), encoding="utf-8")

    manifest = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "runtime_manifest_sha256": completion["runtime_manifest_sha256"],
        "source_sha256": {
            name: completion["output_sha256"][name] for name in SOURCE_FILES
        },
        "figure_count": len(generated),
        "figures": {path.name: file_sha256(path) for path in sorted(generated)},
        "notes_sha256": file_sha256(notes_path),
        "plot_protocol": {
            "repeat_summary": "median with full observed [min, max] over 3 repeats",
            "outlier_removal": False,
            "runtime_scope": "complete pilot-frozen budget, not first-hit time-to-epsilon",
            "training_time_excludes": ["startup", "evaluation", "serialization"],
            "end_to_end_time_includes": ["startup", "evaluation", "serialization"],
            "strong_scaling_method": "NOG-FO only",
            "strong_scaling_speedup": "median(m=1) / median(m)",
            "worker_counts": list(WORKER_ORDER),
            "epsilons": list(EPSILON_ORDER),
        },
        "warnings": [
            "No positive CPU-process strong scaling was observed; do not generalize this single-machine Gloo result to a multi-node cluster.",
            "ME-DOL total work grows with worker count, so its m=1 timing ratio is not a strong-scaling efficiency measurement.",
            "Frozen configurations are not empirical-work matched; runtime ratios do not establish finite-work parity or Section 5 Work complexity.",
            "ME-DOL at epsilon=0.008 is accuracy-censored; full-budget runtime is not an unconditional time-to-epsilon result.",
        ],
    }
    atomic_write_json(output_path / "runtime_figure_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root", default="outputs/distributed_cpu_fo/runtime"
    )
    parser.add_argument(
        "--output-root", default="outputs/distributed_cpu_fo/runtime/figures"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_figures(args.runtime_root, args.output_root)
    print(
        f"phase=runtime-figures status={manifest['status']} "
        f"figures={manifest['figure_count']} output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
