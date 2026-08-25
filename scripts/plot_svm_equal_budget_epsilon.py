#!/usr/bin/env python3
"""Plot SVM epsilon curves at a shared grid of depth and work budgets."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / (
    "outputs/distributed_zo/zo_theory_validation/real_data/"
    "supplement_formal_cpu_v2_seed_shards"
)
DEFAULT_OUTPUT = ROOT / "results/advisor_cpu_completion/svm_equal_budget_epsilon"
METHODS = ("NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+")
DATASETS = ("a9a", "ijcnn1")
COLORS = {
    "NOG-ZO": "#0072B2",
    "ME-DOL-ZO": "#D55E00",
    "DGFM": "#009E73",
    "DGFM+": "#CC79A7",
}
LINEWIDTHS = {
    "NOG-ZO": 3.0,
    "ME-DOL-ZO": 2.2,
    "DGFM": 2.2,
    "DGFM+": 2.2,
}
BUDGET_GRIDS = {
    "communication_depth": [float(768 * index) for index in range(1, 6)],
    "training_work": [float(98304 * index) for index in range(1, 11)],
}
BUDGET_COLUMNS = {
    "communication_depth": "depth",
    "training_work": "total_work",
}
T_CRITICAL_95_DF19 = 2.093024054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def load_rows(source: Path) -> dict[tuple[str, str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    paths = sorted(source.glob("seed_*/partials/*.csv"))
    if len(paths) != 160:
        raise ValueError(f"Expected 160 formal task files, found {len(paths)}.")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty result file: {path}.")
        key = (
            rows[0]["dataset"],
            rows[0]["method"],
            int(rows[0]["formal_seed"]),
        )
        if any(
            (row["dataset"], row["method"], int(row["formal_seed"])) != key
            for row in rows
        ):
            raise ValueError(f"Mixed task identities in {path}.")
        if key in grouped:
            raise ValueError(f"Duplicate task identity: {key}.")
        grouped[key] = rows
    expected = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in range(20)
    }
    if set(grouped) != expected:
        raise ValueError("Formal task identities are incomplete.")
    return grouped


def nearest_native_row(
    rows: list[dict[str, str]],
    budget_column: str,
    target: float,
) -> dict[str, str]:
    return min(
        rows,
        key=lambda row: (
            abs(float(row[budget_column]) - target),
            float(row[budget_column]) > target,
            int(row["iteration"]),
        ),
    )


def summarize(
    grouped: dict[tuple[str, str, int], list[dict[str, str]]],
) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for dataset in DATASETS:
        initial_epsilon = np.asarray(
            [
                float(
                    min(
                        grouped[(dataset, "NOG-ZO", seed)],
                        key=lambda row: int(row["iteration"]),
                    )["stat_proxy"]
                )
                for seed in range(20)
            ],
            dtype=float,
        )
        initial_mean = float(initial_epsilon.mean())
        initial_std = float(initial_epsilon.std(ddof=1))
        initial_half_width = (
            T_CRITICAL_95_DF19 * initial_std / math.sqrt(len(initial_epsilon))
        )
        for view, targets in BUDGET_GRIDS.items():
            for method in METHODS:
                records.append(
                    {
                        "dataset": dataset,
                        "budget_view": view,
                        "target_index": 0,
                        "target_budget": 0.0,
                        "method": method,
                        "point_role": "common_initial_reference",
                        "epsilon_mean": initial_mean,
                        "epsilon_std": initial_std,
                        "epsilon_ci95_low": initial_mean - initial_half_width,
                        "epsilon_ci95_high": initial_mean + initial_half_width,
                        "actual_budget_min": 0.0,
                        "actual_budget_max": 0.0,
                        "seed_count": len(initial_epsilon),
                    }
                )
            budget_column = BUDGET_COLUMNS[view]
            for target_index, target in enumerate(targets, start=1):
                for method in METHODS:
                    selected = [
                        nearest_native_row(
                            grouped[(dataset, method, seed)],
                            budget_column,
                            target,
                        )
                        for seed in range(20)
                    ]
                    epsilon = np.asarray(
                        [float(row["stat_proxy"]) for row in selected],
                        dtype=float,
                    )
                    actual = np.asarray(
                        [float(row[budget_column]) for row in selected],
                        dtype=float,
                    )
                    mean = float(epsilon.mean())
                    standard_deviation = float(epsilon.std(ddof=1))
                    half_width = (
                        T_CRITICAL_95_DF19
                        * standard_deviation
                        / math.sqrt(len(epsilon))
                    )
                    records.append(
                        {
                            "dataset": dataset,
                            "budget_view": view,
                            "target_index": target_index,
                            "target_budget": target,
                            "method": method,
                            "point_role": "matched_budget_checkpoint",
                            "epsilon_mean": mean,
                            "epsilon_std": standard_deviation,
                            "epsilon_ci95_low": mean - half_width,
                            "epsilon_ci95_high": mean + half_width,
                            "actual_budget_min": float(actual.min()),
                            "actual_budget_max": float(actual.max()),
                            "seed_count": len(epsilon),
                        }
                    )
    return records


def save_data(records: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot(records: list[dict[str, float | int | str]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharey=True)
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    maximum_depth_mismatch = 0.0
    maximum_work_mismatch = 0.0

    for row_index, dataset in enumerate(DATASETS):
        for column_index, view in enumerate(BUDGET_GRIDS):
            axis = axes[row_index, column_index]
            for method in METHODS:
                values = sorted(
                    (
                        row
                        for row in records
                        if row["dataset"] == dataset
                        and row["budget_view"] == view
                        and row["method"] == method
                    ),
                    key=lambda row: int(row["target_index"]),
                )
                x = np.asarray([float(row["target_budget"]) for row in values])
                mean = np.asarray([float(row["epsilon_mean"]) for row in values])
                low = np.asarray([float(row["epsilon_ci95_low"]) for row in values])
                high = np.asarray([float(row["epsilon_ci95_high"]) for row in values])
                actual_min = np.asarray(
                    [float(row["actual_budget_min"]) for row in values]
                )
                actual_max = np.asarray(
                    [float(row["actual_budget_max"]) for row in values]
                )
                mismatch = float(
                    np.maximum(np.abs(actual_min - x), np.abs(actual_max - x)).max()
                )
                if view == "communication_depth":
                    maximum_depth_mismatch = max(maximum_depth_mismatch, mismatch)
                else:
                    maximum_work_mismatch = max(maximum_work_mismatch, mismatch)

                axis.fill_between(
                    x,
                    low,
                    high,
                    color=COLORS[method],
                    alpha=0.12,
                    linewidth=0,
                    zorder=1,
                )
                axis.plot(
                    x,
                    mean,
                    color=COLORS[method],
                    linewidth=LINEWIDTHS[method],
                    marker="o",
                    markersize=5.0,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    label=method,
                    zorder=3 if method == "NOG-ZO" else 2,
                )

            initial_reference = next(
                row
                for row in records
                if row["dataset"] == dataset
                and row["budget_view"] == view
                and row["method"] == "NOG-ZO"
                and row["point_role"] == "common_initial_reference"
            )
            initial_mean = float(initial_reference["epsilon_mean"])
            axis.scatter(
                [0.0],
                [initial_mean],
                marker="*",
                s=95,
                color="#222222",
                edgecolor="white",
                linewidth=0.7,
                zorder=6,
            )
            axis.annotate(
                "same initialization",
                (0.0, initial_mean),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8.5,
                color="#333333",
            )
            axis.set_ylim(0.0, 0.405)
            axis.grid(True, color="#B8B8B8", alpha=0.45)
            axis.spines[["top", "right"]].set_visible(False)
            if view == "communication_depth":
                axis.set_xlim(-120, 4100)
                axis.set_xlabel("Common communication-depth target")
                subtitle = "same depth checkpoints"
            else:
                axis.set_xlim(-30000, 1050000)
                axis.xaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value / 1000:.0f}k")
                )
                axis.set_xlabel("Common training-work target")
                subtitle = "same work checkpoints"
            axis.set_title(
                f"{panel_labels[row_index][column_index]} {dataset}: {subtitle}"
            )
            if column_index == 0:
                axis.set_ylabel("Achieved epsilon (lower is better)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Capped-l1 SVM: epsilon curves at matched budget checkpoints",
        fontsize=14,
        y=0.992,
    )
    figure.text(
        0.5,
        0.012,
        "All curves share the black-star initialization reference (estimated from "
        "NOG's first near-zero checkpoint, where objective is approximately 1). "
        "Lines: 20-seed means; bands: 95% CI. "
        f"Nearest native checkpoints; max mismatch is {maximum_depth_mismatch:.0f} depth "
        f"and {maximum_work_mismatch:.0f} work; no interpolation.",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.10,
        top=0.88,
        hspace=0.31,
        wspace=0.11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    grouped = load_rows(source)
    records = summarize(grouped)
    save_data(records, output.with_name(output.name + "_data").with_suffix(".csv"))
    plot(records, output)
    print(
        f"records={len(records)} shared_depth_points="
        f"{len(BUDGET_GRIDS['communication_depth'])} shared_work_points="
        f"{len(BUDGET_GRIDS['training_work'])}"
    )
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
