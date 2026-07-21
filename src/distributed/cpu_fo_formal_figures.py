"""Generate Step 7D paper-candidate figures from audited formal results."""

from __future__ import annotations

import argparse
import json
import math
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
METHOD_STYLE = {
    "NOG-FO": {"color": "#0072B2", "marker": "o"},
    "ME-DOL-FO": {"color": "#D55E00", "marker": "s"},
}
SOURCE_FILES = (
    "formal_results.csv",
    "threshold_summary.csv",
    "threshold_per_seed.csv",
    "method_comparison.csv",
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
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_verified_inputs(
    formal_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    root = Path(formal_root)
    completion = _load_json(root / "formal_analysis_completion.json")
    audit = _load_json(root / "formal_audit_report.json")
    if completion.get("status") != "complete":
        raise ValueError("Formal analysis is not complete.")
    if audit.get("status") != "passed":
        raise ValueError("Formal audit did not pass.")
    for name in SOURCE_FILES:
        expected = completion.get("output_sha256", {}).get(name)
        if expected is None:
            raise ValueError(f"Formal analysis did not record SHA256 for {name}.")
        if file_sha256(root / name) != expected:
            raise ValueError(f"Formal figure input SHA256 mismatch: {name}.")

    trajectories = pd.read_csv(root / "formal_results.csv")
    summaries = pd.read_csv(root / "threshold_summary.csv")
    per_seed = pd.read_csv(root / "threshold_per_seed.csv")
    summaries["full_hit"] = _as_bool(summaries["full_hit"])
    per_seed["hit"] = _as_bool(per_seed["hit"])
    per_seed["censored"] = _as_bool(per_seed["censored"])

    expected_pairs = {
        (method, epsilon)
        for method in METHOD_ORDER
        for epsilon in (0.011, 0.010, 0.009, 0.008, 0.0075)
    }
    observed_pairs = {
        (str(row.method), float(row.epsilon))
        for row in summaries.itertuples()
    }
    if observed_pairs != expected_pairs:
        raise ValueError("Threshold summary method/epsilon coverage mismatch.")
    if len(summaries) != 10 or len(per_seed) != 50 or len(trajectories) != 1130:
        raise ValueError("Formal figure input row counts do not match Step 7C.")
    if not np.isfinite(
        trajectories[
            ["depth", "total_work", "stat_proxy", "training_time"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Formal trajectories contain non-finite plot metrics.")
    return trajectories, summaries, per_seed, completion


def _save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    outputs = []
    for suffix, options in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
    ):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_threshold_metric(
    summaries: pd.DataFrame,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    output_stem: Path,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    for method in METHOD_ORDER:
        style = METHOD_STYLE[method]
        subset = summaries[summaries["method"] == method].sort_values("epsilon")
        x = subset["epsilon"].to_numpy(dtype=float)
        y = subset[mean_column].to_numpy(dtype=float)
        axis.plot(x, y, color=style["color"], linewidth=1.8, label=method)
        for full_hit, marker_face in ((True, style["color"]), (False, "white")):
            points = subset[subset["full_hit"] == full_hit]
            if points.empty:
                continue
            axis.errorbar(
                points["epsilon"],
                points[mean_column],
                yerr=points[std_column].fillna(0.0),
                linestyle="none",
                marker=style["marker"],
                markersize=7,
                markerfacecolor=marker_face,
                markeredgecolor=style["color"],
                markeredgewidth=1.5,
                ecolor=style["color"],
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
        for row in subset[~subset["full_hit"]].itertuples():
            axis.annotate(
                f"{int(row.hit_count)}/{int(row.num_seeds)} hit",
                (float(row.epsilon), float(getattr(row, mean_column))),
                xytext=(4, 7),
                textcoords="offset points",
                fontsize=7.5,
                color=style["color"],
            )

    epsilons = sorted(summaries["epsilon"].unique())
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xticks(epsilons)
    axis.set_xticklabels([f"{value:g}" for value in epsilons])
    axis.set_xlabel(r"Stationarity tolerance $\epsilon$ (stricter $\rightarrow$)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25, linewidth=0.7)
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            marker=METHOD_STYLE[method]["marker"],
            linewidth=1.8,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label="Censored: successful seeds only",
        )
    )
    axis.legend(handles=handles, frameon=False, fontsize=8.5)
    figure.tight_layout()
    return _save_figure(figure, output_stem)


def _trajectory_statistics(
    trajectories: pd.DataFrame,
    config_id: str,
    x_column: str,
) -> pd.DataFrame:
    subset = trajectories[trajectories["formal_config_id"] == config_id]
    aggregate = (
        subset.groupby(x_column, as_index=False)["stat_proxy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(x_column)
    )
    if not (aggregate["count"] == 5).all():
        raise ValueError(f"Trajectory seed coverage mismatch for {config_id}.")
    aggregate["std"] = aggregate["std"].fillna(0.0)
    return aggregate


def plot_trajectory_panels(
    trajectories: pd.DataFrame,
    summaries: pd.DataFrame,
    x_column: str,
    xlabel: str,
    title: str,
    output_stem: Path,
) -> list[Path]:
    epsilons = sorted(summaries["epsilon"].unique(), reverse=True)
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.1), sharey=True)
    flat_axes = list(axes.flat)
    for axis, epsilon in zip(flat_axes, epsilons):
        epsilon_rows = summaries[summaries["epsilon"] == epsilon]
        for method in METHOD_ORDER:
            row = epsilon_rows[epsilon_rows["method"] == method].iloc[0]
            stats = _trajectory_statistics(
                trajectories, str(row["formal_config_id"]), x_column
            )
            style = METHOD_STYLE[method]
            x = stats[x_column].to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].to_numpy(dtype=float)
            axis.plot(
                x,
                mean,
                color=style["color"],
                linewidth=1.7,
                label=method,
            )
            axis.fill_between(
                x,
                np.maximum(mean - std, np.finfo(float).tiny),
                mean + std,
                color=style["color"],
                alpha=0.14,
                linewidth=0,
            )
        axis.axhline(
            float(epsilon), color="#333333", linestyle="--", linewidth=1.0
        )
        hit_labels = ", ".join(
            f"{method.split('-')[0]} {int(epsilon_rows[epsilon_rows['method'] == method]['hit_count'].iloc[0])}/5"
            for method in METHOD_ORDER
        )
        axis.set_title(rf"$\epsilon={epsilon:g}$  ({hit_labels})", fontsize=10)
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.22, linewidth=0.65)
        axis.set_xlabel(xlabel)
    flat_axes[-1].axis("off")
    for axis in (flat_axes[0], flat_axes[3]):
        axis.set_ylabel(r"Stationarity proxy $\|\widehat{\nabla} f_\delta\|$")
    handles = [
        Line2D([0], [0], color=METHOD_STYLE[method]["color"], lw=2, label=method)
        for method in METHOD_ORDER
    ]
    handles.append(
        Line2D([0], [0], color="#333333", lw=1, ls="--", label=r"Target $\epsilon$")
    )
    flat_axes[-1].legend(handles=handles, loc="center", frameon=False)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(figure, output_stem)


def generate_figures(
    formal_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    formal_path = Path(formal_root)
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    trajectories, summaries, _, completion = load_verified_inputs(formal_path)

    generated: list[Path] = []
    generated.extend(
        plot_threshold_metric(
            summaries,
            "first_hit_depth_mean",
            "first_hit_depth_std",
            "First confirmed-hit communication depth",
            "Communication depth vs. stationarity tolerance",
            output_path / "depth_vs_epsilon",
        )
    )
    generated.extend(
        plot_threshold_metric(
            summaries,
            "first_hit_total_work_mean",
            "first_hit_total_work_std",
            "First confirmed-hit total SFO work",
            "Training work vs. stationarity tolerance",
            output_path / "work_vs_epsilon",
        )
    )
    generated.extend(
        plot_trajectory_panels(
            trajectories,
            summaries,
            "depth",
            "Communication depth",
            "Formal trajectories by frozen epsilon-specific configuration",
            output_path / "stat_proxy_vs_depth",
        )
    )
    generated.extend(
        plot_trajectory_panels(
            trajectories,
            summaries,
            "total_work",
            "Total training SFO work",
            "Formal trajectories by frozen epsilon-specific configuration",
            output_path / "stat_proxy_vs_work",
        )
    )

    censored = [
        {
            "method": str(row.method),
            "epsilon": float(row.epsilon),
            "hit_count": int(row.hit_count),
            "num_seeds": int(row.num_seeds),
        }
        for row in summaries[~summaries["full_hit"]].itertuples()
    ]
    manifest = {
        "schema_version": FIGURE_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_manifest_sha256": completion["formal_manifest_sha256"],
        "frozen_config_sha256": completion["frozen_config_sha256"],
        "source_sha256": {
            name: completion["output_sha256"][name] for name in SOURCE_FILES
        },
        "figure_count": len(generated),
        "figures": {
            path.name: file_sha256(path) for path in sorted(generated)
        },
        "censored_method_epsilon_pairs": censored,
        "plot_protocol": {
            "threshold_points": "mean +/- sample std among successful formal seeds",
            "filled_marker": "all 5 formal seeds confirmed hit",
            "hollow_marker": "right-censored; point uses successful seeds only",
            "trajectory_line": "mean across 5 formal seeds",
            "trajectory_band": "+/- one sample std across 5 formal seeds",
            "confirmed_hit_consecutive": 2,
            "training_work_excludes_evaluation_work": True,
        },
        "warnings": [
            "Censored threshold points are conditional on successful seeds and must not be interpreted as unconditional method ratios.",
            "Each epsilon panel uses its own pilot-frozen configuration; curves are not a single universal hyperparameter setting.",
        ],
    }
    atomic_write_json(output_path / "figure_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        default="outputs/distributed_cpu_fo/formal_accuracy",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/distributed_cpu_fo/figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_figures(args.formal_root, args.output_root)
    print(
        f"phase=formal-figures status={manifest['status']} "
        f"figures={manifest['figure_count']} output_root={args.output_root}"
    )


if __name__ == "__main__":
    main()
