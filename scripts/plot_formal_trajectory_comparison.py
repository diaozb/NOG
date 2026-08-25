#!/usr/bin/env python3
"""Plot direct FO/ZO formal trajectory comparisons from completed results.

The figures use the fixed configurations selected before the formal runs.  Lines
show the across-seed median at each native evaluation checkpoint and bands show
the interquartile range.  No interpolation, smoothing, or experiment reruns are
performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FO_ROOT = (
    REPOSITORY_ROOT
    / "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/formal"
)
FO_INPUTS = {
    "NOG-FO": FO_ROOT / "NOG-FO__data-B-total-8/partials",
    "ME-DOL-FO": (
        FO_ROOT / "ME-DOL-FO__epoch-6__mult-100__rounds-3840/partials"
    ),
}
ZO_INPUT = (
    REPOSITORY_ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal/"
    "fixed_work_983040/results.csv"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results/formal_trajectory_comparison"

METHOD_STYLES = {
    "NOG-FO": {"color": "#0072B2", "linewidth": 2.5, "zorder": 5},
    "ME-DOL-FO": {"color": "#D55E00", "linewidth": 1.9, "zorder": 4},
    "NOG-ZO": {"color": "#0072B2", "linewidth": 2.5, "zorder": 6},
    "ME-DOL-ZO": {"color": "#D55E00", "linewidth": 1.9, "zorder": 5},
    "DGFM": {"color": "#009E73", "linewidth": 1.9, "zorder": 4},
    "DGFM+": {"color": "#CC79A7", "linewidth": 1.9, "zorder": 3},
}
FO_METHODS = ("NOG-FO", "ME-DOL-FO")
ZO_METHODS = ("NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+")
REQUIRED_COLUMNS = {"method", "formal_seed", "depth", "total_work", "stat_proxy"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fo() -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    source_files: list[Path] = []
    for method, directory in FO_INPUTS.items():
        paths = sorted(directory.glob("*.json"))
        if len(paths) != 20:
            raise ValueError(f"Expected 20 {method} partials in {directory}, found {len(paths)}")
        rows: list[dict[str, Any]] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("method") != method:
                raise ValueError(f"Method mismatch in {path}")
            rows.extend(payload["rows"])
        frame = pd.DataFrame(rows)
        if set(frame["method"].unique()) != {method}:
            raise ValueError(f"Unexpected method rows in {directory}")
        frames.append(frame)
        source_files.extend(paths)
    return pd.concat(frames, ignore_index=True), source_files


def _load_zo() -> tuple[pd.DataFrame, list[Path]]:
    frame = pd.read_csv(ZO_INPUT)
    observed = set(frame["method"].unique())
    if observed != set(ZO_METHODS):
        raise ValueError(f"Unexpected ZO methods: {sorted(observed)}")
    return frame, [ZO_INPUT]


def _validate(frame: pd.DataFrame, methods: Iterable[str], oracle: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{oracle} input is missing columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS - {"method"})].isna().any().any():
        raise ValueError(f"{oracle} input contains missing plot values")
    numeric = frame[["depth", "total_work", "stat_proxy"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError(f"{oracle} plot values must be finite and positive")

    for method in methods:
        subset = frame[frame["method"] == method]
        seeds = sorted(subset["formal_seed"].astype(int).unique())
        if seeds != list(range(20)):
            raise ValueError(f"{method} does not contain formal seeds 0--19")
        duplicates = subset.duplicated(["formal_seed", "depth"]).any()
        if duplicates:
            raise ValueError(f"{method} has duplicate depth checkpoints within a seed")


def _summarize(frame: pd.DataFrame, methods: Iterable[str]) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for method in methods:
        subset = frame[frame["method"] == method]
        for x_column in ("depth", "total_work"):
            summary = (
                subset.groupby(x_column)["stat_proxy"]
                .agg(
                    median="median",
                    q25=lambda values: values.quantile(0.25),
                    q75=lambda values: values.quantile(0.75),
                    seed_count="count",
                )
                .reset_index()
                .rename(columns={x_column: "x"})
                .sort_values("x")
            )
            if not (summary["seed_count"] == 20).all():
                counts = sorted(summary["seed_count"].unique())
                raise ValueError(
                    f"{method}/{x_column} checkpoint coverage is not 20 seeds: {counts}"
                )
            summary.insert(0, "method", method)
            summary.insert(1, "x_metric", x_column)
            summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot(
    summary: pd.DataFrame,
    methods: Iterable[str],
    oracle: str,
    output_stem: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), sharey=True)
    panel_specs = (
        ("depth", "Communication depth", "(a) Communication depth"),
        ("total_work", f"Total training {oracle} work", "(b) Training work"),
    )
    for axis, (x_metric, xlabel, title) in zip(axes, panel_specs):
        for method in methods:
            values = summary[
                (summary["method"] == method) & (summary["x_metric"] == x_metric)
            ]
            style = METHOD_STYLES[method]
            x = values["x"].to_numpy(dtype=float)
            median = values["median"].to_numpy(dtype=float)
            q25 = values["q25"].to_numpy(dtype=float)
            q75 = values["q75"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                q25,
                q75,
                color=style["color"],
                alpha=0.13,
                linewidth=0,
                zorder=style["zorder"] - 1,
            )
            axis.plot(
                x,
                median,
                color=style["color"],
                linewidth=style["linewidth"],
                label=method,
                zorder=style["zorder"],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(True, which="major", color="#B8B8B8", alpha=0.45, linewidth=0.7)
        axis.grid(True, which="minor", color="#D8D8D8", alpha=0.25, linewidth=0.5)
        axis.tick_params(which="both", direction="out")

    axes[0].set_ylabel("Stationarity proxy (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(tuple(methods)),
        frameon=False,
    )
    figure.text(
        0.5,
        0.018,
        "Median over 20 formal seeds; shaded bands show the interquartile range.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.78, wspace=0.09)

    outputs: list[Path] = []
    for suffix, options in ((".png", {"dpi": 300}), (".pdf", {})):
        path = output_stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    plt.close(figure)
    return outputs


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    fo, fo_sources = _load_fo()
    zo, zo_sources = _load_zo()
    _validate(fo, FO_METHODS, "FO")
    _validate(zo, ZO_METHODS, "ZO")
    fo_summary = _summarize(fo, FO_METHODS)
    zo_summary = _summarize(zo, ZO_METHODS)

    generated = []
    generated.extend(_plot(fo_summary, FO_METHODS, "SFO", output_dir / "fo_full_trajectory"))
    generated.extend(_plot(zo_summary, ZO_METHODS, "SZO", output_dir / "zo_full_trajectory"))

    fo_summary.insert(0, "oracle", "FO")
    zo_summary.insert(0, "oracle", "ZO")
    summary_path = output_dir / "trajectory_summary.csv"
    pd.concat([fo_summary, zo_summary], ignore_index=True).to_csv(summary_path, index=False)
    generated.append(summary_path)

    all_sources = fo_sources + zo_sources
    manifest = {
        "status": "complete",
        "protocol": {
            "formal_seeds": list(range(20)),
            "line": "median across formal seeds at native checkpoints",
            "band": "25th--75th percentiles across formal seeds",
            "interpolation": False,
            "smoothing": False,
            "training_work_excludes_evaluation_work": True,
            "fo_nog_configuration": "NOG-FO data_B_total=8",
            "fo_methods": list(FO_METHODS),
            "zo_methods": list(ZO_METHODS),
        },
        "sources": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha256(path)
            for path in sorted(all_sources)
        },
        "outputs": {
            path.name: _sha256(path) for path in sorted(generated)
        },
    }
    manifest_path = output_dir / "trajectory_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(generated) - 1} figures in {output_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
