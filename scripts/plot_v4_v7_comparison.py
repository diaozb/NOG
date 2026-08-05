"""Generate side-by-side v4--v7 experiment comparison figures.

The four protocols are intentionally plotted in separate rows because their
hyperparameter-selection rules, epsilon ranges, and formal seeds differ.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v4_v7_comparison" / "figures"


def _merge_methods(
    frame: pd.DataFrame,
    *,
    depth_column: str,
    work_column: str,
) -> pd.DataFrame:
    pieces = []
    for method, prefix in (("NOG-FO", "nog"), ("ME-DOL-FO", "me")):
        part = frame.loc[
            frame["method"] == method,
            ["epsilon", depth_column, work_column],
        ].copy()
        part = part.rename(
            columns={depth_column: f"{prefix}_depth", work_column: f"{prefix}_work"}
        )
        pieces.append(part)
    return pieces[0].merge(pieces[1], on="epsilon", validate="one_to_one")


def load_v4() -> pd.DataFrame:
    base = ROOT / "results" / "theory_validation_v4" / "analysis"
    summary = pd.read_csv(base / "formal_summary.csv")
    summary = summary.loc[summary["scope"] == "primary"]
    merged = _merge_methods(
        summary, depth_column="depth_mean", work_column="total_work_mean"
    )
    ratios = pd.read_csv(base / "formal_ratios.csv")
    return merged.merge(ratios, on="epsilon", validate="one_to_one").sort_values(
        "epsilon", ascending=False
    )


def load_v5() -> pd.DataFrame:
    base = ROOT / "results" / "low_epsilon_v5_symmetric" / "analysis"
    summary = pd.read_csv(base / "formal_summary.csv")
    summary = summary.loc[summary["scope"] == "primary"]
    merged = _merge_methods(
        summary, depth_column="depth_mean_hits", work_column="work_mean_hits"
    )
    ratios = pd.read_csv(base / "formal_ratios.csv")
    ratios = ratios.loc[ratios["scope"] == "primary"]
    return merged.merge(ratios, on="epsilon", validate="one_to_one").sort_values(
        "epsilon", ascending=False
    )


def load_v6(regime: str) -> pd.DataFrame:
    base = ROOT / "results" / "v4_v7_comparison" / "summaries" / "v6"
    summary = pd.read_csv(base / "formal_summary.csv")
    summary = summary.loc[summary["regime"] == regime]
    merged = _merge_methods(
        summary, depth_column="depth_mean_hits", work_column="work_mean_hits"
    )
    ratios = pd.read_csv(base / "formal_ratios.csv")
    ratios = ratios.loc[ratios["regime"] == regime]
    return merged.merge(ratios, on="epsilon", validate="one_to_one").sort_values(
        "epsilon", ascending=False
    )


def load_v7() -> pd.DataFrame:
    base = ROOT / "results" / "v4_v7_comparison" / "summaries" / "v7"
    summary = pd.read_csv(base / "formal_summary.csv")
    summary["effective_depth"] = summary["depth_mean_hits"].where(
        summary["hit_count"] == summary["num_seeds"], summary["capped_depth_mean"]
    )
    summary["effective_work"] = summary["work_mean_hits"].where(
        summary["hit_count"] == summary["num_seeds"], summary["capped_work_mean"]
    )
    censored = (
        summary.groupby("epsilon", as_index=False)
        .apply(lambda rows: bool((rows["hit_count"] < rows["num_seeds"]).any()), include_groups=False)
        .rename(columns={None: "censored"})
    )
    merged = _merge_methods(
        summary, depth_column="effective_depth", work_column="effective_work"
    )
    ratios = pd.read_csv(base / "formal_ratios.csv")
    return (
        merged.merge(ratios, on="epsilon", validate="one_to_one")
        .merge(censored, on="epsilon", validate="one_to_one")
        .sort_values("epsilon", ascending=False)
    )


def _format_axis(ax: plt.Axes, *, ratio: bool = False) -> None:
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(r"Target tolerance $\epsilon$ (smaller $\rightarrow$)")
    if ratio:
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle=":", alpha=0.8)


def _plot_absolute_row(
    axes: tuple[plt.Axes, plt.Axes],
    frames: list[tuple[str, pd.DataFrame, str]],
    title: str,
) -> None:
    depth_ax, work_ax = axes
    for label, frame, style in frames:
        depth_ax.plot(
            frame["epsilon"], frame["nog_depth"], style, color="#1f77b4", label=f"NOG {label}"
        )
        depth_ax.plot(
            frame["epsilon"], frame["me_depth"], style, color="#d62728", label=f"ME-DOL {label}"
        )
        work_ax.plot(
            frame["epsilon"], frame["nog_work"], style, color="#1f77b4", label=f"NOG {label}"
        )
        work_ax.plot(
            frame["epsilon"], frame["me_work"], style, color="#d62728", label=f"ME-DOL {label}"
        )
    for ax, ylabel in ((depth_ax, "Mean first-hit depth"), (work_ax, "Mean first-hit work")):
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {ylabel.removeprefix('Mean first-hit ')}")
        _format_axis(ax)
        ax.legend(fontsize=8, ncol=2)


def _plot_ratio_row(
    axes: tuple[plt.Axes, plt.Axes],
    frames: list[tuple[str, pd.DataFrame, str]],
    title: str,
) -> None:
    depth_ax, work_ax = axes
    for label, frame, style in frames:
        depth_ax.plot(
            frame["epsilon"],
            frame["depth_ratio_mean"],
            style,
            label=label or "primary",
        )
        work_ax.plot(
            frame["epsilon"],
            frame["work_ratio_mean"],
            style,
            label=label or "primary",
        )
    for ax, ylabel in (
        (depth_ax, "ME-DOL / NOG depth"),
        (work_ax, "NOG / ME-DOL work"),
    ):
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {ylabel}")
        _format_axis(ax, ratio=True)
        ax.legend(fontsize=8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    v4 = load_v4()
    v5 = load_v5()
    v6_work = load_v6("work_optimal")
    v6_depth = load_v6("depth_optimal")
    v7 = load_v7()

    rows = [
        ("v4 fixed algorithms + matched-work batch", [("", v4, "o-")]),
        ("v5 symmetric global configuration", [("", v5, "o-")]),
        (
            "v6 per-epsilon joint retuning",
            [("work-optimal", v6_work, "o-"), ("depth-optimal", v6_depth, "s--")],
        ),
        ("v7 piecewise theory scaling", [("", v7, "o-")]),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)
    for index, (title, frames) in enumerate(rows):
        _plot_absolute_row((axes[index, 0], axes[index, 1]), frames, title)
    fig.suptitle(
        "v4--v7 absolute first-hit depth and training-oracle work\n"
        "Rows use different frozen protocols and must not be joined into one curve",
        fontsize=16,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"v4_v7_absolute_metrics.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(15, 20), constrained_layout=True)
    for index, (title, frames) in enumerate(rows):
        _plot_ratio_row((axes[index, 0], axes[index, 1]), frames, title)
    censored = v7.loc[v7["censored"]]
    if not censored.empty:
        axes[3, 0].scatter(
            censored["epsilon"], censored["depth_ratio_mean"], marker="x", s=100, color="black", zorder=5
        )
        axes[3, 1].scatter(
            censored["epsilon"], censored["work_ratio_mean"], marker="x", s=100, color="black", zorder=5
        )
        axes[3, 1].text(
            0.98,
            0.96,
            "black x: capped ratio (censored)",
            transform=axes[3, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    fig.suptitle(
        "v4--v7 paired depth/work ratios\n"
        "Depth > 1 favors NOG; work near 1 indicates similar training-oracle work",
        fontsize=16,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"v4_v7_ratio_metrics.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
