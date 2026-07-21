"""Create audited tables and paper-facing figures from formal results."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def final_summary(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        results.sort_values(["communication_round", "iteration"])
        .groupby(["method", "worker_count", "formal_seed"], as_index=False)
        .tail(1)
    )
    aggregate = per_seed.groupby(
        ["method", "base_method", "oracle_type", "worker_count"], as_index=False
    ).agg(
        num_seeds=("formal_seed", "nunique"),
        final_objective_mean=("objective", "mean"),
        final_objective_std=("objective", "std"),
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_std=("stat_proxy", "std"),
        final_total_work_mean=("total_work", "mean"),
        final_per_worker_work_mean=("per_worker_work_max", "mean"),
        final_depth_mean=("communication_round", "mean"),
        final_time_sec_mean=("time_sec", "mean"),
    )
    return per_seed, aggregate


def threshold_tables(
    results: pd.DataFrame,
    thresholds: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_columns = ["method", "base_method", "oracle_type", "worker_count", "formal_seed"]
    for keys, frame in results.groupby(group_columns):
        frame = frame.sort_values(["communication_round", "total_work"])
        final = frame.iloc[-1]
        identity = dict(zip(group_columns, keys))
        for threshold in thresholds:
            hit = frame[frame["stat_proxy"] <= threshold]
            if hit.empty:
                rows.append(
                    {
                        **identity,
                        "threshold": threshold,
                        "hit": False,
                        "first_hit_depth": np.nan,
                        "first_hit_total_work": np.nan,
                        "first_hit_per_worker_work": np.nan,
                        "first_hit_stat_proxy": np.nan,
                        "final_stat_proxy": final["stat_proxy"],
                    }
                )
            else:
                first = hit.iloc[0]
                rows.append(
                    {
                        **identity,
                        "threshold": threshold,
                        "hit": True,
                        "first_hit_depth": first["communication_round"],
                        "first_hit_total_work": first["total_work"],
                        "first_hit_per_worker_work": first["per_worker_work_max"],
                        "first_hit_stat_proxy": first["stat_proxy"],
                        "final_stat_proxy": final["stat_proxy"],
                    }
                )
    per_seed = pd.DataFrame(rows)
    aggregate = per_seed.groupby(
        ["method", "base_method", "oracle_type", "worker_count", "threshold"],
        as_index=False,
    ).agg(
        num_seeds=("formal_seed", "nunique"),
        hit_rate=("hit", "mean"),
        first_hit_depth_mean=("first_hit_depth", "mean"),
        first_hit_depth_std=("first_hit_depth", "std"),
        first_hit_total_work_mean=("first_hit_total_work", "mean"),
        first_hit_total_work_std=("first_hit_total_work", "std"),
        first_hit_per_worker_work_mean=("first_hit_per_worker_work", "mean"),
        final_stat_proxy_mean=("final_stat_proxy", "mean"),
        final_stat_proxy_std=("final_stat_proxy", "std"),
    )
    return per_seed, aggregate


def curve_summary(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(
        ["method", "oracle_type", "worker_count", "communication_round", "total_work"],
        as_index=False,
    ).agg(
        num_seeds=("formal_seed", "nunique"),
        stat_proxy_mean=("stat_proxy", "mean"),
        stat_proxy_std=("stat_proxy", "std"),
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
    )


def plot_track(
    curves: pd.DataFrame,
    oracle_type: str,
    worker_count: int,
    x_column: str,
    out_path: Path,
) -> None:
    frame = curves[
        (curves["oracle_type"] == oracle_type)
        & (curves["worker_count"] == worker_count)
    ]
    plt.figure(figsize=(6.4, 4.4))
    for method in frame["method"].unique():
        method_frame = frame[frame["method"] == method].sort_values(x_column)
        x = method_frame[x_column].to_numpy(dtype=float)
        mean = method_frame["stat_proxy_mean"].to_numpy(dtype=float)
        std = method_frame["stat_proxy_std"].fillna(0).to_numpy(dtype=float)
        plt.plot(x, mean, label=method, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.18)
    if x_column == "total_work":
        plt.xscale("log")
        xlabel = "Training oracle work (SFO calls)" if oracle_type == "sfo" else "Training oracle work (SZO calls)"
    else:
        xlabel = "Communication depth"
    plt.xlabel(xlabel)
    plt.ylabel("Smoothed gradient norm proxy")
    plt.title(f"{oracle_type.upper()} comparison, m={worker_count}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_nog_scaling(final_per_seed: pd.DataFrame, out_path: Path) -> None:
    frame = final_per_seed[final_per_seed["base_method"] == "NOG"]
    grouped = frame.groupby(["method", "worker_count"], as_index=False).agg(
        per_worker_work=("per_worker_work_max", "mean"),
        stat_proxy_mean=("stat_proxy", "mean"),
        stat_proxy_std=("stat_proxy", "std"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for method in grouped["method"].unique():
        method_frame = grouped[grouped["method"] == method].sort_values("worker_count")
        workers = method_frame["worker_count"].to_numpy(dtype=float)
        work = method_frame["per_worker_work"].to_numpy(dtype=float)
        axes[0].plot(workers, work / work[0], marker="o", label=method)
        axes[1].errorbar(
            workers,
            method_frame["stat_proxy_mean"],
            yerr=method_frame["stat_proxy_std"].fillna(0),
            marker="o",
            capsize=3,
            label=method,
        )
    workers = np.array(sorted(grouped["worker_count"].unique()), dtype=float)
    axes[0].plot(workers, 1.0 / workers, "k--", label="ideal 1/m")
    axes[0].set_xlabel("Worker count m")
    axes[0].set_ylabel("Normalized per-worker work")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("Worker count m")
    axes[1].set_ylabel("Final stationarity proxy")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=220)
    plt.close(figure)


def work_accounting_audit(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in results.groupby(["method", "worker_count", "formal_seed"]):
        frame = frame.sort_values(["communication_round", "total_work"])
        parsed = frame["per_worker_work"].map(
            lambda value: ast.literal_eval(value) if isinstance(value, str) else value
        )
        sum_matches = all(
            int(total) == int(sum(per_worker))
            for total, per_worker in zip(frame["total_work"], parsed)
        )
        rows.append(
            {
                "method": keys[0],
                "worker_count": keys[1],
                "formal_seed": keys[2],
                "total_work_monotone": bool(frame["total_work"].is_monotonic_increasing),
                "depth_monotone": bool(frame["communication_round"].is_monotonic_increasing),
                "eval_work_monotone": bool(frame["eval_work"].is_monotonic_increasing),
                "total_equals_sum_workers": bool(sum_matches),
                "finite_metrics": bool(
                    np.isfinite(frame[["objective", "stat_proxy"]].to_numpy()).all()
                ),
            }
        )
    audit = pd.DataFrame(rows)
    check_columns = [
        "total_work_monotone",
        "depth_monotone",
        "eval_work_monotone",
        "total_equals_sum_workers",
        "finite_metrics",
    ]
    audit["all_checks_pass"] = audit[check_columns].all(axis=1)
    return audit


def validate_formal_coverage(results: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    formal_seeds = set(int(value) for value in cfg["run"]["formal_seeds"])
    comparison_worker = int(cfg["distributed"]["comparison_worker"])
    configured_methods = [*cfg["methods"]["sfo"], *cfg["methods"]["szo"]]
    for method in configured_methods:
        observed = set(
            results[
                (results["method"] == method)
                & (results["worker_count"] == comparison_worker)
            ]["formal_seed"].astype(int)
        )
        if observed != formal_seeds:
            raise ValueError(
                f"Incomplete formal coverage for {method}, m={comparison_worker}: "
                f"expected {sorted(formal_seeds)}, got {sorted(observed)}."
            )


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary_markdown(
    final: pd.DataFrame,
    thresholds: pd.DataFrame,
    comparison_worker: int,
    path: Path,
) -> None:
    main_final = final[final["worker_count"] == comparison_worker].sort_values(
        ["oracle_type", "final_stat_proxy_mean"]
    )
    lines = [
        "# Distributed formal experiment summary",
        "",
        f"主对比使用 `{comparison_worker}` workers；所有数值为 5 formal seeds 的统计。",
        "",
        "## Final metrics",
        "",
        markdown_table(main_final),
        "",
        "## First-hit thresholds",
        "",
        markdown_table(
            thresholds[thresholds["worker_count"] == comparison_worker].sort_values(
                ["oracle_type", "threshold", "method"]
            )
        ),
        "",
        "注意：single-process simulation 的 wall-clock time 不能解释为真实 distributed speedup。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.results)
    cfg = load_yaml(args.config)
    validate_formal_coverage(results, cfg)
    thresholds = [float(value) for value in cfg["metrics"]["thresholds"]]
    comparison_worker = int(cfg["distributed"]["comparison_worker"])

    final_per_seed, final = final_summary(results)
    threshold_per_seed, threshold_summary = threshold_tables(results, thresholds)
    curves = curve_summary(results)
    audit = work_accounting_audit(results)

    final_per_seed.to_csv(out_dir / "final_per_seed.csv", index=False)
    final.to_csv(out_dir / "final_summary.csv", index=False)
    threshold_per_seed.to_csv(out_dir / "threshold_per_seed.csv", index=False)
    threshold_summary.to_csv(out_dir / "threshold_summary.csv", index=False)
    curves.to_csv(out_dir / "curve_summary.csv", index=False)
    audit.to_csv(out_dir / "work_accounting_audit.csv", index=False)

    plot_track(curves, "sfo", comparison_worker, "communication_round", out_dir / "sfo_stat_proxy_vs_depth.png")
    plot_track(curves, "sfo", comparison_worker, "total_work", out_dir / "sfo_stat_proxy_vs_total_work.png")
    plot_track(curves, "szo", comparison_worker, "communication_round", out_dir / "szo_stat_proxy_vs_depth.png")
    plot_track(curves, "szo", comparison_worker, "total_work", out_dir / "szo_stat_proxy_vs_total_work.png")
    plot_nog_scaling(final_per_seed, out_dir / "nog_worker_scaling.png")
    write_summary_markdown(final, threshold_summary, comparison_worker, out_dir / "summary.md")

    summary = {
        "num_rows": int(len(results)),
        "formal_seeds": sorted(int(value) for value in results["formal_seed"].unique()),
        "comparison_worker": comparison_worker,
        "audit_all_pass": bool(audit["all_checks_pass"].all()),
        "final_main_comparison": final[
            final["worker_count"] == comparison_worker
        ].to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"summary_written={out_dir}")
    print(f"audit_all_pass={summary['audit_all_pass']}")


if __name__ == "__main__":
    main()
