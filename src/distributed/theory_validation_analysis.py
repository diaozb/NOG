"""Analyze and plot the independent wide-epsilon theory-validation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now
from src.distributed.epsilon_scaling import trend_statistics


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _mean_ci(values: Iterable[float], seed: int, repetitions: int = 2000) -> tuple[float, float]:
    observed = [float(value) for value in values]
    if not observed:
        return math.nan, math.nan
    rng = random.Random(int(seed))
    estimates = sorted(
        statistics.mean(rng.choice(observed) for _ in observed)
        for _ in range(repetitions)
    )
    return estimates[int(0.025 * (repetitions - 1))], estimates[int(0.975 * (repetitions - 1))]


def _fit_exponent(epsilons: List[float], values: List[float]) -> float:
    return trend_statistics(epsilons, values)["log_log_slope"]


def _plot_ratios(rows: List[Dict[str, Any]], path: Path) -> None:
    eps = [float(row["epsilon"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    panels = [
        (
            "depth_ratio_mean",
            "depth_ratio_ci_low",
            "depth_ratio_ci_high",
            "ME-DOL / NOG depth",
            1.0,
        ),
        (
            "work_ratio_mean",
            "work_ratio_ci_low",
            "work_ratio_ci_high",
            "NOG / ME-DOL total work",
            1.0,
        ),
    ]
    for axis, (field, low, high, label, reference) in zip(axes, panels):
        values = [float(row[field]) for row in rows]
        lower = [float(row[low]) for row in rows]
        upper = [float(row[high]) for row in rows]
        axis.plot(eps, values, marker="o", markersize=3.5, linewidth=1.6)
        axis.fill_between(eps, lower, upper, alpha=0.2)
        axis.axhline(reference, color="black", linestyle="--", linewidth=0.9)
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_complexities(summary: List[Dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for method, color in [("NOG-FO", "#1f77b4"), ("ME-DOL-FO", "#d62728")]:
        rows = [row for row in summary if row["method"] == method]
        eps = [float(row["epsilon"]) for row in rows]
        axes[0].plot(eps, [float(row["depth_mean"]) for row in rows], marker="o", markersize=3, label=method, color=color)
        axes[1].plot(eps, [float(row["total_work_mean"]) for row in rows], marker="o", markersize=3, label=method, color=color)
    for axis, ylabel in zip(axes, ["first-hit depth", "first-hit total oracle work"]):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.invert_xaxis()
        axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(
    cfg: Dict[str, Any], freeze_path: Path, formal_root: Path, analysis_root: Path
) -> Dict[str, Any]:
    freeze = _load(freeze_path)
    completion = _load(formal_root / "completion.json")
    result_audit = _load(formal_root / "formal_result_audit.json")
    if result_audit.get("status") != "passed":
        raise ValueError("Formal result audit has not passed.")
    if freeze.get("status") != "frozen":
        raise ValueError("Parameters are not frozen.")
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError("Formal experiment is not complete and failure-free.")
    seeds = [int(value) for value in freeze["formal_seeds"]]
    schedule = {
        float(row["epsilon"]): int(row["data_B_total"])
        for row in freeze["selected_schedule"]
    }
    primary_eps = [float(value) for value in freeze["primary_epsilons"]]
    all_eps = [float(value) for value in cfg["epsilon_scaling"]["epsilons"]]
    consecutive = int(freeze["confirmed_hit_consecutive"])

    payloads: Dict[tuple[str, int], Dict[str, Any]] = {}
    input_paths = [freeze_path, formal_root / "completion.json"] + [formal_root / "formal_result_audit.json"]
    for record in completion["records"]:
        path = Path(record["partial_path"])
        input_paths.append(path)
        payloads[(str(record["label"]), int(record["task"]["formal_seed"]))] = _load(path)
    me_label = "ME-DOL-FO__epoch-6__mult-100__rounds-3840"
    largest_batch = max(int(value) for value in freeze["selected_batches"])

    per_seed: List[Dict[str, Any]] = []
    for epsilon in all_eps:
        primary = epsilon in schedule
        batch = schedule.get(epsilon, largest_batch)
        nog_label = f"NOG-FO__data-B-total-{batch}"
        for method, label in [("NOG-FO", nog_label), ("ME-DOL-FO", me_label)]:
            for seed in seeds:
                payload = payloads[(label, seed)]
                hit = confirmed_hit(payload["rows"], epsilon, consecutive)
                final = payload["rows"][-1]
                per_seed.append(
                    {
                        "scope": "primary" if primary else "exploratory_censored",
                        "method": method,
                        "epsilon": epsilon,
                        "formal_seed": seed,
                        "data_B_total": batch if method == "NOG-FO" else 8,
                        "hit": hit is not None,
                        "first_hit_depth": hit["depth"] if hit else None,
                        "first_hit_total_work": hit["total_work"] if hit else None,
                        "censoring_depth": int(final["depth"]),
                        "censoring_total_work": int(final["total_work"]),
                        "final_stat_proxy": float(final["stat_proxy"]),
                    }
                )

    lookup = {
        (str(row["method"]), float(row["epsilon"]), int(row["formal_seed"])): row
        for row in per_seed
    }
    summary: List[Dict[str, Any]] = []
    ratios: List[Dict[str, Any]] = []
    for epsilon in all_eps:
        for method in ["NOG-FO", "ME-DOL-FO"]:
            rows = [lookup[(method, epsilon, seed)] for seed in seeds]
            hits = [row for row in rows if row["hit"]]
            depths = [float(row["first_hit_depth"]) for row in hits]
            works = [float(row["first_hit_total_work"]) for row in hits]
            summary.append(
                {
                    "scope": rows[0]["scope"],
                    "method": method,
                    "epsilon": epsilon,
                    "num_seeds": len(rows),
                    "hit_count": len(hits),
                    "hit_rate": len(hits) / len(rows),
                    "depth_mean": statistics.mean(depths) if depths else None,
                    "depth_sd": statistics.stdev(depths) if len(depths) > 1 else (0.0 if depths else None),
                    "total_work_mean": statistics.mean(works) if works else None,
                    "total_work_sd": statistics.stdev(works) if len(works) > 1 else (0.0 if works else None),
                    "capped_depth_mean": statistics.mean(
                        float(row["first_hit_depth"] if row["hit"] else row["censoring_depth"])
                        for row in rows
                    ),
                    "capped_total_work_mean": statistics.mean(
                        float(row["first_hit_total_work"] if row["hit"] else row["censoring_total_work"])
                        for row in rows
                    ),
                    "mean_final_stat_proxy": statistics.mean(float(row["final_stat_proxy"]) for row in rows),
                }
            )
        if epsilon not in schedule:
            continue
        depth_values = []
        work_values = []
        paired_hits = 0
        for seed in seeds:
            nog = lookup[("NOG-FO", epsilon, seed)]
            me = lookup[("ME-DOL-FO", epsilon, seed)]
            if nog["hit"] and me["hit"]:
                paired_hits += 1
            nog_depth = float(nog["first_hit_depth"] if nog["hit"] else nog["censoring_depth"])
            me_depth = float(me["first_hit_depth"] if me["hit"] else me["censoring_depth"])
            nog_work = float(nog["first_hit_total_work"] if nog["hit"] else nog["censoring_total_work"])
            me_work = float(me["first_hit_total_work"] if me["hit"] else me["censoring_total_work"])
            depth_values.append(me_depth / nog_depth)
            work_values.append(nog_work / me_work)
        depth_ci = _mean_ci(depth_values, 10000 + int(round(epsilon * 1_000_000)))
        work_ci = _mean_ci(work_values, 20000 + int(round(epsilon * 1_000_000)))
        ratios.append(
            {
                "epsilon": epsilon,
                "data_B_total": schedule[epsilon],
                "paired_hit_count": paired_hits,
                "ratios_use_capped_values": paired_hits != len(seeds),
                "depth_ratio_mean": statistics.mean(depth_values),
                "depth_ratio_sd": statistics.stdev(depth_values),
                "depth_ratio_ci_low": depth_ci[0],
                "depth_ratio_ci_high": depth_ci[1],
                "work_ratio_mean": statistics.mean(work_values),
                "work_ratio_sd": statistics.stdev(work_values),
                "work_ratio_ci_low": work_ci[0],
                "work_ratio_ci_high": work_ci[1],
            }
        )

    primary_summary = [row for row in summary if row["scope"] == "primary"]
    nog_summary = [row for row in primary_summary if row["method"] == "NOG-FO"]
    me_summary = [row for row in primary_summary if row["method"] == "ME-DOL-FO"]
    depth_ratios = [float(row["depth_ratio_mean"]) for row in ratios]
    work_ratios = [float(row["work_ratio_mean"]) for row in ratios]
    trends = {
        "primary_epsilon_count": len(primary_eps),
        "formal_seed_count": len(seeds),
        "depth_ratio": trend_statistics(primary_eps, depth_ratios),
        "work_ratio": trend_statistics(primary_eps, work_ratios),
        "work_ratio_mean": statistics.mean(work_ratios),
        "work_ratio_coefficient_of_variation": statistics.stdev(work_ratios) / statistics.mean(work_ratios),
        "work_ratio_min": min(work_ratios),
        "work_ratio_max": max(work_ratios),
        "observed_exponents": {
            "NOG_depth": _fit_exponent(primary_eps, [float(row["depth_mean"]) for row in nog_summary]),
            "ME_DOL_depth": _fit_exponent(primary_eps, [float(row["depth_mean"]) for row in me_summary]),
            "NOG_work": _fit_exponent(primary_eps, [float(row["total_work_mean"]) for row in nog_summary]),
            "ME_DOL_work": _fit_exponent(primary_eps, [float(row["total_work_mean"]) for row in me_summary]),
        },
        "theory_reference_exponents": {
            "NOG_depth": 5.0 / 3.0,
            "ME_DOL_depth": 3.0,
            "depth_ratio": 4.0 / 3.0,
            "NOG_work": 3.0,
            "ME_DOL_work": 3.0,
            "work_ratio": 0.0,
        },
        "all_primary_pairs_hit": all(int(row["paired_hit_count"]) == len(seeds) for row in ratios),
    }
    criteria = freeze["formal_success_criteria"]
    verdict = {
        "depth_advantage_grows": (
            trends["depth_ratio"]["spearman_rho"] >= float(criteria["minimum_depth_ratio_spearman"])
            and depth_ratios[-1] > depth_ratios[0]
        ),
        "work_is_matched": (
            min(work_ratios) >= float(criteria["work_ratio_lower"])
            and max(work_ratios) <= float(criteria["work_ratio_upper"])
            and trends["work_ratio_coefficient_of_variation"] <= float(criteria["maximum_work_ratio_cv"])
        ),
        "hit_rate_passes": all(
            int(row["hit_count"]) >= int(criteria["minimum_primary_hits_of_20"])
            for row in primary_summary
        ),
    }
    verdict["primary_claim_supported"] = all(verdict.values())
    trends["verdict"] = verdict

    analysis_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(analysis_root / "formal_per_seed.csv", per_seed)
    atomic_write_csv(analysis_root / "formal_summary.csv", summary)
    atomic_write_csv(analysis_root / "formal_ratios.csv", ratios)
    atomic_write_json(analysis_root / "formal_trends.json", trends)
    _plot_ratios(ratios, analysis_root / "figures" / "depth_work_ratios.png")
    _plot_complexities(primary_summary, analysis_root / "figures" / "depth_work_vs_epsilon.png")

    report = [
        "# Wide-epsilon theory-validation result",
        "",
        f"- Primary grid: {len(primary_eps)} epsilon values from {max(primary_eps):g} to {min(primary_eps):g}.",
        f"- Independent formal repetitions: {len(seeds)} seeds per frozen configuration.",
        f"- Depth ratio ME-DOL/NOG: {depth_ratios[0]:.2f}x at epsilon={primary_eps[0]:g} to {depth_ratios[-1]:.2f}x at epsilon={primary_eps[-1]:g}; Spearman rho={trends['depth_ratio']['spearman_rho']:.3f}.",
        f"- Matched-work ratio NOG/ME-DOL: mean={trends['work_ratio_mean']:.2f}x, range={trends['work_ratio_min']:.2f}--{trends['work_ratio_max']:.2f}x, CV={trends['work_ratio_coefficient_of_variation']:.3f}.",
        f"- Primary claim supported by frozen criteria: **{verdict['primary_claim_supported']}**.",
        "",
        "The points below epsilon=0.01 are reported as exploratory censored results; non-hits are retained at their maximum tested depth/work and are never silently dropped.",
        "",
        "![Depth and work ratios](figures/depth_work_ratios.png)",
        "",
        "![Depth and work versus epsilon](figures/depth_work_vs_epsilon.png)",
    ]
    (analysis_root / "theory_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_completion_status": completion["status"],
        "per_seed_rows": len(per_seed),
        "summary_rows": len(summary),
        "ratio_rows": len(ratios),
        "verdict": verdict,
        "input_artifacts": [
            {"path": str(path), "sha256": file_sha256(path)} for path in input_paths
        ],
    }
    atomic_write_json(analysis_root / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_theory_validation_v4.yaml")
    parser.add_argument(
        "--freeze",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/frozen_parameters.json",
    )
    parser.add_argument(
        "--formal-root",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/formal",
    )
    parser.add_argument(
        "--analysis-root",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/analysis",
    )
    args = parser.parse_args()
    result = analyze(
        load_config(args.config),
        Path(args.freeze),
        Path(args.formal_root),
        Path(args.analysis_root),
    )
    print(f"status={result['status']} verdict={result['verdict']}")


if __name__ == "__main__":
    main()
