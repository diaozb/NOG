"""Create censoring-aware statistics for the wide-epsilon formal experiment."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import atomic_write_json, object_sha256, utc_now
from src.distributed.epsilon_scaling import (
    kaplan_meier_restricted_mean,
    paired_ratio_summary,
    summarize_censored,
    trend_statistics,
)


STATISTICS_SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _bootstrap_ci(
    values: Sequence[Any],
    statistic: Callable[[List[Any]], float],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("Bootstrap values cannot be empty.")
    rng = random.Random(int(seed))
    sample = list(values)
    estimates = sorted(
        float(statistic([rng.choice(sample) for _ in sample]))
        for _ in range(int(repetitions))
    )
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return estimates[low_index], estimates[high_index]


def _seed_for(base: int, *parts: Any) -> int:
    digest = object_sha256([base, *parts])
    return int(digest[:12], 16)


def _payload_lookup(root: Path) -> Dict[tuple[str, str, int], Dict[str, Any]]:
    completion = _load(root / "formal_completion.json")
    if completion.get("status") != "complete" or completion.get("completed_tasks") != 120:
        raise ValueError("Formal completion is not 120/120.")
    result = {}
    for record in completion["records"]:
        payload = _load(root / record["partial_path"])
        result[(record["method"], record["region"], int(record["formal_seed"]))] = payload
    if len(result) != 120:
        raise ValueError("Formal payload lookup is incomplete.")
    return result


def _metric_summary(
    values: List[float | None],
    limits: List[float],
    repetitions: int,
    seed: int,
) -> Dict[str, Any]:
    summary = summarize_censored(values, limits)
    capped = [float(value) if value is not None else float(limit) for value, limit in zip(values, limits)]
    events = [value is not None for value in values]
    pairs = list(zip(capped, events))
    capped_ci = _bootstrap_ci(capped, statistics.mean, repetitions, seed)
    restricted_ci = _bootstrap_ci(
        pairs,
        lambda rows: kaplan_meier_restricted_mean(
            [float(row[0]) for row in rows],
            [bool(row[1]) for row in rows],
            max(float(value) for value in limits),
        ),
        repetitions,
        seed + 1,
    )
    hits = [float(value) for value in values if value is not None]
    conditional_ci = (
        _bootstrap_ci(hits, statistics.mean, repetitions, seed + 2) if hits else (None, None)
    )
    return {
        **summary,
        "conditional_mean_ci_low": conditional_ci[0],
        "conditional_mean_ci_high": conditional_ci[1],
        "capped_mean_ci_low": capped_ci[0],
        "capped_mean_ci_high": capped_ci[1],
        "restricted_mean_ci_low": restricted_ci[0],
        "restricted_mean_ci_high": restricted_ci[1],
        "mean_lower_bound": summary["capped_mean"],
        "mean_upper_bound": summary["conditional_mean"] if len(hits) == len(values) else None,
        "upper_bound_is_unbounded": len(hits) != len(values),
    }


def analyze_formal(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    audit = _load(root / "formal_result_audit.json")
    if audit.get("status") != "passed" or audit.get("passed_tasks") != 120:
        raise ValueError("Formal result audit must pass before statistics.")
    payloads = _payload_lookup(root)
    epsilons = [float(value) for value in cfg["epsilon_scaling"]["epsilons"]]
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])
    repetitions = int(cfg["epsilon_scaling"]["statistics"]["bootstrap_repetitions"])
    bootstrap_seed = int(cfg["epsilon_scaling"]["statistics"]["bootstrap_seed"])

    expanded = []
    for method in cfg["methods"]["sfo"]:
        for epsilon in epsilons:
            from src.distributed.epsilon_scaling import epsilon_region

            region = epsilon_region(epsilon)
            for seed in seeds:
                payload = payloads[(method, region, seed)]
                hit = confirmed_hit(payload["rows"], epsilon, consecutive)
                final = payload["rows"][-1]
                expanded.append(
                    {
                        "method": method,
                        "epsilon": epsilon,
                        "region": region,
                        "formal_seed": seed,
                        "hit": hit is not None,
                        "first_hit_iteration": hit["iteration"] if hit else None,
                        "first_hit_depth": hit["depth"] if hit else None,
                        "first_hit_total_work": hit["total_work"] if hit else None,
                        "first_hit_per_worker_work": hit["per_worker_work"] if hit else None,
                        "first_hit_training_time": hit["training_time"] if hit else None,
                        "censoring_depth": int(final["depth"]),
                        "censoring_total_work": int(final["total_work"]),
                        "censoring_per_worker_work": int(final["per_worker_work_max"]),
                        "censoring_training_time": float(final["training_time"]),
                        "final_stat_proxy": float(final["stat_proxy"]),
                    }
                )
    if len(expanded) != 680:
        raise ValueError("Expected 680 expanded formal rows.")

    lookup = {
        (row["method"], float(row["epsilon"]), int(row["formal_seed"])): row
        for row in expanded
    }
    summary_rows = []
    metric_fields = {
        "depth": ("first_hit_depth", "censoring_depth"),
        "total_work": ("first_hit_total_work", "censoring_total_work"),
        "per_worker_work": ("first_hit_per_worker_work", "censoring_per_worker_work"),
        "training_time": ("first_hit_training_time", "censoring_training_time"),
    }
    for method in cfg["methods"]["sfo"]:
        for epsilon in epsilons:
            rows = [lookup[(method, epsilon, seed)] for seed in seeds]
            hit_values = [1.0 if row["hit"] else 0.0 for row in rows]
            hit_ci = _bootstrap_ci(
                hit_values,
                statistics.mean,
                repetitions,
                _seed_for(bootstrap_seed, method, epsilon, "hit"),
            )
            result: Dict[str, Any] = {
                "method": method,
                "epsilon": epsilon,
                "region": rows[0]["region"],
                "num_seeds": len(rows),
                "hit_count": sum(bool(row["hit"]) for row in rows),
                "hit_rate": statistics.mean(hit_values),
                "hit_rate_ci_low": hit_ci[0],
                "hit_rate_ci_high": hit_ci[1],
            }
            for metric, (value_field, limit_field) in metric_fields.items():
                values = [float(row[value_field]) if row[value_field] is not None else None for row in rows]
                limits = [float(row[limit_field]) for row in rows]
                metric_summary = _metric_summary(
                    values,
                    limits,
                    repetitions,
                    _seed_for(bootstrap_seed, method, epsilon, metric),
                )
                result.update({f"{metric}_{key}": value for key, value in metric_summary.items()})
            summary_rows.append(result)

    ratio_rows = []
    for epsilon in epsilons:
        nog = [lookup[("NOG-FO", epsilon, seed)] for seed in seeds]
        me = [lookup[("ME-DOL-FO", epsilon, seed)] for seed in seeds]
        depth = paired_ratio_summary(
            [row["first_hit_depth"] for row in me],
            [row["first_hit_depth"] for row in nog],
            [row["censoring_depth"] for row in me],
            [row["censoring_depth"] for row in nog],
        )
        work = paired_ratio_summary(
            [row["first_hit_total_work"] for row in nog],
            [row["first_hit_total_work"] for row in me],
            [row["censoring_total_work"] for row in nog],
            [row["censoring_total_work"] for row in me],
        )
        depth_capped = [
            (float(left["first_hit_depth"]) if left["first_hit_depth"] is not None else float(left["censoring_depth"]))
            / (float(right["first_hit_depth"]) if right["first_hit_depth"] is not None else float(right["censoring_depth"]))
            for left, right in zip(me, nog)
        ]
        work_capped = [
            (float(left["first_hit_total_work"]) if left["first_hit_total_work"] is not None else float(left["censoring_total_work"]))
            / (float(right["first_hit_total_work"]) if right["first_hit_total_work"] is not None else float(right["censoring_total_work"]))
            for left, right in zip(nog, me)
        ]
        depth_ci = _bootstrap_ci(depth_capped, statistics.mean, repetitions, _seed_for(bootstrap_seed, epsilon, "depth-ratio"))
        work_ci = _bootstrap_ci(work_capped, statistics.mean, repetitions, _seed_for(bootstrap_seed, epsilon, "work-ratio"))
        ratio_rows.append(
            {
                "epsilon": epsilon,
                "region": nog[0]["region"],
                **{f"depth_me_over_nog_{key}": value for key, value in depth.items()},
                "depth_me_over_nog_capped_mean_ci_low": depth_ci[0],
                "depth_me_over_nog_capped_mean_ci_high": depth_ci[1],
                **{f"work_nog_over_me_{key}": value for key, value in work.items()},
                "work_nog_over_me_capped_mean_ci_low": work_ci[0],
                "work_nog_over_me_capped_mean_ci_high": work_ci[1],
            }
        )

    depth_ratios = [float(row["depth_me_over_nog_ratio_of_capped_means"]) for row in ratio_rows]
    work_ratios = [float(row["work_nog_over_me_ratio_of_capped_means"]) for row in ratio_rows]
    trends = {
        "depth_me_over_nog": trend_statistics(epsilons, depth_ratios),
        "work_nog_over_me": trend_statistics(epsilons, work_ratios),
        "work_ratio_mean": statistics.mean(work_ratios),
        "work_ratio_coefficient_of_variation": statistics.stdev(work_ratios) / statistics.mean(work_ratios),
        "interpretation": {
            "positive_depth_slope_means_advantage_grows_as_epsilon_decreases": True,
            "near_zero_work_slope_means_ratio_is_stable": True,
            "all_ratio_statistics_use_capped_values_when either method is censored": True,
        },
    }
    manifest = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_result_audit_status": audit["status"],
        "expanded_rows": len(expanded),
        "summary_rows": len(summary_rows),
        "ratio_rows": len(ratio_rows),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
        "confirmed_hit_consecutive": consecutive,
        "nonhits_retained": True,
        "primary_censoring_statistics": ["capped_mean", "restricted_mean", "mean_lower_bound"],
    }
    analysis_root = root / "analysis"
    atomic_write_csv(analysis_root / "formal_per_seed.csv", expanded)
    atomic_write_csv(analysis_root / "formal_summary.csv", summary_rows)
    atomic_write_csv(analysis_root / "formal_ratios.csv", ratio_rows)
    atomic_write_json(analysis_root / "formal_trends.json", trends)
    atomic_write_json(analysis_root / "formal_statistics_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    result = analyze_formal(cfg, root)
    status = result["status"]
    expanded = result["expanded_rows"]
    summaries = result["summary_rows"]
    ratios = result["ratio_rows"]
    print(f"status={status} expanded={expanded} summaries={summaries} ratios={ratios}")


if __name__ == "__main__":
    main()
