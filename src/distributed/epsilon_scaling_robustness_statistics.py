"""Create censoring-aware worker robustness summaries and paired ratios."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import atomic_write_json, utc_now
from src.distributed.epsilon_scaling import paired_ratio_summary, trend_statistics
from src.distributed.epsilon_scaling_formal_statistics import (
    _bootstrap_ci,
    _metric_summary,
    _seed_for,
)


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _payloads(root: Path, manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stage_records = {}
    for worker in (1, 2, 4):
        completion = _load(root / f"robustness_m{worker}_completion.json")
        stage_records.update({row["task_id"]: row for row in completion["records"]})
    result = {}
    for entry in manifest["tasks"]:
        if entry["source"] == "reuse-formal":
            path = root / entry["source_partial_path"]
        else:
            path = root / stage_records[entry["task_id"]]["partial_path"]
        result[entry["task_id"]] = _load(path)
    if len(result) != 240:
        raise ValueError("Expected 240 robustness payloads.")
    return result


def analyze_robustness(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    audit = _load(root / "robustness_result_audit.json")
    if audit.get("status") != "passed" or audit.get("passed_tasks") != 240:
        raise ValueError("Robustness result audit must pass before statistics.")
    manifest = _load(root / "robustness_manifest.json")
    payloads = _payloads(root, manifest)
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])
    repetitions = int(cfg["epsilon_scaling"]["statistics"]["bootstrap_repetitions"])
    bootstrap_seed = int(cfg["epsilon_scaling"]["statistics"]["bootstrap_seed"])

    expanded = []
    for entry in manifest["tasks"]:
        payload = payloads[entry["task_id"]]
        hit = confirmed_hit(payload["rows"], float(entry["epsilon"]), consecutive)
        final = payload["rows"][-1]
        expanded.append(
            {
                "task_id": entry["task_id"],
                "method": entry["method"],
                "epsilon": entry["epsilon"],
                "region": entry["region"],
                "worker_count": entry["worker_count"],
                "formal_seed": entry["formal_seed"],
                "source": entry["source"],
                "hit": hit is not None,
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
    lookup = {
        (row["method"], float(row["epsilon"]), int(row["worker_count"]), int(row["formal_seed"])): row
        for row in expanded
    }
    workers = [int(value) for value in manifest["workers"]]
    epsilons = [float(value) for value in manifest["epsilons"]]
    seeds = [int(value) for value in manifest["seeds"]]
    metric_fields = {
        "depth": ("first_hit_depth", "censoring_depth"),
        "total_work": ("first_hit_total_work", "censoring_total_work"),
        "per_worker_work": ("first_hit_per_worker_work", "censoring_per_worker_work"),
        "training_time": ("first_hit_training_time", "censoring_training_time"),
    }
    summaries = []
    for method in cfg["methods"]["sfo"]:
        for epsilon in epsilons:
            for worker in workers:
                rows = [lookup[(method, epsilon, worker, seed)] for seed in seeds]
                hit_values = [1.0 if row["hit"] else 0.0 for row in rows]
                hit_ci = _bootstrap_ci(
                    hit_values,
                    statistics.mean,
                    repetitions,
                    _seed_for(bootstrap_seed, "robustness", method, epsilon, worker, "hit"),
                )
                result: Dict[str, Any] = {
                    "method": method,
                    "epsilon": epsilon,
                    "region": rows[0]["region"],
                    "worker_count": worker,
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
                        _seed_for(bootstrap_seed, "robustness", method, epsilon, worker, metric),
                    )
                    result.update({f"{metric}_{key}": value for key, value in metric_summary.items()})
                summaries.append(result)

    ratios = []
    for epsilon in epsilons:
        for worker in workers:
            nog = [lookup[("NOG-FO", epsilon, worker, seed)] for seed in seeds]
            me = [lookup[("ME-DOL-FO", epsilon, worker, seed)] for seed in seeds]
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
            ratios.append(
                {
                    "epsilon": epsilon,
                    "worker_count": worker,
                    **{f"depth_me_over_nog_{key}": value for key, value in depth.items()},
                    **{f"work_nog_over_me_{key}": value for key, value in work.items()},
                }
            )

    trends = {}
    for epsilon in epsilons:
        rows = sorted(
            [row for row in ratios if float(row["epsilon"]) == epsilon],
            key=lambda row: int(row["worker_count"]),
        )
        inverse_workers = [1.0 / int(row["worker_count"]) for row in rows]
        trends[str(epsilon)] = {
            "workers": [int(row["worker_count"]) for row in rows],
            "depth_ratio_vs_workers": trend_statistics(
                inverse_workers,
                [float(row["depth_me_over_nog_ratio_of_capped_means"]) for row in rows],
            ),
            "work_ratio_vs_workers": trend_statistics(
                inverse_workers,
                [float(row["work_nog_over_me_ratio_of_capped_means"]) for row in rows],
            ),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "expanded_rows": len(expanded),
        "summary_rows": len(summaries),
        "ratio_rows": len(ratios),
        "workers": workers,
        "epsilons": epsilons,
        "seeds": seeds,
        "nonhits_retained": True,
        "bootstrap_repetitions": repetitions,
    }
    analysis_root = root / "analysis"
    atomic_write_csv(analysis_root / "robustness_per_seed.csv", expanded)
    atomic_write_csv(analysis_root / "robustness_summary.csv", summaries)
    atomic_write_csv(analysis_root / "robustness_ratios.csv", ratios)
    atomic_write_json(analysis_root / "robustness_trends.json", trends)
    atomic_write_json(analysis_root / "robustness_statistics_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    result = analyze_robustness(cfg, root)
    status = result["status"]
    expanded = result["expanded_rows"]
    summaries = result["summary_rows"]
    ratios = result["ratio_rows"]
    print(f"status={status} expanded={expanded} summaries={summaries} ratios={ratios}")


if __name__ == "__main__":
    main()
