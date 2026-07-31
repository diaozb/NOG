"""Analyze v6 work-optimal and depth-optimal formal selections."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now
from src.distributed.epsilon_scaling import trend_statistics
from src.distributed.low_epsilon_runner import me_label, nog_label
from src.distributed.low_epsilon_v6_runner import DEFAULT_CONFIG, DEFAULT_ROOT


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _mean_ci(
    values: Iterable[float], seed: int, repetitions: int = 2000
) -> tuple[float | None, float | None]:
    observed = [float(value) for value in values]
    if not observed:
        return None, None
    rng = random.Random(int(seed))
    estimates = sorted(
        statistics.mean(rng.choice(observed) for _ in observed)
        for _ in range(repetitions)
    )
    return (
        estimates[int(0.025 * (repetitions - 1))],
        estimates[int(0.975 * (repetitions - 1))],
    )


def _payloads(
    completion_path: Path,
) -> tuple[Dict[tuple[str, int], Dict[str, Any]], list[Path]]:
    completion = _load(completion_path)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError("Formal completion is not complete and failure-free.")
    result: Dict[tuple[str, int], Dict[str, Any]] = {}
    paths = [completion_path]
    for record in completion["records"]:
        path = Path(record["partial_path"])
        key = (str(record["label"]), int(record["task"]["formal_seed"]))
        result[key] = _load(path)
        paths.append(path)
    return result, paths


def _formal_label(row: Dict[str, Any], depth: int) -> str:
    if row["method"] == "NOG-FO":
        return nog_label(row["M"], row["eta"], row["batch_total"], depth)
    return me_label(
        row["epoch_length"],
        row["theory_multiplier"],
        row["batch_total"],
        depth,
    )


def _parameter_text(row: Dict[str, Any]) -> str:
    if row["method"] == "NOG-FO":
        return f"M={int(row['M'])};eta={float(row['eta']):g};batch={int(row['batch_total'])}"
    return (
        f"H={int(row['epoch_length'])};multiplier="
        f"{float(row['theory_multiplier']):g};batch={int(row['batch_total'])}"
    )


def analyze(
    cfg: Dict[str, Any], freeze_path: Path, formal_root: Path, analysis_root: Path
) -> Dict[str, Any]:
    freeze = _load(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("v6 parameters are not frozen.")
    payloads, input_paths = _payloads(formal_root / "completion.json")
    input_paths.append(freeze_path)
    seeds = [int(value) for value in freeze["formal_seeds"]]
    consecutive = int(cfg["low_epsilon_extension"]["confirmed_hit_consecutive"])
    formal_depth = int(freeze["formal_max_depth"])
    per_seed: list[Dict[str, Any]] = []
    summary: list[Dict[str, Any]] = []
    ratios: list[Dict[str, Any]] = []

    for regime, selected_by_epsilon in freeze["selected"].items():
        for epsilon_key, by_method in selected_by_epsilon.items():
            epsilon = float(epsilon_key)
            current: Dict[tuple[str, int], Dict[str, Any]] = {}
            for method in ["NOG-FO", "ME-DOL-FO"]:
                selected = by_method[method]
                label = _formal_label(selected, formal_depth)
                for seed in seeds:
                    payload = payloads[(label, seed)]
                    hit = confirmed_hit(payload["rows"], epsilon, consecutive)
                    final = payload["rows"][-1]
                    row = {
                        "regime": regime,
                        "method": method,
                        "epsilon": epsilon,
                        "formal_seed": seed,
                        "parameter": _parameter_text(selected),
                        "batch_total": int(selected["batch_total"]),
                        "hit": hit is not None,
                        "first_hit_depth": hit["depth"] if hit else None,
                        "first_hit_total_work": hit["total_work"] if hit else None,
                        "censoring_depth": int(final["depth"]),
                        "censoring_total_work": int(final["total_work"]),
                        "final_stat_proxy": float(final["stat_proxy"]),
                    }
                    per_seed.append(row)
                    current[(method, seed)] = row

                rows = [current[(method, seed)] for seed in seeds]
                hits = [row for row in rows if row["hit"]]
                summary.append(
                    {
                        "regime": regime,
                        "method": method,
                        "epsilon": epsilon,
                        "parameter": _parameter_text(selected),
                        "batch_total": int(selected["batch_total"]),
                        "num_seeds": len(seeds),
                        "hit_count": len(hits),
                        "hit_rate": len(hits) / len(seeds),
                        "depth_mean_hits": statistics.mean(
                            float(row["first_hit_depth"]) for row in hits
                        ) if hits else None,
                        "depth_sd_hits": statistics.stdev(
                            float(row["first_hit_depth"]) for row in hits
                        ) if len(hits) > 1 else 0.0 if hits else None,
                        "work_mean_hits": statistics.mean(
                            float(row["first_hit_total_work"]) for row in hits
                        ) if hits else None,
                        "work_sd_hits": statistics.stdev(
                            float(row["first_hit_total_work"]) for row in hits
                        ) if len(hits) > 1 else 0.0 if hits else None,
                    }
                )

            depth_values, work_values = [], []
            paired_hits = 0
            for seed in seeds:
                nog = current[("NOG-FO", seed)]
                me = current[("ME-DOL-FO", seed)]
                if nog["hit"] and me["hit"]:
                    paired_hits += 1
                nog_depth = float(
                    nog["first_hit_depth"] if nog["hit"] else nog["censoring_depth"]
                )
                me_depth = float(
                    me["first_hit_depth"] if me["hit"] else me["censoring_depth"]
                )
                nog_work = float(
                    nog["first_hit_total_work"]
                    if nog["hit"]
                    else nog["censoring_total_work"]
                )
                me_work = float(
                    me["first_hit_total_work"]
                    if me["hit"]
                    else me["censoring_total_work"]
                )
                depth_values.append(me_depth / nog_depth)
                work_values.append(nog_work / me_work)
            depth_ci = _mean_ci(
                depth_values,
                51000 + int(round(epsilon * 10_000_000)) + len(regime),
            )
            work_ci = _mean_ci(
                work_values,
                61000 + int(round(epsilon * 10_000_000)) + len(regime),
            )
            ratios.append(
                {
                    "regime": regime,
                    "epsilon": epsilon,
                    "num_seeds": len(seeds),
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

    trends: Dict[str, Any] = {}
    for regime in freeze["selected"]:
        selected = [row for row in ratios if row["regime"] == regime]
        epsilon_values = [float(row["epsilon"]) for row in selected]
        depth_values = [float(row["depth_ratio_mean"]) for row in selected]
        work_values = [float(row["work_ratio_mean"]) for row in selected]
        trends[regime] = {
            "depth_ratio": trend_statistics(epsilon_values, depth_values),
            "work_ratio": trend_statistics(epsilon_values, work_values),
            "depth_ratio_start": depth_values[0],
            "depth_ratio_end": depth_values[-1],
            "work_ratio_mean": statistics.mean(work_values),
            "work_ratio_min": min(work_values),
            "work_ratio_max": max(work_values),
            "work_ratio_coefficient_of_variation": (
                statistics.stdev(work_values) / statistics.mean(work_values)
            ),
            "all_paired_hits": all(
                int(row["paired_hit_count"]) == int(row["num_seeds"])
                for row in selected
            ),
        }

    analysis_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(analysis_root / "formal_per_seed.csv", per_seed)
    atomic_write_csv(analysis_root / "formal_summary.csv", summary)
    atomic_write_csv(analysis_root / "formal_ratios.csv", ratios)
    atomic_write_json(analysis_root / "formal_trends.json", trends)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    colors = {"work_optimal": "#1f77b4", "depth_optimal": "#d62728"}
    for regime in freeze["selected"]:
        selected = [row for row in ratios if row["regime"] == regime]
        epsilon_values = [float(row["epsilon"]) for row in selected]
        for axis, field, ylabel in [
            (axes[0], "depth_ratio_mean", "ME-DOL / NOG depth"),
            (axes[1], "work_ratio_mean", "NOG / ME-DOL work"),
        ]:
            axis.plot(
                epsilon_values,
                [float(row[field]) for row in selected],
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=regime.replace("_", "-"),
                color=colors.get(regime),
            )
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        axis.invert_xaxis()
        axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    figure_path = analysis_root / "figures" / "joint_retune_ratios.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_used_formal_results": False,
        "per_seed_rows": len(per_seed),
        "summary_rows": len(summary),
        "ratio_rows": len(ratios),
        "trends": trends,
        "input_artifacts": [
            {"path": str(path), "sha256": file_sha256(path)} for path in input_paths
        ],
    }
    atomic_write_json(analysis_root / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", default=str(DEFAULT_ROOT / "frozen_parameters.json"))
    parser.add_argument("--formal-root", default=str(DEFAULT_ROOT / "formal"))
    parser.add_argument("--analysis-root", default=str(DEFAULT_ROOT / "analysis"))
    args = parser.parse_args()
    result = analyze(
        load_config(args.config),
        Path(args.freeze),
        Path(args.formal_root),
        Path(args.analysis_root),
    )
    print(f"status={result['status']} trends={json.dumps(result['trends'])}")


if __name__ == "__main__":
    main()
