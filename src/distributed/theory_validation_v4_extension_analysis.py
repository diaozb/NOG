"""Audit and analyze the extended-budget continuation of v4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now
from src.distributed.theory_validation_audit import audit


CONFIG = "configs/distributed_cpu_fo_theory_validation_v4_extended_budget.yaml"
TIMING_FIELDS = {"time_sec", "training_time", "communication_time", "evaluation_time", "smooth_B", "data_B_per_worker"}


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return statistics.mean(rows) if rows else None


def _sd(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    if not rows:
        return None
    return statistics.stdev(rows) if len(rows) > 1 else 0.0


def _numeric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key not in TIMING_FIELDS}


def validate_v4_prefix(
    payloads: Dict[tuple[str, int], Dict[str, Any]], source_root: Path
) -> Dict[str, Any]:
    completion = _load(source_root / "completion.json")
    source = {}
    for record in completion["records"]:
        method = str(record["task"]["method"])
        if method == "NOG-FO" and "data-B-total-16" not in str(record["label"]):
            continue
        source[(method, int(record["task"]["formal_seed"]))] = _load(
            Path(record["partial_path"])
        )

    errors = []
    checked_rows = 0
    for identity, old in source.items():
        new = payloads.get(identity)
        if new is None:
            errors.append(f"missing extended payload for {identity}")
            continue
        if len(new["rows"]) <= len(old["rows"]):
            errors.append(f"extended trajectory is not longer for {identity}")
            continue
        for index, old_row in enumerate(old["rows"]):
            if _numeric_row(old_row) != _numeric_row(new["rows"][index]):
                errors.append(f"v4 prefix mismatch for {identity} at row {index}")
                break
            checked_rows += 1
    return {
        "status": "passed" if not errors else "failed",
        "source_root": str(source_root),
        "checked_tasks": len(source),
        "checked_rows": checked_rows,
        "ignored_fields": sorted(TIMING_FIELDS),
        "errors": errors,
    }


def _plot(summary: List[Dict[str, Any]], ratios: List[Dict[str, Any]], path: Path) -> None:
    epsilons = sorted({float(row["epsilon"]) for row in summary}, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for method, color in [("NOG-FO", "#1f77b4"), ("ME-DOL-FO", "#d62728")]:
        lookup = {float(row["epsilon"]): float(row["hit_rate"]) for row in summary if row["method"] == method}
        axes[0].plot(epsilons, [lookup[e] for e in epsilons], marker="o", label=method, color=color)
    full = [row for row in ratios if row["all_pairs_hit"]]
    if full:
        axes[1].plot(
            [float(row["epsilon"]) for row in full],
            [float(row["depth_ratio_mean"]) for row in full],
            marker="o",
            label="ME-DOL/NOG depth",
        )
        axes[1].plot(
            [float(row["epsilon"]) for row in full],
            [float(row["work_ratio_mean"]) for row in full],
            marker="s",
            label="NOG/ME-DOL work",
        )
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)")
        axis.legend()
    axes[0].set_ylabel("confirmed-hit rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("paired ratio (only 20/20 points)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(cfg: Dict[str, Any], formal_root: Path, analysis_root: Path) -> Dict[str, Any]:
    protocol = _load(formal_root / "extension_protocol.json")
    completion = _load(formal_root / "completion.json")
    audit_result = audit(formal_root / "extension_protocol.json", formal_root)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError("Extended-budget run is not complete and failure-free.")
    if audit_result["status"] != "passed":
        raise ValueError("Extended-budget artifact audit failed.")

    payloads: Dict[tuple[str, int], Dict[str, Any]] = {}
    input_paths = [formal_root / "extension_protocol.json", formal_root / "completion.json", formal_root / "formal_result_audit.json"]
    for record in completion["records"]:
        path = Path(record["partial_path"])
        payloads[(str(record["task"]["method"]), int(record["task"]["formal_seed"]))] = _load(path)
        input_paths.append(path)

    prefix = validate_v4_prefix(payloads, Path(cfg["extended_budget"]["source_formal_root"]))
    if prefix["status"] != "passed":
        raise ValueError(f"Extended trajectories do not reproduce the original v4 prefix: {prefix['errors'][:3]}")

    seeds = [int(value) for value in protocol["formal_seeds"]]
    epsilons = [float(value) for value in protocol["epsilons"]]
    consecutive = int(protocol["confirmed_hit_consecutive"])
    per_seed: List[Dict[str, Any]] = []
    lookup = {}
    for epsilon in epsilons:
        for method in ["NOG-FO", "ME-DOL-FO"]:
            for seed in seeds:
                payload = payloads[(method, seed)]
                hit = confirmed_hit(payload["rows"], epsilon, consecutive)
                final = payload["rows"][-1]
                row = {
                    "stage": protocol["stage"],
                    "method": method,
                    "epsilon": epsilon,
                    "formal_seed": seed,
                    "hit": hit is not None,
                    "first_hit_depth": hit["depth"] if hit else None,
                    "first_hit_total_work": hit["total_work"] if hit else None,
                    "censoring_depth": int(final["depth"]),
                    "censoring_total_work": int(final["total_work"]),
                    "final_stat_proxy": float(final["stat_proxy"]),
                }
                per_seed.append(row)
                lookup[(method, epsilon, seed)] = row

    summary: List[Dict[str, Any]] = []
    ratios: List[Dict[str, Any]] = []
    for epsilon in epsilons:
        for method in ["NOG-FO", "ME-DOL-FO"]:
            rows = [lookup[(method, epsilon, seed)] for seed in seeds]
            hits = [row for row in rows if row["hit"]]
            depths = [float(row["first_hit_depth"]) for row in hits]
            works = [float(row["first_hit_total_work"]) for row in hits]
            summary.append(
                {
                    "stage": protocol["stage"],
                    "method": method,
                    "epsilon": epsilon,
                    "num_seeds": len(rows),
                    "hit_count": len(hits),
                    "hit_rate": len(hits) / len(rows),
                    "depth_mean_hits": _mean(depths),
                    "depth_sd_hits": _sd(depths),
                    "total_work_mean_hits": _mean(works),
                    "total_work_sd_hits": _sd(works),
                    "capped_depth_mean": _mean(float(row["first_hit_depth"] if row["hit"] else row["censoring_depth"]) for row in rows),
                    "capped_total_work_mean": _mean(float(row["first_hit_total_work"] if row["hit"] else row["censoring_total_work"]) for row in rows),
                }
            )

        paired = []
        capped = []
        for seed in seeds:
            nog = lookup[("NOG-FO", epsilon, seed)]
            me = lookup[("ME-DOL-FO", epsilon, seed)]
            nd = float(nog["first_hit_depth"] if nog["hit"] else nog["censoring_depth"])
            md = float(me["first_hit_depth"] if me["hit"] else me["censoring_depth"])
            nw = float(nog["first_hit_total_work"] if nog["hit"] else nog["censoring_total_work"])
            mw = float(me["first_hit_total_work"] if me["hit"] else me["censoring_total_work"])
            capped.append((md / nd, nw / mw))
            if nog["hit"] and me["hit"]:
                paired.append((md / nd, nw / mw))
        all_hit = len(paired) == len(seeds)
        ratios.append(
            {
                "stage": protocol["stage"],
                "epsilon": epsilon,
                "paired_hit_count": len(paired),
                "all_pairs_hit": all_hit,
                "depth_ratio_mean": _mean(value[0] for value in paired) if all_hit else None,
                "depth_ratio_sd": _sd(value[0] for value in paired) if all_hit else None,
                "work_ratio_mean": _mean(value[1] for value in paired) if all_hit else None,
                "work_ratio_sd": _sd(value[1] for value in paired) if all_hit else None,
                "capped_depth_ratio_mean": _mean(value[0] for value in capped),
                "capped_work_ratio_mean": _mean(value[1] for value in capped),
            }
        )

    analysis_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(analysis_root / "extended_per_seed.csv", per_seed)
    atomic_write_csv(analysis_root / "extended_summary.csv", summary)
    atomic_write_csv(analysis_root / "extended_ratios.csv", ratios)
    atomic_write_json(analysis_root / "prefix_validation.json", prefix)
    _plot(summary, ratios, analysis_root / "figures" / "extended_hit_and_ratios.png")

    summary_lookup = {(row["method"], float(row["epsilon"])): row for row in summary}
    lines = [
        f"# v4 extended-budget continuation: {protocol['stage']}",
        "",
        f"- Budgets: NOG={protocol['budgets']['NOG-FO']} rounds; ME-DOL={protocol['budgets']['ME-DOL-FO']} rounds.",
        "- All problem, algorithm, batch, evaluation, worker, and seed settings are frozen from v4.",
        f"- Original v4 deterministic prefix validation: **{prefix['status']}** ({prefix['checked_tasks']} tasks).",
        "- A finite ratio is reported only when both methods hit on all 20 paired seeds.",
        "",
        "| epsilon | NOG hit | ME hit | ME/NOG depth | NOG/ME work |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in ratios:
        epsilon = float(row["epsilon"])
        n = summary_lookup[("NOG-FO", epsilon)]
        m = summary_lookup[("ME-DOL-FO", epsilon)]
        depth = f"{float(row['depth_ratio_mean']):.3f}x" if row["all_pairs_hit"] else "--"
        work = f"{float(row['work_ratio_mean']):.3f}x" if row["all_pairs_hit"] else "--"
        lines.append(f"| {epsilon:.4f} | {n['hit_count']}/20 | {m['hit_count']}/20 | {depth} | {work} |")
    (analysis_root / "extended_budget_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "stage": protocol["stage"],
        "budgets": protocol["budgets"],
        "artifact_audit": audit_result["status"],
        "prefix_validation": prefix,
        "fully_observed_epsilons": [float(row["epsilon"]) for row in ratios if row["all_pairs_hit"]],
        "input_artifacts": [{"path": str(path), "sha256": file_sha256(path)} for path in input_paths],
    }
    atomic_write_json(analysis_root / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage1", "stage2"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--formal-root", default=None)
    parser.add_argument("--analysis-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    formal_root = Path(args.formal_root or Path(cfg["run"]["out_dir"]) / args.stage)
    analysis_root = Path(args.analysis_root or formal_root / "analysis")
    result = analyze(cfg, formal_root, analysis_root)
    print(
        f"stage={result['stage']} status={result['status']} "
        f"fully_observed={len(result['fully_observed_epsilons'])}"
    )


if __name__ == "__main__":
    main()
