"""Analyze symmetric low-epsilon formal results and confirm large ratio drops."""

from __future__ import annotations

import argparse
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
from src.distributed.low_epsilon_runner import formal, me_label, nog_label

SCHEMA_VERSION = 2


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


def _completion_payloads(
    completion_path: Path,
) -> tuple[Dict[tuple[str, int], Dict[str, Any]], List[Path]]:
    completion = _load(completion_path)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError(f"Completion is not complete: {completion_path}")
    payloads: Dict[tuple[str, int], Dict[str, Any]] = {}
    paths = [completion_path]
    for record in completion["records"]:
        path = Path(record["partial_path"])
        payloads[(str(record["label"]), int(record["task"]["formal_seed"]))] = _load(path)
        paths.append(path)
    return payloads, paths


def _label(
    freeze: Dict[str, Any], method: str, batch: int
) -> str:
    rounds = int(freeze["common_max_depth"])
    selected = freeze["selected_algorithms"][method]
    if method == "NOG-FO":
        return nog_label(selected["M"], selected["eta"], batch, rounds)
    return me_label(
        selected["epoch_length"],
        selected["theory_multiplier"],
        batch,
        rounds,
    )


def _rows_for(
    freeze: Dict[str, Any],
    payloads: Dict[tuple[str, int], Dict[str, Any]],
    seeds: List[int],
    schedule: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    consecutive = int(freeze["confirmed_hit_consecutive"])
    per_seed: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    ratios: List[Dict[str, Any]] = []

    for item in schedule:
        epsilon = float(item["epsilon"])
        scope = str(item["scope"])
        batches = {
            "NOG-FO": int(item["nog_batch_total"]),
            "ME-DOL-FO": int(item["me_batch_total"]),
        }
        for method in ["NOG-FO", "ME-DOL-FO"]:
            label = _label(freeze, method, batches[method])
            for seed in seeds:
                payload = payloads[(label, seed)]
                hit = confirmed_hit(payload["rows"], epsilon, consecutive)
                final = payload["rows"][-1]
                per_seed.append(
                    {
                        "scope": scope,
                        "method": method,
                        "epsilon": epsilon,
                        "formal_seed": seed,
                        "data_B_total": batches[method],
                        "hit": hit is not None,
                        "first_hit_depth": hit["depth"] if hit else None,
                        "first_hit_total_work": hit["total_work"] if hit else None,
                        "censoring_depth": int(final["depth"]),
                        "censoring_total_work": int(final["total_work"]),
                        "final_stat_proxy": float(final["stat_proxy"]),
                    }
                )

    lookup = {
        (row["method"], float(row["epsilon"]), row["scope"], int(row["formal_seed"])): row
        for row in per_seed
    }
    for item in schedule:
        epsilon = float(item["epsilon"])
        scope = str(item["scope"])
        for method in ["NOG-FO", "ME-DOL-FO"]:
            rows = [lookup[(method, epsilon, scope, seed)] for seed in seeds]
            hits = [row for row in rows if row["hit"]]
            summary.append(
                {
                    "scope": scope,
                    "method": method,
                    "epsilon": epsilon,
                    "data_B_total": int(
                        item["nog_batch_total"]
                        if method == "NOG-FO"
                        else item["me_batch_total"]
                    ),
                    "num_seeds": len(rows),
                    "hit_count": len(hits),
                    "hit_rate": len(hits) / len(rows),
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
                    "capped_depth_mean": statistics.mean(
                        float(row["first_hit_depth"] if row["hit"] else row["censoring_depth"])
                        for row in rows
                    ),
                    "capped_work_mean": statistics.mean(
                        float(row["first_hit_total_work"] if row["hit"] else row["censoring_total_work"])
                        for row in rows
                    ),
                    "mean_final_stat_proxy": statistics.mean(
                        float(row["final_stat_proxy"]) for row in rows
                    ),
                }
            )

        depth_ratios, work_ratios = [], []
        complete_depth, complete_work = [], []
        paired_hits = 0
        for seed in seeds:
            nog = lookup[("NOG-FO", epsilon, scope, seed)]
            me = lookup[("ME-DOL-FO", epsilon, scope, seed)]
            if nog["hit"] and me["hit"]:
                paired_hits += 1
                complete_depth.append(float(me["first_hit_depth"]) / float(nog["first_hit_depth"]))
                complete_work.append(float(nog["first_hit_total_work"]) / float(me["first_hit_total_work"]))
            nog_depth = float(nog["first_hit_depth"] if nog["hit"] else nog["censoring_depth"])
            me_depth = float(me["first_hit_depth"] if me["hit"] else me["censoring_depth"])
            nog_work = float(nog["first_hit_total_work"] if nog["hit"] else nog["censoring_total_work"])
            me_work = float(me["first_hit_total_work"] if me["hit"] else me["censoring_total_work"])
            depth_ratios.append(me_depth / nog_depth)
            work_ratios.append(nog_work / me_work)
        depth_ci = _mean_ci(depth_ratios, 31000 + int(round(epsilon * 10_000_000)))
        work_ci = _mean_ci(work_ratios, 41000 + int(round(epsilon * 10_000_000)))
        ratios.append(
            {
                "scope": scope,
                "epsilon": epsilon,
                "nog_batch_total": int(item["nog_batch_total"]),
                "me_batch_total": int(item["me_batch_total"]),
                "num_seeds": len(seeds),
                "paired_hit_count": paired_hits,
                "ratios_use_capped_values": paired_hits != len(seeds),
                "depth_ratio_mean": statistics.mean(depth_ratios),
                "depth_ratio_sd": statistics.stdev(depth_ratios),
                "depth_ratio_ci_low": depth_ci[0],
                "depth_ratio_ci_high": depth_ci[1],
                "work_ratio_mean": statistics.mean(work_ratios),
                "work_ratio_sd": statistics.stdev(work_ratios),
                "work_ratio_ci_low": work_ci[0],
                "work_ratio_ci_high": work_ci[1],
                "complete_case_depth_ratio_mean": statistics.mean(complete_depth) if complete_depth else None,
                "complete_case_work_ratio_mean": statistics.mean(complete_work) if complete_work else None,
            }
        )
    return per_seed, summary, ratios


def _anomalies(
    primary_ratios: List[Dict[str, Any]], drop_fraction: float
) -> List[Dict[str, Any]]:
    found = []
    for loose, strict in zip(primary_ratios, primary_ratios[1:]):
        previous = float(loose["depth_ratio_mean"])
        current = float(strict["depth_ratio_mean"])
        drop = (previous - current) / previous
        if drop > drop_fraction:
            found.append(
                {
                    "loose_epsilon": float(loose["epsilon"]),
                    "strict_epsilon": float(strict["epsilon"]),
                    "loose_depth_ratio": previous,
                    "strict_depth_ratio": current,
                    "drop_fraction": drop,
                }
            )
    return found


def _fixed_batch_rows(
    freeze: Dict[str, Any],
    payloads: Dict[tuple[str, int], Dict[str, Any]],
    seeds: List[int],
) -> List[Dict[str, Any]]:
    common = sorted(
        set(int(v) for v in freeze["selected_batches"]["NOG-FO"])
        & set(int(v) for v in freeze["selected_batches"]["ME-DOL-FO"])
    )
    rows = []
    consecutive = int(freeze["confirmed_hit_consecutive"])
    for batch in common:
        nog_label_value = _label(freeze, "NOG-FO", batch)
        me_label_value = _label(freeze, "ME-DOL-FO", batch)
        for epsilon in freeze["all_requested_epsilons"]:
            pairs = []
            for seed in seeds:
                nog = confirmed_hit(payloads[(nog_label_value, seed)]["rows"], float(epsilon), consecutive)
                me = confirmed_hit(payloads[(me_label_value, seed)]["rows"], float(epsilon), consecutive)
                if nog and me:
                    pairs.append((nog, me))
            rows.append(
                {
                    "batch_total": batch,
                    "epsilon": float(epsilon),
                    "num_seeds": len(seeds),
                    "paired_hit_count": len(pairs),
                    "depth_ratio_mean": statistics.mean(
                        float(me["depth"]) / float(nog["depth"]) for nog, me in pairs
                    ) if pairs else None,
                    "work_ratio_mean": statistics.mean(
                        float(nog["total_work"]) / float(me["total_work"]) for nog, me in pairs
                    ) if pairs else None,
                }
            )
    return rows


def _plot_ratios(rows: List[Dict[str, Any]], path: Path) -> None:
    primary = [row for row in rows if row["scope"] == "primary"]
    diagnostic = [row for row in rows if row["scope"] == "diagnostic_midpoint"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for axis, field, low, high, ylabel in [
        (axes[0], "depth_ratio_mean", "depth_ratio_ci_low", "depth_ratio_ci_high", "ME-DOL / NOG depth"),
        (axes[1], "work_ratio_mean", "work_ratio_ci_low", "work_ratio_ci_high", "NOG / ME-DOL work"),
    ]:
        for selected, marker, color, label in [
            (primary, "o", "#1f77b4", "preregistered schedule"),
            (diagnostic, "D", "#9467bd", "diagnostic midpoint"),
        ]:
            if not selected:
                continue
            eps = [float(row["epsilon"]) for row in selected]
            vals = [float(row[field]) for row in selected]
            axis.plot(eps, vals, marker=marker, color=color, linewidth=1.5, markersize=4, label=label)
            axis.fill_between(eps, [float(row[low]) for row in selected], [float(row[high]) for row in selected], color=color, alpha=0.15)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        axis.invert_xaxis(); axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)")
        axis.set_ylabel(ylabel); axis.grid(True, alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_hits(summary: List[Dict[str, Any]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    for method, marker, color in [("NOG-FO", "o", "#1f77b4"), ("ME-DOL-FO", "s", "#d62728")]:
        rows = [row for row in summary if row["method"] == method and row["scope"] == "primary"]
        axis.plot([float(row["epsilon"]) for row in rows], [float(row["hit_rate"]) for row in rows], marker=marker, color=color, label=method)
    axis.invert_xaxis(); axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)"); axis.set_ylabel("confirmed-hit rate")
    axis.grid(True, alpha=0.25); axis.legend(); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_fixed(rows: List[Dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for batch in sorted({int(row["batch_total"]) for row in rows}):
        selected = [row for row in rows if int(row["batch_total"]) == batch and int(row["paired_hit_count"]) == int(row["num_seeds"])]
        if not selected:
            continue
        for axis, field, ylabel in [(axes[0], "depth_ratio_mean", "ME-DOL / NOG depth"), (axes[1], "work_ratio_mean", "NOG / ME-DOL work")]:
            axis.plot([float(row["epsilon"]) for row in selected], [float(row[field]) for row in selected], marker="o", markersize=3, label=f"total batch {batch}")
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9); axis.invert_xaxis()
        axis.set_xlabel(r"target $\epsilon$ (decreasing $\rightarrow$)"); axis.grid(True, alpha=0.25); axis.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(
    cfg: Dict[str, Any], freeze_path: Path, formal_root: Path,
    extra_root: Path, analysis_root: Path, auto_confirm: bool = True,
) -> Dict[str, Any]:
    freeze = _load(freeze_path)
    payloads, input_paths = _completion_payloads(formal_root / "completion.json")
    audit_path = formal_root / "formal_result_audit.json"
    if audit_path.exists(): input_paths.append(audit_path)
    seeds = [int(value) for value in freeze["formal_seeds"]]
    schedule = [
        {
            "epsilon": float(row["epsilon"]), "scope": "primary",
            "nog_batch_total": int(row["NOG-FO"]["batch_total"]),
            "me_batch_total": int(row["ME-DOL-FO"]["batch_total"]),
        }
        for row in freeze["selected_schedule"]
    ]
    per_seed, summary, ratios = _rows_for(freeze, payloads, seeds, schedule)
    threshold = float(freeze["anomaly_confirmation"]["adjacent_depth_ratio_drop_fraction"])
    initial_anomalies = _anomalies(ratios, threshold)
    extra_used = False
    midpoint_pairs = {(row["loose_epsilon"], row["strict_epsilon"]) for row in initial_anomalies}
    if initial_anomalies and auto_confirm:
        extra_completion = extra_root / "completion.json"
        if not extra_completion.exists(): formal(cfg, freeze_path, extra_root, extra=True)
        extra_payloads, extra_paths = _completion_payloads(extra_completion)
        payloads.update(extra_payloads); input_paths.extend(extra_paths)
        extra_audit = extra_root / "formal_result_audit.json"
        if extra_audit.exists(): input_paths.append(extra_audit)
        seeds = sorted(seeds + [int(v) for v in freeze["anomaly_confirmation"]["extra_formal_seeds"]])
        midpoints = []
        for loose, strict in zip(schedule, schedule[1:]):
            if (loose["epsilon"], strict["epsilon"]) in midpoint_pairs:
                midpoints.append({
                    "epsilon": (loose["epsilon"] + strict["epsilon"]) / 2.0,
                    "scope": "diagnostic_midpoint",
                    "nog_batch_total": strict["nog_batch_total"],
                    "me_batch_total": strict["me_batch_total"],
                })
        expanded = sorted(schedule + midpoints, key=lambda row: row["epsilon"], reverse=True)
        per_seed, summary, ratios = _rows_for(freeze, payloads, seeds, expanded)
        extra_used = True
    primary = [row for row in ratios if row["scope"] == "primary"]
    remaining_anomalies = _anomalies(primary, threshold)
    depth_values = [float(row["depth_ratio_mean"]) for row in primary]
    work_values = [float(row["work_ratio_mean"]) for row in primary]
    eps = [float(row["epsilon"]) for row in primary]
    depth_trend = trend_statistics(eps, depth_values)
    work_trend = trend_statistics(eps, work_values)
    work_cv = statistics.stdev(work_values) / statistics.mean(work_values)
    criteria = freeze["formal_success_criteria"]
    primary_summary = [row for row in summary if row["scope"] == "primary"]
    required_hits = math.ceil(
        int(criteria["minimum_hits_of_20"]) / 20.0 * len(seeds)
    )
    verdict = {
        "depth_advantage_grows": depth_trend["spearman_rho"] >= float(criteria["minimum_depth_ratio_spearman"]) and depth_values[-1] > depth_values[0],
        "work_is_matched": min(work_values) >= float(criteria["work_ratio_lower"]) and max(work_values) <= float(criteria["work_ratio_upper"]) and work_cv <= float(criteria["maximum_work_ratio_cv"]),
        "hit_rate_passes": all(int(row["hit_count"]) >= required_hits for row in primary_summary),
        "no_unresolved_large_depth_drop": not remaining_anomalies,
    }
    verdict["low_epsilon_claim_supported"] = all(verdict.values())
    fixed = _fixed_batch_rows(freeze, payloads, seeds)
    trends = {
        "primary_epsilon_count": len(primary), "formal_seed_count": len(seeds),
        "extra_confirmation_used": extra_used, "required_hits": required_hits,
        "initial_adjacent_depth_drop_anomalies": initial_anomalies,
        "adjacent_depth_drop_anomalies": remaining_anomalies,
        "depth_ratio": depth_trend, "work_ratio": work_trend,
        "depth_ratio_start": depth_values[0], "depth_ratio_end": depth_values[-1],
        "work_ratio_mean": statistics.mean(work_values), "work_ratio_min": min(work_values),
        "work_ratio_max": max(work_values), "work_ratio_coefficient_of_variation": work_cv,
        "verdict": verdict,
    }
    analysis_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(analysis_root / "formal_per_seed.csv", per_seed)
    atomic_write_csv(analysis_root / "formal_summary.csv", summary)
    atomic_write_csv(analysis_root / "formal_ratios.csv", ratios)
    atomic_write_csv(analysis_root / "fixed_batch_diagnostics.csv", fixed)
    atomic_write_json(analysis_root / "formal_trends.json", trends)
    _plot_ratios(ratios, analysis_root / "figures" / "low_epsilon_ratios.png")
    _plot_hits(summary, analysis_root / "figures" / "low_epsilon_hit_rates.png")
    _plot_fixed(fixed, analysis_root / "figures" / "fixed_batch_diagnostics.png")
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "complete", "created_at_utc": utc_now(),
        "per_seed_rows": len(per_seed), "summary_rows": len(summary), "ratio_rows": len(ratios),
        "fixed_batch_rows": len(fixed), "verdict": verdict,
        "input_artifacts": [{"path": str(path), "sha256": file_sha256(path)} for path in input_paths],
    }
    atomic_write_json(analysis_root / "analysis_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric")
    parser.add_argument("--config", default="configs/distributed_cpu_fo_low_epsilon_v5.yaml")
    parser.add_argument("--freeze", default=str(root / "frozen_parameters.json"))
    parser.add_argument("--formal-root", default=str(root / "formal"))
    parser.add_argument("--extra-root", default=str(root / "formal_extra"))
    parser.add_argument("--analysis-root", default=str(root / "analysis"))
    parser.add_argument("--no-auto-confirm", action="store_true")
    args = parser.parse_args()
    result = analyze(load_config(args.config), Path(args.freeze), Path(args.formal_root), Path(args.extra_root), Path(args.analysis_root), auto_confirm=not args.no_auto_confirm)
    print(f"status={result['status']} verdict={result['verdict']}")


if __name__ == "__main__":
    main()
