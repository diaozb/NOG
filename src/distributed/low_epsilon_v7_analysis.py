"""Analyze formal theory-scaled v7 trajectories without post-hoc selection."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now
from src.distributed.epsilon_scaling import trend_statistics
from src.distributed.low_epsilon_v7_runner import (
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    me_label,
    nog_label,
)


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _bootstrap_ci(values: list[float], seed: int, repetitions: int = 2000) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty sample.")
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choice(values) for _ in values) for _ in range(repetitions)
    )
    return means[int(0.025 * repetitions)], means[min(repetitions - 1, int(0.975 * repetitions))]


def _segment(segments: list[Dict[str, Any]], epsilon: float) -> Dict[str, Any]:
    matches = [
        row
        for row in segments
        if float(row["epsilon_min"]) - 1e-12
        <= epsilon
        <= float(row["epsilon_max"]) + 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(f"epsilon={epsilon} maps to {len(matches)} segments.")
    return matches[0]


def _payloads(
    completion_path: Path, expected_seeds: list[int]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    completion = _load(completion_path)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError("v7 formal completion is not complete and failure-free.")
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for record in completion["records"]:
        label = str(record["label"])
        seed = int(record["task"]["formal_seed"])
        result.setdefault(label, {})[seed] = _load(Path(record["partial_path"]))
    expected = set(expected_seeds)
    if any(set(rows) != expected for rows in result.values()):
        raise ValueError("One or more v7 formal configurations have incomplete seeds.")
    return result


def analyze(
    cfg: Dict[str, Any],
    freeze_path: Path,
    completion_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    freeze = _load(freeze_path)
    if freeze.get("status") != "frozen":
        raise ValueError("v7 parameters are not frozen.")
    ext = cfg["theory_scaling"]
    epsilons = [float(value) for value in ext["epsilons"]]
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    consecutive = int(ext["confirmed_hit_consecutive"])
    depth = int(ext["formal_max_depth"])
    eta_scale = float(freeze["selected_global_constants"]["NOG-FO"]["eta_scale"])
    multiplier = float(
        freeze["selected_global_constants"]["ME-DOL-FO"]["theory_multiplier"]
    )
    payloads = _payloads(completion_path, seeds)
    expected_labels = {
        *(nog_label(row, eta_scale, depth) for row in ext["nog_segments"]),
        *(me_label(row, multiplier, depth) for row in ext["me_segments"]),
    }
    if set(payloads) != expected_labels:
        raise ValueError("Formal labels differ from the frozen theory schedule.")

    per_seed: list[Dict[str, Any]] = []
    summary: list[Dict[str, Any]] = []
    paired_by_epsilon: Dict[float, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for epsilon in epsilons:
        paired_by_epsilon[epsilon] = {seed: {} for seed in seeds}
        for method in ["NOG-FO", "ME-DOL-FO"]:
            segments = ext["nog_segments"] if method == "NOG-FO" else ext["me_segments"]
            segment = _segment(segments, epsilon)
            label = (
                nog_label(segment, eta_scale, depth)
                if method == "NOG-FO"
                else me_label(segment, multiplier, depth)
            )
            hit_depths, hit_works = [], []
            capped_depths, capped_works = [], []
            for seed in seeds:
                rows = payloads[label][seed]["rows"]
                hit = confirmed_hit(rows, epsilon, consecutive)
                final = rows[-1]
                value = hit if hit is not None else final
                capped_depth = float(value["depth"])
                capped_work = float(value["total_work"])
                if hit is not None:
                    hit_depths.append(float(hit["depth"]))
                    hit_works.append(float(hit["total_work"]))
                capped_depths.append(capped_depth)
                capped_works.append(capped_work)
                row = {
                    "method": method,
                    "epsilon": epsilon,
                    "formal_seed": seed,
                    "segment_id": str(segment["id"]),
                    "hit": hit is not None,
                    "first_hit_depth": float(hit["depth"]) if hit is not None else None,
                    "first_hit_total_work": float(hit["total_work"]) if hit is not None else None,
                    "capped_depth": capped_depth,
                    "capped_total_work": capped_work,
                }
                per_seed.append(row)
                paired_by_epsilon[epsilon][seed][method] = row
            d_mean, d_sd = _mean_sd(hit_depths) if hit_depths else (None, None)
            w_mean, w_sd = _mean_sd(hit_works) if hit_works else (None, None)
            if method == "NOG-FO":
                parameter = (
                    f"M={int(segment['M'])};eta={float(segment['eta']) * eta_scale:g};"
                    f"batch={int(segment['batch_total'])}"
                )
            else:
                parameter = (
                    f"H={int(segment['epoch_length'])};multiplier={multiplier:g};"
                    f"batch={int(segment['batch_total'])}"
                )
            summary.append(
                {
                    "method": method,
                    "epsilon": epsilon,
                    "segment_id": str(segment["id"]),
                    "parameter": parameter,
                    "batch_total": int(segment["batch_total"]),
                    "num_seeds": len(seeds),
                    "hit_count": len(hit_depths),
                    "hit_rate": len(hit_depths) / len(seeds),
                    "depth_mean_hits": d_mean,
                    "depth_sd_hits": d_sd,
                    "work_mean_hits": w_mean,
                    "work_sd_hits": w_sd,
                    "capped_depth_mean": statistics.mean(capped_depths),
                    "capped_work_mean": statistics.mean(capped_works),
                }
            )

    ratios = []
    for index, epsilon in enumerate(epsilons):
        values = paired_by_epsilon[epsilon]
        paired_hits = [
            seed
            for seed in seeds
            if values[seed]["NOG-FO"]["hit"] and values[seed]["ME-DOL-FO"]["hit"]
        ]
        use_capped = len(paired_hits) != len(seeds)
        selected_seeds = seeds if use_capped else paired_hits
        depth_ratios, work_ratios = [], []
        for seed in selected_seeds:
            n, m = values[seed]["NOG-FO"], values[seed]["ME-DOL-FO"]
            dk = "capped_depth" if use_capped else "first_hit_depth"
            wk = "capped_total_work" if use_capped else "first_hit_total_work"
            depth_ratios.append(float(m[dk]) / float(n[dk]))
            work_ratios.append(float(n[wk]) / float(m[wk]))
        dm, ds = _mean_sd(depth_ratios)
        wm, ws = _mean_sd(work_ratios)
        dl, dh = _bootstrap_ci(depth_ratios, 71000 + index)
        wl, wh = _bootstrap_ci(work_ratios, 72000 + index)
        ratios.append(
            {
                "epsilon": epsilon,
                "num_seeds": len(seeds),
                "paired_hit_count": len(paired_hits),
                "ratios_use_capped_values": use_capped,
                "depth_ratio_mean": dm,
                "depth_ratio_sd": ds,
                "depth_ratio_ci_low": dl,
                "depth_ratio_ci_high": dh,
                "work_ratio_mean": wm,
                "work_ratio_sd": ws,
                "work_ratio_ci_low": wl,
                "work_ratio_ci_high": wh,
            }
        )

    depth_values = [float(row["depth_ratio_mean"]) for row in ratios]
    work_values = [float(row["work_ratio_mean"]) for row in ratios]
    trends = {
        "depth_ratio": trend_statistics(epsilons, depth_values),
        "work_ratio": trend_statistics(epsilons, work_values),
        "depth_ratio_start": depth_values[0],
        "depth_ratio_end": depth_values[-1],
        "work_ratio_mean": statistics.mean(work_values),
        "work_ratio_min": min(work_values),
        "work_ratio_max": max(work_values),
        "work_ratio_coefficient_of_variation": statistics.stdev(work_values)
        / statistics.mean(work_values),
        "all_paired_hits": all(row["paired_hit_count"] == len(seeds) for row in ratios),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "formal_per_seed.csv", per_seed)
    atomic_write_csv(output_dir / "formal_summary.csv", summary)
    atomic_write_csv(output_dir / "formal_ratios.csv", ratios)
    atomic_write_json(output_dir / "formal_trends.json", trends)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_used_formal_results": False,
        "epsilon_count": len(epsilons),
        "formal_seed_count": len(seeds),
        "all_paired_hits": trends["all_paired_hits"],
        "freeze": {"path": str(freeze_path), "sha256": file_sha256(freeze_path)},
        "completion": {
            "path": str(completion_path),
            "sha256": file_sha256(completion_path),
        },
    }
    atomic_write_json(output_dir / "analysis_manifest.json", manifest)
    return {"manifest": manifest, "trends": trends}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--freeze", default=str(DEFAULT_ROOT / "frozen_parameters.json"))
    parser.add_argument("--completion", default=str(DEFAULT_ROOT / "formal" / "completion.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT / "analysis"))
    args = parser.parse_args()
    result = analyze(
        load_config(args.config),
        Path(args.freeze),
        Path(args.completion),
        Path(args.output_dir),
    )
    print(json.dumps(result["trends"], indent=2))


if __name__ == "__main__":
    main()
