"""Pilot-only calibration and freeze for the theory-scaled v7 protocol."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import (
    atomic_write_json,
    file_sha256,
    object_sha256,
    utc_now,
)
from src.distributed.low_epsilon_v7_runner import (
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    pilot_descriptors,
)


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _payloads(
    completion_path: Path, expected_seeds: list[int]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    completion = _load(completion_path)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError("v7 pilot is not complete and failure-free.")
    result: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for record in completion["records"]:
        label = str(record["label"])
        seed = int(record["task"]["formal_seed"])
        result.setdefault(label, {})[seed] = _load(Path(record["partial_path"]))
    expected = set(expected_seeds)
    if any(set(rows) != expected for rows in result.values()):
        raise ValueError("One or more v7 pilot candidates have incomplete seeds.")
    return result


def _measure(
    descriptor: Dict[str, Any],
    rows: Dict[int, Dict[str, Any]],
    seeds: list[int],
    consecutive: int,
) -> Dict[str, Any]:
    epsilon = float(descriptor["segment"]["representative_epsilon"])
    hits = [confirmed_hit(rows[seed]["rows"], epsilon, consecutive) for seed in seeds]
    good = [hit for hit in hits if hit is not None]
    depths = [float(hit["depth"]) for hit in good]
    works = [float(hit["total_work"]) for hit in good]
    return {
        "label": str(descriptor["label"]),
        "method": str(descriptor["method"]),
        "segment_id": str(descriptor["segment"]["id"]),
        "representative_epsilon": epsilon,
        "eta_scale": descriptor.get("eta_scale"),
        "theory_multiplier": descriptor.get("theory_multiplier"),
        "hit_count": len(good),
        "mean_depth_hits": statistics.mean(depths) if depths else None,
        "mean_total_work_hits": statistics.mean(works) if works else None,
    }


def _candidate_summary(
    rows: list[Dict[str, Any]], method: str, constant_key: str, required: int
) -> list[Dict[str, Any]]:
    values = sorted({float(row[constant_key]) for row in rows if row["method"] == method})
    result = []
    for value in values:
        selected = [
            row
            for row in rows
            if row["method"] == method and float(row[constant_key]) == value
        ]
        full = [row for row in selected if int(row["hit_count"]) == required]
        log_work = (
            statistics.mean(math.log(float(row["mean_total_work_hits"])) for row in full)
            if full
            else math.inf
        )
        log_depth = (
            statistics.mean(math.log(float(row["mean_depth_hits"])) for row in full)
            if full
            else math.inf
        )
        result.append(
            {
                "method": method,
                "constant_key": constant_key,
                "constant_value": value,
                "full_anchor_count": len(full),
                "anchor_count": len(selected),
                "total_hits": sum(int(row["hit_count"]) for row in selected),
                "geometric_mean_work": math.exp(log_work) if math.isfinite(log_work) else None,
                "geometric_mean_depth": math.exp(log_depth) if math.isfinite(log_depth) else None,
            }
        )
    return result


def _select(rows: list[Dict[str, Any]], anchor_count: int) -> Dict[str, Any]:
    eligible = [row for row in rows if int(row["full_anchor_count"]) == anchor_count]
    if not eligible:
        detail = ", ".join(
            f"{row['constant_value']}:anchors={row['full_anchor_count']}/{anchor_count},"
            f"hits={row['total_hits']}" for row in rows
        )
        raise ValueError(f"No global constant covered all pilot anchors: {detail}")
    return min(
        eligible,
        key=lambda row: (
            float(row["geometric_mean_work"]),
            float(row["geometric_mean_depth"]),
            float(row["constant_value"]),
        ),
    )


def freeze(
    cfg: Dict[str, Any], completion_path: Path, output_path: Path
) -> Dict[str, Any]:
    if list((output_path.parent / "formal").glob("**/partials/*.json")):
        raise ValueError("Formal artifacts exist; refusing post-hoc v7 calibration.")
    ext = cfg["theory_scaling"]
    seeds = [int(value) for value in ext["pilot_seeds"]]
    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if set(seeds) & set(formal_seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    consecutive = int(ext["confirmed_hit_consecutive"])
    descriptors = pilot_descriptors(cfg)
    payloads = _payloads(completion_path, seeds)
    if set(payloads) != set(descriptors):
        raise ValueError("Pilot labels do not equal the preregistered v7 candidates.")
    measurements = [
        _measure(descriptor, payloads[label], seeds, consecutive)
        for label, descriptor in sorted(descriptors.items())
    ]
    required = int(ext["calibration"]["require_hits_per_anchor"])
    if required != len(seeds):
        raise ValueError("v7 requires full pilot-seed coverage at every anchor.")
    nog_summary = _candidate_summary(measurements, "NOG-FO", "eta_scale", required)
    me_summary = _candidate_summary(
        measurements, "ME-DOL-FO", "theory_multiplier", required
    )
    nog_choice = _select(nog_summary, len(ext["nog_segments"]))
    me_choice = _select(me_summary, len(ext["me_segments"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "selection_happened_before_formal_runs": True,
        "selection_used_formal_results": False,
        "selection_rule": (
            "independently require 5/5 hits at every theory-scaling anchor, then "
            "minimize geometric mean first-hit total work; depth and constant tie-break"
        ),
        "pilot_seeds": seeds,
        "formal_seeds": formal_seeds,
        "pilot_max_depth": int(ext["pilot_max_depth"]),
        "formal_max_depth": int(ext["formal_max_depth"]),
        "selected_global_constants": {
            "NOG-FO": {"eta_scale": float(nog_choice["constant_value"])},
            "ME-DOL-FO": {
                "theory_multiplier": float(me_choice["constant_value"])
            },
        },
        "nog_segments": ext["nog_segments"],
        "me_segments": ext["me_segments"],
        "candidate_summaries": nog_summary + me_summary,
        "pilot_completion": {
            "path": str(completion_path),
            "sha256": file_sha256(completion_path),
        },
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "pilot_calibration.csv", measurements)
    atomic_write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--completion", default=str(DEFAULT_ROOT / "pilot" / "completion.json")
    )
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "frozen_parameters.json"))
    args = parser.parse_args()
    result = freeze(load_config(args.config), Path(args.completion), Path(args.output))
    print(
        "status=frozen eta_scale={} me_multiplier={}".format(
            result["selected_global_constants"]["NOG-FO"]["eta_scale"],
            result["selected_global_constants"]["ME-DOL-FO"]["theory_multiplier"],
        )
    )


if __name__ == "__main__":
    main()
