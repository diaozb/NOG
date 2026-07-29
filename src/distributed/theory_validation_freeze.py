"""Freeze a pilot-calibrated matched-work schedule before formal seeds run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import (
    atomic_write_json,
    file_sha256,
    object_sha256,
    utc_now,
)


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def select_monotone_schedule(
    per_epsilon: List[List[Dict[str, Any]]], switch_penalty: float = 0.05
) -> List[Dict[str, Any]]:
    """Match work as closely as possible while batch never decreases with accuracy."""

    if not per_epsilon or any(not rows for rows in per_epsilon):
        raise ValueError("Every primary epsilon needs at least one eligible candidate.")
    states: Dict[int, tuple[float, int, List[Dict[str, Any]]]] = {}
    for row in per_epsilon[0]:
        batch = int(row["data_B_total"])
        states[batch] = (abs(math.log(float(row["work_ratio_nog_over_me"]))), 0, [row])
    for rows in per_epsilon[1:]:
        next_states: Dict[int, tuple[float, int, List[Dict[str, Any]]]] = {}
        for row in rows:
            batch = int(row["data_B_total"])
            local = abs(math.log(float(row["work_ratio_nog_over_me"])))
            options = []
            for previous_batch, (cost, switches, path) in states.items():
                if batch < previous_batch:
                    continue
                changed = int(batch != previous_batch)
                options.append(
                    (
                        cost + local + switch_penalty * changed,
                        switches + changed,
                        path + [row],
                    )
                )
            if options:
                next_states[batch] = min(options, key=lambda item: (item[0], item[1]))
        if not next_states:
            raise ValueError("No nondecreasing batch schedule covers all epsilons.")
        states = next_states
    return min(states.values(), key=lambda item: (item[0], item[1]))[2]


def freeze_parameters(
    cfg: Dict[str, Any], batch_root: Path, me_root: Path, output_path: Path
) -> Dict[str, Any]:
    completion_path = batch_root / "completion.json"
    existing_formal = list((output_path.parent / "formal").glob("**/partials/*.json"))
    if existing_formal:
        raise ValueError("Formal partials already exist; refusing post-hoc parameter freeze.")
    completion = _load(completion_path)
    if completion.get("status") != "complete" or completion.get("completed_tasks") != 40:
        raise ValueError("Batch-grid pilot must be complete 40/40 before freezing.")
    pilot_seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if set(pilot_seeds) & set(formal_seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    epsilons = [
        float(value)
        for value in cfg["epsilon_scaling"]["epsilons"]
        if float(value) >= 0.01
    ]
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])

    nog_payloads: Dict[int, Dict[int, Dict[str, Any]]] = {}
    input_paths = [completion_path]
    for record in completion["records"]:
        batch = int(record["label"].rsplit("-", 1)[1])
        seed = int(record["task"]["formal_seed"])
        path = Path(record["partial_path"])
        input_paths.append(path)
        nog_payloads.setdefault(batch, {})[seed] = _load(path)
    if any(set(rows) != set(pilot_seeds) for rows in nog_payloads.values()):
        raise ValueError("A NOG batch candidate is missing pilot seeds.")

    me_paths = sorted((me_root / "rounds_3840" / "partials").glob("*.json"))
    me_payloads = {int(_load(path)["formal_seed"]): _load(path) for path in me_paths}
    input_paths.extend(me_paths)
    if set(me_payloads) != set(pilot_seeds):
        raise ValueError("ME-DOL high-resolution pilot is not complete for five seeds.")

    calibration_rows: List[Dict[str, Any]] = []
    eligible_by_epsilon: List[List[Dict[str, Any]]] = []
    for epsilon in epsilons:
        me_hits = [
            confirmed_hit(me_payloads[seed]["rows"], epsilon, consecutive)
            for seed in pilot_seeds
        ]
        if any(hit is None for hit in me_hits):
            raise ValueError(f"ME-DOL did not hit primary epsilon={epsilon} on all pilot seeds.")
        me_depth = statistics.mean(float(hit["depth"]) for hit in me_hits if hit)
        me_work = statistics.mean(float(hit["total_work"]) for hit in me_hits if hit)
        eligible = []
        for batch in sorted(nog_payloads):
            hits = [
                confirmed_hit(nog_payloads[batch][seed]["rows"], epsilon, consecutive)
                for seed in pilot_seeds
            ]
            hit_count = sum(hit is not None for hit in hits)
            nog_depth = (
                statistics.mean(float(hit["depth"]) for hit in hits if hit)
                if hit_count
                else None
            )
            nog_work = (
                statistics.mean(float(hit["total_work"]) for hit in hits if hit)
                if hit_count
                else None
            )
            row = {
                "epsilon": epsilon,
                "data_B_total": batch,
                "nog_hit_count": hit_count,
                "me_hit_count": len(me_hits),
                "nog_mean_depth": nog_depth,
                "me_mean_depth": me_depth,
                "depth_ratio_me_over_nog": me_depth / nog_depth if nog_depth else None,
                "nog_mean_total_work": nog_work,
                "me_mean_total_work": me_work,
                "work_ratio_nog_over_me": nog_work / me_work if nog_work else None,
                "eligible_full_hit": hit_count == len(pilot_seeds),
            }
            calibration_rows.append(row)
            if row["eligible_full_hit"]:
                eligible.append(row)
        if not eligible:
            raise ValueError(f"No full-hit NOG batch candidate for epsilon={epsilon}.")
        eligible_by_epsilon.append(eligible)

    selected = select_monotone_schedule(eligible_by_epsilon)
    selected_batches = [int(row["data_B_total"]) for row in selected]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "selection_happened_before_formal_runs": True,
        "pilot_seeds": pilot_seeds,
        "formal_seeds": formal_seeds,
        "seed_sets_disjoint": True,
        "primary_epsilons": epsilons,
        "confirmed_hit_consecutive": consecutive,
        "selection_rule": {
            "objective": "minimize sum(abs(log(NOG_work/ME_work)))",
            "constraint": "5/5 pilot hits and NOG batch nondecreasing as epsilon decreases",
            "switch_penalty": 0.05,
            "purpose": "compare depth at approximately matched total oracle work",
        },
        "formal_success_criteria": {
            "minimum_primary_hits_of_20": 18,
            "minimum_depth_ratio_spearman": 0.7,
            "work_ratio_lower": 0.5,
            "work_ratio_upper": 2.0,
            "maximum_work_ratio_cv": 0.25,
        },
        "fixed_algorithm_parameters": {
            "NOG-FO": {"M": 2, "eta": 1.0, "smooth_B": 1, "rounds": 960},
            "ME-DOL-FO": {
                "epoch_length": 6,
                "theory_multiplier": 100.0,
                "rounds": 3840,
            },
        },
        "selected_schedule": [
            {
                "epsilon": float(row["epsilon"]),
                "data_B_total": int(row["data_B_total"]),
                "pilot_depth_ratio_me_over_nog": float(row["depth_ratio_me_over_nog"]),
                "pilot_work_ratio_nog_over_me": float(row["work_ratio_nog_over_me"]),
            }
            for row in selected
        ],
        "selected_batches": sorted(set(selected_batches)),
        "input_artifacts": [
            {"path": str(path), "sha256": file_sha256(path)} for path in input_paths
        ],
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "pilot_calibration.csv", calibration_rows)
    atomic_write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_theory_validation_v4.yaml"
    )
    parser.add_argument(
        "--batch-root",
        default=(
            "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/"
            "pilot/batch_grid"
        ),
    )
    parser.add_argument(
        "--me-root",
        default=(
            "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/pilot/"
            "fine_eval/ME-DOL-FO__epoch-6__mult-100__r3840"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/"
            "frozen_parameters.json"
        ),
    )
    args = parser.parse_args()
    result = freeze_parameters(
        load_config(args.config),
        Path(args.batch_root),
        Path(args.me_root),
        Path(args.output),
    )
    print(
        f"status={result['status']} epsilons={len(result['primary_epsilons'])} "
        f"batches={result['selected_batches']}"
    )


if __name__ == "__main__":
    main()
