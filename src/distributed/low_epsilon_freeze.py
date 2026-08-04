"""Freeze symmetric low-epsilon choices before independent formal runs."""

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
from src.distributed.low_epsilon_runner import (
    DEFAULT_ROOT,
    me_label,
    nog_label,
)


SCHEMA_VERSION = 2


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _completion_payloads(
    completion_path: Path, expected_seeds: List[int]
) -> tuple[Dict[str, Dict[int, Dict[str, Any]]], List[Path]]:
    completion = _load(completion_path)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError(f"Pilot is not complete and failure-free: {completion_path}")
    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    paths = [completion_path]
    for record in completion["records"]:
        label = str(record["label"])
        seed = int(record["task"]["formal_seed"])
        path = Path(record["partial_path"])
        grouped.setdefault(label, {})[seed] = _load(path)
        paths.append(path)
    expected = set(expected_seeds)
    if any(set(rows) != expected for rows in grouped.values()):
        raise ValueError("A pilot candidate is missing one or more pilot seeds.")
    return grouped, paths


def _algorithm_candidates(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ext = cfg["low_epsilon_extension"]
    rounds = int(ext["max_depth"])
    batch = int(ext["common_algorithm_selection_batch_total"])
    candidates: Dict[str, Dict[str, Any]] = {}
    for item in ext["algorithm_candidates"]["nog"]:
        candidate = {
            "method": "NOG-FO",
            "M": int(item["M"]),
            "eta": float(item["eta"]),
        }
        label = nog_label(candidate["M"], candidate["eta"], batch, rounds)
        candidates[label] = candidate
    for item in ext["algorithm_candidates"]["me_dol"]:
        candidate = {
            "method": "ME-DOL-FO",
            "epoch_length": int(item["epoch_length"]),
            "theory_multiplier": float(item["theory_multiplier"]),
        }
        label = me_label(
            candidate["epoch_length"],
            candidate["theory_multiplier"],
            batch,
            rounds,
        )
        candidates[label] = candidate
    return candidates


def _score_candidate(
    by_seed: Dict[int, Dict[str, Any]],
    seeds: List[int],
    epsilons: List[float],
    consecutive: int,
) -> tuple[tuple[Any, ...], List[Dict[str, Any]]]:
    prefix_len = 0
    prefix_open = True
    total_hits = 0
    hit_works: List[float] = []
    rows = []
    for epsilon in epsilons:
        hits = [
            confirmed_hit(by_seed[seed]["rows"], epsilon, consecutive)
            for seed in seeds
        ]
        good = [hit for hit in hits if hit is not None]
        total_hits += len(good)
        if prefix_open and len(good) == len(seeds):
            prefix_len += 1
        else:
            prefix_open = False
        depths = [float(hit["depth"]) for hit in good]
        works = [float(hit["total_work"]) for hit in good]
        hit_works.extend(works)
        rows.append(
            {
                "epsilon": epsilon,
                "hit_count": len(good),
                "mean_depth_hits": statistics.mean(depths) if depths else None,
                "mean_total_work_hits": statistics.mean(works) if works else None,
            }
        )
    mean_work = statistics.mean(hit_works) if hit_works else math.inf
    return (-prefix_len, -total_hits, mean_work), rows


def freeze_algorithms(
    cfg: Dict[str, Any], grid_root: Path, output_path: Path
) -> Dict[str, Any]:
    ext = cfg["low_epsilon_extension"]
    seeds = [int(value) for value in ext["pilot_seeds"]]
    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if set(seeds) & set(formal_seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    epsilons = [float(value) for value in ext["epsilons"]]
    consecutive = int(ext["confirmed_hit_consecutive"])
    completion_path = grid_root / "completion.json"
    payloads, input_paths = _completion_payloads(completion_path, seeds)
    candidates = _algorithm_candidates(cfg)
    if set(payloads) != set(candidates):
        raise ValueError("Observed algorithm labels differ from the frozen grid.")
    counts = {
        method: sum(c["method"] == method for c in candidates.values())
        for method in ["NOG-FO", "ME-DOL-FO"]
    }
    if counts["NOG-FO"] != counts["ME-DOL-FO"]:
        raise ValueError("Symmetric protocol requires equal candidate counts.")

    selected: Dict[str, Dict[str, Any]] = {}
    grid_rows: List[Dict[str, Any]] = []
    for method in ["NOG-FO", "ME-DOL-FO"]:
        scores = []
        for label, candidate in sorted(candidates.items()):
            if candidate["method"] != method:
                continue
            score, rows = _score_candidate(
                payloads[label], seeds, epsilons, consecutive
            )
            scores.append((*score, label))
            for row in rows:
                grid_rows.append({"method": method, "label": label, **row})
        selected_label = min(scores)[-1]
        selected[method] = {
            key: value
            for key, value in candidates[selected_label].items()
            if key != "method"
        }
        selected[method]["label"] = selected_label

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "stage": "algorithm_selection",
        "created_at_utc": utc_now(),
        "selection_happened_before_batch_pilot_and_formal": True,
        "pilot_seeds": seeds,
        "formal_seeds": formal_seeds,
        "candidate_count_per_method": counts,
        "common_max_depth": int(ext["max_depth"]),
        "max_recorded_depth_by_method": {"NOG-FO": int(ext["max_depth"]) + 2, "ME-DOL-FO": int(ext["max_depth"])},
        "common_eval_every": int(ext["common_eval_every"]),
        "common_batch_total": int(ext["common_algorithm_selection_batch_total"]),
        "selection_rule": str(ext["selection"]["algorithm_rule"]),
        "selected_algorithms": selected,
        "input_artifacts": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in input_paths
        ],
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "algorithm_pilot_grid.csv", grid_rows)
    atomic_write_json(output_path, manifest)
    return manifest


def _select_min_work_schedule(
    eligible_by_epsilon: List[List[Dict[str, Any]]], switch_penalty: float
) -> List[Dict[str, Any]]:
    states: Dict[int, tuple[float, List[Dict[str, Any]]]] = {}
    for index, eligible in enumerate(eligible_by_epsilon):
        min_work = min(float(row["mean_total_work_hits"]) for row in eligible)
        new_states: Dict[int, tuple[float, List[Dict[str, Any]]]] = {}
        for row in eligible:
            batch = int(row["batch_total"])
            local_cost = math.log(float(row["mean_total_work_hits"]) / min_work)
            if index == 0:
                new_states[batch] = (local_cost, [row])
                continue
            options = []
            for previous_batch, (cost, path) in states.items():
                if previous_batch <= batch:
                    switch = switch_penalty if previous_batch != batch else 0.0
                    options.append((cost + local_cost + switch, path + [row]))
            if options:
                new_states[batch] = min(options, key=lambda item: item[0])
        if not new_states:
            raise ValueError("No nondecreasing full-hit batch schedule exists.")
        states = new_states
    return min(states.values(), key=lambda item: item[0])[1]


def freeze_final(
    cfg: Dict[str, Any],
    algorithm_freeze_path: Path,
    batch_root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    formal_root = output_path.parent / "formal"
    if list(formal_root.glob("**/partials/*.json")):
        raise ValueError("Formal partials exist; refusing post-hoc freeze.")
    ext = cfg["low_epsilon_extension"]
    seeds = [int(value) for value in ext["pilot_seeds"]]
    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    epsilons = [float(value) for value in ext["epsilons"]]
    consecutive = int(ext["confirmed_hit_consecutive"])
    rounds = int(ext["max_depth"])
    algorithm_freeze = _load(algorithm_freeze_path)
    if algorithm_freeze.get("status") != "frozen":
        raise ValueError("Algorithm parameters have not been frozen.")
    selected = algorithm_freeze["selected_algorithms"]
    completion_path = batch_root / "completion.json"
    payloads, input_paths = _completion_payloads(completion_path, seeds)

    expected_labels: Dict[str, Dict[int, str]] = {"NOG-FO": {}, "ME-DOL-FO": {}}
    for batch_value in ext["batch_total_candidates"]:
        batch = int(batch_value)
        expected_labels["NOG-FO"][batch] = nog_label(
            selected["NOG-FO"]["M"], selected["NOG-FO"]["eta"], batch, rounds
        )
        expected_labels["ME-DOL-FO"][batch] = me_label(
            selected["ME-DOL-FO"]["epoch_length"],
            selected["ME-DOL-FO"]["theory_multiplier"],
            batch,
            rounds,
        )
    if set(payloads) != {
        label for labels in expected_labels.values() for label in labels.values()
    }:
        raise ValueError("Observed batch labels differ from the symmetric grid.")

    calibration_rows: List[Dict[str, Any]] = []
    eligible: Dict[str, List[List[Dict[str, Any]]]] = {
        "NOG-FO": [],
        "ME-DOL-FO": [],
    }
    primary_epsilons: List[float] = []
    for epsilon in epsilons:
        current: Dict[str, List[Dict[str, Any]]] = {}
        for method in ["NOG-FO", "ME-DOL-FO"]:
            current[method] = []
            for batch, label in sorted(expected_labels[method].items()):
                hits = [
                    confirmed_hit(payloads[label][seed]["rows"], epsilon, consecutive)
                    for seed in seeds
                ]
                good = [hit for hit in hits if hit is not None]
                row = {
                    "method": method,
                    "epsilon": epsilon,
                    "batch_total": batch,
                    "hit_count": len(good),
                    "mean_depth_hits": (
                        statistics.mean(float(hit["depth"]) for hit in good)
                        if good
                        else None
                    ),
                    "mean_total_work_hits": (
                        statistics.mean(float(hit["total_work"]) for hit in good)
                        if good
                        else None
                    ),
                    "eligible_full_hit": len(good) == len(seeds),
                }
                calibration_rows.append(row)
                if row["eligible_full_hit"]:
                    current[method].append(row)
        if not current["NOG-FO"] or not current["ME-DOL-FO"]:
            break
        primary_epsilons.append(epsilon)
        for method in current:
            eligible[method].append(current[method])
    if len(primary_epsilons) < 2:
        raise ValueError("Pilot did not establish a shared low-epsilon interval.")

    schedules = {
        method: _select_min_work_schedule(
            eligible[method], float(ext["selection"]["switch_penalty"])
        )
        for method in ["NOG-FO", "ME-DOL-FO"]
    }
    selected_batches = {
        method: sorted({int(row["batch_total"]) for row in rows})
        for method, rows in schedules.items()
    }
    schedule_rows = []
    for index, epsilon in enumerate(primary_epsilons):
        schedule_rows.append(
            {
                "epsilon": epsilon,
                "NOG-FO": {
                    key: schedules["NOG-FO"][index][key]
                    for key in ["batch_total", "mean_depth_hits", "mean_total_work_hits"]
                },
                "ME-DOL-FO": {
                    key: schedules["ME-DOL-FO"][index][key]
                    for key in ["batch_total", "mean_depth_hits", "mean_total_work_hits"]
                },
            }
        )
    criteria = ext["formal_success_criteria"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "stage": "final",
        "created_at_utc": utc_now(),
        "selection_happened_before_formal_runs": True,
        "pilot_seeds": seeds,
        "formal_seeds": formal_seeds,
        "seed_sets_disjoint": not bool(set(seeds) & set(formal_seeds)),
        "all_requested_epsilons": epsilons,
        "primary_epsilons": primary_epsilons,
        "exploratory_censored_epsilons": epsilons[len(primary_epsilons) :],
        "confirmed_hit_consecutive": consecutive,
        "common_max_depth": rounds,
        "max_recorded_depth_by_method": {"NOG-FO": rounds + 2, "ME-DOL-FO": rounds},
        "common_eval_every": int(ext["common_eval_every"]),
        "selected_algorithms": {
            method: {key: value for key, value in values.items() if key != "label"}
            for method, values in selected.items()
        },
        "algorithm_freeze_sha256": file_sha256(algorithm_freeze_path),
        "batch_selection_rule": str(ext["selection"]["batch_rule"]),
        "selected_schedule": schedule_rows,
        "selected_batches": selected_batches,
        "formal_success_criteria": criteria,
        "anomaly_confirmation": ext["anomaly_confirmation"],
        "input_artifacts": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in [algorithm_freeze_path, *input_paths]
        ],
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "batch_pilot_grid.csv", calibration_rows)
    atomic_write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["freeze-algorithms", "freeze-final"]
    )
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_low_epsilon_v5.yaml"
    )
    parser.add_argument(
        "--algorithm-grid", default=str(DEFAULT_ROOT / "pilot" / "algorithm_grid")
    )
    parser.add_argument(
        "--batch-grid", default=str(DEFAULT_ROOT / "pilot" / "batch_grid")
    )
    parser.add_argument(
        "--algorithm-freeze", default=str(DEFAULT_ROOT / "algorithm_freeze.json")
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_ROOT / "frozen_parameters.json")
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "freeze-algorithms":
        result = freeze_algorithms(
            cfg, Path(args.algorithm_grid), Path(args.algorithm_freeze)
        )
    else:
        result = freeze_final(
            cfg,
            Path(args.algorithm_freeze),
            Path(args.batch_grid),
            Path(args.output),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
