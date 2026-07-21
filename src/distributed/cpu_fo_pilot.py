"""Auditable staged pilot tuning for the real CPU-process FO comparison.

``prepare`` is safe: it validates the preregistered grid and writes only a
manifest.  ``coarse`` is deliberately separate because it launches the long
960-round process grid.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.distributed.common import validate_experiment_config
from src.distributed.cpu_fo_correctness import (
    atomic_write_csv,
    compare_trajectories,
    load_config,
    process_config_from_experiment,
)
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    effective_task_config,
    object_sha256,
    run_task_set,
    utc_now,
)


PILOT_SCHEMA_VERSION = 1
SELECTION_RULE_VERSION = "confirmed2-full-hit-pareto-time-depth-v1"


def _slug(value: Any) -> str:
    return str(value).replace(".", "p").replace("+", "plus").replace("-", "m")


def candidate_id(method: str, parameters: Dict[str, Any]) -> str:
    pieces = [method.replace("+", "plus")]
    pieces.extend(f"{key}-{_slug(value)}" for key, value in sorted(parameters.items()))
    return "__".join(pieces)


def coarse_candidates(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pilot = cfg["pilot"]
    candidates: List[Dict[str, Any]] = []
    for block in pilot["nog"]["M"]:
        for eta in pilot["nog"]["eta"]:
            parameters = {
                "M": int(block),
                "eta": float(eta),
                "smooth_B": int(pilot["nog"]["coarse_smooth_B"]),
                "data_B_total": int(cfg["oracle"]["data_B_total"]),
            }
            candidates.append(
                {
                    "candidate_id": candidate_id("NOG-FO", parameters),
                    "method": "NOG-FO",
                    "phase": "coarse",
                    "parameters": parameters,
                }
            )
    for epoch in pilot["me_dol"]["epoch_length"]:
        for multiplier in pilot["me_dol"]["theory_multiplier"]:
            parameters = {
                "epoch_length": int(epoch),
                "theory_multiplier": float(multiplier),
            }
            candidates.append(
                {
                    "candidate_id": candidate_id("ME-DOL-FO", parameters),
                    "method": "ME-DOL-FO",
                    "phase": "coarse",
                    "parameters": parameters,
                }
            )
    return candidates


def candidate_config(
    base_cfg: Dict[str, Any],
    candidate: Dict[str, Any],
    rounds: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["rounds"] = int(rounds)
    parameters = candidate["parameters"]
    if candidate["method"] == "NOG-FO":
        cfg["nog"]["M"] = int(parameters["M"])
        cfg["nog"]["eta"] = float(parameters["eta"])
        cfg["oracle"]["smooth_B"] = int(parameters["smooth_B"])
        cfg["oracle"]["data_B_total"] = int(parameters["data_B_total"])
    elif candidate["method"] == "ME-DOL-FO":
        cfg["me_dol"]["epoch_length"] = int(parameters["epoch_length"])
        cfg["me_dol"]["theory_multiplier"] = float(
            parameters["theory_multiplier"]
        )
    else:
        raise ValueError(f"Unknown candidate method: {candidate['method']}.")
    validate_experiment_config(cfg)
    stride = (
        int(cfg["nog"]["M"])
        if candidate["method"] == "NOG-FO"
        else int(cfg["me_dol"]["epoch_length"])
    )
    if rounds % stride != 0:
        raise ValueError(f"rounds={rounds} is not divisible by stride={stride}.")
    return cfg


def validate_pilot_config(cfg: Dict[str, Any]) -> None:
    validate_experiment_config(cfg)
    pilot = cfg["pilot"]
    pilot_seeds = {int(value) for value in pilot["seeds"]}
    formal_seeds = {int(value) for value in cfg["run"]["formal_seeds"]}
    overlap = pilot_seeds & formal_seeds
    if overlap:
        raise ValueError(f"Pilot/formal seeds overlap: {sorted(overlap)}.")
    if cfg["oracle"].get("evaluation_seed_mode") != "fixed_bank":
        raise ValueError("Pilot requires oracle.evaluation_seed_mode=fixed_bank.")
    if cfg.get("wandb", {}).get("enabled", False):
        raise ValueError("W&B must remain disabled until pilot selection is frozen.")
    worker = int(pilot["reference_worker"])
    if worker != int(cfg["distributed"]["comparison_worker"]):
        raise ValueError("Pilot reference_worker must equal comparison_worker.")
    if worker > 32:
        raise ValueError("Pilot worker count exceeds the Step-5 frozen maximum 32.")
    epsilons = [float(value) for value in pilot["epsilons"]]
    if not epsilons or any(value <= 0.0 for value in epsilons):
        raise ValueError("Pilot epsilons must be positive.")
    if int(pilot["confirmed_hit_consecutive"]) < 2:
        raise ValueError("Confirmed hit must require at least two checkpoints.")
    budgets = [int(value) for value in pilot["budgets"]]
    if budgets != sorted(set(budgets)):
        raise ValueError("Pilot budgets must be unique and increasing.")
    for candidate in coarse_candidates(cfg):
        for rounds in budgets:
            candidate_config(cfg, candidate, rounds)


def prepare_manifest(cfg: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    validate_pilot_config(cfg)
    root = Path(output_root)
    candidates = coarse_candidates(cfg)
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    counts = {
        method: sum(candidate["method"] == method for candidate in candidates)
        for method in ("NOG-FO", "ME-DOL-FO")
    }
    task_counts = {method: count * len(seeds) for method, count in counts.items()}
    estimates = cfg["pilot"].get("estimated_960_seconds", {})
    estimated_seconds = sum(
        task_counts[method] * float(estimates.get(method, 0.0))
        for method in task_counts
    )
    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "pilot_seeds": seeds,
        "formal_seeds_excluded": [int(v) for v in cfg["run"]["formal_seeds"]],
        "reference_worker": int(cfg["pilot"]["reference_worker"]),
        "epsilons": [float(v) for v in cfg["pilot"]["epsilons"]],
        "budgets": [int(v) for v in cfg["pilot"]["budgets"]],
        "evaluation_seed_mode": cfg["oracle"]["evaluation_seed_mode"],
        "confirmed_hit_consecutive": int(
            cfg["pilot"]["confirmed_hit_consecutive"]
        ),
        "candidate_count": len(candidates),
        "candidate_count_by_method": counts,
        "coarse_task_count": len(candidates) * len(seeds),
        "coarse_task_count_by_method": task_counts,
        "estimated_coarse_seconds": estimated_seconds,
        "estimated_coarse_hours": estimated_seconds / 3600.0,
        "candidate_grid_sha256": object_sha256(candidates),
        "effective_config_sha256": object_sha256(effective_task_config(cfg)),
        "candidates": candidates,
        "launches_started": 0,
        "notes": [
            "One trajectory is reused for all epsilon thresholds.",
            "The prepare phase launches no worker process.",
            "Only eligible/Pareto candidates may advance beyond 960 rounds.",
        ],
    }
    atomic_write_json(root / "config_used.json", effective_task_config(cfg))
    atomic_write_json(root / "candidate_manifest.json", manifest)
    return manifest


def confirmed_hit(
    rows: Iterable[Dict[str, Any]],
    epsilon: float,
    consecutive: int = 2,
) -> Dict[str, Any] | None:
    ordered = sorted(rows, key=lambda row: int(row["iteration"]))
    run_length = 0
    for index, row in enumerate(ordered):
        value = float(row["stat_proxy"])
        run_length = run_length + 1 if math.isfinite(value) and value <= epsilon else 0
        if run_length >= consecutive:
            first = ordered[index - consecutive + 1]
            return {
                "iteration": int(first["iteration"]),
                "depth": int(first["depth"]),
                "total_work": int(first["total_work"]),
                "per_worker_work": int(first["per_worker_work_max"]),
                "training_time": float(first["training_time"]),
                "stat_proxy": float(first["stat_proxy"]),
            }
    return None


def pareto_frontier(
    rows: List[Dict[str, Any]],
    metrics: tuple[str, ...] = ("median_depth", "median_total_work", "median_training_time"),
) -> List[Dict[str, Any]]:
    frontier = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and all(float(other[key]) <= float(candidate[key]) for key in metrics)
            and any(float(other[key]) < float(candidate[key]) for key in metrics)
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def summarize_candidate(
    candidate: Dict[str, Any],
    payloads: Iterable[Dict[str, Any]],
    epsilon: float,
    consecutive: int,
) -> Dict[str, Any]:
    payload_list = list(payloads)
    hits = [confirmed_hit(payload["rows"], epsilon, consecutive) for payload in payload_list]
    observed = [hit for hit in hits if hit is not None]
    finals = [float(payload["rows"][-1]["stat_proxy"]) for payload in payload_list]
    row: Dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "epsilon": float(epsilon),
        "num_seeds": len(payload_list),
        "hit_count": len(observed),
        "hit_rate": len(observed) / len(payload_list),
        "full_hit": len(observed) == len(payload_list),
        "median_final_stat_proxy": statistics.median(finals),
        "parameters": json.dumps(candidate["parameters"], sort_keys=True),
    }
    for output, key in (
        ("median_depth", "depth"),
        ("median_total_work", "total_work"),
        ("median_per_worker_work", "per_worker_work"),
        ("median_training_time", "training_time"),
    ):
        row[output] = statistics.median(hit[key] for hit in observed) if observed else math.inf
    return row


def seed_hit_rows(
    candidate: Dict[str, Any],
    payloads: Iterable[Dict[str, Any]],
    epsilon: float,
    consecutive: int,
) -> List[Dict[str, Any]]:
    rows = []
    for payload in payloads:
        hit = confirmed_hit(payload["rows"], epsilon, consecutive)
        final = payload["rows"][-1]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "method": candidate["method"],
                "epsilon": float(epsilon),
                "formal_seed": int(payload["formal_seed"]),
                "hit": hit is not None,
                "censored": hit is None,
                "first_hit_iteration": hit["iteration"] if hit else None,
                "first_hit_depth": hit["depth"] if hit else None,
                "first_hit_total_work": hit["total_work"] if hit else None,
                "first_hit_per_worker_work": (
                    hit["per_worker_work"] if hit else None
                ),
                "first_hit_training_time": hit["training_time"] if hit else None,
                "first_hit_stat_proxy": hit["stat_proxy"] if hit else None,
                "final_iteration": int(final["iteration"]),
                "final_depth": int(final["depth"]),
                "final_total_work": int(final["total_work"]),
                "final_stat_proxy": float(final["stat_proxy"]),
                "parameters": json.dumps(candidate["parameters"], sort_keys=True),
            }
        )
    return rows


def select_for_epsilon(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select from an empty candidate list.")
    best_hit_rate = max(float(row["hit_rate"]) for row in rows)
    hit_eligible = [row for row in rows if float(row["hit_rate"]) == best_hit_rate]
    full = [row for row in hit_eligible if bool(row["full_hit"])]
    if full:
        frontier = pareto_frontier(full)
        selected = min(
            frontier,
            key=lambda row: (
                float(row["median_training_time"]),
                float(row["median_depth"]),
                float(row["median_total_work"]),
                row["candidate_id"],
            ),
        )
        status = "full-hit"
    else:
        frontier = []
        selected = min(
            hit_eligible,
            key=lambda row: (
                float(row["median_final_stat_proxy"]),
                row["candidate_id"],
            ),
        )
        status = "pilot-infeasible-censored"
    finite_hit_eligible = [
        row
        for row in hit_eligible
        if math.isfinite(float(row["median_depth"]))
        and math.isfinite(float(row["median_total_work"]))
    ]
    return {
        "status": status,
        "selected_candidate_id": selected["candidate_id"],
        "frontier_candidate_ids": [row["candidate_id"] for row in frontier],
        "min_depth_candidate_id": (
            min(
                finite_hit_eligible,
                key=lambda row: (float(row["median_depth"]), row["candidate_id"]),
            )["candidate_id"]
            if finite_hit_eligible
            else None
        ),
        "min_work_candidate_id": (
            min(
                finite_hit_eligible,
                key=lambda row: (
                    float(row["median_total_work"]),
                    row["candidate_id"],
                ),
            )["candidate_id"]
            if finite_hit_eligible
            else None
        ),
    }


def advancement_candidates(
    grouped_rows: Dict[tuple[str, float], List[Dict[str, Any]]],
    top_n_censored: int,
) -> tuple[set[str], Dict[str, List[str]]]:
    """Choose 1920-round candidates without using formal seeds.

    Full-hit groups advance their entire Pareto frontier.  If a threshold has
    no full-hit candidate, only its best ``top_n_censored`` candidates advance,
    ordered by hit rate and final stationarity.  This prevents a blind
    extension of every clearly dominated failed configuration.
    """

    advancing: set[str] = set()
    reasons: Dict[str, List[str]] = {}
    for (method, epsilon), rows in sorted(grouped_rows.items()):
        selection = select_for_epsilon(rows)
        if selection["status"] == "full-hit":
            eligible = list(selection["frontier_candidate_ids"])
            reason = "pareto-frontier"
        else:
            eligible = [
                row["candidate_id"]
                for row in sorted(
                    rows,
                    key=lambda row: (
                        -float(row["hit_rate"]),
                        float(row["median_final_stat_proxy"]),
                        row["candidate_id"],
                    ),
                )[:top_n_censored]
            ]
            reason = "top-censored"
        for identifier in eligible:
            advancing.add(identifier)
            reasons.setdefault(identifier, []).append(
                f"{method}:epsilon={epsilon:g}:{reason}"
            )
    return advancing, reasons


def _load_coarse_payloads(
    root: Path,
    candidates: List[Dict[str, Any]],
    rounds: int,
    seeds: List[int],
) -> Dict[str, List[Dict[str, Any]]]:
    coarse_completion_path = root / "coarse_completion.json"
    if not coarse_completion_path.exists():
        raise FileNotFoundError("coarse_completion.json is missing.")
    with open(coarse_completion_path, "r", encoding="utf-8") as handle:
        coarse_completion = json.load(handle)
    if coarse_completion.get("status") != "complete":
        raise ValueError("Coarse sweep is not complete.")

    expected_seeds = sorted(seeds)
    payloads_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_root = (
            root / "coarse" / candidate["candidate_id"] / f"rounds_{rounds}"
        )
        completion_path = candidate_root / "completion_manifest.json"
        with open(completion_path, "r", encoding="utf-8") as handle:
            completion = json.load(handle)
        if completion.get("status") != "complete":
            raise ValueError(f"Candidate is incomplete: {candidate['candidate_id']}.")
        payloads = []
        for record in completion["records"]:
            if record["status"] not in {"completed", "resumed", "recovered"}:
                raise ValueError(
                    f"Invalid task status for {candidate['candidate_id']}: "
                    f"{record['status']}."
                )
            with open(
                candidate_root / record["partial_path"],
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
            if payload["method"] != candidate["method"]:
                raise ValueError("Partial method does not match candidate method.")
            payloads.append(payload)
        observed_seeds = sorted(int(payload["formal_seed"]) for payload in payloads)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"Pilot seeds mismatch for {candidate['candidate_id']}: "
                f"{observed_seeds} != {expected_seeds}."
            )
        payloads_by_candidate[candidate["candidate_id"]] = payloads
    return payloads_by_candidate


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_coarse(cfg: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    """Summarize the completed 960-round sweep without launching processes."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    candidates = coarse_candidates(cfg)
    candidate_lookup = {row["candidate_id"]: row for row in candidates}
    rounds = int(cfg["pilot"]["budgets"][0])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    epsilons = [float(value) for value in cfg["pilot"]["epsilons"]]
    consecutive = int(cfg["pilot"]["confirmed_hit_consecutive"])
    payloads = _load_coarse_payloads(root, candidates, rounds, seeds)

    candidate_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_payloads = payloads[candidate["candidate_id"]]
        for epsilon in epsilons:
            summary = summarize_candidate(
                candidate,
                candidate_payloads,
                epsilon,
                consecutive,
            )
            candidate_rows.append(summary)
            grouped.setdefault((candidate["method"], epsilon), []).append(summary)
            per_seed_rows.extend(
                seed_hit_rows(
                    candidate,
                    candidate_payloads,
                    epsilon,
                    consecutive,
                )
            )

    selections: Dict[str, Dict[str, Any]] = {}
    pareto_rows: List[Dict[str, Any]] = []
    for (method, epsilon), rows in sorted(grouped.items()):
        selected = select_for_epsilon(rows)
        frontier_ids = set(selected["frontier_candidate_ids"])
        for row in rows:
            row["pareto_frontier"] = row["candidate_id"] in frontier_ids
            row["selected"] = row["candidate_id"] == selected["selected_candidate_id"]
            row["selection_status"] = selected["status"]
            if row["pareto_frontier"]:
                pareto_rows.append(dict(row))
        selected_row = next(
            row
            for row in rows
            if row["candidate_id"] == selected["selected_candidate_id"]
        )
        epsilon_key = f"{epsilon:g}"
        selections.setdefault(epsilon_key, {})[method] = {
            **selected,
            "selected_parameters": candidate_lookup[
                selected["selected_candidate_id"]
            ]["parameters"],
            "selected_hit_rate": float(selected_row["hit_rate"]),
            "selected_median_final_stat_proxy": float(
                selected_row["median_final_stat_proxy"]
            ),
            "selected_median_depth": _finite_or_none(
                selected_row["median_depth"]
            ),
            "selected_median_total_work": _finite_or_none(
                selected_row["median_total_work"]
            ),
            "selected_median_training_time": _finite_or_none(
                selected_row["median_training_time"]
            ),
        }

    advancing_ids, advancement_reasons = advancement_candidates(
        grouped,
        int(cfg["pilot"]["top_n_refinement"]),
    )
    advancement_records = [
        {
            **candidate,
            "reasons": advancement_reasons[candidate["candidate_id"]],
        }
        for candidate in candidates
        if candidate["candidate_id"] in advancing_ids
    ]
    status_counts: Dict[str, int] = {}
    for by_method in selections.values():
        for selection in by_method.values():
            status = selection["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "complete",
        "analysis_stage": "coarse-960",
        "created_at_utc": utc_now(),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "advancement_rule_version": "full-hit-pareto-or-top3-censored-v1",
        "rounds": rounds,
        "candidate_count": len(candidates),
        "task_count": len(candidates) * len(seeds),
        "candidate_epsilon_rows": len(candidate_rows),
        "per_seed_epsilon_rows": len(per_seed_rows),
        "selection_status_counts": status_counts,
        "advancing_candidate_count": len(advancement_records),
        "advancing_candidate_count_by_method": {
            method: sum(row["method"] == method for row in advancement_records)
            for method in ("NOG-FO", "ME-DOL-FO")
        },
        "selections": selections,
        "advancement_to_1920": advancement_records,
        "launches_started": 0,
    }
    atomic_write_csv(root / "pilot_hits_per_seed.csv", per_seed_rows)
    atomic_write_csv(root / "candidate_results.csv", candidate_rows)
    atomic_write_csv(root / "pareto_by_epsilon.csv", pareto_rows)
    atomic_write_json(root / "selected_coarse_by_epsilon.yaml", selections)
    atomic_write_json(
        root / "advancement_to_1920.json",
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "status": "prepared",
            "source_rounds": rounds,
            "target_rounds": int(cfg["pilot"]["budgets"][1]),
            "rule_version": report["advancement_rule_version"],
            "candidate_count": len(advancement_records),
            "candidates": advancement_records,
            "launches_started": 0,
        },
    )
    atomic_write_json(root / "coarse_analysis_report.json", report)
    return report


def validate_advancement_manifest(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if manifest.get("status") != "prepared":
        raise ValueError("Advancement manifest must have status=prepared.")
    budgets = [int(value) for value in cfg["pilot"]["budgets"]]
    if int(manifest.get("source_rounds", -1)) != budgets[0]:
        raise ValueError("Advancement source_rounds does not match pilot budget.")
    if int(manifest.get("target_rounds", -1)) != budgets[1]:
        raise ValueError("Advancement target_rounds does not match pilot budget.")
    if int(manifest.get("launches_started", -1)) != 0:
        raise ValueError("Advancement manifest must be frozen before launch.")
    expected = {row["candidate_id"]: row for row in coarse_candidates(cfg)}
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Advancement manifest has no candidates.")
    observed_ids = [row.get("candidate_id") for row in candidates]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Advancement manifest contains duplicate candidate IDs.")
    if int(manifest.get("candidate_count", -1)) != len(candidates):
        raise ValueError("Advancement candidate_count does not match candidates.")
    for candidate in candidates:
        identifier = candidate.get("candidate_id")
        if identifier not in expected:
            raise ValueError(f"Unknown advancement candidate: {identifier}.")
        canonical = expected[identifier]
        if candidate.get("method") != canonical["method"]:
            raise ValueError(f"Method mismatch for advancement candidate {identifier}.")
        if candidate.get("parameters") != canonical["parameters"]:
            raise ValueError(
                f"Parameter mismatch for advancement candidate {identifier}."
            )
    return candidates


def run_extension_1920(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    """Run only the candidates frozen by the coarse-analysis gate."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    manifest_path = root / "advancement_to_1920.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        advancement = json.load(handle)
    candidates = validate_advancement_manifest(cfg, advancement)
    rounds = int(advancement["target_rounds"])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    worker = int(cfg["pilot"]["reference_worker"])
    process = process_config_from_experiment(cfg)
    completion_records = []
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"extension_candidate={index}/{len(candidates)} "
            f"id={candidate['candidate_id']} status=starting",
            flush=True,
        )
        current_cfg = candidate_config(cfg, candidate, rounds)
        candidate_root = (
            root / "extension" / f"rounds_{rounds}" / candidate["candidate_id"]
        )
        completion = run_task_set(
            current_cfg,
            [CpuFoTask(candidate["method"], seed, worker) for seed in seeds],
            candidate_root,
            process,
            continue_on_error=True,
        )
        record = {
            "candidate_id": candidate["candidate_id"],
            "method": candidate["method"],
            "status": completion["status"],
            "completed_tasks": completion["completed_tasks"],
            "failed_tasks": completion["failed_tasks"],
            "output_root": str(candidate_root.relative_to(root)),
        }
        completion_records.append(record)
        print(
            f"extension_candidate={index}/{len(candidates)} "
            f"id={candidate['candidate_id']} status={record['status']} "
            f"tasks={record['completed_tasks']}/3 failures={record['failed_tasks']}",
            flush=True,
        )
    result = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": (
            "complete"
            if all(record["status"] == "complete" for record in completion_records)
            else "incomplete"
        ),
        "source_manifest": str(manifest_path.relative_to(root)),
        "rounds": rounds,
        "candidate_count": len(candidates),
        "expected_tasks": len(candidates) * len(seeds),
        "completed_tasks": sum(row["completed_tasks"] for row in completion_records),
        "failed_tasks": sum(row["failed_tasks"] for row in completion_records),
        "records": completion_records,
    }
    atomic_write_json(root / "extension_1920_completion.json", result)
    atomic_write_csv(root / "extension_1920_completion.csv", completion_records)
    return result


def unresolved_advancement_candidates(
    grouped_rows: Dict[tuple[str, float], List[Dict[str, Any]]],
    top_n_censored: int,
) -> tuple[set[str], Dict[str, List[str]]]:
    """Advance only candidates needed by a threshold with no full-hit config."""

    advancing: set[str] = set()
    reasons: Dict[str, List[str]] = {}
    for (method, epsilon), rows in sorted(grouped_rows.items()):
        selection = select_for_epsilon(rows)
        if selection["status"] == "full-hit":
            continue
        eligible = [
            row["candidate_id"]
            for row in sorted(
                rows,
                key=lambda row: (
                    -float(row["hit_rate"]),
                    float(row["median_final_stat_proxy"]),
                    row["candidate_id"],
                ),
            )[:top_n_censored]
        ]
        for identifier in eligible:
            advancing.add(identifier)
            reasons.setdefault(identifier, []).append(
                f"{method}:epsilon={epsilon:g}:unresolved-top-censored"
            )
    return advancing, reasons


def _load_extension_payloads(
    root: Path,
    candidates: List[Dict[str, Any]],
    rounds: int,
    seeds: List[int],
) -> Dict[str, List[Dict[str, Any]]]:
    completion_path = root / "extension_1920_completion.json"
    with open(completion_path, "r", encoding="utf-8") as handle:
        stage_completion = json.load(handle)
    if stage_completion.get("status") != "complete":
        raise ValueError("1920-round extension is not complete.")
    if int(stage_completion.get("rounds", -1)) != rounds:
        raise ValueError("Extension completion rounds do not match analysis rounds.")

    expected_seeds = sorted(seeds)
    payloads_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_root = root / "extension" / f"rounds_{rounds}" / candidate[
            "candidate_id"
        ]
        with open(
            candidate_root / "completion_manifest.json",
            "r",
            encoding="utf-8",
        ) as handle:
            completion = json.load(handle)
        if completion.get("status") != "complete":
            raise ValueError(
                f"Extension candidate is incomplete: {candidate['candidate_id']}."
            )
        payloads = []
        for record in completion["records"]:
            with open(
                candidate_root / record["partial_path"],
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)
            if payload["method"] != candidate["method"]:
                raise ValueError("Extension partial method mismatch.")
            payloads.append(payload)
        observed_seeds = sorted(int(payload["formal_seed"]) for payload in payloads)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"Extension seeds mismatch for {candidate['candidate_id']}: "
                f"{observed_seeds} != {expected_seeds}."
            )
        payloads_by_candidate[candidate["candidate_id"]] = payloads
    return payloads_by_candidate


def _prefix_consistency_audit(
    candidates: List[Dict[str, Any]],
    coarse_payloads: Dict[str, List[Dict[str, Any]]],
    extension_payloads: Dict[str, List[Dict[str, Any]]],
    prefix_rounds: int,
) -> List[Dict[str, Any]]:
    audit_rows = []
    for candidate in candidates:
        identifier = candidate["candidate_id"]
        coarse_by_seed = {
            int(payload["formal_seed"]): payload
            for payload in coarse_payloads[identifier]
        }
        extension_by_seed = {
            int(payload["formal_seed"]): payload
            for payload in extension_payloads[identifier]
        }
        for seed, coarse in sorted(coarse_by_seed.items()):
            extension = extension_by_seed[seed]
            raw_prefix_rows = [
                row
                for row in extension["rows"]
                if int(row["iteration"]) <= prefix_rounds
            ]
            extension_iterations = {
                int(row["iteration"]) for row in raw_prefix_rows
            }
            shared_coarse_rows = [
                row
                for row in coarse["rows"]
                if int(row["iteration"]) in extension_iterations
            ]
            shared_iterations = {
                int(row["iteration"]) for row in shared_coarse_rows
            }
            shared_extension_rows = [
                row
                for row in raw_prefix_rows
                if int(row["iteration"]) in shared_iterations
            ]
            coarse_only_iterations = sorted(
                int(row["iteration"])
                for row in coarse["rows"]
                if int(row["iteration"]) not in extension_iterations
            )
            expected_boundary_difference = coarse_only_iterations in (
                [],
                [prefix_rounds],
            )
            numerical_passed, max_difference, errors = compare_trajectories(
                shared_coarse_rows,
                shared_extension_rows,
                rel_tol=1.0e-5,
                abs_tol=1.0e-7,
            )
            if not expected_boundary_difference:
                errors.append(
                    "unexpected coarse-only checkpoints: "
                    f"{coarse_only_iterations}"
                )
            passed = numerical_passed and expected_boundary_difference
            audit_rows.append(
                {
                    "candidate_id": identifier,
                    "method": candidate["method"],
                    "formal_seed": seed,
                    "prefix_rounds": prefix_rounds,
                    "coarse_checkpoint_count": len(coarse["rows"]),
                    "extension_prefix_checkpoint_count": len(raw_prefix_rows),
                    "shared_checkpoint_count": len(shared_coarse_rows),
                    "coarse_only_iterations": json.dumps(
                        coarse_only_iterations
                    ),
                    "expected_boundary_difference": expected_boundary_difference,
                    "passed": passed,
                    "max_abs_difference": max_difference,
                    "errors": " | ".join(errors),
                }
            )
    return audit_rows


def analyze_extension_1920(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    """Audit and summarize the staged 1920-round extension."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    with open(root / "advancement_to_1920.json", "r", encoding="utf-8") as handle:
        advancement = json.load(handle)
    candidates = validate_advancement_manifest(cfg, advancement)
    candidate_lookup = {row["candidate_id"]: row for row in candidates}
    source_rounds = int(advancement["source_rounds"])
    rounds = int(advancement["target_rounds"])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    epsilons = [float(value) for value in cfg["pilot"]["epsilons"]]
    consecutive = int(cfg["pilot"]["confirmed_hit_consecutive"])
    coarse_payloads = _load_coarse_payloads(root, candidates, source_rounds, seeds)
    payloads = _load_extension_payloads(root, candidates, rounds, seeds)
    prefix_audit = _prefix_consistency_audit(
        candidates,
        coarse_payloads,
        payloads,
        source_rounds,
    )
    atomic_write_csv(root / "prefix_consistency_960_vs_1920.csv", prefix_audit)
    if not all(row["passed"] for row in prefix_audit):
        raise RuntimeError("960/1920 deterministic prefix consistency audit failed.")

    candidate_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_payloads = payloads[candidate["candidate_id"]]
        for epsilon in epsilons:
            summary = summarize_candidate(
                candidate,
                candidate_payloads,
                epsilon,
                consecutive,
            )
            candidate_rows.append(summary)
            grouped.setdefault((candidate["method"], epsilon), []).append(summary)
            per_seed_rows.extend(
                seed_hit_rows(candidate, candidate_payloads, epsilon, consecutive)
            )

    selections: Dict[str, Dict[str, Any]] = {}
    pareto_rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    for (method, epsilon), rows in sorted(grouped.items()):
        selected = select_for_epsilon(rows)
        status_counts[selected["status"]] = (
            status_counts.get(selected["status"], 0) + 1
        )
        frontier_ids = set(selected["frontier_candidate_ids"])
        for row in rows:
            row["pareto_frontier"] = row["candidate_id"] in frontier_ids
            row["selected"] = row["candidate_id"] == selected["selected_candidate_id"]
            row["selection_status"] = selected["status"]
            if row["pareto_frontier"]:
                pareto_rows.append(dict(row))
        selected_row = next(
            row
            for row in rows
            if row["candidate_id"] == selected["selected_candidate_id"]
        )
        selections.setdefault(f"{epsilon:g}", {})[method] = {
            **selected,
            "selected_parameters": candidate_lookup[
                selected["selected_candidate_id"]
            ]["parameters"],
            "selected_hit_rate": float(selected_row["hit_rate"]),
            "selected_median_final_stat_proxy": float(
                selected_row["median_final_stat_proxy"]
            ),
            "selected_median_depth": _finite_or_none(
                selected_row["median_depth"]
            ),
            "selected_median_total_work": _finite_or_none(
                selected_row["median_total_work"]
            ),
            "selected_median_training_time": _finite_or_none(
                selected_row["median_training_time"]
            ),
        }

    advancing_ids, reasons = unresolved_advancement_candidates(
        grouped,
        int(cfg["pilot"]["top_n_refinement"]),
    )
    advancement_records = [
        {
            **candidate,
            "reasons": reasons[candidate["candidate_id"]],
        }
        for candidate in candidates
        if candidate["candidate_id"] in advancing_ids
    ]
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "complete",
        "analysis_stage": "extension-1920",
        "created_at_utc": utc_now(),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "advancement_rule_version": "unresolved-threshold-top3-censored-v1",
        "rounds": rounds,
        "candidate_count": len(candidates),
        "task_count": len(candidates) * len(seeds),
        "candidate_epsilon_rows": len(candidate_rows),
        "per_seed_epsilon_rows": len(per_seed_rows),
        "prefix_audit_rows": len(prefix_audit),
        "prefix_audit_passed": all(row["passed"] for row in prefix_audit),
        "prefix_max_abs_difference": max(
            float(row["max_abs_difference"]) for row in prefix_audit
        ),
        "selection_status_counts": status_counts,
        "advancing_candidate_count": len(advancement_records),
        "advancing_candidate_count_by_method": {
            method: sum(row["method"] == method for row in advancement_records)
            for method in ("NOG-FO", "ME-DOL-FO")
        },
        "selections": selections,
        "advancement_to_3840": advancement_records,
        "launches_started": 0,
    }
    atomic_write_csv(root / "extension_1920_hits_per_seed.csv", per_seed_rows)
    atomic_write_csv(root / "extension_1920_candidate_results.csv", candidate_rows)
    atomic_write_csv(root / "extension_1920_pareto_by_epsilon.csv", pareto_rows)
    atomic_write_json(root / "selected_1920_by_epsilon.yaml", selections)
    atomic_write_json(
        root / "advancement_to_3840.json",
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "status": "prepared" if advancement_records else "not-needed",
            "source_rounds": rounds,
            "target_rounds": int(cfg["pilot"]["budgets"][2]),
            "rule_version": report["advancement_rule_version"],
            "candidate_count": len(advancement_records),
            "candidates": advancement_records,
            "launches_started": 0,
        },
    )
    atomic_write_json(root / "extension_1920_analysis_report.json", report)
    return report


def validate_3840_advancement_manifest(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if manifest.get("status") != "prepared":
        raise ValueError("3840 advancement manifest must have status=prepared.")
    budgets = [int(value) for value in cfg["pilot"]["budgets"]]
    if int(manifest.get("source_rounds", -1)) != budgets[1]:
        raise ValueError("3840 advancement source_rounds mismatch.")
    if int(manifest.get("target_rounds", -1)) != budgets[2]:
        raise ValueError("3840 advancement target_rounds mismatch.")
    if int(manifest.get("launches_started", -1)) != 0:
        raise ValueError("3840 advancement manifest must be frozen before launch.")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("3840 advancement manifest has no candidates.")
    if int(manifest.get("candidate_count", -1)) != len(candidates):
        raise ValueError("3840 advancement candidate_count mismatch.")
    observed_ids = [row.get("candidate_id") for row in candidates]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("3840 advancement contains duplicate candidate IDs.")

    coarse_lookup = {row["candidate_id"]: row for row in coarse_candidates(cfg)}
    with open(
        Path(cfg["run"]["out_dir"])
        / cfg["run"]["name"]
        / "advancement_to_1920.json",
        "r",
        encoding="utf-8",
    ) as handle:
        prior_manifest = json.load(handle)
    prior_ids = {row["candidate_id"] for row in prior_manifest["candidates"]}
    for candidate in candidates:
        identifier = candidate.get("candidate_id")
        if identifier not in prior_ids:
            raise ValueError(
                f"3840 candidate did not pass the 1920 gate: {identifier}."
            )
        canonical = coarse_lookup.get(identifier)
        if canonical is None:
            raise ValueError(f"Unknown 3840 candidate: {identifier}.")
        if candidate.get("method") != canonical["method"]:
            raise ValueError(f"3840 method mismatch for {identifier}.")
        if candidate.get("parameters") != canonical["parameters"]:
            raise ValueError(f"3840 parameter mismatch for {identifier}.")
    return candidates


def run_extension_3840(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    """Run the final max-budget candidates frozen by Step 6E."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    manifest_path = root / "advancement_to_3840.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        advancement = json.load(handle)
    # Resolve the prior-gate file relative to the actual output root, including
    # callers that override --output-root.
    expected_prior_path = (
        Path(cfg["run"]["out_dir"])
        / cfg["run"]["name"]
        / "advancement_to_1920.json"
    )
    if root != expected_prior_path.parent:
        cfg = copy.deepcopy(cfg)
        cfg["run"]["out_dir"] = str(root.parent)
        cfg["run"]["name"] = root.name
    candidates = validate_3840_advancement_manifest(cfg, advancement)
    rounds = int(advancement["target_rounds"])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    worker = int(cfg["pilot"]["reference_worker"])
    process = process_config_from_experiment(cfg)
    completion_records = []
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"final_extension_candidate={index}/{len(candidates)} "
            f"id={candidate['candidate_id']} status=starting",
            flush=True,
        )
        current_cfg = candidate_config(cfg, candidate, rounds)
        candidate_root = (
            root / "extension" / f"rounds_{rounds}" / candidate["candidate_id"]
        )
        completion = run_task_set(
            current_cfg,
            [CpuFoTask(candidate["method"], seed, worker) for seed in seeds],
            candidate_root,
            process,
            continue_on_error=True,
        )
        record = {
            "candidate_id": candidate["candidate_id"],
            "method": candidate["method"],
            "status": completion["status"],
            "completed_tasks": completion["completed_tasks"],
            "failed_tasks": completion["failed_tasks"],
            "output_root": str(candidate_root.relative_to(root)),
        }
        completion_records.append(record)
        print(
            f"final_extension_candidate={index}/{len(candidates)} "
            f"id={candidate['candidate_id']} status={record['status']} "
            f"tasks={record['completed_tasks']}/3 failures={record['failed_tasks']}",
            flush=True,
        )
    result = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": (
            "complete"
            if all(row["status"] == "complete" for row in completion_records)
            else "incomplete"
        ),
        "source_manifest": str(manifest_path.relative_to(root)),
        "rounds": rounds,
        "candidate_count": len(candidates),
        "expected_tasks": len(candidates) * len(seeds),
        "completed_tasks": sum(row["completed_tasks"] for row in completion_records),
        "failed_tasks": sum(row["failed_tasks"] for row in completion_records),
        "records": completion_records,
    }
    atomic_write_json(root / "extension_3840_completion.json", result)
    atomic_write_csv(root / "extension_3840_completion.csv", completion_records)
    return result


def _load_extension_3840_payloads(
    root: Path,
    candidates: List[Dict[str, Any]],
    rounds: int,
    seeds: List[int],
) -> Dict[str, List[Dict[str, Any]]]:
    with open(
        root / "extension_3840_completion.json",
        "r",
        encoding="utf-8",
    ) as handle:
        stage_completion = json.load(handle)
    if stage_completion.get("status") != "complete":
        raise ValueError("3840-round extension is not complete.")
    if int(stage_completion.get("rounds", -1)) != rounds:
        raise ValueError("3840 completion rounds do not match analysis rounds.")
    expected_seeds = sorted(seeds)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        candidate_root = root / "extension" / f"rounds_{rounds}" / candidate[
            "candidate_id"
        ]
        with open(
            candidate_root / "completion_manifest.json",
            "r",
            encoding="utf-8",
        ) as handle:
            completion = json.load(handle)
        if completion.get("status") != "complete":
            raise ValueError(
                f"3840 candidate is incomplete: {candidate['candidate_id']}."
            )
        payloads = []
        for record in completion["records"]:
            with open(
                candidate_root / record["partial_path"],
                "r",
                encoding="utf-8",
            ) as handle:
                payloads.append(json.load(handle))
        observed_seeds = sorted(int(row["formal_seed"]) for row in payloads)
        if observed_seeds != expected_seeds:
            raise ValueError(
                f"3840 seeds mismatch for {candidate['candidate_id']}."
            )
        result[candidate["candidate_id"]] = payloads
    return result


def prepare_nog_batch_refinement(
    cfg: Dict[str, Any],
    grouped_nog_rows: Dict[tuple[str, float], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Create the union of top-3 NOG Pareto bases and new batch variants."""

    top_n = int(cfg["pilot"]["top_n_refinement"])
    reasons: Dict[str, List[str]] = {}
    per_epsilon: Dict[str, List[str]] = {}
    for (method, epsilon), rows in sorted(grouped_nog_rows.items()):
        if method != "NOG-FO":
            continue
        selection = select_for_epsilon(rows)
        frontier_lookup = {
            row["candidate_id"]: row
            for row in rows
            if row["candidate_id"] in selection["frontier_candidate_ids"]
        }
        ordered = sorted(
            frontier_lookup.values(),
            key=lambda row: (
                float(row["median_training_time"]),
                float(row["median_depth"]),
                float(row["median_total_work"]),
                row["candidate_id"],
            ),
        )[:top_n]
        identifiers = [row["candidate_id"] for row in ordered]
        per_epsilon[f"{epsilon:g}"] = identifiers
        for identifier in identifiers:
            reasons.setdefault(identifier, []).append(
                f"NOG-FO:epsilon={epsilon:g}:top{top_n}-pareto"
            )

    canonical = {row["candidate_id"]: row for row in coarse_candidates(cfg)}
    base_candidates = [
        {**canonical[identifier], "reasons": reasons[identifier]}
        for identifier in canonical
        if identifier in reasons
    ]
    smooth_values = [
        int(value) for value in cfg["pilot"]["nog"]["refinement_smooth_B"]
    ]
    coarse_smooth = int(cfg["pilot"]["nog"]["coarse_smooth_B"])
    variants = []
    runnable = []
    reused = []
    for base in base_candidates:
        for smooth_batch in smooth_values:
            parameters = dict(base["parameters"])
            parameters["smooth_B"] = smooth_batch
            variant = {
                "candidate_id": candidate_id("NOG-FO", parameters),
                "method": "NOG-FO",
                "phase": "batch-refinement",
                "parameters": parameters,
                "base_candidate_id": base["candidate_id"],
                "reasons": base["reasons"],
                "reuse_existing": smooth_batch == coarse_smooth,
            }
            variants.append(variant)
            (reused if variant["reuse_existing"] else runnable).append(variant)
    seed_count = len(cfg["pilot"]["seeds"])
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "prepared",
        "source_stage": "extension-1920",
        "initial_rounds": int(cfg["pilot"]["budgets"][0]),
        "selection_rule": f"union-top{top_n}-pareto-by-epsilon",
        "per_epsilon_base_candidates": per_epsilon,
        "base_candidate_count": len(base_candidates),
        "smooth_B_values": smooth_values,
        "variant_count": len(variants),
        "reused_variant_count": len(reused),
        "runnable_variant_count": len(runnable),
        "runnable_task_count": len(runnable) * seed_count,
        "base_candidates": base_candidates,
        "variants": variants,
        "runnable_variants": runnable,
        "reused_variants": reused,
        "launches_started": 0,
    }


def analyze_extension_3840(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    """Finalize max-budget ME-DOL analysis and prepare NOG refinement."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    with open(root / "advancement_to_3840.json", "r", encoding="utf-8") as handle:
        advancement = json.load(handle)
    candidates = validate_3840_advancement_manifest(cfg, advancement)
    candidate_lookup = {row["candidate_id"]: row for row in candidates}
    source_rounds = int(advancement["source_rounds"])
    rounds = int(advancement["target_rounds"])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    epsilons = [float(value) for value in cfg["pilot"]["epsilons"]]
    consecutive = int(cfg["pilot"]["confirmed_hit_consecutive"])

    with open(root / "advancement_to_1920.json", "r", encoding="utf-8") as handle:
        prior_advancement = json.load(handle)
    prior_candidates = validate_advancement_manifest(cfg, prior_advancement)
    prior_lookup = {row["candidate_id"]: row for row in prior_candidates}
    selected_prior_candidates = [prior_lookup[row["candidate_id"]] for row in candidates]
    prior_payloads = _load_extension_payloads(
        root,
        selected_prior_candidates,
        source_rounds,
        seeds,
    )
    payloads = _load_extension_3840_payloads(root, candidates, rounds, seeds)
    prefix_audit = _prefix_consistency_audit(
        candidates,
        prior_payloads,
        payloads,
        source_rounds,
    )
    atomic_write_csv(root / "prefix_consistency_1920_vs_3840.csv", prefix_audit)
    if not all(row["passed"] for row in prefix_audit):
        raise RuntimeError("1920/3840 deterministic prefix consistency audit failed.")

    candidate_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    selections_3840: Dict[str, Dict[str, Any]] = {}
    pareto_rows: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}
    for candidate in candidates:
        for epsilon in epsilons:
            summary = summarize_candidate(
                candidate,
                payloads[candidate["candidate_id"]],
                epsilon,
                consecutive,
            )
            candidate_rows.append(summary)
            grouped.setdefault((candidate["method"], epsilon), []).append(summary)
            per_seed_rows.extend(
                seed_hit_rows(
                    candidate,
                    payloads[candidate["candidate_id"]],
                    epsilon,
                    consecutive,
                )
            )
    for (method, epsilon), rows in sorted(grouped.items()):
        selected = select_for_epsilon(rows)
        status_counts[selected["status"]] = (
            status_counts.get(selected["status"], 0) + 1
        )
        frontier_ids = set(selected["frontier_candidate_ids"])
        for row in rows:
            row["pareto_frontier"] = row["candidate_id"] in frontier_ids
            row["selected"] = row["candidate_id"] == selected["selected_candidate_id"]
            row["selection_status"] = selected["status"]
            if row["pareto_frontier"]:
                pareto_rows.append(dict(row))
        selected_row = next(
            row for row in rows if row["candidate_id"] == selected["selected_candidate_id"]
        )
        selections_3840.setdefault(f"{epsilon:g}", {})[method] = {
            **selected,
            "selected_parameters": candidate_lookup[
                selected["selected_candidate_id"]
            ]["parameters"],
            "selected_hit_rate": float(selected_row["hit_rate"]),
            "selected_median_final_stat_proxy": float(
                selected_row["median_final_stat_proxy"]
            ),
            "selected_median_depth": _finite_or_none(selected_row["median_depth"]),
            "selected_median_total_work": _finite_or_none(
                selected_row["median_total_work"]
            ),
            "selected_median_training_time": _finite_or_none(
                selected_row["median_training_time"]
            ),
        }

    with open(
        root / "extension_1920_analysis_report.json",
        "r",
        encoding="utf-8",
    ) as handle:
        analysis_1920 = json.load(handle)
    staged_selections = copy.deepcopy(analysis_1920["selections"])
    strict_key = f"{min(epsilons):g}"
    staged_selections[strict_key]["ME-DOL-FO"] = {
        **selections_3840[strict_key]["ME-DOL-FO"],
        "selected_stage_rounds": rounds,
        "max_budget_censored": (
            selections_3840[strict_key]["ME-DOL-FO"]["status"]
            != "full-hit"
        ),
    }
    for epsilon_key, methods in staged_selections.items():
        for method, selection in methods.items():
            selection.setdefault("selected_stage_rounds", source_rounds)
            selection.setdefault("max_budget_censored", False)

    with open(
        root / "extension_1920_candidate_results.csv",
        "r",
        encoding="utf-8",
    ) as handle:
        import csv

        nog_rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == "NOG-FO"
        ]
    typed_nog_rows: List[Dict[str, Any]] = []
    for row in nog_rows:
        typed = dict(row)
        for key in (
            "epsilon",
            "hit_rate",
            "median_final_stat_proxy",
            "median_depth",
            "median_total_work",
            "median_per_worker_work",
            "median_training_time",
        ):
            typed[key] = float(typed[key])
        typed["full_hit"] = typed["full_hit"] == "True"
        typed_nog_rows.append(typed)
    grouped_nog: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for row in typed_nog_rows:
        grouped_nog.setdefault(("NOG-FO", float(row["epsilon"])), []).append(row)
    refinement = prepare_nog_batch_refinement(cfg, grouped_nog)

    strict_selection = selections_3840[strict_key]["ME-DOL-FO"]
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "complete",
        "analysis_stage": "extension-3840",
        "created_at_utc": utc_now(),
        "rounds": rounds,
        "candidate_count": len(candidates),
        "task_count": len(candidates) * len(seeds),
        "prefix_audit_rows": len(prefix_audit),
        "prefix_audit_passed": all(row["passed"] for row in prefix_audit),
        "prefix_max_abs_difference": max(
            float(row["max_abs_difference"]) for row in prefix_audit
        ),
        "selection_status_counts": status_counts,
        "strict_epsilon": float(strict_key),
        "strict_epsilon_hit_rate": strict_selection["selected_hit_rate"],
        "strict_epsilon_status": strict_selection["status"],
        "max_budget_reached": True,
        "further_round_extension_allowed": False,
        "staged_selections": staged_selections,
        "batch_refinement": {
            "base_candidate_count": refinement["base_candidate_count"],
            "variant_count": refinement["variant_count"],
            "reused_variant_count": refinement["reused_variant_count"],
            "runnable_variant_count": refinement["runnable_variant_count"],
            "runnable_task_count": refinement["runnable_task_count"],
        },
        "launches_started": 0,
    }
    atomic_write_csv(root / "extension_3840_hits_per_seed.csv", per_seed_rows)
    atomic_write_csv(root / "extension_3840_candidate_results.csv", candidate_rows)
    atomic_write_csv(root / "extension_3840_pareto_by_epsilon.csv", pareto_rows)
    atomic_write_json(root / "selected_staged_by_epsilon.yaml", staged_selections)
    atomic_write_json(root / "nog_batch_refinement_manifest.json", refinement)
    atomic_write_json(root / "extension_3840_analysis_report.json", report)
    return report


def validate_nog_refinement_manifest(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if manifest.get("status") != "prepared":
        raise ValueError("NOG refinement manifest must have status=prepared.")
    if int(manifest.get("launches_started", -1)) != 0:
        raise ValueError("NOG refinement manifest must be frozen before launch.")
    if int(manifest.get("initial_rounds", -1)) != int(cfg["pilot"]["budgets"][0]):
        raise ValueError("NOG refinement rounds do not match pilot budget.")
    runnable = manifest.get("runnable_variants")
    if not isinstance(runnable, list) or not runnable:
        raise ValueError("NOG refinement has no runnable variants.")
    if int(manifest.get("runnable_variant_count", -1)) != len(runnable):
        raise ValueError("NOG refinement runnable variant count mismatch.")
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    if int(manifest.get("runnable_task_count", -1)) != len(runnable) * len(seeds):
        raise ValueError("NOG refinement task count mismatch.")
    identifiers = [row.get("candidate_id") for row in runnable]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("NOG refinement contains duplicate candidate IDs.")

    base_lookup = {
        row["candidate_id"]: row for row in manifest.get("base_candidates", [])
    }
    canonical = {row["candidate_id"]: row for row in coarse_candidates(cfg)}
    allowed_smooth = {
        int(value) for value in cfg["pilot"]["nog"]["refinement_smooth_B"]
    }
    reused_smooth = int(cfg["pilot"]["nog"]["coarse_smooth_B"])
    rounds = int(manifest["initial_rounds"])
    for variant in runnable:
        if variant.get("method") != "NOG-FO":
            raise ValueError("NOG refinement variant has a non-NOG method.")
        if variant.get("reuse_existing") is not False:
            raise ValueError("Runnable NOG refinement variant cannot be reused.")
        base_id = variant.get("base_candidate_id")
        if base_id not in base_lookup or base_id not in canonical:
            raise ValueError(f"Unknown NOG refinement base: {base_id}.")
        base_parameters = canonical[base_id]["parameters"]
        parameters = variant.get("parameters", {})
        smooth_batch = int(parameters.get("smooth_B", -1))
        if smooth_batch not in allowed_smooth or smooth_batch == reused_smooth:
            raise ValueError(f"Invalid runnable smooth_B={smooth_batch}.")
        for key in ("M", "eta", "data_B_total"):
            if parameters.get(key) != base_parameters[key]:
                raise ValueError(
                    f"NOG refinement parameter {key} changed outside batch grid."
                )
        if variant.get("candidate_id") != candidate_id("NOG-FO", parameters):
            raise ValueError("NOG refinement candidate ID does not match parameters.")
        candidate_config(cfg, variant, rounds)
    return runnable


def run_nog_batch_refinement(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    """Run the frozen non-reused NOG batch variants from Step 6G."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    manifest_path = root / "nog_batch_refinement_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    variants = validate_nog_refinement_manifest(cfg, manifest)
    rounds = int(manifest["initial_rounds"])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    worker = int(cfg["pilot"]["reference_worker"])
    process = process_config_from_experiment(cfg)
    records = []
    for index, variant in enumerate(variants, start=1):
        print(
            f"refinement_variant={index}/{len(variants)} "
            f"id={variant['candidate_id']} status=starting",
            flush=True,
        )
        current_cfg = candidate_config(cfg, variant, rounds)
        variant_root = (
            root
            / "batch_refinement"
            / f"rounds_{rounds}"
            / variant["candidate_id"]
        )
        completion = run_task_set(
            current_cfg,
            [CpuFoTask("NOG-FO", seed, worker) for seed in seeds],
            variant_root,
            process,
            continue_on_error=True,
        )
        record = {
            "candidate_id": variant["candidate_id"],
            "base_candidate_id": variant["base_candidate_id"],
            "smooth_B": int(variant["parameters"]["smooth_B"]),
            "status": completion["status"],
            "completed_tasks": completion["completed_tasks"],
            "failed_tasks": completion["failed_tasks"],
            "output_root": str(variant_root.relative_to(root)),
        }
        records.append(record)
        print(
            f"refinement_variant={index}/{len(variants)} "
            f"id={variant['candidate_id']} status={record['status']} "
            f"tasks={record['completed_tasks']}/3 failures={record['failed_tasks']}",
            flush=True,
        )
    result = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": (
            "complete"
            if all(row["status"] == "complete" for row in records)
            else "incomplete"
        ),
        "source_manifest": str(manifest_path.relative_to(root)),
        "rounds": rounds,
        "variant_count": len(variants),
        "expected_tasks": len(variants) * len(seeds),
        "completed_tasks": sum(row["completed_tasks"] for row in records),
        "failed_tasks": sum(row["failed_tasks"] for row in records),
        "records": records,
    }
    atomic_write_json(root / "batch_refinement_completion.json", result)
    atomic_write_csv(root / "batch_refinement_completion.csv", records)
    return result


def _load_nog_refinement_payloads(
    root: Path,
    variants: List[Dict[str, Any]],
    rounds: int,
    seeds: List[int],
) -> Dict[str, List[Dict[str, Any]]]:
    with open(
        root / "batch_refinement_completion.json",
        "r",
        encoding="utf-8",
    ) as handle:
        completion = json.load(handle)
    if completion.get("status") != "complete":
        raise ValueError("NOG batch refinement is not complete.")
    if int(completion.get("rounds", -1)) != rounds:
        raise ValueError("NOG batch refinement rounds mismatch.")
    expected_seeds = sorted(seeds)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for variant in variants:
        variant_root = (
            root
            / "batch_refinement"
            / f"rounds_{rounds}"
            / variant["candidate_id"]
        )
        with open(
            variant_root / "completion_manifest.json",
            "r",
            encoding="utf-8",
        ) as handle:
            variant_completion = json.load(handle)
        if variant_completion.get("status") != "complete":
            raise ValueError(
                f"Refinement variant incomplete: {variant['candidate_id']}."
            )
        payloads = []
        for record in variant_completion["records"]:
            with open(
                variant_root / record["partial_path"],
                "r",
                encoding="utf-8",
            ) as handle:
                payloads.append(json.load(handle))
        observed = sorted(int(row["formal_seed"]) for row in payloads)
        if observed != expected_seeds:
            raise ValueError(
                f"Refinement seeds mismatch for {variant['candidate_id']}."
            )
        result[variant["candidate_id"]] = payloads
    return result


def nog_refinement_work_audit(
    cfg: Dict[str, Any],
    variants: List[Dict[str, Any]],
    payloads: Dict[str, List[Dict[str, Any]]],
    rounds: int,
) -> List[Dict[str, Any]]:
    rows = []
    for variant in variants:
        parameters = variant["parameters"]
        expected_total = (
            (rounds + 2)
            * int(parameters["smooth_B"])
            * int(parameters["data_B_total"])
        )
        for payload in payloads[variant["candidate_id"]]:
            observed = int(payload["rows"][-1]["total_work"])
            rows.append(
                {
                    "candidate_id": variant["candidate_id"],
                    "formal_seed": int(payload["formal_seed"]),
                    "smooth_B": int(parameters["smooth_B"]),
                    "rounds": rounds,
                    "expected_final_total_work": expected_total,
                    "observed_final_total_work": observed,
                    "passed": observed == expected_total,
                }
            )
    return rows


def analyze_nog_batch_refinement(
    cfg: Dict[str, Any],
    output_root: str | Path,
) -> Dict[str, Any]:
    """Finalize pilot selections after the NOG batch sweep."""

    validate_pilot_config(cfg)
    root = Path(output_root)
    with open(
        root / "nog_batch_refinement_manifest.json",
        "r",
        encoding="utf-8",
    ) as handle:
        refinement = json.load(handle)
    runnable = validate_nog_refinement_manifest(cfg, refinement)
    reused = refinement["reused_variants"]
    variants = refinement["variants"]
    rounds = int(refinement["initial_rounds"])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    epsilons = [float(value) for value in cfg["pilot"]["epsilons"]]
    consecutive = int(cfg["pilot"]["confirmed_hit_consecutive"])

    payloads = _load_nog_refinement_payloads(root, runnable, rounds, seeds)
    reused_payloads = _load_coarse_payloads(root, reused, rounds, seeds)
    payloads.update(reused_payloads)
    if set(payloads) != {row["candidate_id"] for row in variants}:
        raise ValueError("Refinement payload coverage does not match all variants.")

    work_audit = nog_refinement_work_audit(cfg, variants, payloads, rounds)
    atomic_write_csv(root / "batch_refinement_work_audit.csv", work_audit)
    if not all(row["passed"] for row in work_audit):
        raise RuntimeError("NOG batch-refinement work audit failed.")

    variant_lookup = {row["candidate_id"]: row for row in variants}
    candidate_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for variant in variants:
        for epsilon in epsilons:
            summary = summarize_candidate(
                variant,
                payloads[variant["candidate_id"]],
                epsilon,
                consecutive,
            )
            candidate_rows.append(summary)
            grouped.setdefault(("NOG-FO", epsilon), []).append(summary)
            per_seed_rows.extend(
                seed_hit_rows(
                    variant,
                    payloads[variant["candidate_id"]],
                    epsilon,
                    consecutive,
                )
            )

    nog_selections: Dict[str, Dict[str, Any]] = {}
    pareto_rows: List[Dict[str, Any]] = []
    for (_, epsilon), rows in sorted(grouped.items()):
        selected = select_for_epsilon(rows)
        frontier_ids = set(selected["frontier_candidate_ids"])
        for row in rows:
            row["pareto_frontier"] = row["candidate_id"] in frontier_ids
            row["selected"] = row["candidate_id"] == selected["selected_candidate_id"]
            row["selection_status"] = selected["status"]
            if row["pareto_frontier"]:
                pareto_rows.append(dict(row))
        selected_row = next(
            row for row in rows if row["candidate_id"] == selected["selected_candidate_id"]
        )
        nog_selections[f"{epsilon:g}"] = {
            **selected,
            "selected_parameters": variant_lookup[
                selected["selected_candidate_id"]
            ]["parameters"],
            "selected_hit_rate": float(selected_row["hit_rate"]),
            "selected_median_final_stat_proxy": float(
                selected_row["median_final_stat_proxy"]
            ),
            "selected_median_depth": _finite_or_none(selected_row["median_depth"]),
            "selected_median_total_work": _finite_or_none(
                selected_row["median_total_work"]
            ),
            "selected_median_training_time": _finite_or_none(
                selected_row["median_training_time"]
            ),
            "selected_stage_rounds": rounds,
            "max_budget_censored": selected["status"] != "full-hit",
        }

    with open(
        root / "selected_staged_by_epsilon.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        staged = json.load(handle)
    frozen: Dict[str, Dict[str, Any]] = {}
    for epsilon in epsilons:
        key = f"{epsilon:g}"
        frozen[key] = {
            "NOG-FO": nog_selections[key],
            "ME-DOL-FO": staged[key]["ME-DOL-FO"],
        }

    formal_config: Dict[str, Any] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "candidate_grid_sha256": object_sha256(
            {
                "coarse": coarse_candidates(cfg),
                "batch_refinement": variants,
            }
        ),
        "pilot_seeds": seeds,
        "formal_seeds": [int(value) for value in cfg["run"]["formal_seeds"]],
        "reference_worker": int(cfg["pilot"]["reference_worker"]),
        "delta": float(cfg["oracle"]["delta"]),
        "evaluation_seed_mode": cfg["oracle"]["evaluation_seed_mode"],
        "confirmed_hit_consecutive": consecutive,
        "by_epsilon": frozen,
    }
    formal_config["frozen_config_sha256"] = object_sha256(formal_config)
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "complete",
        "analysis_stage": "batch-refinement-final",
        "created_at_utc": utc_now(),
        "variant_count": len(variants),
        "new_variant_count": len(runnable),
        "reused_variant_count": len(reused),
        "task_count": len(runnable) * len(seeds),
        "work_audit_rows": len(work_audit),
        "work_audit_passed": all(row["passed"] for row in work_audit),
        "candidate_epsilon_rows": len(candidate_rows),
        "per_seed_epsilon_rows": len(per_seed_rows),
        "frozen_config_sha256": formal_config["frozen_config_sha256"],
        "unresolved_max_budget": [
            {
                "method": method,
                "epsilon": float(epsilon),
                "hit_rate": selection["selected_hit_rate"],
            }
            for epsilon, methods in frozen.items()
            for method, selection in methods.items()
            if selection.get("max_budget_censored", False)
        ],
        "selections": frozen,
        "pilot_complete": True,
        "formal_runs_started": 0,
    }
    atomic_write_csv(root / "batch_refinement_hits_per_seed.csv", per_seed_rows)
    atomic_write_csv(root / "batch_refinement_candidate_results.csv", candidate_rows)
    atomic_write_csv(root / "batch_refinement_pareto_by_epsilon.csv", pareto_rows)
    atomic_write_json(root / "selected_config_by_epsilon.yaml", formal_config)
    atomic_write_json(root / "pilot_final_report.json", report)
    return report


def run_coarse(cfg: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    """Launch the 960-round coarse grid. Call only after explicit approval."""

    manifest = prepare_manifest(cfg, output_root)
    root = Path(output_root)
    rounds = int(cfg["pilot"]["budgets"][0])
    seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    worker = int(cfg["pilot"]["reference_worker"])
    process = process_config_from_experiment(cfg)
    completion_records = []
    for candidate in manifest["candidates"]:
        current_cfg = candidate_config(cfg, candidate, rounds)
        candidate_root = root / "coarse" / candidate["candidate_id"] / f"rounds_{rounds}"
        completion = run_task_set(
            current_cfg,
            [CpuFoTask(candidate["method"], seed, worker) for seed in seeds],
            candidate_root,
            process,
            continue_on_error=True,
        )
        completion_records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": completion["status"],
                "completed_tasks": completion["completed_tasks"],
                "failed_tasks": completion["failed_tasks"],
                "output_root": str(candidate_root.relative_to(root)),
            }
        )
    result = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": (
            "complete"
            if all(record["status"] == "complete" for record in completion_records)
            else "incomplete"
        ),
        "rounds": rounds,
        "records": completion_records,
    }
    atomic_write_json(root / "coarse_completion.json", result)
    atomic_write_csv(root / "coarse_completion.csv", completion_records)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--phase",
        choices=[
            "prepare",
            "coarse",
            "analyze",
            "extend1920",
            "analyze1920",
            "extend3840",
            "analyze3840",
            "refine-nog-batch",
            "analyze-refinement",
        ],
        default="prepare",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    )
    if args.phase == "prepare" or args.dry_run:
        manifest = prepare_manifest(cfg, output_root)
        print(
            f"phase=prepare candidates={manifest['candidate_count']} "
            f"tasks={manifest['coarse_task_count']} "
            f"estimated_hours={manifest['estimated_coarse_hours']:.2f}"
        )
        print(f"output_root={output_root}")
        print("launches_started=0; no worker processes launched")
        return
    if args.phase == "analyze":
        report = analyze_coarse(cfg, output_root)
        print(
            f"phase=analyze status={report['status']} "
            f"rows={report['candidate_epsilon_rows']} "
            f"advancing={report['advancing_candidate_count']}"
        )
        print("launches_started=0; no worker processes launched")
        return
    if args.phase == "extend1920":
        result = run_extension_1920(cfg, output_root)
        print(
            f"phase=extend1920 status={result['status']} "
            f"tasks={result['completed_tasks']}/{result['expected_tasks']}"
        )
        return
    if args.phase == "analyze1920":
        report = analyze_extension_1920(cfg, output_root)
        print(
            f"phase=analyze1920 status={report['status']} "
            f"prefix_audit={report['prefix_audit_passed']} "
            f"advancing={report['advancing_candidate_count']}"
        )
        print("launches_started=0; no worker processes launched")
        return
    if args.phase == "extend3840":
        result = run_extension_3840(cfg, output_root)
        print(
            f"phase=extend3840 status={result['status']} "
            f"tasks={result['completed_tasks']}/{result['expected_tasks']}"
        )
        return
    if args.phase == "analyze3840":
        report = analyze_extension_3840(cfg, output_root)
        print(
            f"phase=analyze3840 status={report['status']} "
            f"prefix_audit={report['prefix_audit_passed']} "
            f"strict_hit_rate={report['strict_epsilon_hit_rate']:.3f} "
            f"refinement_tasks={report['batch_refinement']['runnable_task_count']}"
        )
        print("launches_started=0; no worker processes launched")
        return
    if args.phase == "refine-nog-batch":
        result = run_nog_batch_refinement(cfg, output_root)
        print(
            f"phase=refine-nog-batch status={result['status']} "
            f"tasks={result['completed_tasks']}/{result['expected_tasks']}"
        )
        return
    if args.phase == "analyze-refinement":
        report = analyze_nog_batch_refinement(cfg, output_root)
        print(
            f"phase=analyze-refinement status={report['status']} "
            f"work_audit={report['work_audit_passed']} "
            f"pilot_complete={report['pilot_complete']}"
        )
        print("formal_runs_started=0; no worker processes launched")
        return
    result = run_coarse(cfg, output_root)
    print(f"phase=coarse status={result['status']} candidates={len(result['records'])}")


if __name__ == "__main__":
    main()
