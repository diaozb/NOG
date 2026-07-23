"""Region-level analysis for the initial wide-epsilon pilot sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import (
    atomic_write_csv,
    compare_trajectories,
    load_config,
)
from src.distributed.cpu_fo_pilot import (
    _load_coarse_payloads,
    candidate_id,
    coarse_candidates,
    confirmed_hit,
    pareto_frontier,
)
from src.distributed.cpu_fo_tasks import atomic_write_json, object_sha256, utc_now
from src.distributed.epsilon_scaling import epsilon_region, validate_scaling_protocol
from src.distributed.epsilon_scaling_pilot import pilot_runner_config


ANALYSIS_SCHEMA_VERSION = 1
REGION_SELECTION_VERSION = "representative-coverage-pareto-v1"


def validate_refinement_manifest(
    cfg: Dict[str, Any], manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if manifest.get("status") != "prepared" or manifest.get("launches_started") != 0:
        raise ValueError("Refinement manifest is not frozen before launch.")
    expected_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if expected_hash != object_sha256(unhashed):
        raise ValueError("Refinement manifest SHA256 mismatch.")
    allowed_smooth = {
        int(value)
        for value in cfg["epsilon_scaling"]["pilot"]["nog"]["refinement_smooth_B"]
    }
    coarse_smooth = int(
        cfg["epsilon_scaling"]["pilot"]["nog"]["coarse_smooth_B"]
    )
    runnable = manifest.get("runnable_variants", [])
    expected_tasks = len(runnable) * len(cfg["epsilon_scaling"]["pilot_seeds"])
    if int(manifest.get("runnable_task_count", -1)) != expected_tasks:
        raise ValueError("Refinement task count mismatch.")
    identifiers = set()
    for variant in runnable:
        parameters = variant.get("parameters", {})
        smooth = int(parameters.get("smooth_B", -1))
        if smooth not in allowed_smooth or smooth == coarse_smooth:
            raise ValueError("Refinement smooth_B is outside the frozen grid.")
        if variant.get("reuse_existing") is not False:
            raise ValueError("Runnable refinement variant cannot reuse an artifact.")
        if variant.get("candidate_id") != candidate_id("NOG-FO", parameters):
            raise ValueError("Refinement candidate identity mismatch.")
        if variant["candidate_id"] in identifiers:
            raise ValueError("Duplicate refinement candidate identity.")
        identifiers.add(variant["candidate_id"])
    return runnable


def _median_or_inf(values: List[float]) -> float:
    return statistics.median(values) if values else math.inf


def summarize_region_candidate(
    candidate: Dict[str, Any],
    payloads: List[Dict[str, Any]],
    region: str,
    representatives: List[float],
    consecutive: int,
) -> Dict[str, Any]:
    hits = []
    hit_rates = []
    final_values = []
    for epsilon in representatives:
        epsilon_hits = [
            confirmed_hit(payload["rows"], epsilon, consecutive)
            for payload in payloads
        ]
        observed = [hit for hit in epsilon_hits if hit is not None]
        hits.extend(observed)
        hit_rates.append(len(observed) / len(payloads))
        final_values.extend(float(payload["rows"][-1]["stat_proxy"]) for payload in payloads)
    possible = len(payloads) * len(representatives)
    return {
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "region": region,
        "representative_epsilons": json.dumps(representatives),
        "num_pilot_seeds": len(payloads),
        "hit_count": len(hits),
        "possible_hit_count": possible,
        "coverage": len(hits) / possible,
        "worst_epsilon_hit_rate": min(hit_rates),
        "full_coverage": len(hits) == possible,
        "median_depth": _median_or_inf([float(hit["depth"]) for hit in hits]),
        "median_total_work": _median_or_inf([float(hit["total_work"]) for hit in hits]),
        "median_training_time": _median_or_inf([float(hit["training_time"]) for hit in hits]),
        "median_final_stat_proxy": statistics.median(final_values),
        "parameters": json.dumps(candidate["parameters"], sort_keys=True),
    }


def select_region(rows: List[Dict[str, Any]], top_n_censored: int) -> Dict[str, Any]:
    best_coverage = max(float(row["coverage"]) for row in rows)
    eligible = [row for row in rows if float(row["coverage"]) == best_coverage]
    full = [row for row in eligible if bool(row["full_coverage"])]
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
        advancing = [row["candidate_id"] for row in frontier]
        status = "full-coverage"
    else:
        frontier = []
        ranked = sorted(
            rows,
            key=lambda row: (
                -float(row["coverage"]),
                -float(row["worst_epsilon_hit_rate"]),
                float(row["median_final_stat_proxy"]),
                row["candidate_id"],
            ),
        )
        selected = ranked[0]
        advancing = [row["candidate_id"] for row in ranked[:top_n_censored]]
        status = "censored-top-candidates"
    return {
        "status": status,
        "selected_candidate_id": selected["candidate_id"],
        "selected_coverage": float(selected["coverage"]),
        "frontier_candidate_ids": [row["candidate_id"] for row in frontier],
        "advancing_candidate_ids": advancing,
    }


def analyze_initial_pilot(
    cfg: Dict[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    validate_scaling_protocol(cfg)
    runner_cfg = pilot_runner_config(cfg)
    root = Path(output_root)
    candidates = coarse_candidates(runner_cfg)
    candidate_lookup = {row["candidate_id"]: row for row in candidates}
    seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    rounds = int(cfg["epsilon_scaling"]["budgets"][0])
    payloads = _load_coarse_payloads(root, candidates, rounds, seeds)
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])
    top_n = int(cfg["epsilon_scaling"]["pilot"]["top_n_censored"])

    rows = []
    selections: Dict[str, Dict[str, Any]] = {}
    for region, region_cfg in cfg["epsilon_scaling"]["regions"].items():
        representatives = [float(value) for value in region_cfg["representative_epsilons"]]
        if any(epsilon_region(value) != region for value in representatives):
            raise ValueError(f"Representative epsilon escaped region {region}.")
        for method in cfg["methods"]["sfo"]:
            method_rows = [
                summarize_region_candidate(
                    candidate,
                    payloads[candidate["candidate_id"]],
                    region,
                    representatives,
                    consecutive,
                )
                for candidate in candidates
                if candidate["method"] == method
            ]
            selection = select_region(method_rows, top_n)
            selections.setdefault(region, {})[method] = selection
            frontier = set(selection["frontier_candidate_ids"])
            advancing = set(selection["advancing_candidate_ids"])
            for row in method_rows:
                row["pareto_frontier"] = row["candidate_id"] in frontier
                row["advancing"] = row["candidate_id"] in advancing
                row["selected"] = row["candidate_id"] == selection["selected_candidate_id"]
            rows.extend(method_rows)

    nog_base_ids = sorted(
        {
            identifier
            for methods in selections.values()
            for identifier in methods["NOG-FO"]["advancing_candidate_ids"]
        }
    )
    smooth_values = [
        int(value)
        for value in cfg["epsilon_scaling"]["pilot"]["nog"]["refinement_smooth_B"]
    ]
    coarse_smooth = int(
        cfg["epsilon_scaling"]["pilot"]["nog"]["coarse_smooth_B"]
    )
    variants = []
    for base_id in nog_base_ids:
        base = candidate_lookup[base_id]
        for smooth_batch in smooth_values:
            parameters = dict(base["parameters"])
            parameters["smooth_B"] = smooth_batch
            variants.append(
                {
                    "candidate_id": candidate_id("NOG-FO", parameters),
                    "base_candidate_id": base_id,
                    "method": "NOG-FO",
                    "parameters": parameters,
                    "reuse_existing": smooth_batch == coarse_smooth,
                }
            )
    refinement = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "rounds": rounds,
        "pilot_seeds": seeds,
        "base_candidate_ids": nog_base_ids,
        "variants": variants,
        "runnable_variants": [row for row in variants if not row["reuse_existing"]],
        "runnable_task_count": sum(not row["reuse_existing"] for row in variants) * len(seeds),
        "launches_started": 0,
    }
    refinement["manifest_sha256"] = object_sha256(refinement)
    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_rule_version": REGION_SELECTION_VERSION,
        "rounds": rounds,
        "candidate_region_rows": len(rows),
        "selections": selections,
        "refinement_base_count": len(nog_base_ids),
        "refinement_runnable_task_count": refinement["runnable_task_count"],
        "launches_started": 0,
    }
    atomic_write_csv(root / "initial_region_candidate_results.csv", rows)
    atomic_write_json(root / "initial_region_selection.json", report)
    atomic_write_json(root / "region_batch_refinement_manifest.json", refinement)
    return report


def _load_task_payloads(root: Path) -> List[Dict[str, Any]]:
    with open(root / "completion_manifest.json", "r", encoding="utf-8") as handle:
        completion = json.load(handle)
    if completion.get("status") != "complete":
        raise ValueError(f"Incomplete task set: {root}.")
    payloads = []
    for record in completion["records"]:
        with open(root / record["partial_path"], "r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))
    return payloads


def analyze_refinement_and_prepare_extension(
    cfg: Dict[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    validate_scaling_protocol(cfg)
    runner_cfg = pilot_runner_config(cfg)
    root = Path(output_root)
    rounds = int(cfg["epsilon_scaling"]["budgets"][0])
    target_rounds = int(cfg["epsilon_scaling"]["budgets"][1])
    seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])
    top_n = int(cfg["epsilon_scaling"]["pilot"]["top_n_censored"])
    coarse = coarse_candidates(runner_cfg)
    coarse_lookup = {row["candidate_id"]: row for row in coarse}
    coarse_payloads = _load_coarse_payloads(root, coarse, rounds, seeds)
    with open(root / "region_batch_refinement_manifest.json", "r", encoding="utf-8") as handle:
        refinement = json.load(handle)
    variants = refinement["variants"]
    candidate_lookup = {
        row["candidate_id"]: {
            "candidate_id": row["candidate_id"],
            "method": "NOG-FO",
            "phase": "region-refinement",
            "parameters": row["parameters"],
        }
        for row in variants
    }
    candidate_lookup.update(
        {row["candidate_id"]: row for row in coarse if row["method"] == "ME-DOL-FO"}
    )
    payload_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for variant in variants:
        identifier = variant["candidate_id"]
        if variant["reuse_existing"]:
            payload_lookup[identifier] = coarse_payloads[variant["base_candidate_id"]]
        else:
            payload_lookup[identifier] = _load_task_payloads(
                root / "region_batch_refinement" / identifier
            )
    for candidate in coarse:
        if candidate["method"] == "ME-DOL-FO":
            payload_lookup[candidate["candidate_id"]] = coarse_payloads[candidate["candidate_id"]]

    rows = []
    selections: Dict[str, Dict[str, Any]] = {}
    advancing_ids = set()
    for region, region_cfg in cfg["epsilon_scaling"]["regions"].items():
        representatives = [float(value) for value in region_cfg["representative_epsilons"]]
        for method in cfg["methods"]["sfo"]:
            method_rows = [
                summarize_region_candidate(
                    candidate,
                    payload_lookup[candidate["candidate_id"]],
                    region,
                    representatives,
                    consecutive,
                )
                for candidate in candidate_lookup.values()
                if candidate["method"] == method
            ]
            selection = select_region(method_rows, top_n)
            selected = candidate_lookup[selection["selected_candidate_id"]]
            selection["selected_parameters"] = selected["parameters"]
            selection["selected_stage_rounds"] = rounds
            selections.setdefault(region, {})[method] = selection
            if selection["status"] != "full-coverage":
                advancing_ids.update(selection["advancing_candidate_ids"])
            for row in method_rows:
                row["selected"] = row["candidate_id"] == selection["selected_candidate_id"]
                row["advancing"] = row["candidate_id"] in selection["advancing_candidate_ids"]
            rows.extend(method_rows)

    advancing = [candidate_lookup[identifier] for identifier in sorted(advancing_ids)]
    extension = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "source_rounds": rounds,
        "target_rounds": target_rounds,
        "pilot_seeds": seeds,
        "candidates": advancing,
        "candidate_count": len(advancing),
        "task_count": len(advancing) * len(seeds),
        "launches_started": 0,
    }
    extension["manifest_sha256"] = object_sha256(extension)
    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "rounds": rounds,
        "candidate_region_rows": len(rows),
        "selections": selections,
        "advancing_candidate_count": len(advancing),
        "extension_task_count": extension["task_count"],
        "launches_started": 0,
    }
    atomic_write_csv(root / "refined_region_candidate_results.csv", rows)
    atomic_write_json(root / "refined_region_selection.json", report)
    atomic_write_json(root / "region_extension_manifest.json", extension)
    return report


def validate_extension_manifest(
    cfg: Dict[str, Any], manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if manifest.get("status") != "prepared" or manifest.get("launches_started") != 0:
        raise ValueError("Extension manifest is not frozen before launch.")
    unhashed = dict(manifest)
    expected = unhashed.pop("manifest_sha256", None)
    if expected != object_sha256(unhashed):
        raise ValueError("Extension manifest SHA256 mismatch.")
    budgets = [int(value) for value in cfg["epsilon_scaling"]["budgets"]]
    source, target = int(manifest["source_rounds"]), int(manifest["target_rounds"])
    if source not in budgets or target not in budgets or budgets.index(target) != budgets.index(source) + 1:
        raise ValueError("Extension budgets are not adjacent preregistered stages.")
    candidates = manifest.get("candidates", [])
    if int(manifest.get("task_count", -1)) != len(candidates) * len(cfg["epsilon_scaling"]["pilot_seeds"]):
        raise ValueError("Extension task count mismatch.")
    identifiers = [row["candidate_id"] for row in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Extension candidates are duplicated.")
    return candidates


def analyze_extension_stage(
    cfg: Dict[str, Any], output_root: str | Path
) -> Dict[str, Any]:
    validate_scaling_protocol(cfg)
    root = Path(output_root)
    with open(root / "region_extension_manifest.json", "r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    candidates = validate_extension_manifest(cfg, source_manifest)
    source_rounds = int(source_manifest["source_rounds"])
    rounds = int(source_manifest["target_rounds"])
    budgets = [int(value) for value in cfg["epsilon_scaling"]["budgets"]]
    seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])
    top_n = int(cfg["epsilon_scaling"]["pilot"]["top_n_censored"])
    representatives = [
        float(value)
        for value in cfg["epsilon_scaling"]["regions"]["fine"]["representative_epsilons"]
    ]
    payload_lookup = {
        candidate["candidate_id"]: _load_task_payloads(
            root / f"region_extension_{rounds}" / candidate["candidate_id"]
        )
        for candidate in candidates
    }

    prefix_rows = []
    for candidate in candidates:
        identifier = candidate["candidate_id"]
        if source_rounds == budgets[0]:
            if candidate["method"] == "ME-DOL-FO":
                source_root = root / "coarse" / identifier / f"rounds_{source_rounds}"
            else:
                source_root = root / "region_batch_refinement" / identifier
        else:
            source_root = root / f"region_extension_{source_rounds}" / identifier
        source_payloads = {
            int(payload["formal_seed"]): payload for payload in _load_task_payloads(source_root)
        }
        for payload in payload_lookup[identifier]:
            seed = int(payload["formal_seed"])
            observed_prefix = [
                row for row in payload["rows"] if int(row["iteration"]) <= source_rounds
            ]
            observed_iterations = {int(row["iteration"]) for row in observed_prefix}
            expected_common = [
                row
                for row in source_payloads[seed]["rows"]
                if int(row["iteration"]) in observed_iterations
            ]
            excluded_source_iterations = [
                int(row["iteration"])
                for row in source_payloads[seed]["rows"]
                if int(row["iteration"]) not in observed_iterations
            ]
            passed, maximum, errors = compare_trajectories(
                expected_common, observed_prefix, 1e-6, 1e-8
            )
            if excluded_source_iterations not in ([], [source_rounds]):
                passed = False
                errors.append(
                    f"unexpected source-only checkpoints: {excluded_source_iterations}"
                )
            prefix_rows.append(
                {
                    "candidate_id": identifier,
                    "formal_seed": seed,
                    "source_rounds": source_rounds,
                    "target_rounds": rounds,
                    "passed": passed,
                    "max_abs_difference": maximum,
                    "compared_checkpoint_count": len(expected_common),
                    "excluded_source_iterations": json.dumps(excluded_source_iterations),
                    "errors": json.dumps(errors),
                }
            )
    if not all(row["passed"] for row in prefix_rows):
        raise ValueError("Extension trajectory prefix audit failed.")

    candidate_rows = []
    selections = {}
    advancing_ids = set()
    for method in cfg["methods"]["sfo"]:
        method_rows = [
            summarize_region_candidate(
                candidate,
                payload_lookup[candidate["candidate_id"]],
                "fine",
                representatives,
                consecutive,
            )
            for candidate in candidates
            if candidate["method"] == method
        ]
        selection = select_region(method_rows, top_n)
        selected = next(row for row in candidates if row["candidate_id"] == selection["selected_candidate_id"])
        selection["selected_parameters"] = selected["parameters"]
        selection["selected_stage_rounds"] = rounds
        selections[method] = selection
        if selection["status"] != "full-coverage":
            advancing_ids.update(selection["advancing_candidate_ids"])
        for row in method_rows:
            row["selected"] = row["candidate_id"] == selection["selected_candidate_id"]
            row["advancing"] = row["candidate_id"] in selection["advancing_candidate_ids"]
        candidate_rows.extend(method_rows)

    atomic_write_json(root / f"region_extension_{rounds}_manifest.json", source_manifest)
    atomic_write_csv(root / f"region_extension_{rounds}_prefix_audit.csv", prefix_rows)
    atomic_write_csv(root / f"region_extension_{rounds}_candidate_results.csv", candidate_rows)
    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "source_rounds": source_rounds,
        "rounds": rounds,
        "prefix_audit_passed": True,
        "prefix_max_abs_difference": max(row["max_abs_difference"] for row in prefix_rows),
        "selections": {"fine": selections},
        "fine_resolved": all(value["status"] == "full-coverage" for value in selections.values()),
        "launches_started": 0,
    }
    if rounds != budgets[-1] and advancing_ids:
        next_rounds = budgets[budgets.index(rounds) + 1]
        next_candidates = [
            candidate for candidate in candidates if candidate["candidate_id"] in advancing_ids
        ]
        next_manifest = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "status": "prepared",
            "created_at_utc": utc_now(),
            "source_rounds": rounds,
            "target_rounds": next_rounds,
            "pilot_seeds": seeds,
            "candidates": next_candidates,
            "candidate_count": len(next_candidates),
            "task_count": len(next_candidates) * len(seeds),
            "launches_started": 0,
        }
        next_manifest["manifest_sha256"] = object_sha256(next_manifest)
        atomic_write_json(root / "region_extension_manifest.json", next_manifest)
        report["next_rounds"] = next_rounds
        report["next_candidate_count"] = len(next_candidates)
        report["next_task_count"] = next_manifest["task_count"]
    else:
        report["next_rounds"] = None
        report["next_candidate_count"] = 0
        report["next_task_count"] = 0
    atomic_write_json(root / f"region_extension_{rounds}_analysis.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml"
    )
    parser.add_argument(
        "--phase", choices=["initial", "refinement", "extension"], default="initial"
    )
    parser.add_argument(
        "--output-root",
        default="outputs/distributed_cpu_fo_v2/epsilon_scaling_v2/pilot",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.phase == "initial":
        report = analyze_initial_pilot(cfg, args.output_root)
    elif args.phase == "refinement":
        report = analyze_refinement_and_prepare_extension(cfg, args.output_root)
    else:
        report = analyze_extension_stage(cfg, args.output_root)
    if args.phase == "initial":
        print(
            f"status={report['status']} rows={report['candidate_region_rows']} "
            f"refinement_bases={report['refinement_base_count']} "
            f"refinement_tasks={report['refinement_runnable_task_count']}"
        )
    elif args.phase == "refinement":
        print(
            f"status={report['status']} rows={report['candidate_region_rows']} "
            f"advancing={report['advancing_candidate_count']} "
            f"extension_tasks={report['extension_task_count']}"
        )
    else:
        print(
            f"status={report['status']} rounds={report['rounds']} "
            f"prefix_audit={report['prefix_audit_passed']} "
            f"fine_resolved={report['fine_resolved']} "
            f"next_rounds={report['next_rounds']} next_tasks={report['next_task_count']}"
        )


if __name__ == "__main__":
    main()
