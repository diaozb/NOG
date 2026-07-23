"""Freeze pilot selections and prepare the formal wide-epsilon task manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import (
    atomic_write_json,
    file_sha256,
    object_sha256,
    utc_now,
)
from src.distributed.epsilon_scaling import epsilon_region, validate_scaling_protocol


FREEZE_SCHEMA_VERSION = 1
FREEZE_RULE_VERSION = "max-budget-censor-aware-v1"


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def freeze_selections(cfg: Dict[str, Any], pilot_root: Path) -> Dict[str, Any]:
    """Freeze all regions without silently converting censored pilots to hits."""
    validate_scaling_protocol(cfg)
    refined_path = pilot_root / "refined_region_selection.json"
    fine_path = pilot_root / "region_extension_61440_analysis.json"
    refined = _load(refined_path)
    fine = _load(fine_path)
    if refined.get("status") != "complete" or fine.get("status") != "complete":
        raise ValueError("Pilot selection inputs must be complete.")
    if not fine.get("prefix_audit_passed") or fine.get("prefix_max_abs_difference") != 0.0:
        raise ValueError("The 61440-step prefix audit did not pass exactly.")
    budgets = [int(value) for value in cfg["epsilon_scaling"]["budgets"]]
    if int(fine["rounds"]) != max(budgets) or fine.get("next_rounds") is not None:
        raise ValueError("Fine selection is not the terminal maximum-budget result.")

    frozen: Dict[str, Dict[str, Any]] = {}
    for region in ("coarse", "medium", "fine"):
        source = fine["selections"]["fine"] if region == "fine" else refined["selections"][region]
        frozen[region] = {}
        for method in cfg["methods"]["sfo"]:
            selected = source[method]
            status = str(selected["status"])
            if region != "fine" and status != "full-coverage":
                raise ValueError(f"{region}/{method} is not full-coverage.")
            if region == "fine" and status not in {"full-coverage", "censored-top-candidates"}:
                raise ValueError(f"Unsupported terminal fine status: {status}.")
            frozen[region][method] = {
                "candidate_id": selected["selected_candidate_id"],
                "parameters": selected["selected_parameters"],
                "rounds": int(selected["selected_stage_rounds"]),
                "selection_status": status,
                "pilot_coverage": float(selected["selected_coverage"]),
                "censored_at_freeze": status != "full-coverage",
            }

    result = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "freeze_rule_version": FREEZE_RULE_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "launches_started": 0,
        "pilot_seeds": [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]],
        "formal_seeds_disjoint": not bool(
            set(cfg["run"]["formal_seeds"]) & set(cfg["epsilon_scaling"]["pilot_seeds"])
        ),
        "regions": frozen,
        "source_files": {
            refined_path.name: file_sha256(refined_path),
            fine_path.name: file_sha256(fine_path),
        },
        "notes": [
            "Formal seeds were not used for candidate selection.",
            "Fine parameters are frozen at the preregistered maximum budget.",
            "A censored freeze remains censored and is not reported as a hit.",
        ],
    }
    if not result["formal_seeds_disjoint"]:
        raise ValueError("Pilot and formal seeds overlap.")
    result["freeze_sha256"] = object_sha256(result)
    return result


def prepare_formal_manifest(cfg: Dict[str, Any], freeze: Dict[str, Any]) -> Dict[str, Any]:
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    worker = int(cfg["epsilon_scaling"]["reference_worker"])
    epsilons = [float(value) for value in cfg["epsilon_scaling"]["epsilons"]]
    groups: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    for method in cfg["methods"]["sfo"]:
        for region in ("coarse", "medium", "fine"):
            selection = freeze["regions"][region][method]
            region_eps = [value for value in epsilons if epsilon_region(value) == region]
            group_id = f"{method}__{region}"
            groups.append(
                {
                    "group_id": group_id,
                    "method": method,
                    "region": region,
                    "worker_count": worker,
                    "rounds": selection["rounds"],
                    "epsilons": region_eps,
                    "parameters": selection["parameters"],
                    "selection_status": selection["selection_status"],
                    "pilot_coverage": selection["pilot_coverage"],
                    "formal_seeds": seeds,
                }
            )
            tasks.extend(
                {
                    "task_id": f"{group_id}__seed{seed}",
                    "group_id": group_id,
                    "method": method,
                    "region": region,
                    "formal_seed": seed,
                    "worker_count": worker,
                    "rounds": selection["rounds"],
                    "epsilons": region_eps,
                    "parameters": selection["parameters"],
                }
                for seed in seeds
            )
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "launches_started": 0,
        "freeze_sha256": freeze["freeze_sha256"],
        "trajectory_groups": groups,
        "tasks": tasks,
        "physical_task_count": len(tasks),
        "logical_method_epsilon_seed_rows": len(cfg["methods"]["sfo"]) * len(epsilons) * len(seeds),
        "max_total_worker_processes": int(cfg["cpu_process"]["max_total_worker_processes"]),
        "task_timeout_seconds": int(cfg["cpu_process"]["task_timeout_seconds"]),
    }
    manifest["manifest_sha256"] = object_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    parser.add_argument("--pilot-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    pilot_root = Path(args.pilot_root) if args.pilot_root else root / "pilot"
    freeze = freeze_selections(cfg, pilot_root)
    manifest = prepare_formal_manifest(cfg, freeze)
    atomic_write_json(root / "frozen_region_configs.json", freeze)
    atomic_write_json(root / "formal_manifest.json", manifest)
    group_count = len(manifest["trajectory_groups"])
    physical_count = manifest["physical_task_count"]
    logical_count = manifest["logical_method_epsilon_seed_rows"]
    print(
        f"status=frozen groups={group_count} "
        f"physical_tasks={physical_count} "
        f"logical_rows={logical_count}"
    )


if __name__ == "__main__":
    main()
