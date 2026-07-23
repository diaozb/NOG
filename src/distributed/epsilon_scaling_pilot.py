"""Adapter for the initial staged pilot of the wide-epsilon protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_pilot import candidate_config, prepare_manifest, run_coarse
from src.distributed.cpu_fo_correctness import process_config_from_experiment
from src.distributed.cpu_fo_tasks import CpuFoTask, atomic_write_json, run_task_set, utc_now
from src.distributed.epsilon_scaling import validate_scaling_protocol


def pilot_runner_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map the new protocol onto the hash-audited legacy coarse runner."""
    validate_scaling_protocol(cfg)
    result = copy.deepcopy(cfg)
    scaling = result["epsilon_scaling"]
    source = scaling["pilot"]
    representatives = []
    for region in ("coarse", "medium", "fine"):
        representatives.extend(
            float(value)
            for value in scaling["regions"][region]["representative_epsilons"]
        )
    result["pilot"] = {
        "seeds": [int(value) for value in scaling["pilot_seeds"]],
        "reference_worker": int(scaling["reference_worker"]),
        "epsilons": representatives,
        "budgets": [int(value) for value in scaling["budgets"]],
        "confirmed_hit_consecutive": int(scaling["confirmed_hit_consecutive"]),
        "top_n_refinement": int(source["top_n_censored"]),
        "nog": {
            "M": [int(value) for value in source["nog"]["M"]],
            "eta": [float(value) for value in source["nog"]["eta"]],
            "coarse_smooth_B": int(source["nog"]["coarse_smooth_B"]),
            "refinement_smooth_B": [
                int(value) for value in source["nog"]["refinement_smooth_B"]
            ],
        },
        "me_dol": {
            "epoch_length": [int(value) for value in source["me_dol"]["epoch_length"]],
            "theory_multiplier": [
                float(value) for value in source["me_dol"]["theory_multiplier"]
            ],
        },
        "estimated_960_seconds": dict(source["estimated_960_seconds"]),
    }
    return result


def run_region_refinement(cfg: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    from src.distributed.epsilon_scaling_pilot_analysis import validate_refinement_manifest

    runner_cfg = pilot_runner_config(cfg)
    root = Path(output_root)
    with open(root / "region_batch_refinement_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    variants = validate_refinement_manifest(cfg, manifest)
    seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    rounds = int(manifest["rounds"])
    worker = int(cfg["epsilon_scaling"]["reference_worker"])
    process = process_config_from_experiment(runner_cfg)
    records = []
    for variant in variants:
        current = candidate_config(runner_cfg, variant, rounds)
        variant_root = root / "region_batch_refinement" / variant["candidate_id"]
        completion = run_task_set(
            current,
            [CpuFoTask("NOG-FO", seed, worker) for seed in seeds],
            variant_root,
            process,
            continue_on_error=True,
        )
        records.append(
            {
                "candidate_id": variant["candidate_id"],
                "status": completion["status"],
                "completed_tasks": completion["completed_tasks"],
                "failed_tasks": completion["failed_tasks"],
            }
        )
        print(
            f"refinement={variant['candidate_id']} status={completion['status']} "
            f"tasks={completion['completed_tasks']}/{len(seeds)}",
            flush=True,
        )
    result = {
        "schema_version": 1,
        "status": "complete" if all(row["status"] == "complete" for row in records) else "incomplete",
        "created_at_utc": utc_now(),
        "expected_tasks": len(variants) * len(seeds),
        "completed_tasks": sum(row["completed_tasks"] for row in records),
        "failed_tasks": sum(row["failed_tasks"] for row in records),
        "records": records,
    }
    atomic_write_json(root / "region_batch_refinement_completion.json", result)
    return result


def run_region_extension(cfg: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    from src.distributed.epsilon_scaling_pilot_analysis import validate_extension_manifest

    runner_cfg = pilot_runner_config(cfg)
    root = Path(output_root)
    with open(root / "region_extension_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    candidates = validate_extension_manifest(cfg, manifest)
    seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    rounds = int(manifest["target_rounds"])
    worker = int(cfg["epsilon_scaling"]["reference_worker"])
    process = process_config_from_experiment(runner_cfg)
    records = []
    for candidate in candidates:
        current = candidate_config(runner_cfg, candidate, rounds)
        candidate_root = root / f"region_extension_{rounds}" / candidate["candidate_id"]
        completion = run_task_set(
            current,
            [CpuFoTask(candidate["method"], seed, worker) for seed in seeds],
            candidate_root,
            process,
            continue_on_error=True,
        )
        records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": completion["status"],
                "completed_tasks": completion["completed_tasks"],
                "failed_tasks": completion["failed_tasks"],
            }
        )
        print(
            f"extension={candidate['candidate_id']} status={completion['status']} "
            f"tasks={completion['completed_tasks']}/{len(seeds)}",
            flush=True,
        )
    result = {
        "schema_version": 1,
        "status": "complete" if all(row["status"] == "complete" for row in records) else "incomplete",
        "created_at_utc": utc_now(),
        "rounds": rounds,
        "expected_tasks": len(candidates) * len(seeds),
        "completed_tasks": sum(row["completed_tasks"] for row in records),
        "failed_tasks": sum(row["failed_tasks"] for row in records),
        "records": records,
    }
    atomic_write_json(root / f"region_extension_{rounds}_completion.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml"
    )
    parser.add_argument(
        "--output-root",
        default="outputs/distributed_cpu_fo_v2/epsilon_scaling_v2/pilot",
    )
    parser.add_argument(
        "--phase",
        choices=["prepare", "run-initial", "run-refinement", "run-extension"],
        default="prepare",
    )
    args = parser.parse_args()
    cfg = pilot_runner_config(load_config(args.config))
    root = Path(args.output_root)
    if args.phase == "prepare":
        manifest = prepare_manifest(cfg, root)
        print(
            f"status={manifest['status']} candidates={manifest['candidate_count']} "
            f"tasks={manifest['coarse_task_count']} "
            f"estimated_hours={manifest['estimated_coarse_hours']:.2f} launches_started=0"
        )
        return
    if args.phase == "run-refinement":
        result = run_region_refinement(load_config(args.config), root)
        print(
            f"status={result['status']} tasks={result['completed_tasks']}/"
            f"{result['expected_tasks']} failures={result['failed_tasks']}"
        )
        if result["status"] != "complete":
            raise SystemExit(1)
        return
    if args.phase == "run-extension":
        result = run_region_extension(load_config(args.config), root)
        print(
            f"status={result['status']} tasks={result['completed_tasks']}/"
            f"{result['expected_tasks']} failures={result['failed_tasks']}"
        )
        if result["status"] != "complete":
            raise SystemExit(1)
        return
    result = run_coarse(cfg, root)
    print(
        f"status={result['status']} candidates={len(result['records'])} "
        f"launches_started={len(result['records'])}"
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
