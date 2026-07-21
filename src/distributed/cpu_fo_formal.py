"""Frozen-config formal accuracy runner preparation.

Step 7A validates and deduplicates the pilot-selected configurations.  It
writes an auditable task manifest but never launches worker processes.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.common import validate_experiment_config
from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_correctness import process_config_from_experiment
from src.distributed.cpu_fo_pilot import (
    PILOT_SCHEMA_VERSION,
    candidate_config,
    candidate_id,
    validate_pilot_config,
)
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    effective_task_config,
    object_sha256,
    run_task_set,
    utc_now,
)


FORMAL_SCHEMA_VERSION = 1


def validate_frozen_selection(
    cfg: Dict[str, Any],
    frozen: Dict[str, Any],
    pilot_report: Dict[str, Any],
) -> None:
    validate_pilot_config(cfg)
    if frozen.get("status") != "frozen":
        raise ValueError("Formal selection must have status=frozen.")
    observed_hash = frozen.get("frozen_config_sha256")
    hash_payload = copy.deepcopy(frozen)
    hash_payload.pop("frozen_config_sha256", None)
    expected_hash = object_sha256(hash_payload)
    if observed_hash != expected_hash:
        raise ValueError("Frozen config SHA256 validation failed.")
    if pilot_report.get("status") != "complete" or not pilot_report.get(
        "pilot_complete", False
    ):
        raise ValueError("Pilot report is not complete.")
    if int(pilot_report.get("formal_runs_started", -1)) != 0:
        raise ValueError("Pilot report says formal runs already started.")
    if pilot_report.get("frozen_config_sha256") != observed_hash:
        raise ValueError("Pilot report/frozen config SHA256 mismatch.")

    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    pilot_seeds = [int(value) for value in cfg["pilot"]["seeds"]]
    if [int(value) for value in frozen.get("formal_seeds", [])] != formal_seeds:
        raise ValueError("Frozen formal seeds do not match config.")
    if [int(value) for value in frozen.get("pilot_seeds", [])] != pilot_seeds:
        raise ValueError("Frozen pilot seeds do not match config.")
    if set(formal_seeds) & set(pilot_seeds):
        raise ValueError("Formal and pilot seeds overlap.")
    if int(frozen.get("reference_worker", -1)) != int(
        cfg["pilot"]["reference_worker"]
    ):
        raise ValueError("Frozen reference worker does not match config.")
    if float(frozen.get("delta", -1.0)) != float(cfg["oracle"]["delta"]):
        raise ValueError("Frozen delta does not match config.")
    if frozen.get("evaluation_seed_mode") != "fixed_bank":
        raise ValueError("Formal runs require the frozen fixed evaluation bank.")

    expected_epsilons = {f"{float(value):g}" for value in cfg["pilot"]["epsilons"]}
    by_epsilon = frozen.get("by_epsilon", {})
    if set(by_epsilon) != expected_epsilons:
        raise ValueError("Frozen epsilon coverage does not match pilot config.")
    for epsilon, methods in by_epsilon.items():
        if set(methods) != {"NOG-FO", "ME-DOL-FO"}:
            raise ValueError(f"Frozen methods are incomplete for epsilon={epsilon}.")
        for method, selection in methods.items():
            if not isinstance(selection.get("selected_parameters"), dict):
                raise ValueError(f"Frozen parameters missing for {method}/{epsilon}.")
            rounds = int(selection.get("selected_stage_rounds", -1))
            if rounds not in {int(value) for value in cfg["pilot"]["budgets"]}:
                raise ValueError(f"Invalid frozen rounds for {method}/{epsilon}.")


def unique_formal_configs(
    cfg: Dict[str, Any],
    frozen: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deduplicate trajectories shared by several epsilon thresholds."""

    lookup: Dict[str, Dict[str, Any]] = {}
    for epsilon, methods in sorted(
        frozen["by_epsilon"].items(),
        key=lambda item: float(item[0]),
        reverse=True,
    ):
        for method, selection in sorted(methods.items()):
            parameters = copy.deepcopy(selection["selected_parameters"])
            rounds = int(selection["selected_stage_rounds"])
            identity_payload = {
                "method": method,
                "parameters": parameters,
                "rounds": rounds,
            }
            identity_hash = object_sha256(identity_payload)
            identifier = (
                f"{candidate_id(method, parameters)}__rounds-{rounds}"
                f"__frozen-{identity_hash[:12]}"
            )
            record = lookup.setdefault(
                identity_hash,
                {
                    "formal_config_id": identifier,
                    "formal_config_sha256": identity_hash,
                    "method": method,
                    "parameters": parameters,
                    "rounds": rounds,
                    "epsilons": [],
                    "pilot_selection_status_by_epsilon": {},
                },
            )
            record["epsilons"].append(float(epsilon))
            record["pilot_selection_status_by_epsilon"][epsilon] = selection[
                "status"
            ]
    configs = list(lookup.values())
    for record in configs:
        record["epsilons"] = sorted(record["epsilons"], reverse=True)
        candidate = {
            "method": record["method"],
            "parameters": record["parameters"],
        }
        candidate_config(cfg, candidate, int(record["rounds"]))
    return sorted(configs, key=lambda row: (row["method"], row["formal_config_id"]))


def _pilot_partial_pattern(
    pilot_root: Path,
    formal_config: Dict[str, Any],
) -> str:
    method = formal_config["method"]
    rounds = int(formal_config["rounds"])
    identifier = candidate_id(method, formal_config["parameters"])
    if method == "ME-DOL-FO":
        return str(
            pilot_root
            / "extension"
            / f"rounds_{rounds}"
            / identifier
            / "partials"
            / "*.json"
        )
    smooth_batch = int(formal_config["parameters"]["smooth_B"])
    coarse_batch = 2
    if smooth_batch == coarse_batch:
        return str(
            pilot_root
            / "coarse"
            / identifier
            / f"rounds_{rounds}"
            / "partials"
            / "*.json"
        )
    return str(
        pilot_root
        / "batch_refinement"
        / f"rounds_{rounds}"
        / identifier
        / "partials"
        / "*.json"
    )


def add_runtime_estimates(
    formal_configs: List[Dict[str, Any]],
    pilot_root: Path,
    formal_seed_count: int,
) -> None:
    import glob

    for record in formal_configs:
        paths = sorted(glob.glob(_pilot_partial_pattern(pilot_root, record)))
        if not paths:
            raise FileNotFoundError(
                f"No pilot timing partials for {record['formal_config_id']}."
            )
        observed = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            observed.append(float(payload["launch"]["end_to_end_time"]))
        median_seconds = statistics.median(observed)
        record["pilot_timing_samples"] = len(observed)
        record["pilot_median_end_to_end_seconds"] = median_seconds
        record["estimated_formal_seconds"] = median_seconds * formal_seed_count


def prepare_formal_manifest(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    pilot_path = Path(pilot_root)
    formal_path = Path(output_root)
    with open(
        pilot_path / "selected_config_by_epsilon.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        frozen = json.load(handle)
    with open(
        pilot_path / "pilot_final_report.json",
        "r",
        encoding="utf-8",
    ) as handle:
        pilot_report = json.load(handle)
    validate_frozen_selection(cfg, frozen, pilot_report)
    validate_experiment_config(cfg)

    configs = unique_formal_configs(cfg, frozen)
    formal_seeds = [int(value) for value in frozen["formal_seeds"]]
    add_runtime_estimates(configs, pilot_path, len(formal_seeds))
    for record in configs:
        record["task_count"] = len(formal_seeds)
        record["output_root"] = str(
            Path("raw") / record["formal_config_id"]
        )
        record["tasks"] = [
            {
                "method": record["method"],
                "formal_seed": seed,
                "worker_count": int(frozen["reference_worker"]),
            }
            for seed in formal_seeds
        ]

    by_method = {
        method: sum(row["method"] == method for row in configs)
        for method in ("NOG-FO", "ME-DOL-FO")
    }
    estimated_seconds = sum(
        float(row["estimated_formal_seconds"]) for row in configs
    )
    manifest = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "source_frozen_config": str(
            (pilot_path / "selected_config_by_epsilon.yaml").resolve()
        ),
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "pilot_seeds_excluded": [int(value) for value in frozen["pilot_seeds"]],
        "formal_seeds": formal_seeds,
        "reference_worker": int(frozen["reference_worker"]),
        "epsilon_count": len(frozen["by_epsilon"]),
        "method_epsilon_pairs": len(frozen["by_epsilon"]) * 2,
        "unique_config_count": len(configs),
        "unique_config_count_by_method": by_method,
        "task_count": len(configs) * len(formal_seeds),
        "raw_estimated_seconds": estimated_seconds,
        "raw_estimated_hours": estimated_seconds / 3600.0,
        "conservative_estimated_hours": estimated_seconds * 1.5 / 3600.0,
        "formal_configs": configs,
        "formal_manifest_sha256": object_sha256(
            {
                "frozen_config_sha256": frozen["frozen_config_sha256"],
                "formal_seeds": formal_seeds,
                "formal_configs": [
                    {
                        key: row[key]
                        for key in (
                            "formal_config_id",
                            "formal_config_sha256",
                            "method",
                            "parameters",
                            "rounds",
                            "epsilons",
                        )
                    }
                    for row in configs
                ],
            }
        ),
        "launches_started": 0,
    }
    atomic_write_json(formal_path / "base_config.json", effective_task_config(cfg))
    atomic_write_json(formal_path / "formal_task_manifest.json", manifest)
    return manifest


def validate_formal_manifest(
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
    frozen: Dict[str, Any],
    pilot_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    validate_frozen_selection(cfg, frozen, pilot_report)
    if manifest.get("status") != "prepared":
        raise ValueError("Formal task manifest must have status=prepared.")
    if int(manifest.get("launches_started", -1)) != 0:
        raise ValueError("Formal manifest must be frozen before launch.")
    if manifest.get("frozen_config_sha256") != frozen["frozen_config_sha256"]:
        raise ValueError("Formal/frozen config SHA256 mismatch.")
    formal_seeds = [int(value) for value in frozen["formal_seeds"]]
    if [int(value) for value in manifest.get("formal_seeds", [])] != formal_seeds:
        raise ValueError("Formal manifest seeds mismatch.")
    if set(formal_seeds) & {
        int(value) for value in manifest.get("pilot_seeds_excluded", [])
    }:
        raise ValueError("Formal manifest includes a pilot seed.")
    worker = int(frozen["reference_worker"])
    if int(manifest.get("reference_worker", -1)) != worker:
        raise ValueError("Formal manifest worker mismatch.")

    expected_configs = unique_formal_configs(cfg, frozen)
    expected_lookup = {row["formal_config_id"]: row for row in expected_configs}
    configs = manifest.get("formal_configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Formal manifest has no configs.")
    if int(manifest.get("unique_config_count", -1)) != len(configs):
        raise ValueError("Formal unique config count mismatch.")
    if int(manifest.get("task_count", -1)) != len(configs) * len(formal_seeds):
        raise ValueError("Formal task count mismatch.")
    observed_ids = [row.get("formal_config_id") for row in configs]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Formal manifest contains duplicate config IDs.")
    if set(observed_ids) != set(expected_lookup):
        raise ValueError("Formal manifest config coverage mismatch.")
    for record in configs:
        expected = expected_lookup[record["formal_config_id"]]
        for key in (
            "formal_config_sha256",
            "method",
            "parameters",
            "rounds",
            "epsilons",
        ):
            if record.get(key) != expected[key]:
                raise ValueError(
                    f"Formal config field {key} mismatch for "
                    f"{record['formal_config_id']}."
                )
        tasks = record.get("tasks")
        expected_tasks = [
            {
                "method": record["method"],
                "formal_seed": seed,
                "worker_count": worker,
            }
            for seed in formal_seeds
        ]
        if tasks != expected_tasks:
            raise ValueError(
                f"Formal task identities mismatch for {record['formal_config_id']}."
            )
    expected_manifest_hash = object_sha256(
        {
            "frozen_config_sha256": frozen["frozen_config_sha256"],
            "formal_seeds": formal_seeds,
            "formal_configs": [
                {
                    key: row[key]
                    for key in (
                        "formal_config_id",
                        "formal_config_sha256",
                        "method",
                        "parameters",
                        "rounds",
                        "epsilons",
                    )
                }
                for row in configs
            ],
        }
    )
    if manifest.get("formal_manifest_sha256") != expected_manifest_hash:
        raise ValueError("Formal manifest SHA256 validation failed.")
    return configs


def run_formal_tasks(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    pilot_path = Path(pilot_root)
    formal_path = Path(output_root)
    with open(
        pilot_path / "selected_config_by_epsilon.yaml",
        "r",
        encoding="utf-8",
    ) as handle:
        frozen = json.load(handle)
    with open(
        pilot_path / "pilot_final_report.json",
        "r",
        encoding="utf-8",
    ) as handle:
        pilot_report = json.load(handle)
    with open(
        formal_path / "formal_task_manifest.json",
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)
    configs = validate_formal_manifest(cfg, manifest, frozen, pilot_report)
    process = process_config_from_experiment(cfg)
    records = []
    for index, record in enumerate(configs, start=1):
        print(
            f"formal_config={index}/{len(configs)} "
            f"id={record['formal_config_id']} status=starting",
            flush=True,
        )
        candidate = {
            "method": record["method"],
            "parameters": record["parameters"],
        }
        current_cfg = candidate_config(cfg, candidate, int(record["rounds"]))
        config_root = formal_path / record["output_root"]
        completion = run_task_set(
            current_cfg,
            [
                CpuFoTask(
                    task["method"],
                    int(task["formal_seed"]),
                    int(task["worker_count"]),
                )
                for task in record["tasks"]
            ],
            config_root,
            process,
            continue_on_error=True,
        )
        completion_record = {
            "formal_config_id": record["formal_config_id"],
            "formal_config_sha256": record["formal_config_sha256"],
            "method": record["method"],
            "rounds": int(record["rounds"]),
            "epsilons": record["epsilons"],
            "status": completion["status"],
            "completed_tasks": completion["completed_tasks"],
            "failed_tasks": completion["failed_tasks"],
            "output_root": record["output_root"],
        }
        records.append(completion_record)
        print(
            f"formal_config={index}/{len(configs)} "
            f"id={record['formal_config_id']} status={completion['status']} "
            f"tasks={completion['completed_tasks']}/{len(record['tasks'])} "
            f"failures={completion['failed_tasks']}",
            flush=True,
        )
    result = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": (
            "complete"
            if all(row["status"] == "complete" for row in records)
            else "incomplete"
        ),
        "created_at_utc": utc_now(),
        "source_manifest": "formal_task_manifest.json",
        "formal_manifest_sha256": manifest["formal_manifest_sha256"],
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "unique_config_count": len(configs),
        "expected_tasks": int(manifest["task_count"]),
        "completed_tasks": sum(row["completed_tasks"] for row in records),
        "failed_tasks": sum(row["failed_tasks"] for row in records),
        "records": records,
    }
    atomic_write_json(formal_path / "formal_completion.json", result)
    from src.distributed.cpu_fo_correctness import atomic_write_csv

    atomic_write_csv(formal_path / "formal_completion.csv", records)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-root", default="outputs/distributed_cpu_fo/pilot")
    parser.add_argument(
        "--output-root",
        default="outputs/distributed_cpu_fo/formal_accuracy",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", choices=["prepare", "run"], default="prepare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.phase == "prepare" or args.dry_run:
        manifest = prepare_formal_manifest(
            cfg,
            args.pilot_root,
            args.output_root,
        )
        print(
            f"phase=formal-prepare status={manifest['status']} "
            f"unique_configs={manifest['unique_config_count']} "
            f"tasks={manifest['task_count']} "
            f"raw_estimated_hours={manifest['raw_estimated_hours']:.2f} "
            f"conservative_estimated_hours={manifest['conservative_estimated_hours']:.2f}"
        )
        print(f"output_root={args.output_root}")
        print("launches_started=0; no formal worker processes launched")
        return
    result = run_formal_tasks(cfg, args.pilot_root, args.output_root)
    print(
        f"phase=formal-run status={result['status']} "
        f"tasks={result['completed_tasks']}/{result['expected_tasks']} "
        f"failures={result['failed_tasks']}"
    )


if __name__ == "__main__":
    main()
