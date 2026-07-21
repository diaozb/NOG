"""Prepare the frozen Step 8 CPU runtime-scaling benchmark.

Step 8A validates all upstream artifacts, deduplicates identical workloads,
and writes an auditable task order.  It never launches worker processes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.distributed.cpu_fo_correctness import (
    atomic_write_csv,
    load_config,
    process_config_from_experiment,
)
from src.distributed.cpu_fo_formal import validate_frozen_selection
from src.distributed.cpu_fo_pilot import candidate_config, candidate_id
from src.distributed.cpu_fo_profile import cpu_resource_record
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    effective_task_config,
    file_sha256,
    object_sha256,
    run_task_set,
    utc_now,
)


RUNTIME_SCHEMA_VERSION = 1
EXPECTED_EPSILONS = [0.010, 0.009, 0.008]
EXPECTED_METHODS = ["NOG-FO", "ME-DOL-FO"]
EXPECTED_WORKERS = [1, 2, 4, 8, 16, 32]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def validate_runtime_protocol(
    cfg: Dict[str, Any],
    frozen: Dict[str, Any],
    pilot_report: Dict[str, Any],
    step7_completion: Dict[str, Any],
) -> None:
    validate_frozen_selection(cfg, frozen, pilot_report)
    runtime = cfg.get("runtime", {})
    if [float(value) for value in runtime.get("epsilons", [])] != EXPECTED_EPSILONS:
        raise ValueError("Runtime epsilon protocol differs from the frozen plan.")
    if list(runtime.get("methods", [])) != EXPECTED_METHODS:
        raise ValueError("Runtime method protocol differs from the frozen plan.")
    if [int(value) for value in runtime.get("workers", [])] != EXPECTED_WORKERS:
        raise ValueError("Runtime worker protocol differs from the frozen plan.")
    if int(runtime.get("benchmark_seed", -1)) != 0:
        raise ValueError("Runtime benchmark seed must remain zero.")
    if int(runtime.get("repeats", -1)) != 3:
        raise ValueError("Runtime repeats must remain three.")
    if runtime.get("warmup_mode") != "one_update_unit":
        raise ValueError("Runtime warm-up protocol changed.")
    if not bool(runtime.get("reuse_identical_configs", False)):
        raise ValueError("Identical runtime workloads must be deduplicated.")
    if runtime.get("order_rule") != "alternating-workers-rotating-config-v1":
        raise ValueError("Runtime order rule changed.")
    if step7_completion.get("status") != "complete":
        raise ValueError("Step 7 must be complete before runtime preparation.")
    if step7_completion.get("frozen_config_sha256") != frozen.get(
        "frozen_config_sha256"
    ):
        raise ValueError("Step 7/frozen config SHA256 mismatch.")
    if int(step7_completion.get("verified_figure_count", -1)) != 8:
        raise ValueError("Step 7 figure package is incomplete.")


def unique_runtime_configs(
    cfg: Dict[str, Any],
    frozen: Dict[str, Any],
) -> List[Dict[str, Any]]:
    runtime_epsilons = [float(value) for value in cfg["runtime"]["epsilons"]]
    lookup: Dict[str, Dict[str, Any]] = {}
    for epsilon in runtime_epsilons:
        epsilon_key = f"{epsilon:g}"
        for method in cfg["runtime"]["methods"]:
            selection = frozen["by_epsilon"][epsilon_key][method]
            parameters = copy.deepcopy(selection["selected_parameters"])
            rounds = int(selection["selected_stage_rounds"])
            identity = {
                "method": method,
                "parameters": parameters,
                "rounds": rounds,
            }
            digest = object_sha256(identity)
            record = lookup.setdefault(
                digest,
                {
                    "runtime_config_id": (
                        f"{candidate_id(method, parameters)}__rounds-{rounds}"
                        f"__runtime-{digest[:12]}"
                    ),
                    "runtime_config_sha256": digest,
                    "method": method,
                    "parameters": parameters,
                    "rounds": rounds,
                    "epsilons": [],
                    "selection_status_by_epsilon": {},
                },
            )
            record["epsilons"].append(epsilon)
            record["selection_status_by_epsilon"][epsilon_key] = selection[
                "status"
            ]
    configs = list(lookup.values())
    for record in configs:
        record["epsilons"] = sorted(record["epsilons"], reverse=True)
        candidate_config(
            cfg,
            {"method": record["method"], "parameters": record["parameters"]},
            int(record["rounds"]),
        )
        record["warmup_rounds"] = (
            int(record["parameters"]["M"])
            if record["method"] == "NOG-FO"
            else int(record["parameters"]["epoch_length"])
        )
    return sorted(
        configs,
        key=lambda row: (
            row["method"],
            -max(row["epsilons"]),
            row["runtime_config_id"],
        ),
    )


def _rotating_config_order(
    configs: List[Dict[str, Any]],
    offset: int,
) -> List[Dict[str, Any]]:
    nog = sorted(
        (row for row in configs if row["method"] == "NOG-FO"),
        key=lambda row: -max(row["epsilons"]),
    )
    me = [row for row in configs if row["method"] == "ME-DOL-FO"]
    if len(nog) != 3 or len(me) != 1:
        raise ValueError("Expected three unique NOG configs and one ME-DOL config.")
    rotated_nog = nog[offset % len(nog) :] + nog[: offset % len(nog)]
    insertion = offset % (len(rotated_nog) + 1)
    return rotated_nog[:insertion] + me + rotated_nog[insertion:]


def runtime_task_order(
    cfg: Dict[str, Any],
    configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    workers = [int(value) for value in cfg["runtime"]["workers"]]
    repeats = int(cfg["runtime"]["repeats"])
    seed = int(cfg["runtime"]["benchmark_seed"])
    rows: List[Dict[str, Any]] = []

    def append_task(
        phase: str,
        runtime_config: Dict[str, Any],
        worker: int,
        repeat: int | None,
        rounds: int,
    ) -> None:
        identity = {
            "phase": phase,
            "runtime_config_sha256": runtime_config["runtime_config_sha256"],
            "worker_count": worker,
            "benchmark_seed": seed,
            "repeat": repeat,
            "rounds": rounds,
        }
        task_hash = object_sha256(identity)
        repeat_label = "warmup" if repeat is None else f"repeat_{repeat}"
        rows.append(
            {
                "sequence": len(rows),
                "runtime_task_id": f"runtime-task-{task_hash[:16]}",
                "runtime_task_sha256": task_hash,
                "phase": phase,
                "include_in_summary": phase == "measured",
                "repeat": repeat,
                "runtime_config_id": runtime_config["runtime_config_id"],
                "runtime_config_sha256": runtime_config[
                    "runtime_config_sha256"
                ],
                "method": runtime_config["method"],
                "epsilons": runtime_config["epsilons"],
                "parameters": runtime_config["parameters"],
                "full_rounds": int(runtime_config["rounds"]),
                "rounds": rounds,
                "worker_count": worker,
                "benchmark_seed": seed,
                "output_root": str(
                    Path("raw")
                    / runtime_config["runtime_config_id"]
                    / f"m{worker}"
                    / repeat_label
                ),
            }
        )

    for worker_index, worker in enumerate(workers):
        for runtime_config in _rotating_config_order(configs, worker_index):
            append_task(
                "warmup",
                runtime_config,
                worker,
                None,
                int(runtime_config["warmup_rounds"]),
            )

    for repeat in range(repeats):
        repeat_workers = workers if repeat % 2 == 0 else list(reversed(workers))
        for worker_index, worker in enumerate(repeat_workers):
            for runtime_config in _rotating_config_order(
                configs, repeat + worker_index
            ):
                append_task(
                    "measured",
                    runtime_config,
                    worker,
                    repeat,
                    int(runtime_config["rounds"]),
                )
    return rows


def _profile_rows(profile_csv: Path) -> List[Dict[str, Any]]:
    with open(profile_csv, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _formal_m8_medians(
    formal_root: Path,
    configs: Iterable[Dict[str, Any]],
) -> Dict[str, float]:
    formal_manifest = _load_json(formal_root / "formal_task_manifest.json")
    formal_lookup = {
        row["formal_config_sha256"]: row
        for row in formal_manifest["formal_configs"]
    }
    medians: Dict[str, float] = {}
    for runtime_config in configs:
        digest = runtime_config["runtime_config_sha256"]
        formal_config = formal_lookup.get(digest)
        if formal_config is None:
            raise ValueError(
                f"Runtime config missing from formal manifest: {digest}."
            )
        config_root = formal_root / formal_config["output_root"]
        completion = _load_json(config_root / "completion_manifest.json")
        observed = []
        for record in completion["records"]:
            payload = _load_json(config_root / record["partial_path"])
            observed.append(float(payload["launch"]["end_to_end_time"]))
        if len(observed) != 5:
            raise ValueError("Runtime estimate requires five formal m=8 samples.")
        medians[digest] = statistics.median(observed)
    return medians


def add_time_estimates(
    configs: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    formal_root: Path,
    profile_csv: Path,
    conservative_multiplier: float,
) -> Dict[str, Any]:
    formal_medians = _formal_m8_medians(formal_root, configs)
    profiles = _profile_rows(profile_csv)
    end_to_end = {
        (row["method"], int(row["rounds"]), int(row["worker_count"])): float(
            row["end_to_end_time"]
        )
        for row in profiles
    }
    ratios: Dict[tuple[str, int], float] = {}
    for method in EXPECTED_METHODS:
        reference = end_to_end[(method, 96, 8)]
        for worker in EXPECTED_WORKERS:
            ratios[(method, worker)] = end_to_end[(method, 96, worker)] / reference

    estimated_rows = []
    for task in tasks:
        if task["phase"] == "warmup":
            seconds = end_to_end[(task["method"], 12, task["worker_count"])]
            source = "step5-12-round-end-to-end"
        else:
            seconds = (
                formal_medians[task["runtime_config_sha256"]]
                * ratios[(task["method"], task["worker_count"])]
            )
            source = "formal-m8-median-times-step5-96-round-scaling-ratio"
        task["estimated_end_to_end_seconds"] = seconds
        task["estimate_source"] = source
        estimated_rows.append(seconds)
    raw_seconds = sum(estimated_rows)
    return {
        "estimate_rule": "formal-m8-anchor-step5-scaling-v1",
        "raw_seconds": raw_seconds,
        "raw_hours": raw_seconds / 3600.0,
        "conservative_multiplier": conservative_multiplier,
        "conservative_seconds": raw_seconds * conservative_multiplier,
        "conservative_hours": raw_seconds * conservative_multiplier / 3600.0,
        "formal_m8_median_end_to_end_by_config": formal_medians,
        "step5_m_over_m8_end_to_end_ratio": {
            f"{method}__m{worker}": ratios[(method, worker)]
            for method in EXPECTED_METHODS
            for worker in EXPECTED_WORKERS
        },
        "caveats": [
            "Step 5 scaling ratios came from one diagnostic seed and a different short configuration.",
            "System load and process startup can increase observed wall time.",
            "Warm-ups are separate short process launches and are excluded from summaries.",
        ],
    }


def validate_runtime_manifest(
    cfg: Dict[str, Any],
    frozen: Dict[str, Any],
    pilot_report: Dict[str, Any],
    step7_completion: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    validate_runtime_protocol(cfg, frozen, pilot_report, step7_completion)
    if manifest.get("status") != "prepared":
        raise ValueError("Runtime manifest must have status=prepared.")
    if int(manifest.get("launches_started", -1)) != 0:
        raise ValueError("Runtime manifest must be frozen before launch.")
    if manifest.get("frozen_config_sha256") != frozen.get(
        "frozen_config_sha256"
    ):
        raise ValueError("Runtime/frozen config SHA256 mismatch.")
    if manifest.get("runtime_protocol") != cfg["runtime"]:
        raise ValueError("Runtime manifest protocol mismatch.")

    expected_configs = unique_runtime_configs(cfg, frozen)
    observed_configs = manifest.get("runtime_configs")
    if observed_configs != expected_configs:
        raise ValueError("Runtime manifest config coverage/identity mismatch.")
    expected_tasks = runtime_task_order(cfg, expected_configs)
    observed_tasks = manifest.get("tasks")
    if not isinstance(observed_tasks, list) or len(observed_tasks) != len(
        expected_tasks
    ):
        raise ValueError("Runtime manifest task count mismatch.")
    for expected, observed in zip(expected_tasks, observed_tasks):
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ValueError(
                    f"Runtime task order/identity mismatch at sequence "
                    f"{expected['sequence']} field={key}."
                )
        if not isinstance(observed.get("estimated_end_to_end_seconds"), (int, float)):
            raise ValueError("Runtime task is missing a time estimate.")
        if float(observed["estimated_end_to_end_seconds"]) <= 0:
            raise ValueError("Runtime task time estimate must be positive.")

    warmup_count = sum(row["phase"] == "warmup" for row in observed_tasks)
    measured_count = sum(row["phase"] == "measured" for row in observed_tasks)
    expected_counts = {
        "unique_runtime_config_count": len(expected_configs),
        "unique_setting_count": len(expected_configs) * len(EXPECTED_WORKERS),
        "warmup_task_count": warmup_count,
        "measured_task_count": measured_count,
        "physical_task_count": len(observed_tasks),
    }
    for key, value in expected_counts.items():
        if int(manifest.get(key, -1)) != value:
            raise ValueError(f"Runtime manifest {key} mismatch.")
    if len({row["runtime_task_id"] for row in observed_tasks}) != len(
        observed_tasks
    ):
        raise ValueError("Runtime manifest contains duplicate task IDs.")
    hash_payload = {
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "runtime_protocol": cfg["runtime"],
        "runtime_configs": observed_configs,
        "tasks": observed_tasks,
    }
    if manifest.get("runtime_manifest_sha256") != object_sha256(hash_payload):
        raise ValueError("Runtime manifest SHA256 validation failed.")


def runtime_task_config(
    cfg: Dict[str, Any],
    task_record: Dict[str, Any],
) -> Dict[str, Any]:
    candidate = {
        "method": task_record["method"],
        "parameters": copy.deepcopy(task_record["parameters"]),
    }
    return candidate_config(cfg, candidate, int(task_record["rounds"]))


def _runtime_result_record(
    runtime_root: Path,
    task_record: Dict[str, Any],
    task_completion: Dict[str, Any],
    attempts: int,
) -> Dict[str, Any]:
    output_root = runtime_root / task_record["output_root"]
    completion_record = task_completion["records"][0]
    failure_count = len(list((output_root / "failures").glob("*.json")))
    base = {
        "sequence": int(task_record["sequence"]),
        "runtime_task_id": task_record["runtime_task_id"],
        "phase": task_record["phase"],
        "include_in_summary": bool(task_record["include_in_summary"]),
        "repeat": task_record["repeat"],
        "runtime_config_id": task_record["runtime_config_id"],
        "runtime_config_sha256": task_record["runtime_config_sha256"],
        "method": task_record["method"],
        "epsilons": json.dumps(task_record["epsilons"]),
        "rounds": int(task_record["rounds"]),
        "worker_count": int(task_record["worker_count"]),
        "benchmark_seed": int(task_record["benchmark_seed"]),
        "attempts_this_invocation": attempts,
        "failure_attempt_count": failure_count,
        "output_root": task_record["output_root"],
        "estimated_end_to_end_seconds": float(
            task_record["estimated_end_to_end_seconds"]
        ),
    }
    if task_completion.get("status") != "complete":
        return {
            **base,
            "status": "failed",
            "task_status": completion_record.get("status"),
            "error_type": completion_record.get("error_type"),
            "message": completion_record.get("message"),
            "task_key": None,
            "partial_path": None,
            "partial_sha256": None,
            "checkpoint_count": None,
            "training_time": None,
            "communication_time": None,
            "evaluation_time": None,
            "end_to_end_time": None,
            "final_depth": None,
            "final_total_work": None,
            "final_per_worker_work": None,
            "timing_invariants": False,
        }

    partial_path = output_root / completion_record["partial_path"]
    task_manifest_path = output_root / completion_record["manifest_path"]
    payload = _load_json(partial_path)
    task_manifest = _load_json(task_manifest_path)
    final = payload["rows"][-1]
    training_time = float(final["training_time"])
    communication_time = float(final["communication_time"])
    evaluation_time = float(final["evaluation_time"])
    end_to_end_time = float(payload["launch"]["end_to_end_time"])
    timing_invariants = (
        all(
            math.isfinite(value) and value >= 0
            for value in (
                training_time,
                communication_time,
                evaluation_time,
                end_to_end_time,
            )
        )
        and training_time > 0
        and communication_time <= training_time + 1.0e-12
        and end_to_end_time >= training_time
        and len(set(payload["launch"]["child_pids"]))
        == int(task_record["worker_count"])
        and task_manifest.get("partial_sha256") == file_sha256(partial_path)
    )
    return {
        **base,
        "status": "complete" if timing_invariants else "failed",
        "task_status": completion_record["status"],
        "error_type": None if timing_invariants else "TimingInvariantError",
        "message": None if timing_invariants else "Runtime timing invariant failed.",
        "task_key": completion_record["task_key"],
        "partial_path": str(partial_path.relative_to(runtime_root)),
        "partial_sha256": task_manifest["partial_sha256"],
        "checkpoint_count": len(payload["rows"]),
        "training_time": training_time,
        "communication_time": communication_time,
        "evaluation_time": evaluation_time,
        "end_to_end_time": end_to_end_time,
        "final_depth": int(final["depth"]),
        "final_total_work": int(final["total_work"]),
        "final_per_worker_work": int(final["per_worker_work_max"]),
        "timing_invariants": timing_invariants,
    }


def _write_runtime_progress(
    output_root: Path,
    manifest: Dict[str, Any],
    records: List[Dict[str, Any]],
    started_at_utc: str,
    wall_start: float,
    current_task: Dict[str, Any] | None,
    status: str,
) -> Dict[str, Any]:
    completed = sum(row["status"] == "complete" for row in records)
    failed = sum(row["status"] == "failed" for row in records)
    completed_ids = {row["runtime_task_id"] for row in records}
    remaining_estimate = sum(
        float(row["estimated_end_to_end_seconds"])
        for row in manifest["tasks"]
        if row["runtime_task_id"] not in completed_ids
    )
    progress = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": status,
        "updated_at_utc": utc_now(),
        "started_at_utc": started_at_utc,
        "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
        "expected_tasks": int(manifest["physical_task_count"]),
        "processed_tasks": len(records),
        "completed_tasks": completed,
        "failed_tasks": failed,
        "completed_warmups": sum(
            row["status"] == "complete" and row["phase"] == "warmup"
            for row in records
        ),
        "completed_measured": sum(
            row["status"] == "complete" and row["phase"] == "measured"
            for row in records
        ),
        "elapsed_wall_seconds_this_invocation": time.perf_counter() - wall_start,
        "remaining_estimated_seconds": remaining_estimate,
        "current_task": (
            {
                key: current_task[key]
                for key in (
                    "sequence",
                    "runtime_task_id",
                    "phase",
                    "repeat",
                    "runtime_config_id",
                    "method",
                    "epsilons",
                    "rounds",
                    "worker_count",
                )
            }
            if current_task is not None
            else None
        ),
        "records": records,
    }
    atomic_write_json(output_root / "runtime_progress.json", progress)
    atomic_write_csv(output_root / "runtime_progress.csv", records)
    return progress


def run_runtime_manifest(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    step7_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    pilot = Path(pilot_root)
    step7 = Path(step7_root)
    output = Path(output_root)
    frozen = _load_json(pilot / "selected_config_by_epsilon.yaml")
    pilot_report = _load_json(pilot / "pilot_final_report.json")
    step7_completion = _load_json(step7 / "step7_completion.json")
    manifest = _load_json(output / "runtime_task_manifest.json")
    validate_runtime_manifest(
        cfg, frozen, pilot_report, step7_completion, manifest
    )
    process = process_config_from_experiment(cfg)
    max_attempts = int(cfg["runtime"]["max_attempts"])
    started_at = utc_now()
    wall_start = time.perf_counter()
    records: List[Dict[str, Any]] = []

    for task_record in manifest["tasks"]:
        index = int(task_record["sequence"]) + 1
        _write_runtime_progress(
            output,
            manifest,
            records,
            started_at,
            wall_start,
            task_record,
            "running",
        )
        print(
            f"runtime_task={index}/{manifest['physical_task_count']} "
            f"phase={task_record['phase']} method={task_record['method']} "
            f"m={task_record['worker_count']} repeat={task_record['repeat']} "
            f"status=starting",
            flush=True,
        )
        task_cfg = runtime_task_config(cfg, task_record)
        cpu_task = CpuFoTask(
            task_record["method"],
            int(task_record["benchmark_seed"]),
            int(task_record["worker_count"]),
        )
        task_completion = None
        attempts = 0
        for attempts in range(1, max_attempts + 1):
            task_completion = run_task_set(
                task_cfg,
                [cpu_task],
                output / task_record["output_root"],
                process,
                continue_on_error=True,
            )
            if task_completion["status"] == "complete":
                break
            error_type = task_completion["records"][0].get("error_type")
            if error_type == "ResumeValidationError":
                break
            print(
                f"runtime_task={index}/{manifest['physical_task_count']} "
                f"attempt={attempts}/{max_attempts} status=retrying",
                flush=True,
            )
        assert task_completion is not None
        result = _runtime_result_record(
            output, task_record, task_completion, attempts
        )
        records.append(result)
        _write_runtime_progress(
            output,
            manifest,
            records,
            started_at,
            wall_start,
            None,
            "running",
        )
        print(
            f"runtime_task={index}/{manifest['physical_task_count']} "
            f"status={result['status']} task_status={result['task_status']} "
            f"end_to_end={result['end_to_end_time']}",
            flush=True,
        )

    failed = [row for row in records if row["status"] != "complete"]
    result = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "complete" if not failed else "incomplete",
        "created_at_utc": utc_now(),
        "started_at_utc": started_at,
        "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
        "frozen_config_sha256": manifest["frozen_config_sha256"],
        "expected_tasks": int(manifest["physical_task_count"]),
        "completed_tasks": len(records) - len(failed),
        "failed_tasks": len(failed),
        "completed_warmups": sum(
            row["status"] == "complete" and row["phase"] == "warmup"
            for row in records
        ),
        "completed_measured": sum(
            row["status"] == "complete" and row["phase"] == "measured"
            for row in records
        ),
        "elapsed_wall_seconds_this_invocation": time.perf_counter() - wall_start,
        "records": records,
    }
    atomic_write_json(output / "runtime_completion.json", result)
    atomic_write_csv(output / "runtime_completion.csv", records)
    _write_runtime_progress(
        output,
        manifest,
        records,
        started_at,
        wall_start,
        None,
        "complete" if not failed else "incomplete",
    )
    if failed:
        raise RuntimeError(
            f"Runtime benchmark incomplete: {len(failed)} task(s) failed."
        )
    return result


def prepare_runtime_manifest(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    formal_root: str | Path,
    profile_root: str | Path,
    step7_root: str | Path,
    output_root: str | Path,
) -> Dict[str, Any]:
    pilot = Path(pilot_root)
    formal = Path(formal_root)
    profile = Path(profile_root)
    step7 = Path(step7_root)
    output = Path(output_root)
    frozen = _load_json(pilot / "selected_config_by_epsilon.yaml")
    pilot_report = _load_json(pilot / "pilot_final_report.json")
    step7_completion = _load_json(step7 / "step7_completion.json")
    validate_runtime_protocol(cfg, frozen, pilot_report, step7_completion)

    configs = unique_runtime_configs(cfg, frozen)
    tasks = runtime_task_order(cfg, configs)
    estimates = add_time_estimates(
        configs,
        tasks,
        formal,
        profile / "profile.csv",
        float(cfg["runtime"]["conservative_time_multiplier"]),
    )
    warmups = [row for row in tasks if row["phase"] == "warmup"]
    measured = [row for row in tasks if row["phase"] == "measured"]
    expanded_rows = sum(len(config["epsilons"]) for config in configs) * len(
        cfg["runtime"]["workers"]
    ) * int(cfg["runtime"]["repeats"])
    hash_payload = {
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "runtime_protocol": cfg["runtime"],
        "runtime_configs": configs,
        "tasks": tasks,
    }
    manifest = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "purpose": "formal-cpu-runtime-scaling",
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "step7_completion_sha256": file_sha256(
            step7 / "step7_completion.json"
        ),
        "profile_csv_sha256": file_sha256(profile / "profile.csv"),
        "runtime_protocol": copy.deepcopy(cfg["runtime"]),
        "cpu_resources": cpu_resource_record(),
        "unique_runtime_config_count": len(configs),
        "unique_runtime_config_count_by_method": {
            method: sum(row["method"] == method for row in configs)
            for method in EXPECTED_METHODS
        },
        "unique_setting_count": len(configs) * len(EXPECTED_WORKERS),
        "warmup_task_count": len(warmups),
        "measured_task_count": len(measured),
        "physical_task_count": len(tasks),
        "expanded_method_epsilon_worker_repeat_rows": expanded_rows,
        "runtime_configs": configs,
        "tasks": tasks,
        "time_estimate": estimates,
        "interleaving_note": (
            "ME-DOL has one deduplicated config and NOG has three; rotating insertion "
            "balances order positions but cannot alternate every adjacent task."
        ),
        "runtime_manifest_sha256": object_sha256(hash_payload),
        "launches_started": 0,
    }
    atomic_write_json(output / "base_config.json", effective_task_config(cfg))
    atomic_write_json(output / "runtime_task_manifest.json", manifest)
    validate_runtime_manifest(
        cfg, frozen, pilot_report, step7_completion, manifest
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_runtime.yaml"
    )
    parser.add_argument("--pilot-root", default="outputs/distributed_cpu_fo/pilot")
    parser.add_argument(
        "--formal-root", default="outputs/distributed_cpu_fo/formal_accuracy"
    )
    parser.add_argument(
        "--profile-root", default="outputs/distributed_cpu_fo/profile"
    )
    parser.add_argument(
        "--step7-root", default="outputs/distributed_cpu_fo/step7_final"
    )
    parser.add_argument(
        "--output-root", default="outputs/distributed_cpu_fo/runtime"
    )
    parser.add_argument(
        "--phase", choices=["prepare", "run"], default="prepare"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.phase == "run":
        result = run_runtime_manifest(
            cfg, args.pilot_root, args.step7_root, args.output_root
        )
        print(
            f"phase=runtime-run status={result['status']} "
            f"tasks={result['completed_tasks']}/{result['expected_tasks']} "
            f"warmups={result['completed_warmups']} "
            f"measured={result['completed_measured']} "
            f"failures={result['failed_tasks']}"
        )
        return
    manifest = prepare_runtime_manifest(
        cfg,
        args.pilot_root,
        args.formal_root,
        args.profile_root,
        args.step7_root,
        args.output_root,
    )
    print(
        f"phase=runtime-prepare status={manifest['status']} "
        f"configs={manifest['unique_runtime_config_count']} "
        f"settings={manifest['unique_setting_count']} "
        f"warmups={manifest['warmup_task_count']} "
        f"measured={manifest['measured_task_count']} "
        f"raw_hours={manifest['time_estimate']['raw_hours']:.2f} "
        f"conservative_hours={manifest['time_estimate']['conservative_hours']:.2f}"
    )
    print(f"output_root={args.output_root}")
    print("launches_started=0; no runtime worker processes launched")


if __name__ == "__main__":
    main()
