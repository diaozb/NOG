"""Audit and summarize completed Step 8 CPU runtime measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_runtime import (
    EXPECTED_METHODS,
    EXPECTED_WORKERS,
    _load_json,
    runtime_task_config,
    validate_runtime_manifest,
)
from src.distributed.cpu_fo_tasks import (
    atomic_write_json,
    effective_task_config,
    file_sha256,
    object_sha256,
    utc_now,
)


ANALYSIS_SCHEMA_VERSION = 1


def _strictly_increasing(values: Iterable[int]) -> bool:
    sequence = list(values)
    return all(left < right for left, right in zip(sequence, sequence[1:]))


def _nondecreasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(left <= right for left, right in zip(sequence, sequence[1:]))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _median_min_max(values: Iterable[float | int]) -> tuple[float, float, float]:
    sequence = [float(value) for value in values]
    if not sequence:
        raise ValueError("Cannot summarize an empty runtime sequence.")
    return statistics.median(sequence), min(sequence), max(sequence)


def _audit_task(
    cfg: Dict[str, Any],
    runtime_root: Path,
    task: Dict[str, Any],
    completion_record: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    errors: List[str] = []
    output_root = runtime_root / task["output_root"]
    task_completion = _load_json(output_root / "completion_manifest.json")
    if task_completion.get("status") != "complete":
        errors.append("per-task completion is not complete")
    if len(task_completion.get("records", [])) != 1:
        errors.append("per-task completion record count mismatch")
    atomic_record = task_completion["records"][0]
    if atomic_record.get("task_key") != completion_record.get("task_key"):
        errors.append("top-level/per-task task_key mismatch")

    partial_path = runtime_root / completion_record["partial_path"]
    task_manifest_path = output_root / atomic_record["manifest_path"]
    payload = _load_json(partial_path)
    task_manifest = _load_json(task_manifest_path)
    if task_manifest.get("partial_sha256") != file_sha256(partial_path):
        errors.append("partial SHA256 mismatch")
    if completion_record.get("partial_sha256") != task_manifest.get(
        "partial_sha256"
    ):
        errors.append("completion/manifest partial SHA256 mismatch")
    for key in ("task_key", "config_sha256", "task_fingerprint"):
        if payload.get(key) != task_manifest.get(key):
            errors.append(f"payload/task-manifest {key} mismatch")

    expected_cfg = runtime_task_config(cfg, task)
    expected_config_sha256 = object_sha256(effective_task_config(expected_cfg))
    if payload.get("config_sha256") != expected_config_sha256:
        errors.append("effective config SHA256 mismatch")
    expected_identity = {
        "method": task["method"],
        "formal_seed": int(task["benchmark_seed"]),
        "worker_count": int(task["worker_count"]),
    }
    observed_identity = {
        "method": payload.get("method"),
        "formal_seed": payload.get("formal_seed"),
        "worker_count": payload.get("worker_count"),
    }
    if observed_identity != expected_identity:
        errors.append("payload task identity mismatch")
    if task_manifest.get("task") != expected_identity:
        errors.append("task manifest identity mismatch")

    worker = int(task["worker_count"])
    rank_metadata = payload.get("rank_metadata", [])
    launch_pids = payload.get("launch", {}).get("child_pids", [])
    rank_pids = [row.get("pid") for row in rank_metadata]
    if [row.get("rank") for row in rank_metadata] != list(range(worker)):
        errors.append("rank metadata coverage mismatch")
    if len(set(rank_pids)) != worker or set(rank_pids) != set(launch_pids):
        errors.append("rank/launch PID coverage mismatch")
    if any(int(row.get("torch_threads", -1)) != 1 for row in rank_metadata):
        errors.append("rank thread count mismatch")
    if sum(int(row.get("shard_size", -1)) for row in rank_metadata) != int(
        cfg["problem"]["n_data"]
    ):
        errors.append("rank shard-size coverage mismatch")

    rows = payload.get("rows", [])
    if not rows:
        errors.append("trajectory is empty")
    numeric_fields = (
        "iteration",
        "depth",
        "total_work",
        "per_worker_work_max",
        "eval_work",
        "training_time",
        "communication_time",
        "evaluation_time",
        "objective",
        "stat_proxy",
    )
    eval_calls = int(cfg["oracle"]["eval_smooth_B"]) * int(
        cfg["oracle"]["eval_data_B"]
    )
    for index, row in enumerate(rows):
        if any(not _finite(row.get(field)) for field in numeric_fields):
            errors.append(f"row {index} contains non-finite metrics")
            continue
        iteration = int(row["iteration"])
        if row.get("method") != task["method"]:
            errors.append(f"row {index} method mismatch")
        if int(row.get("formal_seed", -1)) != int(task["benchmark_seed"]):
            errors.append(f"row {index} benchmark seed mismatch")
        if int(row.get("worker_count", -1)) != worker:
            errors.append(f"row {index} worker mismatch")
        if int(row["depth"]) != int(row.get("communication_round", -1)):
            errors.append(f"row {index} depth/communication mismatch")
        per_worker = [int(value) for value in row.get("per_worker_work", [])]
        if len(per_worker) != worker:
            errors.append(f"row {index} per-worker work coverage mismatch")
        elif sum(per_worker) != int(row["total_work"]) or max(per_worker) != int(
            row["per_worker_work_max"]
        ):
            errors.append(f"row {index} per-worker work sum/max mismatch")
        if task["method"] == "NOG-FO":
            parameters = task["parameters"]
            expected_work = (
                (iteration + 2)
                * int(parameters["smooth_B"])
                * int(parameters["data_B_total"])
            )
            expected_depth = iteration + 2
            if int(row.get("M", -1)) != int(parameters["M"]):
                errors.append(f"row {index} NOG M mismatch")
            if float(row.get("lr_or_eta", -1)) != float(parameters["eta"]):
                errors.append(f"row {index} NOG eta mismatch")
        else:
            parameters = task["parameters"]
            expected_work = iteration * worker
            expected_depth = iteration
            if int(row.get("epoch_length", -1)) != int(
                parameters["epoch_length"]
            ):
                errors.append(f"row {index} ME-DOL epoch length mismatch")
            if float(row.get("theory_multiplier", -1)) != float(
                parameters["theory_multiplier"]
            ):
                errors.append(f"row {index} ME-DOL multiplier mismatch")
        if int(row["total_work"]) != expected_work:
            errors.append(f"row {index} analytical work mismatch")
        if int(row["depth"]) != expected_depth:
            errors.append(f"row {index} analytical depth mismatch")
        if int(row["eval_work"]) != (index + 1) * eval_calls:
            errors.append(f"row {index} evaluation work mismatch")

    if rows:
        if int(rows[-1]["iteration"]) != int(task["rounds"]):
            errors.append("final iteration does not match task rounds")
        for key in ("iteration", "depth", "total_work", "per_worker_work_max", "eval_work"):
            if not _strictly_increasing(int(row[key]) for row in rows):
                errors.append(f"{key} is not strictly increasing")
        for key in ("training_time", "communication_time", "evaluation_time"):
            if not _nondecreasing(float(row[key]) for row in rows):
                errors.append(f"{key} is not nondecreasing")
        final = rows[-1]
        end_to_end = float(payload["launch"]["end_to_end_time"])
        if end_to_end < float(final["training_time"]):
            errors.append("end-to-end time is below training time")
    else:
        final = {}
        end_to_end = math.nan

    audit = {
        "sequence": int(task["sequence"]),
        "runtime_task_id": task["runtime_task_id"],
        "phase": task["phase"],
        "repeat": task["repeat"],
        "runtime_config_id": task["runtime_config_id"],
        "method": task["method"],
        "epsilons": json.dumps(task["epsilons"]),
        "worker_count": worker,
        "rounds": int(task["rounds"]),
        "checkpoint_count": len(rows),
        "expected_config_sha256": expected_config_sha256,
        "observed_config_sha256": payload.get("config_sha256"),
        "partial_sha256": task_manifest.get("partial_sha256"),
        "final_depth": int(final["depth"]) if rows else None,
        "final_total_work": int(final["total_work"]) if rows else None,
        "final_per_worker_work": (
            int(final["per_worker_work_max"]) if rows else None
        ),
        "training_time": float(final["training_time"]) if rows else None,
        "communication_time": (
            float(final["communication_time"]) if rows else None
        ),
        "evaluation_time": float(final["evaluation_time"]) if rows else None,
        "end_to_end_time": end_to_end,
        "passed": not errors,
        "errors": " | ".join(errors),
    }
    return payload, audit


def _repeat_consistency(
    payload_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in payload_records:
        if record["task"]["phase"] == "measured":
            groups[
                (
                    record["task"]["runtime_config_sha256"],
                    int(record["task"]["worker_count"]),
                )
            ].append(record)
    audits = []
    for (config_sha, worker), records in sorted(groups.items()):
        records.sort(key=lambda row: int(row["task"]["repeat"]))
        errors: List[str] = []
        if [int(row["task"]["repeat"]) for row in records] != [0, 1, 2]:
            errors.append("repeat coverage mismatch")
        reference = records[0]["payload"]
        max_difference = 0.0
        for record in records[1:]:
            observed = record["payload"]
            if len(observed["rows"]) != len(reference["rows"]):
                errors.append("checkpoint count differs across repeats")
                continue
            for index, (left, right) in enumerate(
                zip(reference["rows"], observed["rows"])
            ):
                for key in (
                    "iteration",
                    "depth",
                    "total_work",
                    "per_worker_work_max",
                    "eval_work",
                    "problem_seed",
                    "partition_seed",
                    "method_seed",
                ):
                    if left[key] != right[key]:
                        errors.append(f"row {index} discrete field differs: {key}")
                for key in ("objective", "stat_proxy"):
                    difference = abs(float(left[key]) - float(right[key]))
                    max_difference = max(max_difference, difference)
                    if not math.isclose(
                        float(left[key]),
                        float(right[key]),
                        rel_tol=1.0e-5,
                        abs_tol=1.0e-7,
                    ):
                        errors.append(f"row {index} numerical field differs: {key}")
        first = records[0]["task"]
        audits.append(
            {
                "runtime_config_id": first["runtime_config_id"],
                "runtime_config_sha256": config_sha,
                "method": first["method"],
                "epsilons": json.dumps(first["epsilons"]),
                "worker_count": worker,
                "repeat_count": len(records),
                "passed": not errors,
                "max_abs_objective_or_stat_difference": max_difference,
                "errors": " | ".join(errors[:20]),
            }
        )
    return audits


def audit_and_collect(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    step7_root: str | Path,
    runtime_root: str | Path,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    pilot = Path(pilot_root)
    step7 = Path(step7_root)
    root = Path(runtime_root)
    frozen = _load_json(pilot / "selected_config_by_epsilon.yaml")
    pilot_report = _load_json(pilot / "pilot_final_report.json")
    step7_completion = _load_json(step7 / "step7_completion.json")
    manifest = _load_json(root / "runtime_task_manifest.json")
    completion = _load_json(root / "runtime_completion.json")
    validate_runtime_manifest(cfg, frozen, pilot_report, step7_completion, manifest)
    errors: List[str] = []
    if completion.get("status") != "complete":
        errors.append("runtime completion is not complete")
    if completion.get("runtime_manifest_sha256") != manifest.get(
        "runtime_manifest_sha256"
    ):
        errors.append("completion/manifest SHA256 mismatch")
    if int(completion.get("completed_tasks", -1)) != int(
        manifest["physical_task_count"]
    ) or int(completion.get("failed_tasks", -1)) != 0:
        errors.append("runtime completion task counts mismatch")
    tasks = manifest["tasks"]
    records = completion.get("records", [])
    if len(records) != len(tasks):
        errors.append("runtime completion record coverage mismatch")

    payload_records = []
    audit_rows = []
    for task, record in zip(tasks, records):
        if record.get("runtime_task_id") != task["runtime_task_id"]:
            errors.append(f"completion task order mismatch at {task['sequence']}")
            continue
        payload, audit = _audit_task(cfg, root, task, record)
        payload_records.append({"task": task, "payload": payload})
        audit_rows.append(audit)
    if any(not row["passed"] for row in audit_rows):
        errors.append("one or more task audits failed")

    repeat_audits = _repeat_consistency(payload_records)
    if len(repeat_audits) != 24 or any(
        not row["passed"] for row in repeat_audits
    ):
        errors.append("repeat consistency audit failed")
    failure_paths = sorted((root / "raw").rglob("failures/*.json"))
    failure_cleanup_passed = True
    for path in failure_paths:
        failure = _load_json(path)
        cleanup = failure.get("process_cleanup", {})
        if cleanup.get("alive_after_cleanup") != []:
            failure_cleanup_passed = False
    if not failure_cleanup_passed:
        errors.append("a failed attempt left child processes alive")

    report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
        "expected_tasks": int(manifest["physical_task_count"]),
        "audited_tasks": len(audit_rows),
        "passed_tasks": sum(bool(row["passed"]) for row in audit_rows),
        "warmup_tasks": sum(row["phase"] == "warmup" for row in audit_rows),
        "measured_tasks": sum(row["phase"] == "measured" for row in audit_rows),
        "repeat_setting_audits": len(repeat_audits),
        "repeat_consistency_passed": all(
            bool(row["passed"]) for row in repeat_audits
        ),
        "max_repeat_numerical_difference": max(
            float(row["max_abs_objective_or_stat_difference"])
            for row in repeat_audits
        ),
        "failure_attempt_records": len(failure_paths),
        "failure_cleanup_passed": failure_cleanup_passed,
        "global_errors": errors,
    }
    return report, payload_records, audit_rows, repeat_audits


def expand_measured_repeats(
    payload_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for record in payload_records:
        task = record["task"]
        if task["phase"] != "measured":
            continue
        payload = record["payload"]
        final = payload["rows"][-1]
        training = float(final["training_time"])
        communication = float(final["communication_time"])
        evaluation = float(final["evaluation_time"])
        end_to_end = float(payload["launch"]["end_to_end_time"])
        for epsilon in task["epsilons"]:
            rows.append(
                {
                    "runtime_config_id": task["runtime_config_id"],
                    "method": task["method"],
                    "epsilon": float(epsilon),
                    "worker_count": int(task["worker_count"]),
                    "repeat": int(task["repeat"]),
                    "benchmark_seed": int(task["benchmark_seed"]),
                    "rounds": int(task["rounds"]),
                    "parameters": json.dumps(task["parameters"], sort_keys=True),
                    "depth": int(final["depth"]),
                    "total_work": int(final["total_work"]),
                    "per_worker_work": int(final["per_worker_work_max"]),
                    "training_time": training,
                    "communication_time": communication,
                    "communication_fraction": communication / training,
                    "evaluation_time": evaluation,
                    "end_to_end_time": end_to_end,
                    "non_training_overhead": max(
                        0.0, end_to_end - training - evaluation
                    ),
                    "final_stat_proxy": float(final["stat_proxy"]),
                    "final_objective": float(final["objective"]),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -row["epsilon"],
            row["method"],
            row["worker_count"],
            row["repeat"],
        ),
    )


def summarize_runtime(
    repeats: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[tuple[str, float, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in repeats:
        groups[(row["method"], row["epsilon"], row["worker_count"])].append(row)
    summaries = []
    timing_metrics = (
        "training_time",
        "communication_time",
        "communication_fraction",
        "evaluation_time",
        "end_to_end_time",
        "non_training_overhead",
    )
    for (method, epsilon, worker), rows in sorted(groups.items()):
        if sorted(row["repeat"] for row in rows) != [0, 1, 2]:
            raise ValueError(f"Runtime repeat coverage mismatch: {method}/{epsilon}/{worker}.")
        summary: Dict[str, Any] = {
            "method": method,
            "epsilon": epsilon,
            "worker_count": worker,
            "repeat_count": len(rows),
            "runtime_config_id": rows[0]["runtime_config_id"],
            "parameters": rows[0]["parameters"],
            "rounds": rows[0]["rounds"],
            "depth": rows[0]["depth"],
            "total_work": rows[0]["total_work"],
            "per_worker_work": rows[0]["per_worker_work"],
        }
        for metric in timing_metrics:
            median, minimum, maximum = _median_min_max(row[metric] for row in rows)
            summary[f"{metric}_median"] = median
            summary[f"{metric}_min"] = minimum
            summary[f"{metric}_max"] = maximum
        summaries.append(summary)

    lookup = {
        (row["method"], row["epsilon"], row["worker_count"]): row
        for row in summaries
    }
    speedups = []
    for row in summaries:
        base = lookup[(row["method"], row["epsilon"], 1)]
        training_speedup = base["training_time_median"] / row[
            "training_time_median"
        ]
        end_to_end_speedup = base["end_to_end_time_median"] / row[
            "end_to_end_time_median"
        ]
        speedups.append(
            {
                **row,
                "training_speedup_vs_m1": training_speedup,
                "training_efficiency_vs_m1": training_speedup
                / row["worker_count"],
                "end_to_end_speedup_vs_m1": end_to_end_speedup,
                "end_to_end_efficiency_vs_m1": end_to_end_speedup
                / row["worker_count"],
                "is_strong_scaling_workload": row["method"] == "NOG-FO",
            }
        )

    comparisons = []
    for epsilon in sorted({row["epsilon"] for row in summaries}, reverse=True):
        for worker in EXPECTED_WORKERS:
            nog = lookup[("NOG-FO", epsilon, worker)]
            me = lookup[("ME-DOL-FO", epsilon, worker)]
            comparisons.append(
                {
                    "epsilon": epsilon,
                    "worker_count": worker,
                    "nog_training_time_median": nog["training_time_median"],
                    "me_dol_training_time_median": me["training_time_median"],
                    "training_time_ratio_nog_over_me": nog[
                        "training_time_median"
                    ]
                    / me["training_time_median"],
                    "nog_end_to_end_time_median": nog[
                        "end_to_end_time_median"
                    ],
                    "me_dol_end_to_end_time_median": me[
                        "end_to_end_time_median"
                    ],
                    "end_to_end_time_ratio_nog_over_me": nog[
                        "end_to_end_time_median"
                    ]
                    / me["end_to_end_time_median"],
                    "depth_ratio_nog_over_me": nog["depth"] / me["depth"],
                    "work_ratio_nog_over_me": nog["total_work"]
                    / me["total_work"],
                    "comparison_scope": "full-frozen-budget-not-first-hit",
                }
            )
    return summaries, speedups, comparisons


def render_summary(
    audit: Dict[str, Any],
    speedups: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
) -> str:
    lines = [
        "# Step 8C CPU Runtime Scaling 汇总",
        "",
        "## 审计",
        "",
        f"- 状态：`{audit['status']}`；tasks `{audit['passed_tasks']}/{audit['expected_tasks']}` passed；",
        f"- Warm-ups：`{audit['warmup_tasks']}`（不进入统计）；measured physical runs：`{audit['measured_tasks']}`；",
        f"- 24 个 setting 的 3-repeat trajectory consistency 全部通过，最大 numerical difference 为 `{audit['max_repeat_numerical_difference']:.3g}`；",
        f"- 历史 process failure attempts：`{audit['failure_attempt_records']}`；cleanup passed：`{audit['failure_cleanup_passed']}`。",
        "",
        "## Scaling results",
        "",
        "时间为 3 repeats 的 median [min, max]。`speedup = median(m=1) / median(m)`；只有 NOG 的 total work 随 m 固定，因此只有 NOG 列可解释为 strong scaling。",
        "",
        "| Method | epsilon | m | Training (s) | Training speedup | Efficiency | End-to-end (s) | Comm fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(speedups, key=lambda item: (-item["epsilon"], item["method"], item["worker_count"])):
        lines.append(
            "| {method} | {eps:g} | {m} | {tm:.2f} [{tmin:.2f}, {tmax:.2f}] | {sp:.3f} | {eff:.3f} | {em:.2f} [{emin:.2f}, {emax:.2f}] | {cf:.1%} |".format(
                method=row["method"], eps=row["epsilon"], m=row["worker_count"],
                tm=row["training_time_median"], tmin=row["training_time_min"], tmax=row["training_time_max"],
                sp=row["training_speedup_vs_m1"], eff=row["training_efficiency_vs_m1"],
                em=row["end_to_end_time_median"], emin=row["end_to_end_time_min"], emax=row["end_to_end_time_max"],
                cf=row["communication_fraction_median"],
            )
        )
    lines.extend(
        [
            "",
            "## Method comparison at full frozen budgets",
            "",
            "下表的 NOG/ME-DOL time ratio 小于 1 表示 NOG 更快。它比较各 epsilon 对应 frozen config 的**完整预算**（NOG 960 rounds、ME-DOL 1920 rounds），不是 first-hit time-to-epsilon；epsilon=0.008 的 ME-DOL accuracy 仍然 censored。",
            "",
            "| epsilon | m | Training NOG/ME | End-to-end NOG/ME | Depth NOG/ME | Work NOG/ME |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {eps:g} | {m} | {tr:.3f} | {er:.3f} | {dr:.3f} | {wr:.3f} |".format(
                eps=row["epsilon"], m=row["worker_count"],
                tr=row["training_time_ratio_nog_over_me"],
                er=row["end_to_end_time_ratio_nog_over_me"],
                dr=row["depth_ratio_nog_over_me"], wr=row["work_ratio_nog_over_me"],
            )
        )
    best_rows = []
    for method in EXPECTED_METHODS:
        for epsilon in sorted({row["epsilon"] for row in speedups}, reverse=True):
            candidates = [row for row in speedups if row["method"] == method and row["epsilon"] == epsilon]
            best = min(candidates, key=lambda row: row["training_time_median"])
            best_rows.append((method, epsilon, best["worker_count"], best["training_time_median"]))
    lines.extend(
        [
            "",
            "## 直接观察",
            "",
            *[
                f"- `{method}`, epsilon `{epsilon:g}`：最低 median training time 在 `m={worker}`，为 `{value:.2f}s`。"
                for method, epsilon, worker, value in best_rows
            ],
            "- NOG 是 fixed-total-work strong-scaling workload；若 speedup 小于 1，表示当前 CPU/Gloo overhead 超过了 local parallelism 收益。",
            "- ME-DOL 每个 worker 每轮各做一个 SFO call，total work 随 m 线性增加；其 m1 ratio 不能作为 strong-scaling efficiency 证据。",
            "- `training_time` 排除 process startup/evaluation/serialization；`end_to_end_time` 包含这些开销。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_runtime(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    step7_root: str | Path,
    runtime_root: str | Path,
) -> Dict[str, Any]:
    root = Path(runtime_root)
    audit, payloads, audit_rows, repeat_audits = audit_and_collect(
        cfg, pilot_root, step7_root, root
    )
    atomic_write_csv(root / "runtime_work_audit.csv", audit_rows)
    atomic_write_csv(root / "runtime_repeat_consistency.csv", repeat_audits)
    atomic_write_json(root / "runtime_audit_report.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError("Runtime audit failed: " + " | ".join(audit["global_errors"]))
    repeats = expand_measured_repeats(payloads)
    summaries, speedups, comparisons = summarize_runtime(repeats)
    atomic_write_csv(root / "raw_repeats.csv", repeats)
    atomic_write_csv(root / "runtime_summary.csv", summaries)
    atomic_write_csv(root / "speedup_summary.csv", speedups)
    atomic_write_csv(root / "method_runtime_comparison.csv", comparisons)
    summary_text = render_summary(audit, speedups, comparisons)
    (root / "runtime_summary.md").write_text(summary_text, encoding="utf-8")
    output_names = (
        "runtime_work_audit.csv",
        "runtime_repeat_consistency.csv",
        "runtime_audit_report.json",
        "raw_repeats.csv",
        "runtime_summary.csv",
        "speedup_summary.csv",
        "method_runtime_comparison.csv",
        "runtime_summary.md",
    )
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "runtime_manifest_sha256": audit["runtime_manifest_sha256"],
        "audited_tasks": audit["audited_tasks"],
        "physical_measured_rows": audit["measured_tasks"],
        "expanded_method_epsilon_rows": len(repeats),
        "runtime_summary_rows": len(summaries),
        "speedup_summary_rows": len(speedups),
        "method_comparison_rows": len(comparisons),
        "output_sha256": {name: file_sha256(root / name) for name in output_names},
    }
    atomic_write_json(root / "runtime_analysis_completion.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distributed_cpu_fo_runtime.yaml")
    parser.add_argument("--pilot-root", default="outputs/distributed_cpu_fo/pilot")
    parser.add_argument("--step7-root", default="outputs/distributed_cpu_fo/step7_final")
    parser.add_argument("--runtime-root", default="outputs/distributed_cpu_fo/runtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_runtime(
        load_config(args.config), args.pilot_root, args.step7_root, args.runtime_root
    )
    print(
        f"phase=runtime-analysis status={result['status']} "
        f"audited={result['audited_tasks']} "
        f"expanded_rows={result['expanded_method_epsilon_rows']} "
        f"summary_rows={result['runtime_summary_rows']}"
    )


if __name__ == "__main__":
    main()
