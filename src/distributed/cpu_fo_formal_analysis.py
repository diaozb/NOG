"""Audit and summarize the frozen Step 7 formal-accuracy trajectories.

This module never launches worker processes.  It validates the completed
atomic artifacts, expands deduplicated trajectories back to every requested
epsilon, and applies the frozen consecutive-checkpoint hit rule.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_formal import validate_formal_manifest
from src.distributed.cpu_fo_pilot import candidate_config, confirmed_hit
from src.distributed.cpu_fo_tasks import (
    atomic_write_json,
    effective_task_config,
    file_sha256,
    object_sha256,
    utc_now,
)


ANALYSIS_SCHEMA_VERSION = 1


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _strictly_increasing(values: Iterable[int]) -> bool:
    sequence = list(values)
    return all(left < right for left, right in zip(sequence, sequence[1:]))


def _nondecreasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(left <= right for left, right in zip(sequence, sequence[1:]))


def _mean_std(values: Iterable[float | int]) -> tuple[float | None, float | None]:
    sequence = [float(value) for value in values]
    if not sequence:
        return None, None
    mean = statistics.mean(sequence)
    std = statistics.stdev(sequence) if len(sequence) > 1 else 0.0
    return mean, std


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _task_audit(
    cfg: Dict[str, Any],
    formal_config: Dict[str, Any],
    config_root: Path,
    completion_record: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    task = completion_record["task"]
    partial_path = config_root / completion_record["partial_path"]
    task_manifest_path = config_root / completion_record["manifest_path"]
    payload = _load_json(partial_path)
    task_manifest = _load_json(task_manifest_path)
    errors: List[str] = []

    expected_task = {
        "method": formal_config["method"],
        "formal_seed": int(task["formal_seed"]),
        "worker_count": int(task["worker_count"]),
    }
    observed_task = {
        "method": payload.get("method"),
        "formal_seed": payload.get("formal_seed"),
        "worker_count": payload.get("worker_count"),
    }
    if task != expected_task or observed_task != expected_task:
        errors.append("task identity mismatch")
    if task_manifest.get("task") != expected_task:
        errors.append("task manifest identity mismatch")
    if task_manifest.get("status") != "complete":
        errors.append("task manifest is not complete")
    if task_manifest.get("partial_sha256") != file_sha256(partial_path):
        errors.append("partial SHA256 mismatch")
    for key in ("task_key", "config_sha256", "task_fingerprint"):
        if payload.get(key) != task_manifest.get(key):
            errors.append(f"partial/task-manifest {key} mismatch")
    if completion_record.get("task_key") != task_manifest.get("task_key"):
        errors.append("completion/task-manifest task_key mismatch")

    candidate = {
        "method": formal_config["method"],
        "parameters": formal_config["parameters"],
    }
    task_cfg = candidate_config(cfg, candidate, int(formal_config["rounds"]))
    expected_config_sha256 = object_sha256(effective_task_config(task_cfg))
    if payload.get("config_sha256") != expected_config_sha256:
        errors.append("effective config SHA256 mismatch")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("trajectory is empty")
        rows = []
    if int(task_manifest.get("row_count", -1)) != len(rows):
        errors.append("task manifest row_count mismatch")
    if int(completion_record.get("row_count", -1)) != len(rows):
        errors.append("completion row_count mismatch")

    worker_count = int(expected_task["worker_count"])
    rank_metadata = payload.get("rank_metadata", [])
    launch = payload.get("launch", {})
    ranks = [item.get("rank") for item in rank_metadata]
    rank_pids = [item.get("pid") for item in rank_metadata]
    child_pids = launch.get("child_pids", [])
    if ranks != list(range(worker_count)):
        errors.append("rank metadata coverage mismatch")
    if len(rank_pids) != worker_count or len(set(rank_pids)) != worker_count:
        errors.append("rank PIDs are not unique/complete")
    if set(rank_pids) != set(child_pids):
        errors.append("rank/launcher child PID mismatch")
    if any(int(item.get("torch_threads", -1)) != 1 for item in rank_metadata):
        errors.append("rank torch thread count is not one")
    if sum(int(item.get("shard_size", -1)) for item in rank_metadata) != int(
        cfg["problem"]["n_data"]
    ):
        errors.append("rank shard sizes do not cover n_data")
    if not _finite_number(launch.get("end_to_end_time")) or float(
        launch.get("end_to_end_time", -1)
    ) <= 0:
        errors.append("invalid end-to-end time")

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
    for index, row in enumerate(rows):
        if any(not _finite_number(row.get(key)) for key in numeric_fields):
            errors.append(f"row {index} has non-finite required metrics")
            continue
        if row.get("method") != formal_config["method"]:
            errors.append(f"row {index} method mismatch")
        if int(row.get("formal_seed", -1)) != int(expected_task["formal_seed"]):
            errors.append(f"row {index} formal seed mismatch")
        if int(row.get("worker_count", -1)) != worker_count:
            errors.append(f"row {index} worker count mismatch")
        if float(row.get("delta", -1.0)) != float(cfg["oracle"]["delta"]):
            errors.append(f"row {index} delta mismatch")
        if row.get("oracle_type") != "sfo":
            errors.append(f"row {index} oracle type mismatch")
        if int(row["depth"]) != int(row.get("communication_round", -1)):
            errors.append(f"row {index} depth/communication mismatch")
        per_worker = row.get("per_worker_work", [])
        if len(per_worker) != worker_count:
            errors.append(f"row {index} per-worker work length mismatch")
        elif (
            sum(int(value) for value in per_worker) != int(row["total_work"])
            or max(int(value) for value in per_worker)
            != int(row["per_worker_work_max"])
        ):
            errors.append(f"row {index} work sum/max mismatch")

        iteration = int(row["iteration"])
        if formal_config["method"] == "NOG-FO":
            parameters = formal_config["parameters"]
            expected_work = (
                (iteration + 2)
                * int(parameters["smooth_B"])
                * int(parameters["data_B_total"])
            )
            expected_depth = iteration + 2
            if iteration % int(parameters["M"]) != 0:
                errors.append(f"row {index} is not at a NOG block boundary")
        else:
            parameters = formal_config["parameters"]
            expected_work = iteration * worker_count
            expected_depth = iteration
            if iteration % int(parameters["epoch_length"]) != 0:
                errors.append(f"row {index} is not at a ME-DOL epoch boundary")
        if int(row["total_work"]) != expected_work:
            errors.append(f"row {index} analytical work mismatch")
        if int(row["depth"]) != expected_depth:
            errors.append(f"row {index} analytical depth mismatch")
        expected_eval_work = (index + 1) * int(cfg["oracle"]["eval_smooth_B"]) * int(
            cfg["oracle"]["eval_data_B"]
        )
        if int(row["eval_work"]) != expected_eval_work:
            errors.append(f"row {index} evaluation work mismatch")

    if rows:
        if int(rows[-1]["iteration"]) != int(formal_config["rounds"]):
            errors.append("trajectory does not end at the frozen round budget")
        if not _strictly_increasing(int(row["iteration"]) for row in rows):
            errors.append("iterations are not strictly increasing")
        for key in ("depth", "total_work", "per_worker_work_max", "eval_work"):
            if not _strictly_increasing(int(row[key]) for row in rows):
                errors.append(f"{key} is not strictly increasing")
        for key in ("training_time", "communication_time", "evaluation_time"):
            if not _nondecreasing(float(row[key]) for row in rows):
                errors.append(f"{key} is not nondecreasing")
        if float(launch["end_to_end_time"]) < max(
            float(rows[-1]["training_time"]),
            float(rows[-1]["evaluation_time"]),
        ):
            errors.append("end-to-end time is below a recorded phase time")

    audit_row = {
        "formal_config_id": formal_config["formal_config_id"],
        "method": formal_config["method"],
        "formal_seed": int(expected_task["formal_seed"]),
        "worker_count": worker_count,
        "rounds": int(formal_config["rounds"]),
        "checkpoint_count": len(rows),
        "partial_sha256": task_manifest.get("partial_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "expected_config_sha256": expected_config_sha256,
        "final_depth": int(rows[-1]["depth"]) if rows else None,
        "final_total_work": int(rows[-1]["total_work"]) if rows else None,
        "final_per_worker_work": (
            int(rows[-1]["per_worker_work_max"]) if rows else None
        ),
        "final_eval_work": int(rows[-1]["eval_work"]) if rows else None,
        "passed": not errors,
        "errors": " | ".join(errors),
    }
    return payload, audit_row, rows


def audit_and_collect(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    formal_root: str | Path,
) -> tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    pilot_path = Path(pilot_root)
    root = Path(formal_root)
    frozen = _load_json(pilot_path / "selected_config_by_epsilon.yaml")
    pilot_report = _load_json(pilot_path / "pilot_final_report.json")
    manifest = _load_json(root / "formal_task_manifest.json")
    completion = _load_json(root / "formal_completion.json")
    configs = validate_formal_manifest(cfg, manifest, frozen, pilot_report)

    global_errors: List[str] = []
    if completion.get("status") != "complete":
        global_errors.append("formal completion status is not complete")
    if completion.get("formal_manifest_sha256") != manifest.get(
        "formal_manifest_sha256"
    ):
        global_errors.append("formal completion/manifest SHA256 mismatch")
    if completion.get("frozen_config_sha256") != frozen.get(
        "frozen_config_sha256"
    ):
        global_errors.append("formal completion/frozen SHA256 mismatch")
    if int(completion.get("expected_tasks", -1)) != int(manifest["task_count"]):
        global_errors.append("formal expected task count mismatch")
    if int(completion.get("completed_tasks", -1)) != int(manifest["task_count"]):
        global_errors.append("formal completed task count mismatch")
    if int(completion.get("failed_tasks", -1)) != 0:
        global_errors.append("formal failures are nonzero")

    completion_lookup = {
        row["formal_config_id"]: row for row in completion.get("records", [])
    }
    if set(completion_lookup) != {row["formal_config_id"] for row in configs}:
        global_errors.append("formal completion config coverage mismatch")

    payload_records: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    observed_seeds: set[int] = set()
    for formal_config in configs:
        identifier = formal_config["formal_config_id"]
        completion_config = completion_lookup.get(identifier, {})
        if completion_config.get("status") != "complete":
            global_errors.append(f"config completion is not complete: {identifier}")
            continue
        config_root = root / formal_config["output_root"]
        config_completion = _load_json(config_root / "completion_manifest.json")
        expected_tasks = formal_config["tasks"]
        if config_completion.get("status") != "complete":
            global_errors.append(f"raw config completion is not complete: {identifier}")
        records = config_completion.get("records", [])
        if len(records) != len(expected_tasks):
            global_errors.append(f"raw config task count mismatch: {identifier}")
        record_lookup = {
            int(record["task"]["formal_seed"]): record for record in records
        }
        expected_seeds = {int(task["formal_seed"]) for task in expected_tasks}
        if set(record_lookup) != expected_seeds:
            global_errors.append(f"raw config seed coverage mismatch: {identifier}")
        for seed in sorted(expected_seeds & set(record_lookup)):
            payload, audit_row, rows = _task_audit(
                cfg, formal_config, config_root, record_lookup[seed]
            )
            observed_seeds.add(seed)
            audit_rows.append(audit_row)
            payload_records.append(
                {
                    "formal_config": formal_config,
                    "payload": payload,
                }
            )
            parameters_json = json.dumps(
                formal_config["parameters"], sort_keys=True
            )
            for row in rows:
                trajectory_rows.append(
                    {
                        "formal_config_id": identifier,
                        "method": formal_config["method"],
                        "formal_seed": seed,
                        "worker_count": int(row["worker_count"]),
                        "parameters": parameters_json,
                        "iteration": int(row["iteration"]),
                        "depth": int(row["depth"]),
                        "total_work": int(row["total_work"]),
                        "per_worker_work": int(row["per_worker_work_max"]),
                        "eval_work": int(row["eval_work"]),
                        "training_time": float(row["training_time"]),
                        "communication_time": float(row["communication_time"]),
                        "evaluation_time": float(row["evaluation_time"]),
                        "objective": float(row["objective"]),
                        "stat_proxy": float(row["stat_proxy"]),
                    }
                )

    formal_seeds = {int(seed) for seed in frozen["formal_seeds"]}
    pilot_seeds = {int(seed) for seed in frozen["pilot_seeds"]}
    if observed_seeds != formal_seeds:
        global_errors.append("observed formal seed set mismatch")
    if observed_seeds & pilot_seeds:
        global_errors.append("pilot seed leaked into formal results")
    if len(audit_rows) != int(manifest["task_count"]):
        global_errors.append("audit task row count mismatch")
    if any(not row["passed"] for row in audit_rows):
        global_errors.append("one or more task-level audits failed")

    audit_report = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "passed" if not global_errors else "failed",
        "created_at_utc": utc_now(),
        "formal_manifest_sha256": manifest["formal_manifest_sha256"],
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "expected_tasks": int(manifest["task_count"]),
        "audited_tasks": len(audit_rows),
        "passed_tasks": sum(bool(row["passed"]) for row in audit_rows),
        "failed_tasks": sum(not bool(row["passed"]) for row in audit_rows),
        "unique_configs": len(configs),
        "formal_seeds": sorted(observed_seeds),
        "pilot_seeds_excluded": sorted(pilot_seeds),
        "trajectory_rows": len(trajectory_rows),
        "global_errors": global_errors,
    }
    return audit_report, payload_records, audit_rows, trajectory_rows


def threshold_results(
    payload_records: List[Dict[str, Any]],
    consecutive: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    per_seed: List[Dict[str, Any]] = []
    for record in payload_records:
        formal_config = record["formal_config"]
        payload = record["payload"]
        final = payload["rows"][-1]
        for epsilon in formal_config["epsilons"]:
            hit = confirmed_hit(payload["rows"], float(epsilon), consecutive)
            per_seed.append(
                {
                    "formal_config_id": formal_config["formal_config_id"],
                    "method": formal_config["method"],
                    "epsilon": float(epsilon),
                    "delta": float(final["delta"]),
                    "formal_seed": int(payload["formal_seed"]),
                    "worker_count": int(payload["worker_count"]),
                    "parameters": json.dumps(
                        formal_config["parameters"], sort_keys=True
                    ),
                    "pilot_selection_status": formal_config[
                        "pilot_selection_status_by_epsilon"
                    ][f"{float(epsilon):g}"],
                    "confirmed_consecutive": consecutive,
                    "hit": hit is not None,
                    "censored": hit is None,
                    "first_hit_iteration": hit["iteration"] if hit else None,
                    "first_hit_depth": hit["depth"] if hit else None,
                    "first_hit_total_work": hit["total_work"] if hit else None,
                    "first_hit_per_worker_work": (
                        hit["per_worker_work"] if hit else None
                    ),
                    "first_hit_training_time": (
                        hit["training_time"] if hit else None
                    ),
                    "first_hit_stat_proxy": hit["stat_proxy"] if hit else None,
                    "final_iteration": int(final["iteration"]),
                    "final_depth": int(final["depth"]),
                    "final_total_work": int(final["total_work"]),
                    "final_per_worker_work": int(final["per_worker_work_max"]),
                    "final_stat_proxy": float(final["stat_proxy"]),
                    "end_to_end_time": float(payload["launch"]["end_to_end_time"]),
                }
            )

    summaries: List[Dict[str, Any]] = []
    groups: Dict[tuple[str, float], List[Dict[str, Any]]] = {}
    for row in per_seed:
        groups.setdefault((row["method"], row["epsilon"]), []).append(row)
    metric_map = (
        ("first_hit_depth", "first_hit_depth"),
        ("first_hit_total_work", "first_hit_total_work"),
        ("first_hit_per_worker_work", "first_hit_per_worker_work"),
        ("first_hit_training_time", "first_hit_training_time"),
    )
    for (method, epsilon), rows in sorted(groups.items()):
        hits = [row for row in rows if row["hit"]]
        final_mean, final_std = _mean_std(row["final_stat_proxy"] for row in rows)
        summary: Dict[str, Any] = {
            "method": method,
            "epsilon": epsilon,
            "delta": rows[0]["delta"],
            "num_seeds": len(rows),
            "hit_count": len(hits),
            "hit_rate": len(hits) / len(rows),
            "censored_count": len(rows) - len(hits),
            "full_hit": len(hits) == len(rows),
            "pilot_selection_status": rows[0]["pilot_selection_status"],
            "formal_config_id": rows[0]["formal_config_id"],
            "parameters": rows[0]["parameters"],
            "final_stat_proxy_mean": final_mean,
            "final_stat_proxy_std": final_std,
        }
        for output, key in metric_map:
            mean, std = _mean_std(row[key] for row in hits)
            summary[f"{output}_mean"] = mean
            summary[f"{output}_std"] = std
        summaries.append(summary)

    comparisons: List[Dict[str, Any]] = []
    summary_lookup = {
        (row["method"], row["epsilon"]): row for row in summaries
    }
    epsilons = sorted({row["epsilon"] for row in summaries}, reverse=True)
    for epsilon in epsilons:
        nog = summary_lookup[("NOG-FO", epsilon)]
        me = summary_lookup[("ME-DOL-FO", epsilon)]
        both_have_hits = nog["hit_count"] > 0 and me["hit_count"] > 0
        comparisons.append(
            {
                "epsilon": epsilon,
                "nog_hit_rate": nog["hit_rate"],
                "me_dol_hit_rate": me["hit_rate"],
                "full_hit_comparison": bool(nog["full_hit"] and me["full_hit"]),
                "ratios_are_hit_only_if_not_full_hit": not bool(
                    nog["full_hit"] and me["full_hit"]
                ),
                "nog_depth_mean": nog["first_hit_depth_mean"],
                "me_dol_depth_mean": me["first_hit_depth_mean"],
                "depth_improvement_me_over_nog": (
                    me["first_hit_depth_mean"] / nog["first_hit_depth_mean"]
                    if both_have_hits and nog["first_hit_depth_mean"]
                    else None
                ),
                "nog_total_work_mean": nog["first_hit_total_work_mean"],
                "me_dol_total_work_mean": me["first_hit_total_work_mean"],
                "total_work_ratio_nog_over_me": (
                    nog["first_hit_total_work_mean"]
                    / me["first_hit_total_work_mean"]
                    if both_have_hits and me["first_hit_total_work_mean"]
                    else None
                ),
                "nog_per_worker_work_mean": nog[
                    "first_hit_per_worker_work_mean"
                ],
                "me_dol_per_worker_work_mean": me[
                    "first_hit_per_worker_work_mean"
                ],
                "per_worker_work_ratio_nog_over_me": (
                    nog["first_hit_per_worker_work_mean"]
                    / me["first_hit_per_worker_work_mean"]
                    if both_have_hits and me["first_hit_per_worker_work_mean"]
                    else None
                ),
                "nog_training_time_mean": nog["first_hit_training_time_mean"],
                "me_dol_training_time_mean": me[
                    "first_hit_training_time_mean"
                ],
                "training_time_improvement_me_over_nog": (
                    me["first_hit_training_time_mean"]
                    / nog["first_hit_training_time_mean"]
                    if both_have_hits and nog["first_hit_training_time_mean"]
                    else None
                ),
            }
        )
    return per_seed, summaries, comparisons


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{float(value):.{digits}f}"


def render_summary(
    audit_report: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
    consecutive: int,
) -> str:
    lines = [
        "# Step 7C Formal accuracy 审计与汇总",
        "",
        "本报告只分析已冻结配置的 formal seeds，不包含任何重新调参或新训练。",
        "",
        "## 完整性审计",
        "",
        f"- 状态：`{audit_report['status']}`",
        f"- Tasks：`{audit_report['passed_tasks']}/{audit_report['expected_tasks']}` passed",
        f"- Unique configs：`{audit_report['unique_configs']}`",
        f"- Formal seeds：`{audit_report['formal_seeds']}`；pilot seeds 已排除",
        f"- Raw checkpoint rows：`{audit_report['trajectory_rows']}`",
        "- SFO training work、evaluation work、depth、SHA256、PID/rank、seed、finite metrics 与单调性均逐 task 审计。",
        "",
        "## Confirmed-hit 结果",
        "",
        f"First hit 要求连续 `{consecutive}` 个 evaluation checkpoints 满足 `stat_proxy <= epsilon`，并把第一个 checkpoint 记为命中位置。均值和 sample std 只在命中 seeds 上计算，hit rate/censoring 单独报告。",
        "",
        "| Method | epsilon | hit | Depth mean±std | Total work mean±std | Per-worker work mean±std | Training time mean±std (s) | Final stat mean±std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: (-item["epsilon"], item["method"])):
        lines.append(
            "| {method} | {eps:g} | {hits}/{n} | {dm}±{ds} | {wm}±{ws} | "
            "{pm}±{ps} | {tm}±{ts} | {fm}±{fs} |".format(
                method=row["method"],
                eps=row["epsilon"],
                hits=row["hit_count"],
                n=row["num_seeds"],
                dm=_fmt(row["first_hit_depth_mean"]),
                ds=_fmt(row["first_hit_depth_std"]),
                wm=_fmt(row["first_hit_total_work_mean"]),
                ws=_fmt(row["first_hit_total_work_std"]),
                pm=_fmt(row["first_hit_per_worker_work_mean"]),
                ps=_fmt(row["first_hit_per_worker_work_std"]),
                tm=_fmt(row["first_hit_training_time_mean"]),
                ts=_fmt(row["first_hit_training_time_std"]),
                fm=_fmt(row["final_stat_proxy_mean"], 6),
                fs=_fmt(row["final_stat_proxy_std"], 6),
            )
        )
    lines.extend(
        [
            "",
            "## NOG-FO / ME-DOL-FO 对比",
            "",
            "`Depth improvement = ME-DOL depth / NOG depth`，大于 1 表示 NOG communication depth 更小；`Work ratio = NOG work / ME-DOL work`，大于 1 表示 NOG 使用更多 SFO work。",
            "",
            "| epsilon | NOG hit | ME-DOL hit | Full-hit comparison | Depth improvement | Total-work ratio | Per-worker-work ratio | Training-time improvement |",
            "|---:|---:|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            "| {eps:g} | {nh:.0%} | {mh:.0%} | {full} | {depth} | {work} | {pwork} | {time} |".format(
                eps=row["epsilon"],
                nh=row["nog_hit_rate"],
                mh=row["me_dol_hit_rate"],
                full="yes" if row["full_hit_comparison"] else "no*",
                depth=_fmt(row["depth_improvement_me_over_nog"]),
                work=_fmt(row["total_work_ratio_nog_over_me"]),
                pwork=_fmt(row["per_worker_work_ratio_nog_over_me"]),
                time=_fmt(row["training_time_improvement_me_over_nog"]),
            )
        )
    lines.extend(
        [
            "",
            "`*` 非 full-hit 行的 ratio 仅基于成功 seeds，存在 censoring bias，不能作为无条件算法比较。",
            "",
            "Evaluation work 来自共同 fixed high-precision sample bank，已单独审计；上表 Work 指 training SFO calls，不包含 evaluation calls。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_formal(
    cfg: Dict[str, Any],
    pilot_root: str | Path,
    formal_root: str | Path,
) -> Dict[str, Any]:
    root = Path(formal_root)
    frozen = _load_json(Path(pilot_root) / "selected_config_by_epsilon.yaml")
    consecutive = int(frozen["confirmed_hit_consecutive"])
    audit_report, payloads, audit_rows, trajectories = audit_and_collect(
        cfg, pilot_root, root
    )
    atomic_write_csv(root / "work_accounting_audit.csv", audit_rows)
    atomic_write_json(root / "formal_audit_report.json", audit_report)
    if audit_report["status"] != "passed":
        raise RuntimeError(
            "Formal audit failed: " + " | ".join(audit_report["global_errors"])
        )

    per_seed, summaries, comparisons = threshold_results(payloads, consecutive)
    atomic_write_csv(root / "formal_results.csv", trajectories)
    atomic_write_csv(root / "threshold_per_seed.csv", per_seed)
    atomic_write_csv(root / "threshold_summary.csv", summaries)
    atomic_write_csv(root / "method_comparison.csv", comparisons)
    summary_text = render_summary(audit_report, summaries, comparisons, consecutive)
    summary_path = root / "formal_summary.md"
    summary_path.write_text(summary_text, encoding="utf-8")

    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_manifest_sha256": audit_report["formal_manifest_sha256"],
        "frozen_config_sha256": audit_report["frozen_config_sha256"],
        "confirmed_hit_consecutive": consecutive,
        "audited_tasks": audit_report["audited_tasks"],
        "trajectory_rows": len(trajectories),
        "method_epsilon_seed_rows": len(per_seed),
        "threshold_summary_rows": len(summaries),
        "comparison_rows": len(comparisons),
        "output_sha256": {
            name: file_sha256(root / name)
            for name in (
                "formal_results.csv",
                "threshold_per_seed.csv",
                "threshold_summary.csv",
                "method_comparison.csv",
                "work_accounting_audit.csv",
                "formal_audit_report.json",
                "formal_summary.md",
            )
        },
    }
    atomic_write_json(root / "formal_analysis_completion.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pilot-root", default="outputs/distributed_cpu_fo/pilot")
    parser.add_argument(
        "--formal-root",
        default="outputs/distributed_cpu_fo/formal_accuracy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_formal(
        load_config(args.config), args.pilot_root, args.formal_root
    )
    print(
        f"phase=formal-analysis status={result['status']} "
        f"audited_tasks={result['audited_tasks']} "
        f"threshold_rows={result['method_epsilon_seed_rows']}"
    )


if __name__ == "__main__":
    main()
