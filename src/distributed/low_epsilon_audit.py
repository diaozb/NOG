"""Audit identities, hashes, grids, and work accounting for low-epsilon v5."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now
from src.distributed.low_epsilon_runner import me_label, nog_label

SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _expected_labels(freeze: Dict[str, Any]) -> Dict[str, int]:
    rounds = int(freeze["common_max_depth"])
    algorithms = freeze["selected_algorithms"]
    labels: Dict[str, int] = {}
    for batch in freeze["selected_batches"]["NOG-FO"]:
        labels[
            nog_label(
                algorithms["NOG-FO"]["M"],
                algorithms["NOG-FO"]["eta"],
                int(batch),
                rounds,
            )
        ] = int(batch)
    for batch in freeze["selected_batches"]["ME-DOL-FO"]:
        labels[
            me_label(
                algorithms["ME-DOL-FO"]["epoch_length"],
                algorithms["ME-DOL-FO"]["theory_multiplier"],
                int(batch),
                rounds,
            )
        ] = int(batch)
    return labels


def audit(
    freeze_path: Path, formal_root: Path, extra: bool = False
) -> Dict[str, Any]:
    freeze = _load(freeze_path)
    completion = _load(formal_root / "completion.json")
    seeds = {
        int(value)
        for value in (
            freeze["anomaly_confirmation"]["extra_formal_seeds"]
            if extra
            else freeze["formal_seeds"]
        )
    }
    labels = _expected_labels(freeze)
    expected = len(labels) * len(seeds)
    common_eval = int(freeze["common_eval_every"])
    max_rounds = int(freeze["common_max_depth"])
    errors = []
    task_rows = []
    if completion.get("status") != "complete":
        errors.append("completion status is not complete")
    if int(completion.get("completed_tasks", -1)) != expected:
        errors.append("completion task count does not match freeze")
    if int(completion.get("failed_tasks", -1)) != 0:
        errors.append("completion contains failures")

    identities = set()
    for record in completion.get("records", []):
        label = str(record["label"])
        task = record["task"]
        seed = int(task["formal_seed"])
        method = str(task["method"])
        identity = (label, seed)
        local = []
        if identity in identities:
            local.append("duplicate label/seed identity")
        identities.add(identity)
        if label not in labels:
            local.append("label is not in frozen configuration set")
        if seed not in seeds or int(task["worker_count"]) != 8:
            local.append("seed/worker does not match freeze")
        if not re.search(r"__batch-total-\d+__rounds-\d+$", label):
            local.append("label does not encode total batch and rounds")
        batch = labels.get(label, -1)
        partial_path = Path(record["partial_path"])
        manifest_path = Path(record["manifest_path"])
        payload = _load(partial_path)
        manifest = _load(manifest_path)
        if manifest.get("status") != "complete":
            local.append("task manifest is incomplete")
        if manifest.get("partial_sha256") != file_sha256(partial_path):
            local.append("partial SHA256 mismatch")
        if payload.get("config_sha256") != manifest.get("config_sha256"):
            local.append("config SHA256 mismatch")
        if payload.get("task_fingerprint") != manifest.get("task_fingerprint"):
            local.append("task fingerprint mismatch")
        if payload.get("method") != method or int(payload.get("formal_seed", -1)) != seed:
            local.append("payload task identity mismatch")
        ranks = payload.get("rank_metadata", [])
        if len(ranks) != 8 or {int(row["rank"]) for row in ranks} != set(range(8)):
            local.append("rank metadata is not exactly ranks 0..7")
        pids = payload.get("launch", {}).get("child_pids", [])
        if len(pids) != 8 or len(set(pids)) != 8:
            local.append("launch does not contain eight unique child PIDs")

        rows = payload.get("rows", [])
        iterations = [int(row["iteration"]) for row in rows]
        depths = [int(row["depth"]) for row in rows]
        works = [int(row["total_work"]) for row in rows]
        if not rows:
            local.append("trajectory is empty")
        if iterations != sorted(set(iterations)):
            local.append("iterations are not unique and increasing")
        if rows and iterations[-1] != max_rounds:
            local.append("trajectory does not reach frozen max rounds")
        if any(value % common_eval for value in iterations[:-1]):
            local.append("checkpoint is outside the common evaluation grid")
        if any(right <= left for left, right in zip(depths, depths[1:])):
            local.append("depth is not strictly increasing")
        if any(right <= left for left, right in zip(works, works[1:])):
            local.append("work is not strictly increasing")
        if works != [depth * batch for depth in depths]:
            local.append("total work formula mismatch")
        if any(int(row["per_worker_work_max"]) * 8 != int(row["total_work"]) for row in rows):
            local.append("per-worker work does not sum to total work")
        if any(
            not math.isfinite(float(row["stat_proxy"]))
            or float(row["stat_proxy"]) < 0.0
            for row in rows
        ):
            local.append("invalid stat_proxy")
        expected_final_depth = max_rounds + 2 if method == "NOG-FO" else max_rounds
        if depths and depths[-1] != expected_final_depth:
            local.append("final communication depth mismatch")
        errors.extend(f"{label}/seed{seed}: {message}" for message in local)
        task_rows.append(
            {
                "label": label,
                "method": method,
                "formal_seed": seed,
                "data_B_total": batch,
                "row_count": len(rows),
                "final_depth": depths[-1] if depths else None,
                "final_total_work": works[-1] if works else None,
                "passed": not local,
            }
        )

    if len(identities) != expected:
        errors.append("unique task identity count does not match expectation")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "scope": "formal_extra" if extra else "formal",
        "expected_tasks": expected,
        "audited_tasks": len(task_rows),
        "passed_tasks": sum(bool(row["passed"]) for row in task_rows),
        "failed_tasks": sum(not bool(row["passed"]) for row in task_rows),
        "errors": errors,
        "tasks": sorted(task_rows, key=lambda row: (row["label"], row["formal_seed"])),
    }
    output = formal_root / "formal_result_audit.json"
    atomic_write_json(output, report)
    if errors:
        raise ValueError(f"Formal audit failed with {len(errors)} error(s).")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        default=str(
            Path("outputs/distributed_cpu_fo_v5")
            / "epsilon_low_extension_v5_symmetric"
            / "frozen_parameters.json"
        ),
    )
    parser.add_argument(
        "--formal-root",
        default=str(
            Path("outputs/distributed_cpu_fo_v5")
            / "epsilon_low_extension_v5_symmetric"
            / "formal"
        ),
    )
    parser.add_argument("--extra", action="store_true")
    args = parser.parse_args()
    result = audit(Path(args.freeze), Path(args.formal_root), extra=args.extra)
    print(
        f"status={result['status']} passed={result['passed_tasks']}/"
        f"{result['expected_tasks']}"
    )


if __name__ == "__main__":
    main()
