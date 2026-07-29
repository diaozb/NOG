"""Resumable task runner for the wide-epsilon theory-validation experiment.

Pilot and formal seeds are intentionally disjoint.  Each task owns eight worker
processes and the scheduler admits at most four tasks at once, keeping the
experiment within the preregistered 32-process CPU cap.
"""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.distributed.cpu_fo_correctness import (
    load_config,
    process_config_from_experiment,
)
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    environment_record,
    object_sha256,
    run_or_resume_task,
    utc_now,
)


SCHEMA_VERSION = 1


def nog_batch_config(base: Dict[str, Any], data_batch_total: int) -> Dict[str, Any]:
    """Return the frozen NOG pilot configuration for one global batch size."""

    cfg = copy.deepcopy(base)
    cfg["train"].update({"rounds": 960, "eval_every": 2})
    cfg["oracle"].update({"smooth_B": 1, "data_B_total": int(data_batch_total)})
    cfg["nog"].update({"M": 2, "eta": 1.0})
    cfg["methods"]["sfo"] = ["NOG-FO"]
    if int(data_batch_total) < 8 or int(data_batch_total) % 8:
        raise ValueError("NOG global data batch must be a positive multiple of 8.")
    return cfg


def me_formal_config(base: Dict[str, Any]) -> Dict[str, Any]:
    """Return the frozen extended-budget ME-DOL configuration."""

    cfg = copy.deepcopy(base)
    cfg["train"].update({"rounds": 3840, "eval_every": 6})
    cfg["me_dol"].update({"epoch_length": 6, "theory_multiplier": 100.0})
    cfg["methods"]["sfo"] = ["ME-DOL-FO"]
    return cfg


def _record(result: Any, label: str, config_sha256: str) -> Dict[str, Any]:
    return {
        "label": label,
        "config_sha256": config_sha256,
        "task": result.task.as_dict(),
        "status": result.status,
        "task_key": result.task_key,
        "partial_path": str(result.partial_path),
        "manifest_path": str(result.manifest_path),
        "row_count": result.row_count,
    }


def run_labeled_tasks(
    labeled: Iterable[tuple[str, Dict[str, Any], CpuFoTask, Path]],
    completion_path: Path,
    max_parallel_tasks: int = 4,
) -> Dict[str, Any]:
    """Run independent labeled configurations with atomic per-task resume."""

    work = list(labeled)
    if not work:
        raise ValueError("No theory-validation tasks were requested.")
    if max_parallel_tasks < 1 or max_parallel_tasks > 4:
        raise ValueError("max_parallel_tasks must be between 1 and 4.")
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    def execute(item: tuple[str, Dict[str, Any], CpuFoTask, Path]) -> Dict[str, Any]:
        label, cfg, task, root = item
        process = process_config_from_experiment(cfg)
        result = run_or_resume_task(cfg, task, root, process)
        return _record(result, label, object_sha256(cfg))

    with ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
        futures = {executor.submit(execute, item): item for item in work}
        for future in as_completed(futures):
            label, cfg, task, root = futures[future]
            try:
                records.append(future.result())
            except BaseException as error:
                failures.append(
                    {
                        "label": label,
                        "task": task.as_dict(),
                        "output_root": str(root),
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
            snapshot = {
                "schema_version": SCHEMA_VERSION,
                "updated_at_utc": utc_now(),
                "status": "running",
                "expected_tasks": len(work),
                "completed_tasks": len(records),
                "failed_tasks": len(failures),
                "max_parallel_tasks": max_parallel_tasks,
                "max_worker_processes": 8 * max_parallel_tasks,
                "environment": environment_record(),
                "records": sorted(records, key=lambda row: (row["label"], row["task"]["formal_seed"])),
                "failures": failures,
            }
            atomic_write_json(completion_path, snapshot)

    completion = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "status": "complete" if len(records) == len(work) and not failures else "incomplete",
        "expected_tasks": len(work),
        "completed_tasks": len(records),
        "failed_tasks": len(failures),
        "max_parallel_tasks": max_parallel_tasks,
        "max_worker_processes": 8 * max_parallel_tasks,
        "environment": environment_record(),
        "records": sorted(records, key=lambda row: (row["label"], row["task"]["formal_seed"])),
        "failures": failures,
    }
    atomic_write_json(completion_path, completion)
    if failures:
        raise RuntimeError(f"{len(failures)} theory-validation task(s) failed.")
    return completion


def pilot_batch_grid(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    batches = [8, 16, 24, 32, 40, 48, 56, 64]
    seeds = [int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]]
    labeled = []
    for batch in batches:
        candidate = nog_batch_config(cfg, batch)
        label = f"NOG-FO__data-B-total-{batch}"
        candidate_root = root / label
        atomic_write_json(candidate_root / "config_used.json", candidate)
        for seed in seeds:
            labeled.append(
                (label, candidate, CpuFoTask("NOG-FO", seed, 8), candidate_root)
            )
    return run_labeled_tasks(labeled, root / "completion.json")


def formal_tasks(
    cfg: Dict[str, Any], freeze_path: Path, root: Path
) -> Dict[str, Any]:
    """Run the independent formal seeds for pilot-frozen configurations."""

    with open(freeze_path, "r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "frozen":
        raise ValueError("Theory-validation parameters have not been frozen.")
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if set(seeds) & {int(value) for value in cfg["epsilon_scaling"]["pilot_seeds"]}:
        raise ValueError("Pilot and formal seeds must be disjoint.")

    labeled = []
    for batch in sorted({int(value) for value in freeze["selected_batches"]}):
        candidate = nog_batch_config(cfg, batch)
        label = f"NOG-FO__data-B-total-{batch}"
        candidate_root = root / label
        atomic_write_json(candidate_root / "config_used.json", candidate)
        for seed in seeds:
            labeled.append(
                (label, candidate, CpuFoTask("NOG-FO", seed, 8), candidate_root)
            )

    candidate = me_formal_config(cfg)
    label = "ME-DOL-FO__epoch-6__mult-100__rounds-3840"
    candidate_root = root / label
    atomic_write_json(candidate_root / "config_used.json", candidate)
    for seed in seeds:
        labeled.append(
            (label, candidate, CpuFoTask("ME-DOL-FO", seed, 8), candidate_root)
        )
    return run_labeled_tasks(labeled, root / "completion.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot-batch-grid", "formal"])
    parser.add_argument(
        "--config",
        default="configs/distributed_cpu_fo_theory_validation_v4.yaml",
    )
    parser.add_argument(
        "--output-root",
        default=None,
    )
    parser.add_argument(
        "--freeze",
        default=(
            "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/"
            "frozen_parameters.json"
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "pilot-batch-grid":
        root = Path(args.output_root or (
            "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/"
            "pilot/batch_grid"
        ))
        result = pilot_batch_grid(cfg, root)
    elif args.command == "formal":
        root = Path(args.output_root or (
            "outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4/formal"
        ))
        result = formal_tasks(cfg, Path(args.freeze), root)
    else:  # pragma: no cover - argparse guards this branch.
        raise ValueError(args.command)
    print(
        f"status={result['status']} completed={result['completed_tasks']}/"
        f"{result['expected_tasks']} failed={result['failed_tasks']}"
    )


if __name__ == "__main__":
    main()
