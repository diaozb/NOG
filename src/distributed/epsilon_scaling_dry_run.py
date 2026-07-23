"""Short real-process dry-run for the wide-epsilon experiment protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import (
    load_config,
    process_config_from_experiment,
)
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    atomic_write_json,
    file_sha256,
    run_task_set,
    utc_now,
)
from src.distributed.epsilon_scaling import summarize_censored, validate_scaling_protocol


DRY_RUN_SCHEMA_VERSION = 1
DRY_RUN_SEEDS = (900, 901)
DRY_RUN_WORKERS = 2


def build_dry_run_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    validate_scaling_protocol(cfg)
    result = copy.deepcopy(cfg)
    result["run"]["name"] = "dry_run"
    result["problem"].update({"d": 8, "n_data": 64, "R": 2})
    result["train"].update({"rounds": 24, "eval_every": 4})
    result["oracle"].update(
        {
            "smooth_B": 2,
            "data_B_total": 8,
            "eval_smooth_B": 2,
            "eval_data_B": 4,
        }
    )
    result["nog"].update({"M": 4, "eta": 0.1})
    result["me_dol"].update({"epoch_length": 4, "theory_multiplier": 1.0})
    result["distributed"].update(
        {"comparison_worker": DRY_RUN_WORKERS, "scaling_workers": [DRY_RUN_WORKERS]}
    )
    result["cpu_process"].update(
        {
            "process_group_timeout_seconds": 60,
            "launch_timeout_seconds": 120,
            "max_total_worker_processes": 4,
            "task_timeout_seconds": 300,
        }
    )
    return result


def _load_payloads(root: Path, completion: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads = []
    for record in completion["records"]:
        path = root / record["partial_path"]
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if file_sha256(path) == "":
            raise ValueError("Impossible empty payload hash.")
        payloads.append(payload)
    return payloads


def analyze_dry_run(
    cfg: Dict[str, Any], root: Path, completion: Dict[str, Any]
) -> Dict[str, Any]:
    epsilons = [float(value) for value in cfg["epsilon_scaling"]["epsilons"]]
    consecutive = int(cfg["epsilon_scaling"]["confirmed_hit_consecutive"])
    payloads = _load_payloads(root, completion)
    expanded = []
    summaries = []
    for method in cfg["methods"]["sfo"]:
        method_payloads = [payload for payload in payloads if payload["method"] == method]
        for epsilon in epsilons:
            values: List[float | None] = []
            limits: List[float] = []
            for payload in method_payloads:
                hit = confirmed_hit(payload["rows"], epsilon, consecutive)
                final = payload["rows"][-1]
                values.append(float(hit["depth"]) if hit else None)
                limits.append(float(final["depth"]))
                expanded.append(
                    {
                        "method": method,
                        "formal_seed": int(payload["formal_seed"]),
                        "epsilon": epsilon,
                        "hit": hit is not None,
                        "first_hit_depth": hit["depth"] if hit else None,
                        "censoring_depth": int(final["depth"]),
                    }
                )
            summaries.append(
                {"method": method, "epsilon": epsilon, **summarize_censored(values, limits)}
            )
    expected = len(cfg["methods"]["sfo"]) * len(DRY_RUN_SEEDS) * len(epsilons)
    errors = []
    if completion.get("status") != "complete":
        errors.append("task set is incomplete")
    if len(payloads) != 4:
        errors.append(f"expected 4 payloads, observed {len(payloads)}")
    if len(expanded) != expected:
        errors.append(f"expected {expected} expanded rows, observed {len(expanded)}")
    if any(row["capped_mean"] is None or row["restricted_mean"] is None for row in summaries):
        errors.append("censoring-aware summary contains an empty primary statistic")
    audit = {
        "schema_version": DRY_RUN_SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "created_at_utc": utc_now(),
        "physical_tasks": len(payloads),
        "expanded_method_epsilon_seed_rows": len(expanded),
        "summary_rows": len(summaries),
        "epsilon_count": len(epsilons),
        "errors": errors,
        "completion_statuses": [record["status"] for record in completion["records"]],
        "summaries": summaries,
    }
    atomic_write_json(root / "dry_run_expanded.json", expanded)
    atomic_write_json(root / "dry_run_audit.json", audit)
    return audit


def run_dry_run(cfg: Dict[str, Any], output_root: str | Path) -> Dict[str, Any]:
    dry_cfg = build_dry_run_config(cfg)
    root = Path(output_root)
    tasks = [
        CpuFoTask(method, seed, DRY_RUN_WORKERS)
        for method in dry_cfg["methods"]["sfo"]
        for seed in DRY_RUN_SEEDS
    ]
    completion = run_task_set(
        dry_cfg,
        tasks,
        root,
        process_config_from_experiment(dry_cfg),
        continue_on_error=False,
    )
    return analyze_dry_run(dry_cfg, root, completion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml"
    )
    parser.add_argument(
        "--output-root",
        default="outputs/distributed_cpu_fo_v2/epsilon_scaling_v2/dry_run",
    )
    args = parser.parse_args()
    audit = run_dry_run(load_config(args.config), args.output_root)
    print(
        f"status={audit['status']} physical_tasks={audit['physical_tasks']} "
        f"expanded_rows={audit['expanded_method_epsilon_seed_rows']} "
        f"summary_rows={audit['summary_rows']}"
    )
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
