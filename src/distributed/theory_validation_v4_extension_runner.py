"""Run a frozen-parameter, extended-budget continuation of theory validation v4.

Only the maximum number of rounds changes relative to the v4 low-epsilon
configuration.  Results are written to a separate root so the preregistered v4
artifacts remain immutable.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_correctness import (
    load_config,
    process_config_from_experiment,
)
from src.distributed.cpu_fo_tasks import CpuFoTask, atomic_write_json, file_sha256, object_sha256, utc_now
from src.distributed.theory_validation_runner import run_labeled_tasks


CONFIG = "configs/distributed_cpu_fo_theory_validation_v4_extended_budget.yaml"


def nog_config(base: Dict[str, Any], rounds: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    fixed = cfg["extended_budget"]["fixed_parameters"]["NOG-FO"]
    cfg["train"].update({"rounds": int(rounds), "eval_every": int(fixed["eval_every"])})
    cfg["oracle"].update(
        {
            "smooth_B": int(fixed["smooth_B"]),
            "data_B_total": int(fixed["data_B_total"]),
        }
    )
    cfg["nog"].update({"M": int(fixed["M"]), "eta": float(fixed["eta"])})
    cfg["methods"]["sfo"] = ["NOG-FO"]
    return cfg


def me_config(base: Dict[str, Any], rounds: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    fixed = cfg["extended_budget"]["fixed_parameters"]["ME-DOL-FO"]
    cfg["train"].update({"rounds": int(rounds), "eval_every": int(fixed["eval_every"])})
    cfg["me_dol"].update(
        {
            "epoch_length": int(fixed["epoch_length"]),
            "theory_multiplier": float(fixed["theory_multiplier"]),
        }
    )
    cfg["methods"]["sfo"] = ["ME-DOL-FO"]
    return cfg


def stage_configs(cfg: Dict[str, Any], stage: str) -> Dict[str, Dict[str, Any]]:
    stages = cfg["extended_budget"]["stages"]
    if stage not in stages:
        raise ValueError(f"Unknown extended-budget stage: {stage}")
    budgets = stages[stage]
    return {
        "NOG-FO": nog_config(cfg, int(budgets["NOG-FO"])),
        "ME-DOL-FO": me_config(cfg, int(budgets["ME-DOL-FO"])),
    }


def run_stage(
    cfg: Dict[str, Any], stage: str, root: Path, max_parallel_tasks: int = 4
) -> Dict[str, Any]:
    configs = stage_configs(cfg, stage)
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if seeds != list(range(20)):
        raise ValueError("The v4 continuation requires frozen formal seeds 0..19.")

    labeled = []
    labels = {}
    for method, candidate in configs.items():
        rounds = int(candidate["train"]["rounds"])
        if method == "NOG-FO":
            batch = int(candidate["oracle"]["data_B_total"])
            label = f"NOG-FO__rounds-{rounds}__data-B-total-{batch}"
        else:
            label = f"ME-DOL-FO__epoch-6__mult-100__rounds-{rounds}"
        labels[method] = label
        candidate_root = root / label
        atomic_write_json(candidate_root / "config_used.json", candidate)
        for seed in seeds:
            labeled.append((label, candidate, CpuFoTask(method, seed, 8), candidate_root))

    source_freeze = Path(cfg["extended_budget"]["source_freeze"])
    protocol = {
        "schema_version": 1,
        "status": "frozen_extension",
        "created_at_utc": utc_now(),
        "stage": stage,
        "source_protocol": cfg["extended_budget"]["source_protocol"],
        "source_freeze": str(source_freeze),
        "source_freeze_sha256": file_sha256(source_freeze),
        "formal_seeds": seeds,
        "selected_batches": [16],
        "confirmed_hit_consecutive": int(cfg["extended_budget"]["confirmed_hit_consecutive"]),
        "epsilons": [float(value) for value in cfg["extended_budget"]["epsilons"]],
        "budgets": {method: int(value["train"]["rounds"]) for method, value in configs.items()},
        "labels": labels,
        "config_sha256": {method: object_sha256(value) for method, value in configs.items()},
        "only_protocol_change": "maximum training rounds",
        "max_parallel_tasks": int(max_parallel_tasks),
        "max_worker_processes": 8 * int(max_parallel_tasks),
    }
    atomic_write_json(root / "extension_protocol.json", protocol)
    result = run_labeled_tasks(
        labeled,
        root / "completion.json",
        max_parallel_tasks=max_parallel_tasks,
    )
    result["stage"] = stage
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage1", "stage2"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-parallel-tasks", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(args.output_root or Path(cfg["run"]["out_dir"]) / args.stage)
    result = run_stage(cfg, args.stage, root, args.max_parallel_tasks)
    print(
        f"stage={args.stage} status={result['status']} "
        f"completed={result['completed_tasks']}/{result['expected_tasks']} "
        f"failed={result['failed_tasks']}"
    )


if __name__ == "__main__":
    main()
