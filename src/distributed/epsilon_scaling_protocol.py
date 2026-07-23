"""Prepare the preregistered wide-epsilon experiment without launching workers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, object_sha256, utc_now
from src.distributed.epsilon_scaling import epsilon_region, validate_scaling_protocol


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "wide-epsilon-regions-censoring-v1"


def _region_epsilons(protocol: Dict[str, Any], region: str) -> List[float]:
    return [
        float(value)
        for value in protocol["epsilons"]
        if epsilon_region(float(value)) == region
    ]


def prepare_protocol(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a deterministic manifest; this function never starts a process."""
    validate_scaling_protocol(cfg)
    protocol = cfg["epsilon_scaling"]
    methods = [str(value) for value in cfg["methods"]["sfo"]]
    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    pilot_seeds = [int(value) for value in protocol["pilot_seeds"]]
    regions = []
    for name in ("coarse", "medium", "fine"):
        regions.append(
            {
                "name": name,
                "epsilons": _region_epsilons(protocol, name),
                "representative_epsilons": [
                    float(value)
                    for value in protocol["regions"][name]["representative_epsilons"]
                ],
            }
        )

    primary_trajectories = [
        {
            "method": method,
            "region": region["name"],
            "worker_count": int(protocol["reference_worker"]),
            "formal_seeds": formal_seeds,
            "epsilons": region["epsilons"],
        }
        for method in methods
        for region in regions
    ]
    robustness = protocol["robustness"]
    robustness_settings = [
        {
            "method": method,
            "epsilon": float(epsilon),
            "region": epsilon_region(float(epsilon)),
            "worker_count": int(worker),
            "formal_seeds": [int(value) for value in robustness["seeds"]],
        }
        for method in methods
        for epsilon in robustness["epsilons"]
        for worker in robustness["workers"]
    ]

    pilot_cfg = protocol["pilot"]
    nog_count = len(pilot_cfg["nog"]["M"]) * len(pilot_cfg["nog"]["eta"])
    me_count = (
        len(pilot_cfg["me_dol"]["epoch_length"])
        * len(pilot_cfg["me_dol"]["theory_multiplier"])
    )
    manifest = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "launches_started": 0,
        "epsilons": [float(value) for value in protocol["epsilons"]],
        "epsilon_range": [min(protocol["epsilons"]), max(protocol["epsilons"])],
        "budgets": [int(value) for value in protocol["budgets"]],
        "pilot_seeds": pilot_seeds,
        "formal_seeds": formal_seeds,
        "regions": regions,
        "primary_trajectory_groups": primary_trajectories,
        "primary_physical_task_count": len(primary_trajectories) * len(formal_seeds),
        "primary_logical_method_epsilon_seed_rows": (
            len(methods) * len(protocol["epsilons"]) * len(formal_seeds)
        ),
        "pilot_candidate_count": nog_count + me_count,
        "pilot_candidate_count_by_method": {"NOG-FO": nog_count, "ME-DOL-FO": me_count},
        "pilot_initial_physical_task_count": (nog_count + me_count) * len(pilot_seeds),
        "pilot_refinement_smooth_B": [
            int(value) for value in pilot_cfg["nog"]["refinement_smooth_B"]
        ],
        "robustness_settings": robustness_settings,
        "robustness_physical_task_count": len(robustness_settings) * len(robustness["seeds"]),
        "max_total_worker_processes": int(cfg["cpu_process"]["max_total_worker_processes"]),
        "task_timeout_seconds": int(cfg["cpu_process"]["task_timeout_seconds"]),
        "config_sha256": object_sha256(cfg),
        "notes": [
            "Preparing this manifest launches no worker process.",
            "Each method-region-seed trajectory is reused across all epsilons in that region.",
            "Pilot candidates advance by preregistered staged gates rather than all running to 61440.",
            "Censored seeds remain in every summary.",
        ],
    }
    manifest["manifest_content_sha256"] = object_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/distributed_cpu_fo_epsilon_scaling.yaml",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = prepare_protocol(cfg)
    root = Path(args.output_root or cfg["run"]["out_dir"]) / cfg["run"]["name"]
    atomic_write_json(root / "protocol_manifest.json", manifest)
    print(
        f"prepared={root / 'protocol_manifest.json'} "
        f"primary_tasks={manifest['primary_physical_task_count']} "
        f"pilot_initial_tasks={manifest['pilot_initial_physical_task_count']} "
        f"robustness_tasks={manifest['robustness_physical_task_count']}"
    )


if __name__ == "__main__":
    main()
