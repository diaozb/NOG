"""Run and analyze one selected v4 extended-budget stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, utc_now
from src.distributed.theory_validation_v4_extension_analysis import analyze
from src.distributed.theory_validation_v4_extension_runner import CONFIG, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stage1", "stage2"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--max-parallel-tasks", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_root = Path(cfg["run"]["out_dir"])
    stage_root = output_root / args.stage
    run_result = run_stage(cfg, args.stage, stage_root, args.max_parallel_tasks)
    result = analyze(cfg, stage_root, stage_root / "analysis")
    record = {
        "stage": args.stage,
        "run_status": run_result["status"],
        "completed_tasks": run_result["completed_tasks"],
        "expected_tasks": run_result["expected_tasks"],
        "fully_observed_epsilons": result["fully_observed_epsilons"],
    }
    prior = []
    status_path = output_root / "pipeline_status.json"
    if status_path.exists():
        import json

        prior = [row for row in json.load(open(status_path, "r", encoding="utf-8")).get("stages", []) if row.get("stage") != args.stage]
    stages = prior + [record]
    expected = len(cfg["extended_budget"]["epsilons"])
    atomic_write_json(
        status_path,
        {
            "schema_version": 1,
            "status": "complete",
            "updated_at_utc": utc_now(),
            "expected_epsilon_count": expected,
            "all_epsilons_fully_observed": len(result["fully_observed_epsilons"]) == expected,
            "stages": stages,
        },
    )
    print(
        f"stage={args.stage} status=complete "
        f"fully_observed={len(result['fully_observed_epsilons'])}/{expected}"
    )


if __name__ == "__main__":
    main()
