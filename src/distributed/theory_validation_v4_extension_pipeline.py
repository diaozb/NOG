"""Run, audit, and analyze staged v4 extended-budget continuations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, utc_now
from src.distributed.theory_validation_v4_extension_analysis import analyze
from src.distributed.theory_validation_v4_extension_runner import CONFIG, run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--max-parallel-tasks", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output_root = Path(cfg["run"]["out_dir"])
    expected = len(cfg["extended_budget"]["epsilons"])
    records = []

    for stage in ["stage1", "stage2"]:
        stage_root = output_root / stage
        run_result = run_stage(cfg, stage, stage_root, args.max_parallel_tasks)
        analysis = analyze(cfg, stage_root, stage_root / "analysis")
        record = {
            "stage": stage,
            "run_status": run_result["status"],
            "completed_tasks": run_result["completed_tasks"],
            "expected_tasks": run_result["expected_tasks"],
            "fully_observed_epsilons": analysis["fully_observed_epsilons"],
        }
        records.append(record)
        atomic_write_json(
            output_root / "pipeline_status.json",
            {
                "schema_version": 1,
                "status": "running",
                "updated_at_utc": utc_now(),
                "expected_epsilon_count": expected,
                "stages": records,
            },
        )
        if len(analysis["fully_observed_epsilons"]) == expected:
            break

    final = {
        "schema_version": 1,
        "status": "complete",
        "updated_at_utc": utc_now(),
        "expected_epsilon_count": expected,
        "all_epsilons_fully_observed": len(records[-1]["fully_observed_epsilons"]) == expected,
        "stages": records,
    }
    atomic_write_json(output_root / "pipeline_status.json", final)
    print(
        f"status=complete final_stage={records[-1]['stage']} "
        f"fully_observed={len(records[-1]['fully_observed_epsilons'])}/{expected}"
    )


if __name__ == "__main__":
    main()
