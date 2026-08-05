"""Run two short v4-extension tasks and verify exact numerical prefixes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import CpuFoTask
from src.distributed.theory_validation_runner import run_labeled_tasks
from src.distributed.theory_validation_v4_extension_runner import me_config, nog_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/distributed_cpu_fo_theory_validation_v4_extended_budget.yaml",
    )
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(args.output_root or tempfile.mkdtemp(prefix="nog-v4-ext-smoke-", dir="/tmp"))
    candidates = {
        "NOG-FO": nog_config(cfg, 964),
        "ME-DOL-FO": me_config(cfg, 3846),
    }
    labels = {
        "NOG-FO": "NOG-FO__rounds-964__data-B-total-16",
        "ME-DOL-FO": "ME-DOL-FO__epoch-6__mult-100__rounds-3846",
    }
    items = [
        (method, candidates[method], CpuFoTask(method, 0, 8), root / labels[method])
        for method in ["NOG-FO", "ME-DOL-FO"]
    ]
    completion = run_labeled_tasks(items, root / "completion.json", max_parallel_tasks=2)
    payloads = {
        (str(record["task"]["method"]), int(record["task"]["formal_seed"])): json.load(
            open(record["partial_path"], "r", encoding="utf-8")
        )
        for record in completion["records"]
    }
    source_completion = json.load(
        open(
            Path(cfg["extended_budget"]["source_formal_root"]) / "completion.json",
            "r",
            encoding="utf-8",
        )
    )
    timing = {"time_sec", "training_time", "communication_time", "evaluation_time", "smooth_B", "data_B_per_worker"}
    checked_rows = 0
    for (method, seed), payload in payloads.items():
        record = next(
            row
            for row in source_completion["records"]
            if row["task"]["method"] == method
            and int(row["task"]["formal_seed"]) == seed
            and (method != "NOG-FO" or "data-B-total-16" in row["label"])
        )
        source = json.load(open(record["partial_path"], "r", encoding="utf-8"))
        assert len(payload["rows"]) > len(source["rows"])
        for old, new in zip(source["rows"], payload["rows"]):
            assert {key: value for key, value in old.items() if key not in timing} == {
                key: value for key, value in new.items() if key not in timing
            }
            checked_rows += 1
    print(
        f"status=passed tasks={len(payloads)} rows={checked_rows} "
        f"output_root={root}"
    )


if __name__ == "__main__":
    main()
