#!/usr/bin/env python3
"""Freeze the best validation-confirmed joint batch/M/eta SVM candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.validation_summary = args.validation_summary.resolve()
    args.protocol = args.protocol.resolve()
    args.output = args.output.resolve()

    import pandas as pd

    validation = pd.read_csv(args.validation_summary)
    required = {"dataset", "candidate_parameters", "rank", "seed_count"}
    missing = required - set(validation.columns)
    if missing:
        raise ValueError(f"Validation summary missing columns: {sorted(missing)}")
    selected = validation[validation["rank"].astype(int) == 1].copy()
    if set(selected["dataset"].astype(str)) != {"a9a", "ijcnn1"}:
        raise ValueError("Expected exactly one rank-1 candidate for each dataset.")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    candidates = []
    for _, row in selected.sort_values("dataset").iterrows():
        candidates.append(
            {
                "dataset": str(row["dataset"]),
                "method": "NOG-ZO",
                "parameters": json.loads(str(row["candidate_parameters"])),
                "validation_rank": int(row["rank"]),
                "validation_seed_count": int(row["seed_count"]),
            }
        )

    input_paths = [
        args.protocol,
        args.validation_summary,
        ROOT / "configs/distributed_zo_batch_svm_a9a.yaml",
        ROOT / "configs/distributed_zo_batch_svm_ijcnn1.yaml",
        ROOT / "src/distributed/zo_refine_pilot.py",
        ROOT / "src/distributed/common.py",
        ROOT / "src/distributed/algorithms.py",
        ROOT / "src/distributed/real_data.py",
        ROOT / "scripts/run_retuned_nog_formal.py",
    ]
    payload = {
        "status": "frozen",
        "stage": "svm-batch-retuned-formal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "selection_happened_before_formal_runs": True,
        "test_data_used_for_parameter_selection": False,
        "seed_sets_disjoint": True,
        "search_seeds": protocol["search_seeds"],
        "validation_seeds": protocol["validation_seeds"],
        "formal_seeds": protocol["formal_seeds"],
        "formal_target_total_work": int(protocol["target_total_work"]),
        "worker_count": int(protocol["worker_count"]),
        "eval_every": int(protocol["common_settings"]["pilot_eval_every"]),
        "eval_smooth_B": int(protocol["common_settings"].get("eval_smooth_B", 32)),
        "eval_data_B": int(protocol["common_settings"].get("eval_data_B", 256)),
        "config_by_dataset": {
            "a9a": "configs/distributed_zo_batch_svm_a9a.yaml",
            "ijcnn1": "configs/distributed_zo_batch_svm_ijcnn1.yaml",
        },
        "selected_candidates": candidates,
        "selection_metric": protocol["selection_metric"],
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in input_paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
