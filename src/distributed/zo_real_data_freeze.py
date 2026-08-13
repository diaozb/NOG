"""Freeze Step ZO-9C parameters before any formal real-data trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / (
    "outputs/distributed_zo/zo_theory_validation/real_data/"
    "calibration_final_boundary_work98304"
)
DEFAULT_OUTPUT = ROOT / "zo_experiments/real_data/frozen_parameters.json"
CONFIG = ROOT / "configs/distributed_zo_real_data_calibration_final_boundary.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default=str(CALIBRATION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = Path(args.calibration).resolve()
    output = Path(args.output).resolve()
    progress_path = calibration / "progress.json"
    selected_path = calibration / "selected_parameters.csv"
    progress = json.loads(progress_path.read_text())
    if progress.get("status") != "complete":
        raise ValueError("Final-boundary calibration is not complete.")
    if not progress.get("selection_before_formal_runs"):
        raise ValueError("Calibration does not certify pre-formal selection.")
    pilot_seeds = [int(value) for value in progress["pilot_seeds"]]
    formal_seeds = [int(value) for value in progress["formal_seeds"]]
    if set(pilot_seeds).intersection(formal_seeds):
        raise ValueError("Pilot and formal seeds overlap.")

    selected = pd.read_csv(selected_path)
    expected = {(dataset, method) for dataset in ["a9a", "ijcnn1"] for method in [
        "NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"
    ]}
    observed = set(zip(selected["dataset"], selected["method"]))
    if observed != expected or len(selected) != len(expected):
        raise ValueError("Expected exactly one selected row per dataset and method.")
    if not (selected["seed_count"].astype(int) == len(pilot_seeds)).all():
        raise ValueError("Not every selected candidate has all pilot seeds.")

    entries = []
    for row in selected.sort_values(["dataset", "method"]).itertuples(index=False):
        entries.append(
            {
                "dataset": row.dataset,
                "method": row.method,
                "parameters": json.loads(row.candidate_parameters),
                "calibration_final_objective_mean": row.final_objective_mean,
                "calibration_final_objective_std": row.final_objective_std,
                "calibration_final_stat_proxy_mean": row.final_stat_proxy_mean,
                "calibration_final_stat_proxy_std": row.final_stat_proxy_std,
            }
        )
    audit_dir = output.parent / "calibration_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_sources = {
        selected_path: audit_dir / "selected_parameters.csv",
        calibration / "replicated_summary.csv": audit_dir
        / "replicated_summary.csv",
        progress_path: audit_dir / "progress.json",
        calibration / "data_manifest.json": audit_dir / "data_manifest.json",
    }
    for source, destination in audit_sources.items():
        shutil.copy2(source, destination)
    implementation_inputs = [
        ROOT / "src/distributed/algorithms.py",
        ROOT / "src/distributed/common.py",
        ROOT / "src/distributed/real_data.py",
        ROOT / "src/distributed/zo_refine_pilot.py",
        ROOT / "src/distributed/zo_real_data_formal.py",
    ]
    inputs = [CONFIG, *audit_sources.values(), *implementation_inputs]
    payload = {
        "status": "frozen",
        "stage": "ZO-9C3",
        "selection_happened_before_formal_runs": True,
        "seed_sets_disjoint": True,
        "pilot_seeds": pilot_seeds,
        "formal_seeds": formal_seeds,
        "worker_count": int(progress["worker_count"]),
        "calibration_target_total_work": int(progress["target_total_work"]),
        "formal_target_total_work": 983040,
        "eval_every": 384,
        "eval_smooth_B": 32,
        "eval_data_B": 256,
        "selection_metric": "mean_final_objective_then_stationarity",
        "base_config": str(CONFIG.relative_to(ROOT)),
        "selected_candidates": entries,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in inputs
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
