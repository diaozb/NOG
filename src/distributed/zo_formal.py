"""Resumable formal fixed-work experiment for the four ZO methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.distributed.run_distributed_baselines import (  # noqa: E402
    environment_record,
    load_config,
    save_yaml,
)
from src.distributed.zo_range_pilot import candidate_id  # noqa: E402
from src.distributed.zo_refine_pilot import run_one, summarize  # noqa: E402
from src.synthetic.run_synthetic import get_device  # noqa: E402


DEFAULT_FREEZE = ROOT / "zo_experiments/frozen_parameters.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/distributed_zo/zo_theory_validation/formal/fixed_work_983040"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_and_verify_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "frozen":
        raise ValueError("ZO parameters are not frozen.")
    if not freeze.get("selection_happened_before_formal_runs"):
        raise ValueError("Formal execution requires pre-frozen parameters.")
    if not freeze.get("seed_sets_disjoint"):
        raise ValueError("Pilot and formal seed sets must be disjoint.")
    if set(freeze["pilot_seeds"]).intersection(freeze["formal_seeds"]):
        raise ValueError("Pilot and formal seed sets overlap.")
    for relative, expected in freeze["input_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {relative}: {actual}!={expected}"
            )
    return freeze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", default=str(DEFAULT_FREEZE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_path = Path(args.freeze).resolve()
    freeze = load_and_verify_freeze(freeze_path)
    config_path = ROOT / freeze["base_config"]
    cfg = load_config(config_path)
    cfg["run"]["pilot_selection_complete"] = True
    cfg["pilot"]["refine"]["target_total_work"] = int(
        freeze["target_total_work"]
    )
    cfg["pilot"]["refine"]["eval_every"] = int(freeze["eval_every"])
    cfg["oracle"]["eval_smooth_B"] = int(freeze["eval_smooth_B"])
    cfg["oracle"]["eval_data_B"] = int(freeze["eval_data_B"])

    allowed_seeds = [int(value) for value in freeze["formal_seeds"]]
    seeds = (
        allowed_seeds
        if args.seeds is None
        else [int(value) for value in args.seeds.split(",") if value.strip()]
    )
    if not set(seeds).issubset(allowed_seeds):
        raise ValueError("Requested seeds must be a subset of frozen formal seeds.")

    candidates = [
        (entry["method"], entry["parameters"])
        for entry in freeze["selected_candidates"]
    ]
    target_work = int(freeze["target_total_work"])
    worker_count = int(freeze["worker_count"])
    device = get_device(cfg["run"].get("device", "auto"))
    output = Path(args.output).resolve()
    partials = output / "partials"
    tasks = [
        (method, parameters, seed)
        for method, parameters in candidates
        for seed in seeds
    ]

    print(
        f"device={device} tasks={len(tasks)} methods={len(candidates)} "
        f"seeds={seeds} target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for method, parameters, seed in tasks:
            print(f"{candidate_id(method, parameters)} seed={seed}", flush=True)
        return

    partials.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freeze_path, output / "frozen_parameters.json")
    save_yaml(cfg, output / "config_frozen.yaml")
    atomic_json(environment_record(device), output / "environment.json")

    completed_frames = []
    for index, (method, parameters, seed) in enumerate(tasks, start=1):
        identifier = candidate_id(method, parameters)
        path = partials / f"{identifier}__seed-{seed}.csv"
        if path.exists():
            print(
                f"[{index}/{len(tasks)}] resume {identifier} seed={seed}",
                flush=True,
            )
            frame = pd.read_csv(path)
        else:
            print(
                f"[{index}/{len(tasks)}] run {identifier} seed={seed}",
                flush=True,
            )
            frame = run_one(
                cfg,
                method,
                parameters,
                seed,
                target_work,
                worker_count,
                device,
            )
            atomic_csv(frame, path)
        completed_frames.append(frame)
        atomic_json(
            {
                "status": "running",
                "completed_tasks": index,
                "total_tasks": len(tasks),
                "last_method": method,
                "last_seed": seed,
            },
            output / "progress.json",
        )

    results = pd.concat(completed_frames, ignore_index=True)
    summary = summarize(results, cfg["epsilon_scaling"]["epsilons"])
    atomic_csv(results, output / "results.csv")
    atomic_csv(summary, output / "summary.csv")
    atomic_json(
        {
            "status": "complete",
            "completed_tasks": len(tasks),
            "total_tasks": len(tasks),
            "methods": [method for method, _ in candidates],
            "formal_seeds": seeds,
        },
        output / "progress.json",
    )
    print(summary.to_string(index=False), flush=True)
    print(f"saved={output}", flush=True)


if __name__ == "__main__":
    main()
