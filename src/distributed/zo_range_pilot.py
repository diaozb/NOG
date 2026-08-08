"""Resumable one-seed/multi-seed range search for the four ZO algorithms.

This stage only identifies viable parameter ranges.  It never uses formal seeds
and does not freeze parameters for the paper experiment.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.distributed.common import validate_experiment_config  # noqa: E402
from src.distributed.run_distributed_baselines import (  # noqa: E402
    environment_record,
    load_config,
    run_selected,
    save_yaml,
)
from src.synthetic.run_synthetic import get_device  # noqa: E402


Candidate = Tuple[str, Dict[str, Any]]


def _slug(value: Any) -> str:
    return str(value).replace("+", "plus").replace(".", "p").replace("-", "m")


def candidate_id(method: str, parameters: Dict[str, Any]) -> str:
    pieces = [method.replace("+", "plus")]
    pieces.extend(f"{key}-{_slug(value)}" for key, value in sorted(parameters.items()))
    return "__".join(pieces)


def range_candidates(cfg: Dict[str, Any]) -> List[Candidate]:
    """Return the broad range grid while holding batch constants fixed."""

    configured = set(cfg["methods"]["szo"])
    pilot = cfg["pilot"]
    candidates: List[Candidate] = []

    if "NOG-ZO" in configured:
        for block, eta in itertools.product(
            pilot["nog"]["M"],
            pilot["nog"]["eta"],
        ):
            candidates.append(
                ("NOG-ZO", {"M": int(block), "eta": float(eta)})
            )

    if "ME-DOL-ZO" in configured:
        for epoch, multiplier in itertools.product(
            pilot["me_dol"]["epoch_length"],
            pilot["me_dol"]["theory_multiplier"],
        ):
            candidates.append(
                (
                    "ME-DOL-ZO",
                    {
                        "epoch_length": int(epoch),
                        "theory_multiplier": float(multiplier),
                    },
                )
            )

    if "DGFM" in configured:
        for eta in pilot["dgfm"]["eta"]:
            candidates.append(("DGFM", {"eta": float(eta)}))

    if "DGFM+" in configured:
        for eta in pilot["dgfm_plus"]["eta"]:
            candidates.append(("DGFM+", {"eta": float(eta)}))

    return candidates


def apply_candidate(cfg: Dict[str, Any], method: str, parameters: Dict[str, Any]) -> None:
    if method == "NOG-ZO":
        cfg["nog"].update(parameters)
    elif method == "ME-DOL-ZO":
        cfg["me_dol"].update(parameters)
    elif method == "DGFM":
        cfg["dgfm"].update(parameters)
    elif method == "DGFM+":
        cfg["dgfm_plus"].update(parameters)
    else:
        raise ValueError(f"Unsupported ZO method: {method}.")


def _parse_ints(value: str | None, fallback: Iterable[int]) -> List[int]:
    if value is None:
        return [int(item) for item in fallback]
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_methods(value: str | None, fallback: Iterable[str]) -> List[str]:
    if value is None:
        return [str(item) for item in fallback]
    return [item.strip() for item in value.split(",") if item.strip()]


def run_one(
    base_cfg: Dict[str, Any],
    method: str,
    parameters: Dict[str, Any],
    seed: int,
    rounds: int,
    worker_count: int,
    device: str,
) -> pd.DataFrame:
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["rounds"] = int(rounds)
    apply_candidate(cfg, method, parameters)
    validate_experiment_config(cfg)
    frame = run_selected(cfg, [method], [seed], [worker_count], device)
    frame["candidate_id"] = candidate_id(method, parameters)
    frame["candidate_parameters"] = json.dumps(parameters, sort_keys=True)
    return frame


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "candidate_id", "candidate_parameters", "formal_seed"]
    ordered = results.sort_values("iteration")
    final = ordered.groupby(keys, as_index=False).tail(1)
    minima = (
        ordered.groupby(keys, as_index=False)["stat_proxy"]
        .min()
        .rename(columns={"stat_proxy": "min_stat_proxy"})
    )
    per_seed = final.merge(minima, on=keys, how="left")
    summary = per_seed.groupby(
        ["method", "candidate_id", "candidate_parameters"], as_index=False
    ).agg(
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_std=("stat_proxy", "std"),
        min_stat_proxy_mean=("min_stat_proxy", "mean"),
        final_objective_mean=("objective", "mean"),
        final_depth_mean=("communication_round", "mean"),
        final_total_work_mean=("total_work", "mean"),
        seed_count=("formal_seed", "nunique"),
    )
    return summary.sort_values(
        ["method", "final_stat_proxy_mean", "final_depth_mean"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/distributed_zo_theory_validation.yaml",
    )
    parser.add_argument("--rounds", type=int, default=240)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = _parse_ints(args.seeds, cfg["run"]["pilot_seeds"])
    formal_seeds = {int(seed) for seed in cfg["run"]["formal_seeds"]}
    overlap = formal_seeds.intersection(seeds)
    if overlap:
        raise ValueError(f"Range search cannot use formal seeds: {sorted(overlap)}.")

    requested_methods = set(_parse_methods(args.methods, cfg["methods"]["szo"]))
    candidates = [
        candidate
        for candidate in range_candidates(cfg)
        if candidate[0] in requested_methods
    ]
    unknown = requested_methods - {candidate[0] for candidate in candidates}
    if unknown:
        raise ValueError(f"No range candidates for methods: {sorted(unknown)}.")

    worker_count = int(
        cfg["distributed"]["comparison_worker"]
        if args.workers is None
        else args.workers
    )
    device = get_device(cfg["run"].get("device", "auto"))
    out_dir = (
        Path(cfg["run"]["out_dir"])
        / cfg["run"]["name"]
        / "pilot"
        / f"range_r{args.rounds}"
    )
    partial_dir = out_dir / "partials"

    print(
        f"device={device} candidates={len(candidates)} seeds={seeds} "
        f"workers={worker_count} rounds={args.rounds}",
        flush=True,
    )
    if args.dry_run:
        for method, parameters in candidates:
            print(candidate_id(method, parameters))
        return

    partial_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config_base.yaml")
    with open(out_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(environment_record(device), handle, indent=2)

    frames = []
    tasks = [
        (method, parameters, seed)
        for method, parameters in candidates
        for seed in seeds
    ]
    for index, (method, parameters, seed) in enumerate(tasks, start=1):
        identifier = candidate_id(method, parameters)
        path = partial_dir / f"{identifier}__seed-{seed}.csv"
        if path.exists():
            print(f"[{index}/{len(tasks)}] resume {identifier} seed={seed}", flush=True)
            frame = pd.read_csv(path)
        else:
            print(f"[{index}/{len(tasks)}] run {identifier} seed={seed}", flush=True)
            frame = run_one(
                cfg,
                method,
                parameters,
                seed,
                args.rounds,
                worker_count,
                device,
            )
            frame.to_csv(path, index=False)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results)
    results.to_csv(out_dir / "results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    best = summary.groupby("method", as_index=False).head(1)
    print("best_by_method:", flush=True)
    print(
        best[
            [
                "method",
                "candidate_parameters",
                "final_stat_proxy_mean",
                "min_stat_proxy_mean",
                "final_depth_mean",
                "final_total_work_mean",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
