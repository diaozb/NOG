"""Fixed-work refinement for the four zeroth-order methods.

Every candidate receives at most the same total number of training SZO calls.
The runner stores one partial CSV per candidate and pilot seed.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
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
from src.distributed.zo_range_pilot import candidate_id  # noqa: E402
from src.synthetic.run_synthetic import get_device  # noqa: E402


Candidate = Tuple[str, Dict[str, Any]]


def refinement_candidates(cfg: Dict[str, Any]) -> List[Candidate]:
    grid = cfg["pilot"]["refine"]
    configured = set(cfg["methods"]["szo"])
    candidates: List[Candidate] = []

    if "NOG-ZO" in configured:
        for block, eta, smooth_batch in itertools.product(
            grid["nog"]["M"],
            grid["nog"]["eta"],
            grid["nog"]["smooth_B"],
        ):
            candidates.append(
                (
                    "NOG-ZO",
                    {
                        "M": int(block),
                        "eta": float(eta),
                        "smooth_B": int(smooth_batch),
                    },
                )
            )

    if "ME-DOL-ZO" in configured:
        for epoch, multiplier in itertools.product(
            grid["me_dol"]["epoch_length"],
            grid["me_dol"]["theory_multiplier"],
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
        for eta in grid["dgfm"]["eta"]:
            candidates.append(("DGFM", {"eta": float(eta)}))

    if "DGFM+" in configured:
        for eta in grid["dgfm_plus"]["eta"]:
            candidates.append(("DGFM+", {"eta": float(eta)}))

    return candidates


def nog_low_epsilon_candidates(cfg: Dict[str, Any]) -> List[Candidate]:
    grid = cfg["pilot"]["refine_nog_low_epsilon"]
    candidates: List[Candidate] = []
    for block, eta, smooth_batch in itertools.product(
        grid["M"],
        grid["eta"],
        grid["smooth_B"],
    ):
        candidates.append(
            (
                "NOG-ZO",
                {
                    "M": int(block),
                    "eta": float(eta),
                    "smooth_B": int(smooth_batch),
                },
            )
        )
    return candidates


def apply_candidate(cfg: Dict[str, Any], method: str, parameters: Dict[str, Any]) -> None:
    if method == "NOG-ZO":
        cfg["nog"]["M"] = int(parameters["M"])
        cfg["nog"]["eta"] = float(parameters["eta"])
        cfg["oracle"]["smooth_B"] = int(parameters["smooth_B"])
    elif method == "ME-DOL-ZO":
        cfg["me_dol"]["epoch_length"] = int(parameters["epoch_length"])
        cfg["me_dol"]["theory_multiplier"] = float(
            parameters["theory_multiplier"]
        )
    elif method == "DGFM":
        cfg["dgfm"]["eta"] = float(parameters["eta"])
    elif method == "DGFM+":
        cfg["dgfm_plus"]["eta"] = float(parameters["eta"])
    else:
        raise ValueError(f"Unsupported ZO method: {method}.")


def training_work_for_rounds(
    cfg: Dict[str, Any],
    method: str,
    rounds: int,
    worker_count: int,
) -> int:
    if method == "NOG-ZO":
        per_oracle = (
            2
            * int(cfg["oracle"]["smooth_B"])
            * int(cfg["oracle"]["data_B_total"])
        )
        return per_oracle * (int(rounds) + 2)

    if method == "ME-DOL-ZO":
        per_round = (
            2
            * int(cfg["me_dol"].get("smooth_B", 1))
            * int(cfg["me_dol"].get("data_B_per_worker", 1))
            * int(worker_count)
        )
        return per_round * int(rounds)

    if method == "DGFM":
        per_round = (
            2 * int(cfg["dgfm"].get("batch_size", 1)) * int(worker_count)
        )
        return per_round * int(rounds)

    if method == "DGFM+":
        period = int(cfg["dgfm_plus"]["restart_period"])
        restart_count = math.ceil(int(rounds) / period)
        ordinary_count = int(rounds) - restart_count
        per_worker = (
            restart_count * 2 * int(cfg["dgfm_plus"]["large_batch"])
            + ordinary_count * 4 * int(cfg["dgfm_plus"]["small_batch"])
        )
        return per_worker * int(worker_count)

    raise ValueError(method)


def rounds_at_work_budget(
    cfg: Dict[str, Any],
    method: str,
    target_work: int,
    worker_count: int,
) -> int:
    """Largest valid round count whose exact SZO work does not exceed budget."""

    if method == "NOG-ZO":
        per_oracle = (
            2
            * int(cfg["oracle"]["smooth_B"])
            * int(cfg["oracle"]["data_B_total"])
        )
        raw_rounds = target_work // per_oracle - 2
        divisor = int(cfg["nog"]["M"])
    elif method == "ME-DOL-ZO":
        per_round = training_work_for_rounds(cfg, method, 1, worker_count)
        raw_rounds = target_work // per_round
        divisor = int(cfg["me_dol"]["epoch_length"])
    elif method == "DGFM":
        per_round = training_work_for_rounds(cfg, method, 1, worker_count)
        raw_rounds = target_work // per_round
        divisor = 1
    elif method == "DGFM+":
        raw_rounds = 0
        while training_work_for_rounds(
            cfg, method, raw_rounds + 1, worker_count
        ) <= target_work:
            raw_rounds += 1
        # Align baseline-only runs with the shared NOG checkpoint grid required
        # by the common configuration validator.
        divisor = int(cfg["nog"]["M"])
    else:
        raise ValueError(method)

    rounds = int(raw_rounds // divisor * divisor)
    if rounds < 1:
        raise ValueError(
            f"Budget {target_work} is too small for {method} with current batch."
        )
    return rounds


def _parse_ints(value: str | None, fallback: Iterable[int]) -> List[int]:
    if value is None:
        return [int(item) for item in fallback]
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _parse_methods(value: str | None, fallback: Iterable[str]) -> List[str]:
    if value is None:
        return [str(item) for item in fallback]
    return [item.strip() for item in value.split(",") if item.strip()]


def run_one(
    base_cfg: Dict[str, Any],
    method: str,
    parameters: Dict[str, Any],
    seed: int,
    target_work: int,
    worker_count: int,
    device: str,
) -> pd.DataFrame:
    cfg = copy.deepcopy(base_cfg)
    apply_candidate(cfg, method, parameters)
    cfg["train"]["eval_every"] = int(cfg["pilot"]["refine"]["eval_every"])
    rounds = rounds_at_work_budget(cfg, method, target_work, worker_count)
    cfg["train"]["rounds"] = rounds
    validate_experiment_config(cfg)

    frame = run_selected(cfg, [method], [seed], [worker_count], device)
    actual_work = int(frame.sort_values("iteration").iloc[-1]["total_work"])
    expected_work = training_work_for_rounds(cfg, method, rounds, worker_count)
    if actual_work != expected_work:
        raise RuntimeError(
            f"Work accounting mismatch for {method}: "
            f"runner={actual_work}, formula={expected_work}."
        )
    if actual_work > target_work:
        raise RuntimeError(
            f"Work budget exceeded for {method}: {actual_work}>{target_work}."
        )

    frame["candidate_id"] = candidate_id(method, parameters)
    frame["candidate_parameters"] = json.dumps(parameters, sort_keys=True)
    frame["candidate_rounds"] = rounds
    frame["target_total_work"] = int(target_work)
    return frame


def summarize(results: pd.DataFrame, epsilons: Iterable[float]) -> pd.DataFrame:
    keys = ["method", "candidate_id", "candidate_parameters", "formal_seed"]
    ordered = results.sort_values("iteration")
    final = ordered.groupby(keys, as_index=False).tail(1)
    minima = (
        ordered.groupby(keys, as_index=False)["stat_proxy"]
        .min()
        .rename(columns={"stat_proxy": "min_stat_proxy"})
    )

    hit_rows = []
    epsilon_values = sorted((float(value) for value in epsilons), reverse=True)
    for key, frame in ordered.groupby(keys, sort=False):
        frame = frame.sort_values("iteration").reset_index(drop=True)
        lowest_hit = math.nan
        for epsilon in epsilon_values:
            below = frame["stat_proxy"].le(epsilon)
            confirmed = below & below.shift(-1, fill_value=False)
            if confirmed.any():
                lowest_hit = epsilon
        hit_rows.append((*key, lowest_hit))
    hits = pd.DataFrame(
        hit_rows,
        columns=[*keys, "lowest_confirmed_epsilon"],
    )

    per_seed = final.merge(minima, on=keys, how="left").merge(
        hits, on=keys, how="left"
    )
    summary = per_seed.groupby(
        ["method", "candidate_id", "candidate_parameters"], as_index=False
    ).agg(
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_std=("stat_proxy", "std"),
        min_stat_proxy_mean=("min_stat_proxy", "mean"),
        lowest_confirmed_epsilon_mean=("lowest_confirmed_epsilon", "mean"),
        final_objective_mean=("objective", "mean"),
        final_depth_mean=("communication_round", "mean"),
        final_total_work_mean=("total_work", "mean"),
        candidate_rounds=("candidate_rounds", "first"),
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
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--target-work", type=int, default=None)
    parser.add_argument("--grid", choices=["refine", "nog-low-epsilon"], default="refine")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--nog-etas", default=None)
    parser.add_argument("--top-per-method", type=int, default=None)
    parser.add_argument("--selection-summary", default=None)
    parser.add_argument("--eval-smooth-b", type=int, default=None)
    parser.add_argument("--eval-data-b", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--nog-ms", default=None)
    parser.add_argument("--nog-smooth-bs", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.eval_smooth_b is not None:
        cfg["oracle"]["eval_smooth_B"] = int(args.eval_smooth_b)
    if args.eval_data_b is not None:
        cfg["oracle"]["eval_data_B"] = int(args.eval_data_b)
    if args.eval_every is not None:
        cfg["pilot"]["refine"]["eval_every"] = int(args.eval_every)
    seeds = _parse_ints(args.seeds, cfg["run"]["pilot_seeds"])
    formal_seeds = {int(seed) for seed in cfg["run"]["formal_seeds"]}
    overlap = formal_seeds.intersection(seeds)
    if overlap:
        raise ValueError(f"Refinement cannot use formal seeds: {sorted(overlap)}.")

    if args.grid == "nog-low-epsilon":
        default_methods = ["NOG-ZO"]
        all_candidates = nog_low_epsilon_candidates(cfg)
    else:
        default_methods = cfg["methods"]["szo"]
        all_candidates = refinement_candidates(cfg)
    requested = set(_parse_methods(args.methods, default_methods))
    candidates = [
        candidate
        for candidate in all_candidates
        if candidate[0] in requested
    ]
    if args.nog_etas is not None:
        allowed_etas = set(_parse_floats(args.nog_etas))
        candidates = [
            candidate
            for candidate in candidates
            if float(candidate[1].get("eta", float("nan"))) in allowed_etas
        ]
    if args.nog_ms is not None:
        allowed_blocks = set(_parse_ints(args.nog_ms, []))
        candidates = [
            candidate for candidate in candidates
            if int(candidate[1].get("M", -1)) in allowed_blocks
        ]
    if args.nog_smooth_bs is not None:
        allowed_batches = set(_parse_ints(args.nog_smooth_bs, []))
        candidates = [
            candidate for candidate in candidates
            if int(candidate[1].get("smooth_B", -1)) in allowed_batches
        ]
    unknown = requested - {candidate[0] for candidate in candidates}
    if unknown:
        raise ValueError(f"No refinement candidates for: {sorted(unknown)}.")

    worker_count = int(
        cfg["distributed"]["comparison_worker"]
        if args.workers is None
        else args.workers
    )
    default_target_work = (
        cfg["pilot"]["refine_nog_low_epsilon"]["target_total_work"]
        if args.grid == "nog-low-epsilon"
        else cfg["pilot"]["refine"]["target_total_work"]
    )
    target_work = int(default_target_work if args.target_work is None else args.target_work)
    device = get_device(cfg["run"].get("device", "auto"))
    stage_name = f"refine_work_{target_work}"
    if args.output_tag:
        stage_name = f"{stage_name}_{args.output_tag}"
    out_dir = (
        Path(cfg["run"]["out_dir"])
        / cfg["run"]["name"]
        / "pilot"
        / stage_name
    )
    partial_dir = out_dir / "partials"

    if args.top_per_method is not None:
        if args.top_per_method < 1:
            raise ValueError("top-per-method must be positive.")
        selection_path = (
            Path(args.selection_summary)
            if args.selection_summary is not None
            else out_dir / "summary.csv"
        )
        if not selection_path.exists():
            raise FileNotFoundError(
                "Single-seed summary is required before top-candidate expansion."
            )
        selection = pd.read_csv(selection_path)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / "summary_seed100_all.csv"
        if not args.dry_run and not archive_path.exists():
            selection.to_csv(archive_path, index=False)
        selected = (
            selection.sort_values(
                ["method", "final_stat_proxy_mean", "final_depth_mean"]
            )
            .groupby("method", as_index=False)
            .head(args.top_per_method)
            .reset_index(drop=True)
        )
        selected_ids = set(selected["candidate_id"])
        candidates = [
            row for row in candidates if candidate_id(*row) in selected_ids
        ]
        if not args.dry_run:
            selected.to_csv(out_dir / "advanced_candidates.csv", index=False)

    task_specs = []
    for method, parameters in candidates:
        candidate_cfg = copy.deepcopy(cfg)
        apply_candidate(candidate_cfg, method, parameters)
        rounds = rounds_at_work_budget(
            candidate_cfg, method, target_work, worker_count
        )
        actual_work = training_work_for_rounds(
            candidate_cfg, method, rounds, worker_count
        )
        task_specs.append((method, parameters, rounds, actual_work))

    print(
        f"device={device} candidates={len(candidates)} seeds={seeds} "
        f"workers={worker_count} target_work={target_work}",
        flush=True,
    )
    if args.dry_run:
        for method, parameters, rounds, actual_work in task_specs:
            print(
                f"{candidate_id(method, parameters)} "
                f"rounds={rounds} work={actual_work}"
            )
        return

    partial_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config_base.yaml")
    with open(out_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(environment_record(device), handle, indent=2)

    frames = []
    tasks = [
        (method, parameters, seed)
        for method, parameters, _, _ in task_specs
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
                target_work,
                worker_count,
                device,
            )
            frame.to_csv(path, index=False)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    summary = summarize(results, cfg["epsilon_scaling"]["epsilons"])
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
                "lowest_confirmed_epsilon_mean",
                "final_depth_mean",
                "final_total_work_mean",
                "candidate_rounds",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"saved={out_dir}", flush=True)


if __name__ == "__main__":
    main()
