"""Resumable pilot-grid runner for distributed baselines.

Only pilot seeds may be used here.  Candidate selection is performed at a
common training-work budget within each method, so larger DGFM+ batches do not
win merely because they spend more SZO calls.
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.distributed.run_distributed_baselines import (  # noqa: E402
    load_config,
    run_selected,
    save_yaml,
)
from src.distributed.common import validate_experiment_config  # noqa: E402
from src.synthetic.run_synthetic import get_device  # noqa: E402


def slug(value: Any) -> str:
    return str(value).replace("+", "plus").replace(".", "p").replace("-", "m")


def candidate_id(method: str, parameters: Dict[str, Any]) -> str:
    pieces = [method.replace("+", "plus")]
    pieces.extend(f"{key}-{slug(value)}" for key, value in sorted(parameters.items()))
    return "__".join(pieces)


def apply_parameters(cfg: Dict[str, Any], method: str, parameters: Dict[str, Any]) -> None:
    if method.startswith("ME-DOL"):
        cfg["me_dol"]["theory_multiplier"] = float(parameters["theory_multiplier"])
    elif method == "DGFM":
        cfg["dgfm"]["eta"] = float(parameters["eta"])
    elif method == "DGFM+":
        for key, value in parameters.items():
            cfg["dgfm_plus"][key] = value
    else:
        raise ValueError(f"Unsupported pilot method: {method}.")


def phase_one_candidates(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    pilot = cfg["pilot"]
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for method in ["ME-DOL-FO", "ME-DOL-ZO"]:
        for value in pilot["me_dol"]["theory_multiplier"]:
            candidates.append((method, {"theory_multiplier": float(value)}))
    for value in pilot["dgfm"]["eta"]:
        candidates.append(("DGFM", {"eta": float(value)}))
    defaults = cfg["dgfm_plus"]
    for value in pilot["dgfm_plus"]["eta"]:
        candidates.append(
            (
                "DGFM+",
                {
                    "eta": float(value),
                    "small_batch": int(defaults["small_batch"]),
                    "large_batch": int(defaults["large_batch"]),
                    "restart_period": int(defaults["restart_period"]),
                    "restart_mixing_rounds": int(defaults["restart_mixing_rounds"]),
                },
            )
        )
    return candidates


def phase_two_candidates(cfg: Dict[str, Any], best_eta: float) -> List[Tuple[str, Dict[str, Any]]]:
    grid = cfg["pilot"]["dgfm_plus"]
    candidates = []
    for small, large, period, mixing in itertools.product(
        grid["small_batch"],
        grid["large_batch"],
        grid["restart_period"],
        grid["restart_mixing_rounds"],
    ):
        candidates.append(
            (
                "DGFM+",
                {
                    "eta": float(best_eta),
                    "small_batch": int(small),
                    "large_batch": int(large),
                    "restart_period": int(period),
                    "restart_mixing_rounds": int(mixing),
                },
            )
        )
    return candidates


def run_candidate(
    base_cfg: Dict[str, Any],
    method: str,
    parameters: Dict[str, Any],
    seeds: List[int],
    rounds: int,
    worker_count: int,
    device: str,
) -> pd.DataFrame:
    cfg = copy.deepcopy(base_cfg)
    cfg["train"]["rounds"] = rounds
    apply_parameters(cfg, method, parameters)
    validate_experiment_config(cfg)
    results = run_selected(cfg, [method], seeds, [worker_count], device)
    results["candidate_id"] = candidate_id(method, parameters)
    results["candidate_parameters"] = json.dumps(parameters, sort_keys=True)
    return results


def execute_candidates(
    cfg: Dict[str, Any],
    candidates: Iterable[Tuple[str, Dict[str, Any]]],
    seeds: List[int],
    rounds: int,
    worker_count: int,
    device: str,
    candidate_dir: Path,
) -> pd.DataFrame:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    candidates = list(candidates)
    for index, (method, parameters) in enumerate(candidates, start=1):
        identifier = candidate_id(method, parameters)
        path = candidate_dir / f"{identifier}.csv"
        if path.exists():
            print(f"[{index}/{len(candidates)}] resume {identifier}")
            frame = pd.read_csv(path)
        else:
            print(f"[{index}/{len(candidates)}] run {identifier}")
            frame = run_candidate(
                cfg,
                method,
                parameters,
                seeds,
                rounds,
                worker_count,
                device,
            )
            frame.to_csv(path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score_candidates(results: pd.DataFrame) -> pd.DataFrame:
    """Score candidates at the minimum final work available for each method."""

    scored_frames = []
    for method, method_frame in results.groupby("method"):
        final_rows = (
            method_frame.sort_values("iteration")
            .groupby(["candidate_id", "formal_seed"], as_index=False)
            .tail(1)
        )
        candidate_final_budget = final_rows.groupby("candidate_id")["total_work"].min()
        common_budget = int(candidate_final_budget.min())

        eligible = method_frame[method_frame["total_work"] <= common_budget]
        at_budget = (
            eligible.sort_values("total_work")
            .groupby(["candidate_id", "formal_seed"], as_index=False)
            .tail(1)
        )
        scores = at_budget.groupby("candidate_id", as_index=False).agg(
            method=("method", "first"),
            candidate_parameters=("candidate_parameters", "first"),
            score_stat_proxy_mean=("stat_proxy", "mean"),
            score_stat_proxy_std=("stat_proxy", "std"),
            score_objective_mean=("objective", "mean"),
            scored_work_mean=("total_work", "mean"),
            scored_depth_mean=("communication_round", "mean"),
            num_seeds=("formal_seed", "nunique"),
        )
        scores["common_work_budget"] = common_budget
        scored_frames.append(scores)
    return pd.concat(scored_frames, ignore_index=True)


def best_by_method(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.sort_values(["method", "score_stat_proxy_mean", "scored_depth_mean"])
        .groupby("method", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def write_selected_config(base_cfg: Dict[str, Any], best: pd.DataFrame, path: Path) -> None:
    selected = copy.deepcopy(base_cfg)
    me_values: Dict[str, float] = {}
    for row in best.to_dict(orient="records"):
        method = row["method"]
        parameters = json.loads(row["candidate_parameters"])
        if method == "ME-DOL-FO":
            me_values["sfo"] = float(parameters["theory_multiplier"])
        elif method == "ME-DOL-ZO":
            me_values["szo"] = float(parameters["theory_multiplier"])
        else:
            apply_parameters(selected, method, parameters)
    if me_values:
        selected["me_dol"]["theory_multiplier"] = me_values
    selected["run"]["pilot_selection_complete"] = True
    save_yaml(selected, path)


def log_wandb(
    cfg: Dict[str, Any],
    scores: pd.DataFrame,
    best: pd.DataFrame,
    files: Iterable[Path],
) -> None:
    if not cfg.get("wandb", {}).get("enabled", False):
        return
    try:
        import wandb

        wcfg = cfg["wandb"]
        kwargs = {
            "project": wcfg["project"],
            "name": f"{cfg['run']['name']}-pilot",
            "mode": wcfg.get("mode", "online"),
            "tags": [*wcfg.get("tags", []), "pilot"],
            "config": cfg,
        }
        if wcfg.get("entity"):
            kwargs["entity"] = wcfg["entity"]
        run = wandb.init(**kwargs)
        run.log(
            {
                "candidate_scores": wandb.Table(dataframe=scores),
                "selected_candidates": wandb.Table(dataframe=best),
            }
        )
        artifact = wandb.Artifact(f"{cfg['run']['name']}-pilot", type="pilot-results")
        for path in files:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.finish()
    except Exception as exc:  # W&B must never destroy completed scientific results.
        print(f"wandb_upload_failed={exc!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, default=480)
    parser.add_argument("--phase", choices=["phase1", "phase2", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = [int(value) for value in cfg["run"]["pilot_seeds"]]
    formal = set(int(value) for value in cfg["run"]["formal_seeds"])
    if formal.intersection(seeds):
        raise ValueError("Pilot seeds and formal seeds must be disjoint.")
    worker_count = int(cfg["distributed"]["comparison_worker"])
    device = get_device(cfg["run"].get("device", "auto"))
    out_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / "pilot"
    candidate_dir = out_dir / f"rounds_{args.rounds}" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1 = phase_one_candidates(cfg)
    print(f"phase1_candidates={len(phase1)} seeds={seeds} rounds={args.rounds}")
    if args.dry_run:
        for method, parameters in phase1:
            print(candidate_id(method, parameters))
        print("phase2_candidates=16 (DGFM+ batch/restart grid after eta selection)")
        return

    frames = []
    if args.phase in {"phase1", "all"}:
        frames.append(
            execute_candidates(
                cfg,
                phase1,
                seeds,
                args.rounds,
                worker_count,
                device,
                candidate_dir / "phase1",
            )
        )
    phase1_files = list((candidate_dir / "phase1").glob("*.csv"))
    if not phase1_files:
        raise RuntimeError("Phase 1 results are required before phase 2.")
    phase1_results = pd.concat([pd.read_csv(path) for path in phase1_files], ignore_index=True)
    phase1_scores = score_candidates(phase1_results)
    dgfm_plus_best = best_by_method(phase1_scores)
    dgfm_plus_best = dgfm_plus_best[dgfm_plus_best["method"] == "DGFM+"].iloc[0]
    best_eta = float(json.loads(dgfm_plus_best["candidate_parameters"])["eta"])
    print(f"phase1_best_dgfm_plus_eta={best_eta}")

    if args.phase in {"phase2", "all"}:
        phase2 = phase_two_candidates(cfg, best_eta)
        print(f"phase2_candidates={len(phase2)}")
        frames.append(
            execute_candidates(
                cfg,
                phase2,
                seeds,
                args.rounds,
                worker_count,
                device,
                candidate_dir / "phase2",
            )
        )

    all_files = list(candidate_dir.glob("phase*/*.csv"))
    all_results = pd.concat([pd.read_csv(path) for path in all_files], ignore_index=True)
    scores = score_candidates(all_results)
    best = best_by_method(scores)
    all_results.to_csv(out_dir / "pilot_results.csv", index=False)
    scores.to_csv(out_dir / "pilot_summary.csv", index=False)
    best.to_csv(out_dir / "selected_candidates.csv", index=False)
    selected_path = out_dir / "config_selected.yaml"
    write_selected_config(cfg, best, selected_path)
    print(best[["method", "candidate_parameters", "score_stat_proxy_mean"]].to_string(index=False))

    if not args.no_wandb:
        log_wandb(
            cfg,
            scores,
            best,
            [out_dir / "pilot_summary.csv", out_dir / "selected_candidates.csv", selected_path],
        )


if __name__ == "__main__":
    main()
