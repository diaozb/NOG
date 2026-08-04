"""Resumable joint hyperparameter/batch runner for low-epsilon v6."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.low_epsilon_runner import (
    _append_tasks,
    _run,
    me_config,
    me_label,
    nog_config,
    nog_label,
)


DEFAULT_CONFIG = "configs/distributed_cpu_fo_low_epsilon_v6.yaml"
DEFAULT_ROOT = Path(
    "outputs/distributed_cpu_fo_v6/epsilon_low_extension_v6_joint_retune"
)


def _at_depth(cfg: Dict[str, Any], depth: int) -> Dict[str, Any]:
    candidate = copy.deepcopy(cfg)
    candidate["low_epsilon_extension"]["max_depth"] = int(depth)
    candidate["train"]["rounds"] = int(depth)
    return candidate


def candidate_map(cfg: Dict[str, Any], depth: int) -> Dict[str, Dict[str, Any]]:
    """Enumerate the preregistered joint algorithm-parameter/batch grid."""

    base = _at_depth(cfg, depth)
    ext = base["low_epsilon_extension"]
    result: Dict[str, Dict[str, Any]] = {}
    for batch_value in ext["batch_total_candidates"]:
        batch = int(batch_value)
        for item in ext["algorithm_candidates"]["nog"]:
            M, eta = int(item["M"]), float(item["eta"])
            label = nog_label(M, eta, batch, depth)
            result[label] = {
                "label": label,
                "method": "NOG-FO",
                "M": M,
                "eta": eta,
                "batch_total": batch,
                "config": nog_config(base, batch, M, eta),
            }
        for item in ext["algorithm_candidates"]["me_dol"]:
            epoch = int(item["epoch_length"])
            multiplier = float(item["theory_multiplier"])
            label = me_label(epoch, multiplier, batch, depth)
            result[label] = {
                "label": label,
                "method": "ME-DOL-FO",
                "epoch_length": epoch,
                "theory_multiplier": multiplier,
                "batch_total": batch,
                "config": me_config(base, epoch, multiplier, batch),
            }
    return result


def _run_descriptors(
    cfg: Dict[str, Any],
    descriptors: list[Dict[str, Any]],
    seeds: list[int],
    root: Path,
) -> Dict[str, Any]:
    workers = int(cfg["low_epsilon_extension"]["reference_worker"])
    labeled = []
    for descriptor in descriptors:
        _append_tasks(
            labeled,
            str(descriptor["label"]),
            descriptor["config"],
            str(descriptor["method"]),
            seeds,
            workers,
            root,
        )
    return _run(
        labeled,
        root,
        int(cfg["low_epsilon_extension"]["max_parallel_tasks"]),
    )


def screen(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    ext = cfg["low_epsilon_extension"]
    depth = int(ext["max_depth"])
    descriptors = list(candidate_map(cfg, depth).values())
    seeds = [int(value) for value in ext["screen_seeds"]]
    return _run_descriptors(cfg, descriptors, seeds, root)


def confirmation(
    cfg: Dict[str, Any], shortlist_path: Path, root: Path
) -> Dict[str, Any]:
    with open(shortlist_path, "r", encoding="utf-8") as handle:
        shortlist = json.load(handle)
    if shortlist.get("status") != "shortlisted":
        raise ValueError("Pilot shortlist is not frozen.")
    ext = cfg["low_epsilon_extension"]
    depth = int(ext["max_depth"])
    available = candidate_map(cfg, depth)
    labels = [str(row["label"]) for row in shortlist["candidates"]]
    if any(label not in available for label in labels):
        raise ValueError("Shortlist contains a label outside the preregistered grid.")
    descriptors = [available[label] for label in labels]
    seeds = [int(value) for value in ext["confirmation_pilot_seeds"]]
    return _run_descriptors(cfg, descriptors, seeds, root)


def formal(cfg: Dict[str, Any], freeze_path: Path, root: Path) -> Dict[str, Any]:
    with open(freeze_path, "r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "frozen":
        raise ValueError("v6 parameters are not frozen.")
    ext = cfg["low_epsilon_extension"]
    pilot_seeds = {int(value) for value in ext["pilot_seeds"]}
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if pilot_seeds & set(seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    depth = int(ext["formal_max_depth"])
    available = candidate_map(cfg, depth)
    selected_keys = set()
    descriptors = []
    for regime in freeze["selected"].values():
        for by_method in regime.values():
            for row in by_method.values():
                if row["method"] == "NOG-FO":
                    label = nog_label(
                        int(row["M"]),
                        float(row["eta"]),
                        int(row["batch_total"]),
                        depth,
                    )
                else:
                    label = me_label(
                        int(row["epoch_length"]),
                        float(row["theory_multiplier"]),
                        int(row["batch_total"]),
                        depth,
                    )
                if label not in selected_keys:
                    descriptors.append(available[label])
                    selected_keys.add(label)
    return _run_descriptors(cfg, descriptors, seeds, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["screen", "confirmation", "formal"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--shortlist", default=str(DEFAULT_ROOT / "shortlist.json"))
    parser.add_argument("--freeze", default=str(DEFAULT_ROOT / "frozen_parameters.json"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "screen":
        root = Path(args.output_root or DEFAULT_ROOT / "pilot" / "screen")
        result = screen(cfg, root)
    elif args.command == "confirmation":
        root = Path(args.output_root or DEFAULT_ROOT / "pilot" / "confirmation")
        result = confirmation(cfg, Path(args.shortlist), root)
    else:
        root = Path(args.output_root or DEFAULT_ROOT / "formal")
        result = formal(cfg, Path(args.freeze), root)
    print(
        f"status={result['status']} completed={result['completed_tasks']}/"
        f"{result['expected_tasks']} failed={result['failed_tasks']}"
    )


if __name__ == "__main__":
    main()
