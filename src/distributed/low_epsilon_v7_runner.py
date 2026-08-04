"""Run theory-scaled low-epsilon NOG/ME-DOL pilot and formal tasks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import CpuFoTask, atomic_write_json
from src.distributed.theory_validation_runner import run_labeled_tasks


DEFAULT_CONFIG = "configs/distributed_cpu_fo_low_epsilon_v7.yaml"
DEFAULT_ROOT = Path(
    "outputs/distributed_cpu_fo_v7/epsilon_low_extension_v7_theory_scaling"
)


def _number(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _at_depth(base: Dict[str, Any], depth: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["train"]["rounds"] = int(depth)
    cfg["train"]["strict_eval_grid"] = False
    return cfg


def _validate_depth(depth: int, period: int, method: str) -> None:
    if depth < 1 or period < 1 or depth % period:
        raise ValueError(f"{method} depth {depth} must be divisible by period {period}.")


def nog_config(
    base: Dict[str, Any], segment: Dict[str, Any], eta_scale: float, depth: int
) -> Dict[str, Any]:
    cfg = _at_depth(base, depth)
    workers = int(base["theory_scaling"]["reference_worker"])
    M = int(segment["M"])
    batch = int(segment["batch_total"])
    _validate_depth(depth, M, "NOG")
    if batch < workers or batch % workers:
        raise ValueError("NOG total batch must be a positive multiple of workers.")
    cfg["train"]["eval_every"] = M
    cfg["oracle"].update({"smooth_B": 1, "data_B_total": batch})
    cfg["nog"].update(
        {"M": M, "eta": float(segment["eta"]) * float(eta_scale)}
    )
    cfg["methods"]["sfo"] = ["NOG-FO"]
    return cfg


def me_config(
    base: Dict[str, Any], segment: Dict[str, Any], multiplier: float, depth: int
) -> Dict[str, Any]:
    cfg = _at_depth(base, depth)
    workers = int(base["theory_scaling"]["reference_worker"])
    epoch = int(segment["epoch_length"])
    batch = int(segment["batch_total"])
    _validate_depth(depth, epoch, "ME-DOL")
    if batch < workers or batch % workers:
        raise ValueError("ME-DOL total batch must be a positive multiple of workers.")
    cfg["train"]["eval_every"] = epoch
    cfg["me_dol"].update(
        {
            "epoch_length": epoch,
            "theory_multiplier": float(multiplier),
            "smooth_B": 1,
            "data_B_per_worker": batch // workers,
        }
    )
    cfg["methods"]["sfo"] = ["ME-DOL-FO"]
    return cfg


def nog_label(segment: Dict[str, Any], eta_scale: float, depth: int) -> str:
    eta = float(segment["eta"]) * float(eta_scale)
    return (
        f"NOG-FO__segment-{segment['id']}__M-{int(segment['M'])}"
        f"__eta-{_number(eta)}__batch-total-{int(segment['batch_total'])}"
        f"__rounds-{int(depth)}"
    )


def me_label(segment: Dict[str, Any], multiplier: float, depth: int) -> str:
    return (
        f"ME-DOL-FO__segment-{segment['id']}"
        f"__epoch-{int(segment['epoch_length'])}"
        f"__mult-{_number(multiplier)}"
        f"__batch-total-{int(segment['batch_total'])}__rounds-{int(depth)}"
    )


def _append(
    labeled: list,
    label: str,
    cfg: Dict[str, Any],
    method: str,
    seeds: Iterable[int],
    workers: int,
    root: Path,
) -> None:
    candidate_root = root / label
    atomic_write_json(candidate_root / "config_used.json", cfg)
    for seed in seeds:
        labeled.append(
            (label, cfg, CpuFoTask(method, int(seed), workers), candidate_root)
        )


def pilot_descriptors(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ext = cfg["theory_scaling"]
    depth = int(ext["pilot_max_depth"])
    result: Dict[str, Dict[str, Any]] = {}
    for scale in ext["nog_eta_scale_candidates"]:
        for segment in ext["nog_segments"]:
            label = nog_label(segment, float(scale), depth)
            result[label] = {
                "label": label,
                "method": "NOG-FO",
                "segment": dict(segment),
                "eta_scale": float(scale),
                "config": nog_config(cfg, segment, float(scale), depth),
            }
    for multiplier in ext["me_multiplier_candidates"]:
        for segment in ext["me_segments"]:
            label = me_label(segment, float(multiplier), depth)
            result[label] = {
                "label": label,
                "method": "ME-DOL-FO",
                "segment": dict(segment),
                "theory_multiplier": float(multiplier),
                "config": me_config(cfg, segment, float(multiplier), depth),
            }
    return result


def _run_descriptors(
    cfg: Dict[str, Any],
    descriptors: Iterable[Dict[str, Any]],
    seeds: list[int],
    root: Path,
) -> Dict[str, Any]:
    ext = cfg["theory_scaling"]
    workers = int(ext["reference_worker"])
    labeled: list = []
    for row in descriptors:
        _append(
            labeled,
            str(row["label"]),
            row["config"],
            str(row["method"]),
            seeds,
            workers,
            root,
        )
    return run_labeled_tasks(
        labeled,
        root / "completion.json",
        max_parallel_tasks=int(ext["max_parallel_tasks"]),
    )


def pilot(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    seeds = [int(value) for value in cfg["theory_scaling"]["pilot_seeds"]]
    return _run_descriptors(cfg, pilot_descriptors(cfg).values(), seeds, root)


def formal_descriptors(
    cfg: Dict[str, Any], freeze: Dict[str, Any]
) -> list[Dict[str, Any]]:
    ext = cfg["theory_scaling"]
    depth = int(ext["formal_max_depth"])
    eta_scale = float(freeze["selected_global_constants"]["NOG-FO"]["eta_scale"])
    multiplier = float(
        freeze["selected_global_constants"]["ME-DOL-FO"]["theory_multiplier"]
    )
    result = []
    for segment in ext["nog_segments"]:
        result.append(
            {
                "label": nog_label(segment, eta_scale, depth),
                "method": "NOG-FO",
                "config": nog_config(cfg, segment, eta_scale, depth),
            }
        )
    for segment in ext["me_segments"]:
        result.append(
            {
                "label": me_label(segment, multiplier, depth),
                "method": "ME-DOL-FO",
                "config": me_config(cfg, segment, multiplier, depth),
            }
        )
    return result


def formal(
    cfg: Dict[str, Any], freeze_path: Path, root: Path
) -> Dict[str, Any]:
    with open(freeze_path, "r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "frozen":
        raise ValueError("v7 global constants are not frozen.")
    pilot_seeds = {int(value) for value in cfg["theory_scaling"]["pilot_seeds"]}
    seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if pilot_seeds & set(seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    return _run_descriptors(cfg, formal_descriptors(cfg, freeze), seeds, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot", "formal"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--freeze", default=str(DEFAULT_ROOT / "frozen_parameters.json"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "pilot":
        root = Path(args.output_root or DEFAULT_ROOT / "pilot")
        result = pilot(cfg, root)
    else:
        root = Path(args.output_root or DEFAULT_ROOT / "formal")
        result = formal(cfg, Path(args.freeze), root)
    print(
        f"status={result['status']} completed={result['completed_tasks']}/"
        f"{result['expected_tasks']} failed={result['failed_tasks']}"
    )


if __name__ == "__main__":
    main()
