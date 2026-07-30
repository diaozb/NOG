"""Resumable runner for the symmetric preregistered low-epsilon experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import CpuFoTask, atomic_write_json
from src.distributed.theory_validation_runner import run_labeled_tasks


DEFAULT_ROOT = Path(
    "outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric"
)


def _extension(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg["low_epsilon_extension"]


def _base_candidate(base: Dict[str, Any]) -> Dict[str, Any]:
    ext = _extension(base)
    cfg = copy.deepcopy(base)
    cfg["train"].update(
        {
            "rounds": int(ext["max_depth"]),
            "eval_every": int(ext["common_eval_every"]),
            "strict_eval_grid": True,
        }
    )
    return cfg


def nog_config(
    base: Dict[str, Any], data_batch_total: int, M: int = 2, eta: float = 1.0
) -> Dict[str, Any]:
    cfg = _base_candidate(base)
    workers = int(_extension(base)["reference_worker"])
    if data_batch_total < workers or data_batch_total % workers:
        raise ValueError("NOG total batch must be a positive multiple of workers.")
    if int(_extension(base)["common_eval_every"]) % int(M):
        raise ValueError("NOG M must divide the common evaluation interval.")
    cfg["oracle"].update({"smooth_B": 1, "data_B_total": int(data_batch_total)})
    cfg["nog"].update({"M": int(M), "eta": float(eta)})
    cfg["methods"]["sfo"] = ["NOG-FO"]
    return cfg


def me_config(
    base: Dict[str, Any],
    epoch_length: int,
    theory_multiplier: float,
    data_batch_total: int,
) -> Dict[str, Any]:
    cfg = _base_candidate(base)
    workers = int(_extension(base)["reference_worker"])
    if data_batch_total < workers or data_batch_total % workers:
        raise ValueError("ME-DOL total batch must be a positive multiple of workers.")
    if int(_extension(base)["common_eval_every"]) % int(epoch_length):
        raise ValueError("ME-DOL epoch length must divide the common evaluation interval.")
    if int(_extension(base)["max_depth"]) % int(epoch_length):
        raise ValueError("ME-DOL max depth must be divisible by epoch length.")
    cfg["me_dol"].update(
        {
            "epoch_length": int(epoch_length),
            "theory_multiplier": float(theory_multiplier),
            "smooth_B": 1,
            "data_B_per_worker": int(data_batch_total) // workers,
        }
    )
    cfg["methods"]["sfo"] = ["ME-DOL-FO"]
    return cfg


def _number(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def nog_label(M: int, eta: float, data_batch_total: int, rounds: int) -> str:
    return (
        f"NOG-FO__M-{int(M)}__eta-{_number(eta)}"
        f"__batch-total-{int(data_batch_total)}__rounds-{int(rounds)}"
    )


def me_label(
    epoch_length: int,
    theory_multiplier: float,
    data_batch_total: int,
    rounds: int,
) -> str:
    return (
        f"ME-DOL-FO__epoch-{int(epoch_length)}"
        f"__mult-{_number(theory_multiplier)}"
        f"__batch-total-{int(data_batch_total)}__rounds-{int(rounds)}"
    )


def _run(
    labeled: Iterable[tuple[str, Dict[str, Any], CpuFoTask, Path]],
    root: Path,
    max_parallel_tasks: int,
) -> Dict[str, Any]:
    return run_labeled_tasks(
        labeled,
        root / "completion.json",
        max_parallel_tasks=max_parallel_tasks,
    )


def _append_tasks(
    labeled: list,
    label: str,
    cfg: Dict[str, Any],
    method: str,
    seeds: list[int],
    workers: int,
    root: Path,
) -> None:
    candidate_root = root / label
    atomic_write_json(candidate_root / "config_used.json", cfg)
    for seed in seeds:
        labeled.append(
            (label, cfg, CpuFoTask(method, seed, workers), candidate_root)
        )


def pilot_algorithms(cfg: Dict[str, Any], root: Path) -> Dict[str, Any]:
    ext = _extension(cfg)
    seeds = [int(value) for value in ext["pilot_seeds"]]
    workers = int(ext["reference_worker"])
    rounds = int(ext["max_depth"])
    batch = int(ext["common_algorithm_selection_batch_total"])
    labeled: list = []
    for item in ext["algorithm_candidates"]["nog"]:
        M, eta = int(item["M"]), float(item["eta"])
        candidate = nog_config(cfg, batch, M, eta)
        _append_tasks(
            labeled,
            nog_label(M, eta, batch, rounds),
            candidate,
            "NOG-FO",
            seeds,
            workers,
            root,
        )
    for item in ext["algorithm_candidates"]["me_dol"]:
        epoch = int(item["epoch_length"])
        multiplier = float(item["theory_multiplier"])
        candidate = me_config(cfg, epoch, multiplier, batch)
        _append_tasks(
            labeled,
            me_label(epoch, multiplier, batch, rounds),
            candidate,
            "ME-DOL-FO",
            seeds,
            workers,
            root,
        )
    return _run(labeled, root, int(ext["max_parallel_tasks"]))


def _load_freeze(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "frozen":
        raise ValueError(f"Parameters are not frozen: {path}")
    return freeze


def pilot_batches(
    cfg: Dict[str, Any], algorithm_freeze_path: Path, root: Path
) -> Dict[str, Any]:
    freeze = _load_freeze(algorithm_freeze_path)
    ext = _extension(cfg)
    seeds = [int(value) for value in ext["pilot_seeds"]]
    workers = int(ext["reference_worker"])
    rounds = int(ext["max_depth"])
    selected_nog = freeze["selected_algorithms"]["NOG-FO"]
    selected_me = freeze["selected_algorithms"]["ME-DOL-FO"]
    labeled: list = []
    for batch in [int(value) for value in ext["batch_total_candidates"]]:
        nog = nog_config(
            cfg, batch, int(selected_nog["M"]), float(selected_nog["eta"])
        )
        _append_tasks(
            labeled,
            nog_label(selected_nog["M"], selected_nog["eta"], batch, rounds),
            nog,
            "NOG-FO",
            seeds,
            workers,
            root,
        )
        me = me_config(
            cfg,
            int(selected_me["epoch_length"]),
            float(selected_me["theory_multiplier"]),
            batch,
        )
        _append_tasks(
            labeled,
            me_label(
                selected_me["epoch_length"],
                selected_me["theory_multiplier"],
                batch,
                rounds,
            ),
            me,
            "ME-DOL-FO",
            seeds,
            workers,
            root,
        )
    return _run(labeled, root, int(ext["max_parallel_tasks"]))


def formal(
    cfg: Dict[str, Any], freeze_path: Path, root: Path, extra: bool = False
) -> Dict[str, Any]:
    freeze = _load_freeze(freeze_path)
    ext = _extension(cfg)
    seeds = (
        [int(value) for value in ext["anomaly_confirmation"]["extra_formal_seeds"]]
        if extra
        else [int(value) for value in cfg["run"]["formal_seeds"]]
    )
    if set(seeds) & {int(value) for value in ext["pilot_seeds"]}:
        raise ValueError("Pilot and formal seeds must be disjoint.")
    workers = int(ext["reference_worker"])
    rounds = int(ext["max_depth"])
    selected_nog = freeze["selected_algorithms"]["NOG-FO"]
    selected_me = freeze["selected_algorithms"]["ME-DOL-FO"]
    labeled: list = []
    for batch in freeze["selected_batches"]["NOG-FO"]:
        candidate = nog_config(
            cfg, int(batch), int(selected_nog["M"]), float(selected_nog["eta"])
        )
        _append_tasks(
            labeled,
            nog_label(selected_nog["M"], selected_nog["eta"], batch, rounds),
            candidate,
            "NOG-FO",
            seeds,
            workers,
            root,
        )
    for batch in freeze["selected_batches"]["ME-DOL-FO"]:
        candidate = me_config(
            cfg,
            int(selected_me["epoch_length"]),
            float(selected_me["theory_multiplier"]),
            int(batch),
        )
        _append_tasks(
            labeled,
            me_label(
                selected_me["epoch_length"],
                selected_me["theory_multiplier"],
                batch,
                rounds,
            ),
            candidate,
            "ME-DOL-FO",
            seeds,
            workers,
            root,
        )
    return _run(labeled, root, int(ext["max_parallel_tasks"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["pilot-algorithms", "pilot-batches", "formal", "formal-extra"],
    )
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_low_epsilon_v5.yaml"
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--algorithm-freeze", default=str(DEFAULT_ROOT / "algorithm_freeze.json")
    )
    parser.add_argument(
        "--freeze", default=str(DEFAULT_ROOT / "frozen_parameters.json")
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "pilot-algorithms":
        root = Path(args.output_root or DEFAULT_ROOT / "pilot" / "algorithm_grid")
        result = pilot_algorithms(cfg, root)
    elif args.command == "pilot-batches":
        root = Path(args.output_root or DEFAULT_ROOT / "pilot" / "batch_grid")
        result = pilot_batches(cfg, Path(args.algorithm_freeze), root)
    elif args.command == "formal":
        root = Path(args.output_root or DEFAULT_ROOT / "formal")
        result = formal(cfg, Path(args.freeze), root)
    else:
        root = Path(args.output_root or DEFAULT_ROOT / "formal_extra")
        result = formal(cfg, Path(args.freeze), root, extra=True)
    print(
        f"status={result['status']} completed={result['completed_tasks']}/"
        f"{result['expected_tasks']} failed={result['failed_tasks']}"
    )


if __name__ == "__main__":
    main()
