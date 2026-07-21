"""CLI for the reproducible simulated distributed baseline study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.distributed.algorithms import (  # noqa: E402
    run_dgfm,
    run_dgfm_plus,
    run_me_dol,
    run_nog,
)
from src.distributed.common import (  # noqa: E402
    build_problem,
    make_seed_bundle,
    make_worker_shards,
    validate_experiment_config,
    validate_shards,
)
from src.synthetic.run_synthetic import get_device  # noqa: E402


def parse_csv_ints(value: str | None) -> List[int] | None:
    if value is None:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str | None) -> List[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def environment_record(device: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": device,
    }
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    return record


def selected_methods(cfg: Dict[str, Any], override: List[str] | None) -> List[str]:
    configured = [*cfg["methods"]["sfo"], *cfg["methods"]["szo"]]
    if override is None:
        return configured
    unknown = sorted(set(override) - set(configured))
    if unknown:
        raise ValueError(f"Methods not present in config: {unknown}.")
    return override


def method_oracle_type(method: str) -> str:
    if method.endswith("-FO"):
        return "sfo"
    if method.endswith("-ZO") or method in {"DGFM", "DGFM+"}:
        return "szo"
    raise ValueError(f"Cannot infer oracle type for {method}.")


def worker_counts_for_method(
    cfg: Dict[str, Any],
    method: str,
    override: List[int] | None,
) -> List[int]:
    if override is not None:
        return override
    if method in {"NOG-FO", "NOG-ZO"}:
        return [int(v) for v in cfg["distributed"]["scaling_workers"]]
    return [int(cfg["distributed"]["comparison_worker"])]


def run_selected(
    cfg: Dict[str, Any],
    methods: Iterable[str],
    seeds: Iterable[int],
    worker_override: List[int] | None,
    device: str,
) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    for formal_seed in seeds:
        problem_seed = 100_000 + int(formal_seed)
        problem = build_problem(cfg, device, problem_seed)
        for method in methods:
            for worker_count in worker_counts_for_method(cfg, method, worker_override):
                seed_bundle = make_seed_bundle(formal_seed, method, worker_count)
                shards = make_worker_shards(
                    n_data=problem.n,
                    n_workers=worker_count,
                    device=device,
                    partition_seed=seed_bundle.partition_seed,
                    shuffle=bool(cfg["distributed"].get("shuffle_partitions", True)),
                )
                validate_shards(shards, problem.n)
                if method in {"NOG-FO", "NOG-ZO"}:
                    rows = run_nog(
                        problem=problem,
                        cfg=cfg,
                        shards=shards,
                        seed_bundle=seed_bundle,
                        oracle_type=method_oracle_type(method),
                        method_name=method,
                    )
                elif method in {"ME-DOL-FO", "ME-DOL-ZO"}:
                    rows = run_me_dol(
                        problem=problem,
                        cfg=cfg,
                        shards=shards,
                        seed_bundle=seed_bundle,
                        oracle_type=method_oracle_type(method),
                        method_name=method,
                    )
                elif method == "DGFM":
                    rows = run_dgfm(problem, cfg, shards, seed_bundle)
                elif method == "DGFM+":
                    rows = run_dgfm_plus(problem, cfg, shards, seed_bundle)
                else:
                    raise NotImplementedError(method)
                all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["check", "pilot", "formal"], default="check")
    parser.add_argument("--methods", default=None, help="Comma-separated method names.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--workers", default=None, help="Comma-separated worker counts.")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.rounds is not None:
        cfg["train"]["rounds"] = int(args.rounds)
    if args.name is not None:
        cfg["run"]["name"] = args.name
    validate_experiment_config(cfg)
    if args.stage == "formal" and not cfg["run"].get("pilot_selection_complete", False):
        raise ValueError(
            "Formal runs require the pilot-selected config with "
            "run.pilot_selection_complete=true."
        )

    methods = selected_methods(cfg, parse_csv_strings(args.methods))
    seed_override = parse_csv_ints(args.seeds)
    if seed_override is not None:
        seeds = seed_override
    elif args.stage == "formal":
        seeds = [int(v) for v in cfg["run"]["formal_seeds"]]
    else:
        seeds = [int(v) for v in cfg["run"]["pilot_seeds"]]
    workers = parse_csv_ints(args.workers)
    device = get_device(cfg["run"].get("device", "auto"))
    cfg["run"]["device_used"] = device
    cfg["run"]["stage"] = args.stage

    print(f"config={args.config}")
    print(f"stage={args.stage} device={device}")
    print(f"methods={methods}")
    print(f"seeds={seeds} workers_override={workers}")
    if args.dry_run:
        print("dry_run=true; configuration validated, no experiment executed")
        return

    out_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config_used.yaml")
    with open(out_dir / "environment.json", "w", encoding="utf-8") as handle:
        json.dump(environment_record(device), handle, indent=2)

    partial_dir = out_dir / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in seeds:
        for method in methods:
            for worker_count in worker_counts_for_method(cfg, method, workers):
                tasks.append((int(seed), method, int(worker_count)))

    frames = []
    for index, (seed, method, worker_count) in enumerate(tasks, start=1):
        safe_method = method.replace("+", "plus")
        partial_path = partial_dir / f"{safe_method}__m{worker_count}__seed{seed}.csv"
        if partial_path.exists():
            print(
                f"[{index}/{len(tasks)}] resume method={method} "
                f"workers={worker_count} seed={seed}",
                flush=True,
            )
            frame = pd.read_csv(partial_path)
        else:
            print(
                f"[{index}/{len(tasks)}] run method={method} "
                f"workers={worker_count} seed={seed}",
                flush=True,
            )
            frame = run_selected(cfg, [method], [seed], [worker_count], device)
            frame.to_csv(partial_path, index=False)
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    results.to_csv(out_dir / "results.csv", index=False)
    print(f"saved_rows={len(results)} path={out_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
