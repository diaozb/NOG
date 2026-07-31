"""Pilot-only shortlist and Pareto freeze for low-epsilon v6."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_pilot import confirmed_hit
from src.distributed.cpu_fo_tasks import (
    atomic_write_json,
    file_sha256,
    object_sha256,
    utc_now,
)
from src.distributed.low_epsilon_v6_runner import (
    DEFAULT_CONFIG,
    DEFAULT_ROOT,
    candidate_map,
)


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _completion_payloads(
    path: Path, expected_seeds: Iterable[int]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    completion = _load(path)
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError(f"Pilot completion is not complete and failure-free: {path}")
    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for record in completion["records"]:
        label = str(record["label"])
        seed = int(record["task"]["formal_seed"])
        grouped.setdefault(label, {})[seed] = _load(Path(record["partial_path"]))
    expected = {int(seed) for seed in expected_seeds}
    if any(set(rows) != expected for rows in grouped.values()):
        raise ValueError(f"One or more candidates do not contain seeds {sorted(expected)}.")
    return grouped


def _clean_descriptor(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in descriptor.items() if key != "config"}


def _measure(
    descriptor: Dict[str, Any],
    by_seed: Dict[int, Dict[str, Any]],
    seeds: list[int],
    epsilon: float,
    consecutive: int,
) -> Dict[str, Any]:
    hits = [
        confirmed_hit(by_seed[seed]["rows"], epsilon, consecutive)
        for seed in seeds
    ]
    good = [hit for hit in hits if hit is not None]
    depths = [float(hit["depth"]) for hit in good]
    works = [float(hit["total_work"]) for hit in good]
    return {
        **_clean_descriptor(descriptor),
        "epsilon": float(epsilon),
        "hit_count": len(good),
        "mean_depth_hits": statistics.mean(depths) if depths else None,
        "mean_total_work_hits": statistics.mean(works) if works else None,
    }


def _shortlist_rows(
    rows: list[Dict[str, Any]], top_k: int, near_best_fraction: float
) -> list[Dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot shortlist an empty candidate collection.")
    full_hits = [row for row in rows if row["hit_count"] == max(r["hit_count"] for r in rows)]
    maximum_hits = int(full_hits[0]["hit_count"])
    if maximum_hits <= 0:
        raise ValueError("No candidate hit this epsilon on any screening seed.")
    selected: Dict[str, Dict[str, Any]] = {}
    for metric in ["mean_total_work_hits", "mean_depth_hits"]:
        ordered = sorted(full_hits, key=lambda row: (float(row[metric]), str(row["label"])))
        best = float(ordered[0][metric])
        for row in ordered[:top_k]:
            selected[str(row["label"])] = row
        for row in ordered:
            if float(row[metric]) <= best * (1.0 + near_best_fraction):
                selected[str(row["label"])] = row
    return sorted(selected.values(), key=lambda row: str(row["label"]))


def shortlist(
    cfg: Dict[str, Any], screen_completion: Path, output_path: Path
) -> Dict[str, Any]:
    ext = cfg["low_epsilon_extension"]
    depth = int(ext["max_depth"])
    seeds = [int(value) for value in ext["screen_seeds"]]
    epsilons = [float(value) for value in ext["epsilons"]]
    consecutive = int(ext["confirmed_hit_consecutive"])
    top_k = int(ext["selection"]["screen_top_k_per_epsilon"])
    near = float(ext["selection"]["screen_near_best_fraction"])
    descriptors = candidate_map(cfg, depth)
    payloads = _completion_payloads(screen_completion, seeds)
    if set(payloads) != set(descriptors):
        raise ValueError("Screen output labels do not equal the preregistered joint grid.")

    grid_rows: list[Dict[str, Any]] = []
    selected_labels: set[str] = set()
    by_epsilon: list[Dict[str, Any]] = []
    for epsilon in epsilons:
        record: Dict[str, Any] = {"epsilon": epsilon, "methods": {}}
        for method in ["NOG-FO", "ME-DOL-FO"]:
            rows = [
                _measure(descriptor, payloads[label], seeds, epsilon, consecutive)
                for label, descriptor in descriptors.items()
                if descriptor["method"] == method
            ]
            grid_rows.extend(rows)
            chosen = _shortlist_rows(rows, top_k, near)
            labels = [str(row["label"]) for row in chosen]
            selected_labels.update(labels)
            record["methods"][method] = {
                "maximum_screen_hits": max(int(row["hit_count"]) for row in rows),
                "labels": labels,
            }
        by_epsilon.append(record)

    candidates = [
        _clean_descriptor(descriptors[label]) for label in sorted(selected_labels)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "shortlisted",
        "created_at_utc": utc_now(),
        "selection_used_formal_results": False,
        "screen_seeds": seeds,
        "confirmation_pilot_seeds": [
            int(value) for value in ext["confirmation_pilot_seeds"]
        ],
        "screen_top_k_per_epsilon_per_objective": top_k,
        "screen_near_best_fraction": near,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "by_epsilon": by_epsilon,
        "screen_completion": {
            "path": str(screen_completion),
            "sha256": file_sha256(screen_completion),
        },
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "screen_joint_grid.csv", grid_rows)
    atomic_write_json(output_path, manifest)
    return manifest


def _combine_payloads(
    screen: Dict[str, Dict[int, Dict[str, Any]]],
    confirmation: Dict[str, Dict[int, Dict[str, Any]]],
    labels: set[str],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    combined: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for label in labels:
        if label not in screen or label not in confirmation:
            raise ValueError(f"Shortlisted label is missing pilot data: {label}")
        rows = dict(screen[label])
        if set(rows) & set(confirmation[label]):
            raise ValueError("Screen and confirmation seeds overlap.")
        rows.update(confirmation[label])
        combined[label] = rows
    return combined


def _best(rows: list[Dict[str, Any]], regime: str, required_hits: int) -> Dict[str, Any]:
    eligible = [row for row in rows if int(row["hit_count"]) == required_hits]
    if not eligible:
        raise ValueError(f"No {required_hits}/{required_hits} candidate for {regime}.")
    if regime == "work_optimal":
        key = lambda row: (
            float(row["mean_total_work_hits"]),
            float(row["mean_depth_hits"]),
            str(row["label"]),
        )
    elif regime == "depth_optimal":
        key = lambda row: (
            float(row["mean_depth_hits"]),
            float(row["mean_total_work_hits"]),
            str(row["label"]),
        )
    else:
        raise ValueError(regime)
    return min(eligible, key=key)


def freeze(
    cfg: Dict[str, Any],
    shortlist_path: Path,
    screen_completion: Path,
    confirmation_completion: Path,
    output_path: Path,
) -> Dict[str, Any]:
    formal_root = output_path.parent / "formal"
    if list(formal_root.glob("**/partials/*.json")):
        raise ValueError("Formal artifacts exist; refusing post-hoc parameter freeze.")
    ext = cfg["low_epsilon_extension"]
    shortlist_manifest = _load(shortlist_path)
    if shortlist_manifest.get("status") != "shortlisted":
        raise ValueError("Shortlist is not frozen.")
    depth = int(ext["max_depth"])
    descriptors = candidate_map(cfg, depth)
    labels = {str(row["label"]) for row in shortlist_manifest["candidates"]}
    screen_seeds = [int(value) for value in ext["screen_seeds"]]
    confirmation_seeds = [int(value) for value in ext["confirmation_pilot_seeds"]]
    seeds = screen_seeds + confirmation_seeds
    screen_payloads = _completion_payloads(screen_completion, screen_seeds)
    confirmation_payloads = _completion_payloads(
        confirmation_completion, confirmation_seeds
    )
    payloads = _combine_payloads(screen_payloads, confirmation_payloads, labels)
    epsilons = [float(value) for value in ext["epsilons"]]
    consecutive = int(ext["confirmed_hit_consecutive"])
    regimes = [str(value) for value in ext["selection"]["primary_regimes"]]

    selected: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {
        regime: {} for regime in regimes
    }
    grid_rows: list[Dict[str, Any]] = []
    for epsilon in epsilons:
        epsilon_key = f"{epsilon:.5f}"
        rows_by_method: Dict[str, list[Dict[str, Any]]] = {}
        for method in ["NOG-FO", "ME-DOL-FO"]:
            rows = [
                _measure(descriptors[label], payloads[label], seeds, epsilon, consecutive)
                for label in sorted(labels)
                if descriptors[label]["method"] == method
            ]
            rows_by_method[method] = rows
            grid_rows.extend(rows)
        for regime in regimes:
            selected[regime][epsilon_key] = {
                method: _best(rows_by_method[method], regime, len(seeds))
                for method in ["NOG-FO", "ME-DOL-FO"]
            }

    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    if set(seeds) & set(formal_seeds):
        raise ValueError("Pilot and formal seeds overlap.")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "selection_happened_before_formal_runs": True,
        "selection_used_formal_results": False,
        "pilot_seeds": seeds,
        "formal_seeds": formal_seeds,
        "pilot_max_depth": depth,
        "formal_max_depth": int(ext["formal_max_depth"]),
        "common_eval_every": int(ext["common_eval_every"]),
        "selection_rules": {
            "work_optimal": "minimum pilot mean first-hit total work; depth then label tie-break",
            "depth_optimal": "minimum pilot mean first-hit depth; work then label tie-break",
        },
        "selected": selected,
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in [shortlist_path, screen_completion, confirmation_completion]
        ],
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "confirmation_joint_grid.csv", grid_rows)
    atomic_write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["shortlist", "freeze"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--screen-completion",
        default=str(DEFAULT_ROOT / "pilot" / "screen" / "completion.json"),
    )
    parser.add_argument(
        "--confirmation-completion",
        default=str(DEFAULT_ROOT / "pilot" / "confirmation" / "completion.json"),
    )
    parser.add_argument("--shortlist", default=str(DEFAULT_ROOT / "shortlist.json"))
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "frozen_parameters.json"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "shortlist":
        result = shortlist(cfg, Path(args.screen_completion), Path(args.shortlist))
        print(f"status={result['status']} candidates={result['candidate_count']}")
    else:
        result = freeze(
            cfg,
            Path(args.shortlist),
            Path(args.screen_completion),
            Path(args.confirmation_completion),
            Path(args.output),
        )
        print(f"status={result['status']} regimes={','.join(result['selected'])}")


if __name__ == "__main__":
    main()
