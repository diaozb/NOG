"""Freeze v7 constants from the initial and boundary-extension pilot stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_correctness import atomic_write_csv, load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, object_sha256, utc_now
from src.distributed.low_epsilon_v7_freeze import (
    SCHEMA_VERSION,
    _candidate_summary,
    _measure,
    _payloads,
    _select,
)
from src.distributed.low_epsilon_v7_runner import DEFAULT_CONFIG, DEFAULT_ROOT, pilot_descriptors


DEFAULT_EXTENSION_CONFIG = "configs/distributed_cpu_fo_low_epsilon_v7_extension.yaml"
DEFAULT_EXTENSION_COMPLETION = DEFAULT_ROOT / "pilot_extension" / "completion.json"


def _stage_measurements(
    cfg: Dict[str, Any], completion_path: Path, seeds: list[int], consecutive: int
) -> list[Dict[str, Any]]:
    ext = cfg["theory_scaling"]
    stage_seeds = [int(value) for value in ext["pilot_seeds"]]
    if stage_seeds != seeds:
        raise ValueError("Initial and extension pilot seeds must be identical.")
    if int(ext["confirmed_hit_consecutive"]) != consecutive:
        raise ValueError("Initial and extension confirmed-hit rules must match.")
    descriptors = pilot_descriptors(cfg)
    payloads = _payloads(completion_path, seeds)
    if set(payloads) != set(descriptors):
        raise ValueError(f"Pilot labels do not match descriptors for {completion_path}.")
    return [
        _measure(descriptor, payloads[label], seeds, consecutive)
        for label, descriptor in sorted(descriptors.items())
    ]


def freeze(
    cfg: Dict[str, Any],
    completion_path: Path,
    extension_cfg: Dict[str, Any],
    extension_completion_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    if list((output_path.parent / "formal").glob("**/partials/*.json")):
        raise ValueError("Formal artifacts exist; refusing post-hoc v7 calibration.")
    ext = cfg["theory_scaling"]
    extension = extension_cfg["theory_scaling"]
    if extension["nog_segments"] != ext["nog_segments"]:
        raise ValueError("Initial and extension NOG segments must match.")
    if extension["me_segments"] != ext["me_segments"]:
        raise ValueError("Initial and extension ME-DOL segments must match.")
    seeds = [int(value) for value in ext["pilot_seeds"]]
    formal_seeds = [int(value) for value in cfg["run"]["formal_seeds"]]
    extension_formal_seeds = [
        int(value) for value in extension_cfg["run"]["formal_seeds"]
    ]
    if set(seeds) & (set(formal_seeds) | set(extension_formal_seeds)):
        raise ValueError("Pilot and formal seeds overlap.")
    consecutive = int(ext["confirmed_hit_consecutive"])
    initial_descriptors = pilot_descriptors(cfg)
    extension_descriptors = pilot_descriptors(extension_cfg)
    overlap = set(initial_descriptors) & set(extension_descriptors)
    if overlap:
        raise ValueError(f"Initial and extension pilot labels overlap: {overlap}")
    measurements = _stage_measurements(cfg, completion_path, seeds, consecutive)
    measurements.extend(
        _stage_measurements(
            extension_cfg, extension_completion_path, seeds, consecutive
        )
    )
    required = int(ext["calibration"]["require_hits_per_anchor"])
    if required != len(seeds):
        raise ValueError("v7 requires full pilot-seed coverage at every anchor.")
    nog_summary = _candidate_summary(
        measurements, "NOG-FO", "eta_scale", required
    )
    me_summary = _candidate_summary(
        measurements, "ME-DOL-FO", "theory_multiplier", required
    )
    nog_choice = _select(nog_summary, len(ext["nog_segments"]))
    me_choice = _select(me_summary, len(ext["me_segments"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "selection_happened_before_formal_runs": True,
        "selection_used_formal_results": False,
        "selection_rule": (
            "independently require 5/5 hits at every theory-scaling anchor "
            "across both pilot stages, then minimize geometric mean first-hit "
            "total work; depth and constant tie-break"
        ),
        "extension_reason": (
            "Stage 1 had no global constant with 5/5 hits at all six anchors; "
            "candidate boundaries were extended before any formal run."
        ),
        "pilot_seeds": seeds,
        "formal_seeds": formal_seeds,
        "pilot_max_depth": int(ext["pilot_max_depth"]),
        "formal_max_depth": int(ext["formal_max_depth"]),
        "selected_global_constants": {
            "NOG-FO": {"eta_scale": float(nog_choice["constant_value"])},
            "ME-DOL-FO": {
                "theory_multiplier": float(me_choice["constant_value"])
            },
        },
        "nog_segments": ext["nog_segments"],
        "me_segments": ext["me_segments"],
        "candidate_summaries": nog_summary + me_summary,
        "pilot_inputs": [
            {
                "stage": "initial",
                "path": str(completion_path),
                "sha256": file_sha256(completion_path),
                "config_sha256": object_sha256(cfg),
            },
            {
                "stage": "boundary_extension",
                "path": str(extension_completion_path),
                "sha256": file_sha256(extension_completion_path),
                "config_sha256": object_sha256(extension_cfg),
            },
        ],
        "pilot_completion": {
            "path": str(completion_path),
            "sha256": file_sha256(completion_path),
        },
        "config_sha256": object_sha256(cfg),
    }
    atomic_write_csv(output_path.parent / "pilot_calibration.csv", measurements)
    atomic_write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--extension-config", default=DEFAULT_EXTENSION_CONFIG)
    parser.add_argument(
        "--completion", default=str(DEFAULT_ROOT / "pilot" / "completion.json")
    )
    parser.add_argument(
        "--extension-completion", default=str(DEFAULT_EXTENSION_COMPLETION)
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_ROOT / "frozen_parameters.json")
    )
    args = parser.parse_args()
    result = freeze(
        load_config(args.config),
        Path(args.completion),
        load_config(args.extension_config),
        Path(args.extension_completion),
        Path(args.output),
    )
    print(
        "status=frozen eta_scale={} me_multiplier={}".format(
            result["selected_global_constants"]["NOG-FO"]["eta_scale"],
            result["selected_global_constants"]["ME-DOL-FO"]["theory_multiplier"],
        )
    )


if __name__ == "__main__":
    main()
