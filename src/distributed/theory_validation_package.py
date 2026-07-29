"""Build the compact, version-controlled theory-validation result package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


SCHEMA_VERSION = 1


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def build_package(source_root: Path, destination: Path, config_path: Path) -> Dict[str, Any]:
    analysis = _load(source_root / "analysis" / "analysis_manifest.json")
    completion = _load(source_root / "formal" / "completion.json")
    freeze = _load(source_root / "frozen_parameters.json")
    result_audit = _load(source_root / "formal" / "formal_result_audit.json")
    if analysis.get("status") != "complete":
        raise ValueError("Analysis is not complete.")
    if completion.get("status") != "complete" or completion.get("failed_tasks"):
        raise ValueError("Formal tasks are not complete and failure-free.")
    if freeze.get("status") != "frozen":
        raise ValueError("Parameters are not frozen.")
    if result_audit.get("status") != "passed":
        raise ValueError("Formal result audit has not passed.")

    copies = {
        source_root / "frozen_parameters.json": destination / "frozen_parameters.json",
        source_root / "pilot_calibration.csv": destination / "pilot_calibration.csv",
        source_root / "formal" / "completion.json": destination / "audit" / "formal_completion.json",
        source_root / "formal" / "formal_result_audit.json": destination / "audit" / "formal_result_audit.json",
        source_root / "analysis" / "formal_per_seed.csv": destination / "analysis" / "formal_per_seed.csv",
        source_root / "analysis" / "formal_summary.csv": destination / "analysis" / "formal_summary.csv",
        source_root / "analysis" / "formal_ratios.csv": destination / "analysis" / "formal_ratios.csv",
        source_root / "analysis" / "formal_trends.json": destination / "analysis" / "formal_trends.json",
        source_root / "analysis" / "analysis_manifest.json": destination / "analysis" / "analysis_manifest.json",
        source_root / "analysis" / "theory_validation_report.md": destination / "analysis" / "theory_validation_report.md",
        source_root / "analysis" / "report_manifest.json": destination / "analysis" / "report_manifest.json",
        source_root / "analysis" / "figures" / "hit_rate_vs_epsilon.png": destination / "analysis" / "figures" / "hit_rate_vs_epsilon.png",
        source_root / "analysis" / "figures" / "hit_rate_vs_epsilon.pdf": destination / "analysis" / "figures" / "hit_rate_vs_epsilon.pdf",
        source_root / "analysis" / "figures" / "depth_work_ratios.png": destination / "analysis" / "figures" / "depth_work_ratios.png",
        source_root / "analysis" / "figures" / "depth_work_ratios.pdf": destination / "analysis" / "figures" / "depth_work_ratios.pdf",
        source_root / "analysis" / "figures" / "depth_work_vs_epsilon.png": destination / "analysis" / "figures" / "depth_work_vs_epsilon.png",
        source_root / "analysis" / "figures" / "depth_work_vs_epsilon.pdf": destination / "analysis" / "figures" / "depth_work_vs_epsilon.pdf",
        config_path: destination / "config.yaml",
    }
    for source, target in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    reproduce = """# Reproduce the wide-epsilon theory validation

Run from the repository root with the `NOG` Conda environment.  Pilot seeds
100--104 select a monotone matched-work batch schedule; formal seeds 0--19 are
never used for selection.  The scheduler admits at most four eight-worker
tasks (32 worker processes) at once, and every command resumes valid partials.

```bash
conda run -n NOG python -m src.distributed.theory_validation_runner pilot-batch-grid
conda run -n NOG python -m src.distributed.theory_validation_freeze
conda run -n NOG python -m src.distributed.theory_validation_runner formal
conda run -n NOG python -m src.distributed.theory_validation_audit
conda run -n NOG python -m src.distributed.theory_validation_analysis
conda run -n NOG python -m src.distributed.theory_validation_report
conda run -n NOG python -m src.distributed.theory_validation_package
```

Raw process trajectories are intentionally excluded from Git because they are
large.  `frozen_parameters.json` and `analysis/analysis_manifest.json` retain
SHA256 hashes for every pilot/formal raw input used to make the package.
"""
    (destination / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    packaged_files = sorted(
        path for path in destination.rglob("*") if path.is_file() and path.name != "package_manifest.json"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": utc_now(),
        "formal_tasks": completion["completed_tasks"],
        "formal_failures": completion["failed_tasks"],
        "verdict": analysis["verdict"],
        "files": [
            {
                "path": str(path.relative_to(destination)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in packaged_files
        ],
    }
    atomic_write_json(destination / "package_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="outputs/distributed_cpu_fo_v4/epsilon_theory_validation_v4",
    )
    parser.add_argument("--destination", default="results/theory_validation_v4")
    parser.add_argument(
        "--config", default="configs/distributed_cpu_fo_theory_validation_v4.yaml"
    )
    args = parser.parse_args()
    result = build_package(Path(args.source_root), Path(args.destination), Path(args.config))
    print(
        f"status={result['status']} files={len(result['files'])} "
        f"formal_tasks={result['formal_tasks']}"
    )


if __name__ == "__main__":
    main()
