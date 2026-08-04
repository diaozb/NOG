"""Build a compact, hash-verified package for symmetric low-epsilon v5."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict

from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def build_package(
    config_path: Path, output_root: Path, destination: Path
) -> Dict[str, Any]:
    analysis = output_root / "analysis"
    required = [
        config_path,
        output_root / "algorithm_freeze.json",
        output_root / "frozen_parameters.json",
        output_root / "algorithm_pilot_grid.csv",
        output_root / "batch_pilot_grid.csv",
        output_root / "formal" / "formal_result_audit.json",
        output_root / "formal_extra" / "formal_result_audit.json",
        analysis / "analysis_manifest.json",
        analysis / "formal_per_seed.csv",
        analysis / "formal_summary.csv",
        analysis / "formal_ratios.csv",
        analysis / "fixed_batch_diagnostics.csv",
        analysis / "formal_trends.json",
        analysis / "low_epsilon_report.md",
        analysis / "report_manifest.json",
        analysis / "figures" / "low_epsilon_ratios.png",
        analysis / "figures" / "low_epsilon_ratios.pdf",
        analysis / "figures" / "low_epsilon_hit_rates.png",
        analysis / "figures" / "low_epsilon_hit_rates.pdf",
        analysis / "figures" / "fixed_batch_diagnostics.png",
        analysis / "figures" / "fixed_batch_diagnostics.pdf",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing low-epsilon package inputs: {missing}")
    for path in [analysis / "analysis_manifest.json", analysis / "report_manifest.json"]:
        if _load(path).get("status") != "complete":
            raise ValueError(f"Incomplete manifest: {path}")
    for path in [
        output_root / "formal" / "formal_result_audit.json",
        output_root / "formal_extra" / "formal_result_audit.json",
    ]:
        if _load(path).get("status") != "passed":
            raise ValueError(f"Failed formal audit: {path}")

    destination.mkdir(parents=True, exist_ok=True)
    copies = {
        config_path: destination / "config.yaml",
        output_root / "algorithm_freeze.json": destination / "algorithm_freeze.json",
        output_root / "frozen_parameters.json": destination / "frozen_parameters.json",
        output_root / "algorithm_pilot_grid.csv": destination / "algorithm_pilot_grid.csv",
        output_root / "batch_pilot_grid.csv": destination / "batch_pilot_grid.csv",
        output_root / "formal" / "formal_result_audit.json": destination / "formal_result_audit.json",
        output_root / "formal_extra" / "formal_result_audit.json": destination / "formal_extra_result_audit.json",
    }
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target_analysis = destination / "analysis"
    if target_analysis.exists():
        shutil.rmtree(target_analysis)
    shutil.copytree(analysis, target_analysis)

    reproduce = """# Reproduce symmetric low-epsilon v5

Run from the repository root in the `NOG` Conda environment. Algorithm and
batch selection use pilot seeds 110--114. Formal seeds 20--39 and anomaly
confirmation seeds 40--49 are disjoint. All commands resume only artifacts
whose config fingerprint and SHA256 match.

```bash
conda run -n NOG python -m src.distributed.low_epsilon_runner pilot-algorithms
conda run -n NOG python -m src.distributed.low_epsilon_freeze freeze-algorithms
conda run -n NOG python -m src.distributed.low_epsilon_runner pilot-batches
conda run -n NOG python -m src.distributed.low_epsilon_freeze freeze-final
conda run -n NOG python -m src.distributed.low_epsilon_runner formal
conda run -n NOG python -m src.distributed.low_epsilon_audit
conda run -n NOG python -m src.distributed.low_epsilon_analysis
conda run -n NOG python -m src.distributed.low_epsilon_audit \\
  --formal-root outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric/formal_extra \\
  --extra
conda run -n NOG python -m src.distributed.low_epsilon_report
conda run -n NOG python -m src.distributed.low_epsilon_package
```

The analysis command automatically runs the already-frozen extra seeds when a
preregistered adjacent depth-ratio drop exceeds 20%. Raw trajectories are kept
locally because of size; the compact Git package contains audits, summaries,
figures, freeze manifests, and SHA256 manifests.
"""
    (destination / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")

    artifacts = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "package_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(destination)),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "created_at_utc": utc_now(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    atomic_write_json(destination / "package_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_low_epsilon_v5.yaml")
    parser.add_argument(
        "--output-root",
        default="outputs/distributed_cpu_fo_v5/epsilon_low_extension_v5_symmetric",
    )
    parser.add_argument("--destination", default="results/low_epsilon_v5_symmetric")
    args = parser.parse_args()
    result = build_package(Path(args.config), Path(args.output_root), Path(args.destination))
    print(f"status={result['status']} artifacts={result['artifact_count']} destination={args.destination}")


if __name__ == "__main__":
    main()
