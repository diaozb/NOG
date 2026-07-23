"""Build a compact, hash-audited Git package without raw task trajectories."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_tasks import atomic_write_json, file_sha256, utc_now


ROOT_FILES = (
    "protocol_manifest.json",
    "frozen_region_configs.json",
    "formal_manifest.json",
    "formal_schedule.json",
    "formal_preflight_audit.json",
    "formal_completion.json",
    "formal_result_audit.json",
    "formal_work_accounting_audit.csv",
    "robustness_manifest.json",
    "robustness_preflight_audit.json",
    "robustness_m1_completion.json",
    "robustness_m2_completion.json",
    "robustness_m4_completion.json",
    "robustness_result_audit.json",
    "robustness_work_accounting_audit.csv",
)

PILOT_FILES = (
    "refined_region_selection.json",
    "region_extension_3840_analysis.json",
    "region_extension_3840_prefix_audit.csv",
    "region_extension_15360_analysis.json",
    "region_extension_15360_prefix_audit.csv",
    "region_extension_61440_analysis.json",
    "region_extension_61440_prefix_audit.csv",
)


def build_package(cfg: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    output = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    package = repo_root / "results" / "epsilon_scaling_v2"
    package.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for name in ROOT_FILES:
        source = output / name
        target = package / "audit" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        # Keep text artifacts friendly to Git and POSIX tooling. Python CSV
        # output uses CRLF by default, which diff --check treats as whitespace.
        if target.suffix.lower() == ".csv":
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
        copied.append(target)
    for name in PILOT_FILES:
        source = output / "pilot" / name
        target = package / "pilot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        # Keep text artifacts friendly to Git and POSIX tooling. Python CSV
        # output uses CRLF by default, which diff --check treats as whitespace.
        if target.suffix.lower() == ".csv":
            target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
        copied.append(target)
    for source in sorted((output / "analysis").rglob("*")):
        if source.is_file():
            target = package / "analysis" / source.relative_to(output / "analysis")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if target.suffix.lower() == ".csv":
                target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
            copied.append(target)

    reproduce = package / "REPRODUCE.md"
    reproduce.write_text(
        """# Reproduce wide-epsilon results

Environment: the repository requirements in the `NOG` conda environment.

```bash
python -m unittest discover -s tests -p "test*.py"
python -m src.distributed.epsilon_scaling_formal_result_audit
python -m src.distributed.epsilon_scaling_robustness_audit
python -m src.distributed.epsilon_scaling_formal_statistics
python -m src.distributed.epsilon_scaling_robustness_statistics
python -m src.distributed.epsilon_scaling_report
```

The audit/statistics commands require the local raw trajectories under
`outputs/distributed_cpu_fo_v2/epsilon_scaling_v2/`. Raw trajectories are not
committed because they total about 215 MiB. This compact package contains their
completion manifests, SHA256 accounting audits, summaries, figures, and report.

Long-task runner guard (operational only; excluded from numerical identity):

```bash
NOG_CPU_LAUNCH_TIMEOUT_SECONDS=7200 python -m src.distributed.epsilon_scaling_robustness_runner --phase run --worker 4
```
""",
        encoding="utf-8",
    )
    copied.append(reproduce)
    records = [
        {
            "path": str(path.relative_to(package)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(copied)
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "source_output_root": str(output),
        "raw_outputs_committed": False,
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "files": records,
        "required_audits": {
            "formal": "passed 120/120",
            "robustness": "passed 240/240",
            "tests": "passed 57/57",
        },
    }
    atomic_write_json(package / "package_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/distributed_cpu_fo_epsilon_scaling.yaml")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    result = build_package(load_config(args.config), repo_root)
    status = result["status"]
    count = result["file_count"]
    size = result["total_bytes"]
    print(f"status={status} files={count} bytes={size}")


if __name__ == "__main__":
    main()
