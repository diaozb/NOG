"""Reliable task orchestration for real-process FO experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import tempfile
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Literal

import torch

from src.distributed.cpu_fo_algorithms import (
    SUPPORTED_CPU_FO_METHODS,
    run_cpu_fo_task,
)
from src.distributed.cpu_process import (
    CpuProcessConfig,
    CpuProcessLaunchError,
)


TASK_SCHEMA_VERSION = 1
TaskStatus = Literal["completed", "resumed", "recovered"]


class ResumeValidationError(RuntimeError):
    """Existing artifacts do not match the requested task identity."""


@dataclass(frozen=True)
class CpuFoTask:
    method: str
    formal_seed: int
    worker_count: int

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_CPU_FO_METHODS:
            raise ValueError(f"Unsupported CPU FO method: {self.method}.")
        if self.worker_count < 1:
            raise ValueError(
                f"worker_count must be positive, got {self.worker_count}."
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "formal_seed": int(self.formal_seed),
            "worker_count": int(self.worker_count),
        }


@dataclass(frozen=True)
class TaskRunResult:
    task: CpuFoTask
    status: TaskStatus
    task_key: str
    partial_path: Path
    manifest_path: Path
    row_count: int

    @property
    def resumed(self) -> bool:
        return self.status in {"resumed", "recovered"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Publish JSON atomically inside its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def effective_task_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    effective = copy.deepcopy(cfg)
    effective.setdefault("distributed", {})["rng_mode"] = "rank_schedule"
    effective.setdefault("run", {})["device"] = "cpu"
    return effective


def _process_identity(config: CpuProcessConfig) -> Dict[str, Any]:
    identity = asdict(config)
    # Port and timeouts control launch mechanics, not the numerical task.
    identity.pop("master_port", None)
    identity.pop("process_group_timeout_seconds", None)
    identity.pop("launch_timeout_seconds", None)
    return identity


def task_identity(
    cfg: Dict[str, Any],
    task: CpuFoTask,
    process_config: CpuProcessConfig,
) -> tuple[str, str, str]:
    effective = effective_task_config(cfg)
    config_sha256 = object_sha256(effective)
    fingerprint = object_sha256(
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "task": task.as_dict(),
            "process": _process_identity(process_config),
        }
    )
    method = task.method.replace("+", "plus")
    key = (
        f"{method}__m{task.worker_count}__seed{task.formal_seed}"
        f"__cfg{fingerprint[:12]}"
    )
    return key, config_sha256, fingerprint


def _artifact_paths(root: Path, task_key: str) -> tuple[Path, Path]:
    partial = root / "partials" / f"{task_key}.json"
    manifest = root / "task_manifests" / f"{task_key}.json"
    return partial, manifest


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ResumeValidationError(f"Expected a JSON object in {path}.")
    return value


def _validate_payload(
    payload: Dict[str, Any],
    task: CpuFoTask,
    config_sha256: str,
    fingerprint: str,
) -> None:
    expected = task.as_dict()
    observed = {
        "method": payload.get("method"),
        "formal_seed": payload.get("formal_seed"),
        "worker_count": payload.get("worker_count"),
    }
    if observed != expected:
        raise ResumeValidationError(
            f"Partial task mismatch: expected={expected}, observed={observed}."
        )
    if payload.get("config_sha256") != config_sha256:
        raise ResumeValidationError("Partial config SHA256 does not match.")
    if payload.get("task_fingerprint") != fingerprint:
        raise ResumeValidationError("Partial task fingerprint does not match.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ResumeValidationError("Partial contains no checkpoint rows.")
    launch = payload.get("launch")
    if not isinstance(launch, dict):
        raise ResumeValidationError("Partial is missing launch metadata.")
    child_pids = launch.get("child_pids")
    if not isinstance(child_pids, list) or len(child_pids) != task.worker_count:
        raise ResumeValidationError("Partial child PID count does not match workers.")


def _manifest_payload(
    root: Path,
    partial: Path,
    payload: Dict[str, Any],
    task: CpuFoTask,
    task_key: str,
    config_sha256: str,
    fingerprint: str,
    recovered: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "status": "complete",
        "recovered_manifest": bool(recovered),
        "created_at_utc": utc_now(),
        "task_key": task_key,
        "task": task.as_dict(),
        "config_sha256": config_sha256,
        "task_fingerprint": fingerprint,
        "partial_path": str(partial.relative_to(root)),
        "partial_sha256": file_sha256(partial),
        "row_count": len(payload["rows"]),
        "launch": payload["launch"],
    }


def _validate_manifest(
    manifest: Dict[str, Any],
    partial: Path,
    task_key: str,
    config_sha256: str,
    fingerprint: str,
) -> None:
    if manifest.get("status") != "complete":
        raise ResumeValidationError("Task manifest is not complete.")
    if manifest.get("task_key") != task_key:
        raise ResumeValidationError("Task manifest key does not match.")
    if manifest.get("config_sha256") != config_sha256:
        raise ResumeValidationError("Task manifest config SHA256 does not match.")
    if manifest.get("task_fingerprint") != fingerprint:
        raise ResumeValidationError("Task manifest fingerprint does not match.")
    if manifest.get("partial_sha256") != file_sha256(partial):
        raise ResumeValidationError("Partial checksum does not match task manifest.")


def _resume_existing(
    root: Path,
    task: CpuFoTask,
    task_key: str,
    config_sha256: str,
    fingerprint: str,
    partial: Path,
    manifest: Path,
) -> TaskRunResult | None:
    if manifest.exists() and not partial.exists():
        raise ResumeValidationError(
            f"Manifest exists without its partial: {manifest}."
        )
    if not partial.exists():
        return None

    payload = _load_json(partial)
    _validate_payload(payload, task, config_sha256, fingerprint)
    if manifest.exists():
        manifest_payload = _load_json(manifest)
        _validate_manifest(
            manifest_payload,
            partial,
            task_key,
            config_sha256,
            fingerprint,
        )
        status: TaskStatus = "resumed"
    else:
        manifest_payload = _manifest_payload(
            root,
            partial,
            payload,
            task,
            task_key,
            config_sha256,
            fingerprint,
            recovered=True,
        )
        atomic_write_json(manifest, manifest_payload)
        status = "recovered"

    return TaskRunResult(
        task=task,
        status=status,
        task_key=task_key,
        partial_path=partial,
        manifest_path=manifest,
        row_count=len(payload["rows"]),
    )


def _failure_details(error: BaseException) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    if isinstance(error, CpuProcessLaunchError):
        details["process_cleanup"] = error.as_dict()
    return details


def run_or_resume_task(
    cfg: Dict[str, Any],
    task: CpuFoTask,
    output_root: str | Path,
    process_config: CpuProcessConfig | None = None,
) -> TaskRunResult:
    """Run one task exactly once, or validate and reuse its atomic partial."""

    root = Path(output_root)
    process = process_config or CpuProcessConfig()
    task_key, config_sha256, fingerprint = task_identity(cfg, task, process)
    partial, manifest = _artifact_paths(root, task_key)
    resumed = _resume_existing(
        root,
        task,
        task_key,
        config_sha256,
        fingerprint,
        partial,
        manifest,
    )
    if resumed is not None:
        return resumed

    attempt_id = uuid.uuid4().hex
    started_at = utc_now()
    work_path = root / "partials" / f".{task_key}.{attempt_id}.work.json"
    try:
        run_cpu_fo_task(
            effective_task_config(cfg),
            task.method,
            task.formal_seed,
            task.worker_count,
            work_path,
            process,
        )
        payload = _load_json(work_path)
        payload.update(
            {
                "schema_version": TASK_SCHEMA_VERSION,
                "config_sha256": config_sha256,
                "task_fingerprint": fingerprint,
                "task_key": task_key,
                "attempt_id": attempt_id,
                "started_at_utc": started_at,
                "finished_at_utc": utc_now(),
            }
        )
        _validate_payload(payload, task, config_sha256, fingerprint)
        atomic_write_json(partial, payload)
        manifest_payload = _manifest_payload(
            root,
            partial,
            payload,
            task,
            task_key,
            config_sha256,
            fingerprint,
            recovered=False,
        )
        atomic_write_json(manifest, manifest_payload)
        return TaskRunResult(
            task=task,
            status="completed",
            task_key=task_key,
            partial_path=partial,
            manifest_path=manifest,
            row_count=len(payload["rows"]),
        )
    except BaseException as error:
        failure = {
            "schema_version": TASK_SCHEMA_VERSION,
            "status": "failed",
            "attempt_id": attempt_id,
            "started_at_utc": started_at,
            "failed_at_utc": utc_now(),
            "parent_pid": os.getpid(),
            "task_key": task_key,
            "task": task.as_dict(),
            "config_sha256": config_sha256,
            "task_fingerprint": fingerprint,
            "temporary_output": str(work_path.relative_to(root)),
            **_failure_details(error),
        }
        failure_path = (
            root
            / "failures"
            / f"{task_key}__attempt-{attempt_id}.json"
        )
        atomic_write_json(failure_path, failure)
        raise
    finally:
        work_path.unlink(missing_ok=True)


def environment_record() -> Dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }


def run_task_set(
    cfg: Dict[str, Any],
    tasks: Iterable[CpuFoTask],
    output_root: str | Path,
    process_config: CpuProcessConfig | None = None,
    continue_on_error: bool = False,
) -> Dict[str, Any]:
    """Run/resume a task list and atomically refresh completion_manifest.json."""

    root = Path(output_root)
    task_list = list(tasks)
    records = []
    first_error: BaseException | None = None
    for task in task_list:
        try:
            result = run_or_resume_task(cfg, task, root, process_config)
            records.append(
                {
                    "task": task.as_dict(),
                    "status": result.status,
                    "task_key": result.task_key,
                    "partial_path": str(result.partial_path.relative_to(root)),
                    "manifest_path": str(result.manifest_path.relative_to(root)),
                    "row_count": result.row_count,
                }
            )
        except BaseException as error:
            records.append(
                {
                    "task": task.as_dict(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            if first_error is None:
                first_error = error
            if not continue_on_error:
                break

    completed_count = sum(
        record["status"] in {"completed", "resumed", "recovered"}
        for record in records
    )
    failed_count = sum(record["status"] == "failed" for record in records)
    completion = {
        "schema_version": TASK_SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "status": (
            "complete"
            if completed_count == len(task_list) and failed_count == 0
            else "incomplete"
        ),
        "expected_tasks": len(task_list),
        "attempted_tasks": len(records),
        "completed_tasks": completed_count,
        "failed_tasks": failed_count,
        "environment": environment_record(),
        "records": records,
    }
    atomic_write_json(root / "completion_manifest.json", completion)
    if first_error is not None and not continue_on_error:
        raise first_error
    return completion
