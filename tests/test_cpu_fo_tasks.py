import json
import tempfile
import unittest
from pathlib import Path

from src.distributed.cpu_fo_tasks import (
    CpuFoTask,
    ResumeValidationError,
    atomic_write_json,
    file_sha256,
    run_or_resume_task,
    run_task_set,
)
from src.distributed.cpu_process import (
    CpuProcessConfig,
    CpuProcessLaunchError,
)


def task_config(data_batch_total=4):
    return {
        "run": {"device": "cpu"},
        "problem": {"d": 4, "n_data": 32, "R": 2, "lam": 0.001},
        "train": {"rounds": 4, "eval_every": 2},
        "oracle": {
            "delta": 0.1,
            "smooth_B": 2,
            "data_B_total": data_batch_total,
            "eval_smooth_B": 2,
            "eval_data_B": 2,
        },
        "nog": {"M": 2, "eta": 0.1},
        "me_dol": {"epoch_length": 2, "theory_multiplier": 1.0},
        "distributed": {
            "comparison_worker": 2,
            "scaling_workers": [1, 2],
            "split_mode": "total_batch_fixed",
            "shuffle_partitions": True,
        },
        "methods": {
            "sfo": ["NOG-FO", "ME-DOL-FO"],
            "szo": [],
        },
    }


def process_config():
    return CpuProcessConfig(
        process_group_timeout_seconds=30.0,
        launch_timeout_seconds=60.0,
        intraop_threads=1,
    )


class CpuFoTaskTests(unittest.TestCase):
    def test_atomic_partial_resume_and_manifest_recovery(self):
        cfg = task_config()
        task = CpuFoTask("NOG-FO", formal_seed=7, worker_count=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_or_resume_task(cfg, task, root, process_config())
            self.assertEqual(first.status, "completed")
            initial_sha = file_sha256(first.partial_path)
            initial_mtime = first.partial_path.stat().st_mtime_ns

            second = run_or_resume_task(cfg, task, root, process_config())
            self.assertEqual(second.status, "resumed")
            self.assertEqual(file_sha256(second.partial_path), initial_sha)
            self.assertEqual(second.partial_path.stat().st_mtime_ns, initial_mtime)

            second.manifest_path.unlink()
            third = run_or_resume_task(cfg, task, root, process_config())
            self.assertEqual(third.status, "recovered")
            self.assertEqual(file_sha256(third.partial_path), initial_sha)
            self.assertEqual(third.partial_path.stat().st_mtime_ns, initial_mtime)
            with open(third.manifest_path, "r", encoding="utf-8") as handle:
                recovered_manifest = json.load(handle)
            self.assertTrue(recovered_manifest["recovered_manifest"])

            completion = run_task_set(
                cfg,
                [task],
                root,
                process_config(),
            )
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(completion["completed_tasks"], 1)
            self.assertEqual(completion["records"][0]["status"], "resumed")
            self.assertTrue((root / "completion_manifest.json").exists())

            with open(third.partial_path, "r", encoding="utf-8") as handle:
                tampered = json.load(handle)
            tampered["rows"][0]["stat_proxy"] += 1.0
            atomic_write_json(third.partial_path, tampered)
            with self.assertRaises(ResumeValidationError):
                run_or_resume_task(cfg, task, root, process_config())

    def test_rank_failure_is_audited_and_children_are_cleaned(self):
        cfg = task_config(data_batch_total=3)
        task = CpuFoTask("NOG-FO", formal_seed=9, worker_count=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(CpuProcessLaunchError) as captured:
                run_or_resume_task(cfg, task, root, process_config())

            error = captured.exception
            self.assertEqual(len(error.child_pids), 2)
            self.assertEqual(error.alive_after_cleanup, ())
            failure_paths = list((root / "failures").glob("*.json"))
            self.assertEqual(len(failure_paths), 1)
            with open(failure_paths[0], "r", encoding="utf-8") as handle:
                failure = json.load(handle)
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(
                failure["process_cleanup"]["alive_after_cleanup"],
                [],
            )
            self.assertEqual(
                sorted(failure["process_cleanup"]["child_pids"]),
                sorted(error.child_pids),
            )
            self.assertFalse(list((root / "partials").glob("*.json")))
            self.assertFalse(list((root / "task_manifests").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
