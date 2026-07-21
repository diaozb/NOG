import copy
import json
import unittest
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_runtime import (
    _runtime_result_record,
    runtime_task_order,
    runtime_task_config,
    unique_runtime_configs,
    validate_runtime_manifest,
    validate_runtime_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "distributed_cpu_fo_runtime.yaml"
PILOT_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "pilot"
STEP7_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "step7_final"
RUNTIME_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "runtime"


class CpuFoRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)
        self.frozen = json.loads(
            (PILOT_ROOT / "selected_config_by_epsilon.yaml").read_text()
        )
        self.pilot_report = json.loads(
            (PILOT_ROOT / "pilot_final_report.json").read_text()
        )
        self.step7 = json.loads(
            (STEP7_ROOT / "step7_completion.json").read_text()
        )

    def test_runtime_protocol_deduplicates_identical_me_dol_configs(self):
        validate_runtime_protocol(
            self.cfg, self.frozen, self.pilot_report, self.step7
        )
        configs = unique_runtime_configs(self.cfg, self.frozen)
        self.assertEqual(len(configs), 4)
        self.assertEqual(sum(row["method"] == "NOG-FO" for row in configs), 3)
        self.assertEqual(sum(row["method"] == "ME-DOL-FO" for row in configs), 1)
        me = next(row for row in configs if row["method"] == "ME-DOL-FO")
        self.assertEqual(me["epsilons"], [0.01, 0.009, 0.008])

    def test_runtime_task_order_has_warmup_and_three_unique_repeats(self):
        configs = unique_runtime_configs(self.cfg, self.frozen)
        tasks = runtime_task_order(self.cfg, configs)
        warmups = [row for row in tasks if row["phase"] == "warmup"]
        measured = [row for row in tasks if row["phase"] == "measured"]
        self.assertEqual(len(warmups), 24)
        self.assertEqual(len(measured), 72)
        self.assertEqual(len({row["runtime_task_id"] for row in tasks}), 96)
        self.assertEqual({row["repeat"] for row in measured}, {0, 1, 2})
        self.assertTrue(all(not row["include_in_summary"] for row in warmups))
        self.assertTrue(all(row["include_in_summary"] for row in measured))

    def test_runtime_protocol_tampering_is_rejected(self):
        tampered = copy.deepcopy(self.cfg)
        tampered["runtime"]["workers"].append(64)
        with self.assertRaisesRegex(ValueError, "worker protocol"):
            validate_runtime_protocol(
                tampered, self.frozen, self.pilot_report, self.step7
            )

    def test_prepared_manifest_passes_and_task_tampering_is_rejected(self):
        manifest = json.loads(
            (RUNTIME_ROOT / "runtime_task_manifest.json").read_text()
        )
        validate_runtime_manifest(
            self.cfg,
            self.frozen,
            self.pilot_report,
            self.step7,
            manifest,
        )
        tampered = copy.deepcopy(manifest)
        tampered["tasks"][0]["worker_count"] = 64
        with self.assertRaisesRegex(ValueError, "order/identity"):
            validate_runtime_manifest(
                self.cfg,
                self.frozen,
                self.pilot_report,
                self.step7,
                tampered,
            )

    def test_runtime_task_config_uses_manifest_rounds_and_frozen_parameters(self):
        manifest = json.loads(
            (RUNTIME_ROOT / "runtime_task_manifest.json").read_text()
        )
        warmup = next(
            row
            for row in manifest["tasks"]
            if row["phase"] == "warmup" and row["method"] == "NOG-FO"
        )
        measured = next(
            row
            for row in manifest["tasks"]
            if row["phase"] == "measured" and row["method"] == "ME-DOL-FO"
        )
        warmup_cfg = runtime_task_config(self.cfg, warmup)
        measured_cfg = runtime_task_config(self.cfg, measured)
        self.assertEqual(warmup_cfg["train"]["rounds"], warmup["rounds"])
        self.assertEqual(warmup_cfg["nog"]["M"], warmup["parameters"]["M"])
        self.assertEqual(measured_cfg["train"]["rounds"], 1920)
        self.assertEqual(measured_cfg["me_dol"]["theory_multiplier"], 10.0)

    def test_runtime_result_record_audits_a_completed_atomic_task(self):
        manifest = json.loads(
            (RUNTIME_ROOT / "runtime_task_manifest.json").read_text()
        )
        task = manifest["tasks"][0]
        # This assertion activates after Step 8B starts; before then the
        # executor helper is covered by the task-config test above.
        completion_path = (
            RUNTIME_ROOT / task["output_root"] / "completion_manifest.json"
        )
        if not completion_path.exists():
            self.skipTest("Step 8B has not produced its first atomic task yet.")
        completion = json.loads(completion_path.read_text())
        record = _runtime_result_record(RUNTIME_ROOT, task, completion, 1)
        self.assertEqual(record["status"], "complete")
        self.assertTrue(record["timing_invariants"])


if __name__ == "__main__":
    unittest.main()
