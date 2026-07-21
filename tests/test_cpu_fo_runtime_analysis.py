import unittest
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_runtime_analysis import (
    audit_and_collect,
    expand_measured_repeats,
    summarize_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "distributed_cpu_fo_runtime.yaml"
PILOT_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "pilot"
STEP7_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "step7_final"
RUNTIME_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "runtime"


class CpuFoRuntimeAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        cls.audit, cls.payloads, cls.audit_rows, cls.repeat_audits = (
            audit_and_collect(cls.cfg, PILOT_ROOT, STEP7_ROOT, RUNTIME_ROOT)
        )

    def test_all_runtime_artifacts_and_repeats_pass_audit(self):
        self.assertEqual(self.audit["status"], "passed")
        self.assertEqual(self.audit["audited_tasks"], 96)
        self.assertEqual(self.audit["passed_tasks"], 96)
        self.assertEqual(len(self.repeat_audits), 24)
        self.assertTrue(all(row["passed"] for row in self.repeat_audits))
        self.assertTrue(self.audit["failure_cleanup_passed"])

    def test_expansion_summary_and_m1_speedup_coverage(self):
        repeats = expand_measured_repeats(self.payloads)
        summaries, speedups, comparisons = summarize_runtime(repeats)
        self.assertEqual(len(repeats), 108)
        self.assertEqual(len(summaries), 36)
        self.assertEqual(len(speedups), 36)
        self.assertEqual(len(comparisons), 18)
        m1 = [row for row in speedups if row["worker_count"] == 1]
        self.assertEqual(len(m1), 6)
        self.assertTrue(
            all(abs(row["training_speedup_vs_m1"] - 1.0) < 1e-12 for row in m1)
        )


if __name__ == "__main__":
    unittest.main()
