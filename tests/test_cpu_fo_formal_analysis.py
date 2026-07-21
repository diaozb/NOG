import unittest
from pathlib import Path

from src.distributed.cpu_fo_correctness import load_config
from src.distributed.cpu_fo_formal_analysis import (
    audit_and_collect,
    threshold_results,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "distributed_cpu_fo_pilot.yaml"
PILOT_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "pilot"
FORMAL_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "formal_accuracy"


class CpuFoFormalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG)
        (
            cls.audit,
            cls.payloads,
            cls.audit_rows,
            cls.trajectories,
        ) = audit_and_collect(cls.cfg, PILOT_ROOT, FORMAL_ROOT)

    def test_all_formal_tasks_pass_integrity_and_accounting_audits(self):
        self.assertEqual(self.audit["status"], "passed")
        self.assertEqual(self.audit["audited_tasks"], 30)
        self.assertEqual(self.audit["passed_tasks"], 30)
        self.assertFalse(self.audit["global_errors"])
        self.assertTrue(all(row["passed"] for row in self.audit_rows))
        self.assertEqual(self.audit["formal_seeds"], [0, 1, 2, 3, 4])

    def test_deduplicated_trajectories_expand_to_all_threshold_pairs(self):
        per_seed, summaries, comparisons = threshold_results(
            self.payloads, consecutive=2
        )
        self.assertEqual(len(per_seed), 50)
        self.assertEqual(len(summaries), 10)
        self.assertEqual(len(comparisons), 5)
        self.assertEqual(
            {(row["method"], row["epsilon"]) for row in summaries},
            {
                (method, epsilon)
                for method in ("NOG-FO", "ME-DOL-FO")
                for epsilon in (0.011, 0.010, 0.009, 0.008, 0.0075)
            },
        )


if __name__ == "__main__":
    unittest.main()
