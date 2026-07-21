import tempfile
import unittest
from pathlib import Path

from src.distributed.cpu_fo_formal_report import build_report_package


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "formal_accuracy"
FIGURE_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "figures"


class CpuFoFormalReportTests(unittest.TestCase):
    def test_builds_hash_verified_step7_package_with_claim_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = build_report_package(
                ROOT, FORMAL_ROOT, FIGURE_ROOT, temporary
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["verified_figure_count"], 8)
            self.assertEqual(len(manifest["deliverables"]), 4)
            report = (Path(temporary) / "FINAL_RESULTS.md").read_text(
                encoding="utf-8"
            )
            table = (Path(temporary) / "paper_table.tex").read_text(
                encoding="utf-8"
            )
            self.assertIn("7.15x, 6.14x, 10.71x", report)
            self.assertIn("censoring bias", report)
            self.assertIn("不能可靠拟合", report)
            self.assertIn("](../figures/depth_vs_epsilon.pdf)", report)
            self.assertIn("| 0.008 | 5/5 | 1/5† |", report)
            self.assertNotIn("| 0.008 | 5/5†", report)
            self.assertIn(r"$^{\dagger}$", table)
            self.assertIn(r"0.008 & 5/5 & 1/5$^{\dagger}$", table)
            self.assertIn(r"& 34611 & 9312$^{\dagger}$ & --", table)
            self.assertNotIn("17.22$\\times$", table)


if __name__ == "__main__":
    unittest.main()
