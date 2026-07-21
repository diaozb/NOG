import tempfile
import unittest
from pathlib import Path

from src.distributed.cpu_fo_runtime_report import build_report_package


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "formal_accuracy"
STEP7_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "step7_final"
RUNTIME_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "runtime"
RUNTIME_FIGURES = RUNTIME_ROOT / "figures"


class CpuFoRuntimeReportTests(unittest.TestCase):
    def test_builds_hash_verified_joint_package_with_claim_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = build_report_package(
                ROOT,
                FORMAL_ROOT,
                STEP7_ROOT,
                RUNTIME_ROOT,
                RUNTIME_FIGURES,
                temporary,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["step8_complete"])
            self.assertEqual(manifest["verified_runtime_figure_count"], 8)
            self.assertEqual(len(manifest["deliverables"]), 4)
            output = Path(temporary)
            report = (output / "FINAL_RUNTIME_RESULTS.md").read_text(
                encoding="utf-8"
            )
            captions = (output / "runtime_figure_captions.md").read_text(
                encoding="utf-8"
            )
            advisor = (output / "advisor_message.md").read_text(
                encoding="utf-8"
            )
            assets = (output / "asset_recommendations.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("6.14--10.71x", report)
            self.assertIn("0.539--0.821", report)
            self.assertIn("不是 first-hit time-to-epsilon", report)
            self.assertIn("| 0.008 | NOG 5/5; ME 1/5† |", report)
            self.assertIn("without outlier removal", captions)
            self.assertIn("没有观察到 positive scaling", advisor)
            self.assertIn("never crop away the work ratio", assets)


if __name__ == "__main__":
    unittest.main()
