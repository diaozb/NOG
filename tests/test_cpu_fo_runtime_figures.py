import tempfile
import unittest
from pathlib import Path

from src.distributed.cpu_fo_runtime_figures import (
    generate_figures,
    load_verified_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "runtime"


class CpuFoRuntimeFigureTests(unittest.TestCase):
    def test_verified_inputs_have_complete_runtime_coverage(self):
        repeats, summaries, speedups, comparisons, completion = (
            load_verified_inputs(RUNTIME_ROOT)
        )
        self.assertEqual(len(repeats), 108)
        self.assertEqual(len(summaries), 36)
        self.assertEqual(len(speedups), 36)
        self.assertEqual(len(comparisons), 18)
        self.assertEqual(completion["status"], "complete")
        strong_methods = set(
            speedups.loc[speedups["is_strong_scaling_workload"], "method"]
        )
        self.assertEqual(strong_methods, {"NOG-FO"})

    def test_generates_png_pdf_pairs_and_claim_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_figures(RUNTIME_ROOT, temporary)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["figure_count"], 8)
            self.assertFalse(manifest["plot_protocol"]["outlier_removal"])
            self.assertEqual(len(manifest["warnings"]), 4)
            for stem in (
                "runtime_vs_workers",
                "nog_strong_scaling_speedup",
                "communication_fraction_vs_workers",
                "full_budget_method_comparison",
            ):
                self.assertTrue((Path(temporary) / f"{stem}.png").is_file())
                self.assertTrue((Path(temporary) / f"{stem}.pdf").is_file())
            self.assertTrue(
                (Path(temporary) / "runtime_figure_notes.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
