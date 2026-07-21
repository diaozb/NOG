import tempfile
import unittest
from pathlib import Path

from src.distributed.cpu_fo_formal_figures import (
    generate_figures,
    load_verified_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "outputs" / "distributed_cpu_fo" / "formal_accuracy"


class CpuFoFormalFigureTests(unittest.TestCase):
    def test_verified_inputs_have_frozen_coverage(self):
        trajectories, summaries, per_seed, completion = load_verified_inputs(
            FORMAL_ROOT
        )
        self.assertEqual(len(trajectories), 1130)
        self.assertEqual(len(summaries), 10)
        self.assertEqual(len(per_seed), 50)
        self.assertEqual(completion["status"], "complete")

    def test_generates_png_pdf_pairs_and_censoring_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_figures(FORMAL_ROOT, temporary)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["figure_count"], 8)
            self.assertEqual(len(manifest["censored_method_epsilon_pairs"]), 3)
            for stem in (
                "depth_vs_epsilon",
                "work_vs_epsilon",
                "stat_proxy_vs_depth",
                "stat_proxy_vs_work",
            ):
                self.assertTrue((Path(temporary) / f"{stem}.png").is_file())
                self.assertTrue((Path(temporary) / f"{stem}.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
