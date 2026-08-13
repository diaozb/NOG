import copy
import unittest

import pandas as pd

from src.distributed.run_distributed_baselines import load_config
from src.distributed.zo_worker_calibration import (
    ROOT,
    evaluation_interval,
    load_manifest,
    validate_partial,
)


class WorkerCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(
            ROOT / "zo_experiments/worker_scaling_calibration_manifest.json"
        )

    def test_frozen_evaluation_intervals(self):
        expected = {
            "NOG-ZO": [4, 4, 4, 4],
            "ME-DOL-ZO": [96, 48, 24, 12],
            "DGFM": [32, 16, 8, 4],
            "DGFM+": [32, 16, 8, 4],
        }
        workers = [1, 2, 4, 8]
        for method, values in expected.items():
            self.assertEqual(
                [evaluation_interval(self.manifest, method, m) for m in workers],
                values,
            )

    def test_unknown_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_interval(self.manifest, "DGFM", 16)

    def test_validation_checks_per_worker_accounting(self):
        cfg = load_config(
            ROOT
            / "outputs/distributed_zo/zo_theory_validation/pilot"
            / "refine_work_983040_baselines_dense_eval4/config_base.yaml"
        )
        cfg["pilot"]["refine"]["eval_every"] = 32
        parameters = {"eta": 0.05}
        frame = pd.DataFrame(
            [
                {
                    "method": "DGFM",
                    "formal_seed": 301,
                    "worker_count": 1,
                    "worker_scaling_count": 1,
                    "evaluation_interval_rounds": 32,
                    "candidate_parameters": '{"eta": 0.05}',
                    "depth": 2,
                    "total_work": 32,
                    "stat_proxy": 0.1,
                    "candidate_rounds": 1,
                    "per_worker_work": "[32]",
                    "per_worker_work_max": 32,
                }
            ]
        )
        validate_partial(
            frame, cfg, "DGFM", parameters, 1, 301, 32, 32
        )
        broken = copy.deepcopy(frame)
        broken.loc[0, "per_worker_work"] = "[31]"
        with self.assertRaisesRegex(ValueError, "Per-worker sum"):
            validate_partial(
                broken, cfg, "DGFM", parameters, 1, 301, 32, 32
            )


if __name__ == "__main__":
    unittest.main()
