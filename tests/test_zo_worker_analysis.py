import unittest

import numpy as np
import pandas as pd

from src.distributed.zo_worker_analysis import (
    build_per_seed,
    build_relative_to_m1,
    build_trends,
)


class WorkerAnalysisTests(unittest.TestCase):
    @staticmethod
    def trajectories() -> pd.DataFrame:
        rows = []
        for worker in [1, 2, 4, 8]:
            for method, depth_factor in [("NOG-ZO", 1.0), ("DGFM", 2.0)]:
                for seed in [0, 1]:
                    for index, proxy in enumerate([0.08, 0.04, 0.02], start=1):
                        depth = index * 4 * depth_factor
                        work = index * 100 * worker
                        rows.append(
                            {
                                "worker_count": worker,
                                "method": method,
                                "formal_seed": seed,
                                "depth": depth,
                                "total_work": work,
                                "per_worker_work_max": work / worker,
                                "stat_proxy": proxy,
                            }
                        )
        return pd.DataFrame(rows)

    def test_confirmed_hit_preserves_per_worker_work(self):
        per_seed = build_per_seed(
            self.trajectories(), [0.05], 2, [0.05]
        )
        self.assertEqual(len(per_seed), 16)
        self.assertTrue(per_seed["hit"].all())
        self.assertTrue(
            np.allclose(per_seed["first_hit_per_worker_work"], 200.0)
        )

    def test_same_seed_ratios_use_m1_reference(self):
        per_seed = build_per_seed(
            self.trajectories(), [0.05], 2, [0.05]
        )
        ratios = build_relative_to_m1(per_seed, [0.05], 100, 11)
        row = ratios.loc[
            ratios["method"].eq("NOG-ZO")
            & ratios["worker_count"].eq(8)
        ].iloc[0]
        self.assertEqual(row["paired_hits"], 2)
        self.assertAlmostEqual(row["mean_depth_ratio"], 1.0)
        self.assertAlmostEqual(row["mean_work_ratio"], 8.0)
        self.assertAlmostEqual(row["mean_per_worker_work_ratio"], 1.0)

    def test_joint_worker_slope_recovers_constructed_scaling(self):
        records = []
        for worker in [1, 2, 4, 8]:
            for method in ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"]:
                for seed in range(4):
                    records.append(
                        {
                            "worker_count": worker,
                            "method": method,
                            "formal_seed": seed,
                            "epsilon": 0.05,
                            "hit": True,
                            "first_hit_depth": 10.0,
                            "first_hit_work": 100.0,
                            "first_hit_per_worker_work": 100.0 / worker,
                        }
                    )
        trends = build_trends(pd.DataFrame(records), [0.05], 100, 13)
        self.assertEqual(len(trends), 4)
        self.assertTrue(np.allclose(trends["depth_slope"], 0.0))
        self.assertTrue(np.allclose(trends["work_slope"], 0.0))
        self.assertTrue(np.allclose(trends["per_worker_work_slope"], -1.0))


if __name__ == "__main__":
    unittest.main()
