import unittest

import numpy as np
import pandas as pd

from src.distributed.zo_dimension_analysis import (
    build_dimension_trends,
    build_per_seed,
    build_ratios,
)


class DimensionAnalysisTests(unittest.TestCase):
    @staticmethod
    def trajectories() -> pd.DataFrame:
        rows = []
        for dimension in [25, 50]:
            for method, multiplier in [("NOG-ZO", 1), ("ME-DOL-ZO", 2)]:
                for seed in [0, 1]:
                    for depth, proxy in [(4, 0.08), (8, 0.04), (12, 0.02)]:
                        rows.append(
                            {
                                "dimension": dimension,
                                "method": method,
                                "formal_seed": seed,
                                "depth": depth * multiplier,
                                "total_work": depth * multiplier * 10,
                                "stat_proxy": proxy,
                            }
                        )
        return pd.DataFrame(rows)

    def test_dimension_and_confirmed_hit_are_preserved(self):
        per_seed = build_per_seed(
            self.trajectories(), [0.05], 2, [0.05]
        )
        self.assertEqual(len(per_seed), 8)
        self.assertTrue(per_seed["hit"].all())
        self.assertEqual(set(per_seed["scope"]), {"primary"})
        nog = per_seed.loc[per_seed["method"] == "NOG-ZO"]
        self.assertTrue(np.allclose(nog["first_hit_depth"], 8.0))

    def test_ratio_uses_same_seed_pairs(self):
        per_seed = build_per_seed(
            self.trajectories(), [0.05], 2, [0.05]
        )
        ratios = build_ratios(per_seed, [0.05], 100, 7)
        me = ratios.loc[ratios["baseline"] == "ME-DOL-ZO"]
        self.assertTrue((me["paired_hits"] == 2).all())
        self.assertTrue(me["complete_pairing"].all())
        self.assertTrue(np.allclose(me["mean_depth_ratio"], 2.0))
        self.assertTrue(np.allclose(me["mean_work_ratio"], 2.0))

    def test_joint_dimension_slope_recovers_constructed_power(self):
        records = []
        for dimension in [25, 50, 100, 200]:
            for seed in range(4):
                records.extend(
                    [
                        {
                            "dimension": dimension,
                            "method": "NOG-ZO",
                            "formal_seed": seed,
                            "epsilon": 0.05,
                            "hit": True,
                            "first_hit_depth": 10.0,
                            "first_hit_work": 10.0,
                        },
                        {
                            "dimension": dimension,
                            "method": "ME-DOL-ZO",
                            "formal_seed": seed,
                            "epsilon": 0.05,
                            "hit": True,
                            "first_hit_depth": 10.0 * dimension**0.5,
                            "first_hit_work": 20.0,
                        },
                    ]
                )
        trends = build_dimension_trends(
            pd.DataFrame(records),
            [0.05],
            {"ME-DOL-ZO/NOG-ZO": {"depth": 0.5, "work": 0.0}},
            100,
            11,
        )
        row = trends.iloc[0]
        self.assertAlmostEqual(row["observed_depth_ratio_slope"], 0.5)
        self.assertAlmostEqual(row["observed_work_ratio_slope"], 0.0)
        self.assertTrue(row["exact_depth_power_inside_ci"])
        self.assertTrue(row["exact_work_power_inside_ci"])


if __name__ == "__main__":
    unittest.main()
