import unittest
from pathlib import Path

from src.distributed.cpu_zo_equivalence import (
    DEFAULT_CONFIG,
    expected_final_accounting,
    load_config,
    simulator_rows,
    task_matrix,
)


class CpuZoEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(Path(DEFAULT_CONFIG))

    def test_frozen_tiny_matrix(self):
        self.assertEqual(len(task_matrix(self.cfg)), 6)
        self.assertEqual(
            {item[0] for item in task_matrix(self.cfg)},
            {"NOG-ZO", "ME-DOL-ZO"},
        )
        self.assertEqual({item[2] for item in task_matrix(self.cfg)}, {1, 2, 8})

    def test_expected_accounting(self):
        self.assertEqual(
            expected_final_accounting(self.cfg, "NOG-ZO", 8, 3),
            {
                "total_work": 320,
                "per_worker_work": [40] * 8,
                "per_worker_work_max": 40,
                "communication_round": 10,
                "depth": 10,
                "eval_work": 24,
            },
        )
        self.assertEqual(
            expected_final_accounting(self.cfg, "ME-DOL-ZO", 8, 3),
            {
                "total_work": 256,
                "per_worker_work": [32] * 8,
                "per_worker_work_max": 32,
                "communication_round": 8,
                "depth": 8,
                "eval_work": 24,
            },
        )

    def test_simulator_is_deterministic(self):
        first = simulator_rows(self.cfg, "NOG-ZO", 0, 2)
        second = simulator_rows(self.cfg, "NOG-ZO", 0, 2)
        for rows in [first, second]:
            for row in rows:
                row.pop("time_sec")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
