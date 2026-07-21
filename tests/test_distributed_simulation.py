import copy
import math
import random
import unittest

import numpy as np
import torch

from src.distributed.algorithms import run_dgfm, run_dgfm_plus, run_me_dol, run_nog
from src.distributed.common import (
    WorkAccounting,
    build_problem,
    complete_graph_mixing,
    isolated_torch_seed,
    make_seed_bundle,
    make_worker_shards,
    mix_worker_vectors,
    validate_experiment_config,
    validate_mixing_matrix,
    validate_shards,
)


def tiny_config():
    return {
        "run": {"device": "cpu"},
        "problem": {"d": 4, "n_data": 32, "R": 2, "lam": 0.001},
        "train": {"rounds": 4, "eval_every": 2},
        "oracle": {
            "delta": 0.1,
            "smooth_B": 2,
            "data_B_total": 4,
            "eval_smooth_B": 2,
            "eval_data_B": 2,
        },
        "nog": {"M": 2, "eta": 0.1},
        "me_dol": {"epoch_length": 2, "theory_multiplier": 1.0},
        "dgfm": {"eta": 0.001, "batch_size": 1},
        "dgfm_plus": {
            "eta": 0.001,
            "small_batch": 2,
            "large_batch": 4,
            "restart_period": 2,
            "restart_mixing_rounds": 2,
        },
        "distributed": {
            "comparison_worker": 2,
            "scaling_workers": [1, 2],
            "split_mode": "total_batch_fixed",
        },
        "methods": {
            "sfo": ["NOG-FO", "ME-DOL-FO"],
            "szo": ["NOG-ZO", "ME-DOL-ZO", "DGFM", "DGFM+"],
        },
    }


class CommonSimulationTests(unittest.TestCase):
    def test_shards_and_complete_mixing(self):
        shards = make_worker_shards(32, 4, "cpu", partition_seed=17)
        validate_shards(shards, 32)
        matrix = complete_graph_mixing(4, "cpu")
        validate_mixing_matrix(matrix)
        values = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        mixed = mix_worker_vectors(matrix, values)
        expected = values.mean(dim=0).expand_as(values)
        self.assertTrue(torch.allclose(mixed, expected))

    def test_accounting(self):
        accounting = WorkAccounting("szo", 3)
        accounting.add_training([2, 4, 6])
        accounting.add_evaluation(9)
        accounting.communicate(2)
        self.assertEqual(accounting.total_work, 12)
        self.assertEqual(accounting.per_worker_work_max, 6)
        self.assertEqual(accounting.snapshot()["depth"], 2)

    def test_evaluation_rng_isolation(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        expected_python = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1)

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        with isolated_torch_seed(999, "cpu"):
            random.random()
            np.random.rand()
            torch.rand(1)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))

    def test_all_method_work_and_depth(self):
        cfg = tiny_config()
        validate_experiment_config(cfg)
        problem = build_problem(cfg, "cpu", problem_seed=100)
        shards = make_worker_shards(32, 2, "cpu", partition_seed=200)
        validate_shards(shards, 32)

        runners = {
            "NOG-FO": lambda seed: run_nog(problem, cfg, shards, seed, "sfo", "NOG-FO"),
            "NOG-ZO": lambda seed: run_nog(problem, cfg, shards, seed, "szo", "NOG-ZO"),
            "ME-DOL-FO": lambda seed: run_me_dol(
                problem, cfg, shards, seed, "sfo", "ME-DOL-FO"
            ),
            "ME-DOL-ZO": lambda seed: run_me_dol(
                problem, cfg, shards, seed, "szo", "ME-DOL-ZO"
            ),
            "DGFM": lambda seed: run_dgfm(problem, cfg, shards, seed),
            "DGFM+": lambda seed: run_dgfm_plus(problem, cfg, shards, seed),
        }
        expected = {
            "NOG-FO": (48, 24, 6),
            "NOG-ZO": (96, 48, 6),
            "ME-DOL-FO": (8, 4, 4),
            "ME-DOL-ZO": (16, 8, 4),
            "DGFM": (16, 8, 8),
            "DGFM+": (64, 32, 10),
        }

        for method, runner in runners.items():
            with self.subTest(method=method):
                seed = make_seed_bundle(0, method, 2)
                rows = runner(seed)
                final = rows[-1]
                observed = (
                    final["total_work"],
                    final["per_worker_work_max"],
                    final["communication_round"],
                )
                self.assertEqual(observed, expected[method])
                self.assertTrue(math.isfinite(final["objective"]))
                self.assertTrue(math.isfinite(final["stat_proxy"]))

    def test_reproducible_trajectory_ignoring_time(self):
        cfg = tiny_config()
        problem = build_problem(cfg, "cpu", problem_seed=100)
        shards = make_worker_shards(32, 2, "cpu", partition_seed=200)
        seed = make_seed_bundle(0, "NOG-FO", 2)
        first = copy.deepcopy(run_nog(problem, cfg, shards, seed, "sfo", "NOG-FO"))
        second = copy.deepcopy(run_nog(problem, cfg, shards, seed, "sfo", "NOG-FO"))
        for rows in (first, second):
            for row in rows:
                row.pop("time_sec")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
