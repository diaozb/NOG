import json
import math
import tempfile
import unittest
from pathlib import Path

from src.distributed.algorithms import run_me_dol, run_nog
from src.distributed.common import (
    build_problem,
    make_seed_bundle,
    make_worker_shards,
)
from src.distributed.cpu_fo_algorithms import run_cpu_fo_task
from src.distributed.cpu_process import CpuProcessConfig


def equivalence_config():
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
        "distributed": {
            "comparison_worker": 2,
            "scaling_workers": [1, 2],
            "split_mode": "total_batch_fixed",
            "shuffle_partitions": True,
            "rng_mode": "rank_schedule",
        },
        "methods": {
            "sfo": ["NOG-FO", "ME-DOL-FO"],
            "szo": [],
        },
    }


class CpuFoEquivalenceTests(unittest.TestCase):
    def test_problem_feature_scale_is_configurable(self):
        cfg = equivalence_config()
        baseline = build_problem(cfg, "cpu", 123)
        cfg["problem"]["feature_scale"] = 4.0
        scaled = build_problem(cfg, "cpu", 123)
        self.assertTrue((scaled.A == 4.0 * baseline.A).all())

    def test_problem_phase_and_common_bias_are_configurable(self):
        cfg = equivalence_config()
        cfg["problem"].update({"R": 1, "phase_mode": "zero", "common_feature_bias": 0.25})
        problem = build_problem(cfg, "cpu", 123)
        self.assertTrue((problem.b == 0.0).all())
        cfg["problem"].update({"phase_mode": "random", "common_feature_bias": 0.0})
        baseline = build_problem(cfg, "cpu", 123)
        self.assertTrue((problem.A[:, :, 0] == baseline.A[:, :, 0] + 0.25).all())

    def assert_rows_close(self, expected, observed):
        excluded = {
            "time_sec",
            "training_time",
            "communication_time",
            "evaluation_time",
        }
        self.assertEqual(len(expected), len(observed))
        for expected_row, observed_row in zip(expected, observed):
            expected_keys = set(expected_row) - excluded
            observed_keys = set(observed_row) - excluded
            self.assertEqual(expected_keys, observed_keys)
            for key in expected_keys:
                expected_value = expected_row[key]
                observed_value = observed_row[key]
                if isinstance(expected_value, float):
                    self.assertTrue(
                        math.isclose(
                            expected_value,
                            float(observed_value),
                            rel_tol=1e-5,
                            abs_tol=1e-7,
                        ),
                        msg=(
                            f"{key}: simulator={expected_value}, "
                            f"cpu_process={observed_value}"
                        ),
                    )
                else:
                    self.assertEqual(expected_value, observed_value, msg=key)

    def _simulator_rows(self, cfg, method, formal_seed, world_size):
        seed_bundle = make_seed_bundle(formal_seed, method, world_size)
        problem = build_problem(cfg, "cpu", seed_bundle.problem_seed)
        shards = make_worker_shards(
            problem.n,
            world_size,
            "cpu",
            seed_bundle.partition_seed,
            shuffle=True,
        )
        if method == "NOG-FO":
            return run_nog(
                problem, cfg, shards, seed_bundle, "sfo", method
            )
        return run_me_dol(
            problem, cfg, shards, seed_bundle, "sfo", method
        )

    def test_two_process_trajectories_match_simulator(self):
        cfg = equivalence_config()
        world_size = 2
        formal_seed = 3
        with tempfile.TemporaryDirectory() as directory:
            for method in ("NOG-FO", "ME-DOL-FO"):
                with self.subTest(method=method):
                    expected = self._simulator_rows(
                        cfg, method, formal_seed, world_size
                    )
                    output = Path(directory) / f"{method}.json"
                    launch = run_cpu_fo_task(
                        cfg,
                        method,
                        formal_seed,
                        world_size,
                        output,
                        CpuProcessConfig(
                            process_group_timeout_seconds=30.0,
                            launch_timeout_seconds=60.0,
                            intraop_threads=1,
                        ),
                    )
                    with open(output, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)

                    self.assert_rows_close(expected, payload["rows"])
                    self.assertEqual(len(set(launch.child_pids)), world_size)
                    self.assertTrue(
                        all(
                            item["torch_threads"] == 1
                            for item in payload["rank_metadata"]
                        )
                    )


if __name__ == "__main__":
    unittest.main()
