import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.distributed.check_cpu_process import cpu_process_check_worker
from src.distributed.common import make_worker_shards
from src.distributed.cpu_process import (
    CpuProcessConfig,
    launch_cpu_processes,
    make_rank_shard,
)


class CpuProcessTests(unittest.TestCase):
    def test_rank_shards_match_existing_simulator(self):
        expected = make_worker_shards(31, 4, "cpu", partition_seed=23)
        observed = [
            make_rank_shard(31, 4, rank, partition_seed=23)
            for rank in range(4)
        ]
        for expected_shard, observed_shard in zip(expected, observed):
            self.assertTrue(torch.equal(expected_shard, observed_shard))

    def test_two_real_processes_reduce_and_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "check.json"
            launch = launch_cpu_processes(
                cpu_process_check_worker,
                world_size=2,
                worker_args=(str(output), 31, 29),
                config=CpuProcessConfig(
                    launch_timeout_seconds=60.0,
                    process_group_timeout_seconds=30.0,
                    intraop_threads=1,
                ),
            )
            with open(output, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertTrue(payload["ok"])
        self.assertEqual(len(set(launch.child_pids)), 2)
        self.assertEqual(
            sorted(
                index
                for metadata in payload["rank_metadata"]
                for index in metadata["shard"]
            ),
            list(range(31)),
        )
        self.assertTrue(
            all(metadata["torch_threads"] == 1 for metadata in payload["rank_metadata"])
        )
        self.assertTrue(
            all(metadata["reduced_mean"] == 1.5 for metadata in payload["rank_metadata"])
        )
        self.assertGreater(payload["max_rank_timings"]["training_time"], 0.0)
        self.assertGreaterEqual(
            payload["max_rank_timings"]["training_time"],
            payload["max_rank_timings"]["communication_time"],
        )


if __name__ == "__main__":
    unittest.main()
