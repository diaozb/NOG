"""Small auditable check for the real CPU/Gloo process layer."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.distributed.cpu_process import (
    CpuProcessConfig,
    RankTimingRecorder,
    all_gather_objects,
    all_reduce_mean_,
    launch_cpu_processes,
    make_rank_shard,
    max_rank_timings,
)


def cpu_process_check_worker(
    rank: int,
    world_size: int,
    output_path: str,
    n_data: int,
    partition_seed: int,
) -> None:
    """Worker used by both the CLI and integration test."""

    shard = make_rank_shard(n_data, world_size, rank, partition_seed)
    timer = RankTimingRecorder()
    local_value = torch.tensor([float(rank + 1)], dtype=torch.float64)
    with timer.phase("training_time", synchronize_start=True):
        all_reduce_mean_(local_value, timer)

    rank_metadata = {
        "rank": rank,
        "pid": os.getpid(),
        "shard": shard.tolist(),
        "reduced_mean": float(local_value.item()),
        "torch_threads": torch.get_num_threads(),
    }
    all_metadata = all_gather_objects(rank_metadata)
    timings = max_rank_timings(timer)

    if rank == 0:
        flat_shards = [
            index
            for metadata in all_metadata
            for index in metadata["shard"]
        ]
        expected_mean = (world_size + 1.0) / 2.0
        payload: dict[str, Any] = {
            "ok": (
                len({item["pid"] for item in all_metadata}) == world_size
                and sorted(flat_shards) == list(range(n_data))
                and all(
                    abs(item["reduced_mean"] - expected_mean) < 1e-12
                    for item in all_metadata
                )
            ),
            "world_size": world_size,
            "rank_metadata": all_metadata,
            "max_rank_timings": timings,
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--n-data", type=int, default=32)
    parser.add_argument("--partition-seed", type=int, default=17)
    parser.add_argument("--threads-per-rank", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporary_output = args.output is None
    if temporary_output:
        descriptor, output_path = tempfile.mkstemp(
            prefix="nog_cpu_process_check_",
            suffix=".json",
        )
        os.close(descriptor)
    else:
        output_path = args.output

    config = CpuProcessConfig(
        launch_timeout_seconds=args.timeout_seconds,
        process_group_timeout_seconds=args.timeout_seconds,
        intraop_threads=args.threads_per_rank,
    )
    launch = launch_cpu_processes(
        cpu_process_check_worker,
        args.world_size,
        (output_path, args.n_data, args.partition_seed),
        config,
    )
    with open(output_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["launch"] = launch.as_dict()
    if not temporary_output:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    else:
        Path(output_path).unlink(missing_ok=True)
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit("CPU process check failed.")


if __name__ == "__main__":
    main()
