"""Reusable primitives for real multi-process CPU experiments.

This module does not implement an optimization method. It owns process
lifecycle, Gloo initialization, rank sharding, collective means, and timing
aggregation so NOG and baselines can share exactly the same systems layer.
"""

from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, Iterator, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WorkerEntryPoint = Callable[..., None]


@dataclass(frozen=True)
class CpuProcessConfig:
    """Process-group settings shared by every rank."""

    master_addr: str = "127.0.0.1"
    master_port: int | None = None
    backend: str = "gloo"
    process_group_timeout_seconds: float = 120.0
    launch_timeout_seconds: float = 300.0
    intraop_threads: int = 1


@dataclass(frozen=True)
class LaunchSummary:
    """Parent-side facts, including process startup/teardown time."""

    world_size: int
    child_pids: tuple[int, ...]
    master_addr: str
    master_port: int
    end_to_end_time: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "child_pids": list(self.child_pids),
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "end_to_end_time": self.end_to_end_time,
        }


class CpuProcessLaunchError(RuntimeError):
    """A rank failed or timed out; sibling cleanup has already completed."""

    def __init__(
        self,
        message: str,
        child_pids: Sequence[int],
        alive_after_cleanup: Sequence[int],
        cause: BaseException,
    ) -> None:
        super().__init__(message)
        self.child_pids = tuple(int(pid) for pid in child_pids)
        self.alive_after_cleanup = tuple(int(pid) for pid in alive_after_cleanup)
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "cause_type": self.cause_type,
            "cause_message": self.cause_message,
            "child_pids": list(self.child_pids),
            "alive_after_cleanup": list(self.alive_after_cleanup),
        }


class RankTimingRecorder:
    """Accumulate local rank time without adding collectives to each event."""

    def __init__(self) -> None:
        self._seconds: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str, synchronize_start: bool = False) -> Iterator[None]:
        if synchronize_start:
            require_process_group()
            dist.barrier()
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed

    def snapshot(self) -> Dict[str, float]:
        return dict(self._seconds)


def find_free_tcp_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an unused local TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def configure_cpu_threads(intraop_threads: int) -> None:
    """Prevent every rank from creating a second large thread pool."""

    threads = int(intraop_threads)
    if threads < 1:
        raise ValueError(f"intraop_threads must be positive, got {threads}.")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work.
        # Spawned experiment ranks normally take the successful branch.
        pass


def require_process_group() -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("A torch.distributed process group is required.")


@contextmanager
def gloo_process_group(
    rank: int,
    world_size: int,
    config: CpuProcessConfig,
    master_port: int,
) -> Iterator[None]:
    """Initialize and always tear down one local Gloo rank."""

    if config.backend != "gloo":
        raise ValueError(f"CPU experiments require backend='gloo', got {config.backend!r}.")
    configure_cpu_threads(config.intraop_threads)
    dist.init_process_group(
        backend=config.backend,
        init_method=f"tcp://{config.master_addr}:{master_port}",
        rank=int(rank),
        world_size=int(world_size),
        timeout=timedelta(seconds=float(config.process_group_timeout_seconds)),
    )
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def make_rank_shard(
    n_data: int,
    world_size: int,
    rank: int,
    partition_seed: int,
    shuffle: bool = True,
) -> torch.Tensor:
    """Return the rank-local member of a deterministic exhaustive partition."""

    if n_data < 1:
        raise ValueError(f"n_data must be positive, got {n_data}.")
    if world_size < 1 or world_size > n_data:
        raise ValueError(
            f"world_size must be in [1, n_data], got {world_size} for n_data={n_data}."
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")

    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(partition_seed))
        indices = torch.randperm(n_data, generator=generator)
    else:
        indices = torch.arange(n_data)
    return torch.tensor_split(indices, world_size)[rank].contiguous()


def all_reduce_mean_(
    value: torch.Tensor,
    timer: RankTimingRecorder | None = None,
) -> torch.Tensor:
    """In-place complete-graph mean, with optional local communication timing."""

    require_process_group()
    timing = timer.phase("communication_time") if timer is not None else _null_context()
    with timing:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(dist.get_world_size())
    return value


@contextmanager
def _null_context() -> Iterator[None]:
    yield


def all_gather_objects(value: Any) -> list[Any]:
    require_process_group()
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def max_rank_timings(recorder: RankTimingRecorder) -> Dict[str, float]:
    """Return max-rank elapsed time for each recorded phase on every rank."""

    snapshots = all_gather_objects(recorder.snapshot())
    keys = sorted({key for item in snapshots for key in item})
    return {
        key: max(float(item.get(key, 0.0)) for item in snapshots)
        for key in keys
    }


def _process_bootstrap(
    rank: int,
    world_size: int,
    config: CpuProcessConfig,
    master_port: int,
    entrypoint: WorkerEntryPoint,
    worker_args: Sequence[Any],
) -> None:
    with gloo_process_group(rank, world_size, config, master_port):
        entrypoint(rank, world_size, *worker_args)


def launch_cpu_processes(
    entrypoint: WorkerEntryPoint,
    world_size: int,
    worker_args: Sequence[Any] = (),
    config: CpuProcessConfig | None = None,
) -> LaunchSummary:
    """Spawn real OS processes and wait for clean completion.

    The entrypoint must be defined at module scope so Python's spawn start
    method can import it. PyTorch propagates rank exceptions and terminates
    sibling processes; an explicit wall-clock timeout also prevents silent
    deadlocks from running forever.
    """

    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    process_config = config or CpuProcessConfig()
    master_port = (
        find_free_tcp_port(process_config.master_addr)
        if process_config.master_port is None
        else int(process_config.master_port)
    )
    start = time.perf_counter()
    context = mp.start_processes(
        _process_bootstrap,
        args=(
            int(world_size),
            process_config,
            master_port,
            entrypoint,
            tuple(worker_args),
        ),
        nprocs=int(world_size),
        join=False,
        start_method="spawn",
    )
    child_pids = tuple(int(process.pid) for process in context.processes)
    # A long pilot stage may need a larger operational wall-clock guard without
    # changing the experiment configuration (and therefore its resume identity).
    launch_timeout_seconds = float(
        os.environ.get(
            "NOG_CPU_LAUNCH_TIMEOUT_SECONDS",
            process_config.launch_timeout_seconds,
        )
    )
    deadline = start + launch_timeout_seconds
    try:
        while not context.join(timeout=0.25):
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"CPU process launch exceeded {launch_timeout_seconds}s "
                    f"for world_size={world_size}."
                )
    except BaseException as error:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5.0)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        for process in context.processes:
            process.join(timeout=5.0)
        alive_after_cleanup = tuple(
            int(process.pid)
            for process in context.processes
            if process.is_alive()
        )
        raise CpuProcessLaunchError(
            (
                f"CPU process task failed for world_size={world_size}; "
                f"alive_after_cleanup={list(alive_after_cleanup)}."
            ),
            child_pids,
            alive_after_cleanup,
            error,
        ) from error

    return LaunchSummary(
        world_size=int(world_size),
        child_pids=child_pids,
        master_addr=process_config.master_addr,
        master_port=master_port,
        end_to_end_time=time.perf_counter() - start,
    )
