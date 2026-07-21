"""Shared building blocks for simulated distributed NOG experiments.

The experiment is intentionally a *logical* distributed simulation: workers have
separate data shards and optimizer state, but are executed sequentially on one
device.  Consequently, wall-clock time is diagnostic only.  Scientific plots
must use communication depth and oracle work from :class:`WorkAccounting`.
"""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

import torch
import numpy as np

from src.synthetic.run_synthetic import (
    SyntheticMaxSinL1,
    sample_ball,
    seed_all,
)


OracleType = Literal["sfo", "szo"]


@dataclass(frozen=True)
class SeedBundle:
    """Seeds that must be logged separately for paired comparisons."""

    formal_seed: int
    problem_seed: int
    partition_seed: int
    method_seed: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "formal_seed": self.formal_seed,
            "problem_seed": self.problem_seed,
            "partition_seed": self.partition_seed,
            "method_seed": self.method_seed,
        }


def stable_method_seed(formal_seed: int, method: str, worker_count: int) -> int:
    """Return a deterministic seed without relying on Python's salted ``hash``."""

    payload = f"{formal_seed}:{method}:{worker_count}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def make_seed_bundle(formal_seed: int, method: str, worker_count: int) -> SeedBundle:
    return SeedBundle(
        formal_seed=int(formal_seed),
        problem_seed=100_000 + int(formal_seed),
        partition_seed=200_000 + int(formal_seed),
        method_seed=stable_method_seed(int(formal_seed), method, int(worker_count)),
    )


def scheduled_rank_seed(
    seed_bundle: SeedBundle,
    stream: str,
    step: int,
    rank: int,
) -> int:
    """Stable stateless seed shared by simulator and real-process runners."""

    payload = (
        f"{seed_bundle.method_seed}:{seed_bundle.problem_seed}:"
        f"{stream}:{int(step)}:{int(rank)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def uses_rank_seed_schedule(cfg: Dict[str, Any]) -> bool:
    """Whether to isolate every rank/call RNG for process equivalence checks."""

    return cfg["distributed"].get("rng_mode", "legacy_stream") == "rank_schedule"


def evaluation_seed(
    seed_bundle: SeedBundle,
    iteration: int,
    mode: str = "checkpoint",
) -> int:
    """Method-independent evaluation seed with optional fixed sample bank."""

    if mode == "fixed_bank":
        return (seed_bundle.problem_seed * 1_000_003 + 65_537) & 0x7FFFFFFF
    if mode == "checkpoint":
        return (
            seed_bundle.problem_seed * 1_000_003 + int(iteration) * 97
        ) & 0x7FFFFFFF
    raise ValueError(f"Unknown evaluation seed mode: {mode}.")


def build_problem(cfg: Dict[str, Any], device: str, problem_seed: int) -> SyntheticMaxSinL1:
    """Build exactly one reproducible problem instance for a formal seed."""

    seed_all(problem_seed)
    pcfg = cfg["problem"]
    return SyntheticMaxSinL1(
        d=int(pcfg["d"]),
        n_data=int(pcfg["n_data"]),
        R=int(pcfg["R"]),
        lam=float(pcfg["lam"]),
        device=device,
    )


def make_worker_shards(
    n_data: int,
    n_workers: int,
    device: str,
    partition_seed: int,
    shuffle: bool = True,
) -> List[torch.Tensor]:
    """Create disjoint, exhaustive and deterministic worker data shards."""

    if n_workers < 1:
        raise ValueError(f"n_workers must be positive, got {n_workers}.")
    if n_workers > n_data:
        raise ValueError(f"n_workers={n_workers} exceeds n_data={n_data}.")

    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(partition_seed))
        indices = torch.randperm(n_data, generator=generator)
    else:
        indices = torch.arange(n_data)

    return [shard.to(device) for shard in torch.tensor_split(indices, n_workers)]


def complete_graph_mixing(n_workers: int, device: str) -> torch.Tensor:
    """Return P=(1/m)11^T, the common communication model in the main study."""

    if n_workers < 1:
        raise ValueError(f"n_workers must be positive, got {n_workers}.")
    return torch.full(
        (n_workers, n_workers),
        fill_value=1.0 / n_workers,
        device=device,
    )


def validate_mixing_matrix(matrix: torch.Tensor, atol: float = 1e-6) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Mixing matrix must be square, got {tuple(matrix.shape)}.")
    if torch.any(matrix < -atol):
        raise ValueError("Mixing matrix has negative entries.")

    ones = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    if not torch.allclose(matrix.sum(dim=0), ones, atol=atol, rtol=0):
        raise ValueError("Mixing matrix is not column stochastic.")
    if not torch.allclose(matrix.sum(dim=1), ones, atol=atol, rtol=0):
        raise ValueError("Mixing matrix is not row stochastic.")


def mix_worker_vectors(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Mix worker vectors whose leading dimension indexes workers."""

    if values.ndim < 2 or values.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"Expected values with leading worker dimension {matrix.shape[0]}, "
            f"got {tuple(values.shape)}."
        )
    return torch.einsum("ij,j...->i...", matrix, values)


def sample_sphere(num: int, dim: int, device: str) -> torch.Tensor:
    z = torch.randn(num, dim, device=device)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def grad_at_indices(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    idx_pool: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if idx_pool.numel() < 1:
        raise ValueError("A worker shard is empty.")

    positions = torch.randint(
        0,
        idx_pool.numel(),
        (batch_size,),
        device=problem.device,
    )
    indices = idx_pool[positions]
    x_var = x.detach().clone().requires_grad_(True)
    loss = problem.loss(x_var, indices)
    return torch.autograd.grad(loss, x_var)[0].detach()


def first_order_smoothed_estimator(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    delta: float,
    smooth_batch: int,
    data_batch: int,
    idx_pool: torch.Tensor,
) -> torch.Tensor:
    """First-order randomized-smoothing estimator (Proposition 4.2)."""

    gradients = []
    for _ in range(int(smooth_batch)):
        u = sample_ball(1, problem.d, problem.device).squeeze(0)
        gradients.append(grad_at_indices(problem, x + delta * u, idx_pool, data_batch))
    return torch.stack(gradients).mean(dim=0)


def zeroth_order_two_point_estimator(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    delta: float,
    smooth_batch: int,
    data_batch: int,
    idx_pool: torch.Tensor,
) -> torch.Tensor:
    """Mini-batch two-point estimator; every sample costs two SZO calls."""

    estimates = []
    for _ in range(int(smooth_batch)):
        direction = sample_sphere(1, problem.d, problem.device).squeeze(0)
        positions = torch.randint(
            0,
            idx_pool.numel(),
            (int(data_batch),),
            device=problem.device,
        )
        indices = idx_pool[positions]
        with torch.no_grad():
            value_plus = problem.loss(x + delta * direction, indices)
            value_minus = problem.loss(x - delta * direction, indices)
        estimate = (
            problem.d
            / (2.0 * delta)
            * (value_plus - value_minus)
            * direction
        )
        estimates.append(estimate)
    return torch.stack(estimates).mean(dim=0)


def local_batch_size(cfg: Dict[str, Any], n_workers: int) -> int:
    dcfg = cfg["distributed"]
    ocfg = cfg["oracle"]
    split_mode = dcfg.get("split_mode", "total_batch_fixed")

    if split_mode == "total_batch_fixed":
        total = int(ocfg["data_B_total"])
        if total % n_workers != 0:
            raise ValueError(
                f"data_B_total={total} must be divisible by n_workers={n_workers}."
            )
        return total // n_workers
    if split_mode == "per_worker_batch_fixed":
        return int(ocfg["data_B_per_worker"])
    raise ValueError(f"Unknown split_mode: {split_mode}.")


def calls_per_worker(
    oracle_type: OracleType,
    smooth_batch: int,
    data_batch: int,
) -> int:
    multiplier = 1 if oracle_type == "sfo" else 2
    return multiplier * int(smooth_batch) * int(data_batch)


def distributed_mean_oracle(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    cfg: Dict[str, Any],
    shards: Sequence[torch.Tensor],
    oracle_type: OracleType,
    data_batch: int | None = None,
    smooth_batch: int | None = None,
    seed_bundle: SeedBundle | None = None,
    oracle_call_index: int | None = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Evaluate local estimators sequentially and return their exact mean."""

    ocfg = cfg["oracle"]
    n_workers = len(shards)
    local_data_batch = (
        local_batch_size(cfg, n_workers) if data_batch is None else int(data_batch)
    )
    local_smooth_batch = (
        int(ocfg["smooth_B"]) if smooth_batch is None else int(smooth_batch)
    )
    delta = float(ocfg["delta"])
    estimator = (
        first_order_smoothed_estimator
        if oracle_type == "sfo"
        else zeroth_order_two_point_estimator
    )
    if (seed_bundle is None) != (oracle_call_index is None):
        raise ValueError(
            "seed_bundle and oracle_call_index must be provided together."
        )

    local_estimates = []
    for rank, shard in enumerate(shards):
        if seed_bundle is None:
            estimate = estimator(
                problem=problem,
                x=x,
                delta=delta,
                smooth_batch=local_smooth_batch,
                data_batch=local_data_batch,
                idx_pool=shard,
            )
        else:
            seed = scheduled_rank_seed(
                seed_bundle,
                f"{oracle_type}_mean_oracle",
                int(oracle_call_index),
                rank,
            )
            with isolated_torch_seed(seed, problem.device):
                estimate = estimator(
                    problem=problem,
                    x=x,
                    delta=delta,
                    smooth_batch=local_smooth_batch,
                    data_batch=local_data_batch,
                    idx_pool=shard,
                )
        local_estimates.append(estimate)
    per_worker_calls = [
        calls_per_worker(oracle_type, local_smooth_batch, local_data_batch)
        for _ in shards
    ]
    return torch.stack(local_estimates).mean(dim=0), per_worker_calls


@contextmanager
def isolated_torch_seed(seed: int, device: str):
    """Run evaluation RNG without perturbing the method's training RNG stream."""

    cpu_state = torch.random.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_state = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
    seed_all(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


def evaluate_point(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    cfg: Dict[str, Any],
    eval_seed: int,
) -> Tuple[Dict[str, float], int]:
    """Common centralized evaluation used for every SFO and SZO method."""

    ocfg = cfg["oracle"]
    eval_smooth_batch = int(ocfg["eval_smooth_B"])
    eval_data_batch = int(ocfg["eval_data_B"])
    all_indices = torch.arange(problem.n, device=problem.device)
    with isolated_torch_seed(eval_seed, problem.device):
        gradient = first_order_smoothed_estimator(
            problem=problem,
            x=x,
            delta=float(ocfg["delta"]),
            smooth_batch=eval_smooth_batch,
            data_batch=eval_data_batch,
            idx_pool=all_indices,
        )
    with torch.no_grad():
        objective = float(problem.loss(x).item())
    metrics = {
        "objective": objective,
        "stat_proxy": float(gradient.norm().item()),
    }
    return metrics, eval_smooth_batch * eval_data_batch


@dataclass
class WorkAccounting:
    """Exact method-agnostic training/evaluation/communication counters."""

    oracle_type: OracleType
    worker_count: int
    per_worker_work: List[int] = field(init=False)
    eval_work: int = 0
    communication_round: int = 0

    def __post_init__(self) -> None:
        if self.oracle_type not in ("sfo", "szo"):
            raise ValueError(f"Unknown oracle type: {self.oracle_type}.")
        if self.worker_count < 1:
            raise ValueError("worker_count must be positive.")
        self.per_worker_work = [0 for _ in range(self.worker_count)]

    def add_training(self, calls: int | Sequence[int]) -> None:
        if isinstance(calls, int):
            values = [calls] * self.worker_count
        else:
            values = [int(v) for v in calls]
        if len(values) != self.worker_count:
            raise ValueError(
                f"Expected {self.worker_count} per-worker call counts, got {len(values)}."
            )
        if any(v < 0 for v in values):
            raise ValueError("Oracle call increments cannot be negative.")
        self.per_worker_work = [old + new for old, new in zip(self.per_worker_work, values)]

    def add_evaluation(self, calls: int) -> None:
        if calls < 0:
            raise ValueError("Evaluation calls cannot be negative.")
        self.eval_work += int(calls)

    def communicate(self, rounds: int = 1) -> None:
        if rounds < 0:
            raise ValueError("Communication rounds cannot be negative.")
        self.communication_round += int(rounds)

    @property
    def total_work(self) -> int:
        return int(sum(self.per_worker_work))

    @property
    def per_worker_work_max(self) -> int:
        return int(max(self.per_worker_work))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "oracle_type": self.oracle_type,
            "work": self.total_work,
            "total_work": self.total_work,
            "per_worker_work_max": self.per_worker_work_max,
            "per_worker_work": list(self.per_worker_work),
            "eval_work": int(self.eval_work),
            "communication_round": int(self.communication_round),
            "depth": int(self.communication_round),
        }


def validate_shards(shards: Iterable[torch.Tensor], n_data: int) -> None:
    shards = list(shards)
    flat = torch.cat([shard.detach().cpu() for shard in shards])
    if flat.numel() != n_data:
        raise ValueError(f"Shards contain {flat.numel()} indices, expected {n_data}.")
    unique = torch.unique(flat)
    if unique.numel() != n_data:
        raise ValueError("Worker shards overlap or omit data indices.")
    if int(unique.min()) != 0 or int(unique.max()) != n_data - 1:
        raise ValueError("Worker shards do not cover [0, n_data).")


def validate_experiment_config(cfg: Dict[str, Any]) -> None:
    required = ["run", "problem", "train", "oracle", "nog", "distributed", "methods"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}.")

    rounds = int(cfg["train"]["rounds"])
    block = int(cfg["nog"]["M"])
    if rounds < 1 or block < 1 or rounds % block != 0:
        raise ValueError(f"NOG requires rounds % M == 0, got rounds={rounds}, M={block}.")

    workers = [int(v) for v in cfg["distributed"]["scaling_workers"]]
    comparison_worker = int(cfg["distributed"]["comparison_worker"])
    if comparison_worker not in workers:
        raise ValueError("comparison_worker must be included in scaling_workers.")
    for n_workers in workers:
        local_batch_size(cfg, n_workers)

    tracks = cfg["methods"]
    expected = {"sfo", "szo"}
    if set(tracks) != expected:
        raise ValueError(f"methods must define exactly {sorted(expected)}, got {sorted(tracks)}.")
