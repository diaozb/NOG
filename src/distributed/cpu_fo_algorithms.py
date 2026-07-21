"""Real CPU-process implementations of the first-order comparison methods."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist

from src.distributed.common import (
    SeedBundle,
    WorkAccounting,
    build_problem,
    evaluation_seed,
    evaluate_point,
    first_order_smoothed_estimator,
    isolated_torch_seed,
    local_batch_size,
    make_seed_bundle,
    scheduled_rank_seed,
)
from src.distributed.cpu_process import (
    CpuProcessConfig,
    LaunchSummary,
    RankTimingRecorder,
    all_gather_objects,
    all_reduce_mean_,
    launch_cpu_processes,
    make_rank_shard,
    max_rank_timings,
)
from src.synthetic.run_synthetic import project_l2_ball


SUPPORTED_CPU_FO_METHODS = {"NOG-FO", "ME-DOL-FO"}


def _evaluation_seed(
    seed_bundle: SeedBundle,
    iteration: int,
    cfg: Dict[str, Any],
) -> int:
    mode = cfg.get("oracle", {}).get("evaluation_seed_mode", "checkpoint")
    return evaluation_seed(seed_bundle, iteration, mode)


def _rank_local_fo_oracle(
    problem: Any,
    point: torch.Tensor,
    shard: torch.Tensor,
    cfg: Dict[str, Any],
    seed_bundle: SeedBundle,
    stream: str,
    oracle_call_index: int,
    rank: int,
    smooth_batch: int,
    data_batch: int,
) -> torch.Tensor:
    seed = scheduled_rank_seed(
        seed_bundle,
        stream,
        oracle_call_index,
        rank,
    )
    with isolated_torch_seed(seed, problem.device):
        return first_order_smoothed_estimator(
            problem=problem,
            x=point,
            delta=float(cfg["oracle"]["delta"]),
            smooth_batch=int(smooth_batch),
            data_batch=int(data_batch),
            idx_pool=shard,
        )


def _scheduled_uniform(
    seed_bundle: SeedBundle,
    stream: str,
    step: int,
    rank: int,
) -> torch.Tensor:
    seed = scheduled_rank_seed(seed_bundle, stream, step, rank)
    with isolated_torch_seed(seed, "cpu"):
        return torch.rand(())


def _rank_metadata(rank: int, shard: torch.Tensor) -> Dict[str, Any]:
    shard_bytes = shard.detach().cpu().numpy().tobytes()
    return {
        "rank": rank,
        "pid": os.getpid(),
        "shard_size": int(shard.numel()),
        "shard_sha256": hashlib.sha256(shard_bytes).hexdigest(),
        "torch_threads": torch.get_num_threads(),
    }


def _validate_distributed_shards(shard: torch.Tensor, n_data: int) -> None:
    shards = all_gather_objects(shard.tolist())
    flat = [index for rank_shard in shards for index in rank_shard]
    if sorted(flat) != list(range(n_data)):
        raise ValueError("CPU process shards overlap or do not exhaust the dataset.")


def _rank0_evaluate(
    rank: int,
    problem: Any,
    point: torch.Tensor,
    cfg: Dict[str, Any],
    eval_seed: int,
    timer: RankTimingRecorder,
) -> tuple[Dict[str, float] | None, int]:
    metrics = None
    eval_calls = (
        int(cfg["oracle"]["eval_smooth_B"])
        * int(cfg["oracle"]["eval_data_B"])
    )
    with timer.phase("evaluation_time", synchronize_start=True):
        if rank == 0:
            metrics, observed_calls = evaluate_point(problem, point, cfg, eval_seed)
            if observed_calls != eval_calls:
                raise ValueError(
                    f"Evaluation work mismatch: {observed_calls} != {eval_calls}."
                )
        dist.barrier()
    return metrics, eval_calls


def _run_nog_fo_rank(
    rank: int,
    world_size: int,
    problem: Any,
    shard: torch.Tensor,
    cfg: Dict[str, Any],
    seed_bundle: SeedBundle,
    timer: RankTimingRecorder,
) -> List[Dict[str, Any]]:
    rounds = int(cfg["train"]["rounds"])
    eval_every = int(cfg["train"]["eval_every"])
    block_size = int(cfg["nog"]["M"])
    eta = float(cfg["nog"]["eta"])
    delta = float(cfg["oracle"]["delta"])
    radius = delta / block_size
    if rounds % block_size != 0:
        raise ValueError(f"NOG requires rounds % M == 0, got {rounds} % {block_size}.")

    smooth_batch = int(cfg["oracle"]["smooth_B"])
    data_batch = local_batch_size(cfg, world_size)
    calls_per_rank = smooth_batch * data_batch
    accounting = WorkAccounting("sfo", world_size)
    x = torch.zeros(problem.d)
    update = torch.zeros_like(x)
    oracle_call_index = 0

    def mean_oracle(point: torch.Tensor) -> torch.Tensor:
        nonlocal oracle_call_index
        local_gradient = _rank_local_fo_oracle(
            problem,
            point,
            shard,
            cfg,
            seed_bundle,
            "sfo_mean_oracle",
            oracle_call_index,
            rank,
            smooth_batch,
            data_batch,
        )
        oracle_call_index += 1
        all_reduce_mean_(local_gradient, timer)
        accounting.add_training(calls_per_rank)
        accounting.communicate()
        return local_gradient

    with timer.phase("training_time", synchronize_start=True):
        grad_tm2 = mean_oracle(x)
        grad_tm1 = mean_oracle(x)

    rows: List[Dict[str, Any]] = []
    block_points: List[torch.Tensor] = []
    block_oracles: List[torch.Tensor] = []
    last_eval_iteration = 0

    for iteration in range(1, rounds + 1):
        with timer.phase("training_time"):
            update = project_l2_ball(
                update - 2.0 * eta * grad_tm1 + eta * grad_tm2,
                radius=radius,
            )
            interpolation = _scheduled_uniform(
                seed_bundle, "nog_interpolation", iteration, 0
            )
            y = x + interpolation * update
            x = (x + update).detach()
            block_points.append(y.detach().clone())
            grad_new = mean_oracle(y)
            block_oracles.append(grad_new.detach().clone())
            grad_tm2, grad_tm1 = grad_tm1, grad_new

        if iteration % block_size != 0:
            continue

        block_id = iteration // block_size
        y_bar = torch.stack(block_points).mean(dim=0)
        block_oracle_norm = float(torch.stack(block_oracles).mean(dim=0).norm().item())
        should_evaluate = (
            block_id == 1
            or iteration == rounds
            or iteration - last_eval_iteration >= eval_every
        )
        if should_evaluate:
            metrics, eval_calls = _rank0_evaluate(
                rank,
                problem,
                y_bar,
                cfg,
                _evaluation_seed(seed_bundle, iteration, cfg),
                timer,
            )
            accounting.add_evaluation(eval_calls)
            last_eval_iteration = iteration
            timings = max_rank_timings(timer)
            if rank == 0:
                rows.append(
                    {
                        "method": "NOG-FO",
                        "base_method": "NOG",
                        "iteration": iteration,
                        "round": iteration,
                        "block_id": block_id,
                        "worker_count": world_size,
                        "eval_point": "y_bar",
                        "delta": delta,
                        "M": block_size,
                        "lr_or_eta": eta,
                        "block_oracle_norm": block_oracle_norm,
                        "time_sec": timings["training_time"],
                        "training_time": timings["training_time"],
                        "communication_time": timings["communication_time"],
                        "evaluation_time": timings["evaluation_time"],
                        **seed_bundle.as_dict(),
                        **accounting.snapshot(),
                        **metrics,
                    }
                )

        block_points.clear()
        block_oracles.clear()

    return rows


def _run_me_dol_fo_rank(
    rank: int,
    world_size: int,
    problem: Any,
    shard: torch.Tensor,
    cfg: Dict[str, Any],
    seed_bundle: SeedBundle,
    timer: RankTimingRecorder,
) -> List[Dict[str, Any]]:
    rounds = int(cfg["train"]["rounds"])
    eval_every = int(cfg["train"]["eval_every"])
    epoch_length = int(cfg["me_dol"]["epoch_length"])
    if rounds % epoch_length != 0:
        raise ValueError(
            f"ME-DOL requires rounds divisible by epoch_length, got "
            f"{rounds} and {epoch_length}."
        )

    delta = float(cfg["oracle"]["delta"])
    multiplier_config = cfg["me_dol"]["theory_multiplier"]
    multiplier = float(
        multiplier_config["sfo"]
        if isinstance(multiplier_config, dict)
        else multiplier_config
    )
    radius = multiplier * delta / (4.0 * epoch_length * world_size**0.5)
    learning_rate = radius / epoch_length**0.5
    accounting = WorkAccounting("sfo", world_size)
    y = torch.zeros(problem.d)
    rows: List[Dict[str, Any]] = []
    last_eval_iteration = 0
    oracle_call_index = 0

    dist.barrier()
    for epoch_start in range(0, rounds, epoch_length):
        delta_half = torch.zeros_like(y)
        previous_gradient = torch.zeros_like(y)
        epoch_points: List[torch.Tensor] = []

        for inner in range(1, epoch_length + 1):
            global_iteration = epoch_start + inner
            with timer.phase("training_time"):
                action = project_l2_ball(
                    delta_half - learning_rate * previous_gradient,
                    radius,
                )
                x = y + action
                interpolation = _scheduled_uniform(
                    seed_bundle,
                    "me_dol_interpolation",
                    global_iteration,
                    rank,
                )
                w = y + interpolation * action

                payload = torch.cat([action, x])
                all_reduce_mean_(payload, timer)
                delta_half = payload[: problem.d].clone()
                y = payload[problem.d :].clone()

                previous_gradient = _rank_local_fo_oracle(
                    problem,
                    w,
                    shard,
                    cfg,
                    seed_bundle,
                    "sfo_local_oracle",
                    oracle_call_index,
                    rank,
                    smooth_batch=1,
                    data_batch=1,
                )
                oracle_call_index += 1
                accounting.add_training(1)
                accounting.communicate()
                epoch_points.append(w.detach().clone())

        iteration = epoch_start + epoch_length
        should_evaluate = (
            iteration == epoch_length
            or iteration == rounds
            or iteration - last_eval_iteration >= eval_every
        )
        if should_evaluate:
            local_epoch_mean = torch.stack(epoch_points).mean(dim=0)
            metrics = None
            eval_calls = (
                int(cfg["oracle"]["eval_smooth_B"])
                * int(cfg["oracle"]["eval_data_B"])
            )
            with timer.phase("evaluation_time", synchronize_start=True):
                w_bar = local_epoch_mean.clone()
                dist.all_reduce(w_bar, op=dist.ReduceOp.SUM)
                w_bar.div_(world_size)
                if rank == 0:
                    metrics, observed_calls = evaluate_point(
                        problem,
                        w_bar,
                        cfg,
                        _evaluation_seed(seed_bundle, iteration, cfg),
                    )
                    if observed_calls != eval_calls:
                        raise ValueError(
                            f"Evaluation work mismatch: {observed_calls} != {eval_calls}."
                        )
                dist.barrier()

            accounting.add_evaluation(eval_calls)
            last_eval_iteration = iteration
            timings = max_rank_timings(timer)
            if rank == 0:
                rows.append(
                    {
                        "method": "ME-DOL-FO",
                        "base_method": "ME-DOL",
                        "iteration": iteration,
                        "round": iteration,
                        "worker_count": world_size,
                        "eval_point": "epoch_w_bar",
                        "time_sec": timings["training_time"],
                        "training_time": timings["training_time"],
                        "communication_time": timings["communication_time"],
                        "evaluation_time": timings["evaluation_time"],
                        "delta": delta,
                        "epoch_length": epoch_length,
                        "domain_radius": radius,
                        "lr_or_eta": learning_rate,
                        "theory_multiplier": multiplier,
                        **seed_bundle.as_dict(),
                        **accounting.snapshot(),
                        **metrics,
                    }
                )

    return rows


def cpu_fo_worker(
    rank: int,
    world_size: int,
    cfg: Dict[str, Any],
    method: str,
    formal_seed: int,
    output_path: str,
) -> None:
    if method not in SUPPORTED_CPU_FO_METHODS:
        raise ValueError(f"Unsupported CPU FO method: {method}.")
    if cfg["distributed"].get("rng_mode") != "rank_schedule":
        raise ValueError("CPU FO runner requires distributed.rng_mode=rank_schedule.")

    seed_bundle = make_seed_bundle(formal_seed, method, world_size)
    problem = build_problem(cfg, "cpu", seed_bundle.problem_seed)
    shard = make_rank_shard(
        problem.n,
        world_size,
        rank,
        seed_bundle.partition_seed,
        shuffle=bool(cfg["distributed"].get("shuffle_partitions", True)),
    )
    _validate_distributed_shards(shard, problem.n)
    timer = RankTimingRecorder()

    if method == "NOG-FO":
        rows = _run_nog_fo_rank(
            rank, world_size, problem, shard, cfg, seed_bundle, timer
        )
    else:
        rows = _run_me_dol_fo_rank(
            rank, world_size, problem, shard, cfg, seed_bundle, timer
        )

    metadata = all_gather_objects(_rank_metadata(rank, shard))
    if rank == 0:
        payload = {
            "method": method,
            "formal_seed": formal_seed,
            "worker_count": world_size,
            "rng_mode": "rank_schedule",
            "rank_metadata": metadata,
            "rows": rows,
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def run_cpu_fo_task(
    cfg: Dict[str, Any],
    method: str,
    formal_seed: int,
    world_size: int,
    output_path: str | Path,
    process_config: CpuProcessConfig | None = None,
) -> LaunchSummary:
    """Launch one method/seed/worker-count task and append parent timing."""

    task_cfg = copy.deepcopy(cfg)
    task_cfg.setdefault("distributed", {})["rng_mode"] = "rank_schedule"
    task_cfg.setdefault("run", {})["device"] = "cpu"
    output = Path(output_path)
    launch = launch_cpu_processes(
        cpu_fo_worker,
        world_size,
        (task_cfg, method, int(formal_seed), str(output)),
        process_config,
    )
    with open(output, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["launch"] = launch.as_dict()
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return launch
