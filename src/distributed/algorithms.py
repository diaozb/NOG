"""Algorithms used by the single-process distributed comparison runner."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import torch

from src.distributed.common import (
    OracleType,
    SeedBundle,
    WorkAccounting,
    complete_graph_mixing,
    distributed_mean_oracle,
    evaluation_seed,
    evaluate_point,
    first_order_smoothed_estimator,
    isolated_torch_seed,
    mix_worker_vectors,
    sample_sphere,
    scheduled_rank_seed,
    uses_rank_seed_schedule,
    zeroth_order_two_point_estimator,
)
from src.synthetic.run_synthetic import SyntheticMaxSinL1, project_l2_ball, seed_all


def run_nog(
    problem: SyntheticMaxSinL1,
    cfg: Dict[str, Any],
    shards: Sequence[torch.Tensor],
    seed_bundle: SeedBundle,
    oracle_type: OracleType,
    method_name: str,
    optimistic: bool = True,
) -> List[Dict[str, Any]]:
    """Run first- or zeroth-order NOG with theorem-aligned block averages."""

    seed_all(seed_bundle.method_seed)
    tcfg, ocfg, ncfg = cfg["train"], cfg["oracle"], cfg["nog"]
    rounds = int(tcfg["rounds"])
    eval_every = int(tcfg["eval_every"])
    target_delta = float(ocfg.get("target_delta", ocfg["delta"]))
    smoothing_delta = float(
        ncfg.get("smoothing_delta", ocfg["delta"])
    )
    block_size = int(ncfg["M"])
    eta = float(ncfg["eta"])
    radius = smoothing_delta / block_size
    n_workers = len(shards)

    if rounds % block_size != 0:
        raise ValueError(f"NOG requires rounds % M == 0, got {rounds} % {block_size}.")

    accounting = WorkAccounting(oracle_type=oracle_type, worker_count=n_workers)
    x = torch.zeros(problem.d, device=problem.device)
    update = torch.zeros_like(x)
    use_schedule = uses_rank_seed_schedule(cfg)
    oracle_call_index = 0

    def mean_oracle(point: torch.Tensor) -> tuple[torch.Tensor, List[int]]:
        nonlocal oracle_call_index
        result = distributed_mean_oracle(
            problem,
            point,
            cfg,
            shards,
            oracle_type,
            smoothing_delta=smoothing_delta,
            seed_bundle=seed_bundle if use_schedule else None,
            oracle_call_index=oracle_call_index if use_schedule else None,
        )
        oracle_call_index += 1
        return result

    # The two independent initial oracle values are training work and require
    # two logical aggregations. We count both instead of hiding initialization.
    grad_tm2, calls = mean_oracle(x)
    accounting.add_training(calls)
    accounting.communicate()
    grad_tm1, calls = mean_oracle(x)
    accounting.add_training(calls)
    accounting.communicate()

    rows: List[Dict[str, Any]] = []
    block_points: List[torch.Tensor] = []
    block_oracles: List[torch.Tensor] = []
    last_eval_iteration = 0
    start = time.time()

    for iteration in range(1, rounds + 1):
        if optimistic:
            update_argument = update - 2.0 * eta * grad_tm1 + eta * grad_tm2
        else:
            update_argument = update - eta * grad_tm1
        update = project_l2_ball(update_argument, radius=radius)
        if use_schedule:
            interpolation_seed = scheduled_rank_seed(
                seed_bundle, "nog_interpolation", iteration, 0
            )
            with isolated_torch_seed(interpolation_seed, problem.device):
                interpolation = torch.rand((), device=problem.device)
        else:
            interpolation = torch.rand((), device=problem.device)
        y = x + interpolation * update
        x = (x + update).detach()
        block_points.append(y.detach().clone())

        grad_new, calls = mean_oracle(y)
        accounting.add_training(calls)
        accounting.communicate()
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
            eval_seed = _evaluation_seed(seed_bundle, iteration, cfg)
            metrics, eval_calls = evaluate_point(problem, y_bar, cfg, eval_seed)
            accounting.add_evaluation(eval_calls)
            last_eval_iteration = iteration
            row: Dict[str, Any] = {
                "method": method_name,
                "base_method": "NOG" if optimistic else "NOG-nonopt",
                "iteration": iteration,
                "round": iteration,
                "block_id": block_id,
                "worker_count": n_workers,
                "eval_point": "y_bar",
                "delta": target_delta,
                "target_delta": target_delta,
                "smoothing_delta": smoothing_delta,
                "M": block_size,
                "lr_or_eta": eta,
                "block_oracle_norm": block_oracle_norm,
                "time_sec": time.time() - start,
                **seed_bundle.as_dict(),
                **accounting.snapshot(),
                **metrics,
            }
            rows.append(row)

        block_points.clear()
        block_oracles.clear()

    return rows


def _evaluation_seed(
    seed_bundle: SeedBundle,
    iteration: int,
    cfg: Dict[str, Any],
) -> int:
    """Use the configured method-independent evaluation sample bank."""

    mode = cfg.get("oracle", {}).get("evaluation_seed_mode", "checkpoint")
    return evaluation_seed(seed_bundle, iteration, mode)


def _local_oracles(
    problem: SyntheticMaxSinL1,
    points: torch.Tensor,
    shards: Sequence[torch.Tensor],
    oracle_type: OracleType,
    delta: float,
    smooth_batch: int = 1,
    data_batch: int = 1,
    seed_bundle: SeedBundle | None = None,
    oracle_call_index: int | None = None,
) -> tuple[torch.Tensor, List[int]]:
    estimator = (
        first_order_smoothed_estimator
        if oracle_type == "sfo"
        else zeroth_order_two_point_estimator
    )
    if (seed_bundle is None) != (oracle_call_index is None):
        raise ValueError(
            "seed_bundle and oracle_call_index must be provided together."
        )
    estimates = []
    for index, shard in enumerate(shards):
        if seed_bundle is None:
            estimate = estimator(
                problem=problem,
                x=points[index],
                delta=delta,
                smooth_batch=smooth_batch,
                data_batch=data_batch,
                idx_pool=shard,
            )
        else:
            seed = scheduled_rank_seed(
                seed_bundle,
                f"{oracle_type}_local_oracle",
                int(oracle_call_index),
                index,
            )
            with isolated_torch_seed(seed, problem.device):
                estimate = estimator(
                    problem=problem,
                    x=points[index],
                    delta=delta,
                    smooth_batch=smooth_batch,
                    data_batch=data_batch,
                    idx_pool=shard,
                )
        estimates.append(estimate)
    multiplier = 1 if oracle_type == "sfo" else 2
    calls = [multiplier * smooth_batch * data_batch for _ in shards]
    return torch.stack(estimates), calls


def _project_worker_rows(values: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.stack([project_l2_ball(row, radius) for row in values])


def _base_row(
    method_name: str,
    base_method: str,
    iteration: int,
    worker_count: int,
    eval_point: str,
    elapsed: float,
    seed_bundle: SeedBundle,
    accounting: WorkAccounting,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "method": method_name,
        "base_method": base_method,
        "iteration": iteration,
        "round": iteration,
        "worker_count": worker_count,
        "eval_point": eval_point,
        "time_sec": elapsed,
        **seed_bundle.as_dict(),
        **accounting.snapshot(),
        **metrics,
    }


def run_me_dol(
    problem: SyntheticMaxSinL1,
    cfg: Dict[str, Any],
    shards: Sequence[torch.Tensor],
    seed_bundle: SeedBundle,
    oracle_type: OracleType,
    method_name: str,
) -> List[Dict[str, Any]]:
    """ME-DOL Algorithms 1--4 with Euclidean decentralized OGD."""

    seed_all(seed_bundle.method_seed)
    rounds = int(cfg["train"]["rounds"])
    eval_every = int(cfg["train"]["eval_every"])
    epoch_length = int(cfg["me_dol"]["epoch_length"])
    if rounds % epoch_length != 0:
        raise ValueError(
            f"ME-DOL requires rounds divisible by epoch_length, got {rounds} and {epoch_length}."
        )

    n_workers = len(shards)
    target_delta = float(
        cfg["oracle"].get("target_delta", cfg["oracle"]["delta"])
    )
    smoothing_delta = float(cfg["me_dol"].get("smoothing_delta", cfg["oracle"]["delta"]))
    multiplier_config = cfg["me_dol"]["theory_multiplier"]
    multiplier = float(
        multiplier_config[oracle_type]
        if isinstance(multiplier_config, dict)
        else multiplier_config
    )
    smooth_batch = int(cfg["me_dol"].get("smooth_B", 1))
    data_batch = int(cfg["me_dol"].get("data_B_per_worker", 1))
    if smooth_batch < 1 or data_batch < 1:
        raise ValueError("ME-DOL oracle batch sizes must be positive.")
    theory_radius = target_delta / (4.0 * epoch_length * n_workers**0.5)
    radius = multiplier * theory_radius
    learning_rate = radius / epoch_length**0.5
    mixing = complete_graph_mixing(n_workers, problem.device)
    accounting = WorkAccounting(oracle_type=oracle_type, worker_count=n_workers)

    y = torch.zeros(n_workers, problem.d, device=problem.device)
    rows: List[Dict[str, Any]] = []
    last_eval_iteration = 0
    start = time.time()
    use_schedule = uses_rank_seed_schedule(cfg)
    oracle_call_index = 0

    for epoch_start in range(0, rounds, epoch_length):
        delta_half = torch.zeros_like(y)
        previous_gradients = torch.zeros_like(y)
        epoch_points: List[torch.Tensor] = []

        for inner in range(1, epoch_length + 1):
            actions = _project_worker_rows(
                delta_half - learning_rate * previous_gradients,
                radius,
            )
            delta_half = mix_worker_vectors(mixing, actions)
            x = y + actions
            if use_schedule:
                global_iteration = epoch_start + inner
                interpolation_values = []
                for rank in range(n_workers):
                    interpolation_seed = scheduled_rank_seed(
                        seed_bundle,
                        "me_dol_interpolation",
                        global_iteration,
                        rank,
                    )
                    with isolated_torch_seed(interpolation_seed, problem.device):
                        interpolation_values.append(
                            torch.rand((), device=problem.device)
                        )
                interpolation = torch.stack(interpolation_values).reshape(n_workers, 1)
            else:
                interpolation = torch.rand(n_workers, 1, device=problem.device)
            w = y + interpolation * actions
            y = mix_worker_vectors(mixing, x)

            gradients, calls = _local_oracles(
                problem,
                w,
                shards,
                oracle_type,
                smoothing_delta,
                smooth_batch=smooth_batch,
                data_batch=data_batch,
                seed_bundle=seed_bundle if use_schedule else None,
                oracle_call_index=oracle_call_index if use_schedule else None,
            )
            oracle_call_index += 1
            accounting.add_training(calls)
            # delta_half and y can be concatenated into one communication payload.
            accounting.communicate()
            previous_gradients = gradients
            epoch_points.append(w.detach().clone())

        iteration = epoch_start + epoch_length
        w_bar = torch.stack(epoch_points).mean(dim=(0, 1))
        should_evaluate = (
            iteration == epoch_length
            or iteration == rounds
            or iteration - last_eval_iteration >= eval_every
        )
        if should_evaluate:
            metrics, eval_calls = evaluate_point(
                problem,
                w_bar,
                cfg,
                _evaluation_seed(seed_bundle, iteration, cfg),
            )
            accounting.add_evaluation(eval_calls)
            last_eval_iteration = iteration
            row = _base_row(
                method_name,
                "ME-DOL",
                iteration,
                n_workers,
                "epoch_w_bar",
                time.time() - start,
                seed_bundle,
                accounting,
                metrics,
            )
            row.update(
                {
                    "delta": target_delta,
                    "target_delta": target_delta,
                    "smoothing_delta": smoothing_delta,
                    "epoch_length": epoch_length,
                    "domain_radius": radius,
                    "lr_or_eta": learning_rate,
                    "theory_multiplier": multiplier,
                    "smooth_B": smooth_batch,
                    "data_B_per_worker": data_batch,
                }
            )
            rows.append(row)

    return rows


def run_dgfm(
    problem: SyntheticMaxSinL1,
    cfg: Dict[str, Any],
    shards: Sequence[torch.Tensor],
    seed_bundle: SeedBundle,
    method_name: str = "DGFM",
) -> List[Dict[str, Any]]:
    """DGFM Algorithm 1 with explicit gradient-tracking communications."""

    seed_all(seed_bundle.method_seed)
    rounds = int(cfg["train"]["rounds"])
    eval_every = int(cfg["train"]["eval_every"])
    target_delta = float(
        cfg["oracle"].get("target_delta", cfg["oracle"]["delta"])
    )
    smoothing_delta = float(cfg["dgfm"].get("smoothing_delta", cfg["oracle"]["delta"]))
    eta = float(cfg["dgfm"]["eta"])
    batch_size = int(cfg["dgfm"].get("batch_size", 1))
    n_workers = len(shards)
    mixing = complete_graph_mixing(n_workers, problem.device)
    accounting = WorkAccounting(oracle_type="szo", worker_count=n_workers)

    x = torch.zeros(n_workers, problem.d, device=problem.device)
    tracker = torch.zeros_like(x)
    previous_gradients = torch.zeros_like(x)
    rows: List[Dict[str, Any]] = []
    start = time.time()

    for iteration in range(1, rounds + 1):
        gradients, calls = _local_oracles(
            problem,
            x,
            shards,
            "szo",
            smoothing_delta,
            smooth_batch=batch_size,
            data_batch=1,
        )
        accounting.add_training(calls)
        tracker = mix_worker_vectors(mixing, tracker + gradients - previous_gradients)
        accounting.communicate()
        x = mix_worker_vectors(mixing, x - eta * tracker).detach()
        accounting.communicate()
        previous_gradients = gradients

        should_evaluate = iteration == 1 or iteration == rounds or iteration % eval_every == 0
        if should_evaluate:
            x_bar = x.mean(dim=0)
            metrics, eval_calls = evaluate_point(
                problem,
                x_bar,
                cfg,
                _evaluation_seed(seed_bundle, iteration, cfg),
            )
            accounting.add_evaluation(eval_calls)
            row = _base_row(
                method_name,
                "DGFM",
                iteration,
                n_workers,
                "x_bar",
                time.time() - start,
                seed_bundle,
                accounting,
                metrics,
            )
            row.update(
                {
                    "delta": target_delta,
                    "target_delta": target_delta,
                    "smoothing_delta": smoothing_delta,
                    "lr_or_eta": eta,
                    "batch_size": batch_size,
                }
            )
            rows.append(row)

    return rows


def _draw_zo_pairs(
    problem: SyntheticMaxSinL1,
    shard: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.randint(0, shard.numel(), (batch_size,), device=problem.device)
    return shard[positions], sample_sphere(batch_size, problem.d, problem.device)


def _zo_from_pairs(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    delta: float,
    indices: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        value_plus = problem.component_losses(
            x.unsqueeze(0) + delta * directions,
            indices,
        )
        value_minus = problem.component_losses(
            x.unsqueeze(0) - delta * directions,
            indices,
        )
        estimates = (
            problem.d
            / (2.0 * delta)
            * (value_plus - value_minus).unsqueeze(1)
            * directions
        )
    return estimates.mean(dim=0)


def run_dgfm_plus(
    problem: SyntheticMaxSinL1,
    cfg: Dict[str, Any],
    shards: Sequence[torch.Tensor],
    seed_bundle: SeedBundle,
    method_name: str = "DGFM+",
) -> List[Dict[str, Any]]:
    """DGFM+ Algorithm 2 with SPIDER and exact SZO/communication counts."""

    seed_all(seed_bundle.method_seed)
    rounds = int(cfg["train"]["rounds"])
    eval_every = int(cfg["train"]["eval_every"])
    dcfg = cfg["dgfm_plus"]
    target_delta = float(
        cfg["oracle"].get("target_delta", cfg["oracle"]["delta"])
    )
    smoothing_delta = float(dcfg.get("smoothing_delta", cfg["oracle"]["delta"]))
    eta = float(dcfg["eta"])
    small_batch = int(dcfg["small_batch"])
    large_batch = int(dcfg["large_batch"])
    restart_period = int(dcfg["restart_period"])
    restart_mixing_rounds = int(dcfg["restart_mixing_rounds"])
    n_workers = len(shards)
    mixing = complete_graph_mixing(n_workers, problem.device)
    accounting = WorkAccounting(oracle_type="szo", worker_count=n_workers)

    x = torch.zeros(n_workers, problem.d, device=problem.device)
    previous_x = x.clone()
    tracker = torch.zeros_like(x)
    previous_v = torch.zeros_like(x)
    rows: List[Dict[str, Any]] = []
    start = time.time()

    for zero_based_iteration in range(rounds):
        iteration = zero_based_iteration + 1
        is_restart = zero_based_iteration % restart_period == 0
        current_v_rows = []

        if is_restart:
            for worker, shard in enumerate(shards):
                indices, directions = _draw_zo_pairs(problem, shard, large_batch)
                current_v_rows.append(
                    _zo_from_pairs(problem, x[worker], smoothing_delta, indices, directions)
                )
            current_v = torch.stack(current_v_rows)
            accounting.add_training(2 * large_batch)
            tracker = current_v
            for _ in range(restart_mixing_rounds):
                tracker = mix_worker_vectors(mixing, tracker)
            accounting.communicate(restart_mixing_rounds)
        else:
            for worker, shard in enumerate(shards):
                indices, directions = _draw_zo_pairs(problem, shard, small_batch)
                gradient_current = _zo_from_pairs(
                    problem, x[worker], smoothing_delta, indices, directions
                )
                gradient_previous = _zo_from_pairs(
                    problem, previous_x[worker], smoothing_delta, indices, directions
                )
                current_v_rows.append(
                    previous_v[worker] + gradient_current - gradient_previous
                )
            current_v = torch.stack(current_v_rows)
            accounting.add_training(4 * small_batch)
            tracker = mix_worker_vectors(
                mixing,
                tracker + current_v - previous_v,
            )
            accounting.communicate()

        next_x = mix_worker_vectors(mixing, x - eta * tracker).detach()
        accounting.communicate()
        previous_x, x = x, next_x
        previous_v = current_v

        should_evaluate = iteration == 1 or iteration == rounds or iteration % eval_every == 0
        if should_evaluate:
            x_bar = x.mean(dim=0)
            metrics, eval_calls = evaluate_point(
                problem,
                x_bar,
                cfg,
                _evaluation_seed(seed_bundle, iteration, cfg),
            )
            accounting.add_evaluation(eval_calls)
            row = _base_row(
                method_name,
                "DGFM+",
                iteration,
                n_workers,
                "x_bar",
                time.time() - start,
                seed_bundle,
                accounting,
                metrics,
            )
            row.update(
                {
                    "delta": target_delta,
                    "target_delta": target_delta,
                    "smoothing_delta": smoothing_delta,
                    "lr_or_eta": eta,
                    "small_batch": small_batch,
                    "large_batch": large_batch,
                    "restart_period": restart_period,
                    "restart_mixing_rounds": restart_mixing_rounds,
                }
            )
            rows.append(row)

    return rows
