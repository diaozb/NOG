"""Protocol validation and censoring-aware statistics for epsilon scaling."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, Iterable, List, Sequence


REGION_ORDER = ("coarse", "medium", "fine")


def epsilon_region(epsilon: float) -> str:
    value = float(epsilon)
    if value >= 0.05:
        return "coarse"
    if value >= 0.01:
        return "medium"
    return "fine"


def validate_scaling_protocol(cfg: Dict[str, Any]) -> None:
    protocol = cfg["epsilon_scaling"]
    epsilons = [float(value) for value in protocol["epsilons"]]
    if not epsilons or any(value <= 0.0 for value in epsilons):
        raise ValueError("Scaling epsilons must be positive.")
    if epsilons != sorted(set(epsilons), reverse=True):
        raise ValueError("Scaling epsilons must be unique and decreasing.")
    if min(epsilons) > 0.002 or max(epsilons) < 0.2:
        raise ValueError("Scaling epsilon range must cover [0.002, 0.2].")

    budgets = [int(value) for value in protocol["budgets"]]
    if budgets != sorted(set(budgets)) or budgets[-1] != 61440:
        raise ValueError("Scaling budgets must increase to the frozen 61440 cap.")
    if any(budget % int(cfg["train"]["eval_every"]) for budget in budgets):
        raise ValueError("Every budget must align with eval_every.")

    pilot = {int(value) for value in protocol["pilot_seeds"]}
    formal = {int(value) for value in cfg["run"]["formal_seeds"]}
    if pilot & formal:
        raise ValueError("Pilot and formal seeds overlap.")
    if len(pilot) != 5 or len(formal) != 20:
        raise ValueError("Protocol requires 5 pilot and 20 formal seeds.")
    if int(protocol["reference_worker"]) != 8:
        raise ValueError("Primary epsilon-scaling experiment must use m=8.")
    if int(cfg["cpu_process"]["max_total_worker_processes"]) > 32:
        raise ValueError("CPU process cap exceeds 32.")
    if int(protocol["confirmed_hit_consecutive"]) < 2:
        raise ValueError("Confirmed hit requires at least two checkpoints.")

    regions = protocol["regions"]
    if tuple(regions) != REGION_ORDER:
        raise ValueError("Regions must be ordered coarse, medium, fine.")
    for name in REGION_ORDER:
        representatives = [
            float(value) for value in regions[name]["representative_epsilons"]
        ]
        if not representatives:
            raise ValueError(f"Region {name} has no representative epsilon.")
        if any(epsilon_region(value) != name for value in representatives):
            raise ValueError(f"Region {name} contains an invalid representative.")
        if any(value not in epsilons for value in representatives):
            raise ValueError(f"Region {name} representative is not preregistered.")

    robustness = protocol["robustness"]
    if [int(value) for value in robustness["workers"]] != [1, 2, 4, 8]:
        raise ValueError("Robustness worker grid must be [1,2,4,8].")
    if [float(value) for value in robustness["epsilons"]] != [0.1, 0.01, 0.005]:
        raise ValueError("Robustness epsilon grid changed from preregistration.")
    if len({int(value) for value in robustness["seeds"]}) != 10:
        raise ValueError("Robustness study requires 10 seeds.")


def kaplan_meier_restricted_mean(
    times: Sequence[float], events: Sequence[bool], horizon: float | None = None
) -> float:
    """Return restricted mean time using a right-censored Kaplan-Meier curve."""
    if len(times) != len(events) or not times:
        raise ValueError("times/events must be non-empty and equally sized.")
    observed = [float(value) for value in times]
    if any(not math.isfinite(value) or value < 0.0 for value in observed):
        raise ValueError("times must be finite and nonnegative.")
    tau = max(observed) if horizon is None else float(horizon)
    if tau < 0.0 or any(value > tau for value in observed):
        raise ValueError("horizon must cover all observed times.")

    survival = 1.0
    area = 0.0
    previous = 0.0
    for time in sorted(set(value for value in observed if value <= tau)):
        area += survival * (time - previous)
        at_risk = sum(value >= time for value in observed)
        deaths = sum(
            value == time and bool(event)
            for value, event in zip(observed, events)
        )
        if at_risk:
            survival *= 1.0 - deaths / at_risk
        previous = time
    area += survival * (tau - previous)
    return area


def summarize_censored(
    values: Sequence[float | None], censor_limits: Sequence[float]
) -> Dict[str, Any]:
    if len(values) != len(censor_limits) or not values:
        raise ValueError("values/censor_limits must be non-empty and equally sized.")
    events = [value is not None for value in values]
    times = [
        float(value) if value is not None else float(limit)
        for value, limit in zip(values, censor_limits)
    ]
    hits = [float(value) for value in values if value is not None]
    horizon = max(float(value) for value in censor_limits)
    return {
        "num_seeds": len(values),
        "hit_count": len(hits),
        "hit_rate": len(hits) / len(values),
        "conditional_mean": statistics.mean(hits) if hits else None,
        "conditional_median": statistics.median(hits) if hits else None,
        "capped_mean": statistics.mean(times),
        "restricted_mean": kaplan_meier_restricted_mean(times, events, horizon),
        "censoring_horizon": horizon,
    }


def bootstrap_interval(
    values: Sequence[float], repetitions: int, seed: int, confidence: float = 0.95
) -> tuple[float, float]:
    if not values or repetitions < 2:
        raise ValueError("Bootstrap requires values and at least two repetitions.")
    sample = [float(value) for value in values]
    rng = random.Random(int(seed))
    estimates = sorted(
        statistics.mean(rng.choice(sample) for _ in sample)
        for _ in range(int(repetitions))
    )
    tail = (1.0 - float(confidence)) / 2.0
    low = estimates[max(0, math.floor(tail * (len(estimates) - 1)))]
    high = estimates[min(len(estimates) - 1, math.ceil((1.0 - tail) * (len(estimates) - 1)))]
    return low, high


def paired_ratio_summary(
    numerator: Sequence[float | None],
    denominator: Sequence[float | None],
    numerator_limits: Sequence[float],
    denominator_limits: Sequence[float],
) -> Dict[str, Any]:
    lengths = {len(numerator), len(denominator), len(numerator_limits), len(denominator_limits)}
    if len(lengths) != 1 or not numerator:
        raise ValueError("Paired ratio inputs must have equal non-zero lengths.")
    paired = [
        float(left) / float(right)
        for left, right in zip(numerator, denominator)
        if left is not None and right is not None and float(right) > 0.0
    ]
    capped_num = [
        float(value) if value is not None else float(limit)
        for value, limit in zip(numerator, numerator_limits)
    ]
    capped_den = [
        float(value) if value is not None else float(limit)
        for value, limit in zip(denominator, denominator_limits)
    ]
    capped_ratios = [left / right for left, right in zip(capped_num, capped_den) if right > 0.0]
    return {
        "num_seeds": len(numerator),
        "paired_hit_count": len(paired),
        "paired_hit_ratio_median": statistics.median(paired) if paired else None,
        "paired_hit_ratio_mean": statistics.mean(paired) if paired else None,
        "capped_paired_ratio_mean": statistics.mean(capped_ratios),
        "ratio_of_capped_means": statistics.mean(capped_num) / statistics.mean(capped_den),
        "ratios_are_conditional": len(paired) != len(numerator),
    }


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def trend_statistics(epsilons: Sequence[float], ratios: Sequence[float]) -> Dict[str, float]:
    if len(epsilons) != len(ratios) or len(epsilons) < 2:
        raise ValueError("Trend statistics require paired epsilon/ratio values.")
    x = [math.log(1.0 / float(value)) for value in epsilons]
    y = [math.log(float(value)) for value in ratios]
    if any(not math.isfinite(value) for value in x + y):
        raise ValueError("Epsilons and ratios must be positive and finite.")
    x_mean, y_mean = statistics.mean(x), statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0.0:
        raise ValueError("Epsilons must not all be equal.")
    slope = sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y)) / denominator
    rx, ry = _ranks(x), _ranks(y)
    rx_mean, ry_mean = statistics.mean(rx), statistics.mean(ry)
    covariance = sum((left - rx_mean) * (right - ry_mean) for left, right in zip(rx, ry))
    scale = math.sqrt(
        sum((value - rx_mean) ** 2 for value in rx)
        * sum((value - ry_mean) ** 2 for value in ry)
    )
    return {"log_log_slope": slope, "spearman_rho": covariance / scale if scale else 0.0}
