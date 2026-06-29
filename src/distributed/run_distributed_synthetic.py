import argparse
import json
import random
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import trange

# Allow importing from repo root when running:
# python src/distributed/run_distributed_synthetic.py --config ...
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.synthetic.run_synthetic import (  # noqa: E402
    SyntheticMaxSinL1,
    seed_all,
    get_device,
    load_yaml,
    save_yaml,
    sample_ball,
    project_l2_ball,
    smoothed_oracle as global_smoothed_oracle,
)


# ============================================================
# Distributed data partition and local oracles
# ============================================================

def make_worker_shards(n_data: int, n_workers: int, device: str, seed: int, shuffle: bool = True) -> List[torch.Tensor]:
    if shuffle:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        perm = torch.randperm(n_data, generator=g)
    else:
        perm = torch.arange(n_data)

    return [s.to(device) for s in torch.tensor_split(perm, n_workers)]


def grad_at_indices(problem: SyntheticMaxSinL1, x: torch.Tensor, idx_pool: torch.Tensor, batch_size: int) -> torch.Tensor:
    pos = torch.randint(0, idx_pool.numel(), (batch_size,), device=problem.device)
    idx = idx_pool[pos]

    x_var = x.detach().clone().requires_grad_(True)
    loss = problem.loss(x_var, idx)
    grad = torch.autograd.grad(loss, x_var)[0]
    return grad.detach()


def local_smoothed_oracle(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    delta: float,
    smooth_B: int,
    local_data_B: int,
    idx_pool: torch.Tensor,
) -> torch.Tensor:
    grads = []
    for _ in range(smooth_B):
        u = sample_ball(1, problem.d, problem.device).squeeze(0)
        grads.append(grad_at_indices(problem, x + delta * u, idx_pool, local_data_B))
    return torch.stack(grads, dim=0).mean(dim=0)


def distributed_smoothed_oracle(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    delta: float,
    smooth_B: int,
    local_data_B: int,
    worker_shards: List[torch.Tensor],
) -> torch.Tensor:
    local_grads = [
        local_smoothed_oracle(problem, x, delta, smooth_B, local_data_B, shard)
        for shard in worker_shards
    ]
    return torch.stack(local_grads, dim=0).mean(dim=0)


def get_local_batch_size(cfg: Dict[str, Any], n_workers: int) -> int:
    split_mode = cfg["distributed"].get("split_mode", "total_batch_fixed")
    oracle_cfg = cfg["oracle"]

    if split_mode == "total_batch_fixed":
        total_B = int(oracle_cfg["data_B_total"])
        if total_B % n_workers != 0:
            raise ValueError(
                f"data_B_total must be divisible by n_workers under total_batch_fixed. "
                f"Got data_B_total={total_B}, n_workers={n_workers}."
            )
        return total_B // n_workers

    if split_mode == "per_worker_batch_fixed":
        return int(oracle_cfg.get("data_B_per_worker", oracle_cfg.get("data_B_total", 64)))

    raise ValueError(f"Unknown split_mode: {split_mode}")


def get_train_sfo_per_oracle(cfg: Dict[str, Any], n_workers: int) -> Tuple[int, int]:
    smooth_B = int(cfg["oracle"]["smooth_B"])
    local_B = get_local_batch_size(cfg, n_workers)
    per_worker = smooth_B * local_B
    total = n_workers * per_worker
    return total, per_worker


# ============================================================
# Evaluation
# ============================================================

def eval_objective(problem: SyntheticMaxSinL1, x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(problem.loss(x).item())


def eval_stat_proxy(problem: SyntheticMaxSinL1, x: torch.Tensor, cfg: Dict[str, Any]) -> float:
    ocfg = cfg["oracle"]
    g = global_smoothed_oracle(
        problem,
        x,
        float(ocfg["delta"]),
        int(ocfg["eval_smooth_B"]),
        int(ocfg["eval_data_B"]),
    )
    return float(g.norm().item())


def evaluate(problem: SyntheticMaxSinL1, x: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "objective": eval_objective(problem, x),
        "stat_proxy": eval_stat_proxy(problem, x, cfg),
    }


def get_eval_sfo_calls(cfg: Dict[str, Any]) -> int:
    ocfg = cfg["oracle"]
    return int(ocfg["eval_smooth_B"]) * int(ocfg["eval_data_B"])


# ============================================================
# Distributed NOG
# ============================================================

def run_distributed_nog(
    problem: SyntheticMaxSinL1,
    x0: torch.Tensor,
    cfg: Dict[str, Any],
    seed: int,
    n_workers: int,
) -> List[Dict[str, Any]]:
    seed_all(seed)

    tcfg, ocfg, ncfg, dcfg = cfg["train"], cfg["oracle"], cfg["nog"], cfg["distributed"]

    rounds = int(tcfg["rounds"])
    eval_every = int(tcfg["eval_every"])
    delta = float(ocfg["delta"])
    smooth_B = int(ocfg["smooth_B"])
    M = int(ncfg["M"])
    eta = float(ncfg["eta"])
    D = delta / M

    assert rounds % M == 0, f"NOG requires rounds divisible by M. Got rounds={rounds}, M={M}."

    local_B = get_local_batch_size(cfg, n_workers)
    total_sfo_per_oracle, per_worker_sfo_per_oracle = get_train_sfo_per_oracle(cfg, n_workers)

    worker_shards = make_worker_shards(
        n_data=problem.n,
        n_workers=n_workers,
        device=problem.device,
        seed=seed,
        shuffle=bool(dcfg.get("shuffle_partitions", True)),
    )

    x = x0.detach().clone()
    Delta = torch.zeros_like(x)

    # Paper initialization: x0 = y0 = y_{-1}; two independent distributed oracle calls.
    g_tm2 = distributed_smoothed_oracle(problem, x, delta, smooth_B, local_B, worker_shards)
    g_tm1 = distributed_smoothed_oracle(problem, x, delta, smooth_B, local_B, worker_shards)

    total_work = 2 * total_sfo_per_oracle
    per_worker_work = 2 * per_worker_sfo_per_oracle
    eval_work = 0

    rows = []
    start = time.time()

    ys_block = []
    oracle_block = []
    last_eval_round = 0
    label = f"NOG-m{n_workers}"

    for t in trange(1, rounds + 1, desc=f"{label} seed={seed}"):
        Delta = project_l2_ball(Delta - 2.0 * eta * g_tm1 + eta * g_tm2, radius=D)

        s = torch.rand((), device=problem.device)
        y = x + s * Delta
        x = (x + Delta).detach()

        ys_block.append(y.detach().clone())

        g_new = distributed_smoothed_oracle(problem, y, delta, smooth_B, local_B, worker_shards)
        oracle_block.append(g_new.detach().clone())

        total_work += total_sfo_per_oracle
        per_worker_work += per_worker_sfo_per_oracle
        g_tm2, g_tm1 = g_tm1, g_new

        if t % M == 0:
            block_id = t // M
            y_bar = torch.stack(ys_block, dim=0).mean(dim=0)
            block_oracle_norm = float(torch.stack(oracle_block, dim=0).mean(dim=0).norm().item())

            should_eval = block_id == 1 or t == rounds or (t - last_eval_round) >= eval_every
            if should_eval:
                metrics = evaluate(problem, y_bar, cfg)
                eval_work += get_eval_sfo_calls(cfg)
                last_eval_round = t

                rows.append({
                    "method": label,
                    "base_method": "DistributedNOG",
                    "seed": seed,
                    "round": t,
                    "communication_round": t,
                    "block_id": block_id,
                    "worker_count": n_workers,
                    "local_data_B": local_B,
                    "split_mode": dcfg.get("split_mode", "total_batch_fixed"),
                    "work": total_work,
                    "total_work": total_work,
                    "per_worker_work": per_worker_work,
                    "eval_work": eval_work,
                    "total_sfo_per_oracle": total_sfo_per_oracle,
                    "per_worker_sfo_per_oracle": per_worker_sfo_per_oracle,
                    "time_sec": time.time() - start,
                    "d": problem.d,
                    "delta": delta,
                    "M": M,
                    "lr_or_eta": eta,
                    "eval_point": "y_bar",
                    "objective": metrics["objective"],
                    "stat_proxy": metrics["stat_proxy"],
                    "block_oracle_norm": block_oracle_norm,
                })

            ys_block = []
            oracle_block = []

    return rows


# ============================================================
# Plotting
# ============================================================

def plot_mean_curve(grouped: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure()
    for method in grouped["method"].unique():
        sub = grouped[grouped["method"] == method].sort_values(x_col)
        x = sub[x_col].to_numpy()
        y = sub[f"{y_col}_mean"].to_numpy()
        y_std = sub[f"{y_col}_std"].fillna(0.0).to_numpy()
        plt.plot(x, y, label=method)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    grouped = df.groupby(["method", "round"], as_index=False).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        stat_proxy_mean=("stat_proxy", "mean"),
        stat_proxy_std=("stat_proxy", "std"),
        total_work_mean=("total_work", "mean"),
        per_worker_work_mean=("per_worker_work", "mean"),
    )

    plot_mean_curve(
        grouped, "round", "objective",
        out_dir / "objective_vs_communication_rounds.png",
        "Objective vs communication rounds", "Communication rounds / depth", "Objective value",
    )
    plot_mean_curve(
        grouped, "round", "stat_proxy",
        out_dir / "stat_proxy_vs_communication_rounds.png",
        "Stationarity proxy vs communication rounds", "Communication rounds / depth", "Smoothed gradient norm proxy",
    )

    grouped_total = grouped.rename(columns={"total_work_mean": "total_work"})
    plot_mean_curve(
        grouped_total, "total_work", "stat_proxy",
        out_dir / "stat_proxy_vs_total_work.png",
        "Stationarity proxy vs total work", "Total SFO calls across workers", "Smoothed gradient norm proxy",
    )

    grouped_local = grouped.rename(columns={"per_worker_work_mean": "per_worker_work"})
    plot_mean_curve(
        grouped_local, "per_worker_work", "stat_proxy",
        out_dir / "stat_proxy_vs_per_worker_work.png",
        "Stationarity proxy vs per-worker work", "Per-worker SFO calls", "Smoothed gradient norm proxy",
    )


# ============================================================
# Threshold summaries
# ============================================================

def compute_threshold_summaries(df: pd.DataFrame, thresholds: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for threshold in thresholds:
        for (worker_count, seed), sub in df.sort_values("round").groupby(["worker_count", "seed"]):
            hit = sub[sub["stat_proxy"] <= threshold]
            last = sub.iloc[-1]
            if len(hit) > 0:
                first = hit.iloc[0]
                rows.append({
                    "threshold": threshold,
                    "worker_count": int(worker_count),
                    "method": first["method"],
                    "seed": seed,
                    "hit": True,
                    "first_hit_round": float(first["round"]),
                    "first_hit_total_work": float(first["total_work"]),
                    "first_hit_per_worker_work": float(first["per_worker_work"]),
                    "first_hit_time_sec": float(first["time_sec"]),
                    "first_hit_stat_proxy": float(first["stat_proxy"]),
                    "first_hit_objective": float(first["objective"]),
                    "final_stat_proxy": float(last["stat_proxy"]),
                    "final_objective": float(last["objective"]),
                })
            else:
                rows.append({
                    "threshold": threshold,
                    "worker_count": int(worker_count),
                    "method": last["method"],
                    "seed": seed,
                    "hit": False,
                    "first_hit_round": np.nan,
                    "first_hit_total_work": np.nan,
                    "first_hit_per_worker_work": np.nan,
                    "first_hit_time_sec": np.nan,
                    "first_hit_stat_proxy": np.nan,
                    "first_hit_objective": np.nan,
                    "final_stat_proxy": float(last["stat_proxy"]),
                    "final_objective": float(last["objective"]),
                })

    per_seed = pd.DataFrame(rows)
    aggregate = per_seed.groupby(["threshold", "worker_count", "method"], as_index=False).agg(
        hit_rate=("hit", "mean"),
        first_hit_round_mean=("first_hit_round", "mean"),
        first_hit_round_std=("first_hit_round", "std"),
        first_hit_total_work_mean=("first_hit_total_work", "mean"),
        first_hit_total_work_std=("first_hit_total_work", "std"),
        first_hit_per_worker_work_mean=("first_hit_per_worker_work", "mean"),
        first_hit_per_worker_work_std=("first_hit_per_worker_work", "std"),
        first_hit_time_sec_mean=("first_hit_time_sec", "mean"),
        first_hit_time_sec_std=("first_hit_time_sec", "std"),
        final_stat_proxy_mean=("final_stat_proxy", "mean"),
        final_objective_mean=("final_objective", "mean"),
    )
    return per_seed, aggregate


def compute_worker_speedup(threshold_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold, sub in threshold_summary.groupby("threshold"):
        base_rows = sub[sub["worker_count"] == 1]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]

        for _, row in sub.iterrows():
            m = int(row["worker_count"])
            rows.append({
                "threshold": threshold,
                "worker_count": m,
                "hit_rate": row["hit_rate"],
                "first_hit_round_mean": row["first_hit_round_mean"],
                "round_ratio_m_over_1": row["first_hit_round_mean"] / base["first_hit_round_mean"]
                    if pd.notna(row["first_hit_round_mean"]) and pd.notna(base["first_hit_round_mean"]) and base["first_hit_round_mean"] > 0 else np.nan,
                "first_hit_total_work_mean": row["first_hit_total_work_mean"],
                "total_work_ratio_m_over_1": row["first_hit_total_work_mean"] / base["first_hit_total_work_mean"]
                    if pd.notna(row["first_hit_total_work_mean"]) and pd.notna(base["first_hit_total_work_mean"]) and base["first_hit_total_work_mean"] > 0 else np.nan,
                "first_hit_per_worker_work_mean": row["first_hit_per_worker_work_mean"],
                "per_worker_work_ratio_m_over_1": row["first_hit_per_worker_work_mean"] / base["first_hit_per_worker_work_mean"]
                    if pd.notna(row["first_hit_per_worker_work_mean"]) and pd.notna(base["first_hit_per_worker_work_mean"]) and base["first_hit_per_worker_work_mean"] > 0 else np.nan,
                "ideal_per_worker_ratio": 1.0 / m,
            })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    if args.name is not None:
        cfg["run"]["name"] = args.name
    if args.d is not None:
        cfg["problem"]["d"] = args.d
    if args.rounds is not None:
        cfg["train"]["rounds"] = args.rounds
    if args.seeds is not None:
        cfg["run"]["seeds"] = [int(x) for x in args.seeds.split(",")]
    return cfg


def build_problem(cfg: Dict[str, Any], device: str) -> SyntheticMaxSinL1:
    pcfg = cfg["problem"]
    return SyntheticMaxSinL1(
        d=int(pcfg["d"]),
        n_data=int(pcfg["n_data"]),
        R=int(pcfg["R"]),
        lam=float(pcfg["lam"]),
        device=device,
    )


def main() -> None:
    args = parse_args()

    cfg = apply_overrides(load_yaml(args.config), args)
    device = get_device(cfg["run"].get("device", "auto"))
    cfg["run"]["device_used"] = device

    out_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config_used.yaml")

    workers = [int(w) for w in cfg["distributed"]["workers"]]

    print("=" * 80)
    print("Simulated Distributed NOG Experiment")
    print("=" * 80)
    print(f"config     : {args.config}")
    print(f"run name   : {cfg['run']['name']}")
    print(f"out dir    : {out_dir}")
    print(f"device     : {device}")
    print(f"seeds      : {cfg['run']['seeds']}")
    print(f"d          : {cfg['problem']['d']}")
    print(f"rounds     : {cfg['train']['rounds']}")
    print(f"M, eta     : {cfg['nog']['M']}, {cfg['nog']['eta']}")
    print(f"workers    : {workers}")
    print(f"split mode : {cfg['distributed'].get('split_mode', 'total_batch_fixed')}")
    print("=" * 80)

    all_rows = []
    for seed in cfg["run"]["seeds"]:
        seed_all(seed)
        problem = build_problem(cfg, device)
        x0 = 0.1 * torch.zeros(problem.d, device=device)

        for m in workers:
            print(f"[seed={seed}] running workers={m}, local_data_B={get_local_batch_size(cfg, m)}")
            rows = run_distributed_nog(
                problem=problem,
                x0=x0,
                cfg=cfg,
                seed=seed + 10000 + 1000 * m,
                n_workers=m,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)
    plot_curves(df, out_dir)

    final_df = df.sort_values("round").groupby(["worker_count", "seed"]).tail(1)
    final_summary = final_df.groupby(["worker_count", "method"], as_index=False).agg(
        final_objective_mean=("objective", "mean"),
        final_objective_std=("objective", "std"),
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_std=("stat_proxy", "std"),
        final_total_work_mean=("total_work", "mean"),
        final_per_worker_work_mean=("per_worker_work", "mean"),
        final_eval_work_mean=("eval_work", "mean"),
        final_time_sec_mean=("time_sec", "mean"),
    )
    final_summary.to_csv(out_dir / "final_summary_by_workers.csv", index=False)

    summary = {
        "num_rows": int(len(df)),
        "workers": workers,
        "seeds": cfg["run"]["seeds"],
        "note": "One-process simulation. Use communication_round, total_work, and per_worker_work as evidence; wall-clock time is not the main metric.",
        "work_accounting": {
            "work": "alias of total_work",
            "total_work": "total training SFO calls across all simulated workers",
            "per_worker_work": "training SFO calls on each local worker",
            "eval_work": "centralized evaluation SFO calls; not included in total_work",
        },
        "final_summary_by_workers": final_summary.to_dict(orient="records"),
    }

    thresholds = cfg.get("metrics", {}).get("thresholds", [])
    if thresholds:
        per_seed, threshold_summary = compute_threshold_summaries(df, thresholds)
        per_seed.to_csv(out_dir / "threshold_per_seed.csv", index=False)
        threshold_summary.to_csv(out_dir / "threshold_summary.csv", index=False)

        worker_speedup = compute_worker_speedup(threshold_summary)
        worker_speedup.to_csv(out_dir / "worker_speedup_vs_m1.csv", index=False)
        summary["worker_speedup_vs_m1"] = worker_speedup.to_dict(orient="records")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"Saved results to        : {out_dir / 'results.csv'}")
    print(f"Saved final summary to  : {out_dir / 'final_summary_by_workers.csv'}")
    if (out_dir / "worker_speedup_vs_m1.csv").exists():
        print(f"Saved worker speedup to : {out_dir / 'worker_speedup_vs_m1.csv'}")
    print(f"Saved figures to        : {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
