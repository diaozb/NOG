import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import trange
import copy
import itertools


# ============================================================
# Basic utils
# ============================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def sample_ball(num: int, dim: int, device: str) -> torch.Tensor:
    z = torch.randn(num, dim, device=device)
    z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = torch.rand(num, 1, device=device).pow(1.0 / dim)
    return r * z


def project_l2_ball(v: torch.Tensor, radius: float) -> torch.Tensor:
    norm = v.norm()
    if float(norm.item()) <= radius:
        return v
    return radius * v / norm.clamp_min(1e-12)


# ============================================================
# Synthetic problem
# ============================================================

class SyntheticMaxSinL1:
    """
    F(x; xi) = max_r sin(a_{xi,r}^T x + b_{xi,r}) + lambda * ||x||_1

    This toy problem is nonsmooth and nonconvex:
    - max creates nonsmoothness
    - sin creates nonconvexity
    - L1 creates additional nonsmoothness
    """

    def __init__(
        self,
        d: int,
        n_data: int,
        R: int,
        lam: float,
        device: str,
    ):
        self.d = d
        self.n = n_data
        self.R = R
        self.lam = lam
        self.device = device

        self.A = torch.randn(n_data, R, d, device=device) / math.sqrt(d)
        self.b = 2.0 * math.pi * torch.rand(n_data, R, device=device)

    def loss(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if idx is None:
            A = self.A
            b = self.b
        else:
            A = self.A[idx]
            b = self.b[idx]

        vals = torch.sin(torch.einsum("nrd,d->nr", A, x) + b)
        max_vals = vals.max(dim=1).values
        return max_vals.mean() + self.lam * x.abs().sum()


# ============================================================
# First-order smoothed oracle
# ============================================================

def grad_at(problem: SyntheticMaxSinL1, x: torch.Tensor, batch_size: int) -> torch.Tensor:
    idx = torch.randint(problem.n, (batch_size,), device=problem.device)

    x_var = x.detach().clone().requires_grad_(True)
    loss = problem.loss(x_var, idx)
    grad = torch.autograd.grad(loss, x_var)[0]

    return grad.detach()


def smoothed_oracle(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    delta: float,
    smooth_B: int,
    data_B: int,
) -> torch.Tensor:
    grads = []

    for _ in range(smooth_B):
        u = sample_ball(1, problem.d, problem.device).squeeze(0)
        g = grad_at(problem, x + delta * u, data_B)
        grads.append(g)

    return torch.stack(grads, dim=0).mean(dim=0)


# ============================================================
# Evaluation
# ============================================================

def eval_objective(problem: SyntheticMaxSinL1, x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(problem.loss(x).item())


def eval_smoothed_grad_norm(
    problem: SyntheticMaxSinL1,
    x: torch.Tensor,
    cfg: Dict[str, Any],
) -> float:
    oracle_cfg = cfg["oracle"]

    g = smoothed_oracle(
        problem=problem,
        x=x,
        delta=oracle_cfg["delta"],
        smooth_B=oracle_cfg["eval_smooth_B"],
        data_B=oracle_cfg["eval_data_B"],
    )

    return float(g.norm().item())


def evaluate(problem: SyntheticMaxSinL1, x: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "objective": eval_objective(problem, x),
        "stat_proxy": eval_smoothed_grad_norm(problem, x, cfg),
    }


def get_eval_sfo_calls(cfg: Dict[str, Any]) -> int:
    """
    Number of SFO calls used by one stationarity-proxy evaluation.

    We keep this separate from `work`, because the paper's work/depth complexity
    refers to training oracle calls. Evaluation oracle calls are logged as
    `eval_work` for transparency, but are not added to the main `work` column.
    """
    oracle_cfg = cfg["oracle"]
    return int(oracle_cfg["eval_smooth_B"]) * int(oracle_cfg["eval_data_B"])


# ============================================================
# Methods
# ============================================================

def run_nog(
    problem: SyntheticMaxSinL1,
    x0: torch.Tensor,
    cfg: Dict[str, Any],
    seed: int,
    wandb_run=None,
) -> List[Dict[str, Any]]:
    """
    NOG with block-average evaluation.

    The primary logged metrics `objective` and `stat_proxy` are evaluated at y_bar,
    where y_bar is the average of y_t over one full block of length M.

    Extra metrics are also saved:
    - objective_x, stat_proxy_x: evaluation at current x_t
    - objective_ybar, stat_proxy_ybar: evaluation at block average y_bar
    - block_oracle_norm: || mean_{t in block} O(y_t) ||, a cheap theorem-aligned proxy
      based on the oracle estimates already computed during the block.
    """
    seed_all(seed)

    train_cfg = cfg["train"]
    oracle_cfg = cfg["oracle"]
    nog_cfg = cfg["nog"]

    rounds = int(train_cfg["rounds"])
    eval_every = int(train_cfg["eval_every"])

    delta = float(oracle_cfg["delta"])
    smooth_B = int(oracle_cfg["smooth_B"])
    data_B = int(oracle_cfg["data_B"])

    M = int(nog_cfg["M"])
    eta = float(nog_cfg["eta"])
    D = delta / M

    assert rounds % M == 0, (
        f"For NOG, train.rounds should be K * M. "
        f"Got rounds={rounds}, M={M}, rounds % M={rounds % M}."
    )

    x = x0.detach().clone()
    Delta = torch.zeros_like(x)

    # Paper initialization: x0 = y0 = y_{-1}; here both oracle calls are sampled independently.
    g_tm2 = smoothed_oracle(problem, x, delta, smooth_B, data_B)
    g_tm1 = smoothed_oracle(problem, x, delta, smooth_B, data_B)

    # Main work column counts training SFO calls only.
    # The two initialization oracle estimates are part of training computation.
    work = 2 * smooth_B * data_B

    # Evaluation work is logged separately and is not included in `work`.
    eval_work = 0
    rows = []
    start = time.time()

    ys_block: List[torch.Tensor] = []
    oracle_block: List[torch.Tensor] = []
    last_eval_round = 0

    for t in trange(1, rounds + 1, desc=f"NOG seed={seed}"):
        Delta = project_l2_ball(
            Delta - 2.0 * eta * g_tm1 + eta * g_tm2,
            radius=D,
        )

        s = torch.rand((), device=problem.device)
        y = x + s * Delta
        x = (x + Delta).detach()

        ys_block.append(y.detach().clone())

        g_new = smoothed_oracle(problem, y, delta, smooth_B, data_B)
        oracle_block.append(g_new.detach().clone())
        work += smooth_B * data_B

        g_tm2, g_tm1 = g_tm1, g_new

        # Only evaluate after a complete block, because y_bar is defined block-wise.
        if t % M == 0:
            block_id = t // M
            y_bar = torch.stack(ys_block, dim=0).mean(dim=0)
            block_oracle_mean = torch.stack(oracle_block, dim=0).mean(dim=0)
            block_oracle_norm = float(block_oracle_mean.norm().item())

            should_eval = (
                block_id == 1
                or t == rounds
                or (t - last_eval_round) >= eval_every
            )

            if should_eval:
                metrics_x = evaluate(problem, x, cfg)
                metrics_ybar = evaluate(problem, y_bar, cfg)
                # Two stationarity evaluations are performed here: one at x and one at y_bar.
                eval_work += 2 * get_eval_sfo_calls(cfg)
                last_eval_round = t

                row = {
                    "method": "NOG",
                    "seed": seed,
                    "round": t,
                    "block_id": block_id,
                    "work": work,
                    "eval_work": eval_work,
                    "communication_round": t,
                    "time_sec": time.time() - start,
                    "d": problem.d,
                    "delta": delta,
                    "M": M,
                    "lr_or_eta": eta,
                    "eval_point": "y_bar",

                    # Primary columns used by plotting and summaries.
                    "objective": metrics_ybar["objective"],
                    "stat_proxy": metrics_ybar["stat_proxy"],

                    # Extra columns for checking theory alignment.
                    "objective_x": metrics_x["objective"],
                    "stat_proxy_x": metrics_x["stat_proxy"],
                    "objective_ybar": metrics_ybar["objective"],
                    "stat_proxy_ybar": metrics_ybar["stat_proxy"],
                    "block_oracle_norm": block_oracle_norm,
                }

                rows.append(row)
                log_to_wandb(wandb_run, row)

            ys_block = []
            oracle_block = []

    return rows


def run_smoothed_sgd(
    problem: SyntheticMaxSinL1,
    x0: torch.Tensor,
    cfg: Dict[str, Any],
    seed: int,
    wandb_run=None,
) -> List[Dict[str, Any]]:
    seed_all(seed)

    train_cfg = cfg["train"]
    oracle_cfg = cfg["oracle"]
    ssgd_cfg = cfg["ssgd"]

    rounds = int(train_cfg["rounds"])
    eval_every = int(train_cfg["eval_every"])

    delta = float(oracle_cfg["delta"])
    smooth_B = int(oracle_cfg["smooth_B"])
    data_B = int(oracle_cfg["data_B"])

    lr = float(ssgd_cfg["lr"])

    x = x0.detach().clone()
    # Main work column counts training SFO calls only.
    work = 0

    # Evaluation work is logged separately and is not included in `work`.
    eval_work = 0
    rows = []
    start = time.time()

    for t in trange(1, rounds + 1, desc=f"SmoothedSGD seed={seed}"):
        g = smoothed_oracle(problem, x, delta, smooth_B, data_B)
        x = (x - lr * g).detach()
        work += smooth_B * data_B

        if t % eval_every == 0 or t == 1 or t == rounds:
            metrics = evaluate(problem, x, cfg)
            # One stationarity evaluation is performed here.
            eval_work += get_eval_sfo_calls(cfg)

            row = {
                "method": "SmoothedSGD",
                "seed": seed,
                "round": t,
                "block_id": np.nan,
                "work": work,
                "eval_work": eval_work,
                "communication_round": t,
                "time_sec": time.time() - start,
                "d": problem.d,
                "delta": delta,
                "M": np.nan,
                "lr_or_eta": lr,
                "eval_point": "x",

                # Primary columns.
                "objective": metrics["objective"],
                "stat_proxy": metrics["stat_proxy"],

                # Extra columns, kept consistent with NOG output schema.
                "objective_x": metrics["objective"],
                "stat_proxy_x": metrics["stat_proxy"],
                "objective_ybar": np.nan,
                "stat_proxy_ybar": np.nan,
                "block_oracle_norm": np.nan,
            }

            rows.append(row)
            log_to_wandb(wandb_run, row)

    return rows


# ============================================================
# Plotting
# ============================================================

def plot_mean_curve(
    grouped: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
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


def plot_nog_x_vs_ybar(df: pd.DataFrame, out_dir: Path) -> None:
    nog = df[df["method"] == "NOG"].copy()
    if nog.empty:
        return

    grouped = nog.groupby("round", as_index=False).agg(
        stat_proxy_x_mean=("stat_proxy_x", "mean"),
        stat_proxy_x_std=("stat_proxy_x", "std"),
        stat_proxy_ybar_mean=("stat_proxy_ybar", "mean"),
        stat_proxy_ybar_std=("stat_proxy_ybar", "std"),
        objective_x_mean=("objective_x", "mean"),
        objective_x_std=("objective_x", "std"),
        objective_ybar_mean=("objective_ybar", "mean"),
        objective_ybar_std=("objective_ybar", "std"),
    )

    plt.figure()
    for label, mean_col, std_col in [
        ("NOG current x", "stat_proxy_x_mean", "stat_proxy_x_std"),
        ("NOG block average y_bar", "stat_proxy_ybar_mean", "stat_proxy_ybar_std"),
    ]:
        x = grouped["round"].to_numpy()
        y = grouped[mean_col].to_numpy()
        y_std = grouped[std_col].fillna(0.0).to_numpy()
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    plt.xlabel("Rounds / depth")
    plt.ylabel("Smoothed gradient norm proxy")
    plt.title("NOG: current x vs block average y_bar")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nog_x_vs_ybar_stat_proxy.png", dpi=200)
    plt.close()

    plt.figure()
    for label, mean_col, std_col in [
        ("NOG current x", "objective_x_mean", "objective_x_std"),
        ("NOG block average y_bar", "objective_ybar_mean", "objective_ybar_std"),
    ]:
        x = grouped["round"].to_numpy()
        y = grouped[mean_col].to_numpy()
        y_std = grouped[std_col].fillna(0.0).to_numpy()
        plt.plot(x, y, label=label)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    plt.xlabel("Rounds / depth")
    plt.ylabel("Objective value")
    plt.title("NOG: current x vs block average y_bar")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nog_x_vs_ybar_objective.png", dpi=200)
    plt.close()


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_round = df.groupby(["method", "round"], as_index=False).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        stat_proxy_mean=("stat_proxy", "mean"),
        stat_proxy_std=("stat_proxy", "std"),
        work_mean=("work", "mean"),
        time_sec_mean=("time_sec", "mean"),
    )

    plot_mean_curve(
        grouped=grouped_round,
        x_col="round",
        y_col="objective",
        out_path=out_dir / "objective_vs_rounds.png",
        title="Objective vs rounds",
        xlabel="Rounds / depth",
        ylabel="Objective value",
    )

    plot_mean_curve(
        grouped=grouped_round,
        x_col="round",
        y_col="stat_proxy",
        out_path=out_dir / "stat_proxy_vs_rounds.png",
        title="Stationarity proxy vs rounds",
        xlabel="Rounds / depth",
        ylabel="Smoothed gradient norm proxy",
    )

    grouped_work = grouped_round.rename(columns={"work_mean": "work"})

    plot_mean_curve(
        grouped=grouped_work,
        x_col="work",
        y_col="stat_proxy",
        out_path=out_dir / "stat_proxy_vs_work.png",
        title="Stationarity proxy vs work",
        xlabel="SFO calls / work",
        ylabel="Smoothed gradient norm proxy",
    )

    plot_nog_x_vs_ybar(df, out_dir)


# ============================================================
# wandb
# ============================================================

def init_wandb(cfg: Dict[str, Any], out_dir: Path):
    wandb_cfg = cfg.get("wandb", {})

    if not wandb_cfg.get("enabled", False):
        return None

    import wandb

    run = wandb.init(
        project=wandb_cfg.get("project", "nog-synthetic"),
        entity=wandb_cfg.get("entity") or None,
        name=cfg["run"]["name"],
        mode=wandb_cfg.get("mode", "online"),
        tags=wandb_cfg.get("tags", []),
        config=cfg,
    )

    try:
        wandb.save(str(out_dir / "config_used.yaml"))
    except Exception:
        pass

    return run


def log_to_wandb(wandb_run, row: Dict[str, Any]) -> None:
    if wandb_run is None:
        return

    method = row["method"]
    seed = row["seed"]
    prefix = f"{method}/seed_{seed}"

    payload = {
        "round": row["round"],
        "work": row["work"],
        "time_sec": row["time_sec"],
        "eval_work": row.get("eval_work", 0),
        "communication_round": row.get("communication_round", row["round"]),
        f"{prefix}/objective": row["objective"],
        f"{prefix}/stat_proxy": row["stat_proxy"],
        f"{prefix}/work": row["work"],
        f"{prefix}/time_sec": row["time_sec"],
        f"{prefix}/eval_work": row.get("eval_work", 0),
    }

    for key in [
        "objective_x",
        "stat_proxy_x",
        "objective_ybar",
        "stat_proxy_ybar",
        "block_oracle_norm",
    ]:
        val = row.get(key, None)
        if val is not None and not pd.isna(val):
            payload[f"{prefix}/{key}"] = val

    wandb_run.log(payload)


def finish_wandb(wandb_run, out_dir: Path, df: pd.DataFrame) -> None:
    if wandb_run is None:
        return

    import wandb

    try:
        wandb_run.log({"results_table": wandb.Table(dataframe=df)})
    except Exception as exc:
        print(f"[wandb warning] failed to upload table: {exc}")

    for fname in [
        "objective_vs_rounds.png",
        "stat_proxy_vs_rounds.png",
        "stat_proxy_vs_work.png",
        "nog_x_vs_ybar_stat_proxy.png",
        "nog_x_vs_ybar_objective.png",
    ]:
        fpath = out_dir / fname
        if fpath.exists():
            wandb_run.log({f"figures/{fname}": wandb.Image(str(fpath))})

    artifact = wandb.Artifact(
        name=f"synthetic_results_{wandb_run.id}",
        type="experiment-results",
    )

    for path in out_dir.glob("*"):
        if path.is_file():
            artifact.add_file(str(path))

    wandb_run.log_artifact(artifact)

    final_df = df.sort_values("round").groupby(["method", "seed"]).tail(1)
    summary = final_df.groupby("method").agg(
        final_objective_mean=("objective", "mean"),
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_work_mean=("work", "mean"),
        final_eval_work_mean=("eval_work", "mean"),
        final_time_sec_mean=("time_sec", "mean"),
    )

    for method, row in summary.iterrows():
        wandb_run.summary[f"{method}/final_objective_mean"] = float(row["final_objective_mean"])
        wandb_run.summary[f"{method}/final_stat_proxy_mean"] = float(row["final_stat_proxy_mean"])
        wandb_run.summary[f"{method}/final_work_mean"] = float(row["final_work_mean"])
        if "final_eval_work_mean" in row:
            wandb_run.summary[f"{method}/final_eval_work_mean"] = float(row["final_eval_work_mean"])
        wandb_run.summary[f"{method}/final_time_sec_mean"] = float(row["final_time_sec_mean"])

    wandb_run.finish()


# ============================================================
# NOG hyperparameter sweep
# ============================================================

def as_list(x):
    if isinstance(x, list):
        return x
    return [x]


def make_nog_sweep_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand cfg['nog']['M'] and cfg['nog']['eta'] into a list of configs.

    Example:
      nog:
        M: [4, 8, 16]
        eta: [0.1, 0.3]

    gives 6 NOG configs.
    """
    M_list = as_list(cfg["nog"]["M"])
    eta_list = as_list(cfg["nog"]["eta"])

    sweep_cfgs = []

    for M, eta in itertools.product(M_list, eta_list):
        new_cfg = copy.deepcopy(cfg)
        new_cfg["nog"]["M"] = int(M)
        new_cfg["nog"]["eta"] = float(eta)

        new_cfg["sweep"] = {
            "M": int(M),
            "eta": float(eta),
            "grid_id": f"M{int(M)}_eta{float(eta):g}",
        }

        sweep_cfgs.append(new_cfg)

    return sweep_cfgs


def add_sweep_metadata(rows: List[Dict[str, Any]], sweep_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Add M/eta/grid_id columns to each row.
    """
    sweep_info = sweep_cfg.get("sweep", {})
    for row in rows:
        row["sweep_M"] = sweep_info.get("M", row.get("M"))
        row["sweep_eta"] = sweep_info.get("eta", row.get("lr_or_eta"))
        row["sweep_grid_id"] = sweep_info.get("grid_id", "default")
    return rows


def select_best_nog_config(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Select the best NOG config by final mean stat_proxy across seeds.

    Smaller stat_proxy is better.
    """
    nog_df = df[df["method"] == "NOG"].copy()

    if nog_df.empty:
        raise ValueError("No NOG results found, cannot select best M/eta.")

    final_nog = (
        nog_df.sort_values("round")
        .groupby(["sweep_grid_id", "seed"])
        .tail(1)
    )

    sweep_summary = (
        final_nog.groupby(["sweep_grid_id", "sweep_M", "sweep_eta"], as_index=False)
        .agg(
            final_stat_proxy_mean=("stat_proxy", "mean"),
            final_stat_proxy_std=("stat_proxy", "std"),
            final_objective_mean=("objective", "mean"),
            final_objective_std=("objective", "std"),
            final_work_mean=("work", "mean"),
            final_time_sec_mean=("time_sec", "mean"),
        )
        .sort_values("final_stat_proxy_mean", ascending=True)
    )

    best = sweep_summary.iloc[0].to_dict()

    return {
        "best_grid_id": best["sweep_grid_id"],
        "best_M": int(best["sweep_M"]),
        "best_eta": float(best["sweep_eta"]),
        "best_final_stat_proxy_mean": float(best["final_stat_proxy_mean"]),
        "best_final_objective_mean": float(best["final_objective_mean"]),
        "sweep_summary": sweep_summary,
    }


def filter_best_plot_df(df: pd.DataFrame, best_grid_id: str) -> pd.DataFrame:
    """
    Keep only:
      - NOG rows from best M/eta
      - SmoothedSGD rows
    """
    keep = (
        ((df["method"] == "NOG") & (df["sweep_grid_id"] == best_grid_id))
        | (df["method"] == "SmoothedSGD")
    )
    return df[keep].copy()


# ============================================================
# Threshold summary
# ============================================================

def compute_threshold_summaries(
    df: pd.DataFrame,
    thresholds: List[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each threshold, report the first round/work at which each method/seed
    reaches stat_proxy <= threshold.

    This table is useful because the paper's claim is about the amount of
    depth/work needed to reach a stationarity target, not only about final curves.
    """
    per_seed_rows = []

    for threshold in thresholds:
        for (method, seed), sub in df.sort_values("round").groupby(["method", "seed"]):
            hit = sub[sub["stat_proxy"] <= threshold]
            last = sub.iloc[-1]
            
            if len(hit) > 0:
                first = hit.iloc[0]
                per_seed_rows.append({
                    "threshold": threshold,
                    "method": method,
                    "seed": seed,
                    "hit": True,
                    "first_hit_round": float(first["round"]),
                    "first_hit_work": float(first["work"]),
                    "first_hit_eval_work": float(first.get("eval_work", 0.0)),
                    "first_hit_time_sec": float(first["time_sec"]),
                    "first_hit_stat_proxy": float(first["stat_proxy"]),
                    "first_hit_objective": float(first["objective"]),
                    "final_stat_proxy": float(last["stat_proxy"]),
                    "final_objective": float(last["objective"]),
                })
            else:
                last = sub.iloc[-1]
                per_seed_rows.append({
                    "threshold": threshold,
                    "method": method,
                    "seed": seed,
                    "hit": False,
                    "first_hit_round": np.nan,
                    "first_hit_work": np.nan,
                    "first_hit_eval_work": np.nan,
                    "first_hit_time_sec": np.nan,
                    "first_hit_stat_proxy": np.nan,
                    "first_hit_objective": np.nan,
                    "final_stat_proxy": float(last["stat_proxy"]),
                    "final_objective": float(last["objective"]),
                })

    per_seed = pd.DataFrame(per_seed_rows)

    aggregate = per_seed.groupby(["threshold", "method"], as_index=False).agg(
        hit_rate=("hit", "mean"),
        first_hit_round_mean=("first_hit_round", "mean"),
        first_hit_round_std=("first_hit_round", "std"),
        first_hit_work_mean=("first_hit_work", "mean"),
        first_hit_work_std=("first_hit_work", "std"),
        first_hit_time_sec_mean=("first_hit_time_sec", "mean"),
        first_hit_time_sec_std=("first_hit_time_sec", "std"),
        final_stat_proxy_mean=("final_stat_proxy", "mean"),
        final_objective_mean=("final_objective", "mean"),
    )

    return per_seed, aggregate

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

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)

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

    if args.use_wandb:
        cfg["wandb"]["enabled"] = True

    if args.wandb_project is not None:
        cfg["wandb"]["project"] = args.wandb_project

    if args.wandb_mode is not None:
        cfg["wandb"]["mode"] = args.wandb_mode

    return cfg


def build_problem(cfg: Dict[str, Any], device: str) -> SyntheticMaxSinL1:
    problem_cfg = cfg["problem"]

    return SyntheticMaxSinL1(
        d=problem_cfg["d"],
        n_data=problem_cfg["n_data"],
        R=problem_cfg["R"],
        lam=problem_cfg["lam"],
        device=device,
    )


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args)

    device = get_device(cfg["run"].get("device", "auto"))
    cfg["run"]["device_used"] = device

    out_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    save_yaml(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("Synthetic NOG Experiment")
    print("=" * 80)
    print(f"config      : {args.config}")
    print(f"run name    : {cfg['run']['name']}")
    print(f"out dir     : {out_dir}")
    print(f"device      : {device}")
    print(f"methods     : {cfg['methods']}")
    print(f"seeds       : {cfg['run']['seeds']}")
    print(f"d           : {cfg['problem']['d']}")
    print(f"rounds      : {cfg['train']['rounds']}")
    print(f"wandb       : {cfg['wandb']['enabled']}")
    print("=" * 80)

    wandb_run = init_wandb(cfg, out_dir)

    all_rows = []

    nog_sweep_cfgs = make_nog_sweep_configs(cfg)

    print("=" * 80)
    print("NOG sweep configs")
    print("=" * 80)
    for sweep_cfg in nog_sweep_cfgs:
        print(
            f"{sweep_cfg['sweep']['grid_id']}: "
            f"M={sweep_cfg['nog']['M']}, eta={sweep_cfg['nog']['eta']}"
        )
    print("=" * 80)

    for seed in cfg["run"]["seeds"]:
        seed_all(seed)

        # Same synthetic problem and same x0 for all methods/configs under this seed.
        problem = build_problem(cfg, device)
        x0 = 0.1 * torch.zeros(problem.d, device=device)

        if "NOG" in cfg["methods"]:
            for sweep_id, sweep_cfg in enumerate(nog_sweep_cfgs):
                M = int(sweep_cfg["nog"]["M"])
                rounds = int(sweep_cfg["train"]["rounds"])

                if rounds % M != 0:
                    raise ValueError(
                        f"NOG requires train.rounds divisible by M. "
                        f"Got rounds={rounds}, M={M} for grid "
                        f"{sweep_cfg['sweep']['grid_id']}."
                    )

                # Use comparable randomness across M/eta configs.
                method_seed = seed + 10000

                rows = run_nog(
                    problem=problem,
                    x0=x0,
                    cfg=sweep_cfg,
                    seed=method_seed,
                    wandb_run=wandb_run,
                )
                rows = add_sweep_metadata(rows, sweep_cfg)
                all_rows.extend(rows)

        if "SmoothedSGD" in cfg["methods"]:
            # Run SmoothedSGD only once, not once per M/eta.
            ssgd_cfg = copy.deepcopy(cfg)
            ssgd_cfg["sweep"] = {
                "M": np.nan,
                "eta": float(ssgd_cfg["ssgd"]["lr"]),
                "grid_id": "SmoothedSGD",
            }

            method_seed = seed + 20000

            rows = run_smoothed_sgd(
                problem=problem,
                x0=x0,
                cfg=ssgd_cfg,
                seed=method_seed,
                wandb_run=wandb_run,
            )
            rows = add_sweep_metadata(rows, ssgd_cfg)
            all_rows.extend(rows)


    df = pd.DataFrame(all_rows)

    all_csv_path = out_dir / "all_results.csv"
    df.to_csv(all_csv_path, index=False)

    best_info = select_best_nog_config(df)
    best_grid_id = best_info["best_grid_id"]

    sweep_summary = best_info["sweep_summary"]
    sweep_summary.to_csv(out_dir / "sweep_summary.csv", index=False)

    best_plot_df = filter_best_plot_df(df, best_grid_id)

    best_csv_path = out_dir / "best_results.csv"
    best_plot_df.to_csv(best_csv_path, index=False)

    best_cfg = copy.deepcopy(cfg)
    best_cfg["nog"]["M"] = best_info["best_M"]
    best_cfg["nog"]["eta"] = best_info["best_eta"]
    best_cfg["best_selection"] = {
        "criterion": "minimum final mean NOG stat_proxy across seeds",
        "best_grid_id": best_info["best_grid_id"],
        "best_M": best_info["best_M"],
        "best_eta": best_info["best_eta"],
        "best_final_stat_proxy_mean": best_info["best_final_stat_proxy_mean"],
        "best_final_objective_mean": best_info["best_final_objective_mean"],
    }

    save_yaml(best_cfg, out_dir / "best_config.yaml")

    # Only plot best NOG config plus SmoothedSGD.
    plot_curves(best_plot_df, out_dir)


    final_df = df.sort_values("round").groupby(["method", "seed"]).tail(1)
    summary_df = final_df.groupby("method").agg(
        final_objective_mean=("objective", "mean"),
        final_objective_std=("objective", "std"),
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_std=("stat_proxy", "std"),
        final_work_mean=("work", "mean"),
        final_eval_work_mean=("eval_work", "mean"),
        final_time_sec_mean=("time_sec", "mean"),
    )

    summary = {
        "num_all_rows": int(len(df)),
        "num_best_plot_rows": int(len(best_plot_df)),
        "methods": list(best_plot_df["method"].unique()),
        "seeds": cfg["run"]["seeds"],
        "primary_metric_note": "For NOG, objective/stat_proxy are evaluated at block-average y_bar.",
        "selection_criterion": "Choose M/eta with the smallest final mean NOG stat_proxy across seeds.",
        "best_M": best_info["best_M"],
        "best_eta": best_info["best_eta"],
        "best_grid_id": best_info["best_grid_id"],
        "best_final_stat_proxy_mean": best_info["best_final_stat_proxy_mean"],
        "best_final_objective_mean": best_info["best_final_objective_mean"],
        "final_by_method": summary_df.to_dict(),
    }


    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    thresholds = cfg.get("metrics", {}).get("thresholds", [])
    if thresholds:
        threshold_per_seed, threshold_summary = compute_threshold_summaries(best_plot_df, thresholds)
        threshold_per_seed.to_csv(out_dir / "threshold_per_seed.csv", index=False)
        threshold_summary.to_csv(out_dir / "threshold_summary.csv", index=False)


    finish_wandb(wandb_run, out_dir, df)

    print("=" * 80)
    print("Best NOG hyperparameters")
    print("=" * 80)
    print(f"best M      : {best_info['best_M']}")
    print(f"best eta    : {best_info['best_eta']}")
    print(f"best grid   : {best_info['best_grid_id']}")
    print(f"best stat   : {best_info['best_final_stat_proxy_mean']}")
    print(f"best obj    : {best_info['best_final_objective_mean']}")
    print("=" * 80)
    print(f"Saved all sweep results to : {all_csv_path}")
    print(f"Saved best results to      : {best_csv_path}")
    print(f"Saved sweep summary to     : {out_dir / 'sweep_summary.csv'}")
    print(f"Saved best config to       : {out_dir / 'best_config.yaml'}")
    print(f"Saved figures to           : {out_dir}")
    print("Done.")



if __name__ == "__main__":
    main()
