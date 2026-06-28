import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from tqdm import trange


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


def get_eval_sfo_calls(cfg: Dict[str, Any]) -> int:
    oracle_cfg = cfg["oracle"]
    return int(oracle_cfg["eval_smooth_B"]) * int(oracle_cfg["eval_data_B"])


# ============================================================
# CIFAR data
# ============================================================

class BatchIterator:
    def __init__(self, loader, device: str):
        self.loader = loader
        self.device = device
        self.iterator = iter(loader)

    def next(self):
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            images, labels = next(self.iterator)

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return images, labels


def build_cifar_loaders(cfg: Dict[str, Any], device: str, seed: int):
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for CIFAR experiments. "
            "Install it with: pip install torchvision"
        ) from exc

    data_cfg = cfg["data"]
    oracle_cfg = cfg["oracle"]

    root = data_cfg.get("root", "data")
    download = bool(data_cfg.get("download", True))
    num_workers = int(data_cfg.get("num_workers", 2))

    train_batch_size = int(oracle_cfg["data_B"])
    eval_batch_size = int(oracle_cfg["eval_data_B"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transform,
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )

    subset_train = data_cfg.get("subset_train", None)
    subset_test = data_cfg.get("subset_test", None)

    rng = np.random.default_rng(seed)

    if subset_train is not None:
        subset_train = int(subset_train)
        train_indices = rng.choice(len(train_set), size=subset_train, replace=False)
        train_set = Subset(train_set, train_indices)

    if subset_test is not None:
        subset_test = int(subset_test)
        test_indices = rng.choice(len(test_set), size=subset_test, replace=False)
        test_set = Subset(test_set, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    eval_train_loader = DataLoader(
        train_set,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    return train_loader, eval_train_loader, test_loader


# ============================================================
# Model
# ============================================================

class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10 smoke tests.

    It uses ReLU, so the training objective is nonconvex and nonsmooth.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(cfg: Dict[str, Any], device: str) -> nn.Module:
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "SmallCNN")

    if name != "SmallCNN":
        raise ValueError(f"Unsupported model: {name}. First version only supports SmallCNN.")

    model = SmallCNN(num_classes=int(model_cfg.get("num_classes", 10)))
    return model.to(device)


# ============================================================
# Parameter vector helpers
# ============================================================

def get_model_vector(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(
        [p.detach() for p in model.parameters()]
    ).detach()


def load_model_vector(model: nn.Module, theta: torch.Tensor) -> None:
    with torch.no_grad():
        torch.nn.utils.vector_to_parameters(theta.detach(), model.parameters())


def get_grad_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads, dim=0)


# ============================================================
# First-order smoothed oracle on model parameters
# ============================================================

def grad_at_theta(
    model: nn.Module,
    theta: torch.Tensor,
    batch: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    Compute mini-batch gradient at model parameter vector theta.
    """
    load_model_vector(model, theta)

    model.train()
    model.zero_grad(set_to_none=True)

    images, labels = batch
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad = get_grad_vector(model)
    return grad.detach(), float(loss.item())


def smoothed_oracle(
    model: nn.Module,
    theta: torch.Tensor,
    delta: float,
    smooth_B: int,
    batch_iter: BatchIterator,
) -> torch.Tensor:
    """
    Estimate grad of smoothed CIFAR objective:
        E_u,batch[ grad F(theta + delta*u; batch) ].
    """
    grads = []
    dim = theta.numel()
    device = str(theta.device)

    for _ in range(smooth_B):
        u = sample_ball(1, dim, device).squeeze(0)
        batch = batch_iter.next()
        g, _ = grad_at_theta(model, theta + delta * u, batch)
        grads.append(g)

    load_model_vector(model, theta)
    return torch.stack(grads, dim=0).mean(dim=0)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    theta: torch.Tensor,
    loader,
    device: str,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    load_model_vector(model, theta)

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="sum")

        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += int(labels.numel())

    if total_count == 0:
        return {"loss": float("nan"), "acc": float("nan")}

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
    }


def eval_smoothed_grad_norm(
    model: nn.Module,
    theta: torch.Tensor,
    cfg: Dict[str, Any],
    eval_batch_iter: BatchIterator,
) -> float:
    oracle_cfg = cfg["oracle"]

    g = smoothed_oracle(
        model=model,
        theta=theta,
        delta=float(oracle_cfg["delta"]),
        smooth_B=int(oracle_cfg["eval_smooth_B"]),
        batch_iter=eval_batch_iter,
    )

    load_model_vector(model, theta)
    return float(g.norm().item())


def evaluate(
    model: nn.Module,
    theta: torch.Tensor,
    cfg: Dict[str, Any],
    eval_train_loader,
    test_loader,
    eval_batch_iter: BatchIterator,
    device: str,
) -> Dict[str, float]:
    metrics_cfg = cfg.get("metrics", {})

    train_eval_batches = metrics_cfg.get("eval_train_batches", 10)
    test_eval_batches = metrics_cfg.get("eval_test_batches", None)

    train_metrics = eval_loss_acc(
        model=model,
        theta=theta,
        loader=eval_train_loader,
        device=device,
        max_batches=train_eval_batches,
    )

    test_metrics = eval_loss_acc(
        model=model,
        theta=theta,
        loader=test_loader,
        device=device,
        max_batches=test_eval_batches,
    )

    stat_proxy = eval_smoothed_grad_norm(
        model=model,
        theta=theta,
        cfg=cfg,
        eval_batch_iter=eval_batch_iter,
    )

    load_model_vector(model, theta)

    return {
        "objective": train_metrics["loss"],
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "stat_proxy": stat_proxy,
    }


# ============================================================
# Methods
# ============================================================

def run_nog(
    model: nn.Module,
    theta0: torch.Tensor,
    cfg: Dict[str, Any],
    train_batch_iter: BatchIterator,
    eval_batch_iter: BatchIterator,
    eval_train_loader,
    test_loader,
    seed: int,
    device: str,
) -> List[Dict[str, Any]]:
    seed_all(seed)

    train_cfg = cfg["train"]
    oracle_cfg = cfg["oracle"]
    nog_cfg = cfg["nog"]
    metrics_cfg = cfg.get("metrics", {})

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

    theta = theta0.detach().clone()
    Delta = torch.zeros_like(theta)

    g_tm2 = smoothed_oracle(model, theta, delta, smooth_B, train_batch_iter)
    g_tm1 = smoothed_oracle(model, theta, delta, smooth_B, train_batch_iter)

    work = 2 * smooth_B * data_B
    eval_work = 0
    rows = []
    start = time.time()

    y_sum = torch.zeros_like(theta)
    oracle_sum = torch.zeros_like(theta)
    last_eval_round = 0

    eval_current_x = bool(metrics_cfg.get("eval_current_x", False))

    for t in trange(1, rounds + 1, desc=f"CIFAR NOG seed={seed}"):
        Delta = project_l2_ball(
            Delta - 2.0 * eta * g_tm1 + eta * g_tm2,
            radius=D,
        )

        s = torch.rand((), device=theta.device)
        y = theta + s * Delta
        theta = (theta + Delta).detach()

        y_sum += y.detach()

        g_new = smoothed_oracle(model, y, delta, smooth_B, train_batch_iter)
        oracle_sum += g_new.detach()
        work += smooth_B * data_B

        g_tm2, g_tm1 = g_tm1, g_new

        if t % M == 0:
            block_id = t // M
            y_bar = y_sum / M
            block_oracle_norm = float((oracle_sum / M).norm().item())

            should_eval = (
                block_id == 1
                or t == rounds
                or (t - last_eval_round) >= eval_every
            )

            if should_eval:
                metrics_ybar = evaluate(
                    model=model,
                    theta=y_bar,
                    cfg=cfg,
                    eval_train_loader=eval_train_loader,
                    test_loader=test_loader,
                    eval_batch_iter=eval_batch_iter,
                    device=device,
                )
                eval_work += get_eval_sfo_calls(cfg)

                metrics_x = {}
                if eval_current_x:
                    metrics_x = evaluate(
                        model=model,
                        theta=theta,
                        cfg=cfg,
                        eval_train_loader=eval_train_loader,
                        test_loader=test_loader,
                        eval_batch_iter=eval_batch_iter,
                        device=device,
                    )
                    eval_work += get_eval_sfo_calls(cfg)

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
                    "delta": delta,
                    "M": M,
                    "lr_or_eta": eta,
                    "eval_point": "y_bar",

                    "objective": metrics_ybar["objective"],
                    "train_loss": metrics_ybar["train_loss"],
                    "train_acc": metrics_ybar["train_acc"],
                    "test_loss": metrics_ybar["test_loss"],
                    "test_acc": metrics_ybar["test_acc"],
                    "stat_proxy": metrics_ybar["stat_proxy"],

                    "objective_ybar": metrics_ybar["objective"],
                    "stat_proxy_ybar": metrics_ybar["stat_proxy"],
                    "block_oracle_norm": block_oracle_norm,
                }

                if eval_current_x:
                    row.update({
                        "objective_x": metrics_x["objective"],
                        "stat_proxy_x": metrics_x["stat_proxy"],
                        "test_acc_x": metrics_x["test_acc"],
                    })
                else:
                    row.update({
                        "objective_x": np.nan,
                        "stat_proxy_x": np.nan,
                        "test_acc_x": np.nan,
                    })

                rows.append(row)

            y_sum.zero_()
            oracle_sum.zero_()

    load_model_vector(model, theta)
    return rows


def run_smoothed_sgd(
    model: nn.Module,
    theta0: torch.Tensor,
    cfg: Dict[str, Any],
    train_batch_iter: BatchIterator,
    eval_batch_iter: BatchIterator,
    eval_train_loader,
    test_loader,
    seed: int,
    device: str,
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

    theta = theta0.detach().clone()
    work = 0
    eval_work = 0
    rows = []
    start = time.time()

    for t in trange(1, rounds + 1, desc=f"CIFAR SmoothedSGD seed={seed}"):
        g = smoothed_oracle(model, theta, delta, smooth_B, train_batch_iter)
        theta = (theta - lr * g).detach()
        work += smooth_B * data_B

        if t % eval_every == 0 or t == 1 or t == rounds:
            metrics = evaluate(
                model=model,
                theta=theta,
                cfg=cfg,
                eval_train_loader=eval_train_loader,
                test_loader=test_loader,
                eval_batch_iter=eval_batch_iter,
                device=device,
            )
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
                "delta": delta,
                "M": np.nan,
                "lr_or_eta": lr,
                "eval_point": "theta",

                "objective": metrics["objective"],
                "train_loss": metrics["train_loss"],
                "train_acc": metrics["train_acc"],
                "test_loss": metrics["test_loss"],
                "test_acc": metrics["test_acc"],
                "stat_proxy": metrics["stat_proxy"],

                "objective_x": metrics["objective"],
                "stat_proxy_x": metrics["stat_proxy"],
                "objective_ybar": np.nan,
                "stat_proxy_ybar": np.nan,
                "test_acc_x": metrics["test_acc"],
                "block_oracle_norm": np.nan,
            }

            rows.append(row)

    load_model_vector(model, theta)
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


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_round = df.groupby(["method", "round"], as_index=False).agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        stat_proxy_mean=("stat_proxy", "mean"),
        stat_proxy_std=("stat_proxy", "std"),
        work_mean=("work", "mean"),
    )

    plot_mean_curve(
        grouped=grouped_round,
        x_col="round",
        y_col="train_loss",
        out_path=out_dir / "train_loss_vs_rounds.png",
        title="Train loss vs rounds",
        xlabel="Rounds / depth",
        ylabel="Train loss",
    )

    plot_mean_curve(
        grouped=grouped_round,
        x_col="round",
        y_col="test_acc",
        out_path=out_dir / "test_acc_vs_rounds.png",
        title="Test accuracy vs rounds",
        xlabel="Rounds / depth",
        ylabel="Test accuracy",
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


# ============================================================
# Threshold summary
# ============================================================

def compute_threshold_summaries(
    df: pd.DataFrame,
    thresholds: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                    "first_hit_train_loss": float(first["train_loss"]),
                    "first_hit_test_acc": float(first["test_acc"]),
                    "final_stat_proxy": float(last["stat_proxy"]),
                    "final_train_loss": float(last["train_loss"]),
                    "final_test_acc": float(last["test_acc"]),
                })
            else:
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
                    "first_hit_train_loss": np.nan,
                    "first_hit_test_acc": np.nan,
                    "final_stat_proxy": float(last["stat_proxy"]),
                    "final_train_loss": float(last["train_loss"]),
                    "final_test_acc": float(last["test_acc"]),
                })

    per_seed = pd.DataFrame(per_seed_rows)

    aggregate = per_seed.groupby(["threshold", "method"], as_index=False).agg(
        hit_rate=("hit", "mean"),
        first_hit_round_mean=("first_hit_round", "mean"),
        first_hit_round_std=("first_hit_round", "std"),
        first_hit_work_mean=("first_hit_work", "mean"),
        first_hit_work_std=("first_hit_work", "std"),
        final_stat_proxy_mean=("final_stat_proxy", "mean"),
        final_train_loss_mean=("final_train_loss", "mean"),
        final_test_acc_mean=("final_test_acc", "mean"),
    )

    return per_seed, aggregate


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    if args.name is not None:
        cfg["run"]["name"] = args.name

    if args.rounds is not None:
        cfg["train"]["rounds"] = args.rounds

    if args.seeds is not None:
        cfg["run"]["seeds"] = [int(x) for x in args.seeds.split(",")]

    return cfg


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
    print("CIFAR-10 NOG Experiment")
    print("=" * 80)
    print(f"config      : {args.config}")
    print(f"run name    : {cfg['run']['name']}")
    print(f"out dir     : {out_dir}")
    print(f"device      : {device}")
    print(f"methods     : {cfg['methods']}")
    print(f"seeds       : {cfg['run']['seeds']}")
    print(f"rounds      : {cfg['train']['rounds']}")
    print("=" * 80)

    all_rows = []

    for seed in cfg["run"]["seeds"]:
        seed_all(seed)

        train_loader, eval_train_loader, test_loader = build_cifar_loaders(
            cfg=cfg,
            device=device,
            seed=seed,
        )

        model = build_model(cfg, device)
        theta0 = get_model_vector(model).detach().clone()

        print(f"[seed={seed}] parameter dimension: {theta0.numel()}")

        if "NOG" in cfg["methods"]:
            train_iter = BatchIterator(train_loader, device)
            eval_iter = BatchIterator(eval_train_loader, device)

            rows = run_nog(
                model=model,
                theta0=theta0,
                cfg=cfg,
                train_batch_iter=train_iter,
                eval_batch_iter=eval_iter,
                eval_train_loader=eval_train_loader,
                test_loader=test_loader,
                seed=seed + 10000,
                device=device,
            )

            all_rows.extend(rows)

        if "SmoothedSGD" in cfg["methods"]:
            train_iter = BatchIterator(train_loader, device)
            eval_iter = BatchIterator(eval_train_loader, device)

            rows = run_smoothed_sgd(
                model=model,
                theta0=theta0,
                cfg=cfg,
                train_batch_iter=train_iter,
                eval_batch_iter=eval_iter,
                eval_train_loader=eval_train_loader,
                test_loader=test_loader,
                seed=seed + 20000,
                device=device,
            )

            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    plot_curves(df, out_dir)

    final_df = df.sort_values("round").groupby(["method", "seed"]).tail(1)
    summary_df = final_df.groupby("method").agg(
        final_train_loss_mean=("train_loss", "mean"),
        final_train_loss_std=("train_loss", "std"),
        final_test_acc_mean=("test_acc", "mean"),
        final_test_acc_std=("test_acc", "std"),
        final_stat_proxy_mean=("stat_proxy", "mean"),
        final_stat_proxy_std=("stat_proxy", "std"),
        final_work_mean=("work", "mean"),
        final_eval_work_mean=("eval_work", "mean"),
        final_time_sec_mean=("time_sec", "mean"),
    )

    summary = {
        "num_rows": int(len(df)),
        "methods": list(df["method"].unique()),
        "seeds": cfg["run"]["seeds"],
        "primary_metric_note": "For NOG, train_loss/test_acc/stat_proxy are evaluated at block-average y_bar.",
        "final_by_method": summary_df.to_dict(),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    thresholds = cfg.get("metrics", {}).get("thresholds", [])
    if thresholds:
        threshold_per_seed, threshold_summary = compute_threshold_summaries(df, thresholds)
        threshold_per_seed.to_csv(out_dir / "threshold_per_seed.csv", index=False)
        threshold_summary.to_csv(out_dir / "threshold_summary.csv", index=False)

    print(f"Saved results to: {csv_path}")
    print(f"Saved figures to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
