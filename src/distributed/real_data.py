"""Reproducible LIBSVM capped-l1 SVM problem used by Step ZO-9C."""

from __future__ import annotations

import bz2
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests
import torch


LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
DATASETS = {
    "a9a": {
        "file": "a9a",
        "compressed": False,
        "n": 32561,
        "d": 123,
        "sha256": "f5d5ffd8d865ff41328e7ee043e4b020816914ff6843ff15b98905ddbedce906",
    },
    "ijcnn1": {
        "file": "ijcnn1.bz2",
        "compressed": True,
        "n": 49990,
        "d": 22,
        "sha256": "16506cad788cf7c9607454150ed1994788204bac2ff4c9cb3b320036b6950d3f",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(name: str, root: Path) -> Path:
    """Download one official LIBSVM training file without hidden preprocessing."""

    if name not in DATASETS:
        raise ValueError(f"Unsupported real dataset: {name}.")
    spec = DATASETS[name]
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / f"{name}.libsvm"
    if raw_path.exists():
        if sha256(raw_path) != str(spec["sha256"]):
            raise ValueError(f"Cached dataset hash mismatch: {raw_path}.")
        return raw_path
    source_path = root / str(spec["file"])
    if not source_path.exists():
        url = f"{LIBSVM_BASE}/{spec['file']}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        temporary = source_path.with_suffix(source_path.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(source_path)
    raw_path.write_bytes(
        bz2.decompress(source_path.read_bytes())
        if bool(spec["compressed"])
        else source_path.read_bytes()
    )
    if sha256(raw_path) != str(spec["sha256"]):
        raw_path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded dataset hash mismatch: {name}.")
    return raw_path


@lru_cache(maxsize=8)
def load_libsvm_dense(
    path: Path,
    dimension: int,
    expected_rows: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse a finite-dimensional LIBSVM file using only core dependencies."""

    features = []
    labels = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            fields = raw.strip().split()
            if not fields:
                continue
            label = float(fields[0])
            if label not in {-1.0, 1.0}:
                raise ValueError(f"Invalid label {label} at line {line_number}.")
            row = torch.zeros(dimension, dtype=torch.float32)
            for token in fields[1:]:
                index_text, value_text = token.split(":", maxsplit=1)
                index = int(index_text) - 1
                if index < 0 or index >= dimension:
                    raise ValueError(
                        f"Feature {index + 1} outside [1,{dimension}] at line "
                        f"{line_number}."
                    )
                row[index] = float(value_text)
            features.append(row)
            labels.append(label)
    if expected_rows is not None and len(features) != expected_rows:
        raise ValueError(f"Row count {len(features)} != expected {expected_rows}.")
    if not features:
        raise ValueError(f"No examples parsed from {path}.")
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


class CappedL1SVM:
    """Finite-sum hinge-loss SVM with a nonconvex capped-l1 penalty."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        cap: float,
        lam: float,
        device: str,
        normalize_rows: bool = True,
    ) -> None:
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError("Expected a feature matrix and label vector.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Feature/label row counts do not match.")
        if cap <= 0 or lam < 0:
            raise ValueError("cap must be positive and lambda nonnegative.")
        values = features.to(device)
        if normalize_rows:
            values = values / values.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.features = values
        self.labels = labels.to(device)
        self.cap = float(cap)
        self.lam = float(lam)
        self.device = device
        self.n, self.d = values.shape

    def component_losses(
        self, x: torch.Tensor, idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        features = self.features if idx is None else self.features[idx]
        labels = self.labels if idx is None else self.labels[idx]
        if x.ndim == 1:
            points = x.unsqueeze(0).expand(features.shape[0], -1)
        elif x.ndim == 2 and x.shape[0] == features.shape[0]:
            points = x
        else:
            raise ValueError("x must be one point or one point per selected sample.")
        margins = labels * (features * points).sum(dim=1)
        hinge = torch.relu(1.0 - margins)
        penalty = self.lam * torch.minimum(
            points.abs(), torch.full_like(points, self.cap)
        ).sum(dim=1)
        return hinge + penalty

    def loss(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.component_losses(x, idx).mean()

    def accuracy(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            scores = self.features @ x
            predictions = torch.where(scores >= 0, 1.0, -1.0)
            return float((predictions == self.labels).float().mean().item())


def build_real_problem(cfg: dict, device: str) -> CappedL1SVM:
    pcfg = cfg["problem"]
    name = str(pcfg["dataset"])
    if name not in DATASETS:
        raise ValueError(f"Unsupported real dataset: {name}.")
    spec = DATASETS[name]
    path = download_dataset(name, Path(pcfg.get("data_root", "data/libsvm")))
    features, labels = load_libsvm_dense(
        path, int(spec["d"]), expected_rows=int(spec["n"])
    )
    lam_cfg = pcfg.get("lam", "paper")
    lam = 1.0e-5 / int(spec["n"]) if lam_cfg == "paper" else float(lam_cfg)
    return CappedL1SVM(
        features,
        labels,
        cap=float(pcfg.get("cap", 2.0)),
        lam=lam,
        device=device,
        normalize_rows=bool(pcfg.get("normalize_rows", True)),
    )
