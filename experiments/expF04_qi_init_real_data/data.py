"""Dataset loading + featurization for expF04.

Four real tasks, each reduced to a fixed-length feature vector and normalized to
[-1, 1] (min-max, fit on train). Regression targets are min-max'd to [-1, 1] too so
MSE is comparable across arms/tasks.

  fashion_mnist      784 -> 10   classification   (idx images)
  uci_har            561 ->  6   classification   (engineered sensor features)
  superconductivity   81 ->  1   regression       (material -> critical temp)
  sarcos              21 ->  1   regression       (arm state -> joint-1 torque)

Data must be fetched first (fetch_data.py); loaders read from <data_dir>.
--smoke fabricates random tensors of the right shape so the pipeline can be tested
without any download.
"""
from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# name -> (d_in, d_out, kind)
TASK_META = {
    "fashion_mnist": (784, 10, "classif"),
    "uci_har": (561, 6, "classif"),
    "superconductivity": (81, 1, "regr"),
    "sarcos": (21, 1, "regr"),
}
TASK_ORDER = ["fashion_mnist", "uci_har", "superconductivity", "sarcos"]
PRETTY = {
    "fashion_mnist": "Fashion-MNIST",
    "uci_har": "UCI HAR",
    "superconductivity": "Superconductivity",
    "sarcos": "SARCOS",
}


@dataclass
class Task:
    name: str
    kind: str            # "classif" | "regr"
    d_in: int
    d_out: int
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


# ----------------------------- normalization -----------------------------

def _minmax_fit(x: np.ndarray):
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    scale = hi - lo
    scale[scale == 0] = 1.0
    return lo, scale


def _minmax_apply(x: np.ndarray, lo, scale, clip=True):
    z = 2.0 * (x - lo) / scale - 1.0
    if clip:
        z = np.clip(z, -1.0, 1.0)
    return z


def _pack(name, xtr, ytr, xte, yte) -> Task:
    d_in, d_out, kind = TASK_META[name]
    lo, scale = _minmax_fit(xtr)
    xtr = _minmax_apply(xtr, lo, scale)
    xte = _minmax_apply(xte, lo, scale)
    if kind == "regr":
        ylo, yscale = _minmax_fit(ytr.reshape(-1, 1))
        ytr = _minmax_apply(ytr.reshape(-1, 1), ylo, yscale, clip=False).ravel()
        yte = _minmax_apply(yte.reshape(-1, 1), ylo, yscale, clip=False).ravel()
        yt_tr = torch.tensor(ytr, dtype=torch.float32)
        yt_te = torch.tensor(yte, dtype=torch.float32)
    else:
        yt_tr = torch.tensor(ytr, dtype=torch.long)
        yt_te = torch.tensor(yte, dtype=torch.long)
    return Task(
        name=name, kind=kind, d_in=d_in, d_out=d_out,
        x_train=torch.tensor(xtr, dtype=torch.float32),
        y_train=yt_tr,
        x_test=torch.tensor(xte, dtype=torch.float32),
        y_test=yt_te,
    )


# ----------------------------- loaders -----------------------------

def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * r * c)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, r * c).astype(np.float32)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def _load_fashion_mnist(d: Path):
    root = d / "fashion_mnist"
    xtr = _read_idx_images(root / "train-images-idx3-ubyte.gz")
    ytr = _read_idx_labels(root / "train-labels-idx1-ubyte.gz")
    xte = _read_idx_images(root / "t10k-images-idx3-ubyte.gz")
    yte = _read_idx_labels(root / "t10k-labels-idx1-ubyte.gz")
    return xtr, ytr, xte, yte


def _find(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def _load_uci_har(d: Path):
    root = d / "uci_har"
    xtr = np.loadtxt(_find(root, "X_train.txt"))
    ytr = np.loadtxt(_find(root, "y_train.txt")).astype(np.int64) - 1   # 1..6 -> 0..5
    xte = np.loadtxt(_find(root, "X_test.txt"))
    yte = np.loadtxt(_find(root, "y_test.txt")).astype(np.int64) - 1
    return xtr, ytr, xte, yte


def _load_superconductivity(d: Path, seed: int = 0):
    root = d / "superconductivity"
    arr = np.loadtxt(_find(root, "train.csv"), delimiter=",", skiprows=1)
    x, y = arr[:, :-1], arr[:, -1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_te = int(0.2 * len(x))
    te, tr = perm[:n_te], perm[n_te:]
    return x[tr], y[tr], x[te], y[te]


def _load_sarcos(d: Path):
    from scipy.io import loadmat
    root = d / "sarcos"
    tr = loadmat(_find(root, "sarcos_inv.mat"))["sarcos_inv"]
    te = loadmat(_find(root, "sarcos_inv_test.mat"))["sarcos_inv_test"]
    # cols 0..20 = state (21), cols 21..27 = 7 torques; predict joint-1 torque (col 21)
    return tr[:, :21], tr[:, 21], te[:, :21], te[:, 21]


_LOADERS = {
    "fashion_mnist": _load_fashion_mnist,
    "uci_har": _load_uci_har,
    "superconductivity": _load_superconductivity,
    "sarcos": _load_sarcos,
}


def load_task(name: str, data_dir: Path) -> Task:
    xtr, ytr, xte, yte = _LOADERS[name](Path(data_dir))
    return _pack(name, xtr, ytr, xte, yte)


def smoke_task(name: str, n_train=512, n_test=256, seed=0) -> Task:
    """Random data of the right shape -- pipeline test only, no download."""
    d_in, d_out, kind = TASK_META[name]
    g = np.random.default_rng(seed)
    xtr = g.standard_normal((n_train, d_in)).astype(np.float32)
    xte = g.standard_normal((n_test, d_in)).astype(np.float32)
    if kind == "classif":
        ytr = g.integers(0, d_out, n_train)
        yte = g.integers(0, d_out, n_test)
    else:
        ytr = g.standard_normal(n_train).astype(np.float32)
        yte = g.standard_normal(n_test).astype(np.float32)
    return _pack(name, xtr, ytr, xte, yte)
