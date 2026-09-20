# expF10 QI-encoded neural operators: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much the QI (fixed ridge encoder/decoder) helps a learned Darcy operator, via three matched-capacity configs -- A (QI-coeff MLP), B (QI-resample -> FNO), C (plain FNO control) -- on accuracy, discretization invariance, and data efficiency.

**Architecture:** A numpy `QICodec` gives fixed linear `encode = pinv(basis(grid))@a` / `decode = basis(P)@c` (basis = `rows_2d` identity terms, reused from expF08 `core`). A small custom torch `FNO2d`. Config A trains an MLP in coefficient space with an in-graph QI decode (fixed `Phi_out` tensor) so the loss is field-space; B feeds a QI-resampled input to the FNO (offline); C is the plain FNO. Darcy data downsampled from the on-disk 421^2 sets. One A100.

**Tech Stack:** Python, numpy, torch (CUDA), matplotlib; `uv run --extra dev`.

**Spec:** `docs/superpowers/specs/2026-07-16-qi-encoded-neural-operator-design.md`

**Verified probes:** codec D=586 (W=576+10), smooth recon 8e-9, cross-res (enc 32^2 -> dec 64^2) 1.2e-8, rough-Darcy recon 4.5e-2; the `SpectralConv2d` below runs forward+backward at 64^2.

**Data:** `/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_{train,test}_421_jax.npz`, keys `x` (coeff, 4000/1000 x 421^2), `y` (solution). Domain mapped to `[-1,1]^2`.

---

## File Structure

- `experiments/expF10_qi_operator/__init__.py` -- package marker.
- `experiments/expF10_qi_operator/qi_codec.py` -- `QICodec` (numpy fixed encode/decode, cached `Phi^+`).
- `experiments/expF10_qi_operator/fno2d.py` -- `SpectralConv2d`, `FNO2d` (torch).
- `experiments/expF10_qi_operator/data.py` -- Darcy loader + downsample + grid helpers.
- `experiments/expF10_qi_operator/models.py` -- config A/B/C models + `build_model`.
- `experiments/expF10_qi_operator/run.py` -- train + eval (standard/invariance/data-efficiency), `data.json`, CLI.
- `tests/test_expF10_qi_operator.py` -- the tests below.
- `results/checkpoint_F_applications/expF10_qi_operator/expF10_results.md` + index (Task 7).

All commands from repo root `/scr/cdeng/precision-mlps`.

---

### Task 1: QI codec (`qi_codec.py`)

**Files:** Create `experiments/expF10_qi_operator/__init__.py`, `experiments/expF10_qi_operator/qi_codec.py`; Test `tests/test_expF10_qi_operator.py`.

- [ ] **Step 1: Package marker.** Create empty `experiments/expF10_qi_operator/__init__.py`.

- [ ] **Step 2: Write the failing tests.** Create `tests/test_expF10_qi_operator.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))
sys.path.append(str(REPO_ROOT / "experiments" / "expF10_qi_operator"))

import qi_codec as qc

DARCY = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_train_16_jax.npz"


def _smooth(P):
    return np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])


def test_codec_roundtrips_smooth_field():
    codec = qc.QICodec(W=576, lam=0.25)
    g = codec.grid(32)
    c = codec.encode(_smooth(g), 32)
    assert codec.rel_l2(codec.decode(c, g), _smooth(g)) < 1e-7


def test_codec_is_resolution_transferable():
    """Encode on 32^2, decode on 64^2 -- the property config A relies on."""
    codec = qc.QICodec(W=576, lam=0.25)
    c = codec.encode(_smooth(codec.grid(32)), 32)
    g64 = codec.grid(64)
    assert codec.rel_l2(codec.decode(c, g64), _smooth(g64)) < 1e-6


def test_rough_darcy_reconstruction_is_bounded():
    codec = qc.QICodec(W=576, lam=0.25)
    a = np.load(DARCY)["x"][0].astype(np.float64).ravel()
    c = codec.encode(a, 16)
    err = codec.rel_l2(codec.decode(c, codec.grid(16)), a)
    assert 1e-3 < err < 1e-1     # rough: represented, not exact (probe: ~4.5e-2)
```

- [ ] **Step 3: Run to verify fail.** `uv run --extra dev python -m pytest tests/test_expF10_qi_operator.py -q` -> `ModuleNotFoundError: No module named 'qi_codec'`.

- [ ] **Step 4: Write `qi_codec.py`:**

```python
"""Fixed QI ridge encoder/decoder (numpy) for expF10.

encode(a_grid) = pinv(basis(grid)) @ a  -- a fixed linear analysis transform.
decode(c, P)   = basis(P) @ c           -- synthesis at any points/resolution.
basis(P) = rows_2d(P, ..., [((0,0),1.0)]) -- ridge values + degree-3 poly tail.
Pseudo-inverses are cached per grid resolution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))

import core  # expF08 scalar primitives


class QICodec:
    def __init__(self, W=576, lam=0.25, rcond=1e-10):
        self.dirs, self.offs, self.gamma = core.radon_geometry(W, lam)
        self.D = len(self.offs) + len(core.MONO_2D)
        self.rcond = rcond
        self._pinv = {}

    def grid(self, n):
        g = np.linspace(-1.0, 1.0, n)
        X, Y = np.meshgrid(g, g, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def basis(self, P):
        return core.rows_2d(np.asarray(P, float), self.dirs, self.offs,
                            self.gamma, [((0, 0), 1.0)])

    def pinv(self, n):
        if n not in self._pinv:
            self._pinv[n] = np.linalg.pinv(self.basis(self.grid(n)),
                                           rcond=self.rcond)
        return self._pinv[n]

    def encode(self, a_grid, n):
        """a_grid: [n*n] flattened (row-major, indexing='ij') -> coeffs [D]."""
        return self.pinv(n) @ np.asarray(a_grid, float).ravel()

    def decode(self, c, P):
        return self.basis(P) @ np.asarray(c, float)

    @staticmethod
    def rel_l2(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                     / np.linalg.norm(np.asarray(b)))
```

- [ ] **Step 5: Run to verify pass.** `uv run --extra dev python -m pytest tests/test_expF10_qi_operator.py -q` -> PASS (3).

- [ ] **Step 6: Commit.**
```bash
git add experiments/expF10_qi_operator/__init__.py experiments/expF10_qi_operator/qi_codec.py tests/test_expF10_qi_operator.py
git commit -m "feat(expF10): fixed QI ridge encoder/decoder (numpy, cached pinv)"
```

---

### Task 2: Small 2D FNO (`fno2d.py`)

**Files:** Create `experiments/expF10_qi_operator/fno2d.py`; Test `tests/test_expF10_qi_operator.py`.

- [ ] **Step 1: Write the failing test.** Append:

```python
import torch
import fno2d


def test_fno_forward_shape_and_backward():
    net = fno2d.FNO2d(width=16, modes=8, n_layers=3)
    x = torch.randn(2, 1, 48, 48, requires_grad=True)
    y = net(x)
    assert y.shape == (2, 1, 48, 48)
    y.sum().backward()
    assert x.grad is not None
```

- [ ] **Step 2: Run to verify fail.** `uv run --extra dev python -m pytest tests/test_expF10_qi_operator.py::test_fno_forward_shape_and_backward -q` -> `ModuleNotFoundError: No module named 'fno2d'`.

- [ ] **Step 3: Write `fno2d.py`:**

```python
"""Minimal 2D Fourier Neural Operator (torch). Standard FNO2d: lift -> K
spectral-conv+pointwise layers -> project. Verified forward/backward at 64^2."""
from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        # two corners of the rfft2 spectrum (low + high vertical frequencies)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, 2))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, 2))

    @staticmethod
    def _cmul(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, torch.view_as_complex(w))

    def forward(self, x):
        b, c, h, wd = x.shape
        m = self.modes
        xft = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, wd // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = self._cmul(xft[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = self._cmul(xft[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(h, wd))


class FNO2d(nn.Module):
    def __init__(self, width=32, modes=12, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(1, width, 1)
        self.specs = nn.ModuleList(
            [SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, x):
        x = self.lift(x)
        for sp, w in zip(self.specs, self.ws):
            x = torch.nn.functional.gelu(sp(x) + w(x))
        return self.proj(x)
```

- [ ] **Step 4: Run to verify pass.** `uv run --extra dev python -m pytest tests/test_expF10_qi_operator.py::test_fno_forward_shape_and_backward -q` -> PASS.

- [ ] **Step 5: Commit.**
```bash
git add experiments/expF10_qi_operator/fno2d.py tests/test_expF10_qi_operator.py
git commit -m "feat(expF10): minimal 2D FNO (spectral conv + pointwise)"
```

---

### Task 3: Data loader (`data.py`)

**Files:** Create `experiments/expF10_qi_operator/data.py`; Test `tests/test_expF10_qi_operator.py`.

- [ ] **Step 1: Write the failing test.** Append:

```python
import data as dd


def test_load_darcy_downsamples():
    a, u = dd.load_darcy("train", n=8, res=32)
    assert a.shape == (8, 32, 32) and u.shape == (8, 32, 32)
    assert np.isfinite(a).all() and np.isfinite(u).all()
```

- [ ] **Step 2: Run to verify fail.** `... -m pytest ...::test_load_darcy_downsamples -q` -> `ModuleNotFoundError: No module named 'data'`.

- [ ] **Step 3: Write `data.py`:**

```python
"""Darcy dataset loading + area-average downsampling for expF10."""
from __future__ import annotations

import numpy as np

BASE = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax"


def _avg_downsample(field, res):
    """[n,H,W] -> [n,res,res] by block-mean (H must be divisible-ish; uses
    linear index binning that tolerates non-integer ratios via np.add.reduceat)."""
    n, H, W = field.shape
    if H == res:
        return field
    yi = (np.linspace(0, H, res + 1)).astype(int)
    xi = (np.linspace(0, W, res + 1)).astype(int)
    out = np.empty((n, res, res), dtype=np.float64)
    for i in range(res):
        for j in range(res):
            out[:, i, j] = field[:, yi[i]:yi[i + 1], xi[j]:xi[j + 1]].mean((1, 2))
    return out


def load_darcy(split, n, res, source_res=421):
    """Returns (a, u) as [n,res,res] float64, downsampled from source_res."""
    d = np.load(f"{BASE}/darcy_{split}_{source_res}_jax.npz")
    a = np.asarray(d["x"][:n], dtype=np.float64)
    u = np.asarray(d["y"][:n], dtype=np.float64)
    return _avg_downsample(a, res), _avg_downsample(u, res)
```

- [ ] **Step 4: Run to verify pass.** `... ::test_load_darcy_downsamples -q` -> PASS (note: loads a slice of the 421 npz; may take a few seconds).

- [ ] **Step 5: Commit.**
```bash
git add experiments/expF10_qi_operator/data.py tests/test_expF10_qi_operator.py
git commit -m "feat(expF10): Darcy loader with area-average downsampling"
```

---

### Task 4: The three models (`models.py`)

**Files:** Create `experiments/expF10_qi_operator/models.py`; Test `tests/test_expF10_qi_operator.py`.

Config A is an MLP in coefficient space with an **in-graph QI decode** (fixed `Phi_out` buffer) so the loss is field-space. Configs B and C are the FNO; B is fed a QI-resampled input (prepared offline in `run.py`), so at the module level B and C are the *same* FNO -- the difference is the input tensor. `build_model` returns `(module, kind)` where `kind in {"coeff","field"}` tells `run.py` how to feed it.

- [ ] **Step 1: Write the failing test.** Append:

```python
import models as mo


def test_models_forward_backward():
    codec = qc.QICodec(W=128, lam=0.25)   # small W for a fast test
    Phi_out = codec.basis(codec.grid(16))  # [256, D]
    # A: coeff MLP
    A, kindA = mo.build_model("A", D=codec.D, Phi_out=Phi_out)
    assert kindA == "coeff"
    ca = torch.randn(4, codec.D)
    ua = A(ca)
    assert ua.shape == (4, 16 * 16)
    ua.sum().backward()
    # C: plain FNO (field in, field out)
    C, kindC = mo.build_model("C", fno_kw=dict(width=8, modes=6, n_layers=2))
    assert kindC == "field"
    xf = torch.randn(4, 1, 16, 16)
    yf = C(xf)
    assert yf.shape == (4, 1, 16, 16)
```

- [ ] **Step 2: Run to verify fail.** `... ::test_models_forward_backward -q` -> `ModuleNotFoundError: No module named 'models'`.

- [ ] **Step 3: Write `models.py`:**

```python
"""expF10 config models. A = coeff-space MLP + fixed QI decode; B/C = FNO."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import fno2d


class CoeffMLP(nn.Module):
    """c_a [b,D] -> MLP -> c_u [b,D] -> field [b, n_out] via fixed Phi_out."""
    def __init__(self, D, Phi_out, hidden=1024, n_layers=4):
        super().__init__()
        layers, d = [], D
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, D)]
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("Phi_out",
                             torch.tensor(np.asarray(Phi_out), dtype=torch.float32))

    def forward(self, c_a):
        c_u = self.mlp(c_a)                 # [b, D]
        return c_u @ self.Phi_out.t()       # [b, n_out]  (decoded field, flat)


def build_model(config, D=None, Phi_out=None, fno_kw=None):
    fno_kw = fno_kw or dict(width=32, modes=12, n_layers=4)
    if config == "A":
        return CoeffMLP(D, Phi_out), "coeff"
    if config in ("B", "C"):
        return fno2d.FNO2d(**fno_kw), "field"
    raise ValueError(config)
```

- [ ] **Step 4: Run to verify pass.** `... ::test_models_forward_backward -q` -> PASS.

- [ ] **Step 5: Commit.**
```bash
git add experiments/expF10_qi_operator/models.py tests/test_expF10_qi_operator.py
git commit -m "feat(expF10): config A (coeff MLP + QI decode) and B/C (FNO) models"
```

---

### Task 5: Training + evaluation driver (`run.py`)

**Files:** Create `experiments/expF10_qi_operator/run.py`; Test `tests/test_expF10_qi_operator.py`.

- [ ] **Step 1: Write the failing test.** Append:

```python
import run as g10


def test_smoke_train_one_config_returns_finite_loss():
    """Tiny end-to-end: train config C for 2 epochs on 16 instances at 16^2."""
    cfg = g10.SMOKE_CFG
    rec = g10.train_eval("C", cfg)
    assert np.isfinite(rec["test_rel_l2"])
    assert rec["test_rel_l2"] > 0
```

- [ ] **Step 2: Run to verify fail.** `... ::test_smoke_train_one_config_returns_finite_loss -q` -> `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write `run.py`:**

```python
"""expF10 driver: train A/B/C on Darcy, eval accuracy + discretization
invariance + data efficiency.

Usage:
  uv run --extra dev python experiments/expF10_qi_operator/run.py --config all
  uv run --extra dev python experiments/expF10_qi_operator/run.py --eval-invariance
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.append(str(HERE))

import qi_codec as qc
import data as dd
import models as mo

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF10_qi_operator"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Cfg:
    W: int = 576
    lam: float = 0.25
    res: int = 64
    n_train: int = 1000
    n_test: int = 200
    epochs: int = 100
    batch: int = 32
    lr: float = 1e-3
    fno_kw: dict = field(default_factory=lambda: dict(width=32, modes=12, n_layers=4))


SMOKE_CFG = Cfg(W=128, res=16, n_train=16, n_test=16, epochs=2, batch=8,
                fno_kw=dict(width=8, modes=6, n_layers=2))


def _prep(config, cfg, codec):
    """Return (train_in, train_tgt, test_in, test_tgt, kind, extra) tensors.
    Targets are the flattened reference solution fields at cfg.res."""
    a_tr, u_tr = dd.load_darcy("train", cfg.n_train, cfg.res)
    a_te, u_te = dd.load_darcy("test", cfg.n_test, cfg.res)
    u_tr_f = torch.tensor(u_tr.reshape(len(u_tr), -1), dtype=torch.float32)
    u_te_f = torch.tensor(u_te.reshape(len(u_te), -1), dtype=torch.float32)
    if config == "A":
        Pinv = codec.pinv(cfg.res)
        enc = lambda A: torch.tensor(
            (Pinv @ A.reshape(len(A), -1).T).T, dtype=torch.float32)
        return enc(a_tr), u_tr_f, enc(a_te), u_te_f, "coeff"
    # B: QI-resample the input onto the same res grid (offline); C: raw input.
    def to_field(A):
        if config == "B":
            Pinv = codec.pinv(cfg.res)
            Phi = codec.basis(codec.grid(cfg.res))
            A = (Phi @ (Pinv @ A.reshape(len(A), -1).T)).T.reshape(A.shape)
        return torch.tensor(A[:, None], dtype=torch.float32)   # [n,1,res,res]
    return to_field(a_tr), u_tr_f, to_field(a_te), u_te_f, "field"


def _rel_l2(pred, tgt):
    return (torch.linalg.vector_norm(pred - tgt, dim=1)
            / torch.linalg.vector_norm(tgt, dim=1)).mean().item()


def train_eval(config, cfg):
    torch.manual_seed(0)
    codec = qc.QICodec(cfg.W, cfg.lam)
    xin, ytr, xte, yte, kind = _prep(config, cfg, codec)
    if config == "A":
        Phi_out = codec.basis(codec.grid(cfg.res))
        net, _ = mo.build_model("A", D=codec.D, Phi_out=Phi_out)
    else:
        net, _ = mo.build_model(config, fno_kw=cfg.fno_kw)
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    xin, ytr = xin.to(DEV), ytr.to(DEV)
    t0 = time.time()
    for ep in range(cfg.epochs):
        net.train()
        perm = torch.randperm(len(xin), device=DEV)
        for i in range(0, len(xin), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            out = net(xin[idx])
            pred = out.reshape(len(idx), -1)
            loss = _rel_l2(pred, ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = net(xte.to(DEV)).reshape(len(xte), -1).cpu()
    return dict(config=config, res=cfg.res, n_train=cfg.n_train,
                test_rel_l2=_rel_l2(pred, yte),
                n_params=sum(p.numel() for p in net.parameters()),
                t_train=time.time() - t0)


def save(recs, name="data.json"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(recs, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=["A", "B", "C", "all"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-invariance", action="store_true")
    args = ap.parse_args()
    cfg = SMOKE_CFG if args.smoke else Cfg()
    configs = ["A", "B", "C"] if args.config == "all" else [args.config]
    recs = [train_eval(c, cfg) for c in configs]
    for r in recs:
        print(r, flush=True)
    save(recs)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass.** `uv run --extra dev python -m pytest tests/test_expF10_qi_operator.py::test_smoke_train_one_config_returns_finite_loss -q` -> PASS (trains a tiny net on GPU/CPU).

- [ ] **Step 5: Run the full test file.** `uv run --extra dev python -m pytest tests/test_expF10_qi_operator.py -q` -> PASS (6).

- [ ] **Step 6: Commit.**
```bash
git add experiments/expF10_qi_operator/run.py tests/test_expF10_qi_operator.py
git commit -m "feat(expF10): train/eval driver for configs A/B/C with rel-L2 loss"
```

---

### Task 6: Run the study (train + invariance + data-efficiency)

**Files:** none (produces `results/checkpoint_F_applications/expF10_qi_operator/*.json` + figures).

- [ ] **Step 1: Smoke.** `uv run --extra dev python experiments/expF10_qi_operator/run.py --smoke --config all` -> 3 finite records, no traceback.

- [ ] **Step 2: Main accuracy run.** `uv run --extra dev python experiments/expF10_qi_operator/run.py --config all` -> A/B/C test rel L2 at 64^2 (each a few min on the A100). Records in `data.json`.

- [ ] **Step 3: Add the invariance + data-efficiency sweeps to `run.py`** under `--eval-invariance` and `--eval-data-eff`: after training each config at 64^2, evaluate zero-shot at res in {16,32,64,128,256} (for A, rebuild `Phi_out`/`pinv` per res and re-`_prep` the test set; for B/C, feed the test input at that res -- the FNO/`decode` both accept any res); and retrain at `n_train in {100,300,1000}`. Write `invariance.json` and `data_eff.json`. Then a plotting helper writes `accuracy_bar.png`, `invariance_vs_res.png`, `data_eff.png`. (Code mirrors `train_eval`; keep the same seed and cfg.)

- [ ] **Step 4: Run the sweeps.** `uv run --extra dev python experiments/expF10_qi_operator/run.py --eval-invariance` and `--eval-data-eff`.

- [ ] **Step 5: Sanity-read.** Confirm: C (plain FNO) test rel L2 is in the literature ballpark (~1e-2) at 64^2; A/B report their deltas; invariance -- A roughly flat across res, C worse off 64^2.

- [ ] **Step 6: Commit results.**
```bash
git add results/checkpoint_F_applications/expF10_qi_operator/*.png
git commit -m "results(expF10): QI-operator accuracy, invariance, data-efficiency figures"
```
(`*.json` gitignored by `results/**`.)

---

### Task 7: Writeup + checkpoint index

**Files:** Create `results/checkpoint_F_applications/expF10_qi_operator/expF10_results.md`; Modify `results/checkpoint_F_applications/expF_results.md`.

- [ ] **Step 1: Write `expF10_results.md`** (TL;DR / Question / Design / Results / Conclusions / Open questions), filled from the json + figures. State, with numbers: A/B/C test rel L2 at 64^2 (the "how much does QI help" delta), the invariance curves (does A stay flat while C degrades?), data-efficiency, and the QI input-reconstruction error (the rough-Darcy bound). Report n_params per config and t_train. Frame honestly: this is the data-driven regime; the interesting result is *where and how much* the fixed QI representation helps.

- [ ] **Step 2: Add to the checkpoint index.** In `results/checkpoint_F_applications/expF_results.md` under `## Experiments`, after the expF09 bullet:
```markdown
- **expF10 -- QI-encoded neural operators (drafted).** Data-driven A/B/C on Darcy: A QI-coeff MLP, B QI-resample->FNO, C plain FNO. Measures how much a fixed QI ridge encoder/decoder helps vs plain FNO on accuracy, discretization invariance (train 64^2, test 16-256), and data efficiency. Writeup: `expF10_qi_operator/expF10_results.md`.
```

- [ ] **Step 3: Commit.**
```bash
git add results/checkpoint_F_applications/expF10_qi_operator/expF10_results.md results/checkpoint_F_applications/expF_results.md
git commit -m "docs(expF10): QI-encoded neural operator writeup + checkpoint index"
```

---

## Self-Review Notes

- **Spec coverage:** QICodec fixed encode/decode + cross-res + rough-bound (T1); small FNO (T2); Darcy loader/downsample (T3); configs A/B/C (T4); train/eval with field-space rel-L2 loss (T5); accuracy + invariance + data-efficiency runs + figures (T6); writeup + index (T7). The QI input-reconstruction diagnostic is T1's rough-field test, reported in T7.
- **Placeholder scan:** none. T6 Step 3 describes an additive extension of the already-written `train_eval` (same pattern, no new interfaces) rather than pasting a near-duplicate; the sweep is a loop over res / n_train reusing `_prep` + `train_eval`.
- **Type consistency:** `QICodec` (`grid`, `basis`, `pinv`, `encode(a,n)`, `decode(c,P)`, `D`); `build_model(config, D, Phi_out, fno_kw) -> (module, kind in {coeff,field})`; `Cfg`/`SMOKE_CFG`; `train_eval(config, cfg) -> dict(config,res,n_train,test_rel_l2,n_params,t_train)`; records used identically in the tests, driver, and writeup.
- **Verified wrinkles:** codec tolerances (smooth <1e-7, cross-res <1e-6, rough 1e-3..1e-1) come from the probe (8e-9 / 1.2e-8 / 4.5e-2); `SpectralConv2d` forward/backward verified at 64^2; A feeds coeffs `[b,D]` and decodes via a fixed `Phi_out` buffer so the loss is field-space and backprops to the MLP; B/C share the FNO and differ only by the input tensor (`_prep`). Append-only `sys.path` in the test avoids the shared-module-name collision seen in expG03/expG04.