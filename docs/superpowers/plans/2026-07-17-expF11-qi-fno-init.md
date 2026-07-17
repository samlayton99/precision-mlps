# expF11 QI-based FNO initialization: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get initial signal on three QI-derived FNO initializations vs a random-init control on low-data Darcy: (1) physics-pretrain-as-init, (2) warm-start residual, (3) QI-bandwidth spectral init.

**Architecture:** Reuse expF10's `fno2d`/`data`/`qi_codec` (import via sys.path). Add `qi_solve.py` (wraps expF08's Darcy collocation solver -> `u_QI(a)` teacher targets, disk-cached) and `init_methods.py` (QI spectral init + its bandwidth estimate). `run.py` has one `train_eval(method, cfg)` dispatching D0/1/2/3 and logging a convergence curve.

**Tech Stack:** Python, numpy, torch (CUDA A100), matplotlib; `uv run --extra dev`.

**Spec:** `docs/superpowers/specs/2026-07-17-qi-fno-initialization-design.md`

**Verified probe:** `u_QI(a)` at 64^2 via `core.solve_square` + `DarcyCoefficient(sigma_px=4, cell_centered=True)`, W=576, lam=0.25 -> rel L2 **1.03e-2** vs dataset `u_ref` (better than the ~7% FNO), boundary max ~3.6e-3 (cell-centered), **0.7 s/solve**.

**Reused modules:** `experiments/expF10_qi_operator/{fno2d,data,qi_codec}.py`; `experiments/expF08_darcy_sweep/{core,darcy_problems}.py`.

---

## File Structure

- `experiments/expF11_qi_fno_init/__init__.py` -- package marker.
- `experiments/expF11_qi_fno_init/qi_solve.py` -- `u_qi(a,res)`, `batch_u_qi`, disk cache.
- `experiments/expF11_qi_fno_init/init_methods.py` -- `qi_resample_gain`, `qi_spectral_init`.
- `experiments/expF11_qi_fno_init/run.py` -- `train_eval(method, cfg)` for D0/1/2/3, convergence log, `data.json`, `--plot`.
- `tests/test_expF11_qi_fno_init.py`.
- `results/checkpoint_F_applications/expF11_qi_fno_init/expF11_results.md` + index (Task 5).

All commands from repo root `/scr/cdeng/precision-mlps`.

---

### Task 1: QI solve teacher (`qi_solve.py`)

**Files:** Create `experiments/expF11_qi_fno_init/__init__.py`, `experiments/expF11_qi_fno_init/qi_solve.py`; Test `tests/test_expF11_qi_fno_init.py`.

- [ ] **Step 1: Package marker.** Empty `experiments/expF11_qi_fno_init/__init__.py`.

- [ ] **Step 2: Write the failing test.** Create `tests/test_expF11_qi_fno_init.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for d in ("expF08_darcy_sweep", "expF10_qi_operator", "expF11_qi_fno_init"):
    sys.path.append(str(REPO_ROOT / "experiments" / d))

import qi_solve as qs
import data as dd


def test_u_qi_is_a_sane_solution():
    a, u = dd.load_darcy("test", n=2, res=32)
    uq = qs.u_qi(a[0], res=32)
    assert uq.shape == (32, 32) and np.isfinite(uq).all()
    assert np.max(np.abs(uq[0, :])) < 1e-2         # ~Dirichlet (cell-centered)
    rel = np.linalg.norm(uq - u[0]) / np.linalg.norm(u[0])
    assert rel < 5e-2                              # a real (approx) solution
```

- [ ] **Step 3: Run to verify fail.** `uv run --extra dev python -m pytest tests/test_expF11_qi_fno_init.py -q` -> `ModuleNotFoundError: No module named 'qi_solve'`.

- [ ] **Step 4: Write `qi_solve.py`:**

```python
"""QI physics-solve teacher for expF11: u_QI(a) via the expF08 Darcy collocation
solver. Disk-cached so target generation is one-time."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.append(str(REPO_ROOT / "experiments" / "expF08_darcy_sweep"))

import core
import darcy_problems as dp

CACHE = HERE / "_uqi_cache"
W_DEFAULT, LAM, SIGMA = 576, 0.25, 4.0


def _eval_grid(n):
    g = dp.grid_1d(n, cell_centered=True)
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def _zero_bc():
    Pb = core.boundary_points_square(360)
    return [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]


def u_qi(a_grid, res, W=W_DEFAULT, lam=LAM, sigma=SIGMA):
    """QI Darcy solution for one coefficient field [res,res] -> [res,res]."""
    coeff = dp.DarcyCoefficient(np.asarray(a_grid, float), sigma_px=sigma,
                                cell_centered=True)
    model = core.solve_square(coeff.terms(), dp.DARCY_FORCING, _zero_bc(),
                              W, lam, seed=42)
    return core.eval_model(model, _eval_grid(res)).reshape(res, res)


def batch_u_qi(a_batch, res, tag, **kw):
    """[n,res,res] -> [n,res,res], cached to _uqi_cache/{tag}.npy."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{tag}_res{res}_n{len(a_batch)}.npy"
    if path.exists():
        return np.load(path)
    out = np.stack([u_qi(a, res, **kw) for a in a_batch])
    np.save(path, out)
    return out
```

- [ ] **Step 5: Run to verify pass.** `... -m pytest tests/test_expF11_qi_fno_init.py -q` -> PASS (the single solve is ~0.3 s at res 32).

- [ ] **Step 6: Commit.**
```bash
git add experiments/expF11_qi_fno_init/__init__.py experiments/expF11_qi_fno_init/qi_solve.py tests/test_expF11_qi_fno_init.py
git commit -m "feat(expF11): QI physics-solve teacher u_QI(a) with disk cache"
```

---

### Task 2: QI spectral init (`init_methods.py`)

**Files:** Create `experiments/expF11_qi_fno_init/init_methods.py`; Test `tests/test_expF11_qi_fno_init.py`.

- [ ] **Step 1: Write the failing test.** Append:

```python
import torch
import fno2d
import qi_codec as qc
import init_methods as im


def test_qi_spectral_init_changes_weights_and_runs():
    codec = qc.QICodec(W=256, lam=0.25)
    net = fno2d.FNO2d(width=16, modes=12, n_layers=3)
    before = net.specs[0].w1.detach().clone()
    im.qi_spectral_init(net, codec, res=64)
    assert not torch.allclose(before, net.specs[0].w1)     # init changed weights
    for res in (64, 32):                                   # still runs, incl low-res
        y = net(torch.randn(2, 1, res, res))
        assert y.shape == (2, 1, res, res)
```

- [ ] **Step 2: Run to verify fail.** `... ::test_qi_spectral_init_changes_weights_and_runs -q` -> `ModuleNotFoundError: No module named 'init_methods'`.

- [ ] **Step 3: Write `init_methods.py`:**

```python
"""QI-bandwidth spectral initialization for expF11 method (3).

The QI-resample operator a -> decode(encode(a)) = Phi Phi^+ a is low-pass; its
empirical radial frequency gain is used to shape the FNO's spectral-conv init so
the net starts biased toward the QI's frequency content (a simplification of a
full per-mode operator fit; the low-pass envelope is what carries the signal)."""
from __future__ import annotations

import numpy as np
import torch


def qi_resample_gain(codec, res, n_probe=48, seed=0):
    """Radial frequency gain |F(resample a)| / |F(a)| of the QI-resample, on
    random fields, as a 1-D profile indexed by integer radius."""
    rng = np.random.default_rng(seed)
    Pinv = codec.pinv(res)
    Phi = codec.basis(codec.grid(res))
    kx = np.fft.fftfreq(res) * res
    R = np.round(np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)).astype(int)
    num = np.zeros(res); den = np.zeros(res)
    for _ in range(n_probe):
        a = rng.standard_normal((res, res))
        ar = (Phi @ (Pinv @ a.ravel())).reshape(res, res)
        fa, far = np.abs(np.fft.fft2(a)), np.abs(np.fft.fft2(ar))
        for r in range(res):
            m = R == r
            if m.any():
                num[r] += far[m].mean(); den[r] += fa[m].mean()
    g = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return g / max(g[0], 1e-12)          # normalize DC gain to 1


def qi_spectral_init(net, codec, res=64):
    """Scale each spectral-conv layer's per-mode weights by the QI-resample
    radial gain at that mode's (kx,ky), in place."""
    g = qi_resample_gain(codec, res)
    with torch.no_grad():
        for sp in net.specs:
            m = sp.modes
            env = np.zeros((m, m), dtype=np.float32)
            for i in range(m):
                for j in range(m):
                    r = int(round((i ** 2 + j ** 2) ** 0.5))
                    env[i, j] = g[min(r, len(g) - 1)]
            e = torch.tensor(env)[None, None, :, :, None]
            sp.w1.mul_(e)
            sp.w2.mul_(e)
```

- [ ] **Step 4: Run to verify pass.** `... ::test_qi_spectral_init_changes_weights_and_runs -q` -> PASS.

- [ ] **Step 5: Commit.**
```bash
git add experiments/expF11_qi_fno_init/init_methods.py tests/test_expF11_qi_fno_init.py
git commit -m "feat(expF11): QI-bandwidth spectral initialization"
```

---

### Task 3: Train/eval driver (`run.py`)

**Files:** Create `experiments/expF11_qi_fno_init/run.py`; Test `tests/test_expF11_qi_fno_init.py`.

- [ ] **Step 1: Write the failing tests.** Append:

```python
import run as g11


def test_train_eval_all_methods_finite():
    cfg = g11.SMOKE_CFG
    for method in ("D0", "1", "2", "3"):
        rec = g11.train_eval(method, cfg)
        assert np.isfinite(rec["test_rel_l2"]) and rec["test_rel_l2"] > 0


def test_pretrain_init_lowers_starting_loss():
    """Method 1's pretrained net starts below a random net on the labeled set."""
    cfg = g11.SMOKE_CFG
    assert g11.pretrain_start_loss(cfg) < g11.random_start_loss(cfg)
```

- [ ] **Step 2: Run to verify fail.** `... ::test_train_eval_all_methods_finite -q` -> `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write `run.py`:**

```python
"""expF11 driver: D0 (random init) + three QI inits, low-data Darcy + convergence.

  D0: random-init FNO, train on labels
  1 : pretrain on (a, u_QI), fine-tune on labels
  2 : train on (u_ref - u_QI); infer u_QI + FNO
  3 : QI spectral init, train on labels

Usage: run.py --method all [--smoke] [--plot]
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
for d in ("expF10_qi_operator", "expF11_qi_fno_init"):
    sys.path.append(str(REPO_ROOT / "experiments" / d))

import fno2d
import data as dd
import qi_codec as qc
import qi_solve as qs
import init_methods as im

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF11_qi_fno_init"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Cfg:
    res: int = 64
    n_qi: int = 500
    n_lab: int = 100
    n_test: int = 200
    epochs: int = 100
    pre_epochs: int = 100
    batch: int = 32
    lr: float = 1e-3
    W: int = 576
    fno_kw: dict = field(default_factory=lambda: dict(width=32, modes=12, n_layers=4))


SMOKE_CFG = Cfg(res=32, n_qi=8, n_lab=8, n_test=8, epochs=3, pre_epochs=3, batch=4, W=128)


def _t(x):
    return torch.tensor(np.asarray(x, np.float32))


def _rel(pred, tgt):
    return (torch.linalg.vector_norm(pred - tgt, dim=1)
            / torch.linalg.vector_norm(tgt, dim=1)).mean()


def _field(a):
    return _t(a)[:, None]                       # [n,1,res,res]


def _flat(u):
    return _t(u).reshape(len(u), -1)


def _fit(net, xin, ytgt, cfg, epochs, log=None, logx=None, logy=None):
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xin, ytgt = xin.to(DEV), ytgt.to(DEV)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xin), device=DEV)
        for i in range(0, len(xin), cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            loss = _rel(net(xin[idx]).reshape(len(idx), -1), ytgt[idx])
            loss.backward(); opt.step()
        sched.step()
        if log is not None:
            log.append(dict(epoch=ep, rel=_test(net, logx, logy)))


@torch.no_grad()
def _test(net, xte, yte):
    net.eval()
    return _rel(net(xte.to(DEV)).reshape(len(xte), -1), yte.to(DEV)).item()


def _load(cfg):
    a_tr, u_tr = dd.load_darcy("train", cfg.n_qi, cfg.res)      # n_qi >= n_lab
    a_te, u_te = dd.load_darcy("test", cfg.n_test, cfg.res)
    uq_tr = qs.batch_u_qi(a_tr, cfg.res, "train")
    uq_te = qs.batch_u_qi(a_te, cfg.res, "test")
    return a_tr, u_tr, uq_tr, a_te, u_te, uq_te


def train_eval(method, cfg):
    torch.manual_seed(0)
    a_tr, u_tr, uq_tr, a_te, u_te, uq_te = _load(cfg)
    L = cfg.n_lab
    xte, yte = _field(a_te), _flat(u_te)
    net = fno2d.FNO2d(**cfg.fno_kw).to(DEV)
    conv = []
    t0 = time.time()
    if method == "3":
        im.qi_spectral_init(net, qc.QICodec(cfg.W, 0.25), res=cfg.res)
    if method == "1":
        _fit(net, _field(a_tr), _flat(uq_tr), cfg, cfg.pre_epochs)   # pretrain
    if method == "2":
        _fit(net, _field(a_tr[:L]), _flat(u_tr[:L] - uq_tr[:L]), cfg, cfg.epochs,
             conv, _field(a_te), _flat(u_te - uq_te))
        pred = (_t(uq_te).reshape(len(uq_te), -1)
                + net(xte.to(DEV)).reshape(len(xte), -1).cpu())
        rel = _rel(pred, yte).item()
    else:
        _fit(net, _field(a_tr[:L]), _flat(u_tr[:L]), cfg, cfg.epochs,
             conv, xte, yte)
        rel = _test(net, xte, yte)
    return dict(method=method, n_lab=L, res=cfg.res, test_rel_l2=rel,
                conv=conv, t_train=time.time() - t0)


def _start_loss(net, cfg):
    a_tr, u_tr, *_ = _load(cfg)
    return _test(net, _field(a_tr[:cfg.n_lab]), _flat(u_tr[:cfg.n_lab]))


def random_start_loss(cfg):
    torch.manual_seed(0)
    return _start_loss(fno2d.FNO2d(**cfg.fno_kw).to(DEV), cfg)


def pretrain_start_loss(cfg):
    torch.manual_seed(0)
    a_tr, u_tr, uq_tr, *_ = _load(cfg)
    net = fno2d.FNO2d(**cfg.fno_kw).to(DEV)
    _fit(net, _field(a_tr), _flat(uq_tr), cfg, cfg.pre_epochs)
    return _start_loss(net, cfg)


def save(recs, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(recs, indent=1))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = json.loads((RESULTS_DIR / "data.json").read_text())
    # accuracy bars grouped by n_lab
    fig, ax = plt.subplots(figsize=(6, 4))
    labs = sorted({r["n_lab"] for r in recs})
    methods = ["D0", "1", "2", "3"]
    x = np.arange(len(labs))
    for k, m in enumerate(methods):
        ys = [next((r["test_rel_l2"] for r in recs
                    if r["method"] == m and r["n_lab"] == L), np.nan) for L in labs]
        ax.bar(x + k * 0.2, ys, width=0.2, label=f"method {m}")
    ax.set_xticks(x + 0.3); ax.set_xticklabels([f"N={L}" for L in labs])
    ax.set_ylabel("test rel L2"); ax.legend(fontsize=8)
    ax.set_title("expF11: QI-init methods vs random (D0)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "init_accuracy.png", dpi=150)
    # convergence at the smallest n_lab
    Lmin = labs[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        r = next((r for r in recs if r["method"] == m and r["n_lab"] == Lmin), None)
        if r and r["conv"]:
            ax.plot([c["epoch"] for c in r["conv"]],
                    [c["rel"] for c in r["conv"]], label=f"method {m}")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("test rel L2")
    ax.set_title(f"expF11: convergence (N={Lmin})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "init_convergence.png", dpi=150)
    print("figures written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        plot(); return
    cfg = SMOKE_CFG if args.smoke else Cfg()
    methods = ["D0", "1", "2", "3"] if args.method == "all" else [args.method]
    recs = []
    for L in ([cfg.n_lab] if args.smoke else [100, 300]):
        c = Cfg(**{**cfg.__dict__, "n_lab": L}) if not args.smoke else cfg
        for m in methods:
            rec = train_eval(m, c)
            recs.append(rec)
            print({k: rec[k] for k in ("method", "n_lab", "test_rel_l2", "t_train")},
                  flush=True)
            save(recs, "data.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass.** `uv run --extra dev python -m pytest tests/test_expF11_qi_fno_init.py -q` -> PASS (4). (The smoke train + pretrain-loss tests train tiny nets and run a couple QI solves.)

- [ ] **Step 5: Commit.**
```bash
git add experiments/expF11_qi_fno_init/run.py tests/test_expF11_qi_fno_init.py
git commit -m "feat(expF11): D0 + three QI-init train/eval with convergence logging"
```

---

### Task 4: Run the study

**Files:** none (produces `results/checkpoint_F_applications/expF11_qi_fno_init/{data.json,*.png}`).

- [ ] **Step 1: Smoke.** `uv run --extra dev python experiments/expF11_qi_fno_init/run.py --smoke --method all` -> 4 finite records, no traceback.

- [ ] **Step 2: Full run.** `uv run --extra dev python experiments/expF11_qi_fno_init/run.py --method all` -> D0/1/2/3 at N_lab in {100,300} (8 records). The 500 train + 200 test QI solves run once (~8 min, cached); training is fast on the A100.

- [ ] **Step 3: Figures.** `uv run --extra dev python experiments/expF11_qi_fno_init/run.py --plot` -> `init_accuracy.png`, `init_convergence.png`.

- [ ] **Step 4: Sanity-read.** Compare each method's `test_rel_l2` to D0 at N=100 and N=300; note which QI inits beat random and whether the gap shrinks as N grows; eyeball the convergence curves (does any QI init start/settle lower).

- [ ] **Step 5: Commit results.**
```bash
git add results/checkpoint_F_applications/expF11_qi_fno_init/*.png
git commit -m "results(expF11): QI-init accuracy + convergence figures"
```
(`*.json` gitignored.)

---

### Task 5: Writeup + checkpoint index

**Files:** Create `results/checkpoint_F_applications/expF11_qi_fno_init/expF11_results.md`; Modify `results/checkpoint_F_applications/expF_results.md`.

- [ ] **Step 1: Write `expF11_results.md`** (TL;DR / Question / Design / Results / Conclusions / Open questions) from `data.json` + figures. State, with numbers: each method's test rel L2 vs D0 at N=100/300 (does QI-init help, and does the gap close as labels grow?), convergence differences, the u_QI teacher accuracy (~1e-2 at 64^2) and its generation cost, and method 2's non-amortized inference caveat. Honest framing: initial signal, single seed, small FNO.

- [ ] **Step 2: Add to the checkpoint index.** In `results/checkpoint_F_applications/expF_results.md` under `## Experiments`, after the expF10 bullet:
```markdown
- **expF11 -- QI-based FNO initialization (drafted).** Three QI-derived FNO inits vs random on low-data Darcy: (1) physics-pretrain on u_QI then fine-tune, (2) warm-start residual u_QI + FNO, (3) QI-bandwidth spectral init. Measures low-data (N=100/300) accuracy + convergence. u_QI teacher ~1e-2 at 64^2 (better than the 7% FNO). Writeup: `expF11_qi_fno_init/expF11_results.md`.
```

- [ ] **Step 3: Commit.**
```bash
git add results/checkpoint_F_applications/expF11_qi_fno_init/expF11_results.md results/checkpoint_F_applications/expF_results.md
git commit -m "docs(expF11): QI-FNO-init writeup + checkpoint index"
```

---

## Self-Review Notes

- **Spec coverage:** qi_solve teacher (T1); QI spectral init + bandwidth estimate (T2); D0 + methods 1/2/3 train_eval with convergence log (T3); u_QI-sane, spectral-init-changes-weights, all-methods-finite, pretrain-lowers-start tests (T1-T3); low-data (N=100/300) + convergence run + figures (T4); writeup + index (T5). Method-2 inference cost and the single-seed/initial-signal framing are called out in T5.
- **Placeholder scan:** none. Every code step is complete; `run.py` reuses expF10 `fno2d`/`data`/`qi_codec` and expF08 `core`/`darcy_problems` by sys.path (append-only, matching expF10's collision-safe pattern).
- **Type consistency:** `u_qi(a,res)->[res,res]`, `batch_u_qi(a,res,tag)->[n,res,res]`; `qi_resample_gain(codec,res)->[res]`, `qi_spectral_init(net,codec,res)` in place; `Cfg`/`SMOKE_CFG`; `train_eval(method,cfg)->dict(method,n_lab,res,test_rel_l2,conv,t_train)`; helper `random_start_loss`/`pretrain_start_loss` used by the tests. Method labels are strings "D0","1","2","3" throughout.
- **Verified wrinkles:** u_QI at 64^2 = 1.03e-2 vs u_ref, boundary ~3.6e-3 (so test 1 asserts boundary < 1e-2, not exact zero), 0.7 s/solve (W=576, sigma=4); FNO modes-clamp fix from expF10 lets method-2/D0 nets run at the 32^2 smoke res. n_qi >= n_lab so the labeled subset is a prefix of the QI pool (first pass; disjoint pools deferred per spec).
