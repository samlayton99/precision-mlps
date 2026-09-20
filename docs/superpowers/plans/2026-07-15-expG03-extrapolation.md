# expG03 Extrapolation & Data-Poor Generalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch experiment that runs the fixed uniform-γ construction under three hold-out protocols (edge / beyond-domain / sparse-half) over a λ-sweep, reports region-split error, and visualizes per-neuron basis contributions.

**Architecture:** Reuse expG01's geometry (`-1 + arange(-halo,N+halo+1)·h`, `h=2/N`, γ=λ/h, `default_halo`) and the `src/construction` solver (`build_phi` + `solve_readout_with_bias(method="svd")`, fp64). A small `solver.py` holds the numeric core (geometry / fit / predict / basis / metrics); `protocols.py` defines the three splits; `run.py` orchestrates the sweep + `data.json`; `viz.py` renders figures.

**Tech Stack:** Python, numpy, matplotlib; repo runs tests/scripts via `uv run --extra dev`.

**Spec:** `docs/superpowers/specs/2026-07-15-expG03-extrapolation-design.md`

**Reference (do not import, mirror):** `experiments/expG01_interactive_explorer/app.py` — `geometry()` at lines 103–113, `HALO_LAMBDA=0.25` at line 30.

---

## File Structure

- Create `experiments/expG03_extrapolation/__init__.py` — empty package marker.
- Create `experiments/expG03_extrapolation/solver.py` — geometry, fit, predict, basis contributions, rel_l2/linf. Imports `src.construction`.
- Create `experiments/expG03_extrapolation/protocols.py` — `edge_holdout`, `beyond_domain`, `sparse_half`, plus `in_regions` / disjoint-grid helper.
- Create `experiments/expG03_extrapolation/run.py` — target registry, sweep loop, `data.json` (incremental), `--smoke` / `--plot`.
- Create `experiments/expG03_extrapolation/viz.py` — fit+residual, basis-contribution, summary figures.
- Create `tests/test_expG03_extrapolation.py` — the four spec tests.
- Create `results/checkpoint_G_generalization/expG03_extrapolation/expG03_results.md` — writeup (Task 7).
- Modify `results/checkpoint_G_generalization/expG_results.md` — list expG03 (Task 7).

All commands run from the repo root `/scr/cdeng/precision-mlps`.

---

### Task 1: Numeric core (`solver.py`) with the fp64-floor sanity test

**Files:**
- Create: `experiments/expG03_extrapolation/__init__.py`
- Create: `experiments/expG03_extrapolation/solver.py`
- Test: `tests/test_expG03_extrapolation.py`

- [ ] **Step 1: Create the empty package marker**

```bash
: > experiments/expG03_extrapolation/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_expG03_extrapolation.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver


def test_no_holdout_sine_reaches_fp64_floor():
    """Full-grid fit of sine at lambda=0.25, N=128 hits the fp64 floor,
    matching expG01's 3.6e-14 sanity number."""
    N, lam = 128, 0.25
    centers, gamma_vec = solver.geometry(N, lam)
    x = np.linspace(-1.0, 1.0, N)
    f = lambda t: np.sin(2 * np.pi * t)
    v, bias, info = solver.fit(x, f(x), centers, gamma_vec)
    x_test = np.linspace(-1.0, 1.0, 257)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    assert solver.rel_l2(u_hat, f(x_test)) < 1e-12
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'solver'`.

- [ ] **Step 4: Write `solver.py`**

```python
"""expG03 numeric core: fixed uniform-gamma construction + SVD readout.

Mirrors expG01's geometry() (app.py:103) and the src.construction solver path,
as a small importable core for the batch experiment. fp64 throughout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.construction.qi_mpmath import default_halo
from src.construction.readout import build_phi, solve_readout_with_bias

HALO_LAMBDA = 0.25  # reference lambda for halo sizing (as in expG01/expC04)


def geometry(N, lam, halo=None):
    """Center lattice on [-1,1] plus a halo of ghost nodes per side, and a
    per-center gamma = lam/h (h = 2/N). Returns (centers, gamma_vec)."""
    h = 2.0 / N
    if halo is None:
        halo = default_halo(N, lambda_star=HALO_LAMBDA)
    halo = int(halo)
    n_idx = np.arange(-halo, N + halo + 1)
    centers = -1.0 + n_idx.astype(np.float64) * h
    gamma_vec = np.full(centers.size, lam / h)
    return centers, gamma_vec


def fit(x_train, y_train, centers, gamma_vec):
    """Truncated-SVD min-norm readout with bias on the training points.
    Returns (v, bias, info)."""
    Phi = build_phi(x_train, gamma_vec, centers)
    return solve_readout_with_bias(Phi, np.asarray(y_train, float),
                                   method="svd")


def predict(x, centers, gamma_vec, v, bias):
    return build_phi(x, gamma_vec, centers) @ v + bias


def basis_contributions(x, centers, gamma_vec, v, bias):
    """Per-center weighted ridges c_k*phi_k(x) as columns [n_x, width] and the
    scalar bias. contributions.sum(axis=1) + bias == predict(...)."""
    Phi = build_phi(x, gamma_vec, centers)
    return Phi * np.asarray(v, float)[None, :], float(bias)


def rel_l2(u_hat, u_true):
    u_hat, u_true = np.asarray(u_hat), np.asarray(u_true)
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


def linf(u_hat, u_true):
    return float(np.max(np.abs(np.asarray(u_hat) - np.asarray(u_true))))
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add experiments/expG03_extrapolation/__init__.py experiments/expG03_extrapolation/solver.py tests/test_expG03_extrapolation.py
git commit -m "feat(expG03): numeric core (geometry + SVD readout + basis) with fp64-floor test"
```

---

### Task 2: Hold-out protocols (`protocols.py`)

**Files:**
- Create: `experiments/expG03_extrapolation/protocols.py`
- Test: `tests/test_expG03_extrapolation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expG03_extrapolation.py`:

```python
import protocols


def _min_gap(a, b):
    return float(np.min(np.abs(a[:, None] - b[None, :])))


def test_protocols_disjoint_and_holdout_coverage():
    # edge_holdout and beyond_domain: NO train point in the held-out region.
    for maker in (protocols.edge_holdout, protocols.beyond_domain):
        x_tr, x_te, regions = maker()
        assert _min_gap(x_tr, x_te) > 1e-9                       # disjoint grids
        held = protocols.in_regions(x_tr, regions)
        assert held.sum() == 0                                   # data-free hold-out

    # sparse_half: data-POOR, at most n_sparse train points in (0, 1].
    x_tr, x_te, regions = protocols.sparse_half(n_sparse=3)
    assert _min_gap(x_tr, x_te) > 1e-9
    assert protocols.in_regions(x_tr, regions).sum() <= 3


def test_in_regions_masks_correctly():
    x = np.array([-0.9, 0.0, 0.6, 1.2])
    m = protocols.in_regions(x, [(0.5, 1.0)])
    assert list(m) == [False, False, True, False]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'protocols'`.

- [ ] **Step 3: Write `protocols.py`**

```python
"""Hold-out protocols for expG03.

Each protocol returns (x_train, x_test, holdout_regions):
  x_train, x_test  1D float64 grids, guaranteed disjoint;
  holdout_regions  list of (lo, hi) intervals scored as "held-out".
"""
from __future__ import annotations

import numpy as np


def in_regions(x, regions):
    """Boolean mask: True where x falls in any (lo, hi] region."""
    x = np.asarray(x, float)
    m = np.zeros(x.shape, dtype=bool)
    for lo, hi in regions:
        m |= (x > lo) & (x <= hi)
    return m


def _disjoint_grid(lo, hi, n, x_train, tol=1e-9):
    """Equispaced n points on [lo, hi] with any point coinciding with a train
    point removed, so x_train and the result are disjoint."""
    x = np.linspace(lo, hi, n)
    keep = np.min(np.abs(x[:, None] - np.asarray(x_train)[None, :]), axis=1) > tol
    return x[keep]


def edge_holdout(c=0.5, n_train=96, n_test=200):
    """Train on [-1, c]; held-out (c, 1] (one-sided extrapolation)."""
    x_train = np.linspace(-1.0, c, n_train)
    x_test = _disjoint_grid(-1.0, 1.0, n_test, x_train)
    return x_train, x_test, [(c, 1.0)]


def beyond_domain(delta=0.3, n_train=128, n_test=200):
    """Train on [-1, 1]; held-out [-1-delta, -1] and [1, 1+delta] (past the
    last neuron)."""
    x_train = np.linspace(-1.0, 1.0, n_train)
    x_test = _disjoint_grid(-1.0 - delta, 1.0 + delta, n_test, x_train)
    return x_train, x_test, [(-1.0 - delta, -1.0), (1.0, 1.0 + delta)]


def sparse_half(n_dense=96, n_sparse=3, n_test=200):
    """Dense on [-1, 0] plus n_sparse interior points on (0, 1); held-out
    (0, 1] scored as the data-poor half (data-poor, NOT data-free)."""
    x_dense = np.linspace(-1.0, 0.0, n_dense)
    x_sparse = np.linspace(0.0, 1.0, n_sparse + 2)[1:-1]
    x_train = np.concatenate([x_dense, x_sparse])
    x_test = _disjoint_grid(-1.0, 1.0, n_test, x_train)
    return x_train, x_test, [(0.0, 1.0)]


PROTOCOLS = {
    "edge_holdout": edge_holdout,
    "beyond_domain": beyond_domain,
    "sparse_half": sparse_half,
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/expG03_extrapolation/protocols.py tests/test_expG03_extrapolation.py
git commit -m "feat(expG03): edge/beyond/sparse hold-out protocols with disjoint grids"
```

---

### Task 3: Basis-decomposition identity test

**Files:**
- Test: `tests/test_expG03_extrapolation.py` (solver already implements it)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expG03_extrapolation.py`:

```python
def test_basis_contributions_sum_to_fit():
    """sum_k c_k*phi_k(x) + bias must equal predict(...) to fp precision."""
    N, lam = 64, 0.10
    centers, gamma_vec = solver.geometry(N, lam)
    x = np.linspace(-1.0, 1.0, N)
    y = 1.0 / (1.0 + 25.0 * x**2)
    v, bias, _ = solver.fit(x, y, centers, gamma_vec)
    x_dense = np.linspace(-1.0, 1.0, 311)
    contrib, b = solver.basis_contributions(x_dense, centers, gamma_vec, v, bias)
    recon = contrib.sum(axis=1) + b
    assert np.max(np.abs(recon - solver.predict(x_dense, centers, gamma_vec, v, bias))) < 1e-9
```

- [ ] **Step 2: Run the test to verify it passes** (solver.basis_contributions already exists from Task 1)

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py::test_basis_contributions_sum_to_fit -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_expG03_extrapolation.py
git commit -m "test(expG03): basis contributions reconstruct the fit to fp precision"
```

---

### Task 4: Sweep runner (`run.py`) writing `data.json`

**Files:**
- Create: `experiments/expG03_extrapolation/run.py`
- Test: `tests/test_expG03_extrapolation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expG03_extrapolation.py`:

```python
import run as g03run


def test_evaluate_cell_holdout_is_finite_and_harder():
    """Runge under edge_holdout at lambda=0.25: held-out rel L2 is finite and
    no better than the unmasked region."""
    rec = g03run.evaluate_cell("edge_holdout", "runge", 0.25, N=128)
    assert np.isfinite(rec["rel_l2_held"])
    assert np.isfinite(rec["coeff_norm"])
    assert rec["rel_l2_held"] >= rec["rel_l2_unmasked"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py::test_evaluate_cell_holdout_is_finite_and_harder -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write `run.py`**

```python
"""expG03 sweep driver: 3 hold-out protocols x targets x lambda.

Usage (from repo root):
    uv run --extra dev python experiments/expG03_extrapolation/run.py [--smoke] [--plot]

Writes results incrementally to
results/checkpoint_G_generalization/expG03_extrapolation/data.json
(safe to re-run: finished cells are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import solver
import protocols as P

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_G_generalization"
               / "expG03_extrapolation")

N_DEFAULT = 128
LAMBDAS = [0.25, 0.10, 0.05]
TARGETS = {
    "sine": lambda x: np.sin(2 * np.pi * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x**2),
    "exp": lambda x: np.exp(x),
}


def _rel_over(u_hat, u_true, mask):
    if mask.sum() == 0:
        return float("nan")
    return solver.rel_l2(u_hat[mask], u_true[mask])


def evaluate_cell(protocol, target, lam, N=N_DEFAULT):
    """Fit one (protocol, target, lambda) cell and return its metric record."""
    t0 = time.time()
    f = TARGETS[target]
    x_train, x_test, regions = P.PROTOCOLS[protocol]()
    centers, gamma_vec = solver.geometry(N, lam)
    v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    u_true = f(x_test)
    held = P.in_regions(x_test, regions)
    return dict(
        protocol=protocol, target=target, lam=lam, N=N,
        rel_l2_entire=solver.rel_l2(u_hat, u_true),
        rel_l2_unmasked=_rel_over(u_hat, u_true, ~held),
        rel_l2_held=_rel_over(u_hat, u_true, held),
        linf_held=float(np.max(np.abs((u_hat - u_true)[held]))) if held.sum() else float("nan"),
        coeff_norm=float(np.linalg.norm(v)),
        t_solve=time.time() - t0,
    )


def cell_key(rec):
    return (rec["protocol"], rec["target"], rec["lam"])


def load_records():
    path = RESULTS_DIR / "data.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_records(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data.json").write_text(json.dumps(records, indent=1))


def run_sweep(smoke=False):
    protocols_ = list(P.PROTOCOLS)
    targets = ["runge"] if smoke else list(TARGETS)
    lams = [0.25] if smoke else LAMBDAS
    records = load_records()
    done = {cell_key(r) for r in records}
    for protocol in protocols_:
        for target in targets:
            for lam in lams:
                if (protocol, target, lam) in done:
                    continue
                rec = evaluate_cell(protocol, target, lam)
                records.append(rec)
                save_records(records)
                print(f"{protocol} {target} lam={lam}: "
                      f"held={rec['rel_l2_held']:.2e} "
                      f"unmasked={rec['rel_l2_unmasked']:.2e} "
                      f"||v||={rec['coeff_norm']:.2e}", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        import viz
        viz.make_all_figures(load_records(), RESULTS_DIR, TARGETS, P, solver, N_DEFAULT)
        return
    records = run_sweep(smoke=args.smoke)
    import viz
    viz.make_all_figures(records, RESULTS_DIR, TARGETS, P, solver, N_DEFAULT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py::test_evaluate_cell_holdout_is_finite_and_harder -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/expG03_extrapolation/run.py tests/test_expG03_extrapolation.py
git commit -m "feat(expG03): sweep runner with region-split metrics and incremental data.json"
```

---

### Task 5: Figures (`viz.py`)

**Files:**
- Create: `experiments/expG03_extrapolation/viz.py`
- Test: `tests/test_expG03_extrapolation.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expG03_extrapolation.py`:

```python
def test_viz_writes_figures(tmp_path):
    import viz
    recs = [g03run.evaluate_cell("edge_holdout", "runge", 0.25, N=128)]
    viz.make_all_figures(recs, tmp_path, g03run.TARGETS, protocols, solver, 128)
    assert (tmp_path / "summary_held_vs_lambda.png").exists()
    assert any(tmp_path.glob("fit_*edge_holdout*runge*.png"))
    assert any(tmp_path.glob("basis_*edge_holdout*runge*.png"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py::test_viz_writes_figures -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz'`.

- [ ] **Step 3: Write `viz.py`**

```python
"""expG03 figures: fit+residual, basis contributions, summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _shade_regions(ax, regions):
    for lo, hi in regions:
        ax.axvspan(lo, hi, color="0.85", zorder=0)


def _fit_residual_fig(out_dir, protocol, target, lam, f, x_test, u_hat, regions):
    u_true = f(x_test)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    _shade_regions(a0, regions)
    a0.plot(x_test, u_true, "b-", lw=1.5, label="target")
    a0.plot(x_test, u_hat, "r-", lw=1.0, alpha=0.7, label="fit")
    a0.set_ylabel("f, f_hat")
    a0.legend(fontsize=8)
    a0.set_title(f"{protocol} / {target} / lam={lam:g}")
    _shade_regions(a1, regions)
    resid = u_true - u_hat
    a1.plot(x_test, np.sign(resid) * np.log10(np.abs(resid) + 1e-18), "k-", lw=0.8)
    a1.set_ylabel("sign(r)·log10|r|")
    a1.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"fit_{protocol}_{target}_lam{lam:g}.png", dpi=140)
    plt.close(fig)


def _basis_fig(out_dir, protocol, target, lam, f, centers, gamma_vec, v, bias,
               regions, solver):
    x = np.linspace(min(-1.3, centers.min()), max(1.3, centers.max()), 600)
    contrib, b = solver.basis_contributions(x, centers, gamma_vec, v, bias)
    in_band = np.zeros(centers.shape, dtype=bool)
    for lo, hi in regions:
        in_band |= (centers > lo - 0.05) & (centers < hi + 0.05)
    fig, ax = plt.subplots(figsize=(7, 5))
    _shade_regions(ax, regions)
    for k in range(centers.size):
        if in_band[k]:
            ax.plot(x, contrib[:, k], color="tab:red", lw=0.6, alpha=0.5, zorder=2)
        else:
            ax.plot(x, contrib[:, k], color="tab:blue", lw=0.4, alpha=0.15, zorder=1)
    ax.plot(x, contrib.sum(axis=1) + b, "k-", lw=1.6, label="sum = f_hat", zorder=3)
    ax.plot(x, f(x), "g--", lw=1.0, label="target", zorder=3)
    ax.set_title(f"basis contributions — {protocol} / {target} / lam={lam:g}\n"
                 f"(red = centers in/near held-out band)")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"basis_{protocol}_{target}_lam{lam:g}.png", dpi=140)
    plt.close(fig)


def _summary_fig(out_dir, records):
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = sorted({(r["protocol"], r["target"]) for r in records})
    for protocol, target in keys:
        cells = sorted((r for r in records
                        if r["protocol"] == protocol and r["target"] == target),
                       key=lambda r: r["lam"])
        lams = [r["lam"] for r in cells]
        held = [r["rel_l2_held"] for r in cells]
        ax.loglog(lams, held, "o-", label=f"{protocol}/{target}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("held-out rel L2")
    ax.set_title("expG03: held-out error vs lambda")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "summary_held_vs_lambda.png", dpi=140)
    plt.close(fig)


def make_all_figures(records, out_dir, targets, protocols_mod, solver, N):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        protocol, target, lam = rec["protocol"], rec["target"], rec["lam"]
        f = targets[target]
        x_train, x_test, regions = protocols_mod.PROTOCOLS[protocol]()
        centers, gamma_vec = solver.geometry(N, lam)
        v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
        u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
        _fit_residual_fig(out_dir, protocol, target, lam, f, x_test, u_hat, regions)
        _basis_fig(out_dir, protocol, target, lam, f, centers, gamma_vec, v, bias,
                   regions, solver)
    if records:
        _summary_fig(out_dir, records)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py::test_viz_writes_figures -q`
Expected: PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run --extra dev python -m pytest tests/test_expG03_extrapolation.py -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add experiments/expG03_extrapolation/viz.py tests/test_expG03_extrapolation.py
git commit -m "feat(expG03): fit/residual, basis-contribution, and summary figures"
```

---

### Task 6: Run the full sweep

**Files:** none (produces `results/checkpoint_G_generalization/expG03_extrapolation/{data.json, *.png}`)

- [ ] **Step 1: Smoke test the driver end to end**

Run: `uv run --extra dev python experiments/expG03_extrapolation/run.py --smoke`
Expected: 3 lines printed (one per protocol, target=runge, lam=0.25); no traceback; `data.json` + a few PNGs written under the results dir.

- [ ] **Step 2: Run the full sweep**

Run: `uv run --extra dev python experiments/expG03_extrapolation/run.py`
Expected: 9 lines (3 protocols x 3 targets) beyond the smoke cell — the loop skips the already-done `*/runge/0.25` cell. Full grid is 27 cells; runtime a few minutes on CPU.

- [ ] **Step 3: Sanity-read the numbers**

Run: `uv run --extra dev python -c "import json,pathlib; d=json.loads(pathlib.Path('results/checkpoint_G_generalization/expG03_extrapolation/data.json').read_text()); [print(r['protocol'], r['target'], r['lam'], f\"held={r['rel_l2_held']:.2e} unmasked={r['rel_l2_unmasked']:.2e} ||v||={r['coeff_norm']:.2e}\") for r in sorted(d, key=lambda r:(r['protocol'],r['target'],r['lam']))]"`
Expected: 27 rows. Sanity: `rel_l2_unmasked` near the fp64 floor for smooth cells; `rel_l2_held` far larger; smaller λ generally lowers held-out error (the tradeoff).

- [ ] **Step 4: Commit results**

```bash
git add results/checkpoint_G_generalization/expG03_extrapolation/*.png
git commit -m "results(expG03): first-pass extrapolation sweep figures"
```

(`data.json` is gitignored by the repo's `results/**` rule — regenerated by `run.py`.)

---

### Task 7: Writeup + checkpoint index

**Files:**
- Create: `results/checkpoint_G_generalization/expG03_extrapolation/expG03_results.md`
- Modify: `results/checkpoint_G_generalization/expG_results.md`

- [ ] **Step 1: Write `expG03_results.md`**

Use the standard format (TL;DR / Question / Experiment design / Results / Conclusions / Open questions), filled from `data.json` and the figures. It must state, with numbers read from the sweep:
- the unmasked (fp64-floor) vs held-out rel L2 gap per protocol;
- how held-out error moves with λ (does small λ tame the blowup, per Sam's note?);
- the coefficient-norm ‖v‖ blowup trend;
- what the basis-contribution figures show about which neurons carry the held-out region (and how far the influence reaches past the last center in `beyond_domain`).
Include the reproduce command and point at the figures.

- [ ] **Step 2: Add expG03 to the checkpoint index**

In `results/checkpoint_G_generalization/expG_results.md`, under `## Experiments`, add a bullet after the expG01 line:

```markdown
- **expG03 -- extrapolation & data-poor generalization (drafted).** Fixed uniform-gamma construction under three hold-out protocols (edge_holdout, beyond_domain, sparse_half) over lambda in {0.25,0.10,0.05}, with per-neuron basis-contribution visualization. Construction-only first pass; Adam/cascade baselines deferred. Writeup: `expG03_extrapolation/expG03_results.md`.
```

- [ ] **Step 3: Commit**

```bash
git add results/checkpoint_G_generalization/expG03_extrapolation/expG03_results.md results/checkpoint_G_generalization/expG_results.md
git commit -m "docs(expG03): results writeup + checkpoint G index entry"
```

---

## Self-Review Notes

- **Spec coverage:** three protocols (T2), λ-sweep + targets + region-split metrics + ‖v‖ (T4), basis-contribution + fit/residual + summary figures (T5), no-holdout fp64 sanity (T1), disjointness + basis-sum tests (T2/T3), sweep run (T6), writeup + index (T7). Adam/cascade baselines are explicitly deferred in the spec and not tasked.
- **Type consistency:** `geometry` → `(centers, gamma_vec)`; `fit` → `(v, bias, info)`; `basis_contributions` → `(contrib[n,W], bias)`; records carry `rel_l2_{entire,unmasked,held}`, `linf_held`, `coeff_norm`; `protocols.PROTOCOLS` and `TARGETS` are the registries used by both `run.py` and `viz.py`. `viz.make_all_figures(records, out_dir, targets, protocols_mod, solver, N)` signature matches both call sites (run.py `--plot` and the test).
- **Known wrinkle:** `beyond_domain` evaluates the fit at |x|>1 where no centers live; `basis_contributions`/`predict` handle it (tanh saturates), and `_basis_fig` widens the x-range to show the decay. `sparse_half` is data-poor not data-free, so its disjointness assertion allows ≤ n_sparse train points in the region (T2).
