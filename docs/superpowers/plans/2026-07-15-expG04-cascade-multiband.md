# expG04 Cascade Multi-Band Geometry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch experiment testing whether a hand-built coarsening cascade (sharp full grid + progressively coarser soft bands, one stacked SVD readout) keeps the fp64 floor on the trained region while extrapolating better than any single band.

**Architecture:** Reuse expG03 wholesale — import its `protocols.py` and `solver.py` (which operate on arbitrary `(centers, gamma_vec)`). Add `cascade.py` that concatenates per-band expG03 geometries; `run.py` ablates the band count {1,2,3} over expG03's protocols×targets; `viz.py` colors basis contributions by band. n_bands=1 is an exact reproduction of expG03's λ=0.25 single-band baseline.

**Tech Stack:** Python, numpy, matplotlib; tests/scripts via `uv run --extra dev`.

**Spec:** `docs/superpowers/specs/2026-07-15-expG04-cascade-multiband-design.md`

**Reused (import, do not copy):** `experiments/expG03_extrapolation/solver.py` (`geometry`, `fit`, `predict`, `basis_contributions`, `rel_l2`, `linf`) and `protocols.py` (`PROTOCOLS`, `in_regions`, `edge_holdout`/…).

**Verified band sizes** (`N=128`, coarsen=2): band0 λ=0.25 → 269 centers; band1 λ=0.10 → 205; band2 λ=0.05 → 173; total 3-band = 647.

---

## File Structure

- Create `experiments/expG04_cascade_multiband/__init__.py` — empty package marker.
- Create `experiments/expG04_cascade_multiband/cascade.py` — `cascade_geometry(N, lambdas, coarsen)` stacking per-band expG03 geometries; imports expG03 `solver`.
- Create `experiments/expG04_cascade_multiband/run.py` — target registry, `evaluate_cell`, band-count sweep, `data.json`, `--smoke`/`--plot`.
- Create `experiments/expG04_cascade_multiband/viz.py` — fit+residual, band-colored basis, summary-vs-n_bands.
- Create `tests/test_expG04_cascade_multiband.py` — the four spec tests.
- Create `results/checkpoint_G_generalization/expG04_cascade_multiband/expG04_results.md` — writeup (Task 5).
- Modify `results/checkpoint_G_generalization/expG_results.md` — list expG04 (Task 5).

All commands run from the repo root `/scr/cdeng/precision-mlps`.

---

### Task 1: Cascade geometry (`cascade.py`)

**Files:**
- Create: `experiments/expG04_cascade_multiband/__init__.py`
- Create: `experiments/expG04_cascade_multiband/cascade.py`
- Test: `tests/test_expG04_cascade_multiband.py`

- [ ] **Step 1: Create the package marker**

Create the empty file `experiments/expG04_cascade_multiband/__init__.py` (no content).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_expG04_cascade_multiband.py`:

```python
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG04_cascade_multiband"))

import solver          # expG03 numeric core
import cascade


def test_n_bands_1_reproduces_expg03_geometry():
    """A single-band cascade must equal expG03's uniform geometry, and a dense
    fit of sine must reach the fp64 floor."""
    c1, g1, b1 = cascade.cascade_geometry(128, [0.25], coarsen=2)
    c0, g0 = solver.geometry(128, 0.25)
    assert np.array_equal(c1, c0) and np.array_equal(g1, g0)
    assert set(np.unique(b1)) == {0}
    x = np.linspace(-1.0, 1.0, 400)
    f = lambda t: np.sin(2 * np.pi * t)
    v, bias, _ = solver.fit(x, f(x), c1, g1)
    xt = np.linspace(-1.0, 1.0, 257)
    assert solver.rel_l2(solver.predict(xt, c1, g1, v, bias), f(xt)) < 1e-12


def test_cascade_concatenation_and_band_index():
    """Three bands concatenate to the known per-band sizes with a matching
    band index; softer bands have fewer in-grid centers."""
    c, g, b = cascade.cascade_geometry(128, [0.25, 0.10, 0.05], coarsen=2)
    counts = [int((b == k).sum()) for k in range(3)]
    assert counts == [269, 205, 173]
    assert c.size == g.size == b.size == 647
    # per-band gamma is constant and strictly decreasing across bands
    gammas = [g[b == k][0] for k in range(3)]
    assert gammas[0] > gammas[1] > gammas[2]
    assert all(np.allclose(g[b == k], gammas[k]) for k in range(3))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'cascade'`.

- [ ] **Step 4: Write `cascade.py`**

```python
"""expG04 cascade multi-band geometry.

Stacks single-band expG03 geometries at decreasing bandwidth and increasing
center spacing into one (centers, gamma_vec) pair, plus a band index. The
concatenated geometry is a drop-in for expG03's solver.{fit,predict,
basis_contributions}, so a cascade is solved by the same stacked SVD readout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver  # expG03 numeric core


def cascade_geometry(N, lambdas, coarsen=2):
    """Multi-band geometry: band k is expG03's uniform geometry on grid
    N // coarsen**k at bandwidth lambdas[k] (sharp band = full N; softer bands
    coarser). Returns (centers, gamma_vec, band_idx), all length = total width.
    """
    centers_list, gamma_list, band_list = [], [], []
    for k, lam in enumerate(lambdas):
        Nk = max(4, N // coarsen**k)
        c, g = solver.geometry(Nk, lam)
        centers_list.append(c)
        gamma_list.append(g)
        band_list.append(np.full(c.size, k, dtype=int))
    return (np.concatenate(centers_list),
            np.concatenate(gamma_list),
            np.concatenate(band_list))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add experiments/expG04_cascade_multiband/__init__.py experiments/expG04_cascade_multiband/cascade.py tests/test_expG04_cascade_multiband.py
git commit -m "feat(expG04): cascade multi-band geometry (stacked per-band expG03 geometries)"
```

---

### Task 2: Sweep runner (`run.py`) with per-band norms

**Files:**
- Create: `experiments/expG04_cascade_multiband/run.py`
- Test: `tests/test_expG04_cascade_multiband.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expG04_cascade_multiband.py`:

```python
import run as g04run


def test_evaluate_cell_precision_preserved_and_per_band():
    """3-band cascade keeps the trained region near the floor, splits ||v|| by
    band, and the held-out region is finite and no easier than unmasked."""
    rec = g04run.evaluate_cell(3, "edge_holdout", "sine")
    assert rec["rel_l2_unmasked"] < 1e-8            # precision preserved
    assert np.isfinite(rec["rel_l2_held"])
    assert rec["rel_l2_held"] >= rec["rel_l2_unmasked"]
    assert sorted(rec["per_band_norm"]) == [0, 1, 2]
    assert np.isclose(
        np.sqrt(sum(n**2 for n in rec["per_band_norm"].values())),
        rec["coeff_norm"], rtol=1e-9)


def test_evaluate_cell_basis_sum_identity():
    """With a cascade geometry, sum_k c_k*phi_k + bias == predict(...)."""
    c, g, _ = g04run.C.cascade_geometry(128, g04run.LAMBDAS, g04run.COARSEN)
    x = np.linspace(-1.0, 1.0, 300)
    y = 1.0 / (1.0 + 25.0 * x**2)
    v, bias, _ = solver.fit(x, y, c, g)
    xd = np.linspace(-1.3, 1.3, 411)
    contrib, b = solver.basis_contributions(xd, c, g, v, bias)
    assert np.max(np.abs(contrib.sum(axis=1) + b
                         - solver.predict(xd, c, g, v, bias))) < 1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write `run.py`**

```python
"""expG04 sweep driver: band-count ablation x protocols x targets.

Usage (from repo root):
    uv run --extra dev python experiments/expG04_cascade_multiband/run.py [--smoke] [--plot]

Writes results incrementally to
results/checkpoint_G_generalization/expG04_cascade_multiband/data.json
(safe to re-run: finished cells are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expG03_extrapolation"))

import solver
import protocols as P
import cascade as C

RESULTS_DIR = (REPO_ROOT / "results" / "checkpoint_G_generalization"
               / "expG04_cascade_multiband")

N_DEFAULT = 128
COARSEN = 2
LAMBDAS = [0.25, 0.10, 0.05]
N_BANDS = [1, 2, 3]
TARGETS = {
    "sine": lambda x: np.sin(2 * np.pi * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x**2),
    "exp": lambda x: np.exp(x),
}


def _rel_over(u_hat, u_true, mask):
    if mask.sum() == 0:
        return float("nan")
    return solver.rel_l2(u_hat[mask], u_true[mask])


def evaluate_cell(n_bands, protocol, target, N=N_DEFAULT, coarsen=COARSEN):
    """Fit one (n_bands, protocol, target) cell and return its metric record."""
    t0 = time.time()
    f = TARGETS[target]
    x_train, x_test, regions = P.PROTOCOLS[protocol]()
    centers, gamma_vec, band_idx = C.cascade_geometry(N, LAMBDAS[:n_bands], coarsen)
    v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
    u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
    u_true = f(x_test)
    held = P.in_regions(x_test, regions)
    per_band = {int(k): float(np.linalg.norm(v[band_idx == k]))
                for k in range(n_bands)}
    return dict(
        n_bands=n_bands, protocol=protocol, target=target, N=N, coarsen=coarsen,
        rel_l2_entire=solver.rel_l2(u_hat, u_true),
        rel_l2_unmasked=_rel_over(u_hat, u_true, ~held),
        rel_l2_held=_rel_over(u_hat, u_true, held),
        linf_held=float(np.max(np.abs((u_hat - u_true)[held]))) if held.sum() else float("nan"),
        coeff_norm=float(np.linalg.norm(v)),
        per_band_norm=per_band,
        t_solve=time.time() - t0,
    )


def cell_key(rec):
    return (rec["n_bands"], rec["protocol"], rec["target"])


def load_records():
    path = RESULTS_DIR / "data.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_records(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data.json").write_text(json.dumps(records, indent=1))


def run_sweep(smoke=False):
    nbands = [3] if smoke else N_BANDS
    targets = ["runge"] if smoke else list(TARGETS)
    records = load_records()
    done = {cell_key(r) for r in records}
    for n_bands in nbands:
        for protocol in P.PROTOCOLS:
            for target in targets:
                if (n_bands, protocol, target) in done:
                    continue
                rec = evaluate_cell(n_bands, protocol, target)
                records.append(rec)
                save_records(records)
                print(f"nb={n_bands} {protocol} {target}: "
                      f"held={rec['rel_l2_held']:.2e} "
                      f"unmasked={rec['rel_l2_unmasked']:.2e} "
                      f"||v||={rec['coeff_norm']:.2e} "
                      f"bands={ {k: round(x,3) for k,x in rec['per_band_norm'].items()} }",
                      flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        import viz
        viz.make_all_figures(load_records(), RESULTS_DIR, TARGETS, P, C, solver,
                             N_DEFAULT, LAMBDAS, COARSEN)
        return
    records = run_sweep(smoke=args.smoke)
    import viz
    viz.make_all_figures(records, RESULTS_DIR, TARGETS, P, C, solver,
                         N_DEFAULT, LAMBDAS, COARSEN)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add experiments/expG04_cascade_multiband/run.py tests/test_expG04_cascade_multiband.py
git commit -m "feat(expG04): band-count sweep with region-split metrics and per-band norms"
```

---

### Task 3: Figures (`viz.py`)

**Files:**
- Create: `experiments/expG04_cascade_multiband/viz.py`
- Test: `tests/test_expG04_cascade_multiband.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expG04_cascade_multiband.py`:

```python
def test_viz_writes_figures(tmp_path):
    import viz
    recs = [g04run.evaluate_cell(3, "edge_holdout", "runge"),
            g04run.evaluate_cell(1, "edge_holdout", "runge")]
    viz.make_all_figures(recs, tmp_path, g04run.TARGETS, __import__("protocols"),
                         cascade, solver, 128, g04run.LAMBDAS, g04run.COARSEN)
    assert (tmp_path / "summary_held_vs_nbands.png").exists()
    assert any(tmp_path.glob("basis_nb3_edge_holdout_runge.png"))
    assert any(tmp_path.glob("fit_nb3_edge_holdout_runge.png"))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py::test_viz_writes_figures -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'viz'`.

- [ ] **Step 3: Write `viz.py`**

```python
"""expG04 figures: fit+residual, band-colored basis contributions, summary."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAND_COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
BAND_NAMES = ["sharp", "mid", "soft", "band3"]


def _shade_regions(ax, regions):
    for lo, hi, *_ in regions:
        ax.axvspan(lo, hi, color="0.85", zorder=0)


def _fit_residual_fig(out_dir, n_bands, protocol, target, f, x_test, u_hat, regions):
    u_true = f(x_test)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    _shade_regions(a0, regions)
    a0.plot(x_test, u_true, "b-", lw=1.5, label="target")
    a0.plot(x_test, u_hat, "r-", lw=1.0, alpha=0.7, label="fit")
    a0.set_ylabel("f, f_hat")
    a0.legend(fontsize=8)
    a0.set_title(f"nb={n_bands} / {protocol} / {target}")
    _shade_regions(a1, regions)
    resid = u_true - u_hat
    a1.plot(x_test, np.sign(resid) * np.log10(np.abs(resid) + 1e-18), "k-", lw=0.8)
    a1.set_ylabel("sign(r)·log10|r|")
    a1.set_xlabel("x")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"fit_nb{n_bands}_{protocol}_{target}.png", dpi=140)
    plt.close(fig)


def _basis_band_fig(out_dir, n_bands, protocol, target, f, centers, gamma_vec,
                    band_idx, v, bias, regions, solver):
    x = np.linspace(min(-1.3, centers.min()), max(1.3, centers.max()), 600)
    contrib, b = solver.basis_contributions(x, centers, gamma_vec, v, bias)
    fig, ax = plt.subplots(figsize=(7, 5))
    _shade_regions(ax, regions)
    for k in range(n_bands):
        cols = np.where(band_idx == k)[0]
        for j in cols:
            ax.plot(x, contrib[:, j], color=BAND_COLORS[k], lw=0.4, alpha=0.2,
                    zorder=1 + k)
        # one labeled proxy line per band
        ax.plot([], [], color=BAND_COLORS[k], lw=1.2,
                label=f"{BAND_NAMES[k]} band")
    ax.plot(x, contrib.sum(axis=1) + b, "k-", lw=1.6, label="sum = f_hat", zorder=9)
    ax.plot(x, f(x), "g--", lw=1.0, label="target", zorder=9)
    ax.set_title(f"basis by band — nb={n_bands} / {protocol} / {target}")
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"basis_nb{n_bands}_{protocol}_{target}.png", dpi=140)
    plt.close(fig)


def _summary_fig(out_dir, records):
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = sorted({(r["protocol"], r["target"]) for r in records})
    for protocol, target in keys:
        cells = sorted((r for r in records
                        if r["protocol"] == protocol and r["target"] == target),
                       key=lambda r: r["n_bands"])
        nb = [r["n_bands"] for r in cells]
        held = [r["rel_l2_held"] for r in cells]
        ax.semilogy(nb, held, "o-", label=f"{protocol}/{target}")
    ax.set_xlabel("n_bands")
    ax.set_ylabel("held-out rel L2")
    ax.set_xticks([1, 2, 3])
    ax.set_title("expG04: held-out error vs number of bands")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "summary_held_vs_nbands.png", dpi=140)
    plt.close(fig)


def make_all_figures(records, out_dir, targets, protocols_mod, cascade_mod,
                     solver, N, lambdas, coarsen):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        n_bands, protocol, target = rec["n_bands"], rec["protocol"], rec["target"]
        f = targets[target]
        x_train, x_test, regions = protocols_mod.PROTOCOLS[protocol]()
        centers, gamma_vec, band_idx = cascade_mod.cascade_geometry(
            N, lambdas[:n_bands], coarsen)
        v, bias, _ = solver.fit(x_train, f(x_train), centers, gamma_vec)
        u_hat = solver.predict(x_test, centers, gamma_vec, v, bias)
        _fit_residual_fig(out_dir, n_bands, protocol, target, f, x_test, u_hat, regions)
        _basis_band_fig(out_dir, n_bands, protocol, target, f, centers, gamma_vec,
                        band_idx, v, bias, regions, solver)
    if records:
        _summary_fig(out_dir, records)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py::test_viz_writes_figures -q`
Expected: PASS.

- [ ] **Step 5: Run the full test file**

Run: `uv run --extra dev python -m pytest tests/test_expG04_cascade_multiband.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add experiments/expG04_cascade_multiband/viz.py tests/test_expG04_cascade_multiband.py
git commit -m "feat(expG04): fit/residual, band-colored basis, and summary-vs-nbands figures"
```

---

### Task 4: Run the sweep

**Files:** none (produces `results/checkpoint_G_generalization/expG04_cascade_multiband/{data.json, *.png}`)

- [ ] **Step 1: Smoke test the driver end to end**

Run: `uv run --extra dev python experiments/expG04_cascade_multiband/run.py --smoke`
Expected: 3 lines (nb=3, target=runge, one per protocol); no traceback; `data.json` + PNGs written.

- [ ] **Step 2: Run the full sweep**

Run: `uv run --extra dev python experiments/expG04_cascade_multiband/run.py`
Expected: the remaining cells of the 27-cell grid (3 n_bands x 3 protocols x 3 targets); runtime a few minutes on CPU. The already-done smoke cells are skipped.

- [ ] **Step 3: Sanity-read the numbers**

Run: `uv run --extra dev python -c "import json,pathlib; d=json.loads(pathlib.Path('results/checkpoint_G_generalization/expG04_cascade_multiband/data.json').read_text()); [print(r['n_bands'], r['protocol'], r['target'], f\"held={r['rel_l2_held']:.2e} unmasked={r['rel_l2_unmasked']:.2e} ||v||={r['coeff_norm']:.2e}\") for r in sorted(d, key=lambda r:(r['protocol'],r['target'],r['n_bands']))]"`
Expected: 27 rows. Sanity: `rel_l2_unmasked` near the fp64 floor at all n_bands (precision preserved); for runge, held-out should drop as bands are added *without* ||v|| exploding (contrast expG03's single-band low-λ blowup).

- [ ] **Step 4: Commit results**

```bash
git add results/checkpoint_G_generalization/expG04_cascade_multiband/*.png
git commit -m "results(expG04): cascade multi-band ablation figures"
```

(`data.json` is gitignored by the repo's `results/**` rule — regenerated by `run.py`.)

---

### Task 5: Writeup + checkpoint index

**Files:**
- Create: `results/checkpoint_G_generalization/expG04_cascade_multiband/expG04_results.md`
- Modify: `results/checkpoint_G_generalization/expG_results.md`

- [ ] **Step 1: Write `expG04_results.md`**

Use the standard format (TL;DR / Question / Experiment design / Results / Conclusions / Open questions), filled from `data.json` and the figures. It must state, with numbers read from the sweep:
- whether precision on the trained region survives as bands are added (unmasked rel L2 vs n_bands);
- whether held-out error drops with more bands, per protocol/target — and specifically whether the cascade tames the Runge blowup that single-band low-λ suffered in expG03 (compare to expG03's runge numbers), and whether it recovers (or fails to recover) the smooth-target extrapolation a pure soft band gave;
- what the per-band coefficient norms and band-colored basis figures show about which scale carries the fit vs the held-out region.
Include the reproduce command and point at the figures. A mixed/null result is a valid finding — report it plainly.

- [ ] **Step 2: Add expG04 to the checkpoint index**

In `results/checkpoint_G_generalization/expG_results.md`, under `## Experiments`, add a bullet after the expG03 line:

```markdown
- **expG04 -- cascade multi-band geometry (drafted).** Hand-built coarsening cascade (sharp full grid at lambda=0.25 + coarser soft bands at 0.10, 0.05) solved by one stacked SVD readout; band-count ablation {1,2,3} over expG03's protocols/targets, per-band coefficient norms + band-colored basis viz. Tests whether multi-band keeps the fp64 floor while extrapolating better than any single band. Writeup: `expG04_cascade_multiband/expG04_results.md`.
```

- [ ] **Step 3: Commit**

```bash
git add results/checkpoint_G_generalization/expG04_cascade_multiband/expG04_results.md results/checkpoint_G_generalization/expG_results.md
git commit -m "docs(expG04): results writeup + checkpoint G index entry"
```

---

## Self-Review Notes

- **Spec coverage:** cascade geometry with coarsening + fixed-reference halo (T1); band-count ablation sweep + region-split metrics + per-band norms (T2); n_bands=1 == expG03 reproduction and precision-preserved tests (T1/T2); basis-sum identity (T2); fit/residual + band-colored basis + summary-vs-nbands figures (T3); sweep run (T4); writeup + index (T5). Coarsen sweep and Adam baseline are explicitly deferred in the spec and not tasked.
- **Type consistency:** `cascade_geometry` → `(centers, gamma_vec, band_idx)` everywhere; records carry `rel_l2_{entire,unmasked,held}`, `linf_held`, `coeff_norm`, `per_band_norm` (dict keyed by int band); `viz.make_all_figures(records, out_dir, targets, protocols_mod, cascade_mod, solver, N, lambdas, coarsen)` matches both call sites (run.py `--plot`/post-sweep and the test). `run.py` exposes `C` (cascade module), `LAMBDAS`, `COARSEN`, `TARGETS`, `evaluate_cell` used by the tests.
- **Known wrinkle (verified):** 3 bands = 647 columns, but expG03's training densities (300/400/250) still reach the floor on the trained region (checked: sine/edge unmasked 3e-13 at n_bands=3), so no density bump is needed. The interesting signal is already visible in the pre-check: cascade tames the Runge edge blowup (single-band low-λ ~3e5 → cascade ~5e-2, ||v|| bounded) but does not recover smooth-target extrapolation — the writeup (T5) must report this mixed result.
