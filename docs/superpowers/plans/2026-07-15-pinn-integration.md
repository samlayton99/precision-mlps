# PINN Integration (expF05/expF06/expF07) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three checkpoint-F experiments: B-spline ridge basis with adaptive knots (expF05), Newton-lstsq for steady 2D Burgers (expF06), and an lstsq precision-finisher for a trained torch PINN (expF07).

**Architecture:** A generalized frozen-ridge collocation core (`ridge_core.py`, living in expF05, imported by F03/F04 via the repo's sys.path pattern) parameterizes the univariate family (tanh vs cubic B-spline) and allows per-neuron γ. Nonlinearity is handled by damped Newton where each step is one block lstsq; the PINN finisher is the same Newton loop warm-started at a frozen torch net supplied as `base_fields`.

**Tech Stack:** numpy/scipy (solves), torch (PINN only), matplotlib (plots), pytest. Run everything with `uv run --extra dev ...` from `/scr/cdeng/precision-mlps`.

**Spec:** `docs/superpowers/specs/2026-07-15-pinn-integration-design.md`

## Context primer (read first)

- Repo: `/scr/cdeng/precision-mlps`. Experiments are self-contained dirs `experiments/expF0N_<name>/` with `run.py` (+ helper modules). Results go to `results/checkpoint_F_applications/expF0N_<name>/`. Every `run.py` supports `--smoke` (tiny grids) and `--plot` (regenerate figures from data.json only).
- Tests live in `tests/test_expF0N_<name>.py`, run via `uv run --extra dev pytest tests/<file> -v`. Mark tests >10s with `@pytest.mark.slow`.
- The reference implementation of the tanh solver is `experiments/expF01_linear_de_zoo/run.py` and (cleaner) `/scr/cdeng/continuous-mlps/experiments/precision_pde/core.py`. This plan vendors and generalizes that core — you do not need to read those files; all code is in this plan.
- Operator convention: a "terms" list `[((ax, ay), coeff)]` means L = Σ coeff · ∂^ax_x ∂^ay_y, where coeff is a float, a callable `P[n,2] -> [n]`, or (new in this plan) a precomputed `[n]` array.
- Darcy data npz: `/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_test_421_jax.npz` (421×421, cell-centered grid, keys `x`/`y` or `a`/`u`). Known baselines from the July-14 continuous-mlps sweep: dense tanh W=2304 on rough instance 0 at σ=0 → rel-L2 ~7.2e-2; presmoothed σ=4 → ~2.8e-3; smooth control → 3.0e-14.
- Commit style: `expF05: <what>`.

## File structure

```
experiments/expF05_spline_ridge/
  ridge_core.py     generalized solver core (families, geometry, rows, lstsq, eval)
  problems.py       Poisson + smooth-Darcy-control manufactured problems (FD-verified)
  darcy_data.py     darcy_421 npz loader + spline coefficient surrogate (vendored)
  adaptive.py       residual->knot insertion + per-neuron gamma from local spacing
  run.py            Part A (tanh vs spline floor) + Part B (--adaptive rough Darcy)
experiments/expF06_newton_burgers/
  problems.py       Taylor-Green manufactured Burgers fields + forcing (FD-verified)
  newton.py         fields abstraction + damped Newton block-lstsq loop
  run.py            nu x W sweep, convergence/floor plots
experiments/expF07_pinn_finisher/
  pinn.py           torch tanh-MLP PINN, residual loss, Adam trainer, fields adapter
  run.py            train -> plateau -> Newton polish (reuses expF06 newton), plots
tests/
  test_expF05_spline_ridge.py
  test_expF06_newton_burgers.py
  test_expF07_pinn_finisher.py
results/checkpoint_F_applications/expF0{2,3,4}_*/   data.json + PNGs (+ PINN ckpt)
```

---

### Task 1: expF05 ridge_core.py (generalized solver core)

**Files:**
- Create: `experiments/expF05_spline_ridge/__init__.py` (empty), `experiments/expF05_spline_ridge/ridge_core.py`
- Test: `tests/test_expF05_spline_ridge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expF05_spline_ridge.py
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))

import ridge_core as rc


def _fd_family(family, order, Z, h=1e-6):
    """Central-difference d/dz of family(order-1, .) as a check of family(order, .)."""
    return (family(order - 1, Z + h) - family(order - 1, Z - h)) / (2 * h)


@pytest.mark.parametrize("family", [rc.tanh_family, rc.bspline_family])
@pytest.mark.parametrize("order", [1, 2])
def test_family_derivatives_match_fd(family, order):
    rng = np.random.default_rng(0)
    Z = rng.uniform(-1.8, 1.8, (64,))
    Z += 0.013  # keep away from spline knots at integers
    ours = family(order, Z)
    fd = _fd_family(family, order, Z)
    assert np.max(np.abs(ours - fd)) < 5e-5


def test_bspline_third_derivative_spot_values():
    # C^2 spline: order-3 is piecewise constant. B''' = 3*sgn(z) inner, -sgn(z) outer.
    Z = np.array([0.5, -0.5, 1.5, -1.5, 2.5])
    out = rc.bspline_family(3, Z)
    assert np.allclose(out, [3.0, -3.0, -1.0, 1.0, 0.0])


def test_bspline_compact_support_and_continuity():
    assert np.all(rc.bspline_family(0, np.array([2.0, -2.0, 3.0])) == 0.0)
    for order in [0, 1, 2]:
        eps = 1e-9
        for knot in [1.0, 2.0]:
            lo = rc.bspline_family(order, np.array([knot - eps]))
            hi = rc.bspline_family(order, np.array([knot + eps]))
            assert abs(lo[0] - hi[0]) < 1e-6, (order, knot)


@pytest.mark.parametrize("family,tol", [(rc.tanh_family, 1e-4), (rc.bspline_family, 1e-3)])
def test_solve_poisson_sanity(family, tol):
    # -lap u = f with u* = sin(pi x) sin(pi y), f = 2 pi^2 u*; u* = 0 on boundary.
    ustar = lambda P: np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1])
    forcing = lambda P: 2 * np.pi**2 * ustar(P)
    terms = [((2, 0), -1.0), ((0, 2), -1.0)]
    Pb = rc.boundary_points_square(200)
    bc = [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]
    model = rc.solve_square(terms, forcing, bc, W=256, lam=0.25, family=family)
    Pe = np.stack(np.meshgrid(np.linspace(-0.98, 0.98, 40),
                              np.linspace(-0.98, 0.98, 40)), -1).reshape(-1, 2)
    err = rc.rel_l2(rc.eval_model(model, Pe), ustar(Pe))
    assert err < tol, err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /scr/cdeng/precision-mlps && uv run --extra dev pytest tests/test_expF05_spline_ridge.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'ridge_core'`

- [ ] **Step 3: Write ridge_core.py**

```python
# experiments/expF05_spline_ridge/ridge_core.py
"""Frozen-ridge collocation-lstsq core on [-1,1]^2, generalized over the
univariate family and per-neuron gamma.

u(p) = sum_m c_m phi(gamma_m (w_m . p - t_m)) + poly_deg<=3(p).
family(order, Z) returns d^order/dz^order phi(z) elementwise. gammas is [M].
All coefficients from ONE min-norm lstsq over stacked PDE + BC rows.
Vendored/generalized from expF01 (and continuous-mlps precision_pde/core.py).

Operator terms: [((ax, ay), coeff)], coeff a float, callable(P)->[n], or [n] array.
"""
from __future__ import annotations

import numpy as np

RCOND = 1e-13
COLLAR_SQUARE = 1.6

MONO_2D = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
           (3, 0), (2, 1), (1, 2), (0, 3)]


def tanh_family(order, Z):
    t = np.tanh(Z)
    if order == 0:
        return t
    if order == 1:
        return 1.0 - t * t
    if order == 2:
        return -2.0 * t * (1.0 - t * t)
    if order == 3:
        s = 1.0 - t * t
        return -2.0 * s * (1.0 - 3.0 * t * t)
    raise ValueError(order)


def bspline_family(order, Z):
    """Cubic B-spline bump: support [-2,2], C^2, closed-form derivatives.
    z in [0,1]: B = 2/3 - z^2 + z^3/2;  z in [1,2]: B = (2-z)^3/6; even in z."""
    z = np.abs(Z)
    sgn = np.sign(Z)
    inner = z <= 1.0
    outer = (z > 1.0) & (z < 2.0)
    out = np.zeros_like(np.asarray(Z, dtype=np.float64))
    zi, zo = z[inner], z[outer]
    if order == 0:
        out[inner] = 2.0 / 3.0 - zi**2 + 0.5 * zi**3
        out[outer] = (2.0 - zo)**3 / 6.0
    elif order == 1:
        out[inner] = (-2.0 * zi + 1.5 * zi**2) * sgn[inner]
        out[outer] = -0.5 * (2.0 - zo)**2 * sgn[outer]
    elif order == 2:
        out[inner] = -2.0 + 3.0 * zi
        out[outer] = 2.0 - zo
    elif order == 3:
        out[inner] = 3.0 * sgn[inner]
        out[outer] = -1.0 * sgn[outer]
    else:
        raise ValueError(order)
    return out


def _ffact(d, o):
    out = 1
    for k in range(o):
        out *= (d - k)
    return out


def _coeff_col(coeff, pts):
    """Coefficient -> column [n,1], scalar float, for a term at pts."""
    if callable(coeff):
        return np.asarray(coeff(pts), dtype=np.float64).reshape(-1, 1)
    arr = np.asarray(coeff, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return arr.reshape(-1, 1)


def pi_thetas(J):
    return np.pi * (np.arange(J) + 0.5) / J


def radon_geometry(W, lam, collar=COLLAR_SQUARE):
    """Uniform Radon tensor geometry. Returns dirs [M,2], offs [M], gammas [M]."""
    J = int(round(np.sqrt(W)))
    M = W // J
    thetas = pi_thetas(J)
    ts = np.linspace(-collar, collar, M)
    dirs = np.repeat(np.stack([np.cos(thetas), np.sin(thetas)], axis=1), M, axis=0)
    offs = np.tile(ts, J)
    h_ref = 2.8 / np.sqrt(J * M)
    gammas = np.full(len(offs), lam / h_ref)
    return dirs, offs, gammas


def rows_2d(P, dirs, offs, gammas, terms, family):
    """[L Phi | L poly] rows at P."""
    Z = (P @ dirs.T - offs[None, :]) * gammas[None, :]
    A = np.zeros_like(Z)
    polys = np.zeros((len(P), len(MONO_2D)))
    x, y = P[:, 0], P[:, 1]
    for (ax, ay), coeff in terms:
        o = ax + ay
        cc = _coeff_col(coeff, P)
        dir_fac = (dirs[:, 0] ** ax * dirs[:, 1] ** ay)[None, :]
        A += cc * (gammas[None, :] ** o) * dir_fac * family(o, Z)
        ccr = cc.ravel() if np.ndim(cc) else cc
        for k, (px, py) in enumerate(MONO_2D):
            if ax <= px and ay <= py:
                mono = (_ffact(px, ax) * _ffact(py, ay)
                        * x ** (px - ax) * y ** (py - ay))
                polys[:, k] += ccr * mono
    return np.hstack([A, polys])


def boundary_points_square(n_per_edge=480):
    s = np.linspace(-1.0, 1.0, n_per_edge)
    edges = [np.stack([s, np.full_like(s, -1.0)], axis=1),
             np.stack([s, np.full_like(s, 1.0)], axis=1),
             np.stack([np.full_like(s, -1.0), s], axis=1),
             np.stack([np.full_like(s, 1.0), s], axis=1)]
    return np.concatenate(edges, axis=0)


def interior_points_square(n_feat, rng):
    n = max(5 * n_feat, 2000)
    return rng.uniform(-1.0, 1.0, (n, 2))


def solve_square(terms, forcing, bc_blocks, W=None, lam=None, family=tanh_family,
                 seed=42, geometry=None):
    """One stacked min-norm lstsq. Either pass W+lam (uniform Radon geometry) or
    geometry=(dirs, offs, gammas) explicitly (adaptive knots)."""
    rng = np.random.default_rng(seed)
    if geometry is None:
        dirs, offs, gammas = radon_geometry(W, lam)
    else:
        dirs, offs, gammas = geometry
    P = interior_points_square(len(offs), rng)
    A_pde = rows_2d(P, dirs, offs, gammas, terms, family)
    y_pde = forcing(P) if callable(forcing) else np.full(len(P), float(forcing))
    s = np.abs(A_pde).max()
    rows, vals = [A_pde / s], [y_pde / s]
    for blk in bc_blocks:
        Pb = np.asarray(blk["points"], dtype=np.float64)
        r = rows_2d(Pb, dirs, offs, gammas, blk["terms"], family)
        w = np.sqrt(len(P) / len(Pb))
        rows.append(w * r)
        vals.append(w * np.asarray(blk["values"], dtype=np.float64))
    A = np.vstack(rows)
    y = np.concatenate(vals)
    sol = np.linalg.lstsq(A, y, rcond=RCOND)[0]
    return dict(dirs=dirs, offs=offs, gammas=gammas, sol=sol, family=family)


def eval_model(model, P, terms=(((0, 0), 1.0),), chunk=4096):
    """Evaluate L[u_hat] at P; default L = identity."""
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        R = rows_2d(P[i:i + chunk], model["dirs"], model["offs"],
                    model["gammas"], list(terms), model["family"])
        out[i:i + chunk] = R @ model["sol"]
    return out


def rel_l2(u_hat, u_true):
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


def linf(u_hat, u_true):
    return float(np.max(np.abs(u_hat - u_true)))
```

Also create the empty `experiments/expF05_spline_ridge/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py -v`
Expected: all PASS. If `test_solve_poisson_sanity[bspline_family]` fails on tolerance (not on error), record the actual number — that is the spec's spline-conditioning risk. Loosen ONLY that test to `5e-3` and note it in the commit message; if worse than 5e-3, stop and report.

- [ ] **Step 5: Commit**

```bash
git add experiments/expF05_spline_ridge/__init__.py experiments/expF05_spline_ridge/ridge_core.py tests/test_expF05_spline_ridge.py
git commit -m "expF05: generalized ridge core (tanh + cubic B-spline families, per-neuron gamma)"
```

---

### Task 2: expF05 problems.py (Poisson + smooth Darcy control)

**Files:**
- Create: `experiments/expF05_spline_ridge/problems.py`
- Test: append to `tests/test_expF05_spline_ridge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expF05_spline_ridge.py`:

```python
def test_problems_fd_verified():
    import problems
    problems.verify_all()  # raises AssertionError on any FD mismatch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py::test_problems_fd_verified -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'problems'`

- [ ] **Step 3: Write problems.py**

The control problem is vendored verbatim from continuous-mlps `precision_pde/darcy_problems.py` (a = 3 + exp(sin πx sin πy), u* = sin πx sin πy + 0.5 sin 2πx sin πy, u* = 0 on the boundary). Poisson reuses the same u*.

```python
# experiments/expF05_spline_ridge/problems.py
"""Part-A problems on [-1,1]^2, all with u* = 0 on the boundary.

  poisson:        -lap u = f
  darcy_control:  -a lap u - grad a . grad u = f   (smooth manufactured a)

Every hand-coded derivative/forcing is FD-verified by verify_all().
"""
from __future__ import annotations

import numpy as np

PI = np.pi


def ustar(P):
    x, y = P[:, 0], P[:, 1]
    return np.sin(PI * x) * np.sin(PI * y) + 0.5 * np.sin(2 * PI * x) * np.sin(PI * y)


def ustar_x(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.cos(PI * x) * np.sin(PI * y) + PI * np.cos(2 * PI * x) * np.sin(PI * y)


def ustar_y(P):
    x, y = P[:, 0], P[:, 1]
    return PI * np.sin(PI * x) * np.cos(PI * y) + 0.5 * PI * np.sin(2 * PI * x) * np.cos(PI * y)


def ustar_lap(P):
    x, y = P[:, 0], P[:, 1]
    uxx = -PI**2 * np.sin(PI * x) * np.sin(PI * y) - 2 * PI**2 * np.sin(2 * PI * x) * np.sin(PI * y)
    uyy = -PI**2 * np.sin(PI * x) * np.sin(PI * y) - 0.5 * PI**2 * np.sin(2 * PI * x) * np.sin(PI * y)
    return uxx + uyy


def a_ctrl(P):
    x, y = P[:, 0], P[:, 1]
    return 3.0 + np.exp(np.sin(PI * x) * np.sin(PI * y))


def a_ctrl_x(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(np.sin(PI * x) * np.sin(PI * y)) * PI * np.cos(PI * x) * np.sin(PI * y)


def a_ctrl_y(P):
    x, y = P[:, 0], P[:, 1]
    return np.exp(np.sin(PI * x) * np.sin(PI * y)) * PI * np.sin(PI * x) * np.cos(PI * y)


def poisson():
    terms = [((2, 0), -1.0), ((0, 2), -1.0)]
    forcing = lambda P: -ustar_lap(P)
    return dict(name="poisson", terms=terms, forcing=forcing, exact=ustar)


def darcy_control():
    terms = [((2, 0), lambda P: -a_ctrl(P)),
             ((0, 2), lambda P: -a_ctrl(P)),
             ((1, 0), lambda P: -a_ctrl_x(P)),
             ((0, 1), lambda P: -a_ctrl_y(P))]
    forcing = lambda P: -(a_ctrl(P) * ustar_lap(P)
                          + a_ctrl_x(P) * ustar_x(P)
                          + a_ctrl_y(P) * ustar_y(P))
    return dict(name="darcy_control", terms=terms, forcing=forcing, exact=ustar)


PROBLEMS = [poisson, darcy_control]


def _fd_partial(f, P, ax, ay, h=1e-5):
    if ax == 0 and ay == 0:
        return f(P)
    if ax + ay == 1:
        col = 0 if ax else 1
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - f(Pm)) / (2 * h)
    if (ax, ay) in [(2, 0), (0, 2)]:
        col = 0 if ax else 1
        Pp, Pm = P.copy(), P.copy()
        Pp[:, col] += h
        Pm[:, col] -= h
        return (f(Pp) - 2 * f(P) + f(Pm)) / h**2
    raise ValueError((ax, ay))


def verify_all(tol=2e-4):
    rng = np.random.default_rng(7)
    P = rng.uniform(-0.9, 0.9, (60, 2))
    checks = [
        ("u_x", ustar_x(P), _fd_partial(ustar, P, 1, 0)),
        ("u_y", ustar_y(P), _fd_partial(ustar, P, 0, 1)),
        ("lap", ustar_lap(P),
         _fd_partial(ustar, P, 2, 0, h=1e-4) + _fd_partial(ustar, P, 0, 2, h=1e-4)),
        ("a_x", a_ctrl_x(P), _fd_partial(a_ctrl, P, 1, 0)),
        ("a_y", a_ctrl_y(P), _fd_partial(a_ctrl, P, 0, 1)),
    ]
    # boundary condition: u* = 0 on all four edges
    s = np.linspace(-1, 1, 41)
    for edge in [np.stack([s, np.full_like(s, -1.0)], 1), np.stack([s, np.full_like(s, 1.0)], 1),
                 np.stack([np.full_like(s, -1.0), s], 1), np.stack([np.full_like(s, 1.0), s], 1)]:
        assert np.max(np.abs(ustar(edge))) < 1e-12
    for name, ours, fd in checks:
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(ours - fd)) / scale
        assert err < tol, f"check '{name}' failed: rel err {err:.2e}"
    # forcing identity for both problems: L[u*] evaluated by FD == forcing
    for prob_fn in PROBLEMS:
        prob = prob_fn()
        lhs = np.zeros(len(P))
        for (ax, ay), coeff in prob["terms"]:
            c = coeff(P) if callable(coeff) else coeff
            lhs += c * _fd_partial(ustar, P, ax, ay, h=1e-4)
        err = np.max(np.abs(lhs - prob["forcing"](P))) / max(1.0, np.max(np.abs(lhs)))
        assert err < tol, f"forcing identity '{prob['name']}': rel err {err:.2e}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py::test_problems_fd_verified -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/expF05_spline_ridge/problems.py tests/test_expF05_spline_ridge.py
git commit -m "expF05: FD-verified poisson + smooth darcy control problems"
```

---

### Task 3: expF05 darcy_data.py (rough-Darcy loader + surrogate)

**Files:**
- Create: `experiments/expF05_spline_ridge/darcy_data.py`
- Test: append to `tests/test_expF05_spline_ridge.py`

- [ ] **Step 1: Write the failing test**

```python
DARCY_NPZ = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_test_421_jax.npz"


@pytest.mark.skipif(not Path(DARCY_NPZ).exists(), reason="darcy npz not present")
def test_darcy_loader_and_surrogate():
    import darcy_data as dd
    a_all, u_all = dd.load_darcy_test(DARCY_NPZ, n_instances=1)
    assert a_all.shape == (1, 421, 421) and u_all.shape == (1, 421, 421)
    coef = dd.DarcyCoefficient(a_all[0], sigma_px=0.0, cell_centered=True)
    # surrogate reproduces grid values at grid nodes (interpolation property)
    g = dd.grid_1d(421, cell_centered=True)
    ii = np.array([10, 100, 210, 400])
    Pg = np.stack(np.meshgrid(g[ii], g[ii], indexing="ij"), -1).reshape(-1, 2)
    vals = coef.a(Pg).reshape(4, 4)
    assert np.allclose(vals, a_all[0][np.ix_(ii, ii)], atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py::test_darcy_loader_and_surrogate -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'darcy_data'` (or SKIP if the npz is missing — if SKIP, verify the path by `ls` and fix before proceeding)

- [ ] **Step 3: Write darcy_data.py** (vendored from continuous-mlps `precision_pde/darcy_problems.py`)

```python
# experiments/expF05_spline_ridge/darcy_data.py
"""darcy_421 benchmark loading + spline coefficient surrogate.

Benchmark convention: -div_s(a grad_s u) = 1 on [0,1]^2, u = 0 on the boundary.
On p = 2s - 1: -a lap_p u - grad_p a . grad_p u = 1/4, so DARCY_FORCING = 0.25.
darcy_421 stores CELL-CENTERED grids (node k at (k+1/2)h).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

DARCY_FORCING = 0.25
DEFAULT_NPZ = "/scr/cdeng/continuous-mlps/data/fno_datasets_jax/darcy_test_421_jax.npz"


def grid_1d(n, cell_centered=False):
    if cell_centered:
        return -1.0 + (2.0 * np.arange(n) + 1.0) / n
    return np.linspace(-1.0, 1.0, n)


class DarcyCoefficient:
    """Cubic-spline surrogate of a gridded coefficient on [-1,1]^2."""

    def __init__(self, a_grid, sigma_px=0.0, cell_centered=False):
        a = np.asarray(a_grid, dtype=np.float64)
        if sigma_px > 0:
            a = gaussian_filter(a, sigma_px, mode="nearest")
        n0, n1 = a.shape
        self._sp = RectBivariateSpline(grid_1d(n0, cell_centered),
                                       grid_1d(n1, cell_centered), a, kx=3, ky=3)

    def a(self, P):
        return self._sp.ev(P[:, 0], P[:, 1])

    def ax(self, P):
        return self._sp.ev(P[:, 0], P[:, 1], dx=1)

    def ay(self, P):
        return self._sp.ev(P[:, 0], P[:, 1], dy=1)

    def terms(self):
        return [((2, 0), lambda P: -self.a(P)),
                ((0, 2), lambda P: -self.a(P)),
                ((1, 0), lambda P: -self.ax(P)),
                ((0, 1), lambda P: -self.ay(P))]


def load_darcy_test(path=DEFAULT_NPZ, n_instances=16):
    d = np.load(path)
    keys = set(d.keys())
    if {"x", "y"} <= keys:
        a, u = d["x"], d["y"]
    elif {"a", "u"} <= keys:
        a, u = d["a"], d["u"]
    else:
        raise KeyError(f"unrecognized darcy npz keys: {sorted(keys)}")
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    a = a.reshape(a.shape[0], a.shape[-2], a.shape[-1])
    u = u.reshape(u.shape[0], u.shape[-2], u.shape[-1])
    return a[:n_instances], u[:n_instances]


def eval_points_and_ref(u_grid, stride=3):
    """Cell-centered eval points and reference values, subsampled by stride."""
    n = u_grid.shape[0]
    g = grid_1d(n, cell_centered=True)[::stride]
    P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return P, u_grid[::stride, ::stride].ravel()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py::test_darcy_loader_and_surrogate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/expF05_spline_ridge/darcy_data.py tests/test_expF05_spline_ridge.py
git commit -m "expF05: vendored darcy_421 loader + spline coefficient surrogate"
```

---

### Task 4: expF05 adaptive.py (knot insertion + local gammas)

**Files:**
- Create: `experiments/expF05_spline_ridge/adaptive.py`
- Test: append to `tests/test_expF05_spline_ridge.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_local_gammas_matches_uniform_grid():
    import adaptive
    dirs, offs, gammas = rc.radon_geometry(256, lam=0.25)
    got = adaptive.local_gammas(dirs, offs, lam=0.25)
    # on the uniform init grid, per-neuron gammas ~ the global expF01 gamma
    assert np.all(np.abs(got / gammas - 1.0) < 0.15)


def test_insert_knots_targets_residual_mass():
    import adaptive
    dirs, offs, _ = rc.radon_geometry(256, lam=0.25)
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, (4000, 2))
    # all residual mass concentrated near x ~ 0.7
    r = np.exp(-((P[:, 0] - 0.7) ** 2) / 0.005)
    nd, no = adaptive.insert_knots(dirs, offs, P, r, n_new=64)
    assert len(no) == 64 and nd.shape == (64, 2)
    # knots for the near-x-axis direction should cluster near s ~ 0.7
    ax_dir = np.abs(nd[:, 1]) < 0.35  # directions mostly along x
    assert np.median(np.abs(no[ax_dir] - 0.7)) < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py -k adaptive_or_knots -v` — actually run by name:
`uv run --extra dev pytest tests/test_expF05_spline_ridge.py::test_local_gammas_matches_uniform_grid tests/test_expF05_spline_ridge.py::test_insert_knots_targets_residual_mass -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive'`

- [ ] **Step 3: Write adaptive.py**

```python
# experiments/expF05_spline_ridge/adaptive.py
"""Residual-guided knot insertion for the spline ridge basis.

insert_knots: project |residual| mass onto each ridge direction (Radon
binning), allocate new offsets across directions proportionally to mass,
place them at the highest-mass bin centers.
local_gammas: per-neuron gamma from local offset spacing within a direction,
gamma = lam / (0.875 * local_gap) — matches the global expF01 gamma
(lam / (2.8/sqrt(W))) on the uniform init grid.
"""
from __future__ import annotations

import numpy as np

from ridge_core import COLLAR_SQUARE

GAP_TO_HREF = 0.875  # (2.8/sqrt(W)) / (3.2/sqrt(W)) on the uniform grid


def _theta_groups(dirs):
    thetas = np.round(np.arctan2(dirs[:, 1], dirs[:, 0]), 9)
    return thetas, np.unique(thetas)


def local_gammas(dirs, offs, lam):
    thetas, uniq = _theta_groups(dirs)
    gammas = np.empty(len(offs))
    for th in uniq:
        idx = np.where(thetas == th)[0]
        order = idx[np.argsort(offs[idx])]
        t = offs[order]
        gaps = np.diff(t)
        g = np.empty(len(t))
        if len(t) == 1:
            g[:] = 2 * COLLAR_SQUARE
        else:
            g[0], g[-1] = gaps[0], gaps[-1]
            g[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
        gammas[order] = lam / (GAP_TO_HREF * np.maximum(g, 1e-6))
    return gammas


def insert_knots(dirs, offs, P_res, r_abs, n_new, collar=COLLAR_SQUARE, n_bins=48):
    """Return (new_dirs [k,2], new_offs [k]) with k == n_new."""
    thetas, uniq = _theta_groups(dirs)
    infos, masses = [], []
    for th in uniq:
        w = np.array([np.cos(th), np.sin(th)])
        s = P_res @ w
        hist, edges = np.histogram(s, bins=n_bins, range=(-collar, collar), weights=r_abs)
        infos.append((w, hist, edges))
        masses.append(hist.sum())
    masses = np.asarray(masses)
    alloc = np.floor(n_new * masses / masses.sum()).astype(int)
    short = n_new - alloc.sum()
    if short > 0:
        alloc[np.argsort(-masses)[:short]] += 1
    new_dirs, new_offs = [], []
    for (w, hist, edges), k in zip(infos, alloc):
        if k == 0:
            continue
        k = int(min(k, len(hist)))
        top = np.argsort(-hist)[:k]
        centers = 0.5 * (edges[top] + edges[top + 1])
        for c in centers:
            new_dirs.append(w)
            new_offs.append(c)
    return np.asarray(new_dirs), np.asarray(new_offs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_expF05_spline_ridge.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/expF05_spline_ridge/adaptive.py tests/test_expF05_spline_ridge.py
git commit -m "expF05: residual-guided knot insertion + local per-neuron gammas"
```

---

### Task 5: expF05 run.py (Part A + Part B drivers)

**Files:**
- Create: `experiments/expF05_spline_ridge/run.py`

- [ ] **Step 1: Write run.py**

```python
# experiments/expF05_spline_ridge/run.py
"""Experiment expF05 -- KAN-style B-spline ridge basis.

Part A (default): tanh vs cubic-B-spline floor on poisson + smooth darcy
control, W in {144,256,576,1024,2304}, lam in {0.2,0.25,0.3}, best-of-lam.
Part B (--adaptive): rough darcy_421 instance 0, sigma=0. Baselines: dense
tanh and dense spline at W=2304 (uniform). Adaptive: spline, start W=1024,
4 rounds x 320 residual-guided knots -> 2304 total (width-matched).

Outputs (results/checkpoint_F_applications/expF05_spline_ridge/):
  error_vs_width.png    rel L2 vs W, 2 problems x 2 families
  adaptive_rounds.png   rel L2 + n_knots per adaptive round vs dense baselines
  data.json             all cells, written incrementally

Usage:
  uv run --extra dev python experiments/expF05_spline_ridge/run.py [--smoke] [--plot] [--adaptive] [--darcy-path PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ridge_core as rc
import problems as pb
import adaptive as ad
import darcy_data as dd

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF05_spline_ridge"
DATA_PATH = RESULTS_DIR / "data.json"

FAMILIES = {"tanh": rc.tanh_family, "bspline": rc.bspline_family}
LAMS = [0.2, 0.25, 0.3]


def load_data():
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return []


def save_data(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=1))


def eval_grid(n=120):
    g = np.linspace(-0.995, 0.995, n)
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def bc_zero(n_per_edge=480):
    Pb = rc.boundary_points_square(n_per_edge)
    return [dict(points=Pb, terms=[((0, 0), 1.0)], values=np.zeros(len(Pb)))]


def part_a(smoke):
    w_grid = [144, 400] if smoke else [144, 256, 576, 1024, 2304]
    data = load_data()
    done = {(c["part"], c.get("problem"), c.get("family"), c.get("W"), c.get("lam"))
            for c in data}
    Pe = eval_grid()
    for prob_fn in pb.PROBLEMS:
        prob = prob_fn()
        u_true = prob["exact"](Pe)
        for fam_name, family in FAMILIES.items():
            for W in w_grid:
                for lam in LAMS:
                    key = ("A", prob["name"], fam_name, W, lam)
                    if key in done:
                        continue
                    t0 = time.time()
                    model = rc.solve_square(prob["terms"], prob["forcing"], bc_zero(),
                                            W=W, lam=lam, family=family)
                    err = rc.rel_l2(rc.eval_model(model, Pe), u_true)
                    cell = dict(part="A", problem=prob["name"], family=fam_name,
                                W=W, lam=lam, rel_l2=err, t_solve=time.time() - t0)
                    print(cell, flush=True)
                    data.append(cell)
                    save_data(data)


def part_b(smoke, darcy_path):
    a_all, u_all = dd.load_darcy_test(darcy_path, n_instances=1)
    coef = dd.DarcyCoefficient(a_all[0], sigma_px=0.0, cell_centered=True)
    P_eval, u_ref = dd.eval_points_and_ref(u_all[0], stride=3)
    lam = 0.25
    data = load_data()
    done = {(c["part"], c.get("method"), c.get("round")) for c in data}

    def record(method, rnd, model, n_knots, t0):
        err = rc.rel_l2(rc.eval_model(model, P_eval), u_ref)
        cell = dict(part="B", method=method, round=rnd, n_knots=int(n_knots),
                    rel_l2=err, t_solve=time.time() - t0)
        print(cell, flush=True)
        data.append(cell)
        save_data(data)

    w_dense = 400 if smoke else 2304
    for fam_name, family in FAMILIES.items():
        if ("B", f"dense_{fam_name}", 0) not in done:
            t0 = time.time()
            model = rc.solve_square(coef.terms(), dd.DARCY_FORCING, bc_zero(),
                                    W=w_dense, lam=lam, family=family)
            record(f"dense_{fam_name}", 0, model, w_dense, t0)

    # adaptive spline: start smaller, insert knots up to the dense budget
    w0 = 144 if smoke else 1024
    n_rounds, n_add = (2, 128) if smoke else (4, 320)
    dirs, offs, _ = rc.radon_geometry(w0, lam)
    rng = np.random.default_rng(1)
    P_res = rng.uniform(-1, 1, (20000, 2))
    for rnd in range(n_rounds + 1):
        if ("B", "adaptive_bspline", rnd) in done:
            continue
        gammas = ad.local_gammas(dirs, offs, lam)
        t0 = time.time()
        model = rc.solve_square(coef.terms(), dd.DARCY_FORCING, bc_zero(),
                                family=rc.bspline_family,
                                geometry=(dirs, offs, gammas))
        record("adaptive_bspline", rnd, model, len(offs), t0)
        if rnd == n_rounds:
            break
        resid = np.abs(rc.eval_model(model, P_res, terms=coef.terms())
                       - dd.DARCY_FORCING)
        nd, no = ad.insert_knots(dirs, offs, P_res, resid, n_new=n_add)
        dirs = np.vstack([dirs, nd])
        offs = np.concatenate([offs, no])


def plot():
    data = load_data()
    a = [c for c in data if c["part"] == "A"]
    if a:
        probs = sorted({c["problem"] for c in a})
        fig, axes = plt.subplots(1, len(probs), figsize=(6 * len(probs), 4.5))
        axes = np.atleast_1d(axes)
        for axi, prob in zip(axes, probs):
            for fam in sorted({c["family"] for c in a}):
                cells = [c for c in a if c["problem"] == prob and c["family"] == fam]
                ws = sorted({c["W"] for c in cells})
                best = [min(c["rel_l2"] for c in cells if c["W"] == w) for w in ws]
                axi.loglog(ws, best, "o-", label=fam)
            axi.set_title(prob)
            axi.set_xlabel("W")
            axi.set_ylabel("rel L2")
            axi.grid(True, which="both", alpha=0.3)
            axi.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "error_vs_width.png", dpi=140)
    b = [c for c in data if c["part"] == "B"]
    if b:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ada = sorted([c for c in b if c["method"] == "adaptive_bspline"],
                     key=lambda c: c["round"])
        if ada:
            ax.semilogy([c["n_knots"] for c in ada], [c["rel_l2"] for c in ada],
                        "o-", label="adaptive bspline")
        for c in b:
            if c["method"].startswith("dense"):
                ax.axhline(c["rel_l2"], ls="--", alpha=0.7,
                           label=f"{c['method']} W={c['n_knots']}")
        ax.set_xlabel("total knots")
        ax.set_ylabel("rel L2 vs reference")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "adaptive_rounds.png", dpi=140)
    print("plots saved to", RESULTS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--darcy-path", default=dd.DEFAULT_NPZ)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot()
        return
    if args.adaptive:
        part_b(args.smoke, args.darcy_path)
    else:
        part_a(args.smoke)
    plot()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run Part A**

Run: `uv run --extra dev python experiments/expF05_spline_ridge/run.py --smoke`
Expected: 2 problems × 2 families × 2 W × 3 λ = 24 cells printed with finite `rel_l2`; tanh at W=400 should reach ≤1e-6 on poisson; `error_vs_width.png` and `data.json` created. Sanity-read: spline within ~1-2 orders of tanh at matched W. If spline is >2 orders worse, note it and continue (full-run data decides the quintic escalation).

- [ ] **Step 3: Smoke-run Part B**

Run: `uv run --extra dev python experiments/expF05_spline_ridge/run.py --smoke --adaptive`
Expected: dense_tanh + dense_bspline cells at W=400, then adaptive rounds 0..2 (144 → 272 → 400 knots) each with finite rel_l2; `adaptive_rounds.png` created. Smoke numbers will be poor (~1e-1) — only checking the machinery runs.

- [ ] **Step 4: Delete smoke data and commit**

```bash
rm results/checkpoint_F_applications/expF05_spline_ridge/data.json
git add experiments/expF05_spline_ridge/run.py
git commit -m "expF05: Part A/B driver with smoke mode and incremental data.json"
```

---

### Task 6: expF05 full runs

- [ ] **Step 1: Run Part A in the background**

Run: `cd /scr/cdeng/precision-mlps && nohup uv run --extra dev python experiments/expF05_spline_ridge/run.py > /tmp/expF05_partA.log 2>&1 &`
Expected: ~60 cells; the W=2304 solves take ~1-2 min each; total ~1-2 h. Poll with `tail /tmp/expF05_partA.log`.

- [ ] **Step 2: When Part A finishes, run Part B**

Run: `uv run --extra dev python experiments/expF05_spline_ridge/run.py --adaptive > /tmp/expF05_partB.log 2>&1`
Expected: dense baselines (dense_tanh should land near the known 7.2e-2), then 5 adaptive cells (rounds 0-4, 1024→2304 knots).

- [ ] **Step 3: Sanity-read results against the spec**

- Part A pass: bspline best-of-λ within ~1 order of tanh at W=2304 on both problems (tanh reference: ~1e-13 poisson, ~3e-14-1e-12 darcy_control).
- Part B: compare adaptive round-4 rel_l2 vs dense_bspline and dense_tanh. Spec success = ≥1 order below 7.2e-2; stretch ≤2.8e-3. Record whatever happened — a negative is a result.

- [ ] **Step 4: Commit results**

```bash
git add results/checkpoint_F_applications/expF05_spline_ridge/
git commit -m "expF05: full Part A floor sweep + Part B adaptive-knot rough darcy results"
```

---

### Task 7: expF06 problems.py (manufactured Burgers)

**Files:**
- Create: `experiments/expF06_newton_burgers/__init__.py` (empty), `experiments/expF06_newton_burgers/problems.py`
- Test: `tests/test_expF06_newton_burgers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expF06_newton_burgers.py
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF06_newton_burgers"))


def test_burgers_manufactured_fd_verified():
    import problems as bp
    bp.verify_all(nu=0.1)
    bp.verify_all(nu=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_expF06_newton_burgers.py -v`
Expected: FAIL — `problems` resolves to expF05's module which has no `verify_all(nu=...)` signature, or import error. NOTE: because both experiments name a module `problems.py`, tests and run scripts must insert the expF06 dir FIRST (last-inserted wins with sys.path.insert(0, ...)). The ordering above (expF05 inserted before expF06) achieves that. In expF06/expF07 code, import expF05 modules by their unique names (`ridge_core`, `adaptive`, `darcy_data`) — those don't collide.

- [ ] **Step 3: Write problems.py**

```python
# experiments/expF06_newton_burgers/problems.py
"""Steady viscous Burgers on [-1,1]^2, manufactured Taylor-Green solution.

  F_u(u,v) = u u_x + v u_y - nu lap u - f_u = 0
  F_v(u,v) = u v_x + v v_y - nu lap v - f_v = 0
  u* = -cos(pi x) sin(pi y),  v* = sin(pi x) cos(pi y)   (period 2)
Dirichlet BCs from the exact solution on all four edges.
"""
from __future__ import annotations

import numpy as np

PI = np.pi


def u_exact(P):
    return -np.cos(PI * P[:, 0]) * np.sin(PI * P[:, 1])


def v_exact(P):
    return np.sin(PI * P[:, 0]) * np.cos(PI * P[:, 1])


def u_x(P):
    return PI * np.sin(PI * P[:, 0]) * np.sin(PI * P[:, 1])


def u_y(P):
    return -PI * np.cos(PI * P[:, 0]) * np.cos(PI * P[:, 1])


def v_x(P):
    return PI * np.cos(PI * P[:, 0]) * np.cos(PI * P[:, 1])


def v_y(P):
    return -PI * np.sin(PI * P[:, 0]) * np.sin(PI * P[:, 1])


def lap_u(P):
    return -2 * PI**2 * u_exact(P)


def lap_v(P):
    return -2 * PI**2 * v_exact(P)


def f_u(P, nu):
    return u_exact(P) * u_x(P) + v_exact(P) * u_y(P) - nu * lap_u(P)


def f_v(P, nu):
    return u_exact(P) * v_x(P) + v_exact(P) * v_y(P) - nu * lap_v(P)


def _fd(f, P, col, h=1e-5):
    Pp, Pm = P.copy(), P.copy()
    Pp[:, col] += h
    Pm[:, col] -= h
    return (f(Pp) - f(Pm)) / (2 * h)


def _fd2(f, P, col, h=1e-4):
    Pp, Pm = P.copy(), P.copy()
    Pp[:, col] += h
    Pm[:, col] -= h
    return (f(Pp) - 2 * f(P) + f(Pm)) / h**2


def verify_all(nu, tol=2e-4):
    rng = np.random.default_rng(7)
    P = rng.uniform(-0.9, 0.9, (60, 2))
    checks = [
        ("u_x", u_x(P), _fd(u_exact, P, 0)),
        ("u_y", u_y(P), _fd(u_exact, P, 1)),
        ("v_x", v_x(P), _fd(v_exact, P, 0)),
        ("v_y", v_y(P), _fd(v_exact, P, 1)),
        ("lap_u", lap_u(P), _fd2(u_exact, P, 0) + _fd2(u_exact, P, 1)),
        ("lap_v", lap_v(P), _fd2(v_exact, P, 0) + _fd2(v_exact, P, 1)),
        ("f_u", f_u(P, nu),
         u_exact(P) * _fd(u_exact, P, 0) + v_exact(P) * _fd(u_exact, P, 1)
         - nu * (_fd2(u_exact, P, 0) + _fd2(u_exact, P, 1))),
        ("f_v", f_v(P, nu),
         u_exact(P) * _fd(v_exact, P, 0) + v_exact(P) * _fd(v_exact, P, 1)
         - nu * (_fd2(v_exact, P, 0) + _fd2(v_exact, P, 1))),
    ]
    for name, ours, fd in checks:
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(ours - fd)) / scale
        assert err < tol, f"burgers check '{name}' failed: rel err {err:.2e}"
```

Also create the empty `experiments/expF06_newton_burgers/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_expF06_newton_burgers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/expF06_newton_burgers/ tests/test_expF06_newton_burgers.py
git commit -m "expF06: FD-verified manufactured Taylor-Green Burgers problem"
```

---

### Task 8: expF06 newton.py (damped Newton block-lstsq)

**Files:**
- Create: `experiments/expF06_newton_burgers/newton.py`
- Test: append to `tests/test_expF06_newton_burgers.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.slow
def test_newton_converges_nu01():
    import problems as bp
    import newton as nt
    res = nt.newton_burgers(nu=0.1, W=256, lam=0.25, max_iter=8, seed=0,
                            u_exact=bp.u_exact, v_exact=bp.v_exact)
    hist = res["history"]
    # damping guarantees monotone residual over accepted steps
    resids = [h["res_norm"] for h in hist]
    assert all(b <= a * 1.0001 for a, b in zip(resids, resids[1:]))
    assert hist[-1]["rel_l2_u"] < 1e-6, hist[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_expF06_newton_burgers.py::test_newton_converges_nu01 -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'newton'`

- [ ] **Step 3: Write newton.py**

```python
# experiments/expF06_newton_burgers/newton.py
"""Damped Newton for steady Burgers; every step is one block collocation lstsq
in a FROZEN ridge basis, so iterates add in coefficient space.

newton_burgers(..., base_fields=None): base_fields(P) -> dict with keys
u, ux, uy, lap_u, v, vx, vy, lap_v (numpy [n]) is an optional frozen warm
start (the trained PINN in expF07); the ridge expansion carries corrections.
"""
from __future__ import annotations

import time

import numpy as np

import ridge_core as rc
import problems as bp

FIELD_TERMS = {
    "": (((0, 0), 1.0),),
    "x": (((1, 0), 1.0),),
    "y": (((0, 1), 1.0),),
    "lap": (((2, 0), 1.0), ((0, 2), 1.0)),
}


def _ridge_fields(geom, family, sol_u, sol_v, P):
    dirs, offs, gammas = geom
    out = {}
    for name, sol in (("u", sol_u), ("v", sol_v)):
        m = dict(dirs=dirs, offs=offs, gammas=gammas, sol=sol, family=family)
        out[name] = rc.eval_model(m, P)
        out[name + "x"] = rc.eval_model(m, P, terms=FIELD_TERMS["x"])
        out[name + "y"] = rc.eval_model(m, P, terms=FIELD_TERMS["y"])
        out["lap_" + name] = rc.eval_model(m, P, terms=FIELD_TERMS["lap"])
    return out


def _total_fields(geom, family, sol_u, sol_v, P, base_fields):
    f = _ridge_fields(geom, family, sol_u, sol_v, P)
    if base_fields is not None:
        b = base_fields(P)
        for k in f:
            f[k] = f[k] + b[k]
    return f


def _residuals(f, P, nu):
    F_u = f["u"] * f["ux"] + f["v"] * f["uy"] - nu * f["lap_u"] - bp.f_u(P, nu)
    F_v = f["u"] * f["vx"] + f["v"] * f["vy"] - nu * f["lap_v"] - bp.f_v(P, nu)
    return F_u, F_v


def newton_burgers(nu, W, lam, family=rc.tanh_family, max_iter=12, seed=42,
                   base_fields=None, u_exact=bp.u_exact, v_exact=bp.v_exact,
                   n_eval=120):
    rng = np.random.default_rng(seed)
    geom = rc.radon_geometry(W, lam)
    n_feat = len(geom[1]) + len(rc.MONO_2D)
    P = rc.interior_points_square(len(geom[1]), rng)
    Pb = rc.boundary_points_square()
    g = np.linspace(-0.995, 0.995, n_eval)
    Pe = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    ue, ve = u_exact(Pe), v_exact(Pe)

    sol_u = np.zeros(n_feat)
    sol_v = np.zeros(n_feat)
    history = []
    t0 = time.time()
    for it in range(max_iter + 1):
        f = _total_fields(geom, family, sol_u, sol_v, P, base_fields)
        F_u, F_v = _residuals(f, P, nu)
        res_norm = float(np.sqrt(np.mean(F_u**2 + F_v**2)))
        fe = _total_fields(geom, family, sol_u, sol_v, Pe, base_fields)
        history.append(dict(iter=it, res_norm=res_norm,
                            rel_l2_u=rc.rel_l2(fe["u"], ue),
                            rel_l2_v=rc.rel_l2(fe["v"], ve),
                            t=time.time() - t0))
        print(history[-1], flush=True)
        if it == max_iter or res_norm < 1e-13:
            break
        # block Jacobian rows: J_uu du + J_uv dv = -F_u ; J_vu du + J_vv dv = -F_v
        A_uu = rc.rows_2d(P, *geom, terms=[((1, 0), f["u"]), ((0, 1), f["v"]),
                                           ((0, 0), f["ux"]),
                                           ((2, 0), -nu), ((0, 2), -nu)],
                          family=family)
        A_uv = rc.rows_2d(P, *geom, terms=[((0, 0), f["uy"])], family=family)
        A_vu = rc.rows_2d(P, *geom, terms=[((0, 0), f["vx"])], family=family)
        A_vv = rc.rows_2d(P, *geom, terms=[((1, 0), f["u"]), ((0, 1), f["v"]),
                                           ((0, 0), f["vy"]),
                                           ((2, 0), -nu), ((0, 2), -nu)],
                          family=family)
        Z = np.zeros((len(P), n_feat))
        A_pde = np.block([[A_uu, A_uv], [A_vu, A_vv]])
        y_pde = np.concatenate([-F_u, -F_v])
        s = np.abs(A_pde).max()
        # Dirichlet BC on the correction: delta = exact - current on the boundary
        fb = _total_fields(geom, family, sol_u, sol_v, Pb, base_fields)
        Rb = rc.rows_2d(Pb, *geom, terms=[((0, 0), 1.0)], family=family)
        Zb = np.zeros_like(Rb)
        wb = np.sqrt(2 * len(P) / (2 * len(Pb)))
        A_bc = np.block([[Rb, Zb], [Zb, Rb]])
        y_bc = np.concatenate([u_exact(Pb) - fb["u"], v_exact(Pb) - fb["v"]])
        A = np.vstack([A_pde / s, wb * A_bc])
        y = np.concatenate([y_pde / s, wb * y_bc])
        dsol = np.linalg.lstsq(A, y, rcond=rc.RCOND)[0]
        du, dv = dsol[:n_feat], dsol[n_feat:]
        # backtracking line search on the collocation residual norm
        alpha = 1.0
        accepted = False
        while alpha > 1.0 / 256:
            tu, tv = sol_u + alpha * du, sol_v + alpha * dv
            ft = _total_fields(geom, family, tu, tv, P, base_fields)
            Fu_t, Fv_t = _residuals(ft, P, nu)
            if np.sqrt(np.mean(Fu_t**2 + Fv_t**2)) < res_norm:
                sol_u, sol_v = tu, tv
                accepted = True
                break
            alpha /= 2
        if not accepted:
            history[-1]["stalled"] = True
            break
    return dict(geom=geom, family=family, sol_u=sol_u, sol_v=sol_v,
                history=history)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_expF06_newton_burgers.py -v`
Expected: both tests PASS (the Newton test takes ~30-90 s; it is marked slow). Watch the printed history: residual should drop fast (near-quadratically) after the first 1-2 iterations.

- [ ] **Step 5: Commit**

```bash
git add experiments/expF06_newton_burgers/newton.py tests/test_expF06_newton_burgers.py
git commit -m "expF06: damped Newton block-lstsq solver with optional frozen base fields"
```

---

### Task 9: expF06 run.py + full run

**Files:**
- Create: `experiments/expF06_newton_burgers/run.py`

- [ ] **Step 1: Write run.py**

```python
# experiments/expF06_newton_burgers/run.py
"""Experiment expF06 -- steady 2D Burgers via Newton-lstsq (solve, don't train).

Grid: nu in {0.1, 0.01} x W in {256, 576, 1024, 2304} (smoke: nu=0.1, W=256),
max 12 Newton iterations, tanh family, lam=0.25.

Outputs (results/checkpoint_F_applications/expF06_newton_burgers/):
  newton_convergence.png   res_norm + rel_l2(u) vs iteration, best W per nu
  error_vs_width.png       final rel_l2(u) vs W per nu
  data.json                every (nu, W) cell with its full Newton history

Usage:
  uv run --extra dev python experiments/expF06_newton_burgers/run.py [--smoke] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import newton as nt

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF06_newton_burgers"
DATA_PATH = RESULTS_DIR / "data.json"
LAM = 0.25


def load_data():
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text())
    return []


def save_data(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(data, indent=1))


def sweep(smoke):
    nus = [0.1] if smoke else [0.1, 0.01]
    ws = [256] if smoke else [256, 576, 1024, 2304]
    data = load_data()
    done = {(c["nu"], c["W"]) for c in data}
    for nu in nus:
        for W in ws:
            if (nu, W) in done:
                continue
            print(f"=== nu={nu} W={W} ===", flush=True)
            res = nt.newton_burgers(nu=nu, W=W, lam=LAM,
                                    max_iter=6 if smoke else 12)
            data.append(dict(nu=nu, W=W, lam=LAM, history=res["history"]))
            save_data(data)


def plot():
    data = load_data()
    if not data:
        return
    nus = sorted({c["nu"] for c in data}, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for nu in nus:
        cells = [c for c in data if c["nu"] == nu]
        best = max(cells, key=lambda c: c["W"])
        it = [h["iter"] for h in best["history"]]
        axes[0].semilogy(it, [h["res_norm"] for h in best["history"]], "o-",
                         label=f"nu={nu} W={best['W']} residual")
        axes[0].semilogy(it, [h["rel_l2_u"] for h in best["history"]], "s--",
                         label=f"nu={nu} rel L2(u)")
        ws = sorted({c["W"] for c in cells})
        finals = [min(c["history"][-1]["rel_l2_u"] for c in cells if c["W"] == w)
                  for w in ws]
        axes[1].loglog(ws, finals, "o-", label=f"nu={nu}")
    axes[0].set_xlabel("Newton iteration")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("final rel L2(u)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "newton_convergence.png", dpi=140)
    print("plots saved to", RESULTS_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot:
        sweep(args.smoke)
    plot()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run**

Run: `uv run --extra dev python experiments/expF06_newton_burgers/run.py --smoke`
Expected: one (0.1, 256) cell, Newton history printed per iteration, final rel_l2_u ≤ 1e-6, plot + data.json created.

- [ ] **Step 3: Delete smoke data, commit driver**

```bash
rm results/checkpoint_F_applications/expF06_newton_burgers/data.json
git add experiments/expF06_newton_burgers/run.py
git commit -m "expF06: nu x W sweep driver"
```

- [ ] **Step 4: Full run**

Run: `nohup uv run --extra dev python experiments/expF06_newton_burgers/run.py > /tmp/expF06.log 2>&1 &`
Expected: ~1-3 h (W=2304 iterations are ~2-4 min each). Spec pass: ν=0.1 final rel_l2 ≤ 1e-10 at the larger widths. ν=0.01 recorded wherever it lands; if it stalls (damping exhausted), that is recorded in the history — the ν-continuation escalation is a separate follow-up, do not improvise it inline.

- [ ] **Step 5: Commit results**

```bash
git add results/checkpoint_F_applications/expF06_newton_burgers/
git commit -m "expF06: full Newton-lstsq results (nu 0.1/0.01, W up to 2304)"
```

---

### Task 10: expF07 pinn.py (torch PINN + fields adapter)

**Files:**
- Create: `experiments/expF07_pinn_finisher/__init__.py` (empty), `experiments/expF07_pinn_finisher/pinn.py`
- Test: `tests/test_expF07_pinn_finisher.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_expF07_pinn_finisher.py
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF06_newton_burgers"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF07_pinn_finisher"))


def test_pinn_trains_and_loss_decreases():
    import pinn
    net, hist = pinn.train_pinn(nu=0.1, steps=60, batch=256, eval_every=30, seed=0)
    assert np.isfinite(hist[-1]["loss"])
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_pinn_fields_match_autograd_fd():
    import pinn
    net, _ = pinn.train_pinn(nu=0.1, steps=5, batch=64, eval_every=5, seed=0)
    fields = pinn.pinn_fields(net)
    rng = np.random.default_rng(0)
    P = rng.uniform(-0.9, 0.9, (40, 2))
    f = fields(P)
    h = 1e-4
    Px_p, Px_m = P.copy(), P.copy()
    Px_p[:, 0] += h
    Px_m[:, 0] -= h
    fd_ux = (fields(Px_p)["u"] - fields(Px_m)["u"]) / (2 * h)
    assert np.max(np.abs(f["ux"] - fd_ux)) < 1e-3
    fd_lap_part = (fields(Px_p)["ux"] - fields(Px_m)["ux"]) / (2 * h)
    Py_p, Py_m = P.copy(), P.copy()
    Py_p[:, 1] += h
    Py_m[:, 1] -= h
    fd_lap = fd_lap_part + (fields(Py_p)["uy"] - fields(Py_m)["uy"]) / (2 * h)
    assert np.max(np.abs(f["lap_u"] - fd_lap)) < 1e-2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_expF07_pinn_finisher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pinn'`

- [ ] **Step 3: Write pinn.py**

```python
# experiments/expF07_pinn_finisher/pinn.py
"""Vanilla torch tanh-MLP PINN for the expF06 Burgers problem, plus a numpy
fields adapter so the frozen net can serve as base_fields in expF06's Newton
loop. Everything float64 on CPU."""
from __future__ import annotations

import numpy as np
import torch

import problems as bp  # expF06's problems (sys.path order set by caller)

torch.set_default_dtype(torch.float64)


class PINN(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 2)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


def _derivs(net, X):
    """u, v and first/second derivatives at X [n,2] (requires_grad)."""
    out = net(X)
    u, v = out[:, 0], out[:, 1]
    d = {}
    for name, w in (("u", u), ("v", v)):
        g = torch.autograd.grad(w.sum(), X, create_graph=True)[0]
        wx, wy = g[:, 0], g[:, 1]
        gxx = torch.autograd.grad(wx.sum(), X, create_graph=True)[0][:, 0]
        gyy = torch.autograd.grad(wy.sum(), X, create_graph=True)[0][:, 1]
        d[name], d[name + "x"], d[name + "y"] = w, wx, wy
        d["lap_" + name] = gxx + gyy
    return d


def pde_loss(net, X, nu):
    d = _derivs(net, X)
    P = X.detach().numpy()
    fu = torch.from_numpy(bp.f_u(P, nu))
    fv = torch.from_numpy(bp.f_v(P, nu))
    Fu = d["u"] * d["ux"] + d["v"] * d["uy"] - nu * d["lap_u"] - fu
    Fv = d["u"] * d["vx"] + d["v"] * d["vy"] - nu * d["lap_v"] - fv
    return (Fu**2).mean() + (Fv**2).mean()


def bc_loss(net, Xb):
    out = net(Xb)
    P = Xb.detach().numpy()
    gu = torch.from_numpy(bp.u_exact(P))
    gv = torch.from_numpy(bp.v_exact(P))
    return ((out[:, 0] - gu)**2).mean() + ((out[:, 1] - gv)**2).mean()


def _eval_rel_l2(net, n=100):
    g = np.linspace(-0.995, 0.995, n)
    P = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    with torch.no_grad():
        out = net(torch.from_numpy(P)).numpy()
    ue = bp.u_exact(P)
    return float(np.linalg.norm(out[:, 0] - ue) / np.linalg.norm(ue))


def train_pinn(nu, steps=50000, batch=1024, n_bc=256, lr=1e-3, bc_weight=10.0,
               eval_every=500, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PINN()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    for step in range(steps):
        X = torch.from_numpy(rng.uniform(-1, 1, (batch, 2))).requires_grad_(True)
        s = rng.uniform(-1, 1, n_bc)
        edge = rng.integers(0, 4, n_bc)
        Pb = np.empty((n_bc, 2))
        Pb[:, 0] = np.where(edge < 2, s, np.where(edge == 2, -1.0, 1.0))
        Pb[:, 1] = np.where(edge >= 2, s, np.where(edge == 0, -1.0, 1.0))
        Xb = torch.from_numpy(Pb)
        loss = pde_loss(net, X, nu) + bc_weight * bc_loss(net, Xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            rec = dict(step=step, loss=float(loss.item()),
                       rel_l2_u=_eval_rel_l2(net))
            history.append(rec)
            print(rec, flush=True)
    return net, history


def pinn_fields(net):
    """Frozen-net numpy fields adapter: P [n,2] -> dict of numpy [n] arrays
    (u, ux, uy, lap_u, v, vx, vy, lap_v) — the base_fields contract of
    expF06 newton.newton_burgers."""
    def fields(P, chunk=2048):
        out = {k: np.empty(len(P)) for k in
               ["u", "ux", "uy", "lap_u", "v", "vx", "vy", "lap_v"]}
        for i in range(0, len(P), chunk):
            X = torch.from_numpy(np.ascontiguousarray(P[i:i + chunk]))
            X.requires_grad_(True)
            d = _derivs(net, X)
            for k in out:
                out[k][i:i + chunk] = d[k].detach().numpy()
        return out
    return fields
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_expF07_pinn_finisher.py -v`
Expected: both PASS (~30-60 s)

- [ ] **Step 5: Commit**

```bash
git add experiments/expF07_pinn_finisher/ tests/test_expF07_pinn_finisher.py
git commit -m "expF07: torch tanh-MLP PINN + frozen-net fields adapter"
```

---

### Task 11: expF07 finisher smoke test + run.py

**Files:**
- Create: `experiments/expF07_pinn_finisher/run.py`
- Test: append to `tests/test_expF07_pinn_finisher.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.slow
def test_finisher_improves_pinn():
    import pinn
    import newton as nt
    net, hist = pinn.train_pinn(nu=0.1, steps=400, batch=512, eval_every=200, seed=0)
    before = hist[-1]["rel_l2_u"]
    res = nt.newton_burgers(nu=0.1, W=256, lam=0.25, max_iter=3, seed=0,
                            base_fields=pinn.pinn_fields(net))
    after = res["history"][-1]["rel_l2_u"]
    assert after < max(1e-4, 0.01 * before), (before, after)
```

- [ ] **Step 2: Run test to verify it fails-then-passes**

Run: `uv run --extra dev pytest tests/test_expF07_pinn_finisher.py::test_finisher_improves_pinn -v`
Expected: PASS directly if Tasks 8/10 are correct (the machinery already exists; this test pins the integration contract). If it FAILS, debug before writing run.py — the most likely bug is sys.path module shadowing (see Task 7 Step 2 note).

- [ ] **Step 3: Write run.py**

```python
# experiments/expF07_pinn_finisher/run.py
"""Experiment expF07 -- lstsq precision finisher for a trained PINN.

Train a vanilla torch tanh-MLP PINN (4x64, Adam) on the expF06 Burgers
problem (nu=0.1) to its plateau, freeze it, then run the expF06 Newton-lstsq
loop warm-started at the PINN (base_fields). Full: 50k Adam steps, polish
W=1024, 3 Newton steps. Smoke: 400 steps, W=256, 2 steps.

Outputs (results/checkpoint_F_applications/expF07_pinn_finisher/):
  finisher_convergence.png  rel L2(u): Adam curve then polish steps
  pinn_ckpt.pt              trained PINN state_dict
  data.json                 training history + polish history + wall clocks

Usage:
  uv run --extra dev python experiments/expF07_pinn_finisher/run.py [--smoke] [--plot]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF05_spline_ridge"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "expF06_newton_burgers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pinn
import newton as nt

RESULTS_DIR = REPO_ROOT / "results" / "checkpoint_F_applications" / "expF07_pinn_finisher"
DATA_PATH = RESULTS_DIR / "data.json"
NU = 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot()
        return
    steps, W, n_polish = (400, 256, 2) if args.smoke else (50000, 1024, 3)
    t0 = time.time()
    net, train_hist = pinn.train_pinn(nu=NU, steps=steps, seed=0)
    t_train = time.time() - t0
    torch.save(net.state_dict(), RESULTS_DIR / "pinn_ckpt.pt")
    t0 = time.time()
    res = nt.newton_burgers(nu=NU, W=W, lam=0.25, max_iter=n_polish, seed=0,
                            base_fields=pinn.pinn_fields(net))
    t_polish = time.time() - t0
    data = dict(nu=NU, steps=steps, W=W, t_train_s=t_train, t_polish_s=t_polish,
                train_history=train_hist, polish_history=res["history"])
    DATA_PATH.write_text(json.dumps(data, indent=1))
    print(f"train {t_train:.0f}s -> rel_l2 {train_hist[-1]['rel_l2_u']:.2e}; "
          f"polish {t_polish:.0f}s -> rel_l2 {res['history'][-1]['rel_l2_u']:.2e}",
          flush=True)
    plot()


def plot():
    data = json.loads(DATA_PATH.read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    th = data["train_history"]
    ax.semilogy([h["step"] for h in th], [h["rel_l2_u"] for h in th],
                "-", label=f"Adam ({data['t_train_s']:.0f}s)")
    last_step = th[-1]["step"]
    ph = data["polish_history"]
    steps = [last_step + (i + 1) * max(1, last_step // 20) for i in range(len(ph))]
    ax.semilogy(steps, [h["rel_l2_u"] for h in ph], "o-",
                label=f"Newton-lstsq polish ({data['t_polish_s']:.0f}s)")
    ax.set_xlabel("Adam step (polish appended)")
    ax.set_ylabel("rel L2(u)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "finisher_convergence.png", dpi=140)
    print("plot saved to", RESULTS_DIR)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke-run**

Run: `uv run --extra dev python experiments/expF07_pinn_finisher/run.py --smoke`
Expected: PINN trains 400 steps (rel_l2 ~1e-1-1e-2), polish drops it several orders in 2 steps; plot + ckpt + data.json created.

- [ ] **Step 5: Delete smoke artifacts, commit**

```bash
rm results/checkpoint_F_applications/expF07_pinn_finisher/data.json results/checkpoint_F_applications/expF07_pinn_finisher/pinn_ckpt.pt
git add experiments/expF07_pinn_finisher/run.py tests/test_expF07_pinn_finisher.py
git commit -m "expF07: PINN-then-polish driver"
```

---

### Task 12: expF07 full run + results.md summary

- [ ] **Step 1: Full run in the background**

Run: `nohup uv run --extra dev python experiments/expF07_pinn_finisher/run.py > /tmp/expF07.log 2>&1 &`
Expected: 50k Adam steps ~30-90 min CPU, polish ~5-15 min. Spec pass: ≥4 orders improvement over the Adam plateau, landing within ~1 order of the expF06 floor at W=1024.

- [ ] **Step 2: Commit results**

```bash
git add results/checkpoint_F_applications/expF07_pinn_finisher/
git commit -m "expF07: full finisher results (Adam plateau -> Newton-lstsq polish)"
```

- [ ] **Step 3: Append a summary block to results.md**

Append to `results/results.md` (create the heading if the file structure differs — check `head results/results.md` first and match its style):

```markdown
## Checkpoint F additions (2026-07): PINN integration

- expF05 (spline ridges): cubic B-spline family floor vs tanh on poisson +
  smooth darcy control: <fill actual numbers from data.json>. Adaptive knots
  on rough darcy_421 (sigma=0, width-matched 2304): <fill: rel_l2 per round
  vs dense 7.2e-2 baseline>.
- expF06 (Newton-lstsq Burgers): nu=0.1 floor <fill>, nu=0.01 <fill>,
  quadratic convergence in <fill> iterations.
- expF07 (PINN finisher): Adam plateau <fill> after <fill>s; +<fill> Newton
  steps (<fill>s) -> <fill>. "<X> minutes of Adam + <Y> seconds of lstsq."
```

Fill every `<fill>` from the data.json files — no placeholders may survive into the commit.

```bash
git add results/results.md
git commit -m "expF05-04: results.md summary for PINN-integration experiments"
```

---

## Self-review notes (already applied)

- Spec coverage: Part A (Task 5/6), Part B adaptive (Tasks 4/5/6), Burgers Newton (Tasks 7-9), PINN + finisher (Tasks 10-12), tests-with-FD-verification throughout, --smoke/--plot everywhere, slow markers on >10s tests.
- Module shadowing: expF05 and expF06 both define `problems.py`; the sys.path insertion order rule is documented in Task 7 Step 2 and followed in every run.py/test file (expF06/expF07 dirs inserted after expF05, i.e., they win).
- The spec's quintic-spline and nu-continuation escalations are deliberately NOT tasks — they trigger only on recorded failures, as separate follow-up work.
- Known-baseline cross-checks: dense tanh Part B should reproduce ~7.2e-2 (continuous-mlps July-14 sweep); if it doesn't land within ~2x, stop and investigate the vendoring (grid convention, forcing 0.25) before trusting anything downstream.
