"""expF14 -- extended-precision reference trajectories for the dysts systems.

A chaotic IVP amplifies any error by exp(lambda t), so a fp64 reference is not
good enough to certify a fp64 solution: DOP853 at rtol=atol=1e-13/1e-14 is
already only ~6e-13 relative on Lorenz over three Lyapunov times (measured in
`crosscheck`).  The reference here is mpmath's Taylor-method `odefun` at 30
decimal digits, rounded to fp64 afterwards -- exactly the repo's mpmath
convention for the QI construction: an offline extended-precision
precomputation producing fp64 constants.

Parameters and the initial condition are passed to mpmath as the *exact fp64
values*, so the reference solves the same IVP the fp64 solver is given.

Trajectories are cached (uncompressed .npz) keyed by (system, T, n_eval, dps).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = (REPO_ROOT / "results" / "checkpoint_F_applications"
             / "expF14_dysts_chaos" / "ref_cache")

DPS = 30
ODE_TOL_EXP = 25       # mpmath odefun tol = 10^-25
DEGREE = 12


def _mp_rhs(sys):
    """Return a scalar-list rhs f(t, y) in mpmath for `sys`."""
    import mpmath as mp
    p = {k: mp.mpf(v) for k, v in sys.params.items()}
    name = sys.name
    if name == "Lorenz":
        def f(t, y):
            x, yy, z = y
            return [p["sigma"] * (yy - x), p["rho"] * x - x * z - yy,
                    x * yy - p["beta"] * z]
    elif name == "Rossler":
        def f(t, y):
            x, yy, z = y
            return [-yy - z, x + p["a"] * yy, p["b"] + z * x - p["c"] * z]
    elif name == "Thomas":
        def f(t, y):
            x, yy, z = y
            a, b = p["a"], p["b"]
            return [-a * x + b * mp.sin(yy), -a * yy + b * mp.sin(z),
                    -a * z + b * mp.sin(x)]
    elif name == "Halvorsen":
        def f(t, y):
            x, yy, z = y
            a, b = p["a"], p["b"]
            return [-a * x - b * yy - b * z - yy ** 2,
                    -a * yy - b * z - b * x - z ** 2,
                    -a * z - b * x - b * yy - x ** 2]
    elif name == "Lorenz96":
        d = sys.d
        FF = p["f"]

        def f(t, y):
            return [(y[(i + 1) % d] - y[(i - 2) % d]) * y[(i - 1) % d] - y[i] + FF
                    for i in range(d)]
    else:
        raise ValueError(name)
    return f


def reference(sys, T, n_eval, dps=DPS, use_cache=True, verbose=True):
    """Dense reference trajectory: (ts, Y) with ts uniform on [0, T], Y (n_eval, d)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{sys.name}_T{T:.10g}_n{n_eval}_dps{dps}.npz"
    path = CACHE_DIR / key
    if use_cache and path.exists():
        z = np.load(path)
        return z["ts"], z["Y"]

    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        f = _mp_rhs(sys)
        ic = [mp.mpf(float(v)) for v in sys.ic]
        t0 = time.time()
        sol = mp.odefun(f, mp.mpf(0), ic, tol=mp.mpf(10) ** (-ODE_TOL_EXP),
                        degree=DEGREE)
        sol(mp.mpf(T))                     # drive the integrator to the end once
        ts = np.linspace(0.0, T, n_eval)
        Y = np.empty((n_eval, sys.d))
        for i, t in enumerate(ts):
            Y[i] = [float(v) for v in sol(mp.mpf(t))]
        if verbose:
            print(f"  mpmath reference {sys.name} T={T:.4f} dps={dps}: "
                  f"{time.time() - t0:.1f}s", flush=True)
    finally:
        mp.mp.dps = old
    np.savez(path, ts=ts, Y=Y)
    return ts, Y


def rk_trajectory(sys, T, ts, rtol, atol, method="DOP853"):
    """fp64 Runge-Kutta trajectory sampled at `ts` (used as warm start / baseline)."""
    from scipy.integrate import solve_ivp

    def g(t, y):
        return sys.F(y[None, :])[0]

    s = solve_ivp(g, [0.0, T], sys.ic, rtol=rtol, atol=atol, method=method,
                  dense_output=True)
    if not s.success:
        raise RuntimeError(f"{sys.name}: solve_ivp failed -- {s.message}")
    return s.sol(ts).T, int(s.nfev)


def crosscheck(sys, T, n_eval=2001, tols=((1e-13, 1e-14), (1e-10, 1e-12))):
    """How good is a *fp64* Runge-Kutta reference? Compare DOP853 to mpmath."""
    ts, Y = reference(sys, T, n_eval, verbose=False)
    out = {}
    for rtol, atol in tols:
        Z, nfev = rk_trajectory(sys, T, ts, rtol, atol)
        out[f"rtol{rtol:g}"] = dict(
            rel_l2=float(np.linalg.norm(Z - Y) / np.linalg.norm(Y)),
            linf=float(np.max(np.abs(Z - Y))), nfev=nfev)
    return out


def selfcheck(sys, T, n_eval=2001, dps_lo=25, dps_hi=40):
    """Reference self-consistency: the same integration at two working precisions."""
    _, Ylo = reference(sys, T, n_eval, dps=dps_lo, verbose=False)
    _, Yhi = reference(sys, T, n_eval, dps=dps_hi, verbose=False)
    return dict(rel_l2=float(np.linalg.norm(Ylo - Yhi) / np.linalg.norm(Yhi)),
                linf=float(np.max(np.abs(Ylo - Yhi))))
