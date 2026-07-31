"""VarPro + LSMR Gauss-Newton -- matrix-free, O(m), machine-precision.

The dense VarPro+Adam->GN hits the fp64 floor but forms J^T J (a (2W)^2 system),
violating the practicality gate. This variant solves the SAME Gauss-Newton step
matrix-free with LSMR (the Krylov least-squares method): it needs only the
Kaufman Jacobian's action J@v and J^T@w, never the dense normal equations. State
is O(m) in Krylov vectors -- Adam's class.

Question: does matrix-free LSMR-GN still reach the fp64 floor from QI geometry,
i.e. can we get mp WITHOUT the dense O(n^2) solve?

Reduced (Kaufman VarPro) problem, geometry theta=(a,b), readout projected out:
  A(theta) = [tanh(a x + b), 1];  c = A^+ y  (readout, solved)
  r = A c - y;  J_theta = d r / d(a,b) with c fixed;  J = (I - Q Q^T) J_theta
  GN step: min_delta || J delta + r ||  -- solved by LSMR, matrix-free.

pure numpy + scipy.sparse.linalg.lsmr, fp64.
Usage: uv run --extra dev python experiments/expD04_varpro/varpro_lsmr_probe.py --mode {smoke,full}
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_D05_PATH = REPO_ROOT / "experiments" / "expD05_scale_init_story" / "run.py"
_spec = importlib.util.spec_from_file_location("expd05_run", _D05_PATH)
d05 = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["expd05_run"] = d05
_spec.loader.exec_module(d05)  # type: ignore[union-attr]

from src.data.targets import get_target  # noqa: E402

OUT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "varpro_lsmr_summary.csv"

TARGETS = ("sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed")
RESOLUTIONS = (32, 64, 128, 256, 512)
SEEDS = (0, 1)
FAMILY = "qi_geom_random_readout"          # free QI geometry, readout projected out
WARMUP = 100
GN_STEPS = 30
LSMR_ATOL = 1e-13
LSMR_MAXIT = 120          # capped inner budget (Newton tolerates a loose inner solve)
N_TRAIN, N_EVAL = 2003, 4001


def rel_l2(pred, y):
    return float(np.linalg.norm(pred - y) / np.linalg.norm(y))


def solve_readout(theta, W, x, y):
    a, b = theta[:W], theta[W:]
    phi = np.tanh(x[:, None] * a[None, :] + b[None, :])
    A = np.concatenate([phi, np.ones((x.size, 1))], axis=1)
    Q, R = np.linalg.qr(A)
    rdiag = np.abs(np.diag(R))
    if rdiag.size and rdiag.min() > 1e-13 * rdiag.max():
        c = np.linalg.solve(R, Q.T @ y)
    else:
        c, *_ = np.linalg.lstsq(A, y, rcond=1e-13)
    r = A @ c - y
    return phi, Q, c, r


def gn_operator(phi, Q, c, W, x, n):
    """Matrix-free Kaufman Jacobian J = (I - QQ^T) J_theta as a scipy LinearOperator.
    J_theta columns: d r/d a_j = v_j sech^2 x,  d r/d b_j = v_j sech^2."""
    v = c[:W]
    sech2 = 1.0 - phi * phi
    vs = v[None, :] * sech2          # (n, W)
    vsx = vs * x[:, None]

    def matvec(d):                    # J @ d,  d in R^{2W} -> R^n
        Jt_d = vsx @ d[:W] + vs @ d[W:]
        return Jt_d - Q @ (Q.T @ Jt_d)

    def rmatvec(w):                   # J^T @ w,  w in R^n -> R^{2W}
        Pw = w - Q @ (Q.T @ w)
        return np.concatenate([vsx.T @ Pw, vs.T @ Pw])

    return LinearOperator((n, 2 * W), matvec=matvec, rmatvec=rmatvec)


def eval_rel(theta, W, xt, yt, xe, ye):
    _, _, c, _ = solve_readout(theta, W, xt, yt)
    a, b = theta[:W], theta[W:]
    pred = np.tanh(xe[:, None] * a[None, :] + b[None, :]) @ c[:W] + c[W]
    return rel_l2(pred, ye)


def run_case(target, resolution, seed):
    geom = d05.geometry_for_resolution(resolution)
    W = geom.width
    st = d05.build_initial_state(FAMILY, target, geom, seed)
    t = get_target(target)
    xt = np.linspace(-1, 1, N_TRAIN); yt = t.fn_numpy(xt).astype(np.float64)
    xe = np.linspace(-1, 1, N_EVAL); ye = t.fn_numpy(xe).astype(np.float64)
    theta = np.concatenate([st.input_weights, st.input_biases]).astype(np.float64)
    n = N_TRAIN

    best = eval_rel(theta, W, xt, yt, xe, ye)

    # Adam warmup on geometry (VarPro reduced gradient = J^T r, matrix-free)
    m = np.zeros_like(theta); vv = np.zeros_like(theta); b1, b2, eps = 0.9, 0.999, 1e-12
    for it in range(1, WARMUP + 1):
        phi, Q, c, r = solve_readout(theta, W, xt, yt)
        g = gn_operator(phi, Q, c, W, xt, n).rmatvec(r)
        if not np.all(np.isfinite(g)):
            break
        m = b1 * m + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
        theta = theta - 1e-2 * (m / (1 - b1 ** it)) / (np.sqrt(vv / (1 - b2 ** it)) + eps)
        if it % 50 == 0:
            best = min(best, eval_rel(theta, W, xt, yt, xe, ye))

    # LSMR Gauss-Newton: each step solve min ||J delta + r|| matrix-free
    prev = np.inf
    for _ in range(GN_STEPS):
        phi, Q, c, r = solve_readout(theta, W, xt, yt)
        loss = float(r @ r)
        J = gn_operator(phi, Q, c, W, xt, n)
        delta = lsmr(J, -r, atol=LSMR_ATOL, btol=LSMR_ATOL, maxiter=LSMR_MAXIT)[0]
        if not np.all(np.isfinite(delta)):
            break
        step, accepted = 1.0, False
        for _bt in range(30):
            cand = theta + step * delta
            _, _, _, rc = solve_readout(cand, W, xt, yt)
            if np.isfinite(rc @ rc) and rc @ rc < loss:
                theta, accepted = cand, True
                break
            step *= 0.5
        best = min(best, eval_rel(theta, W, xt, yt, xe, ye))
        if not accepted or abs(prev - loss) <= 1e-30:
            break
        prev = loss
    return W, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = ap.parse_args(argv)
    targets = ("runge", "sine_mixture") if args.mode == "smoke" else TARGETS
    resolutions = (64, 256) if args.mode == "smoke" else RESOLUTIONS
    seeds = (0,) if args.mode == "smoke" else SEEDS

    rows = []
    total = len(targets) * len(resolutions) * len(seeds)
    i = 0
    for target in targets:
        for resolution in resolutions:
            for seed in seeds:
                i += 1
                W, best = run_case(target, resolution, seed)
                rows.append({"target": target, "resolution": resolution, "width": W,
                             "seed": seed, "lsmr_best_eval_rel_l2": best})
                print(f"[{i:2d}/{total}] {target:>12s} N={resolution:<4d} W={W:<4d} "
                      f"seed={seed} best={best:.2e}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
