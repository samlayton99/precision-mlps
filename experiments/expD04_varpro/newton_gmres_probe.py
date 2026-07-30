"""Matrix-free Newton-GMRES on a SQUARE collocation system (no least squares).

Every mp-reaching method so far solves a linear system by a direct least-squares
(the readout, or the Gauss-Newton normal equations). This probe removes the
least-squares framing entirely: pick exactly as many collocation points as
parameters, so the residual map r(theta): R^p -> R^p is square, and root-find
r(theta)=0 by Newton's method. Each Newton step solves the SQUARE nonsymmetric
system J delta = -r with GMRES, matrix-free (only needs J@d) -- which is exactly
what GMRES is for. No dense Jacobian formed, no readout projection.

Params p = [a, b, v, c] (full network), so #collocation = 3W+1. The J@d matvec
is the analytic tanh-MLP Jacobian-vector product. pure numpy fp64 (GMRES is
sequential; CPU beats GPU kernel-launch overhead for these sizes).

FINDING (negative, but informative): this stalls at ~1e-3, nowhere near mp.
The square Jacobian is badly rank-deficient (e.g. W=205 -> p=616 but rank ~109):
a width-W sum of tanh's has ~O(W) effective DOF, so it cannot satisfy 3W+1
independent interpolation conditions -- r(theta)=0 has no root. GMRES solves each
inner step fine (rel-resid ~1e-13); the FORMULATION is ill-posed. Conclusion:
the least-squares (overdetermined) structure is forced by the model's rank
deficiency, not incidental -- a square/determined formulation cannot reach mp.

Usage:
    uv run --extra dev python experiments/expD04_varpro/newton_gmres_probe.py --mode smoke
    uv run --extra dev python experiments/expD04_varpro/newton_gmres_probe.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

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

OUT = REPO_ROOT / "results" / "checkpoint_D_optimizers" / "expD04_varpro" / "newton_gmres_summary.csv"

TARGETS = ("runge", "sine_mixture")
RESOLUTIONS = (32, 64, 128, 256, 512)
SEEDS = (0, 1)
FAMILY = "qi_geom_random_readout"
NEWTON_STEPS = 40
GMRES_RESTART = 300        # inner Krylov dimension before restart
GMRES_TOL = 1e-13          # relative-residual tolerance for the inner solve
GMRES_MAXIT = 900
N_EVAL = 4001


def gmres(matvec, b, m=300, tol=1e-13, maxit=900):
    """Restarted GMRES(m) for a square operator matvec: v -> A@v (numpy fp64).
    Arnoldi with modified Gram-Schmidt + Givens rotations. Returns (x, rel_resid)."""
    n = b.size
    x = np.zeros(n)
    bnorm = np.linalg.norm(b)
    if bnorm == 0.0:
        return x, 0.0
    r = b - matvec(x)
    it_total = 0
    while it_total < maxit:
        beta = np.linalg.norm(r)
        if beta / bnorm <= tol:
            break
        V = np.zeros((m + 1, n))
        H = np.zeros((m + 1, m))
        cs = np.zeros(m); sn = np.zeros(m)
        g = np.zeros(m + 1); g[0] = beta
        V[0] = r / beta
        k_used = 0
        for k in range(m):
            it_total += 1
            w = matvec(V[k])
            for j in range(k + 1):
                H[j, k] = V[j] @ w
                w = w - H[j, k] * V[j]
            H[k + 1, k] = np.linalg.norm(w)
            if H[k + 1, k] > 1e-300:
                V[k + 1] = w / H[k + 1, k]
            for j in range(k):
                t = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = t
            denom = np.hypot(H[k, k], H[k + 1, k])
            cs[k] = H[k, k] / denom; sn[k] = H[k + 1, k] / denom
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]; g[k] = cs[k] * g[k]
            k_used = k + 1
            if abs(g[k + 1]) / bnorm <= tol or it_total >= maxit:
                break
        y = np.linalg.solve(H[:k_used, :k_used], g[:k_used])
        x = x + V[:k_used].T @ y
        r = b - matvec(x)
    return x, float(np.linalg.norm(r) / bnorm)


def run_case(target, resolution, seed):
    geom = d05.geometry_for_resolution(resolution)
    W = geom.width
    p = 3 * W + 1
    st = d05.build_initial_state(FAMILY, target, geom, seed)
    t = get_target(target)

    xc = np.linspace(-1, 1, p)                       # square: p collocation points
    yc = t.fn_numpy(xc).astype(np.float64)
    xe = np.linspace(-1, 1, N_EVAL)
    ye = t.fn_numpy(xe).astype(np.float64)
    theta = np.concatenate([st.input_weights, st.input_biases,
                            st.readout_weights, [st.readout_bias]]).astype(np.float64)

    def resid(th):
        a, b, v, c = th[:W], th[W:2 * W], th[2 * W:3 * W], th[3 * W]
        return np.tanh(xc[:, None] * a[None, :] + b[None, :]) @ v + c - yc

    def eval_rel(th):
        a, b, v, c = th[:W], th[W:2 * W], th[2 * W:3 * W], th[3 * W]
        pred = np.tanh(xe[:, None] * a[None, :] + b[None, :]) @ v + c
        return float(np.linalg.norm(pred - ye) / np.linalg.norm(ye))

    best = eval_rel(theta)
    for _ in range(NEWTON_STEPS):
        r = resid(theta)
        rnorm = float(np.linalg.norm(r))
        if not np.isfinite(rnorm):
            break
        a, b, v = theta[:W], theta[W:2 * W], theta[2 * W:3 * W]
        phi = np.tanh(xc[:, None] * a[None, :] + b[None, :])
        vs = v[None, :] * (1.0 - phi * phi)          # (p, W)
        vsx = vs * xc[:, None]

        def jvp(d):                                   # analytic J@d, matrix-free
            return vsx @ d[:W] + vs @ d[W:2 * W] + phi @ d[2 * W:3 * W] + d[3 * W]

        delta, _ = gmres(jvp, -r, m=GMRES_RESTART, tol=GMRES_TOL, maxit=GMRES_MAXIT)
        if not np.all(np.isfinite(delta)):
            break
        step, accepted = 1.0, False
        for _bt in range(40):
            cand = theta + step * delta
            rc = float(np.linalg.norm(resid(cand)))
            if np.isfinite(rc) and rc < rnorm:
                theta, accepted = cand, True
                break
            step *= 0.5
        best = min(best, eval_rel(theta))
        if not accepted or rnorm <= 1e-30:
            break
    return W, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    args = ap.parse_args(argv)
    targets = ("runge",) if args.mode == "smoke" else TARGETS
    resolutions = (64, 128) if args.mode == "smoke" else RESOLUTIONS
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
                             "seed": seed, "newton_gmres_best_eval_rel_l2": best})
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
