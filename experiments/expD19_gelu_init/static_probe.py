"""expD19 static probes -- the diagnosis, measured on the feature matrix alone.

No training. Builds the 1-D QI geometry under a given activation and reports,
per variant: feature-column-norm spread by region, the condition number over
the kept singular values, and the truncated-SVD lstsq floor.

This is the gate that fixes the numbers the training study is built on:

    GELU, N=128, lambda=0.707
      [Phi,1]  (current init)            colnorm ratio ~1e303, cond 2.9e11, floor 2.25e-13
      unit-col-normalized + linear term                              floor 2.9e-14
      halo 8/8 + colnorm + linear        W=147                       floor 3.7e-14
      halo 8/0 or 0/8                                                floor 5.3e-9
      no halo                                                        floor 7.5e-9
    tanh, N=128, lambda=0.25
      halo 8/8 -> 4.2e-15  beats  halo 59/59 -> 3.3e-14

Usage:
    uv run --extra dev python experiments/expD19_gelu_init/static_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RCOND = 1e-13
LAMBDA = {"tanh": 0.25, "gelu": 0.707}


def act_np(name, z):
    if name == "tanh":
        return np.tanh(z)
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    raise ValueError(name)


def build(act_name, N, halo_l, halo_r, x):
    """Feature matrix on the 1-D QI geometry with an asymmetric halo."""
    h = 2.0 / N
    gamma = LAMBDA[act_name] / h
    c = np.array([-1.0 + k * h for k in range(-halo_l, N + halo_r + 1)])
    return act_np(act_name, gamma * (x[:, None] - c[None, :])), c


def solve_floor(M, Me, y, ye):
    """Truncated-SVD least squares at the repo's rcond; eval relative L2."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    keep = s > RCOND * s[0]
    beta = Vt[keep].T @ ((U[:, keep].T @ y) / s[keep])
    cond = s[0] / s[keep][-1]
    return float(np.linalg.norm(Me @ beta - ye) / np.linalg.norm(ye)), float(cond)


def variant(act_name, N, halo_l, halo_r, colnorm, linear, x, xe, f):
    A, c = build(act_name, N, halo_l, halo_r, x)
    Ae, _ = build(act_name, N, halo_l, halo_r, xe)
    if colnorm:
        s = np.linalg.norm(A, axis=0)
        s[s < 1e-300] = 1.0                     # dead columns: leave untouched
        A, Ae = A / s, Ae / s
    cols = [A, np.ones((len(x), 1))]
    colse = [Ae, np.ones((len(xe), 1))]
    if linear:
        cols.append(x[:, None])
        colse.append(xe[:, None])
    M, Me = np.hstack(cols), np.hstack(colse)
    floor, cond = solve_floor(M, Me, f(x), f(xe))
    cn = np.linalg.norm(A, axis=0)
    live = cn > 1e-300
    ratio = cn.max() / max(cn[live].min(), 1e-300) if live.any() else np.inf
    raw_ratio = cn.max() / max(cn.min(), 1e-300)
    return dict(W=M.shape[1], floor=floor, cond=cond, colnorm_ratio=raw_ratio,
                colnorm_ratio_live=ratio, centers=c, col_norms=cn)


def region_means(c, cn):
    L, R, I = c < -1, c > 1, np.abs(c) <= 1
    m = lambda msk: float(cn[msk].mean()) if msk.any() else float("nan")
    return m(I), m(L), m(R)


def main():
    N = 128
    x = np.linspace(-1, 1, 2003)
    xe = np.linspace(-1, 1, 4001)
    f = lambda t: np.sin(np.pi * t)
    Rfull = max(59, int(np.ceil(0.4 * N)))      # the repo's default_halo shape

    print(f"expD19 static probes -- 1-D sine, N={N}, rcond={RCOND:g}\n")
    print("GELU (lambda=0.707): column norms by region, conditioning, lstsq floor")
    print(f"{'variant':<40} {'W':>5} {'interior':>9} {'leftHalo':>9} "
          f"{'rightHalo':>10} {'max/min':>10} {'cond':>9} {'floor':>10}")
    rows = {}
    specs = [
        ("baseline [Phi,1]", Rfull, Rfull, False, False),
        ("+ linear term", Rfull, Rfull, False, True),
        ("+ colnorm + linear", Rfull, Rfull, True, True),
        ("halo 8/8, colnorm + linear", 8, 8, True, True),
        ("halo 8/8, raw", 8, 8, False, False),
        ("halo 8/0", 8, 0, False, False),
        ("halo 0/8", 0, 8, False, False),
        ("no halo", 0, 0, False, False),
    ]
    for tag, hl, hr, cnorm, lin in specs:
        v = variant("gelu", N, hl, hr, cnorm, lin, x, xe, f)
        i, l, r = region_means(v["centers"], v["col_norms"])
        rows[tag] = v
        print(f"{tag:<40} {v['W']:5d} {i:9.2f} {l:9.2f} {r:10.2e} "
              f"{v['colnorm_ratio']:10.2e} {v['cond']:9.2e} {v['floor']:10.2e}")

    print("\ntanh (lambda=0.25): is the halo rule oversized here too?")
    print(f"{'variant':<40} {'W':>5} {'interior':>9} {'leftHalo':>9} "
          f"{'rightHalo':>10} {'max/min':>10} {'cond':>9} {'floor':>10}")
    for tag, hl, hr in [("baseline halo 59/59", Rfull, Rfull),
                        ("halo 16/16", 16, 16), ("halo 8/8", 8, 8),
                        ("halo 4/4", 4, 4), ("no halo", 0, 0)]:
        v = variant("tanh", N, hl, hr, False, False, x, xe, f)
        i, l, r = region_means(v["centers"], v["col_norms"])
        print(f"{tag:<40} {v['W']:5d} {i:9.2f} {l:9.2f} {r:10.2e} "
              f"{v['colnorm_ratio']:10.2e} {v['cond']:9.2e} {v['floor']:10.2e}")
    return rows


if __name__ == "__main__":
    main()
