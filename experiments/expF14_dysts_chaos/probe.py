"""Does staying in Newton past the residual floor keep improving the solution?"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, systems, core, reference
from reference import rk_trajectory

def trace(name, N, lyapT=3.0, warm_rtol=1e-8, nsteps=10):
    S = systems.System(name); T = S.horizon(lyapT)
    ts, Yref = reference.reference(S, T, 6001)
    centers, gamma = core.geometry(N)
    W = centers.size; p = core.n_params(centers); n_col = 4 * W
    s_col = np.linspace(-1, 1, n_col); t_col = T * (s_col + 1) / 2
    D0 = core.dict_rows(s_col, centers, gamma, 0)
    D1 = core.dict_rows(s_col, centers, gamma, 1)
    Yrk, _ = rk_trajectory(S, T, t_col, warm_rtol, warm_rtol * 1e-3)
    sigma = np.maximum(np.sqrt(np.mean(Yrk ** 2, axis=0)), 1e-12)
    w_ic = np.sqrt(n_col)
    B, g = core._ic_block(S, centers, gamma, sigma, p, w_ic)
    M = np.vstack([D0, w_ic * core.dict_rows(np.array([-1.0]), centers, gamma, 0)])
    rhs = np.vstack([Yrk / sigma[None, :], w_ic * (S.ic / sigma)[None, :]])
    A = np.linalg.lstsq(M, rhs, rcond=core.RCOND)[0].T
    row = []
    for k in range(nsteps):
        A, h = core.gauss_newton(S, A, D0, D1, sigma, T, B, g, max_it=1)
        cell = dict(A=A, centers=centers, gamma=gamma, sigma=sigma, T=T, p=p)
        e = core.errors(core.model_trajectory(cell, ts), Yref)
        row.append((h[0], e[0]))
    print(f"{name} N={N} warm={warm_rtol:.0e}: " +
          "  ".join(f"[{i+1}] r={r:.1e} e={e:.2e}" for i, (r, e) in enumerate(row)),
          flush=True)

for name in ["Lorenz", "Rossler", "Thomas", "Lorenz96"]:
    trace(name, 384)
