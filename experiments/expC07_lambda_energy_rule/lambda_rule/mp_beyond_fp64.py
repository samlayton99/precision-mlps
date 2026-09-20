"""Beyond fp64, natively: the least-squares fit in mpmath at 100 bits (dps 30) and at ~265 bits (dps 80). tanh only, sin(pi x), N = 32, halo 32
(W = 97), 400 sample points (4 per cell), least squares via the NORMAL EQUATIONS solved by LU in mpmath at the stated dps
(mpmath's QR refuses the near-duplicate saturated halo columns as "numerically singular"; the normal equations square the
condition number, ~1e30, so dps 80 is the trustworthy run and dps 30 is a check of what 100 bits buys), error on 801 points. Prediction: the wall (fiber curve at 10x the floor) moves left from ~0.3 (fp64) to ~0.15 (100 bits);
the constant A_K = 2^-100 is lambda = 0.143. Slow (pure-Python arithmetic); run in the background.
Output: data/mp_beyond_fp64.json, figures/mp_beyond_fp64.png.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent; REPO_ROOT = HERE.parents[2]; sys.path.insert(0, str(HERE))
from rule import log_khat, LAM_GRID_RULE  # noqa: E402
OUT = REPO_ROOT / "results" / "checkpoint_C_geometry" / "expC07_lambda_energy_rule" / "lambda_rule"
N, HALO, NS = 32, 32, 400
LAMS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.26, 0.30, 0.35, 0.40]


def fit_error(lam, dps):
    mp.mp.dps = dps
    h = mp.mpf(2) / N; g = mp.mpf(lam) / h
    cs = [mp.mpf(-1) + k * h for k in range(-HALO, N + HALO + 1)]
    xs = [mp.mpf(-1) + mp.mpf(2) * i / (NS - 1) for i in range(NS)]
    A = mp.matrix(NS, len(cs) + 1)
    for i, x in enumerate(xs):
        for k, c in enumerate(cs):
            A[i, k] = mp.tanh(g * (x - c))
        A[i, len(cs)] = 1
    b = mp.matrix([mp.sin(mp.pi * x) for x in xs])
    AtA = A.T * A; Atb = A.T * b
    ridge = mp.mpf(10) ** (-(dps - 20)) * max(AtA[i, i] for i in range(AtA.rows))
    for i in range(AtA.rows):
        AtA[i, i] += ridge
    sol = mp.lu_solve(AtA, Atb)
    xe = [mp.mpf(-1) + mp.mpf(2) * i / 800 for i in range(801)]
    num = den = mp.mpf(0)
    for x in xe:
        fit = sol[len(cs)]
        for k, c in enumerate(cs):
            fit += sol[k] * mp.tanh(g * (x - c))
        fe = mp.sin(mp.pi * x); num += (fit - fe) ** 2; den += fe ** 2
    return float(mp.sqrt(num / den))


def fiber(lam, k=1, mmax=8):
    th = k * np.pi * 2.0 / N; ms = np.arange(-mmax, mmax + 1); tt = th + 2 * np.pi * ms
    lt = 2 * log_khat("tanh", np.abs(tt) / lam) - 2 * np.log(np.abs(tt)); lt -= lt.max(); v = np.exp(lt)
    return float(np.sqrt(v[ms != 0].sum() / v.sum()))


def main():
    out = {"N": N, "halo": HALO, "n_samples": NS, "lams": LAMS, "dps30": [], "dps80": []}
    t0 = time.time()
    for dps, key in ((30, "dps30"), (80, "dps80")):
        for lam in LAMS:
            e = fit_error(lam, dps); out[key].append(e)
            print(f"dps={dps} lam={lam:.2f} rel_l2={e:.3e} fiber={fiber(lam):.3e} ({time.time()-t0:.0f}s)", flush=True)
            (OUT / "data" / "mp_beyond_fp64.json").write_text(json.dumps(out, indent=1))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ll = np.geomspace(0.06, 0.45, 200)
    ax.loglog(ll, [fiber(l) for l in ll], "k-.", lw=1, label="fiber floor (exact arithmetic)")
    ax.loglog(LAMS, out["dps30"], "o-", color="tab:blue", label="mpmath normal equations, dps 30 (~100 bits)")
    ax.loglog(LAMS, out["dps80"], "s-", color="#d1352b", label="mpmath normal equations, dps 80 (~265 bits)")
    for eps, col, lab in ((2.0 ** -52, "tab:blue", r"$A_K=2^{-52}$"), (2.0 ** -100, "#d1352b", r"$A_K=2^{-100}$")):
        g = LAM_GRID_RULE; ok = np.where(log_khat("tanh", 2 * np.pi / g) - np.log(eps) <= 0)[0]
        ax.axvline(float(g[ok.max()]), color=col, ls=":", lw=1.2, label=f"constant {lab}")
    ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"rel $L_2$"); ax.set_ylim(1e-40, 1e-2); ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9, frameon=False)
    ax.set_title(r"tanh, $\sin\pi x$, $N=32$, halo 32: the wall moves left as precision increases", fontsize=10, y=1.18)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "mp_beyond_fp64.png", dpi=150); print("saved")


if __name__ == "__main__":
    main()
