"""iteration_10 Phase 0e -- the precision law: stall level vs working precision
for the MEMORYLESS (<10-vector) CG core.

Phase 0a-0d closed off memory-free mechanisms one by one and appeared to confirm
iteration_9's "the floor costs rank memory". Then a ground-truth run broke it:
memoryless CG at 60 decimal digits, no reorthogonalization, reaches 8.0e-17 on a
kappa=1e6 problem where the SAME code in fp64 stalls at 1.4e-4.

So precision IS the lever -- and both prior attempts to test it were broken:
  * iteration_9 `b_replacement.py`: re-quenched the carried residual in fp64
    every 200 iterations and zeroed the compensation word; kept alpha in fp64.
  * my own `ddcg10.py` dd_all: takes only the hi word of beta, leaking fp64 back
    into the search direction.

This measures the law cleanly, with no dd hand-rolling at all: run the identical
memoryless CG at mpmath working precision dps in {16, 20, 24, 32, 40, 60} and
read off stall vs precision. If stall ~ u^1 * kappa^p, the exchange rate tells us
exactly how many digits the phi problems need -- i.e. what iteration 10 must
deliver, in COMPUTE (a constant factor) rather than MEMORY (which is what R5
actually forbids).

Everything is mpmath at the stated dps -- matrix entries, matvecs, dots, scalars,
state -- so there is no mixed-precision confound. dps=16 is the fp64 control and
must reproduce the fp64 stall.

Run: uv run python experiments/expD08_qi_init_nlcg/precision_law.py
Out: results/.../iteration_10/phase0/precision_law_{rows.json,.png}
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (REPO_ROOT / "results" / "checkpoint_D_optimizers"
           / "expD08_qi_init_nlcg" / "iteration_10" / "phase0")

DPS = [16, 20, 24, 32, 40, 60]
KAPPAS = [1e4, 1e6, 1e10]
M, R = 40, 120
ITER_MULT = 5            # iterations = ITER_MULT * m
SPECTRA = ["loglin", "phispec"]


def build(kind, kappa, seed=0):
    """Small dense system with a controlled spectrum, y in range (consistent)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(R, M)))
    V, _ = np.linalg.qr(rng.normal(size=(M, M)))
    if kind == "loglin":
        s = np.logspace(0, -math.log10(kappa), M)
    else:
        # the measured [Phi,1] shape: exponential decay over 0.6m, then a cliff
        k = int(0.6 * M)
        s = np.ones(M)
        s[:k] = np.logspace(0, -math.log10(kappa), k)
        s[k:] = s[k - 1] * 1e-4
    A = U @ np.diag(s) @ V.T
    vstar = rng.normal(size=M)
    return A, A @ vstar, s


def cg_at_dps(A, y, dps, iters, mixed="all"):
    """The campaign's memoryless core -- carried residual, exact GN step, PR+ --
    with EVERY quantity at mpmath precision dps. No memory, no reorthogonalization,
    no re-quench. Returns (best rel L2, trajectory)."""
    from mpmath import mp, mpf
    mp.dps = dps
    m, n = A.shape[1], A.shape[0]
    Am = [[mpf(float(A[i, j])) for j in range(m)] for i in range(n)]
    ym = [mpf(float(t)) for t in y]
    ny = mp.sqrt(sum((t * t for t in ym), mpf(0)))

    # mixed="mv_fp64": the JVP/VJP results are rounded to fp64, everything else
    # stays at dps. This is the DEPLOYABLE question -- a network's autodiff is
    # fp64 forever, so if the matvec must be high precision the mechanism needs
    # a compensated forward; if only the state and scalars must be, it ships now.
    rnd = (lambda t: mpf(float(t))) if mixed == "mv_fp64" else (lambda t: t)

    def Av(x):
        return [rnd(sum((Am[i][j] * x[j] for j in range(m)), mpf(0)))
                for i in range(n)]

    def Atv(x):
        return [rnd(sum((Am[i][j] * x[i] for i in range(n)), mpf(0)))
                for j in range(m)]

    v = [mpf(0)] * m
    r = [-t for t in ym]                      # carried residual at v = 0
    g = Atv(r)
    d = [-t for t in g]
    best, traj = mpf(10), []
    checks = set(np.unique(np.logspace(0, math.log10(iters), 26)
                           .astype(int)).tolist()) | {iters}
    for it in range(1, iters + 1):
        u = Av(d)
        c = sum((t * t for t in u), mpf(0))
        if c == 0:
            break
        al = -sum((g[j] * d[j] for j in range(m)), mpf(0)) / c
        v = [v[j] + al * d[j] for j in range(m)]
        r = [r[i] + al * u[i] for i in range(n)]
        gn = Atv(r)
        den = sum((t * t for t in g), mpf(0))
        if den == 0:
            break
        be = (sum((t * t for t in gn), mpf(0))
              - sum((gn[j] * g[j] for j in range(m)), mpf(0))) / den
        if be < 0:
            be = mpf(0)
        d = [-gn[j] + be * d[j] for j in range(m)]
        g = gn
        if it in checks:
            res = [sum((Am[i][j] * v[j] for j in range(m)), mpf(0)) - ym[i]
                   for i in range(n)]
            rel = mp.sqrt(sum((t * t for t in res), mpf(0))) / ny
            traj.append([it, float(rel)])
            if rel < best:
                best = rel
    return float(best), traj


def job_fn(job):
    A, y, s = build(job["spectrum"], job["kappa"])
    iters = ITER_MULT * M
    t0 = time.time()
    best, traj = cg_at_dps(A, y, job["dps"], iters,
                           mixed=job.get("mixed", "all"))
    return {"spectrum": job["spectrum"], "kappa": job["kappa"],
            "dps": job["dps"], "mixed": job.get("mixed", "all"),
            "best": best, "traj": traj,
            "iters": iters, "wall_s": round(time.time() - t0, 1)}


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5), squeeze=False)
    kcol = {1e4: "#1f77b4", 1e6: "#ff7f0e", 1e10: "#d62728"}
    FP64 = 1.1e-16

    for i, sp in enumerate(SPECTRA):
        ax = axes[0][i]
        for kap in KAPPAS:
            sel = sorted([r for r in rows if r["spectrum"] == sp
                          and r["kappa"] == kap and r["mixed"] == "all"],
                         key=lambda r: r["dps"])
            if sel:
                ax.plot([r["dps"] for r in sel],
                        [max(r["best"], 1e-18) for r in sel], "o-",
                        color=kcol[kap], lw=1.8, ms=5.5,
                        label=rf"$\kappa=10^{{{int(math.log10(kap))}}}$")
            mv = sorted([r for r in rows if r["spectrum"] == sp
                         and r["kappa"] == kap and r["mixed"] == "mv_fp64"],
                        key=lambda r: r["dps"])
            if mv:
                ax.plot([r["dps"] for r in mv],
                        [max(r["best"], 1e-18) for r in mv], "s--",
                        color=kcol[kap], lw=1.2, ms=4.5, alpha=0.75)
        ax.axhline(FP64, color="k", ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 10)
        ax.set_xlim(12, 64)
        ax.grid(alpha=.25, which="both", lw=.4)
        ax.set_xlabel("working precision (decimal digits)")
        ax.set_title(f"{sp} spectrum", fontsize=10, pad=30)
        if i == 0:
            ax.set_ylabel("best rel $L_2$  (memoryless, <10 vectors)")

    ax = axes[0][2]
    sel = sorted([r for r in rows if r["spectrum"] == "phispec"
                  and r["kappa"] == 1e10 and r["mixed"] == "all"],
                 key=lambda r: r["dps"])
    cm = plt.cm.viridis(np.linspace(0.05, 0.9, len(sel)))
    for r, c in zip(sel, cm):
        t = np.array(r["traj"])
        if len(t):
            ax.plot(t[:, 0], np.minimum.accumulate(t[:, 1]), lw=1.6,
                    color=c, label=f"{r['dps']} digits")
    ax.axhline(FP64, color="k", ls=":", lw=1.2)
    ax.axvline(M, color="#999999", ls="--", lw=0.9)
    ax.text(M * 1.07, 2e-2, "$m$", fontsize=9, color="#666666")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 10)
    ax.grid(alpha=.25, which="both", lw=.4)
    ax.set_xlabel("CG iteration")
    ax.set_title(r"trajectories, phispec $\kappa=10^{10}$", fontsize=10, pad=30)

    for a in axes[0]:
        a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
                 fontsize=8, frameon=False, borderaxespad=0)
    fig.text(0.335, 0.015,
             "solid = all quantities at stated precision   |   "
             "dashed = matvec result rounded to fp64 (the deployable question)   "
             "|   dotted = fp64 unit roundoff", fontsize=8, color="#444444")
    fig.suptitle("The precision law: the memoryless core reaches the floor with "
                 "ZERO memory, given enough digits in the OPERATOR", y=1.0,
                 fontsize=11)
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    jobs = [{"spectrum": sp, "kappa": k, "dps": d, "mixed": mx}
            for sp in SPECTRA for k in KAPPAS for d in DPS
            for mx in ("all", "mv_fp64")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    print(f"{len(jobs)} runs (m={M}, R={R}, {ITER_MULT}m iterations each)")
    with ProcessPoolExecutor(max_workers=6) as ex:
        for r in ex.map(job_fn, jobs):
            rows.append(r)
            print(f"  {r['spectrum']:<8s} kappa=1e{int(math.log10(r['kappa'])):<3d}"
                  f" dps={r['dps']:<3d} {r['mixed']:<8s} -> best {r['best']:.3e}"
                  f"   ({r['wall_s']}s)", flush=True)
    (OUT_DIR / "precision_law_rows.json").write_text(json.dumps(rows))
    figure(rows, OUT_DIR / "precision_law.png")

    print("\nstall vs precision (decades gained per extra decimal digit):")
    for sp in SPECTRA:
        for kap in KAPPAS:
            sel = sorted([r for r in rows if r["spectrum"] == sp
                          and r["kappa"] == kap and r["mixed"] == "all"],
                         key=lambda r: r["dps"])
            if len(sel) < 2:
                continue
            lo, hi = sel[0], sel[-1]
            if hi["best"] <= 0 or lo["best"] <= 0:
                continue
            slope = (math.log10(lo["best"]) - math.log10(hi["best"])) \
                / (hi["dps"] - lo["dps"])
            print(f"  {sp:<8s} kappa=1e{int(math.log10(kap)):<3d}: "
                  f"dps {lo['dps']}->{hi['dps']}  "
                  f"{lo['best']:.1e} -> {hi['best']:.1e}   "
                  f"slope {slope:.2f} decades/digit")
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT_DIR}/precision_law.png")


if __name__ == "__main__":
    main()
