"""expD11 -- Sam's design, measured across problems, widths, batches, noise.

THE DESIGN (an "episode"):
  freeze the operator -> accumulate rows from the batch stream until n_acc
  -> build a block-Jacobi factor at O(d*k) state from those rows
  -> ONE long UNRESTARTED LSMR run, then a few refinement rounds on the same
     frozen operator (fresh residual each round)
  -> stop on an observable; hand back

No Adam. No sketch. No O(d^2) object. Batch size only sets how many stream
steps the accumulation takes, never the solve.

Baselines per cell, so every number is interpretable:
  floor_pool      truncated SVD on the whole pool  (best possible)
  floor_1batch(b) truncated SVD on ONE batch of b  (no-accumulation reference)
  stream_restart  restarted LSMR tau=1 + same preconditioner (the streaming
                  alternative, same O(dk) state)

Figures re-render after every cell.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsmr, LinearOperator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core11 as c
import figs_frozen

OUT = c.RESULTS / "frozen.jsonl"
K = 64
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def bjac(Arows, d, k=K, rcond=1e-13):
    """Block-Jacobi M^{-1/2} from the block's own SVD (never the Gram).
    State: d*k floats. Applied as V diag V^T with V orthogonal, so the
    application is backward stable (measured 5.4e-16)."""
    facs = []
    for t in range(0, d, k):
        C = np.arange(t, min(t + k, d))
        _, s, Vt = np.linalg.svd(Arows[:, C], full_matrices=False)
        kp = s > rcond * max(s[0], 1e-300)
        facs.append((C, Vt.T, np.where(kp, 1.0 / np.where(kp, s, 1.0), 0.0)))

    def Mh(v):
        o = np.zeros(d)
        for C, V, ih in facs:
            o[C] = V @ (ih * (V.T @ v[C]))
        return o
    return Mh


def episode(prob, b, n_acc_mult, inner, rounds, k=K, seed=0, drift_eta=0.0):
    """One frozen-operator episode. Returns (w_best, info)."""
    A, y = prob["A"], prob["y"]
    n, d = A.shape
    rng = np.random.default_rng(500 + seed)
    Aeff = A if drift_eta == 0 else A * (
        1.0 + drift_eta * rng.standard_normal(d)[None, :])
    # accumulate rows from the stream, b at a time
    tgt = int(min(n, np.ceil(n_acc_mult * d)))
    idx, steps = [], 0
    while len(idx) < tgt:
        idx.extend(rng.choice(n, min(b, n), replace=False).tolist())
        steps += 1
    idx = np.array(idx[:tgt])
    Aa, ya = Aeff[idx], y[idx]
    Mh = bjac(Aa, d, k)
    op = LinearOperator((len(idx), d), matvec=lambda z: Aa @ Mh(z),
                        rmatvec=lambda u: Mh(Aa.T @ u), dtype=float)
    w = np.zeros(d)
    hist = []
    best = (np.inf, w.copy(), 0)
    passes = len(idx) / max(b, 1)
    for rd in range(1, rounds + 1):
        r = ya - Aa @ w
        res = lsmr(op, r, atol=0.0, btol=0.0, conlim=0, maxiter=inner)
        w = w + Mh(res[0])
        passes += 2 * int(res[2]) * len(idx) / max(b, 1)
        obs = float(np.linalg.norm(Aa.T @ (ya - Aa @ w)))
        ev = c.eval_err(prob, w)
        hist.append(dict(round=rd, obs=obs, eval=ev, iters=int(res[2]),
                         passes=passes, wnorm=float(np.linalg.norm(w))))
        if ev < best[0]:
            best = (ev, w.copy(), rd)
    return best[1], dict(history=hist, best=best[0], best_round=best[2],
                         n_acc=int(len(idx)), acc_steps=steps, passes=passes)


def stream_restart(prob, b, inner_total, k=K, seed=0):
    """The streaming alternative at the SAME O(dk) state: restarted tau=1."""
    A, y = prob["A"], prob["y"]
    n, d = A.shape
    rng = np.random.default_rng(900 + seed)
    Mh = bjac(A[rng.choice(n, min(n, 4 * d), replace=False)], d, k)
    w = np.zeros(d)
    best = np.inf
    for _ in range(inner_total):
        S = rng.choice(n, min(b, n), replace=False)
        Ar, yr = A[S], y[S]
        o = LinearOperator((len(S), d), matvec=lambda z: Ar @ Mh(z),
                           rmatvec=lambda u: Mh(Ar.T @ u), dtype=float)
        w = w + Mh(lsmr(o, yr - Ar @ w, atol=0, btol=0, conlim=0, maxiter=1)[0])
        best = min(best, c.eval_err(prob, w))
    return best


CELLS = (
    [("qi1d", dict(N=N, row_mult=8)) for N in (64, 128, 256, 512)]
    + [("qi2d", dict(N=N, geom="radon_tensor", lam=0.12, row_mult=8))
       for N in (256, 576)]
    + [("twin_cluster", dict(d=D, kind="cluster", row_mult=4)) for D in (512, 1024)]
    + [("twin_unstruct", dict(d=D, kind="unstruct", row_mult=4)) for D in (512, 1024)]
    + [("qi1d", dict(N=1024, row_mult=8))]
    + [("twin_cluster", dict(d=2048, kind="cluster", row_mult=4)),
       ("twin_unstruct", dict(d=2048, kind="unstruct", row_mult=4))]
)
BMULTS = [8.0, 1.0, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64]


def build(fam, kw):
    return (c.prob_qi1d(**kw) if fam == "qi1d" else
            c.prob_qi2d(**kw) if fam == "qi2d" else c.prob_twin(**kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-d", type=int, default=2200)
    a = ap.parse_args()
    have = {(r["family"], r["d"], r["arm"], round(r.get("bmult", -1), 6),
             r.get("n_acc_mult", -1), r.get("noise", 0.0))
            for r in c.load(OUT)}
    c.FIGS.mkdir(parents=True, exist_ok=True)
    for fam, kw in CELLS:
        if (kw.get("d") or kw.get("N", 0) * 2) > a.max_d:
            continue
        for nz in (0.0, 1e-6, 1e-2):
            kw2 = dict(kw)
            if nz:
                kw2["noise_rel"] = nz
            p = build(fam, kw2)
            base = dict(family=p["family"], d=p["d"], n=p["n"], label=p["label"],
                        rank=p["rank"], kappa=p["kappa"], noise=nz,
                        floor_pool=p["floor_pool"])
            log(f"{p['label']} sigma={nz:.0e}: d={p['d']} rank={p['rank']} "
                f"kap={p['kappa']:.1e} floor={p['floor_pool']:.2e}")
            for bm in BMULTS:
                b = max(int(bm * p["d"]), 2)
                key = (p["family"], p["d"], "episode", round(bm, 6), 4.0, nz)
                f1 = c.floor_1batch(p, b)
                if key not in have:
                    _, i4 = episode(p, b, 4.0, 4000, 6)
                    c.append(OUT, dict(**base, arm="episode", bmult=bm, b=b,
                                       n_acc_mult=4.0, floor_1batch=f1,
                                       eval=i4["best"], n_acc=i4["n_acc"],
                                       acc_steps=i4["acc_steps"],
                                       passes=i4["passes"],
                                       best_round=i4["best_round"],
                                       hist=i4["history"]))
                    log(f"   episode b/d={bm:<8.4g} b={b:<6d} eval={i4['best']:.2e} "
                        f"(1batch {f1:.1e}) acc_steps={i4['acc_steps']}")
                key2 = (p["family"], p["d"], "stream", round(bm, 6), -1, nz)
                if key2 not in have and nz == 0.0 and bm in (1.0, 0.125, 1 / 64):
                    sv = stream_restart(p, b, 4000)
                    c.append(OUT, dict(**base, arm="stream", bmult=bm, b=b,
                                       floor_1batch=f1, eval=sv))
                figs_frozen.render_all()
            # accumulation curve at one batch size
            for am in (1.0, 2.0, 4.0, 8.0):
                if am * p["d"] > p["n"]:
                    continue
                key3 = (p["family"], p["d"], "accum", 0.125, am, nz)
                if key3 in have:
                    continue
                _, ia = episode(p, max(p["d"] // 8, 2), am, 4000, 6)
                c.append(OUT, dict(**base, arm="accum", bmult=0.125,
                                   b=max(p["d"] // 8, 2), n_acc_mult=am,
                                   eval=ia["best"], n_acc=ia["n_acc"],
                                   passes=ia["passes"]))
            figs_frozen.render_all()
            del p
            gc.collect()
    figs_frozen.render_all()
    log("ALL DONE")


if __name__ == "__main__":
    main()
