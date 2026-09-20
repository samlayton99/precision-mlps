"""expD11 driver. Runs the batching battery in escalating-d order and
re-renders every figure after each completed cell, so progress is watchable.

    uv run python experiments/expD11_batching/run11.py            # everything
    uv run python experiments/expD11_batching/run11.py --only design,sweep
    uv run python experiments/expD11_batching/run11.py --max-d 2048

Resumable: cells already present in the JSONL are skipped.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core11 as c
import figs11

SWEEP = c.RESULTS / "sweep.jsonl"
EXTRA = c.RESULTS / "extra.jsonl"

# batch fractions: 8d down to d/64
BMULTS = [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64]
TOTAL_ITERS = 3072          # constant solver budget for Type A everywhere
K_BLOCK = 64
TAU_BUDGET = 128
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def cells(max_d):
    """(builder, kwargs) in escalating d order so figures populate early."""
    out = []
    for N in (64, 128, 256, 512, 1024):
        out.append(("qi1d", dict(N=N, row_mult=8)))
    for N in (256, 576, 1024):
        out.append(("qi2d", dict(N=N, geom="radon_tensor", lam=0.12, row_mult=8)))
    for d in (512, 1024, 2048, 5000):
        out.append(("twin_cluster", dict(d=d, kind="cluster", row_mult=4)))
        out.append(("twin_unstruct", dict(d=d, kind="unstruct", row_mult=4)))
    built = []
    for fam, kw in out:
        approx = (kw.get("d") or kw.get("N", 0) * 2)
        if approx <= max_d:
            built.append((fam, kw))
    return built


def build(fam, kw):
    if fam == "qi1d":
        return c.prob_qi1d(**kw)
    if fam == "qi2d":
        return c.prob_qi2d(**kw)
    return c.prob_twin(**kw)


def done_keys(path, tag):
    return {(r["family"], r["d"], r["method"], round(r["bmult"], 6))
            for r in c.load(path) if r.get("tag") == tag}


# ------------------------------------------------------------ design arm ---
def arm_design():
    """F7: the three Type A design decisions, on one small cell each."""
    if any(r.get("tag") == "tau" for r in c.load(EXTRA)):
        log("design arm already present, skipping")
        return
    from scipy.sparse.linalg import lsmr, LinearOperator
    p = c.prob_qi1d(128, row_mult=4)
    d, A, y = p["d"], p["A"], p["y"]
    blocks = [np.arange(t, min(t + 64, d)) for t in range(0, d, 64)]

    def mk(rc):
        fac = []
        for C in blocks:
            ev, V = np.linalg.eigh(A[:, C].T @ A[:, C])
            kp = ev > rc * max(ev.max(), 1e-300)
            fac.append((C, V, np.where(kp, np.where(kp, ev, 1.0) ** -0.5, 0.0)))

        def Mh(v):
            o = np.zeros_like(v)
            for C, V, ih in fac:
                o[C] = V @ (ih * (V.T @ v[C]))
            return o
        return Mh

    # (a) truncation sweep
    for rc in (1e-16, 1e-14, 1e-12, 1e-10, 1e-8):
        Mh = mk(rc)
        B = LinearOperator((A.shape[0], d), matvec=lambda v: A @ Mh(v),
                           rmatvec=lambda u: Mh(A.T @ u), dtype=float)
        e = min(c.eval_err(p, Mh(lsmr(B, y, atol=0, btol=0, conlim=0,
                                     maxiter=it)[0]))
                for it in (200, 800, 1600, 3000))
        c.append(EXTRA, dict(tag="tau", kind="trunc", rcond=rc, eval=e))
    log("design: truncation sweep done")

    # (b) restart penalty on a FIXED operator, constant total iterations
    Mh = mk(1e-12)
    B = LinearOperator((A.shape[0], d), matvec=lambda v: A @ Mh(v),
                       rmatvec=lambda u: Mh(A.T @ u), dtype=float)
    for tau in (2, 8, 32, 128, 512):
        w = np.zeros(d)
        for _ in range(3072 // tau):
            w = w + Mh(lsmr(B, y - A @ w, atol=0, btol=0, conlim=0,
                            maxiter=tau)[0])
        c.append(EXTRA, dict(tag="tau", kind="fixed_op", tau=tau,
                             eval=c.eval_err(p, w)))
    w = Mh(lsmr(B, y, atol=0, btol=0, conlim=0, maxiter=3072)[0])
    c.append(EXTRA, dict(tag="tau", kind="unrestarted", tau=3072,
                         eval=c.eval_err(p, w)))
    log("design: restart penalty done")

    # (c) the clamp, across batch sizes
    p8 = c.prob_qi1d(128, row_mult=8)
    for bm in (8.0, 1.0, 0.25, 1 / 16, 1 / 64):
        b = max(int(bm * p8["d"]), 2)
        _, i_cl = c.type_a(p8, K_BLOCK, b, TAU_BUDGET, max(1, TOTAL_ITERS //
                                                           max(2, min(TAU_BUDGET, b // 2))),
                           rec_every=10 ** 9)
        # unclamped: replicate by forcing tau through a shim
        tau_un = TAU_BUDGET
        _, i_pl = c.type_a(p8, K_BLOCK, max(b, 2 * tau_un), tau_un,
                           TOTAL_ITERS // tau_un, rec_every=10 ** 9) \
            if False else (None, None)
        c.append(EXTRA, dict(tag="tau", kind="clamp", bmult=bm, b=b,
                             clamped=i_cl["best"],
                             plain=_unclamped(p8, b, tau_un),
                             floor_1batch=c.floor_1batch(p8, b)))
    log("design: clamp sweep done")
    figs11.render_all()


def _unclamped(prob, b, tau):
    """Type A with the clamp deliberately disabled (for F7 only)."""
    from scipy.sparse.linalg import lsmr, LinearOperator
    A, y = prob["A"], prob["y"]
    n, d = A.shape
    blocks = c.make_blocks(prob, K_BLOCK, seed=0)
    P = c.BlockPrec(blocks, d, variant="ema", beta=0.99, damp_rel=1e-12, seed=0)
    rng = np.random.default_rng(1234)
    for _ in range(len(blocks)):
        P.update(A[rng.choice(n, min(b, n), replace=False)])
    w = np.zeros(d)
    best = np.inf
    for _ in range(max(1, TOTAL_ITERS // tau)):
        S = np.arange(n) if b >= n else rng.choice(n, b, replace=False)
        Ab, yb = A[S], y[S]
        P.update(Ab)
        op = LinearOperator((len(S), d), matvec=lambda z: Ab @ P.half(z),
                            rmatvec=lambda u: P.half(Ab.T @ u), dtype=float)
        w = w + P.half(lsmr(op, yb - Ab @ w, atol=0, btol=0, conlim=0,
                            maxiter=tau)[0])
        best = min(best, c.eval_err(prob, w))
    return float(best)


# ------------------------------------------------------------- main sweep --
def arm_sweep(max_d):
    have = done_keys(SWEEP, "sweep")
    for fam, kw in cells(max_d):
        t0 = time.time()
        p = build(fam, kw)
        base = dict(tag="sweep", family=p["family"], d=p["d"], n=p["n"],
                    label=p["label"], rank=p["rank"], kappa=p["kappa"],
                    floor_pool=p["floor_pool"])
        log(f"{p['label']}: d={p['d']} n={p['n']} rank={p['rank']} "
            f"kap={p['kappa']:.1e} floor={p['floor_pool']:.2e} ({time.time()-t0:.0f}s build)")
        # Type B batch list is trimmed at large d (b-independence already
        # established at small d; the sketch cost is the expensive part)
        b_list_B = BMULTS if p["d"] <= 1100 else [8.0, 1.0, 0.125, 1 / 64]
        for bm in BMULTS:
            b = max(int(bm * p["d"]), 2)
            f1 = c.floor_1batch(p, b)
            if (p["family"], p["d"], "A", round(bm, 6)) not in have:
                tau_eff = max(2, min(TAU_BUDGET, b // 2))
                _, ia = c.type_a(p, K_BLOCK, b, TAU_BUDGET,
                                 max(1, TOTAL_ITERS // tau_eff),
                                 rec_every=max(1, (TOTAL_ITERS // tau_eff) // 8))
                c.append(SWEEP, dict(**base, method="A", bmult=bm, b=b,
                                     floor_1batch=f1, eval=ia["best"],
                                     passes=ia["passes"], tau=ia["tau_used"],
                                     prec_frac=ia["prec_frac"]))
                log(f"   A b/d={bm:<7.4g} b={b:<6d} tau={ia['tau_used']:<4d} "
                    f"eval={ia['best']:.2e}  (1batch {f1:.1e})")
            if bm in b_list_B and (p["family"], p["d"], "B", round(bm, 6)) not in have:
                _, ib = c.type_b(p, b=b, n_acc_mult=min(4.0, p["n"] / p["d"]))
                c.append(SWEEP, dict(**base, method="B", bmult=bm, b=b,
                                     floor_1batch=f1, eval=ib["eval"],
                                     passes=ib["passes"], n_acc=ib["n_acc"],
                                     kappa_AP=ib["kappa_AP"],
                                     reverted=ib["reverted"]))
                log(f"   B b/d={bm:<7.4g} b={b:<6d} eval={ib['eval']:.2e} "
                    f"n_acc={ib['n_acc']} kap(AP)={ib['kappa_AP']:.1f}")
            figs11.render_all()
        del p
        gc.collect()


# ------------------------------------------------------ accumulation arm ---
def arm_accum(max_d):
    have = {(r["family"], r["d"], r["bmult"], r["acc_mult"])
            for r in c.load(EXTRA) if r.get("tag") == "accum"}
    for fam, kw in [("qi1d", dict(N=256, row_mult=8)),
                    ("qi2d", dict(N=576, geom="radon_tensor", lam=0.12, row_mult=8)),
                    ("twin_cluster", dict(d=1024, kind="cluster", row_mult=4))]:
        if (kw.get("d") or kw.get("N", 0) * 2) > max_d:
            continue
        p = build(fam, kw)
        base = dict(tag="accum", family=p["family"], d=p["d"], label=p["label"],
                    floor_pool=p["floor_pool"])
        for bm in (1.0, 0.125, 1 / 64):
            b = max(int(bm * p["d"]), 2)
            for am in (0.5, 1.0, 2.0, 4.0, 8.0):
                if am * p["d"] > p["n"] or (p["family"], p["d"], bm, am) in have:
                    continue
                _, ib = c.type_b(p, b=b, n_acc_mult=am)
                c.append(EXTRA, dict(**base, bmult=bm, b=b, acc_mult=am,
                                     eval=ib["eval"], n_acc=ib["n_acc"],
                                     kappa_AP=ib["kappa_AP"],
                                     reverted=ib["reverted"]))
            log(f"accum {p['label']} b/d={bm:.4g} done")
            figs11.render_all()
        del p
        gc.collect()


# ------------------------------------------------------------- noise arm ---
def arm_noise(max_d):
    have = {(r["family"], r["d"], r["method"], r["bmult"], r["noise_rel"])
            for r in c.load(EXTRA) if r.get("tag") == "noise"}
    specs = [("qi1d", dict(N=256, row_mult=8)),
             ("qi2d", dict(N=576, geom="radon_tensor", lam=0.12, row_mult=8)),
             ("twin_cluster", dict(d=1024, kind="cluster", row_mult=4)),
             ("twin_unstruct", dict(d=1024, kind="unstruct", row_mult=4))]
    for fam, kw in specs:
        if (kw.get("d") or kw.get("N", 0) * 2) > max_d:
            continue
        for nz in (1e-6, 1e-4, 1e-2):
            p = build(fam, dict(kw, noise_rel=nz))
            base = dict(tag="noise", family=p["family"], d=p["d"],
                        label=p["label"].split(" noise")[0], noise_rel=nz,
                        floor_pool=p["floor_pool"], rank=p["rank"])
            for bm in (8.0, 1.0, 0.125, 1 / 64):
                b = max(int(bm * p["d"]), 2)
                tau_eff = max(2, min(TAU_BUDGET, b // 2))
                if (p["family"], p["d"], "A", bm, nz) not in have:
                    _, ia = c.type_a(p, K_BLOCK, b, TAU_BUDGET,
                                     max(1, TOTAL_ITERS // tau_eff),
                                     rec_every=10 ** 9)
                    c.append(EXTRA, dict(**base, method="A", bmult=bm, b=b,
                                         eval=ia["best"]))
                if (p["family"], p["d"], "B", bm, nz) not in have:
                    _, ib = c.type_b(p, b=b, n_acc_mult=min(4.0, p["n"] / p["d"]),
                                     noise_rcond=None)
                    c.append(EXTRA, dict(**base, method="B", bmult=bm, b=b,
                                         eval=ib["eval"], n_acc=ib["n_acc"]))
            log(f"noise {p['label']} sigma={nz:.0e} done")
            figs11.render_all()
            del p
            gc.collect()


# ------------------------------------------------------- drift/guard arm ---
def arm_drift(max_d):
    have_d = {(r["family"], r["d"], r["eta"]) for r in c.load(EXTRA)
              if r.get("tag") == "drift"}
    have_g = {(r["family"], r["d"], r["eta"]) for r in c.load(EXTRA)
              if r.get("tag") == "guard"}
    for fam, kw in [("qi1d", dict(N=256, row_mult=8)),
                    ("twin_cluster", dict(d=1024, kind="cluster", row_mult=4))]:
        if (kw.get("d") or kw.get("N", 0) * 2) > max_d:
            continue
        p = build(fam, kw)
        b = max(p["d"] // 8, 2)
        tau_eff = max(2, min(TAU_BUDGET, b // 2))
        for eta in (0.0, 1e-8, 1e-6, 1e-4, 1e-3):
            if (p["family"], p["d"], eta) not in have_d:
                _, ia = c.type_a(p, K_BLOCK, b, TAU_BUDGET,
                                 max(1, TOTAL_ITERS // tau_eff),
                                 drift=(lambda t, e=eta: e), rec_every=10 ** 9)
                c.append(EXTRA, dict(tag="drift", family=p["family"], d=p["d"],
                                     label=p["label"], eta=eta, b=b,
                                     eval=ia["best"], floor_pool=p["floor_pool"]))
            if (p["family"], p["d"], eta) not in have_g:
                w0 = np.zeros(p["d"])
                # a realistic entry point: a partially converged Type A result
                w0, _ = c.type_a(p, K_BLOCK, b, TAU_BUDGET, 6, rec_every=10 ** 9)
                e_in = c.eval_err(p, w0)
                wo, ib = c.type_b(p, b=b, n_acc_mult=min(4.0, p["n"] / p["d"]),
                                  w0=w0, drift_eta=eta)
                c.append(EXTRA, dict(tag="guard", family=p["family"], d=p["d"],
                                     label=p["label"], eta=eta,
                                     eval_in=e_in, eval_out=c.eval_err(p, wo),
                                     reverted=ib["reverted"]))
        log(f"drift/guard {p['label']} done")
        figs11.render_all()
        del p
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="design,sweep,accum,noise,drift")
    ap.add_argument("--max-d", type=int, default=5200)
    a = ap.parse_args()
    which = set(a.only.split(","))
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    c.FIGS.mkdir(parents=True, exist_ok=True)
    if "design" in which:
        arm_design()
    if "sweep" in which:
        arm_sweep(a.max_d)
    if "accum" in which:
        arm_accum(a.max_d)
    if "noise" in which:
        arm_noise(a.max_d)
    if "drift" in which:
        arm_drift(a.max_d)
    figs11.render_all()
    log("ALL DONE")


if __name__ == "__main__":
    main()
