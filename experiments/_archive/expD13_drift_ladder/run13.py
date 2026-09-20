"""expD13 battery -- the mu ladder on a drifting Phi.

Arms
  ladder   alpha descends; at each level the geometry is perturbed by
           eta = nmult*alpha (fresh draw), the damped solve runs at c=0 warm
           started from the previous level, and the error is measured on the
           SAME perturbed geometry (that is the model's actual error).
           Terminal solve then runs on the TRUE geometry, full reorth.
  nmult    0 (static control) / 0.1 / 1 (matched) / 10 -- does mu want to track
           the geometry noise?
  batch    the matched ladder with a fresh random batch each step
  noise    the matched ladder with noisy y

Figures re-render after every cell.
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
import core13 as c
import figs13

OUT = c.RESULTS / "ladder.jsonl"
BOUT = c.RESULTS / "batch.jsonl"
T0 = time.time()

ALPHAS = [10.0 ** (-e / 2) for e in range(2, 25)]     # 1e-1 .. 1e-12
CAP = 4096
TOL = 1.3


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


PROBS = [
    ("qi1d_256",    lambda: c.QI1D(256)),
    ("qi1d_1024",   lambda: c.QI1D(1024)),
    ("qi2d_256",    lambda: c.QI2D(256)),
    ("qi2d_2048",   lambda: c.QI2D(2048)),
    ("twin_512",    lambda: c.Twin(512,  label="random gapless  d=512")),
    ("twin_2048",   lambda: c.Twin(2048, label="random gapless  d=2048")),
    ("twin_spec2d", lambda: _spec2d()),
]
SWEEP_KEYS = ["qi1d_256", "qi2d_2048", "twin_2048"]


def _spec2d():
    q = c.QI2D(2048)
    p = q.at(0.0)
    s = np.linalg.svd(p["A"], compute_uv=False)
    r = int((s > c.RCOND * s[0]).sum())
    d = p["d"]
    del p, q
    gc.collect()
    return c.Twin(d, spectrum=s[:r].copy(), row_mult=6,
                  label="random, 2-D spectrum  d=2060")


def run_ladder(dp, key, nmult, tag="ladder", seed0=0):
    """Descend the ladder with the geometry converging as eta = nmult*alpha."""
    p0 = dp.at(0.0)                                    # true geometry
    fl_true, r_true, s1 = c.pool_floor(p0)
    d = p0["d"]
    log(f"--- {dp.label} nmult={nmult}: d={d} n={p0['n']} r={r_true} "
        f"s1={s1:.3e} true-floor={fl_true:.2e}")
    w = np.zeros(d)
    cum = 0
    prev_dop = np.inf
    for lv, a in enumerate(ALPHAS):
        eta = nmult * a
        p = p0 if eta == 0 else dp.at(eta, seed=seed0 + 7 * lv)
        U, s, Vt, r = c.spectrum(p)
        mu = float((a * s[0]) ** 2)
        dop = c.eval_err(p, c.damped_exact(U, s, Vt, p["y"], mu))
        if dop > prev_dop * 1.05 and lv > 3:
            log(f"   alpha={a:.1e} damped optimum turned up "
                f"({prev_dop:.2e}->{dop:.2e}); stopping ladder")
            break
        prev_dop = min(prev_dop, dop)
        pr = lambda z, w=w, p=p: c.eval_err(p, w + z)
        x, hist = c.lsqr_damped(p["A"], p["y"] - p["A"] @ w, mu, d, CAP, 0,
                                probe=pr, probe_at=c.probe_sched(CAP),
                                stop_below=TOL * dop)
        if not hist:
            continue
        it, reached = hist[-1][0], hist[-1][1]
        w = w + x
        cum += it
        err_true = c.eval_err(p0, w)
        c.append(OUT, dict(arm=tag, key=key, label=dp.label, nmult=nmult, d=d,
                           n=p0["n"], rank=r_true, sigma1=s1,
                           floor_true=fl_true, noise=dp.noise_rel,
                           alpha=a, eta=eta, mu=mu, damped_opt=dop,
                           iters=it, cum_before=cum - it, reached=reached,
                           err_true=err_true,
                           converged=bool(reached <= TOL * dop),
                           traj=[[int(i), float(e)] for i, e, _ in hist]))
        log(f"   a={a:.1e} eta={eta:.1e} dop={dop:.2e} it={it:<5d} "
            f"reached={reached:.2e} (true-geom err {err_true:.2e}) cum={cum}")
        del p, U, s, Vt
        gc.collect()
        if not (reached <= TOL * dop):
            log(f"   -> c=0 gave out; HANDOFF at alpha={a:.1e}")
            break
    # terminal solve on the TRUE geometry
    t = time.time()
    pr = lambda z, w=w: c.eval_err(p0, w + z)
    mi = min(2 * r_true, 4 * r_true)
    x, hist = c.lsqr_damped(p0["A"], p0["y"] - p0["A"] @ w, 0.0, d, mi, -1,
                            probe=pr, probe_at=c.probe_sched(mi))
    w = w + x
    fin = c.eval_err(p0, w)
    c.append(OUT, dict(arm="terminal", key=key, label=dp.label, nmult=nmult,
                       d=d, n=p0["n"], rank=r_true, sigma1=s1,
                       floor_true=fl_true, noise=dp.noise_rel, alpha=0.0,
                       eta=0.0, mu=0.0, damped_opt=fl_true,
                       iters=(hist[-1][0] if hist else 0), cum_before=cum,
                       reached=fin, err_true=fin, converged=True,
                       secs=time.time() - t,
                       traj=[[int(i), float(e)] for i, e, _ in hist]))
    log(f"   TERMINAL (true geom): {hist[-1][0] if hist else 0} its, "
        f"{time.time()-t:.1f}s -> {fin:.3e}  (true floor {fl_true:.2e})")
    del p0
    gc.collect()


def run_batched(dp, key, bmults, cap_steps=600, inner=8):
    """Matched ladder (nmult=1) with a fresh batch every GN step."""
    p0 = dp.at(0.0)
    fl_true, r_true, s1 = c.pool_floor(p0)
    d, n = p0["d"], p0["n"]
    alphas = ALPHAS[::2]
    for bm in bmults:
        b = max(int(bm * d), 4)
        f1 = c.floor_1batch(p0, b)
        rng = np.random.default_rng(4242)
        w = np.zeros(d)
        best = np.inf
        for lv, a in enumerate(alphas):
            p = dp.at(a, seed=11 * lv)
            mu = float((a * s1) ** 2)
            for _ in range(cap_steps // len(alphas)):
                S = rng.choice(n, min(b, n), replace=False)
                Ab = p["A"][S]
                x, _ = c.lsqr_damped(Ab, p["y"][S] - Ab @ w, mu, d, inner, 0)
                w = w + x
                best = min(best, c.eval_err(p0, w))
            del p
            gc.collect()
        c.append(BOUT, dict(key=key, label=dp.label, d=d, n=n, bmult=bm, b=b,
                            inner=inner, eval=best, floor_1batch=f1,
                            floor_pool=fl_true))
        log(f"   batch b/d={bm:<8.4g} b={b:<6d} best={best:.2e}  "
            f"(1-batch {f1:.1e}, pool {fl_true:.1e})")
    del p0
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-batch", action="store_true")
    a = ap.parse_args()
    c.FIGS.mkdir(parents=True, exist_ok=True)
    done = {(q["key"], q["arm"], q["nmult"], q.get("noise", 0.0))
            for q in c.load(OUT)}

    # main arm: matched noise, every problem
    for key, mk in PROBS:
        if a.only and a.only not in key:
            continue
        if (key, "terminal", 1.0, 0.0) in done:
            log(f"skip {key} nmult=1 (done)")
            continue
        dp = mk()
        run_ladder(dp, key, 1.0)
        figs13.render_all()
        del dp
        gc.collect()

    # matching sweep on three problems
    for key, mk in PROBS:
        if key not in SWEEP_KEYS or (a.only and a.only not in key):
            continue
        for nm in (0.0, 0.1, 10.0):
            if (key, "terminal", nm, 0.0) in done:
                continue
            dp = mk()
            run_ladder(dp, key, nm)
            figs13.render_all()
            del dp
            gc.collect()

    # noisy y, matched
    for key, mkf in (("qi2d_2048", lambda s: c.QI2D(2048, noise_rel=s)),
                     ("twin_2048", lambda s: c.Twin(2048, noise_rel=s,
                                                    label="random gapless  d=2048"))):
        if a.only and a.only not in key:
            continue
        for s in (1e-8, 1e-4, 1e-2):
            if (key, "terminal", 1.0, s) in done:
                continue
            dp = mkf(s)
            log(f"=== NOISE sigma={s:.0e} on {dp.label}")
            run_ladder(dp, key, 1.0)
            figs13.render_all()
            del dp
            gc.collect()

    if not a.skip_batch:
        bm = [8.0, 1.0, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64]
        for key, mk in (("qi2d_2048", lambda: c.QI2D(2048)),
                        ("twin_spec2d", _spec2d)):
            if a.only and a.only not in key:
                continue
            dp = mk()
            log(f"=== BATCHING (drifting) on {dp.label}")
            run_batched(dp, key, bm)
            figs13.render_all()
            del dp
            gc.collect()
    figs13.render_all()
    log("ALL DONE")


if __name__ == "__main__":
    main()
