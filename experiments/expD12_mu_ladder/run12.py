"""expD12 battery -- the progressively-unregularized solve, characterised.

Arms
  ladder     alpha descends; each level solved at c=0, warm-started; then a
             terminal full-reorthogonalization solve from that point
  batch      the same ladder but each step sees a FRESH batch of b rows
             (soft win = beat floor_1batch(b); hard win = reach floor_pool)
  noise      the ladder on noisy y; should stop at the statistical floor

Figures re-render after every problem so a long run is inspectable live.
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
import core12 as k
import figs12

OUT = k.RESULTS / "ladder.jsonl"
BOUT = k.RESULTS / "batch.jsonl"
T0 = time.time()

ALPHAS = [10.0 ** (-e / 2) for e in range(2, 21)]          # 1e-1 .. 1e-10
CAP = 4096
TOL = 1.3


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


PROBS = [
    ("qi1d_256", lambda: k.prob_qi1d(256)),
    ("qi1d_1024", lambda: k.prob_qi1d(1024)),
    ("qi2d_256", lambda: k.prob_qi2d(256)),
    ("qi2d_2048", lambda: k.prob_qi2d(2048)),
    ("twin_512", lambda: k.prob_twin(512, label="random gapless  d=512")),
    ("twin_2048", lambda: k.prob_twin(2048, label="random gapless  d=2048")),
    ("twin_spec2d", lambda: _twin_like_2d()),
]


def _twin_like_2d():
    """Random matrix with the 2-D QI spectrum copied exactly -- same singular
    values, Haar singular vectors, no other structure."""
    p = k.prob_qi2d(2048)
    sig = p["sigma"][:p["rank"]].copy()
    d = p["d"]
    del p
    gc.collect()
    return k.prob_twin(d, spectrum=sig, row_mult=6,
                       label="random, 2-D spectrum  d=2060")


def run_ladder(p, key, cap=CAP, alphas=ALPHAS, tag="ladder"):
    d, r = p["d"], p["rank"]
    s1 = p["sigma1"]
    w = np.zeros(d)
    cum = 0
    log(f"--- {p['label']}: d={d} n={p['n']} r={r} s1={s1:.3e} "
        f"kappa={p['kappa']:.2e} floor={p['floor_pool']:.2e}")
    prev_dop = np.inf
    for a in alphas:
        mu = float((a * s1) ** 2)
        dop = k.eval_err(p, k.damped_exact(p, mu))
        # Noise-limited stop: with noisy y the damped optimum is U-shaped in
        # alpha (bias falls, variance rises). Once it turns upward, lowering mu
        # only fits noise, so the ladder is done.
        if dop > prev_dop:
            log(f"   alpha={a:.2e} damped optimum turned UP "
                f"({prev_dop:.2e} -> {dop:.2e}); noise-limited, stopping ladder")
            break
        prev_dop = dop
        rhs = p["y"] - p["A"] @ w
        pr = lambda z, w=w: k.eval_err(p, w + z)
        x, hist = k.lsqr_damped(p["A"], rhs, mu, d, cap, 0, probe=pr,
                                probe_at=k.probe_sched(cap),
                                stop_below=TOL * dop)
        if not hist:
            continue
        it_used = hist[-1][0]
        reached = hist[-1][1]
        k.append(OUT, dict(arm=tag, key=key, label=p["label"], d=d, n=p["n"],
                           rank=r, kappa=p["kappa"], sigma1=s1,
                           floor_pool=p["floor_pool"], noise=p["noise_rel"],
                           alpha=a, mu=mu, kappa_mu=1.0 / a,
                           damped_opt=dop, iters=it_used, cum_before=cum,
                           reached=reached, converged=bool(reached <= TOL * dop),
                           traj=[[int(i), float(e), float(o)] for i, e, o in hist]))
        w = w + x
        cum += it_used
        log(f"   alpha={a:.2e} kap_mu={1/a:.1e} dampopt={dop:.2e} "
            f"iters={it_used:<5d} reached={reached:.2e} cum={cum}")
        if not (reached <= TOL * dop):
            log(f"   -> c=0 cannot reach this level; HANDOFF at alpha={a:.1e}")
            break
    # terminal solve: full reorthogonalization from the ladder's endpoint
    t = time.time()
    rhs = p["y"] - p["A"] @ w
    pr = lambda z, w=w: k.eval_err(p, w + z)
    x, hist = k.lsqr_damped(p["A"], rhs, 0.0, d, min(2 * r, 4 * r), -1,
                            probe=pr, probe_at=k.probe_sched(min(2 * r, 4 * r)))
    w = w + x
    fin = k.eval_err(p, w)
    k.append(OUT, dict(arm="terminal", key=key, label=p["label"], d=d, n=p["n"],
                       rank=r, kappa=p["kappa"], sigma1=s1,
                       floor_pool=p["floor_pool"], noise=p["noise_rel"],
                       alpha=0.0, mu=0.0, kappa_mu=p["kappa"],
                       damped_opt=p["floor_pool"],
                       iters=(hist[-1][0] if hist else 0), cum_before=cum,
                       reached=fin, converged=True, secs=time.time() - t,
                       state_floats=int(d * r + p["n"] * r),
                       traj=[[int(i), float(e), float(o)] for i, e, o in hist]))
    log(f"   TERMINAL full-reorth: {hist[-1][0] if hist else 0} iters, "
        f"{time.time()-t:.1f}s -> {fin:.3e}  (pool floor {p['floor_pool']:.2e})")
    return w


def run_batched(p, key, bmults, cap_steps=600, inner=8, alphas=None):
    """Same ladder, but every GN step sees a fresh batch of b rows."""
    d, r, n, s1 = p["d"], p["rank"], p["n"], p["sigma1"]
    alphas = alphas or ALPHAS[::2]
    for bm in bmults:
        b = max(int(bm * d), 4)
        f1 = k.floor_1batch(p, b)
        rng = np.random.default_rng(4242)
        w = np.zeros(d)
        best = np.inf
        steps = 0
        for a in alphas:
            mu = float((a * s1) ** 2)
            for _ in range(cap_steps // len(alphas)):
                S = rng.choice(n, min(b, n), replace=False)
                Ab = p["A"][S]
                x, _ = k.lsqr_damped(Ab, p["y"][S] - Ab @ w, mu, d, inner, 0)
                w = w + x
                steps += 1
                best = min(best, k.eval_err(p, w))
        k.append(BOUT, dict(arm="batch", key=key, label=p["label"], d=d, n=n,
                            rank=r, noise=p["noise_rel"], bmult=bm, b=b,
                            steps=steps, inner=inner, eval=best,
                            floor_1batch=f1, floor_pool=p["floor_pool"]))
        log(f"   batch b/d={bm:<8.4g} b={b:<6d} best={best:.2e}  "
            f"(1-batch floor {f1:.1e}, pool floor {p['floor_pool']:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-batch", action="store_true")
    a = ap.parse_args()
    k.FIGS.mkdir(parents=True, exist_ok=True)
    done = {(q["key"], q["arm"], q.get("noise", 0.0)) for q in k.load(OUT)}
    for key, mk in PROBS:
        if a.only and a.only not in key:
            continue
        if (key, "terminal", 0.0) in done:
            log(f"skip {key} (done)")
            continue
        p = mk()
        run_ladder(p, key)
        figs12.render_all()
        del p
        gc.collect()
    # noise check, on one QI and one random matrix
    for key, mk, nz in (("qi2d_2048", lambda s: k.prob_qi2d(2048, noise_rel=s), None),
                        ("twin_2048", lambda s: k.prob_twin(2048, noise_rel=s,
                                                            label="random gapless  d=2048"), None)):
        if a.only and a.only not in key:
            continue
        for s in (1e-8, 1e-4, 1e-2):
            if (key, "ladder", s) in done:
                continue
            p = mk(s)
            log(f"=== NOISE sigma={s:.0e} on {p['label']}")
            run_ladder(p, key)
            figs12.render_all()
            del p
            gc.collect()
    if not a.skip_batch:
        bm = [8.0, 1.0, 0.25, 0.125, 1 / 16, 1 / 32, 1 / 64]
        for key, mk in (("qi2d_2048", lambda: k.prob_qi2d(2048)),
                        ("twin_spec2d", _twin_like_2d)):
            if a.only and a.only not in key:
                continue
            p = mk()
            log(f"=== BATCHING on {p['label']}")
            run_batched(p, key, bm)
            figs12.render_all()
            del p
            gc.collect()
    figs12.render_all()
    log("ALL DONE")


if __name__ == "__main__":
    main()
