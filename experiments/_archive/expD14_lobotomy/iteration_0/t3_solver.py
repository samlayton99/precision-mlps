"""T3 -- swap the reference solve for the O(d) one, on its own.

T2 ran the L block through a direct damped SVD, so the allocation logic could
be judged without the solver's limits in the way.  Here everything is held at
the T2 winner and only the solver changes:

    direct      damped SVD on a materialised J_L      (reference, O(nB))
    lsqr tau    tau iterations of matrix-free damped LSQR per step, restarted
                from scratch on a fresh residual, O(d) state, 2*tau passes

T1/P2 measured that a COLD c=0 LSQR cannot resolve below alpha ~ 1e-3 at any
iteration count (this is expD11's wall).  The question here is whether the
mu-ladder effect rescues it: theta carries the warm start, alpha descends only
as fast as the drift gauge allows, so each step's solve is a small increment on
an already-good iterate.  If that works, the deep solve is affordable at O(d).

Run: uv run --extra dev python experiments/expD14_lobotomy/iteration_0/t3_solver.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core0 as C  # noqa: E402

OUT = C.RESULTS / "t3_solver.jsonl"
BASE = dict(lset="readout", rho_mode="gain", clip=False, astep="linesearch")


def job(spec):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    env = C.build_case(spec["fn"], spec["N"], init=spec["init"])
    t0 = time.time()
    out = C.train_lobo(env, spec["iters"], lsolver=spec["lsolver"],
                       tau=spec["tau"], lr=1e-3, record=10,
                       alpha_mode=spec["alpha_mode"],
                       alpha_fixed=spec.get("alpha_fixed"), **BASE)
    rec = dict(spec)
    rec.update(floor0=env["floor0"], best_rel=out["best_rel"],
               final_rel=out["final_rel"], passes=out["passes"],
               floor_final=out["floor_final"],
               wall=round(time.time() - t0, 1), hist=out["hist"])
    return rec


def main():
    if OUT.exists():
        OUT.unlink()
    specs = []
    for init in ("qi", "rand"):
        for fn in ("sine", "runge"):
            specs.append(dict(fn=fn, N=128, init=init, iters=3000,
                              lsolver="direct", tau=0, arm="direct",
                              alpha_mode="fixed", alpha_fixed=1e-15))
            for tau in (1, 3, 10, 30, 100):
                specs.append(dict(fn=fn, N=128, init=init, iters=3000,
                                  lsolver="lsqr", tau=tau, arm=f"lsqr{tau}",
                                  alpha_mode="drift"))
            for a in (1e-4, 1e-8, 1e-15):
                specs.append(dict(fn=fn, N=128, init=init, iters=3000,
                                  lsolver="lsqr", tau=30, arm=f"lsqr30_a{a:g}",
                                  alpha_mode="fixed", alpha_fixed=a))
    print(f"{len(specs)} runs")
    recs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            recs.append(r)
            C.append(OUT, r)
            print(f"  [{i:2d}/{len(specs)}] {r['init']:5s} {r['fn']:6s} "
                  f"{r['arm']:8s} best {r['best_rel']:.3e} {r['wall']}s")
    print(f"total {time.time()-t0:.0f}s\n" + "=" * 74)
    for init in ("qi", "rand"):
        for fn in ("sine", "runge"):
            sel = [r for r in recs if r["init"] == init and r["fn"] == fn]
            f0 = sel[0]["floor0"]
            print(f"\n{init:5s} {fn:6s}  floor0 = {f0:.3e}")
            print(f"    {'arm':>14} {'best rel L2':>13} {'x floor0':>10} "
                  f"{'alpha_end':>10} {'passes':>10} {'lsqr_it avg':>12}")
            for a in ["direct"] + [f"lsqr{t}" for t in (1, 3, 10, 30, 100)] \
                    + [f"lsqr30_a{x:g}" for x in (1e-4, 1e-8, 1e-15)]:
                r = [x for x in sel if x["arm"] == a][0]
                h = r["hist"]
                print(f"    {a:>14} {r['best_rel']:13.3e} "
                      f"{r['best_rel']/f0:10.2f} {h['alpha'][-1]:10.2e} "
                      f"{r['passes']:10d} {np.mean(h['lsqr_it']):12.1f}")


if __name__ == "__main__":
    main()
