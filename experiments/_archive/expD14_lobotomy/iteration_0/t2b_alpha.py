"""T2b -- the damping rule, tested against the one thing T2 exposed: from a
random init the method lands on the floor of the geometry it STARTED with and
then stops, because once L is solved exactly the residual is orthogonal to the
features and A's learning signal dies (iteration 11, 9.4, now self-reinforcing:
short A steps -> small drift -> deeper solve -> weaker signal).

  drift        alpha = EMA(||J dA||/||y||)                     (T2's rule)
  drift_gain   alpha = max(drift, sqrt(1-rho))                 (the candidate)

sqrt(1-rho) is "the residual level A still owns": gains are quadratic in the
residual and gainA/(gainA+gainL) = 1-rho, so its square root is in the same
units alpha is (T1/P1 measured eval error ~= 0.3 alpha).  The rule reads: never
resolve L below the part of the error A can still remove.  No new constant.

Measured against stock Adam on the same budget -- the goalpost is "monotonically
better than Adam", not "reaches the floor".

Run: uv run --extra dev python experiments/expD14_lobotomy/iteration_0/t2b_alpha.py
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

OUT = C.RESULTS / "t2b_alpha.jsonl"
BASE = dict(lset="readout", rho_mode="gain", clip=False, astep="linesearch")
ARMS = {
    "adam":       dict(lset="none"),
    "nodamp":     dict(alpha_mode="fixed", alpha_fixed=1e-15, **BASE),
    "drift":      dict(alpha_mode="drift", **BASE),
    "drift_gain": dict(alpha_mode="drift_gain", **BASE),
}
TARGETS = ["sine", "sine_8pi", "runge", "exp"]


def job(spec):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    env = C.build_case(spec["fn"], spec["N"], init=spec["init"])
    t0 = time.time()
    out = C.train_lobo(env, spec["iters"], lsolver="direct", lr=1e-3,
                       record=10, **ARMS[spec["arm"]])
    rec = dict(spec)
    rec.update(floor0=env["floor0"], best_rel=out["best_rel"],
               final_rel=out["final_rel"], floor_final=out["floor_final"],
               wall=round(time.time() - t0, 1), hist=out["hist"])
    return rec


def main():
    if OUT.exists():
        OUT.unlink()
    specs = [dict(fn=fn, N=128, init=init, arm=arm, iters=4000)
             for init in ("qi", "rand") for fn in TARGETS for arm in ARMS]
    print(f"{len(specs)} runs")
    recs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            recs.append(r)
            C.append(OUT, r)
            print(f"  [{i:2d}/{len(specs)}] {r['init']:5s} {r['fn']:9s} "
                  f"{r['arm']:11s} best {r['best_rel']:.3e}  {r['wall']}s")
    print(f"total {time.time()-t0:.0f}s\n" + "=" * 76)
    for init in ("qi", "rand"):
        print(f"\n{init} init -- best eval rel L2 in 3000 steps")
        print("      (err = best eval rel L2; fl = floor of the FINAL geometry,"
              " i.e. did the features improve)")
        hdr = f"    {'target':>9} {'floor0':>10}"
        for a in ARMS:
            hdr += f" | {a+' err':>10} {'fl':>9}"
        print(hdr)
        for fn in TARGETS:
            sel = {r["arm"]: r for r in recs
                   if r["init"] == init and r["fn"] == fn}
            if len(sel) < len(ARMS):
                continue
            line = f"    {fn:>9} {sel['adam']['floor0']:10.2e}"
            for a in ARMS:
                line += f" | {sel[a]['best_rel']:10.2e} {sel[a]['floor_final']:9.2e}"
            print(line)


if __name__ == "__main__":
    main()
