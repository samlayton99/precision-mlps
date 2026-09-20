"""T2 -- the assembled optimizer, with the L set given (oracle = the readout)
so that the ENERGY SPLIT and the DAMPING RULE are the only things under test.

Arms -- one ablation per row, everything else held at the proposal:
  adam        stock Adam                                      (the floor to beat)
  lobo        drift alpha + gain rho + line-searched A step    (the proposal)
  clip        the trust region put back on the L step
  adamstep    A takes Adam's own (scale-free) length instead of the line search
  nodamp      alpha pinned to 1e-15 -- exact solve every step, no damping
  alpha1      alpha pinned to 1    -- maximally damped, ~gradient steps on L
  rho1        rho pinned to 1 -- A never moves
  rho0        rho pinned to 0 -- no allocation; A at full length, L unthrottled
  resetmom    Adam's moments zeroed on L after each solve
  adamL       Adam also steps L (iteration 11 ran Adam on every parameter)

Two inits: qi (geometry is right, readout zeroed) and rand (stock nn.Linear).
Solver: the direct damped SVD solve, so the O(d) solver's own limits (T1/P2)
cannot be confused with the allocation logic.  T3 swaps in the LSQR.

Run: uv run --extra dev python experiments/expD14_lobotomy/iteration_0/t2_assembly.py
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

OUT = C.RESULTS / "t2_assembly.jsonl"

PROPOSAL = dict(lset="readout", alpha_mode="drift", rho_mode="gain",
                clip=False, astep="linesearch")


def _v(**over):
    d = dict(PROPOSAL)
    d.update(over)
    return d


ARMS = {
    "adam":     dict(lset="none"),
    "lobo":     _v(),
    "clip":     _v(clip=True),
    "adamstep": _v(astep="adam"),
    "nodamp":   _v(alpha_mode="fixed", alpha_fixed=1e-15),
    "alpha1":   _v(alpha_mode="fixed", alpha_fixed=1.0),
    "rho1":     _v(rho_mode="fixed", rho_fixed=1.0),
    "rho0":     _v(rho_mode="fixed", rho_fixed=0.0),
    "resetmom": _v(reset_moments=True),
    "adamL":    _v(adam_on_L=True),
}


def job(spec):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    env = C.build_case(spec["fn"], spec["N"], init=spec["init"])
    t0 = time.time()
    out = C.train_lobo(env, spec["iters"], lsolver="direct", lr=spec["lr"],
                       record=spec.get("record", 5), **ARMS[spec["arm"]])
    rec = dict(spec)
    rec.update(floor0=env["floor0"], best_rel=out["best_rel"],
               final_rel=out["final_rel"], wall=round(time.time() - t0, 1),
               hist={k: v for k, v in out["hist"].items()})
    return rec


def main():
    if OUT.exists():
        OUT.unlink()
    specs = [dict(fn=fn, N=128, init=init, arm=arm, iters=2500, lr=1e-3)
             for init in ("qi", "rand")
             for fn in ("sine", "runge")
             for arm in ARMS]
    print(f"{len(specs)} runs")
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            recs.append(r)
            C.append(OUT, r)
            print(f"  [{i:2d}/{len(specs)}] {r['init']:5s} {r['fn']:6s} "
                  f"{r['arm']:9s} best {r['best_rel']:.3e} "
                  f"(floor0 {r['floor0']:.2e})  {r['wall']}s")
    print(f"total {time.time()-t0:.0f}s")

    print("\n" + "=" * 78)
    for init in ("qi", "rand"):
        for fn in ("sine", "runge"):
            f0 = [r for r in recs if r["init"] == init and r["fn"] == fn][0]["floor0"]
            print(f"\n{init:5s} {fn:6s}   floor of the INITIAL geometry = {f0:.3e}")
            print(f"    {'arm':>9} {'best rel L2':>13} {'final':>13} "
                  f"{'x floor0':>10} {'alpha_end':>10} {'rho_mean':>9}")
            for a in ARMS:
                r = [x for x in recs if x["init"] == init and x["fn"] == fn
                     and x["arm"] == a][0]
                h = r["hist"]
                al = h["alpha"][-1] if h["alpha"] else float("nan")
                rho = np.mean(h["rho"]) if h["rho"] else float("nan")
                print(f"    {a:>9} {r['best_rel']:13.3e} {r['final_rel']:13.3e} "
                      f"{r['best_rel']/f0:10.2f} {al:10.2e} {rho:9.3f}")


if __name__ == "__main__":
    main()
