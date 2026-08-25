"""T4 -- the headline grid, with the L set DISCOVERED rather than given.

Everything before this handed the optimizer L = (v, c0). Here L comes from
iteration 11's certificate (second-difference nonlinearity probe + Adam's second
moment as the observability test), refreshed on a cadence, so the only thing
assumed is that a linear block might exist.

Four arms on every cell:
  adam        stock Adam
  adam_fin    stock Adam, then ONE exact solve on the readout at the end
              (the "train the features, refit at the end" baseline -- expD02's
              winner and the thing any in-training solve has to beat)
  lobo        the proposal, alpha pinned low (T2/T2b winner on QI init)
  lobo_soft   the proposal with alpha = max(drift, sqrt(1-rho)) -- deliberately
              under-solving so A keeps a learning signal (T2b winner on runge)

Reported for every run: the error it reached, AND the floor of the geometry it
finished on -- which is exactly the error a terminal exact solve would give.
So every arm is scored twice: as-is, and with a finisher.

Run: uv run --extra dev python experiments/expD14_lobotomy/iteration_0/t4_grid.py
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

OUT = C.RESULTS / "t4_grid.jsonl"
BASE = dict(lset="certificate", rho_mode="gain", clip=False,
            astep="linesearch", lsolver="direct")
ARMS = {
    "adam":      dict(lset="none", lsolver="direct"),
    "lobo":      dict(alpha_mode="fixed", alpha_fixed=1e-15, **BASE),
    "lobo_soft": dict(alpha_mode="drift_gain", **BASE),
}
TARGETS = ["sine", "sine_8pi", "runge", "sine_mixture", "exp", "abs_cubed"]
WIDTHS = [64, 128, 256]


def job(spec):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    env = C.build_case(spec["fn"], spec["N"], init=spec["init"])
    t0 = time.time()
    kw = dict(ARMS[spec["arm"]])
    ls = kw.pop("lsolver")
    out = C.train_lobo(env, spec["iters"], lsolver=ls, lr=1e-3, record=20,
                       refresh=200, **kw)
    rec = dict(spec)
    rec.update(floor0=env["floor0"], best_rel=out["best_rel"],
               final_rel=out["final_rel"], floor_final=out["floor_final"],
               B_final=out["B_final"], m=env["m"], W=env["W"],
               n_readout=env["W"] + 1, passes=out["passes"],
               wall=round(time.time() - t0, 1), hist=out["hist"])
    return rec


def main():
    if OUT.exists():
        OUT.unlink()
    specs = [dict(fn=fn, N=N, init=init, arm=arm, iters=3000)
             for init in ("qi", "rand") for N in WIDTHS
             for fn in TARGETS for arm in ARMS]
    print(f"{len(specs)} runs")
    recs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(job, s) for s in specs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            recs.append(r)
            C.append(OUT, r)
            print(f"  [{i:3d}/{len(specs)}] {r['init']:5s} N{r['N']:<4d} "
                  f"{r['fn']:13s} {r['arm']:10s} best {r['best_rel']:.3e} "
                  f"fin {r['floor_final']:.3e} B {r['B_final']}/{r['n_readout']}"
                  f"  {r['wall']}s")
    print(f"total {time.time()-t0:.0f}s")
    table(recs)


def table(recs):
    for init in ("qi", "rand"):
        for tag, key in (("as-is", "best_rel"),
                         ("after a terminal exact solve", "floor_final")):
            print(f"\n{'='*84}\n{init} init -- eval rel L2, {tag}")
            print(f"    {'target':>13} {'N':>5} {'floor0':>10} "
                  f"{'adam':>11} {'lobo':>11} {'lobo_soft':>11}")
            for fn in TARGETS:
                for N in WIDTHS:
                    sel = {r["arm"]: r for r in recs
                           if r["init"] == init and r["fn"] == fn
                           and r["N"] == N}
                    if len(sel) < len(ARMS):
                        continue
                    print(f"    {fn:>13} {N:5d} {sel['adam']['floor0']:10.2e} "
                          + " ".join(f"{sel[a][key]:11.2e}" for a in ARMS))


if __name__ == "__main__":
    main()
