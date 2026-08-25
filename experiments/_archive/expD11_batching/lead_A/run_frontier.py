"""Frontier sweep: attainable accuracy vs reorthogonalization window c,
as d grows with c FIXED.  Decides requirement 2 (O(d*k), k fixed)."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, core11, wlsqr

OUT = HERE / "frontier.jsonl"


def best(h):
    v = [q["ev"] for q in h if q["ev"] is not None and np.isfinite(q["ev"])]
    return (min(v), [q["it"] for q in h if q["ev"] == min(v)][0]) if v else (np.nan, -1)


def go(tag, mk, sizes, cs):
    for a in sizes:
        p = mk(a)
        it = int(min(2600, 3.0 * p["rank"] + 80))
        for c in cs:
            t = time.time()
            x, h = wlsqr.lsqr_win(p["A"], p["y"], it, c=c, mode="recent",
                                  Aev=p["A_ev"], yev=p["y_ev"],
                                  ynorm=p["ye_norm"], probe_every=10)
            b, bi = best(h)
            rec = dict(fam=tag, size=a, d=p["d"], rank=p["rank"],
                       floor=p["floor_pool"], kappa=p["kappa"], c=c,
                       iters_cap=it, best=b, best_it=bi, secs=time.time() - t,
                       state_over_d=(c if c < 1e5 else bi))
            core11.append(OUT, rec)
            print(f"{tag} d={p['d']} r={p['rank']} c={c}: {b:.2e} @{bi} "
                  f"({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "twin"):
        go("twinU", lambda a: core11.prob_twin(a, "unstruct"),
           [256, 512, 1024, 2048], [32, 64, 128, 256, 10**6])
    if which in ("all", "qi1d"):
        go("qi1d", core11.prob_qi1d, [64, 128, 256, 512], [32, 64, 128, 256, 10**6])
    if which in ("all", "qi2d"):
        go("qi2d", lambda a: core11.prob_qi2d(a), [144, 324, 576, 1024],
           [32, 64, 128, 256, 10**6])
