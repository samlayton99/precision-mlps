"""Observable stopping rule for reorthogonalized LSQR: does the Fong-Saunders
observable pick the oracle-best iterate?  (No accuracy claim in this repo is
allowed to rest on best-over-trajectory alone.)"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, core11, wlsqr

OUT = HERE / "stop.jsonl"
CASES = [("qi1d N=128", lambda: core11.prob_qi1d(128)),
         ("qi1d N=256", lambda: core11.prob_qi1d(256)),
         ("qi1d N=512", lambda: core11.prob_qi1d(512)),
         ("qi2d d=571", lambda: core11.prob_qi2d(576)),
         ("qi2d d=1021", lambda: core11.prob_qi2d(1024)),
         ("twinU 512", lambda: core11.prob_twin(512, "unstruct")),
         ("twinU 1024", lambda: core11.prob_twin(1024, "unstruct")),
         ("twinC 1024", lambda: core11.prob_twin(1024, "cluster"))]

if __name__ == "__main__":
    print(f"{'case':12s} {'d':>5s} {'r':>5s} {'floor':>9s} {'oracle':>9s} "
          f"{'@it':>5s} {'stopped':>9s} {'@it':>5s} {'ratio':>6s} {'secs':>6s}")
    for nm, mk in CASES:
        p = mk()
        it = int(2.0 * p["rank"] + 60)
        t = time.time()
        x, h = wlsqr.lsqr_win(p["A"], p["y"], it, c=10**6, mode="recent",
                              Aev=p["A_ev"], yev=p["y_ev"], ynorm=p["ye_norm"],
                              probe_every=1)
        secs = time.time() - t
        ny = np.linalg.norm(p["y"])
        # Fong-Saunders style: first iterate with phibar <= 5*eps*||y||
        tol = 5 * np.finfo(float).eps * ny
        st = next((q for q in h if q["phibar"] <= tol), h[-1])
        ok = min(q["ev"] for q in h)
        oi = [q["it"] for q in h if q["ev"] == ok][0]
        core11.append(OUT, dict(case=nm, d=p["d"], rank=p["rank"],
                                floor=p["floor_pool"], oracle=ok, oracle_it=oi,
                                stopped=st["ev"], stopped_it=st["it"],
                                secs=secs))
        print(f"{nm:12s} {p['d']:5d} {p['rank']:5d} {p['floor_pool']:9.2e} "
              f"{ok:9.2e} {oi:5d} {st['ev']:9.2e} {st['it']:5d} "
              f"{st['ev']/ok:6.2f} {secs:6.1f}", flush=True)
