"""Gate 1 for the multi-level (butterfly) block-whitening idea: does kappa of
the whitened operator actually fall as levels are added?  State = L*d*k.
kappa is taken from the SVD of the whitened matrix itself, never a Gram."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np, core11, ml

OUT = HERE / "gate1.jsonl"

if __name__ == "__main__":
    for nm, mk in [("qi1d128", lambda: core11.prob_qi1d(128)),
                   ("qi2d576", lambda: core11.prob_qi2d(576)),
                   ("twinU512", lambda: core11.prob_twin(512, "unstruct"))]:
        p = mk()
        print(f"\n{nm} d={p['d']} r={p['rank']} kappa={p['kappa']:.2e}", flush=True)
        for kind in ("stride", "random", "offset"):
            for k in (32, 64, 128):
                out = []
                for L in (1, 2, 3, 4):
                    W = ml.MLWhiten(k=k, L=L, kind=kind, rcond=1e-13)
                    B = W.fit(p["A"])
                    ke, rk, _ = ml.kap_eff(B)
                    core11.append(OUT, dict(case=nm, d=p["d"], rank=p["rank"],
                                            kappa0=p["kappa"], kind=kind, k=k,
                                            L=L, kappa=ke, dB=int(B.shape[1]),
                                            state_over_d=W.state_floats() / p["d"]))
                    out.append(f"L{L}:{ke:.1e}({W.state_floats()/p['d']:.0f}d)")
                print(f"  {kind:7s} k={k:3d}  " + "  ".join(out), flush=True)
