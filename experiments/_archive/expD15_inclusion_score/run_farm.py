"""expD15 -- the acid test.

For every case and every candidate signal: rank the parameters by the score,
take the top k, SOLVE ONLY THOSE using the true Jacobian columns for whatever
was selected (geometry included -- the selection is free to pick first-layer
weights and the solve must then include them), apply, and measure eval error.
Sweep k.

A signal is good if its curve dominates: it finds the subset that actually pays.
This needs no ground truth about which parameters "should" be in L, which is
the point -- it measures the thing we care about rather than agreement with a
prior.

Run: uv run --extra dev python experiments/expD15_inclusion_score/run_farm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core15 as C  # noqa: E402
from signals import make_signals  # noqa: E402

OUT = C.RESULTS / "farm.jsonl"
CASES = ["qi", "clustered", "datagap", "random"]


def ks_for(m):
    g = sorted({int(round(x)) for x in np.unique(
        np.concatenate([np.arange(1, min(m, 12) + 1),
                        np.round(np.geomspace(2, m, 22))]))})
    return [k for k in g if 1 <= k <= m]


def run_cell(case, dim, N, warm=True, seed=0):
    env = C.build_case(case, dim=dim, N=N, seed=seed)
    net = env["net"]
    th = net.theta.copy()   # case-provided small-random readout: real work left
    base = C.eval_rel(env, th)
    sig, J, r = make_signals(env, th)
    m = net.m
    lab = net.labels()
    rows = []
    # Matched-k comparison. Reporting best-over-k would pick k using the eval
    # error, which is unavailable in practice and is expD10's T10 "best over
    # trajectory" trap; every signal is instead scored at the SAME budgets.
    # The solve is damped and NOT rank-truncated: a truncated SVD silently
    # discards unresolved directions, which is the very failure an inclusion
    # score is supposed to catch, so leaving it in hides all the signal.
    good, bad, pr = C.region_split(env, th)
    n_v = int((lab == "v").sum())
    k_match = [max(1, n_v // 4), max(1, n_v // 2), n_v, min(m, 2 * n_v)]
    s1 = np.linalg.svd(J, compute_uv=False)[0]
    mu = (1e-8 * s1) ** 2
    for name, d in sig.items():
        order = np.argsort(-d["v"])
        curve = []
        for k in ks_for(m):
            idx = np.sort(order[:k])
            e, _ = C.solve_subset(env, th, idx, mu=mu, rcond=0.0)
            curve.append((k, e))
        at, at_good, at_bad = {}, {}, {}
        for k in k_match:
            idx = np.sort(order[:k])
            _, th2 = C.solve_subset(env, th, idx, mu=mu, rcond=0.0)
            eo, eg, eb = C.eval_rel_split(env, th2, good, bad)
            at[k] = float(eo)
            at_good[k] = float(eg)
            at_bad[k] = float(eb)
        sel = np.sort(order[:n_v])
        rows.append(dict(case=case, dim=dim, N=N, signal=name, cost=d["cost"],
                         feasible=d["feasible"], why=d["why"], base=base,
                         at_matched={str(k): v for k, v in at.items()},
                         at_good={str(k): v for k, v in at_good.items()},
                         at_bad={str(k): v for k, v in at_bad.items()},
                         k_match=k_match, n_v=n_v,
                         best=float(min(v for v in at.values())),
                         curve=[(int(a), float(b)) for a, b in curve],
                         frac_v=float(np.mean(lab[sel] == "v")),
                         frac_geom=float(np.mean(np.isin(lab[sel], ("A", "b"))))))
    return env, th, rows


def main():
    if OUT.exists():
        OUT.unlink()
    t0 = time.time()
    for dim, N in ((1, 24), (2, 72)):
        for case in CASES:
            env, th, rows = run_cell(case, dim, N)
            for rec in rows:
                C.append(OUT, rec)
            base = rows[0]["base"]
            print(f"\n=== {case}-{dim}d  N={N}  m={env['net'].m}  "
                  f"n_train={env['Xtr'].shape[0]}  err before any solve {base:.3e}")
            km = rows[0]["k_match"]
            kk = str(km[2])
            print(f"    {'signal':>14} {'feas':>5} {'all':>11} "
                  f"{'GOOD parity':>12} {'BAD parity':>12} {'%v':>5}")
            for rec in sorted(rows, key=lambda z: z["at_matched"][kk]):
                print(f"    {rec['signal']:>14} {str(rec['feasible']):>5} "
                      f"{rec['at_matched'][kk]:11.3e} {rec['at_good'][kk]:12.3e} "
                      f"{rec['at_bad'][kk]:12.3e} {100*rec['frac_v']:5.0f}")
    print(f"\ntotal {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
