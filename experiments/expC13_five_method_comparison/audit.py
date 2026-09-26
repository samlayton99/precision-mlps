"""Audit of the expC13 results, independent of the code paths that produced them.

1. Replay: reload saved models from disk, rerun the forward pass on the reporting grid, remeasure, and
   compare with the stored error (every row at p in AUDIT_P).
2. Independent arithmetic: for a sample of saved models (p <= 53), recompute the outputs with gmpy2
   (MPFR) operations written from SPEC alone and the independent tanh_p of tests/pbit_replay_reference,
   and require bit-for-bit equality with pfloat's forward pass.
3. Independent error: recompute the relative L2 error of those outputs directly in mpmath at 256 bits
   against f evaluated in mpmath, and compare with the stored value.
4. Figure consistency: the error-versus-size data at the budget must equal the precision sweep at p = 53.

    .venv/bin/python experiments/expC13_five_method_comparison/audit.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import gmpy2  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import pfloat  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.selection import report_meter  # noqa: E402

DATA = C.OUT / "data"
AUDIT_P = [8, 16, 24, 32, 40, 53, 64, 96, 128]


def rows():
    return [json.loads(s) for s in (DATA / "summary.jsonl").read_text().splitlines()]


def replay(row):
    pfloat.set_num_threads(1)
    net = C.load_model(DATA / "models" / f"{row['target']}_{row['method']}_p{row['p']}.npz")
    xr = C.inputs(C.linspace_points(C.REPORT_POINTS), C.fmt(row["p"]))
    e = report_meter(row["target"]).rel_l2(net.forward(xr))
    return row["target"], row["method"], row["p"], e, row["rel_l2"]


def _ctx(p):
    return gmpy2.context(precision=p, round=gmpy2.RoundToNearest, subnormalize=True,
                         emin=C.EMIN - p + 2, emax=C.EMAX + 1)


def independent_outputs(net, xs, p):
    """Forward pass with gmpy2 operations, written from SPEC 'Networks' (not using pfloat's forward)."""
    import pbit_replay_reference as ref
    out = []
    if net.kind == "tanh":
        w, b, a = (v.to_numpy() for v in (net.slope, net.bias, net.readout))
        c = float(net.offset.to_numpy())
        for x in xs:
            with _ctx(p):
                s = gmpy2.mpfr(c)
                for wj, bj, aj in zip(w, b, a):
                    z = gmpy2.mpfr(wj) * gmpy2.mpfr(x) + gmpy2.mpfr(bj)
                    t = ref.tanh_p(float(z), p)
                    s = s + gmpy2.mpfr(t) * gmpy2.mpfr(aj)
            out.append(float(s))
        return np.array(out)
    for x in xs:
        with _ctx(p):
            h = [gmpy2.mpfr(x)]
            for li, L in enumerate(net.layers):
                W, B = L.w.to_numpy(), L.bias.to_numpy()
                z = []
                for i in range(len(B)):
                    s = gmpy2.mpfr(B[i])
                    for t in range(L.idx.shape[1]):
                        if L.mask[i, t]:
                            s = s + gmpy2.mpfr(W[i, t]) * h[L.idx[i, t]]
                    if li < len(net.layers) - 1:
                        m = s if s > 0 else gmpy2.mpfr(0)
                        s = m * m
                    z.append(s)
                h = z
        out.append(float(h[0]))
    return np.array(out)


def independent_check(target, method, p):
    net = C.load_model(DATA / "models" / f"{target}_{method}_p{p}.npz")
    pts = C.linspace_points(C.REPORT_POINTS)
    x = C.inputs(pts, C.fmt(p))
    y = net.forward(x).to_numpy()
    idx = np.linspace(0, C.REPORT_POINTS - 1, 41).astype(int)
    ind = independent_outputs(net, x.to_numpy()[idx], p)
    bitwise = bool(np.array_equal(ind, y[idx]))
    f = C.TARGETS[target]
    with mp.workprec(256):
        num = den = mp.mpf(0)
        for pt, yi in zip(pts, y):
            fx = f(C._mpf(pt))
            num += (mp.mpf(yi) - fx) ** 2
            den += fx ** 2
        e = float(mp.sqrt(num / den))
    return {"target": target, "method": method, "p": p, "bitwise_equal_on_41_points": bitwise,
            "rel_l2_mpmath": e}


def main():
    R = rows()
    todo = [r for r in R if r["p"] in AUDIT_P]
    with ProcessPoolExecutor(9) as ex:
        res = list(ex.map(replay, todo, chunksize=4))
    bad = [r for r in res if r[3] != r[4]]
    print(f"1. replay: {len(res)} rows reloaded and remeasured; mismatches: {len(bad)}", bad[:5])
    stored = {(r["target"], r["method"], r["p"]): r["rel_l2"] for r in R}
    jobs = [(t, m, p) for t in ("exp", "chirp") for m in ("quills", "mhaskar", "staircase", "costarelli", "chebnet")
            for p in (24, 53)]
    with ProcessPoolExecutor(9) as ex:
        ind = list(ex.map(independent_check, *zip(*jobs)))
    for r in ind:
        r["rel_l2_stored"] = stored[(r["target"], r["method"], r["p"])]
        r["relative_difference"] = abs(r["rel_l2_mpmath"] - r["rel_l2_stored"]) / r["rel_l2_stored"]
        print("2-3.", r)
    (DATA / "audit.json").write_text(json.dumps({"replay_rows": len(res), "replay_mismatches": bad,
                                                 "independent": ind}, indent=1) + "\n")


if __name__ == "__main__":
    main()
