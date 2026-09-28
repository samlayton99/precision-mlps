"""Error against network size at p = 53, on the same footing as the precision sweep.

For each target, method and size cap, take the candidate with the smallest validation error among those
with at most `cap` nonzero parameters (the sweep's selection rule, with the cap in place of the budget),
rebuild it, and measure it on the reporting grid. At cap = 3073 (the budget) this is exactly the sweep's
selected model, and the script checks that it reproduces the stored error. ChebNet is continued past
the budget (degrees up to 255) to show what counting neurons instead of parameters buys it.

    .venv/bin/python experiments/expC13_five_method_comparison/size_curves.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pfloat  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.methods import (  # noqa: E402
    chebnet, chebyshev, costarelli, mhaskar, quills, staircase)
from experiments.expC13_five_method_comparison.selection import report_meter  # noqa: E402

P = 53
DATA = C.OUT / "data"
CAPS = [16, 32, 64, 128, 256, 512, 1024, 2048, C.PARAM_BUDGET]
CAPS_BEYOND = [4096, 6144, 8192]          # ChebNet only
METHODS = ["quills", "chebnet", "mhaskar", "staircase", "costarelli"]


def candidates(target, method):
    tabs = [method] if method != "chebnet" else ["chebnet", "chebnet_neurons"]
    out = []
    for t in tabs:
        rows = json.loads((DATA / "candidates" / f"{target}_{t}_p{P}.json").read_text())
        out += [r for r in rows if r.get("status") in ("ok", "over_budget") and r.get("params") is not None]
    return out


def choose(cands, cap):
    ok = [(r["rel_l2"], r["params"], i, r) for i, r in enumerate(cands) if r["params"] <= cap]
    return min(ok, key=lambda t: t[:3])[3] if ok else None


def rebuild(target, method, hp):
    F = C.fmt(P)
    if method == "quills":
        return quills.build(target, P, width=hp["width"], lam=hp["lambda"])[0]
    if method == "staircase":
        return staircase.build(target, P, hp["N"], Fraction(hp["kappa"]), hp["jumps"], hp.get("halo", 0))
    if method == "costarelli":
        return costarelli.build(target, P, hp["w"], hp["K"], hp["centred"])
    if method == "chebnet":
        cheb = chebyshev.projection(target, P, hp["degree"])
        return chebnet.build(cheb, hp["degree"], hp["normalize"])
    # Mhaskar: rebuild exactly as the sweep did (same degree list, same shared arrays)
    n = max(mhaskar.DEGREES)
    b0 = mhaskar.bias_point(F)
    cheb = chebyshev.projection(target, P, n)
    polys = chebyshev.to_monomials(cheb)
    taylor = chebyshev.tanh_taylor(b0, n)
    d = hp["degree"]
    if hp["grid"] == "centered":
        a, slopes, ok = mhaskar.stencils(polys, taylor, mhaskar.DEGREES, hp["step"])
        return mhaskar.network(a[mhaskar.DEGREES.index(d)], slopes, b0, d)
    m = (d + 1) // 2
    return C.TanhNet(mhaskar.common_slopes(F, hp["step"], m), pfloat.ones(2 * m + 1, F) * b0,
                     mhaskar.common_readout(polys[d], taylor, d, hp["step"]), pfloat.array(0, F))


def curve(target, method):
    pfloat.set_num_threads(1)
    cands = candidates(target, method)
    caps = CAPS + (CAPS_BEYOND if method == "chebnet" else [])
    xr = C.inputs(C.linspace_points(C.REPORT_POINTS), C.fmt(P))
    out, seen = [], {}
    for cap in caps:
        r = choose(cands, cap)
        if r is None:
            continue
        key = json.dumps({k: r[k] for k in r if k not in ("rel_l2", "rel_linf", "params", "neurons", "depth",
                                                         "max_abs", "status")}, sort_keys=True)
        if key not in seen:
            net = rebuild(target, method, r)
            seen[key] = report_meter(target).rel_l2(net.forward(xr))
            assert net.params() == r["params"], (target, method, cap)
        out.append({"cap": cap, "params": r["params"], "rel_l2": seen[key], "val_rel_l2": r["rel_l2"],
                    "hp": json.loads(key)})
    return target, method, out


def main():
    jobs = [(t, m) for t in ("exp", "sine", "runge", "chirp") for m in METHODS]
    with ProcessPoolExecutor(9) as ex:
        res = list(ex.map(curve, *zip(*jobs)))
    summary = {(r["target"], r["method"]): r for r in map(json.loads, (DATA / "summary.jsonl").read_text().splitlines())
               if r["p"] == P}
    data = {}
    for target, method, pts in res:
        at_budget = next(pt for pt in pts if pt["cap"] == C.PARAM_BUDGET)
        stored = summary[(target, method)]
        assert at_budget["rel_l2"] == stored["rel_l2"], (target, method, at_budget, stored["rel_l2"])
        data.setdefault(target, {})[method] = pts
        print(target, method, [(pt["params"], f"{pt['rel_l2']:.2g}") for pt in pts])
    (DATA / "size_curves_p53.json").write_text(json.dumps(data, indent=1) + "\n")
    print("the point at the budget equals the sweep's p = 53 value for every target and method")


if __name__ == "__main__":
    main()
