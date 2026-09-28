"""How much does the tanh policy matter? Rebuild and evaluate selected configurations with a correctly
rounded tanh (MPFR mpfr_tanh in the same format: precision p, exponent range, gradual underflow) in
place of tanh_p, keeping every other operation identical, and compare the reporting-grid errors.

    .venv/bin/python experiments/expC13_five_method_comparison/tanh_policy_check.py

Only the elementary function changes; tanh_p is within 3 ulp, the correctly rounded tanh within 1/2.
Writes data/tanh_policy.json.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gmpy2  # noqa: E402
import numpy as np  # noqa: E402
import pfloat  # noqa: E402

from experiments.expC13_five_method_comparison import common as C  # noqa: E402
from experiments.expC13_five_method_comparison.methods import quills, staircase  # noqa: E402
from experiments.expC13_five_method_comparison.selection import report_meter  # noqa: E402


def tanh_cr(z: pfloat.PArray) -> pfloat.PArray:
    """Correctly rounded tanh of every element, in z's format (p <= 53: values are binary64)."""
    F = z.fmt
    ctx = gmpy2.context(precision=F.p, round=gmpy2.RoundToNearest, subnormalize=True,
                        emin=F.emin - F.p + 2, emax=F.emax + 1)
    flat = z.to_numpy().ravel()
    with ctx:
        out = np.array([float(gmpy2.tanh(gmpy2.mpfr(v))) for v in flat])
    return pfloat.array(out.reshape(z.shape), F)


class CRTanhNet(C.TanhNet):
    def features(self, x):
        z = x.reshape(-1, 1) * self.slope.reshape(1, -1) + self.bias.reshape(1, -1)
        return tanh_cr(z)


def quills_cr(target, p):
    F = C.fmt(p)
    lam = quills.rule_lambda(target, p)
    slope, bias, meta = quills.geometry(F, quills.WIDTH, quills.HALO, lam)
    x = C.inputs(C.linspace_points(quills.TRAIN_POINTS), F)
    y = C.sample(target, x)
    shell = CRTanhNet(slope, bias, pfloat.zeros(quills.WIDTH, F), pfloat.array(0, F))
    A = pfloat.concatenate([shell.features(x), pfloat.ones((x.shape[0], 1), F)], axis=1)
    w = pfloat.lstsq(A, y, rcond=Fraction(2) ** (1 - p))[0]
    return CRTanhNet(slope, bias, w[:quills.WIDTH], w[quills.WIDTH])


def main():
    rows = []
    for target in ("exp", "chirp"):
        for p in (16, 24, 53):
            F = C.fmt(p)
            xr = C.inputs(C.linspace_points(C.REPORT_POINTS), F)
            meter = report_meter(target)
            summary = {json.dumps([r["method"], r["target"], r["p"]]): r for r in
                       map(json.loads, (C.OUT / "data" / "summary.jsonl").read_text().splitlines())}
            hp = summary[json.dumps(["staircase", target, p])]["hp"]
            nets = {
                "quills": (quills.build(target, p)[0], quills_cr(target, p)),
            }
            st = staircase.build(target, p, hp["N"], Fraction(hp["kappa"]), hp["jumps"], hp.get("halo", 0))
            nets["staircase"] = (st, CRTanhNet(st.slope, st.bias, st.readout, st.offset))
            for method, (net_p, net_cr) in nets.items():
                t = time.monotonic()
                e_p = meter.rel_l2(net_p.forward(xr))
                e_cr = meter.rel_l2(net_cr.forward(xr))
                rows.append({"target": target, "p": p, "method": method, "rel_l2_tanh_p": e_p, "rel_l2_tanh_cr": e_cr,
                             "ratio": e_cr / e_p, "hp": hp if method == "staircase" else {"width": quills.WIDTH}})
                print(rows[-1], f"{time.monotonic() - t:.0f}s", flush=True)
    (C.OUT / "data" / "tanh_policy.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
