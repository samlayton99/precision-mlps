"""Classical staircase: q_N(x) = f(-1) + sum_{j=1}^N [f(x_j) - f(x_{j-1})] S(k (x - t_j)), S(t) = (1 +
tanh t)/2, x_j = -1 + 2j/N (the paper's appendix; historically Jones 1990, Chen-Chen-Liu, and
Costarelli-Spigler's G_N in Anal. Theory Appl. 29 (2013), restated in J. Integral Eq. Appl. 25 (2013)
Thm 2.2); all at p bits.

As a tanh network: readouts a_j = mul(sub(y_j, y_{j-1}), 1/2) (the halving is exact), slope
w = Q(kappa N), biases neg(mul(w, t_j)), and offset c = mul(add(y_0, y_N), 1/2): the constant parts of
the sigmoids telescope, f(-1) + sum_j (y_j - y_{j-1})/2 = (y_0 + y_N)/2, so the offset is formed with
one rounding instead of a sequential N-term sum. Hyperparameters chosen on the validation grid: N; the
steepness factor kappa (the appendix's network is kappa = 1, tanh slope N = 2/h; the logistic slope
threshold N ln(N-1)/(b-a) of Costarelli-Spigler is kappa = ln(N-1)/4 in these units, inside the grid);
the jump location t_j = x_j (classical) or the cell midpoint -1 + (2j-1)/N (our variant); and, with
midpoint jumps, a halo of K samples per side extrapolated from the three nearest in-domain samples by
the quadratic Lagrange formula (integer weights, sums at p bits; no oracle calls outside [-1,1]),
which removes the first-order error the truncated smoothing kernel leaves at the ends.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pfloat

from experiments.expC13_five_method_comparison import common as C
from experiments.expC13_five_method_comparison.selection import Selection

SIZES = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1008, 1024]
KAPPAS = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4), Fraction(8),
          Fraction(32)]
JUMPS = ["sample", "midpoint"]
HALOS = [0, 8]


def _extrapolate(y0, y1, y2, i, F):
    """Quadratic Lagrange extrapolation i cells beyond y0 (nodes 0, 1, 2 at -i):
    (i+1)(i+2)/2 y0 - i(i+2) y1 + i(i+1)/2 y2, products and sums at p bits in that order."""
    s = C.constant((i + 1) * (i + 2) // 2, F) * y0
    s = s + C.constant(-i * (i + 2), F) * y1
    return s + C.constant(i * (i + 1) // 2, F) * y2


def build(target: str, p: int, N: int, kappa: Fraction, jumps: str, halo: int = 0) -> C.TanhNet:
    F = C.fmt(p)
    x = C.inputs([Fraction(-1) + Fraction(2 * j, N) for j in range(N + 1)], F)
    y = C.sample(target, x)
    if halo:
        assert jumps == "midpoint" and N >= 2
        left = [_extrapolate(y[0], y[1], y[2], i, F) for i in range(halo, 0, -1)]
        right = [_extrapolate(y[N], y[N - 1], y[N - 2], i, F) for i in range(1, halo + 1)]
        y = pfloat.concatenate([pfloat.stack(left), y, pfloat.stack(right)])
    n = y.shape[0] - 1                                           # jumps
    a = (y[1:] - y[:-1]) * C.constant(Fraction(1, 2), F)
    w = C.constant(kappa * N, F)
    if jumps == "sample":
        t = x[1:]
    else:
        t = C.inputs([Fraction(-1) + Fraction(2 * j - 1, N) for j in range(1 - halo, N + halo + 1)], F)
    slope = pfloat.ones(n, F) * w
    bias = -(slope * t)
    c = (y[0] + y[n]) * C.constant(Fraction(1, 2), F)          # telescoped constant part
    return C.TanhNet(slope, bias, a, c)


def select(ctx, sizes=SIZES, kappas=KAPPAS, jumps=JUMPS, faithful_only: bool = False) -> Selection:
    sel = Selection("staircase", ctx.target, ctx.p)
    if faithful_only:
        kappas, jumps = [Fraction(1)], ["sample"]
    for N in sizes:
        for kappa in kappas:
            for jump, halo in [(j, h) for j in jumps for h in (HALOS if j == "midpoint" else [0])]:
                if N + 2 * halo > 1024:
                    continue
                hp = {"N": N, "kappa": float(kappa), "jumps": jump, "halo": halo}
                net = build(ctx.target, ctx.p, N, kappa, jump, halo)
                if net.params() > ctx.budget:
                    sel.offer(hp, None, None, ctx.val_meter, status="over_budget")
                    continue
                sel.offer(hp, net, net.forward(ctx.val_x), ctx.val_meter)
    return sel
