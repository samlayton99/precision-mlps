"""Mhaskar (1996): a Chebyshev polynomial approximant realized by tanh neurons with one bias b0 and
small slopes, the monomials coming from finite differences in the slope; all at p bits.

Two realizations of x^r, both merged into one hidden layer with bias b0 = Q(ln(2)/2) and offset 0:

* grid = "centered" (Mhaskar 1996, Lemma 3.2, eq. 3.19, and the paper's appendix; expC12): x^r ~ H_{r,h}(x) = sum_j (-1)^{r-j} binom(r,j)
  tanh(b0 + (j - r/2) h x) / (h^r r! c_r), c_r = tanh^(r)(b0)/r!, on slopes s h/2, s = -d..d: 2d+1
  neurons for degree d. Stencil assembly follows pbit_kernels.c:stencil_network: term = div(a_r, c_r),
  term = div(term, mul(h, Q(k))) for k = 1..r, negated for odd r; for j = 0..r add term to the neuron
  with slope (2j - r) h/2, then term = neg(mul(term, div(Q(r-j), Q(j+1)))).
* grid = "common" (our variant, not in the paper): the same kind of slope set b0 + j h, j = -m..m, but
  every monomial up to degree 2m realized from all 2m+1 values by
  maximal-order differences, x^k ~ sum_j L_kj tanh(b0 + j h x) / (c_k h^k), L_kj = [t^k] l_j(t) the
  Lagrange basis coefficients on the integer grid (exact rationals, rounded once). Readout
  a_j = sum_k e_k L_kj (sequential in k) with e_k = div(a_k, c_k) divided k times by h; m = ceil(d/2),
  so degree d costs 2 ceil(d/2) + 1 neurons.

The polynomial is the p-bit discrete Chebyshev projection (chebyshev.py). Degree d, step h and the
grid are chosen on the validation grid (the paper leaves d and h to unspecified constants).
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial

import mpmath as mp
import numpy as np
import pfloat

from experiments.expC13_five_method_comparison import common as C
from experiments.expC13_five_method_comparison.methods import chebyshev
from experiments.expC13_five_method_comparison.selection import Selection

# expC12's degree grid up to 128 (its 160..511 entries are never competitive: no degree above 7 won at
# any p <= 53 in expC12, and the needed bits grow like d^2)
DEGREES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88,
           96, 112, 128]
STEPS = [float(s) for s in np.geomspace(1e-4, 4.0, 65)]
# above 53 bits the best steps are smaller: extend downward at the same ratio to about 1e-8 (a superset)
_RATIO = STEPS[1] / STEPS[0]
STEPS_EXTENDED = [STEPS[0] / _RATIO ** k for k in range(56, 0, -1)] + STEPS


def steps_for(p: int) -> list:
    return STEPS if p <= 53 else STEPS_EXTENDED
COMMON_DEGREES = DEGREES


def bias_point(F) -> pfloat.PArray:
    return C.constant(lambda: mp.log(2) / 2, F)


def stencils(polys: list, taylor: pfloat.PArray, degrees: list, step: float):
    """Merged readouts for every degree in `degrees` at one step: array (len(degrees), 2n+1) indexed by
    slope index s + n, s = -n..n (n = max degree), and the slopes. Rows whose readouts are not all
    finite are returned with ok[row] = False."""
    F = taylor.fmt
    n = max(degrees)
    h = C.constant(Fraction(step), F)
    A = pfloat.stack([polys[d][:n + 1] for d in degrees])                 # (D, n+1): a_r of each truncation
    active = np.asarray(A != 0)
    term = A / taylor[:n + 1].reshape(1, -1)
    for k in range(1, n + 1):
        term[:, k:] = term[:, k:] / (h * C.constant(k, F))
    odd = np.arange(n + 1) % 2 == 1
    term[:, odd] = -term[:, odd]
    a = pfloat.zeros((len(degrees), 2 * n + 1), F)
    zero = pfloat.zeros((len(degrees), n + 1), F)
    for j in range(n + 1):
        r = np.arange(j, n + 1)
        g = n + 2 * j - r                                                  # slope index s = 2j - r, shifted by n
        contrib = pfloat.where(active[:, j:], term[:, j:], zero[:, j:])
        a[:, g] = a[:, g] + contrib
        if j < n:
            rr = r[1:]
            ratio = pfloat.array(rr - j, F) / C.constant(j + 1, F)
            term[:, j + 1:] = -(term[:, j + 1:] * ratio.reshape(1, -1))
    half = h / C.constant(2, F)
    slopes = half * pfloat.array(np.arange(-n, n + 1), F)
    vals = a.to_numpy(rounding=True)
    t = taylor.to_numpy(rounding=True)[:n + 1]
    bad = np.flatnonzero(~np.isfinite(t) | (t == 0))
    usable = bad[0] - 1 if bad.size else n                                 # largest r with a usable c_r
    ok = np.all(np.isfinite(vals), axis=1) & (np.array(degrees) <= usable)
    return a, slopes, ok


def network(a_row: pfloat.PArray, slopes: pfloat.PArray, b0: pfloat.PArray, degree: int) -> C.TanhNet:
    n = (slopes.shape[0] - 1) // 2
    sl = slice(n - degree, n + degree + 1)
    F = a_row.fmt
    return C.TanhNet(slopes[sl], pfloat.ones(2 * degree + 1, F) * b0, a_row[sl], pfloat.array(0, F))


@lru_cache(maxsize=None)
def _lagrange_exact(m: int) -> tuple:
    """cols[j][k] = [t^k] l_j(t) for the nodes -m..m, exact: l_j = P(t)/((t - j) P'(j))."""
    nodes = list(range(-m, m + 1))
    P = [1]
    for i in nodes:
        P = [-i * P[0]] + [P[k - 1] - i * P[k] for k in range(1, len(P))] + [P[-1]]
    cols = []
    for j in nodes:
        n = len(P) - 1
        q = [0] * n
        q[n - 1] = P[n]
        for k in range(n - 1, 0, -1):
            q[k - 1] = P[k] + j * q[k]
        den = (-1) ** (m - j) * factorial(m + j) * factorial(m - j)
        cols.append(tuple(Fraction(c, den) for c in q))
    return tuple(cols)


@lru_cache(maxsize=64)
def lagrange_table(m: int, p: int) -> pfloat.PArray:
    """Q(L_kj), shape (2m+1 rows k, 2m+1 columns j)."""
    cols = _lagrange_exact(m)
    return pfloat.array([[cols[j][k] for j in range(2 * m + 1)] for k in range(2 * m + 1)], C.fmt(p))


def common_readout(poly: pfloat.PArray, taylor: pfloat.PArray, d: int, step: float) -> pfloat.PArray:
    F = poly.fmt
    m = (d + 1) // 2
    h = C.constant(Fraction(step), F)
    e = poly[:d + 1] / taylor[:d + 1]
    for k in range(1, d + 1):
        e[k:] = e[k:] / h
    L = lagrange_table(m, F.p)[:d + 1, :]
    return pfloat.matmul(e.reshape(1, -1), L).reshape(-1)


def common_slopes(F, step: float, m: int) -> pfloat.PArray:
    return C.constant(Fraction(step), F) * pfloat.array(np.arange(-m, m + 1), F)


def _offer_common(sel, ctx, polys, taylor, b0, step, degrees):
    F = ctx.F
    M = (max(degrees) + 1) // 2
    slopes = common_slopes(F, step, M)
    shell = C.TanhNet(slopes, pfloat.ones(2 * M + 1, F) * b0, pfloat.zeros(2 * M + 1, F), pfloat.array(0, F))
    phi = shell.features(ctx.val_x)
    ones = pfloat.ones((phi.shape[0], 1), F)
    t = taylor.to_numpy(rounding=True)
    for d in degrees:
        hp = {"degree": d, "step": step, "grid": "common"}
        a = common_readout(polys[d], taylor, d, step)
        if not (np.all(np.isfinite(a.to_numpy(rounding=True))) and np.all(np.isfinite(t[:d + 1])) and np.all(t[:d + 1] != 0)):
            sel.offer(hp, None, None, ctx.val_meter, status="nonfinite")
            continue
        m = (d + 1) // 2
        sl = slice(M - m, M + m + 1)
        net = C.TanhNet(slopes[sl], pfloat.ones(2 * m + 1, F) * b0, a, pfloat.array(0, F))
        w = pfloat.concatenate([net.offset.reshape(1), net.readout])
        sel.offer(hp, net, pfloat.matmul(pfloat.concatenate([ones, phi[:, sl]], axis=1), w), ctx.val_meter)


def select(ctx, degrees=DEGREES, steps=None, faithful_only: bool = False,
           common_degrees=COMMON_DEGREES) -> Selection:
    sel = Selection("mhaskar", ctx.target, ctx.p)
    steps = steps_for(ctx.p) if steps is None else steps
    F = ctx.F
    n = max(degrees)
    b0 = bias_point(F)
    cheb = chebyshev.projection(ctx.target, ctx.p, n)
    polys = chebyshev.to_monomials(cheb)
    taylor = chebyshev.tanh_taylor(b0, n)
    for step in steps:
        a, slopes, ok = stencils(polys, taylor, degrees, step)
        # all candidates at this step share the features of slopes s h/2, s = -n..n, with bias b0
        shell = C.TanhNet(slopes, pfloat.ones(2 * n + 1, F) * b0, pfloat.zeros(2 * n + 1, F), pfloat.array(0, F))
        phi = shell.features(ctx.val_x)
        ones = pfloat.ones((phi.shape[0], 1), F)
        for row, d in enumerate(degrees):
            hp = {"degree": d, "step": step, "grid": "centered"}
            if not ok[row]:
                sel.offer(hp, None, None, ctx.val_meter, status="nonfinite")
                continue
            net = network(a[row], slopes, b0, d)
            sl = slice(n - d, n + d + 1)
            w = pfloat.concatenate([net.offset.reshape(1), net.readout])
            out = pfloat.matmul(pfloat.concatenate([ones, phi[:, sl]], axis=1), w)
            sel.offer(hp, net, out, ctx.val_meter)
        if not faithful_only:
            _offer_common(sel, ctx, polys, taylor, b0, step, common_degrees)
    return sel
