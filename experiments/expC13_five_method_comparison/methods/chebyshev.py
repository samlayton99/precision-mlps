"""p-bit polynomial front end shared by Mhaskar and ChebNet (SPEC "Polynomial front end").

Discrete Chebyshev projection from the oracle's samples at M first-kind nodes, computed with the
three-term recurrence and sequential sums at p bits; the conversion of every truncation to monomial
coefficients (Mhaskar); the scaled tanh derivatives at a point (Mhaskar). The operation order is
that of experiments/expC12_mhaskar_comparison/pbit_kernels.c (polynomial_data), so the two agree
bit for bit on the same p-bit inputs (tests/test_expC13_constructions.py).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pfloat

from experiments.expC13_five_method_comparison import common as C

NODES = 4096


@lru_cache(maxsize=None)
def nodes(p: int, count: int = NODES) -> pfloat.PArray:
    """x_k = Q(cos(pi (k + 1/2) / count)), k = 0..count-1 (sampling locations, rounded once)."""
    return C.inputs(C.chebyshev_points(count), C.fmt(p))


def projection(target: str, p: int, degree: int, count: int = NODES) -> pfloat.PArray:
    """c_0 = (sum_i y_i) / M, c_k = 2 (sum_i y_i T_k(x_i)) / M, with y_i the oracle's p-bit samples,
    T_k by T_{k+1} = mul(mul(2, x), T_k) - T_{k-1}, and each sum sequential over the nodes."""
    F = C.fmt(p)
    x = nodes(p, count)
    y = C.sample(target, x)
    M = C.constant(count, F)
    two = C.constant(2, F)
    out = []
    t0, t1 = pfloat.ones(count, F), x
    for k in range(degree + 1):
        tk = t0 if k == 0 else t1
        if k >= 2:
            t2 = (two * x) * t1 - t0
            t0, t1 = t1, t2
            tk = t2
        s = pfloat.sum(y * tk)
        out.append((s * two if k else s) / M)
    return pfloat.stack(out)


def to_monomials(cheb: pfloat.PArray) -> list:
    """Monomial coefficients of every truncation sum_{k<=d} c_k T_k, d = 0..n (a list of arrays of
    length n+1): poly = poly + c_k * T_k-coefficients, with the T_k coefficient recurrence
    next_j = mul(2, cur_{j-1}) - prev_j at p bits."""
    F = cheb.fmt
    n = cheb.shape[0] - 1
    two = C.constant(2, F)
    zero = pfloat.zeros(1, F)
    poly = pfloat.zeros(n + 1, F)
    prev = None
    cur = pfloat.zeros(n + 1, F)
    cur[0] = pfloat.array(1, F)
    out = []
    for k in range(n + 1):
        poly = poly + cheb[k] * cur
        out.append(poly.copy())
        if k == 0:
            prev = cur
            cur = pfloat.zeros(n + 1, F)
            if n:
                cur[1] = pfloat.array(1, F)
        else:
            shifted = pfloat.concatenate([zero, two * cur[:-1]])
            prev, cur = cur, shifted - prev
    return out


def tanh_taylor(b: pfloat.PArray, degree: int, c0: pfloat.PArray | None = None) -> pfloat.PArray:
    """c_r = tanh^(r)(b) / r!: c_0 = tanh_p(b); (r+1) c_{r+1} = [r == 0] - sum_{j<=r} c_j c_{r-j}
    (the sum from zero, j ascending), divided by Q(r+1). c0 overrides tanh_p(b) (tests only)."""
    F = b.fmt
    c = [pfloat.tanh(b) if c0 is None else c0]
    one = C.constant(1, F)
    for r in range(degree):
        cv = pfloat.stack(c)
        total = pfloat.dot(cv, cv[::-1])
        num = one - total if r == 0 else -total
        c.append(num / C.constant(r + 1, F))
    return pfloat.stack(c)
