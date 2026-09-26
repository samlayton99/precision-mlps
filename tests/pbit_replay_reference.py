"""Independent gmpy2 (MPFR) reference replay of experiments/expC11_true_precision_law/SPEC.md.

Written from the spec alone (no reference to src/precision/). Every add/sub/mul/div/sqrt is a single
MPFR operation in a context of precision p, round-to-nearest-even, no subnormals, and the widest
exponent range, so each result is Q_p of the exact result. Negation, abs, comparisons, ldexp by an
integer and rint are exact.

It replays the model (constants, tanh_p, features, readout). The least-squares solve is checked
separately, against netlib reference LAPACK compiled from source (tests/test_pbit.py).

Public API:
    constants(p) -> dict
    q(x, p) -> float
    tanh_p(z, p) -> float
    features(p, x, N, halo, lam) -> ndarray (M x W)
    forward(p, x_eval, N, halo, lam, weights) -> ndarray
"""

from __future__ import annotations

from fractions import Fraction
from math import factorial

import gmpy2
import numpy as np
from gmpy2 import mpfr, mpq

_HI_PREC = 2000


def _ctx(p: int):
    return gmpy2.context(
        precision=p,
        round=gmpy2.RoundToNearest,
        subnormalize=False,
        emin=gmpy2.get_emin_min(),
        emax=gmpy2.get_emax_max(),
    )


def _to_exact_mpq(x) -> mpq:
    if isinstance(x, Fraction):
        return mpq(x.numerator, x.denominator)
    if isinstance(x, (int, np.integer)):
        return mpq(int(x))
    if isinstance(x, mpfr):
        return mpq(x)  # exact
    fr = Fraction(float(x))  # exact for a float
    return mpq(fr.numerator, fr.denominator)


def _qp(x, p: int) -> mpfr:
    """Q_p as an mpfr: a single correct rounding of an exact value (float, int, Fraction, mpfr)."""
    with _ctx(p):
        return mpfr(_to_exact_mpq(x))


def q(x, p: int) -> float:
    """Round-to-nearest-even of x to p significant bits, unrestricted exponent, returned as float."""
    return float(_qp(x, p))


def _taylor_degree(p: int) -> int:
    d = 1
    bound = Fraction(1, 2 ** (p + 1))
    while Fraction(35, 100) ** d / factorial(d + 1) > bound:
        d += 1
    return d


def _constants_mpfr(p: int) -> dict:
    kb = (p + 4).bit_length()
    phi_prec = max(p - kb, 1)
    with _ctx(_HI_PREC):
        ln2_hi = gmpy2.const_log2()
        invln2_hi = mpfr(1) / ln2_hi
        sat_hi = mpfr(p + 3) * ln2_hi / 2
    with _ctx(phi_prec):
        ln2hi = gmpy2.const_log2()  # correctly rounded to max(p - k_b, 1) bits
    with _ctx(_HI_PREC):
        lo_hi = ln2_hi - ln2hi  # ln2hi is exact at 2000 bits; difference carries ~2000 bits
    with _ctx(p):
        invln2 = mpfr(invln2_hi)
        ln2lo = mpfr(lo_hi)
        sat = mpfr(sat_hi)
        ln2hi_p = mpfr(ln2hi)  # exact (fewer bits than p)
    d = _taylor_degree(p)
    coeffs = [_qp(Fraction(1, factorial(j + 1)), p) for j in range(d)]
    return dict(INVLN2=invln2, LN2HI=ln2hi_p, LN2LO=ln2lo, SAT=sat, coeffs=coeffs, d=d)


def constants(p: int) -> dict:
    c = _constants_mpfr(p)
    return {
        "INVLN2": float(c["INVLN2"]),
        "LN2HI": float(c["LN2HI"]),
        "LN2LO": float(c["LN2LO"]),
        "SAT": float(c["SAT"]),
        "coeffs": [float(a) for a in c["coeffs"]],
    }


# ---------------------------------------------------------------------------------------------
# The p-bit computation. All functions below must be called inside local_context(_ctx(p)).
# Python operators on mpfr operands are single correctly rounded MPFR operations in that context.
# ---------------------------------------------------------------------------------------------


def _rint(x: mpfr) -> int:
    """Exact round-to-nearest-integer, ties to even."""
    m = _to_exact_mpq(x)
    return round(Fraction(int(m.numerator), int(m.denominator)))  # exact, ties to even


def _tanh_p(z: mpfr, C: dict) -> mpfr:
    one = mpfr(1)
    two = mpfr(2)
    a = abs(z)
    if a >= C["SAT"]:
        return one if z >= 0 else -one
    u = -2 * a  # exact (power-of-two scaling); both operands exact, result exact
    k = _rint(u * C["INVLN2"])
    coeffs = C["coeffs"]
    d = len(coeffs)
    if k == 0:
        r = u
    else:
        kf = mpfr(k)  # small integer, exact at p bits
        r = (u - kf * C["LN2HI"]) - kf * C["LN2LO"]
    qv = coeffs[d - 1]
    for j in range(d - 2, -1, -1):
        qv = coeffs[j] + r * qv
    P = r * qv
    if k == 0:
        E = P
    else:
        pow2k = gmpy2.mul_2exp(one, k)  # exact 2^k
        E = gmpy2.mul_2exp(P, k) + (pow2k - one)
    t = -(E / (E + two))
    return t if z >= 0 else -t


def _is_pbit(x: float, p: int) -> bool:
    return x == q(x, p)


def _geometry(p, N, halo, lam):
    """h = 2 / N, c_j = -1 + j h, gamma = lambda / h (inside the p context)."""
    one, two = mpfr(1), mpfr(2)
    h = two / mpfr(N)  # mpfr(N) rounds the integer N to p bits
    centers = [-one + (mpfr(j) * h) for j in range(-halo, N + halo + 1)]
    return centers, mpfr(float(lam)) / h


def features(p, x, N, halo, lam) -> np.ndarray:
    """phi_ij = tanh_p(gamma (x_i - c_j)); x and lam must already be p-bit numbers."""
    x = np.asarray(x, dtype=np.float64)
    if not all(_is_pbit(float(v), p) for v in x) or not _is_pbit(float(lam), p):
        raise ValueError("inputs are not already p-bit")
    C = _constants_mpfr(p)
    with _ctx(p):
        centers, gam = _geometry(p, N, halo, lam)
        return np.array([[float(_tanh_p(gam * (mpfr(float(v)) - c), C)) for c in centers] for v in x])


def forward(p, x_eval, N, halo, lam, weights) -> np.ndarray:
    """s = w_bias; s = s + phi_j(x) w_j, j = 0..W-1, for each x (all p-bit)."""
    x_eval = np.asarray(x_eval, dtype=np.float64)
    w = [float(v) for v in weights]
    if not all(_is_pbit(v, p) for v in [*x_eval.tolist(), *w, float(lam)]):
        raise ValueError("inputs are not already p-bit")
    C = _constants_mpfr(p)
    with _ctx(p):
        centers, gam = _geometry(p, N, halo, lam)
        wm = [mpfr(v) for v in w]
        out = []
        for v in x_eval:
            xv = mpfr(float(v))
            s = wm[-1]
            for j, c in enumerate(centers):
                s = s + _tanh_p(gam * (xv - c), C) * wm[j]
            out.append(float(s))
        return np.array(out)


def tanh_p(z, p: int) -> float:
    """Convenience: the spec's tanh_p on a p-bit input z (float), returned as float."""
    C = _constants_mpfr(p)
    with _ctx(p):
        return float(_tanh_p(mpfr(float(z)), C))
