"""Minimal vectorized double-double arithmetic (Dekker/Knuth) on numpy arrays.

A dd value is a pair (hi, lo) of float64 arrays with |lo| <= ulp(hi)/2.
Gives ~32 decimal digits, eps_dd ~ 1e-32, at ~10-20x fp64 cost and exactly
2x the memory -- still O(d).
"""
import numpy as np

_SPL = 2.0**27 + 1.0


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def two_prod(a, b):
    p = a * b
    ah = (_SPL * a) - ((_SPL * a) - a); al = a - ah
    bh = (_SPL * b) - ((_SPL * b) - b); bl = b - bh
    return p, (((ah * bh - p) + ah * bl) + al * bh) + al * bl


def dd(x):
    return (np.asarray(x, dtype=float), np.zeros_like(np.asarray(x, dtype=float)))


def add(A, B):
    s, e = two_sum(A[0], B[0])
    e = e + (A[1] + B[1])
    return quick_two_sum(s, e)


def sub(A, B):
    return add(A, (-B[0], -B[1]))


def mul(A, B):
    p, e = two_prod(A[0], B[0])
    e = e + (A[0] * B[1] + A[1] * B[0])
    return quick_two_sum(p, e)


def scale(A, s):
    """dd array times a dd scalar (s is a 2-tuple of python floats)."""
    p, e = two_prod(A[0], s[0])
    e = e + (A[0] * s[1] + A[1] * s[0])
    return quick_two_sum(p, e)


def _csum0(H, L):
    """Compensated sum over axis 0 of dd arrays H,L -> dd of shape H.shape[1:]."""
    sh = np.zeros(H.shape[1:]); sl = np.zeros(H.shape[1:])
    for i in range(H.shape[0]):
        sh, e = two_sum(sh, H[i])
        sl = sl + e + L[i]
        sh, sl = quick_two_sum(sh, sl)
    return sh, sl


def dsum(A):
    """Sum a dd array to a dd scalar, compensated."""
    h, l = _csum0(A[0][:, None], A[1][:, None])
    return (float(h[0]), float(l[0]))


def dot(A, B):
    return dsum(mul(A, B))


def matvec(M, X):
    """M: fp64 (n,d).  X: dd length d.  -> dd length n."""
    n = M.shape[0]
    acc = (np.zeros(n), np.zeros(n))
    for k in range(M.shape[1]):
        p, e = two_prod(M[:, k], X[0][k])
        e = e + M[:, k] * X[1][k]
        acc = add(acc, (p, e))
    return acc


def rmatvec(M, X):
    """M: fp64 (n,d).  X: dd length n.  -> dd length d.  Vectorized: the
    reduction runs over ROWS with numpy ops on d-length arrays."""
    P, E = two_prod(M, X[0][:, None])
    E = E + M * X[1][:, None]
    return _csum0(P, E)


def ssqrt(a):
    """dd scalar square root."""
    if a[0] <= 0.0:
        return (0.0, 0.0)
    h = np.sqrt(a[0])
    p, e = two_prod(h, h)
    r = ((a[0] - p) - e) + a[1]
    return quick_two_sum(h, r / (2.0 * h))


def norm(A):
    return ssqrt(dot(A, A))


def sdiv(a, b):
    """dd scalar / dd scalar."""
    q = a[0] / b[0]
    p, e = two_prod(q, b[0])
    r = ((a[0] - p) - e) + a[1] - q * b[1]
    return quick_two_sum(q, r / b[0])


def sadd(a, b):
    s, e = two_sum(a[0], b[0]); e += a[1] + b[1]
    return quick_two_sum(s, e)


def smul(a, b):
    p, e = two_prod(a[0], b[0]); e += a[0] * b[1] + a[1] * b[0]
    return quick_two_sum(p, e)
