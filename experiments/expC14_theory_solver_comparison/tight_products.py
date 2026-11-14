"""Enclose small matrix-product residuals without a loose gamma_m*n floor.

All *input* arrays are ordinary FP64. Diagnostics use five fixed-point
chunks and error-free TwoSum transformations; they never change a solve.

Assumptions: IEEE round-to-nearest FP64 with gradual underflow, conventional
FP64 GEMM (scalar products/additions or FMAs, no lower-precision products),
finite inputs, and no overflow. The supported exponent checks below ensure
the exact digit GEMMs and their power-of-two reweightings stay normal.

For normalized |x|<1, each chunk is an integer multiple of 2^-q with
absolute value <1. A digit dot product is an integer multiple of 2^-2q;
if 2q+ceil(log2(inner))<=52, EVERY partial sum fits exactly in FP64.
Thus its GEMM is exact independent of the summation order. The only
remaining errors are decomposition tails and reconstruction additions.

For reconstruction, TwoSum gives hi_new+err = hi_old+term exactly. The
rounded accumulator lo approximates the sum of these err matrices. Its
Frobenius error is bounded by gamma_t times the SUM of their Frobenius
norms. Subtracting the requested target before final rounding preserves
small residuals instead of paying u*||A B|| for the rounded product.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import gmpy2 as gp
import numpy as np

_PREC = 256
_UQ = gp.mpq(1, 1 << 53)
_MUQ = gp.mpq(1, 1 << 1074)


def _up(value):
    """Smallest adjacent-double enclosure of an already bounded scalar."""
    if value < 0:
        raise ValueError('A nonnegative upper bound was expected')
    ans = float(value)
    if not math.isfinite(ans):
        raise OverflowError('Upper bound cannot be represented in FP64')
    if gp.mpfr(ans) < value:
        ans = float(np.nextafter(ans, np.inf))
    return ans


def _sum_up(*values):
    with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
        return _up(sum(gp.mpfr(v) for v in values))


def _gamma(steps):
    if steps >= (1 << 52):
        raise ValueError('Operation-count rounding allowance is too large')
    return gp.mpq(steps, (1 << 53)-steps)


def _beta(steps):
    return steps * _MUQ / (1-steps*_UQ)


def _as_matrix(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.all(np.isfinite(a)):
        raise ValueError('Expected a finite two-dimensional FP64 array')
    return a


def frobenius_upper(a):
    """Upward Frobenius bound including scaling, dot, and sqrt rounding.

    The ordinary positive FP64 sum is enclosed analytically; unlike a lone
    nextafter(norm(a)), this accounts for all reduction operations. Scaling
    by maxabs avoids overflow and makes underflow allowances explicit.
    """
    a = _as_matrix(a)
    if not a.size:
        return 0.0
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return 0.0
    y = a/scale
    squared_sum = float(np.sum(y*y))
    steps = 2*a.size+2
    with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundDown):
        denominator = gp.mpfr(1-_gamma(steps))
    with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
        # ||a/scale|| <= (||fl(a/scale)||+sqrt(size)*mu)/(1-u).
        ynorm = gp.sqrt((gp.mpfr(squared_sum)+gp.mpfr(_beta(steps)))/denominator)
        exact_scaled = (ynorm+gp.sqrt(gp.mpfr(a.size))*gp.mpfr(_MUQ))/gp.mpfr(1-_UQ)
        return _up(gp.mpfr(scale)*exact_scaled)


def _normalize(a):
    maximum = float(np.max(np.abs(a))) if a.size else 0.0
    exponent = math.frexp(maximum)[1] if maximum else 0
    normalized = np.ldexp(a, -exponent)
    if not np.array_equal(np.ldexp(normalized, exponent), a):
        raise ValueError('Input exponent range is too broad for exact normalization')
    return normalized, exponent


@dataclass
class _Split:
    chunks: list[np.ndarray]
    remainder: np.ndarray
    exponent: int
    chunk_bits: int

    def tail_norm(self):
        with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
            return _up(gp.mul_2exp(gp.mpfr(frobenius_upper(self.remainder)),
                                  self.exponent-self.chunk_bits*len(self.chunks)))


def _split(a, digits, chunk_bits):
    remainder, exponent = _normalize(a)
    chunks = []
    for _ in range(digits):
        digit = np.ldexp(np.trunc(np.ldexp(remainder, chunk_bits)), -chunk_bits)
        # Exact subtraction: digit is zero or within a factor two of remainder.
        remainder = np.ldexp(remainder-digit, chunk_bits)
        chunks.append(digit)
    return _Split(chunks, remainder, exponent, chunk_bits)


def _two_sum(a, b):
    s = a+b
    v = s-a
    e = (a-(s-v))+(b-v)
    return s, e


def _transpose_alias(a, b):
    return (a.shape == b.T.shape and a.strides == b.T.strides and
            a.__array_interface__['data'][0] == b.__array_interface__['data'][0])


def _residual_block(a, b, target, split_a, split_b, bnorm, btail):
    rows, inner = a.shape
    columns = b.shape[1]
    q = split_a.chunk_bits
    count = len(split_a.chunks)
    exponent = split_a.exponent+split_b.exponent
    if not -500 <= exponent <= 500:
        raise ValueError('Combined exponent outside supported [-500,500] range')
    hi = np.zeros((rows, columns))
    lo = np.zeros_like(hi)
    error_norm_sum = 0.0
    terms = 0
    # Largest contributions first; no condition is needed by TwoSum itself.
    for degree in range(2*count-1):
        for i in range(count):
            j = degree-i
            if not 0 <= j < count:
                continue
            term = split_a.chunks[i] @ split_b.chunks[j]
            # Exact digit GEMM, then exact power-of-two scaling.
            term = np.ldexp(term, exponent-q*degree)
            hi, error = _two_sum(hi, term)
            lo = lo+error
            error_norm_sum = _sum_up(error_norm_sum, frobenius_upper(error))
            terms += 1
    if target is None:
        value, last_error = _two_sum(hi, lo)
        final_error = frobenius_upper(last_error)
    else:
        difference, e1 = _two_sum(hi, -target)
        value, e2 = _two_sum(difference, lo)
        final_error = _sum_up(frobenius_upper(e1), frobenius_upper(e2))
    with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
        reconstruction_error = (gp.mpfr(_gamma(terms))*error_norm_sum +
                                gp.sqrt(gp.mpfr(rows*columns))*gp.mpfr(_beta(terms)))
        ta, tb = split_a.tail_norm(), btail
        anorm = frobenius_upper(a)
        # A B - A_trunc B_trunc = R_A B + A_trunc R_B.
        tail_error = gp.mpfr(ta)*bnorm+(gp.mpfr(anorm)+ta)*tb
        radius = _up(reconstruction_error+tail_error+final_error)
    return value, radius, _up(tail_error), _up(reconstruction_error)


@dataclass
class ProductEnclosure:
    """value encloses A@B-target with a Frobenius error radius."""
    value: np.ndarray
    error_frobenius: float
    norm_upper: float
    tail_error_frobenius: float
    reconstruction_error_frobenius: float

    def spectral_upper(self, power_steps=2):
        return _sum_up(spectral_upper(self.value, power_steps), self.error_frobenius)


def enclosed_product(a, b, target=None, *, digits=5, chunk_bits=16, block_rows=256):
    """Return a tight enclosure of the exact product of stored FP64 arrays.

    Passing target into this function is important: cancellation occurs in
    the high/low expansion before the residual is rounded. For a Gram defect
    use enclosed_product(U.T,U,np.eye(n)); for projection error pass the
    computed projection as target. Inputs remain unchanged.
    """
    a, b = _as_matrix(a), _as_matrix(b)
    if a.shape[1] != b.shape[0]:
        raise ValueError('Incompatible matrix dimensions')
    if target is not None:
        target = _as_matrix(target)
        if target.shape != (a.shape[0], b.shape[1]):
            raise ValueError('Target shape differs from product shape')
    if not 1 <= digits <= 5 or not 1 <= chunk_bits <= 16:
        raise ValueError('Supported: 1..5 chunks of 1..16 bits')
    if block_rows < 1 or not a.shape[1]:
        raise ValueError('A positive block size and nonempty inner dimension are required')
    inner_bits = (a.shape[1]-1).bit_length()
    if 2*chunk_bits+inner_bits > 52:
        raise ValueError('Digit dot products would not be exact in FP64')
    split_b = _split(b, digits, chunk_bits)
    bnorm = frobenius_upper(b)
    btail = split_b.tail_norm()
    transpose = _transpose_alias(a, b)
    value = np.empty((a.shape[0], b.shape[1]))
    radius_squared = gp.mpfr(0)
    tail_squared = gp.mpfr(0)
    reconstruction_squared = gp.mpfr(0)
    for lo in range(0, a.shape[0], block_rows):
        hi = min(lo+block_rows, a.shape[0])
        if transpose:
            split_a = _Split([d[:, lo:hi].T for d in split_b.chunks],
                             split_b.remainder[:, lo:hi].T, split_b.exponent, chunk_bits)
        else:
            split_a = _split(a[lo:hi], digits, chunk_bits)
        block, radius, tail, reconstruction = _residual_block(
            a[lo:hi], b, None if target is None else target[lo:hi], split_a, split_b, bnorm, btail)
        value[lo:hi] = block
        with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
            radius_squared += gp.mpfr(radius)**2
            tail_squared += gp.mpfr(tail)**2
            reconstruction_squared += gp.mpfr(reconstruction)**2
    with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
        radius = _up(gp.sqrt(radius_squared))
        norm_bound = _up(gp.mpfr(frobenius_upper(value))+radius)
        tail = _up(gp.sqrt(tail_squared))
        reconstruction = _up(gp.sqrt(reconstruction_squared))
    return ProductEnclosure(value, radius, norm_bound, tail, reconstruction)


def spectral_upper(a, power_steps=2):
    """Valid spectral-norm upper bound from repeated Gram/Frobenius bounds.

    ||X||_2^2=||X.T X||_2. Each ordinary GEMM is enclosed by the standard
    gamma_(2m) Frobenius product bound; it need not itself be high precision.
    Repeated normalization keeps the small residual matrices away from
    underflow. power_steps=0 returns a Frobenius bound.
    """
    a = _as_matrix(a)
    if power_steps < 0 or power_steps > 4:
        raise ValueError('Choose zero through four Gram steps')
    if not a.size or not np.any(a):
        return 0.0
    if power_steps == 0:
        return frobenius_upper(a)
    try:
        normalized, exponent = _normalize(a)
    except ValueError:
        # A tiny entry can be lost when a larger entry requires downward
        # power-of-two scaling. Do not discard it: the Frobenius enclosure
        # already handles arbitrary exponent ranges and is a valid spectral
        # upper bound. This fallback concerns norm estimation only; the
        # exact-digit product normalization remains strict.
        return frobenius_upper(a)
    # Use the smaller Gram matrix; both identities give the same spectral norm.
    if normalized.shape[0] < normalized.shape[1]:
        normalized = normalized.T
    m, n = normalized.shape
    gram = normalized.T @ normalized
    f = frobenius_upper(normalized)
    gram_norm = spectral_upper(gram, power_steps-1)
    with gp.context(gp.get_context(), precision=_PREC, round=gp.RoundUp):
        product_error = gp.mpfr(_gamma(2*m+2))*gp.mpfr(f)**2+n*gp.mpfr(_beta(2*m+2))
        answer = gp.sqrt(gp.mpfr(gram_norm)+product_error)
        return _up(gp.mul_2exp(answer, exponent))
