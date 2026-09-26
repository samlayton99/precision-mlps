"""FP64 specialization of Mhaskar (1996), Lemma 3.2.

Chebyshev truncation -> monomials -> normalized centered tanh differences.
No fitted readout, arbitrary precision, or high-precision intermediates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.chebyshev import cheb2poly
from scipy.fft import dct
from scipy.special import gammaln


def round_bits(values, bits):
    """Significand-only round-to-nearest/even; retain FP64 exponent range."""
    a = np.asarray(values, dtype=np.float64)
    if bits == 53:
        return a.copy()
    if not 2 <= bits < 53:
        raise ValueError("bits must lie in [2, 53]")
    mantissa, exponent = np.frexp(a)
    return np.ldexp(np.rint(np.ldexp(mantissa, bits)), exponent - bits)


@dataclass
class TanhNetwork:
    slope: np.ndarray
    bias: np.ndarray
    readout: np.ndarray
    offset: np.float64

    def __post_init__(self):
        self.slope = np.asarray(self.slope, dtype=np.float64)
        self.bias = np.asarray(self.bias, dtype=np.float64)
        self.readout = np.asarray(self.readout, dtype=np.float64)
        self.offset = np.float64(self.offset)
        if not (self.slope.shape == self.bias.shape == self.readout.shape):
            raise ValueError("Parameter shapes differ")
        if not all(np.all(np.isfinite(a)) for a in
                   (self.slope, self.bias, self.readout, self.offset)):
            raise FloatingPointError("Nonfinite FP64 network parameter")

    @property
    def width(self):
        return self.readout.size

    def quantized(self, bits):
        return TanhNetwork(*(round_bits(a, bits) for a in
                             (self.slope, self.bias, self.readout, self.offset)))

    def evaluate(self, x, bits=53):
        """FP64 evaluation of p-bit parameters, followed by output rounding."""
        model = self.quantized(bits)
        x = np.asarray(x, dtype=np.float64)
        features = np.tanh(x[:, None] * model.slope + model.bias)
        with np.errstate(over="ignore", invalid="ignore"):
            output = features @ model.readout + model.offset
        return round_bits(output, bits)

    def save(self, path):
        np.savez(path, slope=self.slope, bias=self.bias,
                 readout=self.readout, offset=self.offset)


def chirp(x):
    return np.sin(8.0 * np.pi * (np.asarray(x) + 1.0)**2)


def chebyshev_coefficients(target, points):
    """Discrete Chebyshev projection on first-kind nodes using an FP64 DCT."""
    theta = np.pi * (np.arange(points, dtype=np.float64) + .5) / points
    coefficients = dct(target(np.cos(theta)), type=2) / points
    coefficients[0] *= .5
    return coefficients


def tanh_taylor(bias, degree):
    """Return c_r=tanh^(r)(bias)/r!, using y'=1-y^2.

    Keeping Taylor coefficients scaled avoids forming overflowing factorials.
    The recurrence is evaluated entirely in FP64, including its convolution.
    """
    c = np.zeros(degree + 1, dtype=np.float64)
    c[0] = np.tanh(np.float64(bias))
    for r in range(degree):
        c[r+1] = ((1.0 if r == 0 else 0.0) - np.dot(c[:r+1], c[r::-1])) / (r+1)
    return c


def normalized_difference_weights(order, step, taylor_coefficient):
    """binom(r,j)/(h^r r! c_r), with alternating signs.

    Log scaling avoids overflow of factorials that cancel algebraically.
    This does not supply any extra precision to the resulting coefficients.
    """
    j = np.arange(order + 1)
    if step <= 0 or taylor_coefficient == 0:
        raise ValueError("Positive step and nonzero activation derivative required")
    logabs = (-order*np.log(step) - np.log(abs(taylor_coefficient))
              - gammaln(j+1.0) - gammaln(order-j+1.0))
    signs = np.where((order-j) % 2, -1.0, 1.0) * np.sign(taylor_coefficient)
    with np.errstate(over="ignore", under="ignore"):
        return signs * np.exp(logabs)


def construct_mhaskar(cheb_coefficients, step, bias=np.log(2.0)/2.0):
    """Merge all monomial stencils into a single affine tanh network.

    Slopes are integer multiples of h/2. For degree d the union has at most
    2d+1 neurons. Constant-slope neurons remain explicit (no hidden factoring
    or unquantized per-stencil scale factors are retained during evaluation).
    """
    coefficients = np.asarray(cheb_coefficients, dtype=np.float64)
    degree = len(coefficients) - 1
    monomial = cheb2poly(coefficients)
    taylor = tanh_taylor(bias, degree)
    readout = np.zeros(2*degree + 1, dtype=np.float64)
    for r, a in enumerate(monomial):
        if a == 0:
            continue
        j = np.arange(r + 1)
        if taylor[r] == 0:
            raise FloatingPointError("Derivative rounded to zero")
        # Direct products are more accurate than exponentiating log weights.
        # Use logarithmic scaling only when intermediates exceed FP64 range.
        terms = np.empty(r+1, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            edge = a/taylor[r]
            for k in range(1, r+1):
                edge /= step*k
            terms[0] = edge * (-1.0 if r % 2 else 1.0)
            for k in range(r):
                terms[k+1] = -terms[k]*((r-k)/(k+1))
            if not np.all(np.isfinite(terms)) or edge == 0:
                logabs = (np.log(abs(a)) - r*np.log(step) - np.log(abs(taylor[r]))
                          - gammaln(j+1.0) - gammaln(r-j+1.0))
                sign = np.where((r-j) % 2, -1.0, 1.0) * np.sign(a) * np.sign(taylor[r])
                terms = sign*np.exp(logabs)
            readout[2*j-r+degree] += terms
    active = readout != 0
    slopes = .5 * step * np.arange(-degree, degree+1, dtype=np.float64)
    return TanhNetwork(slopes[active], np.full(active.sum(), bias),
                       readout[active], np.float64(0.0))
