"""Mathematical checks of the FP64 Mhaskar specialization and rounding."""
import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval

from experiments.expC12_mhaskar_comparison.construction import (
    TanhNetwork, chebyshev_coefficients, construct_mhaskar,
    normalized_difference_weights, round_bits, tanh_taylor,
)


def test_derivatives_and_local_taylor_series():
    b = np.log(2.)/2
    t = np.tanh(b)
    c = tanh_taylor(b, 16)
    np.testing.assert_allclose(c[:4], [t, 1-t*t, -t*(1-t*t),
                                     (-2+8*t*t-6*t**4)/6], rtol=1e-14)
    z = np.array([-.15, -.03, .04, .12])
    np.testing.assert_allclose(np.polynomial.polynomial.polyval(z, c),
                               np.tanh(b+z), atol=3e-16, rtol=2e-15)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_stencil_moments_normalization_and_lower_order_cancellation(order):
    h = .3
    c = tanh_taylor(np.log(2.)/2, order)
    weights = normalized_difference_weights(order, h, c[order])
    slopes = (np.arange(order+1)-order/2)*h
    for k in range(order):
        assert abs(np.dot(weights, slopes**k)) < 1e-9
    np.testing.assert_allclose(np.dot(weights, slopes**order)*c[order], 1., rtol=3e-14)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_chebyshev_network_second_order_step_error(degree):
    coefficients = np.zeros(degree+1)
    coefficients[-1] = 1
    x = np.linspace(-1, 1, 101)
    truth = chebval(x, coefficients)
    errors = []
    for h in [.04, .02]:
        model = construct_mhaskar(coefficients, h)
        assert model.width <= 2*degree+1
        errors.append(np.max(abs(model.evaluate(x)-truth)))
    assert 3.5 < errors[0]/errors[1] < 4.5


def test_projection_recovers_known_chebyshev_polynomial():
    expected = np.array([.4, -.2, .7, 0., -.3])
    coefficients = chebyshev_coefficients(lambda x: chebval(x, expected), 64)
    np.testing.assert_allclose(coefficients[:5], expected, atol=4e-16)
    assert np.max(abs(coefficients[5:])) < 3e-16


def test_merged_network_matches_separate_monomial_stencils():
    from numpy.polynomial.chebyshev import cheb2poly
    coefficients = np.array([.2, -.3, .5, .1])
    b, h = np.log(2.)/2, .2
    c = tanh_taylor(b, 3)
    x = np.linspace(-1, 1, 71)
    result = np.zeros_like(x)
    for r, a in enumerate(cheb2poly(coefficients)):
        slopes = (np.arange(r+1)-r/2)*h
        result += a * (np.tanh(x[:, None]*slopes+b)
                       @ normalized_difference_weights(r, h, c[r]))
    np.testing.assert_allclose(construct_mhaskar(coefficients, h).evaluate(x), result,
                               rtol=2e-10, atol=2e-10)


def test_parameter_and_output_rounding_protocol():
    model = TanhNetwork(np.array([.27, 2.31]), np.array([.11, -.72]),
                        np.array([1.81, -.93]), np.float64(.38))
    x = np.linspace(-1, 1, 37)
    q = model.quantized(8)
    for a in (q.slope, q.bias, q.readout, q.offset):
        np.testing.assert_array_equal(a, round_bits(a, 8))
    expected = round_bits(np.tanh(x[:, None]*q.slope+q.bias)@q.readout+q.offset, 8)
    np.testing.assert_array_equal(model.evaluate(x, 8), expected)
    assert not np.array_equal(expected, round_bits(model.evaluate(x, 53), 8))


def test_rounding_ties_and_fp64_identity():
    x = np.array([1+2**-8, 1+3*2**-8, -1-2**-8, 0.])
    np.testing.assert_array_equal(round_bits(x, 8), [1., 1+2**-6, -1., 0.])
    np.testing.assert_array_equal(round_bits(x, 53), x)
