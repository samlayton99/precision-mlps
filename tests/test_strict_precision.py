"""Independent high-precision oracles for the p-bit inference kernels."""
import numpy as np
import pytest

g = pytest.importorskip("gmpy2", reason="Install requirements-strict.txt for MPFR inference checks")

from experiments.expC09_bandwidth_figures.strict_precision import features, readout


@pytest.mark.parametrize("p", [8, 11, 24, 32, 40, 53])
def test_features_against_200_bit_operation_oracle(p):
    x = np.array([-1., -.123456789, 0., .333333333, 1.])
    centers = np.array([-1.25, -.2, 0., .4, 1.25])
    gamma = 37.123456789
    with g.context(precision=200, round=g.RoundToNearest):
        q = lambda v: g.mpfr(v, precision=p)
        expected = np.array([[float(q(g.tanh(q(q(gamma)*q(q(a)-q(c))))))
                              for c in centers] for a in x])
    np.testing.assert_array_equal(features(x, centers, gamma, p), expected)


@pytest.mark.parametrize("p", [8, 24, 40, 52, 53])
def test_readout_against_200_bit_sequential_oracle(p):
    rng = np.random.default_rng(57)
    phi, weights = rng.uniform(-1, 1, (7, 23)), rng.normal(size=23)
    bias = .123456789
    expected = []
    with g.context(precision=200, round=g.RoundToNearest):
        q = lambda v: g.mpfr(v, precision=p)
        for row in phi:
            total = q(bias)
            for a, b in zip(row, weights):
                total = q(total+q(q(a)*q(b)))
            expected.append(float(total))
    np.testing.assert_array_equal(readout(phi, weights, bias, p), expected)


def test_fp32_readout_matches_native_separate_products_and_additions():
    rng = np.random.default_rng(31)
    phi = rng.uniform(-1, 1, (17, 1024)).astype(np.float32)
    weights = rng.normal(size=1024).astype(np.float32)
    bias = np.float32(.19)
    expected = np.full(len(phi), bias, dtype=np.float32)
    for j in range(phi.shape[1]):
        product = phi[:, j]*weights[j]
        expected = expected+product
    np.testing.assert_array_equal(readout(phi, weights, float(bias), 24), expected.astype(float))


def test_avoids_fp64_intermediate_double_rounding():
    # True product is just below a p=52 midpoint. Rounding through FP64 first
    # lands on the midpoint and incorrectly rounds upward on the second step.
    a, b = 1+2.**-26, 1+2.**-26-2.**-51
    with g.context(precision=200):
        expected = float(g.mpfr(g.mpfr(a)*g.mpfr(b), precision=52))
        rounded_through_fp64 = float(g.mpfr(a*b, precision=52))
    assert expected != rounded_through_fp64
    assert readout(np.array([[a]]), np.array([b]), 0., 52)[0] == expected
