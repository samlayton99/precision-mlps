"""Independent algebra and exact-network checks for the target-gap proof."""
from fractions import Fraction
import math

import numpy as np
import pytest

from experiments.expD34_readout_race import population_target_gap as audit


@pytest.mark.parametrize("power, expected", [
    (5, ([16, Fraction(-20, 3), Fraction(-28, 3), -7], [-7, Fraction(-14, 3), Fraction(8, 3), 0])),
    (7, ([-272, 224, 216, 124, 53], [53, -18, -68, 8, 0]))])
def test_exact_derivative_remainder_constants(power, expected):
    # Independently differentiate tanh via d/du = (1-t^2)d/dt.
    coefficients = [0, 1]
    for _ in range(power):
        derivative = [i*coefficients[i] for i in range(1, len(coefficients))]
        coefficients = derivative+[0, 0]
        for i, value in enumerate(derivative):
            coefficients[i+2] -= value
    assert all(v == 0 for v in coefficients[1::2])
    polynomial = coefficients[::2]
    n = len(polynomial)-1
    bernstein = [sum(Fraction(polynomial[k]*math.comb(j, k), math.comb(n, k))
                     for k in range(j+1)) for j in range(n+1)]
    left, right = [bernstein[0]], [bernstein[-1]]
    while len(bernstein) > 1:
        bernstein = [(a+b)/2 for a, b in zip(bernstein, bernstein[1:])]
        left.append(bernstein[0]); right.append(bernstein[-1])
    assert (left, right[::-1]) == expected


@pytest.mark.parametrize("degree", [3, 5])
def test_bounds_on_exact_networks_with_concentration_and_large_arguments(degree):
    rng = np.random.default_rng(46+degree)
    x = np.linspace(-1, 1, 257)
    basis, _ = np.linalg.qr(np.polynomial.legendre.legvander(x, 9))
    g = math.sqrt(len(x))*(.6*basis[:, degree+1]+.8*basis[:, 9])
    for width in (1, 7, 31):
        for scale in (.02, .5, 5):
            p = rng.normal(size=(3, width))*scale
            p[:, 0] *= 3  # No per-neuron confinement in the asserted inequalities.
            a, b, c = p
            f = np.tanh(x[:, None]*a+b)@c
            z = f-basis[:, :2]@(basis[:, :2].T@f)
            q = (width*np.sum(np.sum(p*p, axis=0)**2))**.25
            bound = audit.coupling(q, width, degree, 0, 1)[0]
            assert np.linalg.norm(z)/math.sqrt(len(x)) <= audit.A3*q**4/width+1e-12
            assert abs(np.mean(f*g)) <= bound+1e-12
            assert .5*np.mean((f-g)**2) >= .5-bound-1e-12


def test_target_leakage_detects_missing_and_present_low_modes():
    for name, degree, expect_gap in (("moment9", 5, True), ("moment5", 3, True),
                                      ("moment5", 5, False), ("sine", 5, False)):
        split = audit.target_split(*audit.target_data(name), degree)
        assert (split["leakage"] < 1e-12) == expect_gap
        assert math.isclose(split["leakage"]**2+split["high_norm"]**2,
                            split["target_fine_norm"]**2, abs_tol=1e-13)


def test_width_rates_and_discrete_guard_are_not_flow_unit_relabeling():
    split = dict(target_norm=1., target_fine_norm=1., leakage=0., high_norm=1.)
    for degree, exponent in ((3, 1), (5, 1.5)):
        times = []
        for width in (100, 400):
            state = dict(width=width, eta=.002, M4=1., loss=.5, fine_norm=1.)
            full = audit.candidate(state, split, degree, 1.3, "full_flow")
            gd = audit.candidate(state, split, degree, 1.3, "gd")
            assert gd["flow_time"] < full["flow_time"]
            assert gd["guaranteed_updates"]*.002 < gd["flow_time"]
            times.append(full["flow_time"])
        assert math.isclose(times[1]/times[0], 4**exponent, rel_tol=1e-12)
    state["eta"] = 100.
    assert not audit.candidate(state, split, 5, 1.3, "gd")["valid"]


def test_separate_moments_propagate_under_collective_displacement():
    rng = np.random.default_rng(37)
    p0 = rng.normal(size=(3, 50))*.02
    increment = rng.normal(size=p0.shape)*.03
    radius = np.linalg.norm(increment)
    physical = lambda p, k: np.sum(np.linalg.norm(p, axis=0)**k)**(1/k)
    for k in (2, 4, 6, 8):
        assert physical(p0+increment, k) <= physical(p0, k)+radius
    width = p0.shape[1]
    state = dict(width=width, eta=.0001, loss=.5, fine_norm=1.,
                 M=physical(p0, 2)**2,
                 M4=width*physical(p0, 4)**4,
                 M6=width**2*physical(p0, 6)**6,
                 M8=width**3*physical(p0, 8)**8)
    split = dict(target_norm=1., target_fine_norm=1., leakage=0., high_norm=1.)
    result = audit.candidate(state, split, 5, .2, "gd", "separate_moments")
    outer = 2*result["radius"]
    assert math.isclose(result["available_loss"],
                        audit.A7*(physical(p0, 8)+outer)**8, rel_tol=1e-13)
    assert result["valid"]
