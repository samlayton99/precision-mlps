"""Scalar theorem-evaluation checks; run with the campaign's remote tests."""
import math

import pytest
from scipy.integrate import quad

from experiments.expD34_readout_race.population_sensitivity_certificate import (
    initial_sensitivity_certificate, sensitivity_time_integral,
)


@pytest.mark.parametrize('ratio,factor', [(.01, 1.2), (.3, 1.9), (2.9, 1.5)])
def test_regularized_integral_matches_direct_reference(ratio, factor):
    result, _ = sensitivity_time_integral(ratio, factor)
    reference, _ = quad(lambda u: 1 / (ratio + 2 * math.sqrt(2) * (u**3 - 1)),
                        1., factor, epsabs=1e-11, epsrel=1e-11)
    assert result == pytest.approx(reference, rel=1e-10, abs=1e-12)


def test_small_sensitivity_layer_has_logarithmic_delay():
    first, _ = sensitivity_time_integral(1e-20, 1.9)
    second, _ = sensitivity_time_integral(1e-40, 1.9)
    assert second - first == pytest.approx(20 * math.log(10) / (6 * math.sqrt(2)),
                                           rel=1e-10)


def state(ratio):
    B = .05
    return dict(B=B, fine_norm=.2, fine_jacobian_hs_norm=ratio * B**3,
                resolved=True, M=1., Es=.5, coarse_kappa=.1,
                input_variance=1/3, target_norm=.5)


def test_weaker_initial_sensitivity_extends_evolving_envelope():
    weak = initial_sensitivity_certificate(state(.03))
    strong = initial_sensitivity_certificate(state(3.))
    assert weak['hs_certificate_factor'] == strong['hs_certificate_factor']
    assert weak['hs_certificate_flow_time'] > strong['hs_certificate_flow_time']
    assert weak['hs_certificate_relative_fine_floor'] > strong['hs_certificate_relative_fine_floor']
    q = strong['hs_certificate_factor']
    basic_time = (1-q**-2) / (6 * .2 * .05**2)
    assert strong['hs_certificate_flow_time'] >= basic_time


def test_zero_sensitivity_is_stationary_not_quadrature_failure():
    result = initial_sensitivity_certificate(state(0.))
    assert result['hs_certificate_status'] == 'stationary_effective_flow'
    assert result['hs_certificate_relative_fine_floor'] == 1.


def test_large_ratio_preserves_small_positive_integral():
    result, _ = sensitivity_time_integral(1e20, 1.25)
    assert result == pytest.approx(.25 / 1e20, rel=1e-12, abs=0.)


@pytest.mark.parametrize('ratio,factor', [(math.nan, 1.2), (math.inf, 1.2),
                                         (.1, math.nan), (.1, math.inf)])
def test_nonfinite_quadrature_inputs_are_rejected(ratio, factor):
    with pytest.raises(ValueError):
        sensitivity_time_integral(ratio, factor)


def test_nonfinite_initial_data_is_not_certified():
    row = state(.03)
    row['M'] = math.nan
    assert initial_sensitivity_certificate(row)['hs_certificate_status'] == 'invalid_initial_data'
