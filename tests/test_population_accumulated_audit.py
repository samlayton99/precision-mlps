"""Formula and quadrature checks independent of the archived training outcomes."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from experiments.expD34_readout_race.population_accumulated_audit import (
    budgets, crossing_time, cumulative, envelope,
)


def test_irregular_physical_time_quadrature_and_crossing():
    t = np.array([0., .2, 2.])
    speed = np.array([2., 2., 4.])
    assert cumulative(t, speed) == pytest.approx([0., .4, 5.8])
    budget_at_one = .4 + 2 * .8 + (.8**2 / 1.8)
    assert crossing_time(t, speed, budget_at_one) == pytest.approx(1.)
    assert crossing_time(t, speed, .4) == pytest.approx(.2)
    assert crossing_time(t, speed, 0.) == 0.
    assert crossing_time(t, speed, 6.) is None
    # A decaying speed that reaches zero exercises the stable quadratic root.
    assert crossing_time([0., 2.], [4., 0.], 3.) == pytest.approx(1.)


def test_constant_concentration_recovers_pointwise_formula_and_energy_integral():
    W, m4, y0, I, time = 705, 40., .8, 16., 1.3
    clock = math.sqrt(I) * time
    result = envelope(W, m4, y0, 1., .9, [0., clock], [0., time], 2 / 512)
    k = 6 * y0 * math.sqrt(I * m4) / W
    assert result["M4_envelope"][-1] == pytest.approx(m4 / (1 - k * time)**2)
    # Independently integrate the squared instantaneous force envelope.
    reference = quad(lambda t: (3 * y0 * I**.25 * (m4 / (1 - k * t)**2)**.75 / W)**2,
                     0., time, epsabs=1e-13)[0]
    assert result["dissipation_envelope"][-1] == pytest.approx(reference, rel=1e-12)
    assert result["total_travel_envelope"][-1] == pytest.approx(math.sqrt(time * reference))
    assert result["weighted_travel_envelope"][0] == 0.
    assert result["dissipation_envelope"][0] == 0.


def test_equal_accumulated_concentration_allows_a_brief_high_spike():
    # Equal clock integrals: constant sqrt(I)=2 versus baseline 1 and a
    # triangular excursion reaching 20. These are quadrature fixtures,
    # not an assertion that a network realizes either prescribed history.
    baseline = cumulative([0., 1.], [2., 2.])[-1]
    spiked = cumulative([0., 1 / 19, 2 / 19, 1.], [1., 20., 1., 1.])[-1]
    assert spiked == pytest.approx(baseline)
    a = envelope(705, 40., .8, 1., .9, [baseline], [1.], 2 / 512)
    b = envelope(705, 40., .8, 1., .9, [spiked], [1.], 2 / 512)
    assert a["M4_envelope"] == pytest.approx(b["M4_envelope"])
    assert a["combined_relative_floor"] == pytest.approx(b["combined_relative_floor"])


@pytest.mark.parametrize("tolerance", [.01, .001, .0001])
@pytest.mark.parametrize("m4,target_fine", [(40., .9), (2000., .1)])
def test_accuracy_budget_matches_combined_floor(tolerance, m4, target_fine):
    limit = budgets(705, m4, .8, 1., target_fine, tolerance)
    result = envelope(705, m4, .8, 1., target_fine, [limit["combined"]], [1.], 2 / 512)
    assert result["combined_relative_floor"][0] == pytest.approx(tolerance, rel=2e-12)
    doubled = envelope(705, m4, .8, 1., target_fine,
                       [limit["moment_double"]], [1.], 2 / 512)
    assert doubled["M4_envelope"][0] == pytest.approx(2 * m4)


def test_expired_envelope_is_unavailable_not_reentered_after_pole():
    critical = budgets(705, 40., .8, 1., .9)["denominator"]
    result = envelope(705, 40., .8, 1., .9, [critical, 2 * critical], [1., 2.], 2 / 512)
    assert not np.any(result["envelope_defined"])
    assert np.all(np.isnan(result["M4_envelope"]))
    assert np.all(np.isnan(result["combined_relative_floor"]))


def test_physical_time_width_scaling_and_no_extrapolation():
    first = budgets(705, 40., .8, 1., .9)
    second = budgets(1410, 40., .8, 1., .9)
    assert second["denominator"] == pytest.approx(2 * first["denominator"])
    assert crossing_time([0., 1.], [1., 1.], first["denominator"]) is None
    with pytest.raises(ValueError, match="strictly increasing"):
        cumulative([0., 0., 1.], [1., 1., 1.])
