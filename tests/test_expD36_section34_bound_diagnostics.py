import numpy as np
from experiments.expD36_frozen_gamma_probe.section34_bound_diagnostics import bounded_crossing, executed_crossings, theorem_value


def test_integer_prediction_crossing_and_right_censoring():
    value = lambda n: .5**n
    assert bounded_crossing(value, .125, 10) == 3
    assert bounded_crossing(value, .125, 2) is None
    assert bounded_crossing(value, 1., 2) == 0
    assert bounded_crossing(value, .25, 2) == 2


def test_executed_first_hit_survives_later_upcrossing_and_block_boundaries():
    raw = np.array([1., .8, .1, .9, .01, .7])
    assert executed_crossings(raw, [.2, .02, .001], 5, block=2) == [2, 4, None]


def test_unresolved_target_energy_is_omitted_from_theorem():
    arrays = dict(actual_target_weights=np.array([.6, .4]), resolved=np.array([True, False]), rho_upper=np.array([1., 0.]))
    value = theorem_value(arrays)
    np.testing.assert_allclose(value(0), np.sqrt(.6))
    np.testing.assert_allclose(value(3), np.sqrt(.6)*.5**3)
