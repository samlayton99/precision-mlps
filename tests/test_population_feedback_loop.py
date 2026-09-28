import math

import numpy as np
from numpy.testing import assert_allclose

from experiments.expD34_readout_race.population_feedback_loop import (
    audit, critical_response, integrated_travel,
)


def test_twice_integrated_linear_speed_on_unequal_grid():
    t = np.array([0., .1, 1., 3.])
    travel, twice = integrated_travel(t, 2+3*t)
    assert_allclose(travel, 2*t+1.5*t*t)
    assert_allclose(twice, t*t+.5*t**3)


def test_zero_baseline_critical_response_has_closed_form():
    assert_allclose(critical_response(3., .1, 0., 16.),
                    2/(math.e*.1*math.sqrt(8)*9), rtol=1e-12)


def test_constant_feedback_needs_no_excess_response():
    row = audit([0., 1., 4.], [.01]*3, [.2]*3, [16.]*3)
    assert row['required_K'] == 0
    assert row['sampled_bootstrap_pass']
    assert row['concentration_ratio'] == .5


def test_constructed_feedback_travel_response_and_prefix_failure():
    t = np.array([0., 1., 40., 80.])
    # speed is 2, A=2t; d=.1+K*A, so excess integral=K*t^2.
    row = audit(t, np.ones(4), .1+.06*t, np.full(4, 16.))
    assert_allclose(row['required_K'], .03)
    assert_allclose(row['prefix_K_multiplier_required'], 1.)
    rate = np.array([.1, .16, 2.5, 20.])
    changed = audit(t, np.ones(4), rate, np.full(4, 16.))
    assert not changed['prefix_K_factor1_covers']
    assert not changed['sampled_bootstrap_pass']
