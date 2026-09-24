"""Numerical checks for the actual-update and theorem-bound Section 3.4 paths."""
import numpy as np

from experiments.expD36_frozen_gamma_probe.direct_ratio_interval import calculate
from experiments.expD36_frozen_gamma_probe.section34_bounds import curves
from experiments.expD36_frozen_gamma_probe.section34_frozen_gd import self_test


def test_actual_updates_match_independent_gradient_and_spectrum():
    self_test()


def test_error_bound_encloses_reference_with_unresolved_energy_removed():
    x = -1+2*(np.arange(67)+.5)/67
    centers = -1+np.arange(-6, 39)/16
    target = np.sin(3*np.pi*x)+.2*np.cos(7*np.pi*x)
    summary, arrays = calculate(x, centers, 4., target, order=10, padding=20)
    lower, exact, upper = curves(summary, arrays, np.arange(101))
    assert summary['max_resolved_ratio_violation'] < 1e-12
    assert np.max(lower-exact) < 1e-12
    assert np.max(exact-upper) < 1e-12
    np.testing.assert_allclose(exact[0], 1., atol=1e-14)
    assert np.all(np.diff(lower) <= 0.)
    assert np.all(np.diff(exact) <= 0.)
