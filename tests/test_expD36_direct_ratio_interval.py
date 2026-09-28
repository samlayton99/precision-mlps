"""Independent checks of current-gamma matrix and GD interval construction."""
import numpy as np
from scipy.linalg import eigh

from experiments.expD36_frozen_gamma_probe.direct_ratio_interval import (
    mean_zero_coordinates, padded_eigenvalues, endpoint_arrays, calculate,
    quadrature_factor, exterior_factor, lattice_allowance, first_crossing,
)
from experiments.expD36_frozen_gamma_probe.core import design


def test_projector_and_padded_signed_ordering():
    rng = np.random.default_rng(31)
    a = rng.normal(size=(17, 5))
    projected = mean_zero_coordinates(a)
    np.testing.assert_allclose(projected.T@projected,
        a.T@(np.eye(17)-np.ones((17, 17))/17)@a, atol=2e-14)
    np.testing.assert_array_equal(padded_eigenvalues([2., -3., .5], 6), [2., .5, 0., 0., 0., -3.])


def test_interlacing_indices_and_known_rank_zeros():
    low, high, rl, ru = endpoint_arrays(np.array([8., 4., 2., 0.]), .1, .2, .3, 7., 11., 3)
    np.testing.assert_allclose(low, [7., 3.7, 1.7, 0., 0.])
    np.testing.assert_allclose(high, [11., 8.4, 4.4, 2.4, 0.])
    assert rl[0] == ru[0] == 1


def test_corrected_integral_encloses_finite_compression():
    x = np.linspace(-1, 1, 23); centers = np.linspace(-1.25, 1.25, 41); gamma = 2.
    h = centers[1]-centers[0]
    a, meta = quadrature_factor(x, h, gamma, order=18)
    c, ext = exterior_factor(x, centers, gamma)
    s = a@a.T-c@c.T
    projected = mean_zero_coordinates(design(x, centers, gamma))
    difference = projected@projected.T-s
    eigen = eigh(difference, eigvals_only=True)
    tolerance = lattice_allowance(x, h, gamma)+2e-12
    assert eigen[0] >= -tolerance-ext['exterior_lower_tail']
    assert eigen[-1] <= tolerance+meta['integration_upper_tail']


def test_full_small_geometry_ratio_and_error_intervals():
    x = np.linspace(-1, 1, 39); centers = np.linspace(-1.2, 1.2, 25)
    target = np.sin(3*np.pi*x)/np.sqrt(len(x))
    result, arrays = calculate(x, centers, 5., target, order=14)
    assert result['max_resolved_ratio_violation'] < 1e-12
    assert result['lower_curve_max_violation'] < 1e-12
    assert result['upper_curve_max_violation'] < 1e-12
    assert result['necessary_updates'] <= result['spectral_reference_updates']
    if result['sufficient_updates'] is not None:
        assert result['spectral_reference_updates'] <= result['sufficient_updates']
    assert arrays['actual_target_weights'].sum() <= 1+1e-12


def test_first_crossing_preserves_unresolved_floor():
    assert first_crossing([.1], [.99], .01) is None
    assert first_crossing([.5], [1.], 0., epsilon=.25) == 2
    assert first_crossing([], [], 1e-4) == 0


def test_supplied_step_matches_direct_finite_kernel_recurrence():
    x = np.linspace(-1, 1, 21); centers = np.linspace(-1.2, 1.2, 19)
    features = design(x, centers, 4.)
    kernel = features@features.T
    eta = .2/np.linalg.norm(features, 2)**2
    target = np.sin(2*np.pi*x)/np.sqrt(len(x))
    result, arrays = calculate(x, centers, 4., target, eta=eta, order=12)
    assert result['eta'] == eta
    np.testing.assert_allclose(result['eta_mu1'], .2, atol=1e-14)
    index = 10; step = arrays['steps'][index]
    direct = np.linalg.norm(np.linalg.matrix_power(np.eye(len(x))-eta*kernel, step)@target)/np.linalg.norm(target)
    np.testing.assert_allclose(arrays['reference_error'][index], direct, atol=2e-13)
