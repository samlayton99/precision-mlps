"""Selection, finite decomposition, and independent Fourier-gain checks."""
import numpy as np
from scipy.integrate import quad

from experiments.expD36_frozen_gamma_probe.core import design
from experiments.expD36_frozen_gamma_probe.gamma_mechanism_analysis import select_direction, directional_response
from experiments.expD36_frozen_gamma_probe.gamma_ratio_bound import kernel_increment
from experiments.expD36_frozen_gamma_probe.uniform_periodic import attenuation


def test_selection_uses_target_overlap_among_eligible_modes():
    features = np.diag([1., .1, .01, 1e-16])
    target = np.array([5., .2, .7, 100.])
    direction, selected = select_direction(features, target, ratio_cutoff=.02)
    assert selected['rank'] == 3
    assert selected['eligible_count'] == 2
    np.testing.assert_allclose(abs(direction), [0, 0, 1, 0])
    assert selected['initial_target_energy'] == (.7**2)/(target@target)


def test_directional_response_decomposition_and_dense_operator():
    x = np.linspace(-1, 1, 35); centers = np.linspace(-1.3, 1.3, 27)
    old = design(x, centers, 3.)
    v = np.sin(5*x); v /= np.linalg.norm(v)
    new = design(x, centers, 9.)
    row = directional_response(x, centers, v, old, 9., np.linalg.norm(new, 2)**2, 3.)
    np.testing.assert_allclose(row['actual_action'], v@(new@new.T)@v, atol=1e-14)
    assert row['direct_row_max_absolute_error'] < 1e-14
    np.testing.assert_allclose(row['reference_action_plus_gain']+row['finite_correction'],
                               row['actual_action'], atol=1e-15)


def test_explicit_gain_matches_independent_fourier_integral():
    x = np.array([-.8, -.1, .3, .9]); v = np.array([.1, -.7, .2, .6])
    old, new, h, m = 2., 5., .1, len(x)
    def integrand(omega):
        weight = (attenuation(omega, new)**2-attenuation(omega, old)**2)/omega**2 \
            if omega else np.pi**2/12*(old**-2-new**-2)
        return float(weight*abs(np.sum(v*np.exp(-1j*omega*x)))**2)
    fourier = 4/(np.pi*h*m)*quad(integrand, 0, 200, epsabs=1e-12, limit=300)[0]
    direct = v@kernel_increment(x[:, None]-x, old, new, h, m)@v
    np.testing.assert_allclose(fourier, direct, rtol=2e-11, atol=1e-12)
