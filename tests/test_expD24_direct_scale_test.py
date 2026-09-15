"""Independent checks for the center-preserving Fourier tangent diagnostic."""
import numpy as np
import pytest
import torch
from scipy.integrate import quad

from experiments.expD24_gd_residual_spectrum import direct_scale_test as direct
from experiments.expD24_gd_residual_spectrum import whole_line as whole
from experiments.expD24_gd_residual_spectrum import readout_comparison as comparison


@pytest.mark.parametrize("gamma,center,omega", [(1., .3, 2.), (4., -.7, 12.), (64., .2, 80.), (4., .3, -12.), (4., .3, 0.)])
def test_analytic_scale_tangent_matches_spatial_transform(gamma, center, omega):
    # Integrate in transition coordinates so sharp kernels are resolved.
    def integrand(t):
        return t/np.cosh(t)**2*np.exp(-1j*omega*(center+t/gamma))/gamma**2
    expected = quad(lambda t: integrand(t).real, -32, 32, epsabs=1e-13)[0]
    expected += 1j*quad(lambda t: integrand(t).imag, -32, 32, epsabs=1e-13)[0]
    actual = direct.tangent_spectrum(np.array([omega]), np.array([gamma]), np.array([center]))[0, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=2e-13)


def test_phase_aware_pairing_matches_raw_coordinate_chain_rule():
    config = {"target_modes": [2., 6., 14.], "target_amplitudes": [1., .5, .25], "envelope_sigma": .4}
    a = torch.tensor([1., 4., 16., 64.], requires_grad=True)
    b = torch.tensor([-.3, .8, -2., -20.], requires_grad=True)
    v = torch.tensor([.3, -.2, .7, .1])
    k, w = comparison.frequency_quadrature(96)
    target = torch.tensor(whole.target_spectrum(np.pi*k, config))
    residual = comparison.whole_features(a, b, torch.tensor(k))@v.to(torch.complex128)-target
    loss = .5*torch.sum(torch.tensor(w)*(residual.real.square()+residual.imag.square()))
    loss.backward()
    aa, bb, vv = (value.detach().numpy() for value in (a, b, v))
    center = -bb/aa
    J = direct.tangent_spectrum(np.pi*k, aa, center)*(vv-vv.mean())
    expected = np.sum(w[:, None]*np.real(residual.detach().numpy()[:, None]*J.conj()), axis=0)
    np.testing.assert_allclose(a.grad.numpy()-center*b.grad.numpy(), expected, rtol=3e-10, atol=2e-12)
    # A magnitude-only overlap is a bound, not the signed gradient.
    envelope = np.sum(w[:, None]*abs(residual.detach().numpy()[:, None])*abs(J), axis=0)
    assert np.all(abs(expected) <= envelope)
    assert np.any(envelope > 1.5*abs(expected))


def test_magnitude_means_preserve_zeros_and_do_not_cancel_phases():
    am, gm = direct.magnitude_means(np.array([[1j, -1j], [0, 4], [1, -9]], dtype=complex))
    np.testing.assert_allclose(am, [1, 2, 5])
    np.testing.assert_allclose(gm, [1, 0, 3])


def test_tiny_matched_residual_signal_agrees_with_high_precision_quadrature():
    import mpmath as mp
    config = {"target_modes": [14.], "target_amplitudes": [1.], "envelope_sigma": .4}
    k, weights = comparison.frequency_quadrature(64)
    norm = np.sqrt(whole.target_energy(config))
    E = -whole.target_spectrum(np.pi*k, config)/norm
    J = direct.tangent_spectrum(np.pi*k, np.array([1.]), np.array([0.]))[:, 0]
    actual = np.dot(weights, np.real(E*J.conj()))
    with mp.workdps(55):
        sigma = mp.mpf(".4")
        carrier = 14*mp.pi
        norm_mp = mp.sqrt(mp.sqrt(mp.pi)*sigma/2*(1-mp.exp(-sigma**2*carrier**2)))
        def integrand(k):
            omega = mp.pi*k
            gaussian = lambda w: mp.sqrt(2*mp.pi)*sigma*mp.exp(-sigma**2*w*w/2)
            B = (gaussian(omega-carrier)-gaussian(omega+carrier))/(2*norm_mp)
            if not omega:
                return mp.mpf(0)
            s = mp.pi*omega/2
            derivative = mp.pi*(1-s*mp.coth(s))/mp.sinh(s)
            return B*derivative
        expected = float(mp.quad(integrand, [0, 4, 8, 10, 12, 14, 16, 20, 32, mp.inf]))
    assert abs(expected) < 1e-20  # Below the cancellation floor of spatial float64 sums.
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=0)
