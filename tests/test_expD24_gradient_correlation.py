"""Independent checks for the spatial bound and centered-scale diagnostic."""
import numpy as np
from scipy.integrate import quad

from experiments.expD24_gd_residual_spectrum import gradient_correlation as probe


def test_integral_includes_readout_orientation_and_uniform_density():
    state = {"a": np.array([-2.1, 5.3]), "b": np.array([.3, -.7]),
             "v": np.array([.7, -.2, .1])}
    config = {"envelope_sigma": .4, "target_modes": [2., 6., 14.],
              "target_amplitudes": [1., .5, .25]}
    bound, signed = probe.spatial_integrals(state, "gaussian_envelope", config, np.arange(2), 32768)
    a, b, v = (state[key] for key in ("a", "b", "v"))
    for j in range(2):
        z = -b[j]/a[j]
        def pair(x):
            error = np.tanh(a*x+b) @ v[:-1]+v[-1]-probe.matched.target_values("gaussian_envelope", x, config)
            tangent = v[j]*np.sign(a[j])*(x-z)/np.cosh(a[j]*x+b[j])**2
            return .5*error*tangent
        expected_bound = quad(lambda x: abs(pair(x)), -1, 1, epsabs=1e-11, limit=200)[0]
        expected_signed = quad(pair, -1, 1, epsabs=1e-11)[0]
        np.testing.assert_allclose(bound[j], expected_bound, atol=2e-9, rtol=1e-7)
        np.testing.assert_allclose(signed[j], expected_signed, atol=2e-9, rtol=1e-7)


def test_even_residual_can_cancel_with_nonzero_absolute_overlap(monkeypatch):
    state = {"a": np.array([2.]), "b": np.array([0.]), "v": np.array([1., 0.])}
    # Prediction minus this target is x^2, even; the centered tangent is odd.
    monkeypatch.setattr(probe.matched, "target_values", lambda target, x, config: np.tanh(2*x)-x*x)
    bound, signed = probe.spatial_integrals(state, "test", {}, np.array([0]), 4096)
    assert bound[0] > .01
    assert abs(signed[0]) < 1e-16


def test_linear_fit_uses_an_intercept_and_original_values():
    x = np.linspace(0, 1e-8, 100)
    result = probe.linear_fit(x, -.3*x+2e-9)
    np.testing.assert_allclose(result["slope"], -.3)
    np.testing.assert_allclose(result["intercept"], 2e-9, atol=1e-20)
    np.testing.assert_allclose(result["r_squared"], 1.)
    np.testing.assert_allclose(result["pearson_r"], -1.)
