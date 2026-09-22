"""Discrete, full-curvature, and conditional-bound checks on small examples."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import effective_feedback_predict as efp
from experiments.expD34_readout_race import effective_feedback_kernel as kernel
from experiments.expD34_readout_race import persistence_theory as pt


def test_nonnormal_affine_forecast_matches_explicit_discrete_steps():
    p0 = np.array([.1, -.3, .4])
    g0 = np.array([.2, -.1, .3])
    # Nonsymmetric, including a conjugate pair: no orthogonal spectral shortcut.
    derivative = np.array([[.4, 2., 0.], [-.3, .4, 1.], [0., 0., -.01]])
    horizons = [0, 1, 2, 11, 257]
    got = efp.affine_forecast(p0, g0, derivative, .01, horizons)
    state = p0.copy()
    expected = []
    for n in range(258):
        if n in horizons:
            expected.append(state.copy())
        state -= .01*(g0 + derivative @ (state-p0))
    np.testing.assert_allclose(got["p"], expected, atol=2e-14, rtol=2e-13)
    assert got["supported"].all()


def test_zero_modes_keep_finite_time_growth_and_overflow_is_reported():
    horizons = [0, 1, 200000, 5400000]
    got = efp.affine_forecast(np.zeros(2), np.array([1., -2.]),
                              np.zeros((2, 2)), .002, horizons)
    np.testing.assert_allclose(got["p"], -.002*np.array(horizons)[:, None]*[1., -2.])
    values, supported = efp.discrete_at(np.array([[2.]]), np.ones(1), [1, 2, 2048])
    np.testing.assert_array_equal(supported, [True, True, False])
    np.testing.assert_allclose(values[:2, 0], [2., 4.])
    assert np.isnan(values[2, 0])


def example():
    rng = np.random.default_rng(83)
    p = rng.normal(0, .2, 10)
    x = np.linspace(-.98, .98, 48)
    y = np.sin(4*x) + .1*x*x
    return p, x, y


def test_direction_linearization_retains_residual_curvature():
    p, x, y = example()
    def loss(z):
        a, b, c = z[:-1].reshape(3, -1)
        prediction = jnp.tanh(x[:, None]*a+b) @ c+z[-1]
        return jnp.mean((prediction-y)**2)/2
    gradient, derivative = efp.direction_linearization(jax.grad(loss), p)
    independent = pt.tensors(p, x, y)
    np.testing.assert_allclose(gradient, independent["g"], atol=2e-14)
    np.testing.assert_allclose(derivative, independent["hessian"], atol=2e-14)
    assert np.linalg.norm(derivative-independent["gram"]) > .01


def test_frozen_effective_model_keeps_remainder_and_feedback_separate():
    rng = np.random.default_rng(37)
    p = rng.normal(size=7)
    T = rng.normal(size=(7, 3))*.1
    J = T.T
    e0 = rng.normal(size=3)
    remainder = rng.normal(size=7)*.01
    horizons = [0, 1, 10, 100]
    forecast = efp.frozen_effective_forecast(p, T@e0+remainder, T, J, e0,
                                             .02, horizons)
    for arm, correction in (("pure", np.zeros(7)), ("with_remainder", remainder)):
        e, state = e0.copy(), p.copy()
        expected = []
        for n in range(101):
            if n in horizons:
                expected.append(state.copy())
            direction = T@e+correction
            state -= .02*direction
            e -= .02*J@direction
        np.testing.assert_allclose(forecast[arm]["p"], expected, atol=3e-14)


def test_ordinary_gd_enclosure_covers_all_steps_and_affine_error():
    p, x, y = example()
    eta = .002
    horizons = np.arange(31)
    bound = efp.ordinary_gd_bounds(p, x, y, eta, horizons)
    assert bound["closed"].all()
    state = pt.tensors(p, x, y)
    affine = efp.affine_forecast(p, state["g"], state["hessian"], eta, horizons)
    actual = p.copy()
    path = 0.
    for n in horizons:
        assert path <= bound["parameter_path"][n]+1e-14
        assert np.linalg.norm(actual-affine["p"][n]) <= bound["affine_error"][n]+1e-14
        increment = -eta*pt.tensors(actual, x, y)["g"]
        actual += increment
        path += np.linalg.norm(increment)


def test_prediction_bundle_uses_kernel_fields_and_is_serializable():
    p, x, y = example()
    result = efp.predict_case(p, x, y, degree=5, horizons=[0, 1, 2, 10])
    arrays = result["arrays"]
    json.dumps(result["metadata"], allow_nan=False)
    for arm in efp.ARMS:
        np.testing.assert_allclose(arrays[f"{arm}_p"][1],
                                   p-.002*arrays["joint_g0"], atol=3e-14)
        assert arrays[f"{arm}_supported"].all()
        assert arrays[f"{arm}_gamma"].shape == (4, 3)
    np.testing.assert_allclose(arrays["joint_Dg0"], pt.tensors(p, x, y)["hessian"],
                               atol=2e-14)
    context = kernel.fork_context(p, x, y, degree=5)
    perturbation = np.linspace(-.3, .5, len(p))
    for arm in efp.ARMS:
        high = np.asarray(kernel.field(p+1e-5*perturbation, context, arm)[0])
        low = np.asarray(kernel.field(p-1e-5*perturbation, context, arm)[0])
        np.testing.assert_allclose(arrays[f"{arm}_Dg0"] @ perturbation,
                                   (high-low)/2e-5, atol=3e-10, rtol=3e-8)
    fine_high = kernel.field(p+1e-5*perturbation, context)[1]["effective_a"]
    fine_low = kernel.field(p-1e-5*perturbation, context)[1]["effective_a"]
    combined = arrays["map_force_derivative0"]+arrays["error_force_derivative0"]
    np.testing.assert_allclose(combined @ perturbation, (fine_high-fine_low)/2e-5,
                               atol=3e-10, rtol=3e-8)
