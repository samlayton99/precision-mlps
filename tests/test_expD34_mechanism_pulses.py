"""Independent derivative and constraint checks for matched geometry pulses."""
import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import mechanism_pulses as mp
from experiments.expD34_readout_race import effective_feedback_kernel as kernel
from experiments.expD34_readout_race import transport


def example():
    rng = np.random.default_rng(233)
    p = rng.normal(size=31)*.3
    x = np.linspace(-.99, .99, 80)
    y = np.sin(3*x)+.1*x*x
    return p, x, y, transport.basis(x, 8)


def test_force_derivatives_reconstruct_with_remainder():
    p, x, y, q = example()
    system = mp.local_system(p, x, y, q)
    context = dict(x=jnp.asarray(x), y=jnp.asarray(y), q=jnp.asarray(q), p0=jnp.asarray(p))
    def effective(point):
        matrices = kernel.matrices(point, context)
        return matrices['T']@kernel.field(point, context)[1]['eH']
    DF = np.asarray(jax.jacfwd(effective)(jnp.asarray(p)))
    np.testing.assert_allclose(system['hessian']-system['DR'], DF, atol=4e-15, rtol=1e-11)
    np.testing.assert_allclose(system['map_derivative'], DF-system['T']@system['JH'], atol=4e-15)


def test_matched_direction_and_halved_pulse_mismatch():
    p, x, y, q = example()
    system = mp.local_system(p, x, y, q)
    direction, info = mp.matched_direction(p, system)
    assert info['outward_derivative'] > 0
    assert info['normalized_constraint_residual'] < 2e-12
    constraints = np.vstack((system['JC'], system['Dz'], system['hessian'][:10]))
    np.testing.assert_allclose(constraints@direction, 0, atol=1e-13)
    errors = []
    for amplitude in (.01, .005, .0025):
        value = mp.pt.tensors(p+amplitude*direction, x, y)['g'][:10]
        errors.append(np.linalg.norm(value-system['g'][:10]))
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:], 4., rtol=.03)


def test_anchored_predictors_have_identical_first_update():
    p, x, y, q = example()
    system = mp.local_system(p, x, y, q)
    direction, _ = mp.matched_direction(p, system)
    pulse = p+.01*direction
    gradient = mp.pt.tensors(pulse, x, y)['g']
    predictions = [mp.affine_forecast(pulse, gradient, H, .002, [1, 2])['p']
                   for H in (system['frozen_derivative'], system['hessian'])]
    for prediction in predictions:
        np.testing.assert_allclose(prediction[0], pulse-.002*gradient, atol=1e-16)


def test_family_geometric_sums_match_individual_affine_models():
    rng = np.random.default_rng(12)
    points = rng.normal(size=(3, 9))
    gradients = rng.normal(size=(3, 9))
    H = rng.normal(size=(9, 9))*.03
    horizons = [0, 1, 2, 13, 100]
    family, supported = mp.forecast_family(points, gradients, H, .002, horizons)
    assert supported.all()
    for i in range(3):
        reference = mp.affine_forecast(points[i], gradients[i], H, .002, horizons)['p']
        np.testing.assert_allclose(family[i], reference, atol=2e-15, rtol=1e-14)


def test_two_observable_closure_matches_first_two_variational_steps():
    p, x, y, q = example()
    system = mp.local_system(p, x, y, q)
    direction, _ = mp.matched_direction(p, system)
    eta = .002
    values, info = mp.reduced_response(p, x, y, system, direction, eta)
    assert info['reduced_identified']
    width = (len(p)-1)//3
    weight = direction[:width]/(direction[:width]@direction[:width])
    variation = direction.copy()
    point = p.copy()
    exact = [weight@variation[:width]]
    for _ in range(2):
        state = mp.pt.tensors(point, x, y)
        variation = variation-eta*state['hessian']@variation
        point = point-eta*state['g']
        exact.append(weight@variation[:width])
    np.testing.assert_allclose(values[:3], exact, atol=3e-14, rtol=3e-14)
