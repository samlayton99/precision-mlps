import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race.mechanism_polynomial import coefficients_jacobian, modal_setup, field
from experiments.expD34_readout_race.mechanism_splitting_baselines import matrices
from experiments.expD34_readout_race import transport


@pytest.mark.parametrize('degree', [3, 5])
def test_polynomial_coefficients_autodiff_and_schur(degree):
    x = np.linspace(-1, 1, 101)
    p = jnp.asarray(np.random.default_rng(7).normal(size=22)*.17)
    y = np.sin(2*x)+.2*x*x
    transform, target = modal_setup(x, y, degree)
    coefficients, jacobian = coefficients_jacobian(p, degree)
    np.testing.assert_allclose(jacobian, jax.jacfwd(lambda v: coefficients_jacobian(v, degree)[0])(p), atol=1e-14)
    def direct(v):
        a, b, c = v[:-1].reshape(3, 7)
        u = jnp.asarray(x)[:, None]*a+b
        feature = u-u**3/3
        if degree == 5:
            feature = feature+2*u**5/15
        return feature@c+v[-1]
    J = np.asarray(jax.jacfwd(direct)(p))
    np.testing.assert_allclose(np.asarray(x)[:, None]**np.arange(degree+1)@coefficients, direct(p), atol=1e-14)
    q = transport.basis(x, degree)
    modal = q.T@J/len(x)
    JC = modal[:2]
    g = J.T@(np.asarray(direct(p))-y)/len(x)
    exact = g-JC.T@np.linalg.solve(JC@JC.T, JC@g)
    force, _ = field(p, jnp.asarray(transform), jnp.asarray(target), degree)
    np.testing.assert_allclose(force, exact, atol=2e-15)
    np.testing.assert_allclose(JC@force, 0, atol=2e-16)


def test_uniform_parameter_rescaling_force_convergence():
    x = np.linspace(-1, 1, 129); y = np.cos(2*x)+.3*np.sin(3*x)
    base = np.random.default_rng(18).normal(size=25)*.18
    errors = {3: [], 5: []}
    for scale in (1., .5, .25):
        p = base*scale
        _, T, _, e = matrices(p, x, y, np.ones(len(p)), transport.basis(x, 65))
        exact = T@e
        for degree in (3, 5):
            transform, target = modal_setup(x, y, degree)
            force, _ = field(jnp.asarray(p), jnp.asarray(transform), jnp.asarray(target), degree)
            errors[degree].append(np.linalg.norm(np.asarray(force)-exact))
    # Absolute errors should gain at least four/six powers under rescaling;
    # this is a controlled smooth-target convergence check, not a data-fit gate.
    assert min(np.asarray(errors[3][:-1])/errors[3][1:]) > 12
    assert min(np.asarray(errors[5][:-1])/errors[5][1:]) > 40
    assert errors[5][-1] < errors[3][-1]


def test_anchored_first_step_matches_exact_effective_force():
    from experiments.expD34_readout_race.mechanism_polynomial_anchor import predict
    x = np.linspace(-1, 1, 129); y = np.cos(2*x)+.3*np.sin(3*x)
    p = np.random.default_rng(9).normal(size=25)*.12
    _, T, _, e = matrices(p, x, y, np.ones(len(p)), transport.basis(x, 65))
    states, _ = predict(p[None], x, y[None], steps=1)
    np.testing.assert_allclose(states[0], p-.002*(T@e), atol=1e-16)


def test_clamped_force_matches_own_force_at_anchor():
    from experiments.expD34_readout_race.mechanism_polynomial_clamp import clamped_field
    x = np.linspace(-1, 1, 65); y = np.sin(2*x)
    p = jnp.asarray(np.random.default_rng(3).normal(size=22)*.1)
    transform, target = modal_setup(x, y, 5)
    coefficients, _ = coefficients_jacobian(p, 5)
    e0 = (transform@coefficients-target)[2:]
    force, _ = field(p, jnp.asarray(transform), jnp.asarray(target), 5)
    np.testing.assert_allclose(clamped_field(p, jnp.asarray(transform), e0), force, atol=1e-16)
