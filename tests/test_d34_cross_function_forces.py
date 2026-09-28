"""Independent AD and directional differences for sampled force diagnostics."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import cross_function_forces as cf
from experiments.expD34_readout_race import effective_feedback_kernel as ef
from experiments.expD34_readout_race import transport


def example(degree=9):
    rng = np.random.default_rng(429)
    p = jnp.asarray(np.r_[rng.normal(size=15)*.6, .12])
    x = np.linspace(-1+1/128, 1-1/128, 128)
    y = np.sin(2*np.pi*x)+.2*x**2
    context = ef.fork_context(p, x, y, q=transport.basis(x, degree))
    return p, context


def independent(p, context):
    x, y, q = (context[k] for k in ('x', 'y', 'q'))
    def residual(v):
        a, b, c = v[:-1].reshape(3, -1)
        return jnp.tanh(x[:, None]*a+b) @ c+v[-1]-y
    r = residual(p)
    sample_J = jax.jacfwd(residual)(p)
    J = q.T @ sample_J/len(x)
    e = q.T @ r/len(x)
    B = jnp.linalg.solve(J[:2] @ J[:2].T, J[:2] @ J[2:].T)
    D, C = J[2:].T, J[:2].T @ B
    g = jax.grad(lambda v: jnp.mean(residual(v)**2)/2)(p)
    tracking = J[:2].T @ (e[:2]+B @ e[2:])
    omitted = sample_J.T @ (r-q @ e)/len(x)
    return D, C, e[2:], g, tracking, omitted, J[2:]


@pytest.mark.parametrize('degree', [3, 65])
def test_full_reconstruction_modal_signs_and_forecast_defect(degree):
    p0, context = example(degree)
    D0, C0 = cf.gain_parts(p0, context)
    p = p0-.03*ef.field(p0, context)[0]
    e_hat = context['eH0']*.98
    out = jax.jit(cf.make_analyzer(False))(p, context, D0, C0, e_hat)
    D, C, e, g, tracking, omitted, JH = independent(p, context)
    width = 5
    signed = -np.sign(np.asarray(p[:width]))/width
    for key, expected in [('gradient', g), ('direct', D @ e),
                           ('balanced', -C @ e), ('tracking', tracking),
                           ('omitted', omitted), ('effective', (D-C) @ e),
                           ('fine_effective_forcing', JH @ ((D-C) @ e)),
                           ('fine_tracking_forcing', JH @ tracking),
                           ('fine_omitted_forcing', JH @ omitted)]:
        np.testing.assert_allclose(out[key], expected, atol=2e-13, rtol=2e-11)
    np.testing.assert_allclose(out['reconstruction'], 0, atol=3e-14)
    np.testing.assert_allclose(out['defect_identity'], 0, atol=3e-14)
    assert float(out['coarse_projector_identity_norm']) < 3e-13
    assert float(out['Schur_identity_norm']) < 3e-13
    for name in ('fine_effective_forcing', 'fine_tracking_forcing',
                 'fine_omitted_forcing', 'fine_residual_velocity', 'defect_identity'):
        np.testing.assert_allclose(out[name+'_norm'], np.linalg.norm(out[name]),
                                   atol=2e-14)
    np.testing.assert_allclose(out['fine_residual_velocity'],
        -out['fine_effective_forcing']-out['fine_tracking_forcing']
        -out['fine_omitted_forcing'], atol=3e-13)
    np.testing.assert_allclose(out['forecast_defect_a'],
                               g[:width]-(D0-C0)[:width] @ e_hat, atol=3e-14)
    for label, gain in [('direct', D), ('balanced', -C), ('effective', D-C)]:
        expected = gain[:width]*e[None, :]
        np.testing.assert_allclose(out['modal_'+label+'_a'], expected, atol=2e-13)
        np.testing.assert_allclose(out['modal_'+label+'_A'], signed @ gain[:width],
                                   atol=2e-13)
        np.testing.assert_allclose(out['modal_'+label+'_velocity'],
                                   signed @ expected, atol=2e-13)
        np.testing.assert_allclose(np.sum(out['modal_'+label+'_velocity']),
                                   out[label+'_signed_velocity'], atol=2e-13)


def test_block_jvps_against_independent_finite_differences_and_full_ad():
    p, context = example()
    D0, C0 = cf.gain_parts(p, context)
    out = jax.jit(cf.make_analyzer())(p, context, D0, C0, context['eH0'])
    _, _, e, g, _, _, _ = independent(p, context)
    eps = 2e-5
    for block, (lower, upper) in enumerate([(0, 5), (5, 10), (10, 15), (15, 16)]):
        direction = jnp.zeros_like(p).at[lower:upper].set(-g[lower:upper])
        plus = independent(p+eps*direction, context)
        minus = independent(p-eps*direction, context)
        for label, index, sign in [('direct', 0, 1), ('balanced', 1, -1)]:
            expected = sign*((plus[index]-minus[index])/(2*eps))[:5] @ e
            np.testing.assert_allclose(out['block_'+label+'_derivative_a'][block],
                                       expected, rtol=2e-6, atol=2e-10)
    def effective_force(v):
        D, C, residual, *_ = independent(v, context)
        return ((D-C) @ residual)[:5]
    full_ad = jax.jacfwd(effective_force)(p) @ (-g)
    np.testing.assert_allclose(out['full_force_derivative_a'], full_ad,
                               atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(out['derivative_identity'], 0, atol=2e-12)
    assert float(out['derivative_identity_norm']) < 2e-12
    for label in ('direct', 'balanced', 'gain'):
        np.testing.assert_allclose(out['block_'+label+'_derivative_signed'],
            out['block_'+label+'_derivative_a'] @ (-jnp.sign(p[:5])/5), atol=2e-13)
