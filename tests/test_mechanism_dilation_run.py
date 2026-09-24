"""Focused exact-update and signed-accounting checks; run under CPU Slurm."""
import jax
import jax.numpy as jnp
import numpy as np
import json
from types import SimpleNamespace

from experiments.expD34_readout_race import mechanism_dilation_run as runner
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel


def fixture():
    x = jnp.linspace(-.95, .95, 32, dtype=jnp.float64)
    p = jnp.array([.05, -.07, .11, .02, -.03, .01, .7, -.4, .3, .02], dtype=jnp.float64)
    y = .3+.4*x+jnp.sin(3*x)
    return p, x, y


def test_update_matches_autodiff_and_channels_close():
    p, x, y = fixture()
    def loss(v):
        a, b, c = v[:-1].reshape(3, -1)
        return jnp.mean((jnp.tanh(x[:, None]*a+b)@c+v[-1]-y)**2)/2
    gradient = jax.grad(loss)(p)
    state = runner.initial_state(p)
    got = runner.step(state, x, y, .01, .002)
    np.testing.assert_allclose(got['p'], p-.002*gradient, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(got['signed'].sum(0)+got['crossing'],
                               .01*(abs(got['p'][:3])-abs(p[:3])), atol=1e-17)
    np.testing.assert_allclose(got['positive']-got['negative'],
                               got['signed'].sum(0)+got['crossing'], atol=1e-17)
    assert not bool(got['failed'])


def test_crossing_and_failed_state_accounting():
    p, x, y = fixture()
    # A deliberately large finite step exercises absolute-value crossings;
    # this is accounting verification, not an accepted experiment step size.
    state = runner.initial_state(p)
    got = runner.step(state, x, -y, .01, 10.)
    assert np.any(np.asarray(p[:3]*got['p'][:3]) < 0)
    assert np.sum(np.asarray(got['crossing'])) > 0
    np.testing.assert_allclose(got['signed'].sum(0)+got['crossing'],
                               .01*(abs(got['p'][:3])-abs(p[:3])), atol=1e-16)
    state['failed'] = jnp.array(True)
    stopped = runner.step(state, x, y, .01, .002)
    np.testing.assert_array_equal(stopped['p'], state['p'])
    assert int(stopped['count']) == 0


def test_sparse_run_records_invalid_branch_and_own_forecast(tmp_path):
    p, x, y = fixture()
    source = tmp_path/'input.npz'
    cases = [dict(target='fixture', cohort='development', arm='original', h=.01, valid=True),
             dict(target='fixture', cohort='development', arm='invalid', h=.01, valid=False)]
    xe = jnp.linspace(-.99, .99, 65, dtype=jnp.float64)
    ye = .3+.4*xe+jnp.sin(3*xe)
    np.savez(source, p=np.stack([p, p]), x=x, y=np.stack([y, y]), cases=json.dumps(cases),
             x_eval=xe, y_eval=np.stack([ye, ye]))
    output = tmp_path/'output'
    runner.run(SimpleNamespace(input=source, output=output, backend='cpu', eta=.002,
                               steps=2, targets=None, cohorts=None, arms=None))
    manifest = json.loads((output/'manifest.json').read_text())
    assert len(manifest['cases']) == len(manifest['skipped_invalid']) == 1
    with np.load(output/'000000002.npz') as data:
        assert data['p'].shape == (1, 10)
        assert data['q23_generated_signed_rate'].shape == (1, 2)
        assert not data['failed'].any()
        assert data['closure_error'].max() < 1e-16
        np.testing.assert_allclose(data['forecast_lambda'], .01*abs(data['forecast_a']))
        expected_eval = np.linalg.norm(kernel.output(data['p'][0], xe)-ye)/np.linalg.norm(ye)
        np.testing.assert_allclose(data['relative_eval_l2'][0], expected_eval, rtol=1e-13)
        radii2 = 3*np.sum(data['p'][0, :-1].reshape(3, 3)**2, axis=0)
        np.testing.assert_allclose(data['M6'][0], np.mean(radii2**3), rtol=1e-13)


def test_nonlinear_full_complement_agreement_and_rms_derivative():
    p, x, y = fixture()
    p = p.at[:6].multiply(8.)
    gradient, parts, fine, resolved, _, _ = runner.channels(p, x, y)
    reference = kernel.decomposition(p, x, y)
    assert bool(resolved)
    np.testing.assert_allclose(fine, reference['F'], atol=1e-13)
    np.testing.assert_allclose(parts[2], reference['R'], atol=1e-13)
    np.testing.assert_allclose(parts.sum(0), gradient, atol=1e-13)
    q, _ = np.linalg.qr(np.polynomial.polynomial.polyvander(np.asarray(x), 3))
    values = runner.diagnostic(p, x, y, .01, jnp.asarray(q[:, 2:4]*np.sqrt(len(x))))
    derivative = jax.jvp(lambda v: .01*jnp.sqrt(jnp.mean(v[:3]**2)), (p,), (-fine,))[1]
    np.testing.assert_allclose(values['lambda_rms_rates'][0], derivative, atol=1e-14)


def test_unresolved_split_does_not_stop_ordinary_gd():
    p = jnp.zeros(10, dtype=jnp.float64)
    x = jnp.linspace(-1., 1., 33)
    y = .5+jnp.sin(x)
    state = runner.initial_state(p)
    got = runner.step(state, x, y, .01, .002)
    assert not bool(got['failed'])
    assert int(got['diagnostic_unresolved_steps']) == 1
    assert int(got['count']) == 1
    np.testing.assert_allclose(got['p'][-1], .001, atol=1e-16)


def test_raw_error_hits_use_l2_not_mse_and_initial_time():
    p, x, _ = fixture()
    output = kernel.output(p, x)
    y = output/1.05  # relative L2=.05, relative MSE=.0025
    got = runner.step(runner.initial_state(p), x, y, .01, .002)
    np.testing.assert_array_equal(got['error_first_hit'], [0, -1, -1, -1, -1])
