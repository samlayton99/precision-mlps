"""Preserved-history, slope-only numerator/denominator intervention checks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace
import json

from experiments.expD34_readout_race import mechanism_adam as ma
from experiments.expD34_readout_race import adam_forces as af


def fixture():
    rng = np.random.default_rng(31)
    p = rng.normal(size=(1, 22)) * .2
    cm = rng.normal(size=(1, 3, 22)) * .01
    pack = dict(p=p, m=cm.sum(axis=1), v=rng.uniform(.01, .1, (1, 22)),
                channel_m=cm, count=np.array([600000]))
    return jax.tree.map(lambda v: v[0], ma.initial(pack))


@pytest.mark.parametrize('alpha', ma.ARMS)
def test_independent_numpy_moment_formula_and_cross_terms(alpha):
    old = fixture()
    rng = np.random.default_rng(9)
    ch = rng.normal(size=(3, 22))*.1
    ch[2] = 0
    result, _ = ma.moment_step(old, jnp.asarray(ch), jnp.asarray(alpha))
    cm_in = ch.copy(); cv_in = ch.copy()
    cm_in[1, :7] *= alpha[0]; cv_in[1, :7] *= alpha[1]
    m = .9*np.asarray(old['m']) + .1*cm_in.sum(axis=0)
    v = .999*np.asarray(old['v']) + .001*cv_in.sum(axis=0)**2
    delta = -.002*m/(1-.9**600001)/(np.sqrt(v/(1-.999**600001))+1e-8)
    np.testing.assert_allclose(result['m'], m, atol=1e-16)
    np.testing.assert_allclose(result['v'], v, atol=1e-16)
    np.testing.assert_allclose(result['p'], old['p']+delta, atol=1e-16)
    assert result['count'] == 600001
    np.testing.assert_allclose(result['channel_m'].sum(axis=0), result['m'], atol=1e-16)
    # Squaring channel sums must differ from summing channel squares.
    assert np.max(abs(v-(.999*np.asarray(old['v'])+.001*np.sum(cv_in**2, axis=0)))) > 1e-6


def test_baseline_matches_existing_adam_and_nonslope_first_update():
    old = fixture(); x = jnp.linspace(-1, 1, 67); y = jnp.sin(4*x)+.2
    g, _, jc, ec = af.field(old['p'], x, y)
    ch, info = af.split(g, jc, ec)
    assert info['resolved']
    m, v, cm, mh, _, inv = af.moments(g, ch, old['m'], old['v'],
        old['channel_m'], old['count']+1, .9, .999, 1e-8, True)
    baseline, _ = ma.step(old, x, y, jnp.ones(2))
    for k, value in (('m', m), ('v', v), ('channel_m', cm), ('p', old['p']-.002*inv*mh)):
        np.testing.assert_allclose(baseline[k], value, atol=1e-16)
    for alpha in ma.ARMS:
        state, _ = ma.step(old, x, y, jnp.array(alpha))
        for k in ('p', 'm', 'v'):
            np.testing.assert_array_equal(state[k][7:], baseline[k][7:])


def test_resume_and_normalized_signed_travel_with_crossing():
    old = fixture()
    old['p'] = old['p'].at[0].set(1e-10)
    old['m'] = old['m'].at[0].set(.2)
    old['channel_m'] = old['channel_m'].at[:, 0].set(jnp.array([.2, 0., 0.]))
    ch = jnp.zeros((3, 22)).at[0, 0].set(.2)
    alpha = jnp.array([.9, .9])
    def advance(state, n):
        return jax.lax.fori_loop(0, n, lambda _, s: ma.moment_step(s, ch, alpha)[0], state)
    advance = jax.jit(advance)
    full = advance(old, 13)
    part = advance(old, 5)
    # A checkpoint serialization round trip preserves all histories.
    restored = jax.tree.map(lambda a: jnp.asarray(np.asarray(a).copy()), part)
    resumed = advance(restored, 8)
    for k in full:
        np.testing.assert_array_equal(full[k], resumed[k])
    change = ma.H*(np.abs(full['p'][:7])-np.abs(old['p'][:7]))
    np.testing.assert_allclose(full['positive']-full['negative'], change, atol=1e-16)
    np.testing.assert_allclose(full['signed'].sum(axis=0)+full['crossing'], change, atol=1e-16)
    assert full['crossing'][0] > 0
    assert full['p'][0] < 0


def test_zero_tracking_does_not_erase_old_history():
    old = fixture(); channels = jnp.zeros((3, 22))
    baseline, _ = ma.moment_step(old, channels, jnp.ones(2))
    altered, _ = ma.moment_step(old, channels, jnp.array([.9, .9]))
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], altered[key])
    np.testing.assert_allclose(altered['channel_m'][1], .9*old['channel_m'][1])
    assert np.linalg.norm(altered['channel_m'][1]) > 0


def test_forecast_issuance_and_file_resume(tmp_path):
    old = fixture()
    x = np.linspace(-1, 1, 67)
    pack = {k: np.asarray(old[k])[None] for k in ('p', 'm', 'v', 'channel_m', 'count')}
    pack.update(x=x, y=(np.sin(4*x)+.2)[None], alpha=np.array([[.9, 1.]]),
                cases=np.array(json.dumps([dict(target='test', seed=0)])))
    inputs = tmp_path/'inputs.npz'; ma.save(inputs, **pack)
    predictions = tmp_path/'predictions'
    ma.predict(SimpleNamespace(inputs=inputs, output=predictions, horizon=20))
    with pytest.raises(ValueError, match='immutable'):
        ma.predict(SimpleNamespace(inputs=inputs, output=predictions, horizon=20))
    first = np.load(predictions/'first_step.npz')
    forecast = np.load(predictions/'forecasts.npz')
    np.testing.assert_allclose(first['p'], forecast['frozen_p_1'], atol=1e-16)
    np.testing.assert_allclose(first['p'], forecast['two_phase_p_1'], atol=1e-16)
    args = dict(inputs=inputs, predictions=predictions, backend='cpu', max_seconds=1000)
    ma.run(SimpleNamespace(**args, output=tmp_path/'resumed', horizon=10))
    ma.run(SimpleNamespace(**args, output=tmp_path/'resumed', horizon=20))
    ma.run(SimpleNamespace(**args, output=tmp_path/'full', horizon=20))
    full = np.load(tmp_path/'full/state.npz'); resumed = np.load(tmp_path/'resumed/state.npz')
    for key in full.files:
        np.testing.assert_array_equal(full[key], resumed[key])
