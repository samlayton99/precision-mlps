"""Exact expanded-clone checks for the quotient splitting experiment."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import csv
import json
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import adam_forces as af
from experiments.expD34_readout_race import mechanism_splitting as ms
from experiments.expD34_readout_race import mechanism_splitting_analysis as analysis
from experiments.expD34_readout_race import mechanism_splitting_baselines as baselines
from experiments.expD34_readout_race import transport
from experiments.expD34_readout_race import targets


def example():
    rng = np.random.default_rng(192)
    p = jnp.asarray(rng.normal(size=13)*.4)
    x = jnp.asarray(targets.grid(48))
    return p, x, jnp.sin(2*x)+.1*x*x


@pytest.mark.parametrize('arm', ms.ARMS[1:])
def test_expanded_clone_matches_quotient_for_multiple_steps(arm):
    p, x, y = example()
    k, hidden, readout = ms.settings(arm)
    expanded = ms.expand(p, k)
    mass = jnp.asarray(ms.mobility(4, arm))
    expanded_mass = jnp.asarray(np.r_[np.full(8*k, hidden), np.full(4*k, readout), 1.])
    @jax.jit
    def advance(q, c):
        def step(_, state):
            q, c = state
            return (q-.002*mass*af.field(q, x, y)[0],
                    c-.002*expanded_mass*af.field(c, x, y)[0])
        return jax.lax.fori_loop(0, 31, step, (q, c))
    q, c = advance(p, expanded)
    np.testing.assert_allclose(ms.collapse(c, k), q, atol=3e-15, rtol=3e-14)
    np.testing.assert_allclose(af.field(c, x, y)[1], af.field(q, x, y)[1], atol=3e-15)
    _, channels, _, _ = ms.force(q, x, y, mass)
    _, cloned_channels, _, _ = ms.force(c, x, y, expanded_mass)
    for i in range(3):
        # Directions collapse by summing coefficient velocities, exactly as states.
        np.testing.assert_allclose(ms.collapse(cloned_channels[i], k), channels[i], atol=2e-14)


def test_compensation_preserves_trajectory_and_normalized_motion_accounting():
    p, x, y = example()
    state = jax.vmap(lambda q: ms.initial(q, 1/64))(p[None, :])
    results = []
    for arm in ('original', 'k2_full', 'k4_full'):
        advance = ms.advance_factory(x, jnp.asarray(ms.mobility(4, arm)), 1/64, .002)
        results.append(advance(state, y[None, :], 57))
    for result in results:
        np.testing.assert_array_equal(result['p'], results[0]['p'])
        change = (np.abs(result['p'][0, :4])-np.abs(p[:4]))/64
        np.testing.assert_allclose(result['positive'][0]-result['negative'][0], change, atol=1e-17)
        signed = result['effective']+result['tracking']+result['unresolved']+result['crossing']
        np.testing.assert_allclose(signed[0], change, atol=1e-17)
        assert int(result['count'][0]) == 57
        assert int(result['failed'][0]) == 0


def test_tangent_forecast_matches_explicit_linear_model():
    p, x, y = example()
    mass = ms.mobility(4, 'k4_none')
    forecasts, _ = ms.forecast(np.asarray(p), np.asarray(x), np.asarray(y), mass, .002, [1, 31])
    J = np.asarray(jax.jacfwd(lambda q: af.field(q, x, y)[1])(p))
    r = np.asarray(af.field(p, x, y)[1])
    displacement = np.zeros(len(p))
    for n in range(1, 32):
        displacement -= .002*mass*(J.T@(r+J@displacement)/len(x))
        if n in (1, 31):
            np.testing.assert_allclose(forecasts[0 if n == 1 else 1], p+displacement, atol=2e-14)


def test_full_complement_residual_forcing_matches_jvp():
    p, x, y = example()
    _, channels, _, _ = ms.force(p, x, y, jnp.asarray(ms.mobility(4, 'k2_none')))
    measured, ratio = ms.fine_residual_forcing(p, x, channels)
    q = np.stack((np.ones(len(x)), np.asarray(x)/np.sqrt(np.mean(np.asarray(x)**2))), axis=1)
    expected = []
    for v in channels:
        tangent = np.asarray(jax.jvp(lambda z: af.field(z, x, y)[1], (p,), (v,))[1])
        fine = tangent-q@(q.T@tangent/len(x))
        expected.append(np.sqrt(np.mean(fine*fine)))
    np.testing.assert_allclose(measured, expected, atol=2e-16, rtol=2e-12)
    np.testing.assert_allclose(ratio, expected[1]/expected[0], rtol=2e-12)


def test_analysis_accepts_mixed_target_metadata(tmp_path):
    p, x, y = example()
    pp = np.stack([np.asarray(p)]*2)
    cases = [dict(target='sine', seed=0, start=600000),
             dict(target='gauss_left', seed=22, start=600000, family='gaussian')]
    inputs = tmp_path/'inputs.npz'
    np.savez(inputs, p=pp, x=x, y=np.stack([y, y]), cases=np.array(json.dumps(cases)))
    predictions = tmp_path/'predictions'; predictions.mkdir()
    (predictions/'manifest.json').write_text('{}')
    predicted = np.broadcast_to(pp[None, :, None, :], (len(ms.ARMS), 2, 1, len(p)))
    np.savez(predictions/'predictions.npz', horizons=[1], p=predicted,
             effective_pure_p=predicted, effective_remainder_p=predicted)
    source = tmp_path/'runs'; folder = source/'original'/'snapshots'; folder.mkdir(parents=True)
    zeros = np.zeros((2, 4))
    np.savez(folder/'000000001.npz', p=pp, failed=np.zeros(2), relative_mse=np.ones(2),
             positive=zeros, negative=zeros, effective=zeros, tracking=zeros)
    output = tmp_path/'analysis.csv'
    analysis.analyze(SimpleNamespace(predictions=predictions, inputs=inputs, source=source, output=output))
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]['family'] == '' and rows[1]['family'] == 'gaussian'


def test_checkpoint_baselines_use_metric_balance_and_true_initial_gradient():
    p, x, y = example()
    d = ms.mobility(4, 'k4_none')
    q = transport.basis(np.asarray(x), 9)
    g, T, JH, _ = baselines.matrices(np.asarray(p), np.asarray(x), np.asarray(y), d, q)
    reference, _, JC, _ = af.field(p, x, y)
    np.testing.assert_allclose(g, d*np.asarray(reference), atol=3e-16)
    np.testing.assert_allclose(np.asarray(JC)@T, 0, atol=2e-16)
    S = JH@T
    np.testing.assert_allclose(S, S.T, atol=2e-16)
    assert np.linalg.eigvalsh(S).min() >= -1e-16
