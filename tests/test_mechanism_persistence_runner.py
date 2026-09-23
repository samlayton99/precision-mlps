"""Small CPU checks of the actual runner's updates and travel bookkeeping."""
from argparse import Namespace
import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.expD34_readout_race import mechanism_persistence as runner
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel

jax.config.update('jax_enable_x64', True)


def issued(tmp_path, points, x, labels, references, arms, scales):
    path = tmp_path/'issued'
    path.mkdir()
    cases = [dict(source_index=i, arm='natural', h=float(scales[i])) for i in range(len(points))]
    np.savez(path/'inputs.npz', p=points, x=x, y=labels,
             reference_F0=references['F0'], reference_eH0=references['eH0'],
             arms=arms, h=scales, cases=np.array(json.dumps(cases)))
    np.savez(path/'forecasts.npz', unused=np.array(0))
    (path/'manifest.json').write_text(json.dumps(dict(source_hashes=runner.hashes(),
        input_sha256=runner.io.digest(path/'inputs.npz'),
        forecast_sha256=runner.io.digest(path/'forecasts.npz'), cases=cases)))
    return path


def execute(tmp_path, path, steps=2):
    out = tmp_path/'run'
    runner.run(Namespace(backend='cpu', predictions=path, output=out, steps=steps))
    return {n: dict(np.load(out/'snapshots'/f'{n:09d}.npz')) for n in (0, 1, 2) if n <= steps}


def test_runner_first_step_matches_all_arms_and_natural_second_step(tmp_path):
    x = jnp.linspace(-1., 1., 17)
    p = jnp.asarray([.31, -.42, .12, -.18, .38, -.27, .17])
    y = jnp.sin(2.4*x)+.3*x*x+.1
    s = kernel.decomposition(p, x, y)
    count = len(kernel.ARMS)
    points = np.repeat(np.asarray(p)[None], count, axis=0)
    labels = np.repeat(np.asarray(y)[None], count, axis=0)
    refs = dict(F0=np.repeat(np.asarray(s['F'])[None], count, axis=0),
                eH0=np.repeat(np.asarray(s['eH'])[None], count, axis=0))
    path = issued(tmp_path, points, x, labels, refs, kernel.ARMS, np.full(count, 1/64))
    data = execute(tmp_path, path)
    independent_gradient = jax.grad(lambda v: jnp.mean((kernel.output(v, x)-y)**2)/2)
    first = p-runner.ETA*independent_gradient(p)
    np.testing.assert_allclose(data[1]['p'], np.repeat(np.asarray(first)[None], count, axis=0), atol=2e-15)
    second = first-runner.ETA*independent_gradient(first)
    np.testing.assert_allclose(data[2]['p'][0], second, atol=2e-15)
    for n in (1, 2):
        change = (np.abs(data[n]['p'][:, :2])-np.abs(points[:, :2]))/64
        np.testing.assert_allclose(data[n]['positive']-data[n]['negative'], change, atol=2e-17)
        np.testing.assert_allclose(data[n]['signed_effective']+data[n]['signed_tracking']+data[n]['crossing'],
                                   change, atol=2e-17)
        assert np.all(data[n]['count'] == n)
        assert not data[n]['failed'].any()


def test_runner_charges_sign_crossings_and_first_hits(tmp_path, monkeypatch):
    # Both slopes cross zero on update one without changing magnitude;
    # on update two they pass the normalized acquisition threshold.
    p = np.array([[.1, -.2, 0., 0., .3, .4, 0.]])
    x = np.linspace(-1, 1, 5)
    g = jnp.asarray([100., -200., 0., 0., 0., 0., 0.])
    def components(point, xx, yy, ref, arm):
        return g, .75*g, .75*g, .25*g
    monkeypatch.setattr(kernel, 'step_components', components)
    refs = dict(F0=np.zeros_like(p), eH0=np.zeros((1, len(x))))
    path = issued(tmp_path, p, x, np.zeros((1, len(x))), refs, [[1., 1.]], [1.])
    data = execute(tmp_path, path)
    np.testing.assert_allclose(data[1]['crossing'], [[.2, .4]], atol=2e-15)
    np.testing.assert_allclose(data[1]['positive'], 0., atol=2e-15)
    np.testing.assert_array_equal(data[1]['first_hit'], [[-1, -1]])
    np.testing.assert_array_equal(data[2]['first_hit'], [[2, 2]])
    np.testing.assert_allclose(data[2]['positive'], [[.2, .4]], atol=2e-15)
    change = np.abs(data[2]['p'][:, :2])-np.abs(p[:, :2])
    np.testing.assert_allclose(data[2]['signed_effective']+data[2]['signed_tracking']+data[2]['crossing'], change, atol=2e-15)
    np.testing.assert_allclose(data[2]['travel'], 2*runner.ETA*np.linalg.norm(g), atol=2e-15)
    np.testing.assert_allclose(data[2]['action'], 2*runner.ETA*np.sum(np.asarray(.75*g)**2), atol=2e-13)


def test_runner_freezes_failed_states_and_does_not_count_bad_updates(tmp_path, monkeypatch):
    p = np.array([[.1, -.2, 0., 0., .3, .4, 0.]])
    x = np.linspace(-1, 1, 5)
    def components(point, xx, yy, ref, arm):
        g = jnp.full_like(point, jnp.inf)
        return g, g, g, jnp.zeros_like(point)
    monkeypatch.setattr(kernel, 'step_components', components)
    refs = dict(F0=np.zeros_like(p), eH0=np.zeros((1, len(x))))
    path = issued(tmp_path, p, x, np.zeros((1, len(x))), refs, [[1., 1.]], [1.])
    data = execute(tmp_path, path)
    for n in (1, 2):
        np.testing.assert_array_equal(data[n]['p'], p)
        np.testing.assert_array_equal(data[n]['count'], [0])
        np.testing.assert_array_equal(data[n]['failed'], [True])
        np.testing.assert_array_equal(data[n]['positive'], np.zeros((1, 2)))
        np.testing.assert_array_equal(data[n]['travel'], [0.])
