"""Small CPU integration checks for the effective-feedback experiment runner."""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import effective_feedback as run
from experiments.expD34_readout_race import effective_feedback_kernel as kernel
from experiments.expD34_readout_race import targets


def example():
    rng = np.random.default_rng(317)
    pp = rng.normal(size=(2, 16))*.7  # Five neurons and one output bias.
    x = targets.grid(64)
    yy = np.stack([np.sin(2*np.pi*x), np.cos(np.pi*x)+.1*x])
    return jnp.asarray(pp), x, yy, run.context_for(pp, x, yy, 9)


def single_context(context, i):
    return {k: v if k in ('x', 'q') else v[i] for k, v in context.items()}


def test_first_update_is_bitwise_common_and_second_update_changes_only_slopes():
    pp, _, _, context = example()
    eta = .002
    states = jax.vmap(run.initial)(pp)
    first = {arm: run.advance_factory(arm, eta)(states, context, 1) for arm in run.ARMS}
    for arm in run.ARMS:
        for key in states:
            if key in ('effective', 'tracking', 'omitted'):
                # XLA may reassociate auxiliary products across the conditional;
                # parameter updates and realized travel must still be identical.
                np.testing.assert_allclose(first[arm][key], first['joint'][key], atol=5e-18, rtol=5e-13)
            else:
                np.testing.assert_array_equal(first[arm][key], first['joint'][key])
    second = {arm: run.advance_factory(arm, eta)(first[arm], context, 1) for arm in run.ARMS}
    width = (pp.shape[1]-1)//3
    for i in range(len(pp)):
        p1 = first['joint']['p'][i]
        ctx = single_context(context, i)
        _, channels = kernel.field(p1, ctx)
        T0 = np.asarray(ctx['T_a0'])
        T1 = np.asarray(kernel.matrices(p1, ctx)['T_a'])
        e0, e1 = np.asarray(ctx['eH0']), np.asarray(channels['eH'])
        for arm, expected in [
            ('freeze_map', -eta*((T0-T1) @ e1)),
            ('clamp_residual', -eta*(T1 @ (e0-e1)))]:
            delta = second[arm]['p'][i]-second['joint']['p'][i]
            np.testing.assert_array_equal(delta[width:], np.zeros_like(delta[width:]))
            np.testing.assert_allclose(delta[:width], expected, atol=2e-16, rtol=1e-8)


@pytest.mark.parametrize('arm', run.ARMS)
def test_chunk_resume_preserves_own_state_forces_and_all_accumulators(arm):
    pp, _, _, context = example()
    eta = .002
    state = jax.vmap(run.initial)(pp)
    advance = run.advance_factory(arm, eta)
    uninterrupted = advance(state, context, 7)
    resumed = advance(advance(state, context, 3), context, 4)
    for key in state:
        np.testing.assert_array_equal(resumed[key], uninterrupted[key])
    for i in range(len(pp)):
        p = pp[i]
        ctx = single_context(context, i)
        for step in range(7):
            gradient, _ = kernel.field(p, ctx, 'joint' if step == 0 else arm)
            p = p-eta*gradient
        np.testing.assert_allclose(resumed['p'][i], p, rtol=3e-13, atol=2e-15)
    width = (pp.shape[1]-1)//3
    displacement = np.abs(resumed['p'][:, :width])-np.abs(pp[:, :width])
    np.testing.assert_allclose(resumed['positive']-resumed['negative'], displacement, atol=2e-16)
    attributed = sum(resumed[k] for k in ('effective', 'tracking', 'omitted', 'crossing'))
    np.testing.assert_allclose(attributed, displacement, atol=2e-15)


def test_crossings_and_initial_threshold_occupancy_are_retained():
    pp, x, yy, _ = example()
    pp = pp.at[0, 0].set(0.).at[0, 1].set(3.5)
    context = run.context_for(pp, x, yy, 9)
    old = jax.vmap(run.initial)(pp)
    eta = 100.
    new = run.advance_factory('joint', eta)(old, context, 1)
    width = (pp.shape[1]-1)//3
    for i in range(len(pp)):
        p = pp[i]
        grad, _ = kernel.field(p, single_context(context, i))
        a = np.asarray(p[:width]); change = -eta*np.asarray(grad[:width]); after = a+change
        crosses = a*after < 0
        assert crosses.any()
        expected = np.where(a == 0, abs(change), np.where(crosses, 2*abs(after), 0.))
        np.testing.assert_allclose(new['crossing'][i], expected, atol=2e-14, rtol=3e-14)
    assert int(old['first_hit'][0, 1, 1]) == 0
    assert int(new['first_hit'][0, 1, 1]) == 0
    for ti, threshold in enumerate(run.THRESHOLDS):
        initially = np.abs(pp[:, :width]) >= threshold
        after = np.abs(new['p'][:, :width]) >= threshold
        expected_hit = np.where(initially, 0, np.where(after, 1, -1))
        np.testing.assert_array_equal(new['first_hit'][:, ti], expected_hit)


def test_failed_projection_stops_only_its_case_and_failure_is_sticky():
    pp, x, yy, _ = example()
    pp = pp.at[1].set(jnp.zeros(pp.shape[1]))
    context = run.context_for(pp, x, yy, 9)
    old = jax.vmap(run.initial)(pp)
    advance = run.advance_factory('joint', .002)
    new = advance(old, context, 3)
    np.testing.assert_array_equal(new['failed'], [0, 2])
    np.testing.assert_array_equal(new['count'], [3, 0])
    np.testing.assert_array_equal(new['p'][1], pp[1])
    again = advance(new, context, 2)
    for key in old:
        np.testing.assert_array_equal(again[key][1], new[key][1])
    assert int(again['count'][0]) == 5


def test_nonfinite_proposal_stops_before_updating_state():
    pp, x, yy, _ = example()
    yy[1, 0] = np.nan
    context = run.context_for(pp, x, yy, 9)
    old = jax.vmap(run.initial)(pp)
    new = run.advance_factory('joint', .002)(old, context, 2)
    np.testing.assert_array_equal(new['failed'], [0, 1])
    np.testing.assert_array_equal(new['count'], [2, 0])
    np.testing.assert_array_equal(new['p'][1], old['p'][1])


def test_disk_resume_matches_uninterrupted_run_and_preserves_fork(tmp_path):
    pp, x, yy, _ = example()
    inputs = tmp_path/'inputs.npz'
    cases = [dict(target='sine', seed=i, start=100000, eta=.002, width=5, target_scale=1.) for i in range(2)]
    np.savez(inputs, p=np.asarray(pp), x=x, y=yy, cases=np.array(json.dumps(cases)), sources=np.array('{}'))
    def arguments(output, horizon):
        return SimpleNamespace(inputs=inputs, output=output, indices=None, arm='freeze_map',
            degree=9, eta=.002, horizon=horizon, stride=1000, max_seconds=60,
            backend='cpu', predictions='unit-test-verification-only')
    interrupted, reference = tmp_path/'resumed', tmp_path/'reference'
    run.run(arguments(interrupted, 2))
    run.run(arguments(interrupted, 5))
    run.run(arguments(reference, 5))
    with np.load(interrupted/'state.npz') as actual, np.load(reference/'state.npz') as expected:
        assert set(actual.files) == set(expected.files)
        for key in actual.files:
            np.testing.assert_array_equal(actual[key], expected[key])
    status = json.loads((interrupted/'status.json').read_text())
    assert status['complete'] and status['valid'] and status['offset'] == 5
    assert status['motion_identity'] < 1e-14 and status['signed_identity'] < 1e-14
    with np.load(interrupted/'snapshots'/'000000000.npz') as data:
        np.testing.assert_array_equal(data['p'], pp)
        assert 'metric_effective_a' in data.files
