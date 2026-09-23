"""Matched physical pulse verification; run with JAX_ENABLE_X64=1 on CPU."""
import csv
import json

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import mechanism_persistence_pulses as pulses


def numpy_values(p, x, y):
    """Independent empirical Jacobian and full-complement projection."""
    w = (len(p)-1)//3
    a, b, c = p[:-1].reshape(3, w)
    feature = np.tanh(x[:, None]*a+b)
    derivative = 1-feature**2
    residual = feature@c+p[-1]-y
    jacobian = np.column_stack((derivative*x[:, None]*c, derivative*c,
                               feature, np.ones(len(x))))
    centered = x-x.mean()
    basis = np.column_stack((np.ones(len(x)), centered/np.sqrt(np.mean(centered**2))))
    jc = basis.T@jacobian/len(x)
    ec = basis.T@residual/len(x)
    gradient = jacobian.T@residual/len(x)
    raw = gradient-jc.T@ec
    balance = np.linalg.solve(jc@jc.T, jc@raw)
    force = raw-jc.T@balance
    return dict(coarse=ec, z=ec+balance, gradient=gradient, F=force,
                matched=np.r_[ec, ec+balance, gradient[:w], force@force])


@pytest.fixture(scope='module')
def example():
    rng = np.random.default_rng(233)
    p = rng.normal(size=31)*.3
    x = np.linspace(-.99, .99, 80)
    y = np.sin(3*x)+.1*x*x
    system = pulses.local_system(p, x, y)
    direction, info = pulses.matched_direction(p, system)
    return p, x, y, system, direction, info


def test_slope_elimination_and_constrained_optimum():
    rng = np.random.default_rng(75)
    p = rng.normal(size=25)
    matrix = rng.normal(size=(7, 25))
    objective = rng.normal(size=25)
    direction, info = pulses.matched_direction(p, dict(
        constraints=matrix, objective_gradient=objective))
    assert np.array_equal(direction[:8], np.zeros(8))
    np.testing.assert_allclose(matrix@direction, 0, atol=2e-13)
    assert info['intrinsic_k_derivative'] > 0
    np.testing.assert_allclose(info['scaled_rms'], 1, atol=3e-15)
    # Projecting onto the constraint row space must remove all feasible gain.
    with pytest.raises(ValueError, match='No numerically resolved'):
        pulses.matched_direction(p, dict(constraints=matrix,
            objective_gradient=matrix.T@rng.normal(size=7)))


def test_independent_force_values_and_direction_constraints(example):
    p, x, y, system, direction, info = example
    reference = numpy_values(p, x, y)
    np.testing.assert_allclose(system['base']['F'], reference['F'], atol=2e-15, rtol=2e-12)
    np.testing.assert_allclose(system['base']['z'], reference['z'], atol=2e-15)
    np.testing.assert_allclose(system['constraints']@direction, 0, atol=2e-12)
    assert np.array_equal(direction[:10], np.zeros(10))
    assert info['normalized_constraint_residual'] < 2e-11
    assert info['intrinsic_k_derivative'] > 0


def test_physical_matching_is_quadratic_and_initial_slopes_exact(example):
    p, x, y, _, direction, _ = example
    baseline = numpy_values(p, x, y)['matched']
    # Four distinct constraints: coarse output, z, actual g_a, and ||F||².
    groups = (slice(0, 2), slice(2, 4), slice(4, 14), slice(14, 15))
    for sign in (-1, 1):
        errors = []
        for amplitude in (.01, .005, .0025):
            point = p+sign*amplitude*direction
            assert np.array_equal(point[:10], p[:10])
            delta = numpy_values(point, x, y)['matched']-baseline
            errors.append([np.linalg.norm(delta[block]) for block in groups])
        errors = np.asarray(errors)
        # Three amplitudes distinguish a quadratic mismatch from a linear kick.
        np.testing.assert_allclose(errors[:-1]/errors[1:], 4., rtol=.12)


def test_intrinsic_k_direction_matches_a_finite_difference(example):
    p, x, y, system, direction, info = example
    step = 1e-4
    plus = pulses.observables(jnp.asarray(p+step*direction), jnp.asarray(x), jnp.asarray(y))
    minus = pulses.observables(jnp.asarray(p-step*direction), jnp.asarray(x), jnp.asarray(y))
    derivative = float((plus['diagnostics']['k_pure']-minus['diagnostics']['k_pure'])/(2*step))
    np.testing.assert_allclose(derivative, info['intrinsic_k_derivative'], rtol=2e-4, atol=1e-12)
    # Independently evaluate k=-F^T DF[F]/||F||², without the D/C split.
    force = numpy_values(p, x, y)['F']
    df_force = (numpy_values(p+step*force, x, y)['F']-
                numpy_values(p-step*force, x, y)['F'])/(2*step)
    expected = -(force@df_force)/(force@force)
    np.testing.assert_allclose(float(system['base']['diagnostics']['k_pure']),
                               expected, rtol=3e-6, atol=1e-11)


def test_unresolved_case_keeps_its_baseline(monkeypatch):
    def fail(*args):
        raise ValueError('No numerically resolved amplification direction')
    monkeypatch.setattr(pulses, 'local_system', fail)
    p = np.linspace(.1, .7, 7)
    states, metadata = pulses.build_family(p, np.linspace(-1, 1, 8), np.zeros(8))
    assert metadata['status'] == 'unresolved'
    assert metadata['amplitudes'] == (0.,)
    np.testing.assert_array_equal(states, p[None])


def test_prepare_preserves_all_cases_and_evaluation_data(tmp_path, monkeypatch):
    p = np.arange(14, dtype=float).reshape(2, 7)/10
    x = np.linspace(-.9, .9, 8)
    y = np.stack((x, x*x))
    source = tmp_path/'source.npz'
    cases = [dict(target='first', seed=30, start=20000, nref=128),
             dict(target='second', seed=32, start=20000, nref=128)]
    np.savez(source, p=p, x=x, y=y, x_eval=x, y_eval=y,
             cases=np.array(json.dumps(cases)))
    def family(point, xx, yy):
        if point[0] > 0:
            return point[None], dict(direction=np.zeros_like(point), amplitudes=(0.,),
                status='unresolved', reason='synthetic unresolved objective', pulses=[])
        direction = np.r_[0., 0., np.ones(5)]
        return point+np.asarray(pulses.AMPLITUDES)[:, None]*direction, dict(
            direction=direction, amplitudes=pulses.AMPLITUDES, status='resolved', pulses=[])
    monkeypatch.setattr(pulses, 'build_family', family)
    out = tmp_path/'prepared'
    manifest = pulses.prepare(source, out)
    assert manifest['source_cases'] == manifest['retained_baselines'] == 2
    assert manifest['resolved_cases'] == 1
    with np.load(out/'inputs.npz') as saved:
        metadata = json.loads(str(saved['cases']))
        assert len(metadata) == 8
        np.testing.assert_array_equal(saved['p'][0], p[0])
        np.testing.assert_array_equal(saved['p'][7], p[1])
        np.testing.assert_array_equal(saved['y_eval'], y[[0]*7+[1]])
        assert metadata[7]['reference_baseline_index'] == 7
    filtered = tmp_path/'filtered'
    selection = pulses.prepare(source, filtered, seed=32, targets='second')
    assert selection['selected_source_indices'] == [1]
    with np.load(filtered/'inputs.npz') as saved:
        np.testing.assert_array_equal(saved['p'], p[1:])
        np.testing.assert_array_equal(saved['y_eval'], y[1:])
        assert json.loads(str(saved['cases']))[0]['source_index'] == 1
    with pytest.raises(ValueError, match='No source cases'):
        pulses.prepare(source, tmp_path/'none', seed=30, targets='second')


def test_analysis_uses_own_initial_states_and_antisymmetric_responses(tmp_path):
    predictions, run = tmp_path/'predictions', tmp_path/'run'
    predictions.mkdir(); (run/'snapshots').mkdir(parents=True)
    amplitudes = np.asarray(pulses.AMPLITUDES)
    p0 = np.ones(7)+amplitudes[:, None]*np.array([0., 0., 2., -1., 4., 3., 5.])
    response = np.array([.2, -.1, .3, .4, -.5, .6, .7])
    movement = .01+amplitudes[:, None]*response+amplitudes[:, None]**2*.2
    final = p0+movement
    q0 = 1+amplitudes**2
    q1 = q0+.1+.4*amplitudes
    cases = [dict(target='synthetic', seed=30, start=20000, arm='natural',
                  source_index=i, parent_source_index=0, reference_baseline_index=0,
                  amplitude=float(amplitude), h=.1)
             for i, amplitude in enumerate(amplitudes)]
    force = np.zeros_like(p0); force[:, 0] = q0
    np.savez(predictions/'inputs.npz', p=p0, reference_F0=force,
             cases=np.array(json.dumps(cases)))
    np.savez(predictions/'forecasts.npz', amplification=np.stack((p0, final), axis=1),
             q=np.stack((q0, q1), axis=1), valid=np.ones((7, 2), dtype=bool),
             horizons=np.array([0, 2]))
    for n, state, norm in ((0, p0, q0), (2, final, q1)):
        sampled = np.zeros((7, 5)); sampled[:, 0] = norm
        np.savez(run/'snapshots'/f'{n:09d}.npz', p=state, sampled=sampled,
                 failed=np.zeros(7, dtype=bool))
    (predictions/'manifest.json').write_text('{}')
    (run/'manifest.json').write_text('{}')
    out = tmp_path/'analysis'
    summary = pulses.analyze(predictions, run, out)
    assert summary['pulse_contrasts'] == 12
    assert summary['paired_responses'] == 6
    with (out/'paired.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['horizon'] == '2']
    for row in rows:
        np.testing.assert_allclose(float(row['full_actual_norm']), np.linalg.norm(response), atol=2e-13)
        np.testing.assert_allclose(float(row['readout_actual_norm']), np.linalg.norm(response[4:6]), atol=2e-13)
        np.testing.assert_allclose(float(row['actual_q_change_response']), .4, atol=1e-13)
        assert float(row['full_error_norm']) == 0
        assert float(row['initial_mean_lambda_difference']) == 0
