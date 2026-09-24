"""Finite locked-geometry repairs; run on CPU with JAX_ENABLE_X64=1."""
import json

import numpy as np
import pytest

from experiments.expD34_readout_race import mechanism_dilation_repair as repair


def numpy_state(p, x, y):
    """Independent full-complement Jacobian construction."""
    a, b, c = p[:-1].reshape(3, -1)
    h = np.tanh(x[:, None]*a+b)
    derivative = 1-h*h
    jacobian = np.column_stack((derivative*c*x[:, None], derivative*c,
                               h, np.ones(len(x))))
    centered = x-x.mean()
    basis = np.column_stack((np.ones(len(x)), centered/np.sqrt(np.mean(centered**2))))
    residual = h@c+p[-1]-y
    coarse = basis.T@residual/len(x)
    fine = residual-basis@coarse
    jc = basis.T@jacobian/len(x)
    raw = jacobian.T@fine/len(x)
    balance = np.linalg.solve(jc@jc.T, jc@raw)
    return coarse+balance, raw-jc.T@balance, jc


def test_constraint_and_derivative_use_full_complement():
    rng = np.random.default_rng(847)
    p = rng.normal(size=19)*.3
    x = np.linspace(-1, 1, 65)
    y = np.sin(2.3*x)+.07*x*x
    w = 6
    expected, _, _ = numpy_state(p, x, y)
    state = repair._evaluate(p[2*w:], p[:2*w], x, y)
    np.testing.assert_allclose(state['z'], expected, atol=2e-14)
    direction = rng.normal(size=w+1)
    epsilon = 1e-5
    plus, minus = p.copy(), p.copy()
    plus[2*w:] += epsilon*direction
    minus[2*w:] -= epsilon*direction
    numerical = (numpy_state(plus, x, y)[0]-numpy_state(minus, x, y)[0])/(2*epsilon)
    autodiff = np.asarray(repair._jacobian(p[2*w:], p[:2*w], x, y))@direction
    np.testing.assert_allclose(autodiff, numerical, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize('scale', [1., 1.25, 2.])
@pytest.mark.parametrize('reference', ['primary', 'inverse'])
def test_exact_intercept_repair_locks_geometry(scale, reference):
    # Odd features have zero intercept; c=0 and d=.5 is an exact nearest repair.
    p = np.r_[np.array([.2, -.3, .5]), np.zeros(6), 0.]
    x = np.linspace(-1, 1, 65)
    y = np.full_like(x, .5)
    geometry = scale*p[:6]
    vref = p[6:].copy()
    if reference == 'inverse':
        vref[:3] /= scale
    v, info = repair._stage(vref, geometry, vref, x, y, .5)
    point = np.r_[geometry, v]
    np.testing.assert_array_equal(point[:6], scale*p[:6])
    np.testing.assert_allclose(point[6:], [0., 0., 0., .5], atol=2e-14)
    assert info['balance_norm'] <= .5*repair.BALANCE_TOLERANCE


def test_zero_fine_force_does_not_bypass_tracking_gate(monkeypatch):
    # Deterministic roundoff-scale tracking: tiny absolutely, but larger than
    # the allowed relative fine-force budget. Do not weaken the campaign gate.
    p = np.r_[[.2, -.3, .5], np.zeros(7)]
    x = np.linspace(-1, 1, 33)
    y = np.full_like(x, .5)
    original_evaluate = repair._evaluate
    def with_roundoff_tracking(v, geometry, x, y):
        state = original_evaluate(v, geometry, x, y)
        state['F'] = np.zeros_like(state['F'])
        state['R'] = np.full_like(state['R'], 1e-30)
        return state
    monkeypatch.setattr(repair, '_evaluate', with_roundoff_tracking)
    point, info = repair.repair_case(p, x, y, 1.)
    assert not info['valid']
    assert info['reason'] == 'initial_tracking_contamination'
    np.testing.assert_array_equal(point, p)


@pytest.mark.parametrize('reference', ['primary', 'inverse'])
def test_nonlinear_repair_is_balanced_and_stationary(reference):
    p = np.r_[[.2, -.3, .5], [.1, -.05, .08], [.4, -.2, .3], .02]
    x = np.linspace(-1, 1, 81)
    y = np.sin(1.3*x)+.05*x*x
    point, info = repair.repair_case(p, x, y, 1.25, reference)
    assert info['valid'], json.dumps(info, indent=2)
    np.testing.assert_array_equal(point[:6], 1.25*p[:6])
    z, force, jc = numpy_state(point, x, y)
    assert np.linalg.norm(z) <= 2e-12*np.sqrt(np.mean(y*y))
    base_force = numpy_state(p, x, y)[1]
    assert np.linalg.norm((jc.T@z)[:3]) <= .001*max(np.linalg.norm(force[:3]),
                                                               np.linalg.norm(base_force[:3]))
    # Tangent stationarity is checked independently with finite differences.
    derivative = np.empty((2, 4))
    for index in range(4):
        plus, minus = point.copy(), point.copy()
        plus[6+index] += 1e-5
        minus[6+index] -= 1e-5
        derivative[:, index] = (numpy_state(plus, x, y)[0]-numpy_state(minus, x, y)[0])/2e-5
    vref = p[6:].copy()
    if reference == 'inverse':
        vref[:3] /= 1.25
    displacement = point[6:]-vref
    tangent = displacement-derivative.T@np.linalg.solve(derivative@derivative.T,
                                                       derivative@displacement)
    assert np.linalg.norm(tangent) < 2e-7*max(np.linalg.norm(vref), np.linalg.norm(displacement))


def test_zero_target_exact_state_is_not_excluded():
    p = np.r_[[.2, -.3, .5], np.zeros(7)]
    x = np.linspace(-1, 1, 33)
    point, info = repair.repair_case(p, x, np.zeros_like(x), 1.25)
    assert info['valid'], info
    assert info['target_normalizer'] == 1e-12


def test_unresolved_and_failed_repairs_are_not_promoted(monkeypatch):
    x = np.linspace(-1, 1, 33)
    y = np.sin(x)
    point, info = repair.repair_case(np.zeros(10), x, y, 2.)
    assert not info['valid']
    assert info['reason'] == 'unresolved_coarse_gram'
    p = np.r_[[.2, -.3, .5], np.zeros(6), 0.]
    monkeypatch.setattr(repair, 'MAX_ITERATIONS', 0)
    point, info = repair.repair_case(p, x, np.full_like(x, .5), 2.)
    assert not info['valid']
    assert info['reason'] == 'iteration_limit'
    np.testing.assert_array_equal(point[:6], 2*p[:6])
    np.testing.assert_array_equal(point[6:], p[6:])


def test_prepare_retains_all_six_attempts_and_source_metadata(tmp_path, monkeypatch):
    p = np.arange(10, dtype=float)[None, :]/10
    x = np.linspace(-1, 1, 5)
    source = tmp_path/'source.npz'
    output = tmp_path/'prepared.npz'
    case = dict(target='test', seed=1, start=20000, cohort='early', Nref=128, h=2/128)
    np.savez(source, p=p, x=x, y=x[None, :], cases=np.array(json.dumps([case])),
             x_eval=x, y_eval=x[None, :])
    def fake_repair(p, x, y, scale, reference, continuation):
        point = p.copy()
        point[:6] *= scale
        return point, dict(valid=scale != 2, reason='fixture')
    monkeypatch.setattr(repair, 'repair_case', fake_repair)
    repair.prepare(source, output)
    with np.load(output) as archive:
        cases = json.loads(str(archive['cases']))
        assert archive['p'].shape == (6, 10)
        assert archive['y_eval'].shape == (6, 5)
        assert [item['arm'] for item in cases] == ['original', 'repaired', 's125_primary',
                                                 's2_primary', 's125_inverse', 's2_inverse']
        assert [item['valid'] for item in cases] == [True, True, True, False, True, False]
        assert all(item['original_case_id'] == 0 for item in cases)
        assert all(item['Nref'] == 128 and item['cohort'] == 'early' for item in cases)


def test_log_continuation_keeps_final_reference(monkeypatch):
    p = np.r_[[.2, -.3, .5], [.1, -.05, .08], [.4, -.2, .3], .02]
    x = np.linspace(-1, 1, 33)
    seen = []
    stage = repair._stage
    def wrapped(v, geometry, reference, x, y, target_rms):
        seen.append(reference.copy())
        return stage(v, geometry, reference, x, y, target_rms)
    monkeypatch.setattr(repair, '_stage', wrapped)
    point, info = repair.repair_case(p, x, np.sin(1.3*x), 10., 'inverse', 'log')
    assert info['valid'], info
    np.testing.assert_array_equal(point[:6], 10*p[:6])
    assert len(info['stages']) < 40
    for reference in seen:
        np.testing.assert_array_equal(reference, np.r_[p[6:9]/10, p[9]])
