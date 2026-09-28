import numpy as np
import pytest

from experiments.expD34_readout_race import moment_persistence as theorem
from experiments.expD34_readout_race import population_coverage as coverage


def fixture(width=1000):
    p = np.r_[np.full(width, .02), np.full(width, .005), np.full(width, .03), .1]
    x = np.linspace(-1., 1., 17)
    return p, x, .3+np.sin(x)


def test_initial_normalization_and_shift_symmetry():
    p, x, y = fixture()
    initial = theorem.initial_state(p, x, y, .002, .002)
    w = initial['W']
    assert initial['R0'] == pytest.approx(np.sqrt(w*(.02**2+.005**2+.03**2)))
    assert initial['M0'] == pytest.approx(initial['R0']**2)
    assert initial['Es0'] == pytest.approx(w*(.02**2+.03**2))
    shifted = p.copy(); shifted[-1] += .7
    other = theorem.initial_state(shifted, x, y+.7, .002, .002)
    for key in ('R0', 'M0', 'Es0', 'Y0', 'F0', 'z0', 'total_loss0'):
        assert other[key] == pytest.approx(initial[key], abs=1e-13)


def test_conditioning_formula_and_monotone_slacks():
    p, x, y = fixture()
    initial = theorem.initial_state(p, x, y, .002)
    r = 1.1*initial['R0']
    early = theorem.effective_bounds(initial, r, 0.)
    late = theorem.effective_bounds(initial, r, 1e-4)
    assert early['valid']
    assert early['kappa'] == pytest.approx(early['gap']**2)
    assert late['M_star'] >= early['M_star']
    assert late['Es_star'] <= early['Es_star']
    assert late['gap'] <= early['gap']
    assert late['support_slack'] < early['support_slack']


def test_strict_first_exit_and_integer_clock():
    p, x, y = fixture()
    initial = theorem.initial_state(p, x, y, .002, .002)
    result = theorem.evaluate_margin(initial, 1.1)
    assert result['status'] == 'evaluated'
    assert theorem.effective_bounds(initial, result['radius'], result['max_time_lower'])['valid']
    assert not theorem.effective_bounds(initial, result['radius'], result['max_time_upper'])['valid']
    count = result['effective_flow_integer_updates']
    assert theorem.effective_bounds(initial, result['radius'], .002*count/initial['W'])['valid']
    assert not result['ordinary_gd_bound']


def test_failure_is_retained_and_missing_eta_allowed():
    p, x, y = fixture()
    initial = theorem.initial_state(p, x, y, .002)
    assert theorem.evaluate_margin(initial, 1.1)['effective_flow_integer_updates'] is None
    bad = dict(initial, W=1)
    result = theorem.evaluate_margin(bad, 2.)
    assert result['status'] == 'no_initial_region'
    assert result['limiting_condition'] == 'conditioning'
    assert result['max_time_lower'] == 0


def test_zero_force_stationary_case():
    initial = dict(W=1000, h=.002, eta=.002, R0=1., M0=1., Es0=.5, Y0=0., v=1/3, F0=0.)
    result = theorem.evaluate_margin(initial, 1.1)
    assert result['status'] == 'stationary_effective_flow'
    bound = theorem.effective_bounds(initial, 1.1, 100.)
    assert bound['force_bound'] == 0
    assert bound['parameter_travel_bound'] == 0


def test_reject_unverified_spacing_and_outside_domain():
    p, x, y = fixture()
    with pytest.raises(ValueError, match='spacing'):
        theorem.initial_state(p, x, y, np.nan)
    with pytest.raises(ValueError, match='centered inputs'):
        theorem.initial_state(p, x+.01, y, .002)
    asymmetric = x.copy()
    asymmetric[3] += .001; asymmetric[4] -= .001
    with pytest.raises(ValueError, match='symmetric'):
        theorem.initial_state(p, asymmetric, y, .002)


def test_full_fine_energy_dissipation_in_effective_direction():
    x = np.linspace(-1., 1., 41)
    p = np.array([.4, -.2, .1, .03, .6, -.3, .2])
    y = np.sin(2*x)+.3
    field = coverage.geometry(p, x, y)['F']
    epsilon = 1e-4
    plus = theorem.initial_state(p-epsilon*field, x, y, .01)['Y0']**2/2
    minus = theorem.initial_state(p+epsilon*field, x, y, .01)['Y0']**2/2
    np.testing.assert_allclose((plus-minus)/(2*epsilon), -field@field, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize('small_es', [False, True])
def test_independent_exact_field_structure(small_es):
    rng = np.random.default_rng(178)
    w = 33
    xyz = rng.uniform(-.8, .8, (3, w))
    if small_es:
        xyz[[0, 2]] *= .001
    p = np.r_[xyz.ravel()/np.sqrt(w), .13]
    x = np.linspace(-1., 1., 65)
    y = np.sin(2.3*x)+.21*x*x+.3
    initial = theorem.initial_state(p, x, y, .01)
    state = coverage.geometry(p, x, y)
    radius = initial['R0']
    raw_particles = w**1.5*state['raw'][:-1].reshape(3, w)
    assert np.all(np.linalg.norm(raw_particles, axis=0) <= 4*initial['Y0']*radius**2*abs(xyz[0])+2e-10)
    assert np.linalg.norm(state['F']) <= np.linalg.norm(state['raw'])+1e-13
    np.testing.assert_allclose(state['J']@state['F'], 0., atol=1e-12)

    a, b, c = p[:-1].reshape(3, w)
    affine = np.zeros_like(state['J'])
    affine[0, w:2*w], affine[0, 2*w:3*w], affine[0, -1] = c, b, 1.
    affine[1, :w], affine[1, 2*w:3*w] = np.sqrt(initial['v'])*c, np.sqrt(initial['v'])*a
    assert np.linalg.norm(state['J']-affine, ord=2) <= 3*radius**3/w+1e-13

    # τ=t/W: the particle-label metric includes d at its unscaled weight.
    velocity = -w**1.5*state['F'][:-1].reshape(3, w)
    bias_velocity = -w*state['F'][-1]
    label_norm2 = np.sum(velocity**2)/w+bias_velocity**2
    np.testing.assert_allclose(label_norm2, w*w*(state['F']@state['F']), rtol=1e-12)
    moment_rate = 2*np.sum(xyz*velocity)/w
    es_rate = 2*np.sum(xyz[[0, 2]]*velocity[[0, 2]])/w
    assert abs(moment_rate) <= 8*initial['Y0']*radius**2*initial['M0']+1e-12
    assert abs(es_rate) <= 8*initial['Y0']*radius**2*initial['Es0']+1e-12


def test_effective_field_derivative_against_regional_bound():
    rng = np.random.default_rng(291)
    w = 33
    xyz = rng.uniform(-.2, .2, (3, w))
    p = np.r_[xyz.ravel()/np.sqrt(w), .1]
    x = np.linspace(-1., 1., 65)
    y = np.sin(2*x)+.2*x*x
    initial = theorem.initial_state(p, x, y, .01)
    bound = theorem.effective_bounds(initial, 1.1*initial['R0'], 0.)
    assert bound['valid']
    direction = rng.normal(size=len(p)); direction /= np.linalg.norm(direction)
    derivatives = []
    for epsilon in (1e-5, 5e-6):
        plus = coverage.geometry(p+epsilon*direction, x, y)['F']
        minus = coverage.geometry(p-epsilon*direction, x, y)['F']
        derivatives.append(w*(plus-minus)/(2*epsilon))
    np.testing.assert_allclose(derivatives[0], derivatives[1], rtol=2e-5, atol=1e-8)
    assert np.linalg.norm(derivatives[1]) <= bound['L']+1e-8


def test_gd_recurrence_uses_old_tracking_and_strict_exit():
    p, x, y = fixture()
    initial = theorem.initial_state(p, x, y, .002, .0001)
    constants = theorem.gd_constants(initial, 1.1)
    assert constants['valid']
    state = dict(q=0., m=np.sqrt(initial['M0']), s=np.sqrt(initial['Es0']), r=initial['R0'], u=initial['F0'])
    following = theorem.gd_step(state, constants)
    assert following['q'] > 0
    assert following['m'] == pytest.approx((1+constants['a'])*state['m'])
    assert following['s'] == pytest.approx((1-constants['a'])*state['s'])
    assert theorem.gd_exit_reason(dict(state, r=constants['radius']), constants) == 'support'
    result, records = theorem.evaluate_gd_margin(initial, 1.1, (0, 1))
    assert result['valid_updates'] >= 0
    assert 0 in records
    endpoint = {key: result['endpoint_'+key] for key in ('q', 'm', 's', 'r', 'u')}
    assert theorem.gd_exit_reason(endpoint, constants) is None
    assert theorem.gd_exit_reason(theorem.gd_step(endpoint, constants), constants) is not None


def test_gd_metadata_and_step_rejection_are_explicit():
    p, x, y = fixture()
    initial = theorem.initial_state(p, x, y, .002)
    assert theorem.gd_constants(initial, 1.1)['reason'] == 'missing_eta'
    assert theorem.gd_constants(dict(initial, eta=100.), 1.1)['reason'] == 'step_size'


def test_force_coupled_one_step_contains_actual_mixed_state():
    rng = np.random.default_rng(821)
    w = 33
    xyz = rng.uniform(-.2, .2, (3, w))
    p = np.r_[xyz.ravel()/np.sqrt(w), .13]
    x = np.linspace(-1., 1., 65)
    y = np.sin(2.3*x)+.21*x*x+.3
    eta = 1e-5
    initial = theorem.initial_state(p, x, y, .01, eta)
    constants = theorem.gd_constants(initial, 1.1)
    assert constants['valid']
    state = dict(q=initial['z0'], m=np.sqrt(initial['M0']), s=np.sqrt(initial['Es0']), r=initial['R0'], u=initial['F0'])
    bound = theorem.gd_step(state, constants, force_coupled=True)
    assert theorem.gd_exit_reason(bound, constants) is None
    # All recurrences use OLD q and u; q_next is not fed back into this step.
    assert bound['m'] == pytest.approx(state['m']+eta*(state['u']+constants['J']*state['q']))
    decomposition = coverage.geometry(p, x, y)
    gradient = decomposition['F']+decomposition['R']
    for fraction in (.25, .5, 1.):
        actual = theorem.initial_state(p-fraction*eta*gradient, x, y, .01, eta)
        assert actual['R0'] <= bound['r']+1e-12
        assert np.sqrt(actual['M0']) <= bound['m']+1e-12
        assert np.sqrt(actual['Es0']) >= bound['s']-1e-12
        assert actual['coarse_k0'] >= constants['kappa']-1e-12
    assert actual['z0'] <= bound['q']+1e-12
    assert actual['F0'] <= bound['u']+1e-12
    primary, _ = theorem.evaluate_gd_margin(dict(initial, eta=None), 1.1)
    refined, _ = theorem.evaluate_gd_margin(dict(initial, eta=None), 1.1, force_coupled=True)
    assert primary['recurrence_variant'] == 'primary'
    assert refined['recurrence_variant'] == 'force_coupled'
