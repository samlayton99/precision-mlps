import numpy as np

from experiments.expD34_readout_race import population_coverage as audit


def fixture():
    x = np.linspace(-1., 1., 41)
    p = np.array([.12, -.08, .04, -.03, .02, .01, .3, -.2, .02, .01])
    return p, x, np.sin(3*x)


def test_sign_symmetry():
    p, x, y = fixture()
    other = p.copy()
    other[[1, 4, 7]] *= -1
    first = audit.structural(p, x, y, .01)
    second = audit.structural(other, x, y, .01)
    for key in first:
        if isinstance(first[key], (int, float)):
            np.testing.assert_allclose(first[key], second[key], equal_nan=True)
    g1, g2 = audit.geometry(p, x, y), audit.geometry(other, x, y)
    transformed = g1['F'].copy(); transformed[[1, 4, 7]] *= -1
    np.testing.assert_allclose(transformed, g2['F'], atol=1e-14)


def test_projection():
    a, c = audit.project_cone(np.array([1., 2., 1., 0.]), np.array([3., 0., -3., 2.]))
    np.testing.assert_allclose(a, [1., 1., 0., 0.])
    np.testing.assert_allclose(c, [3., 1., 0., 2.])
    pa, pc = audit.project_cone(np.array([-.1]), np.array([10.]))
    np.testing.assert_allclose(pa, [0.]); np.testing.assert_allclose(pc, [10.])


def test_compensation_reconstruction():
    p, x, y = fixture()
    state = audit.geometry(p, x, y)
    rows = audit.attribution(p, state, audit.classes(p), p, .01)
    assert max(r['compensation_reconstruction_error'] for r in rows) < 1e-14
    np.testing.assert_allclose(state['J']@state['F'], 0, atol=1e-14)
    w = (len(p)-1)//3
    velocity = -.01*np.sign(p[:w])*(state['F']+state['R'])[:w]
    np.testing.assert_allclose(sum(r['total_signed_sum_over_width'] for r in rows), velocity.mean(), atol=1e-14)
    moments = audit.structural(p, x, y, .01)
    np.testing.assert_allclose(sum(r['moment_p'] for r in rows), moments['coarse_p'], atol=1e-14)
    np.testing.assert_allclose(sum(r['moment_M3'] for r in rows), moments['M3'], atol=1e-14)


def test_aligned_fixture_and_loading():
    x = np.linspace(-1., 1., 41)
    p = np.r_[.1, .2, 0., 0., .3, .4, 0.]
    plus = audit.structural(p, x, x**3, .01)
    minus = audit.structural(p, x, -x**3, .01)
    assert plus['cone_exact'] and plus['beta_max'] == 0
    assert plus['cone_distance_rms'] < 1e-15
    assert plus['cubic_sign_eligible'] and not minus['cubic_sign_eligible']
    even = audit.structural(p, x, x**2, .01)
    assert even['target_odd_error'] > 0


def test_constant_target_gauge():
    p, x, y = fixture()
    p2 = p.copy(); p2[-1] += .3
    first = audit.structural(p, x, y, .01)
    second = audit.structural(p2, x, y+.3, .01)
    for key in ('target_odd_error', 'centered_output_bias', 'cone_distance_rms', 'm3_times_p'):
        np.testing.assert_allclose(first[key], second[key], atol=1e-14)
    np.testing.assert_allclose(audit.geometry(p, x, y)['F'], audit.geometry(p2, x, y+.3)['F'], atol=1e-14)


def test_global_output_sign_with_zero_slope():
    p, x, y = fixture()
    p[2] = 0.
    other = p.copy(); other[6:] *= -1
    first = audit.structural(p, x, y, .01)
    second = audit.structural(other, x, -y, .01)
    excluded = {'output_bias', 'target_mean', 'centered_output_bias', 'output_sign'}
    for key in first:
        if key not in excluded and isinstance(first[key], (int, float)):
            np.testing.assert_allclose(first[key], second[key], equal_nan=True, atol=1e-14)
    for key in audit.classes(p):
        np.testing.assert_array_equal(audit.classes(p)[key], audit.classes(other)[key])


def test_zero_slope_radial_right_derivative():
    p, x, y = fixture()
    p[2] = 0.
    state = audit.geometry(p, x, y)
    rows = audit.attribution(p, state, audit.classes(p), p, .01)
    w = (len(p)-1)//3
    velocity = -(state['F']+state['R'])[:w]
    radial = np.where(p[:w] == 0, abs(velocity), np.sign(p[:w])*velocity)
    np.testing.assert_allclose(sum(r['total_signed_sum_over_width'] for r in rows), .01*radial.mean(), atol=1e-14)
    assert audit.structural(p, x, y, .01)['zero_slope_count'] == 1


def test_full_gradient_reconstruction():
    p, x, y = fixture()
    state = audit.geometry(p, x, y)
    w = (len(p)-1)//3
    a, b, c = p[:-1].reshape(3, w)
    f = np.tanh(x[:, None]*a+b)
    residual = f@c+p[-1]-y
    weighted = residual[:, None]*(1-f*f)
    gradient = np.r_[c*(x@weighted)/len(x), c*weighted.mean(0), f.T@residual/len(x), residual.mean()]
    np.testing.assert_allclose(state['F']+state['R'], gradient, atol=1e-14)


def test_fifth_loading_changes_generated_and_initial_margins():
    x = np.linspace(-1., 1., 41)
    q, _ = np.linalg.qr(x[:, None]**np.arange(5))
    phi5 = x**5-q@(q.T@(x**5))
    affine = audit.basis(x)
    phi3 = x**3-affine@(affine.T@(x**3)/len(x))
    t3 = np.mean(phi3**2)
    # W=1, alpha=1, zeta=2: both dominance margins change sign at mu=t3/2.
    p = np.array([1., 0., 2., 0.])
    for loading, expected_sign in ((t3/4, 1), (t3, -1), (-t3, 1)):
        y = loading*phi5/np.mean(phi5*x**5)
        row = audit.structural(p, x, y, .01)
        np.testing.assert_allclose(row['fifth_mu'], loading, rtol=1e-12)
        assert row['cubic_cancellation_relative'] < 1e-11
        np.testing.assert_allclose(row['generated_margin'], 2*t3-4*max(loading, 0.), rtol=1e-12)
        np.testing.assert_allclose(row['initial_moment_margin'], 8*t3-16*max(loading, 0.), rtol=1e-12)
        assert expected_sign*row['generated_margin'] > 0
        assert expected_sign*row['initial_moment_margin'] > 0


def test_union_cone_distance_can_reverse_original_slope_orientation():
    x = np.linspace(-1., 1., 41)
    # The second particle keeps the global coarse sign positive. The first
    # projects from (.1, -10) to (0, -10), not to the positive-cone origin.
    p = np.r_[np.array([.1, 1., 0., 0., -10., 2.])/np.sqrt(2), 0.]
    row = audit.structural(p, x, np.sin(3*x), .01)
    assert row['output_sign'] == 1
    np.testing.assert_allclose(row['cone_distance_rms'], .1/np.sqrt(2), atol=1e-14)
    np.testing.assert_allclose(row['cone_distance_max'], .1, atol=1e-14)
