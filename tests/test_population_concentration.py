"""Independent algebra, global inequalities, and discrete comparison checks."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race import population_concentration as pc
from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel


def test_shape_measures_distribution_and_is_scale_invariant():
    for width, active in ((8, 8), (8, 2), (8, 1)):
        p = np.r_[np.r_[np.ones(active), np.zeros(width-active)], np.zeros(2*width+1)]
        expected = (width/active)**2
        assert pc.population_shape(p)['C6'] == pytest.approx(expected)
        for scale in (.01, 3., 100.):
            assert pc.population_shape(scale*p)['C6'] == pytest.approx(expected)


def test_projected_cubic_jacobian_has_the_claimed_moment_identity():
    x = jnp.linspace(-1, 1, 41)
    p = jnp.array([.7, -.2, .1, .3, -.8, .2, 0.])
    basis = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))), axis=1)
    fine = lambda v:v-basis@(basis.T@v/len(x))
    def cubic(z):
        a,b,c = z[:-1].reshape(3, -1)
        return fine(-((x[:,None]*a+b)**3)@c/3)
    jacobian = np.asarray(jax.jacfwd(cubic)(p))
    a,b,c = np.asarray(p[:-1]).reshape(3, -1)
    target = pc.target_structure(x, np.sin(np.asarray(x)))
    expected = (target['s2']*np.sum(4*c*c*a*a*b*b+c*c*a**4+a**4*b*b)
                +target['s3']*np.sum(c*c*a**4+a**6/9))
    assert np.mean(np.sum(jacobian**2, axis=1)) == pytest.approx(expected, rel=2e-13)
    s6 = np.sum((a*a+b*b+c*c)**3)
    assert expected <= target['D3']**2*s6


@pytest.mark.parametrize('scale', [.03, .4, 2., 8.])
def test_global_bounds_against_independent_exact_jacobian(scale):
    rng = np.random.default_rng(29)
    x = jnp.linspace(-1, 1, 51)
    p = jnp.asarray(np.r_[scale*rng.normal(size=15), .13])
    y = jnp.sin(3*x)+.2*x*x
    state = kernel.decomposition(p,x,y)
    jacobian = jax.jacfwd(lambda z:kernel.output(z,x))(p)
    basis = state['basis']
    jh = jacobian-basis@(basis.T@jacobian/len(x))
    e0 = float(jnp.sqrt(jnp.mean(state['eH']**2)))
    row = dict(width=5, **pc.population_shape(p))
    target = pc.target_structure(x,y)
    upper = pc.bounds(np.sqrt(row['M']),row,target,e0)
    exact_j = float(jnp.sqrt(jnp.mean(jnp.sum(jh*jh,axis=1))))
    exact_q = float(jnp.linalg.norm(state['F']))
    exact_f = float(jnp.sqrt(jnp.mean(state['fH']**2)))
    for key in ('jacobian_generic','jacobian_projected'):
        assert exact_j <= upper[key]*(1+1e-12)
    for key in ('generic','projected','target'):
        assert exact_q <= upper[key]*(1+1e-12)
    assert exact_f <= upper['capacity']*(1+1e-12)


def test_target_gap_is_a_property_of_the_target():
    x = np.linspace(-1,1,63)
    q,_ = np.linalg.qr(np.polynomial.legendre.legvander(x,5))
    y = q[:,5]*np.sqrt(len(x))
    target = pc.target_structure(x,y)
    assert target['tau3'] < 5e-15
    assert target['tau5'] == pytest.approx(1.)


def test_concentration_clock_matches_exact_cubic_growth():
    times = np.linspace(0,.3,10)
    radius,e0,w,shape = 1.2,.8,9.,2.
    a = pc.C3*e0*np.sqrt(shape)/w
    exact = radius/np.sqrt(1-2*a*radius**2*times)
    np.testing.assert_allclose(pc.concentration_envelope(radius,np.sqrt(shape)*times,e0,w),exact)
    assert np.isnan(pc.concentration_envelope(radius,1/(2*pc.C3*e0*radius**2/w),e0,w))


def test_native_gd_radius_and_tracking_lump_are_upper_bounds():
    rng = np.random.default_rng(7)
    n,eta,r0,e0,w = 90,.01,.8,.9,11
    shapes = rng.uniform(1,4,n)
    tracking = rng.uniform(0,.03,n)
    coeff = pc.C3*e0*np.sqrt(shapes)/w
    # A rotating two-dimensional trajectory saturates neither triangle bound.
    vector = np.array([r0,0.]); observed = [r0]
    for a,rr in zip(coeff,tracking,strict=True):
        v = np.array([vector[1],-vector[0]])
        vector += eta*(a*np.linalg.norm(vector)**2*v+np.array([rr,0.]))
        observed.append(np.linalg.norm(vector))
    recurrence = pc.gd_radius_envelope(r0,[lambda r,a=a:a*r**3 for a in coeff],tracking,eta)
    clock = np.r_[0,np.cumsum(eta*np.sqrt(shapes))]
    path = np.r_[0,np.cumsum(eta*tracking)]
    closed = pc.concentration_envelope(r0,clock,e0,w,path)
    assert np.all(np.asarray(observed) <= recurrence+1e-14)
    assert np.all(recurrence <= closed+1e-14)


def test_comparison_coefficients_do_not_use_saved_force_or_reinforcement():
    x = np.linspace(-1,1,51)
    p = np.r_[np.linspace(-.2,.3,12),0.]
    row = dict(width=4,**pc.population_shape(p))
    target = pc.target_structure(x,np.sin(x))
    before = pc.bounds(1.,row,target,1.)
    row.update(q=1e100,kappa=1e100,rotation=1e100,coefficient=1e100)
    after = pc.bounds(1.,row,target,1.)
    for key in before:
        if key != 'reference': assert before[key] == after[key]
