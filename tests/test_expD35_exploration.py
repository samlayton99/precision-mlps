import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.expD35_optimization_exploration import core
from experiments.expD06_fixed_center_scales import core as old, difference_training as previous


def case(**extra):
    return dict(n=64, seed=0, coordinates='individual', optimizer='gd', eta=1e-4, **extra)


@pytest.mark.parametrize('coordinates', core.COORDINATES)
def test_physical_pairing_and_gradient(coordinates):
    c = case(); c['coordinates'] = coordinates
    g = old.geometry(c['n']); state = core.initialize(c)
    w, gamma = old.initial_physical(g, 0, 'xavier_a_reference')
    wr, gr = core.physical(state['z'], g, coordinates)
    np.testing.assert_allclose(wr, w, atol=2e-15, rtol=2e-14)
    np.testing.assert_array_equal(gr, gamma)
    x = jnp.linspace(-1, 1, 129); y = core.target(x, 'mixed')
    analytic = core.field(state['z'], x, y, g, coordinates)[1]
    def loss(z):
        w, ga = core.physical(z, g, coordinates)
        return .5*jnp.mean((w[0]+old.tanh((x[:,None]-g.centers)*ga)@w[1:]-y)**2)
    np.testing.assert_allclose(analytic, jax.grad(loss)(state['z']), rtol=3e-13, atol=2e-14)


@pytest.mark.parametrize('optimizer', ['gd','adam'])
def test_unmodified_matches_existing_training(optimizer):
    c = case(); c['optimizer'] = optimizer
    state = core.initialize(c); hp = core.hyperparameters(c)
    batch = lambda v: jax.tree.map(lambda a:a[None], v)
    actual, _ = core.chunk(64, 'individual', optimizer, 'sine', 20)(batch(state), batch(hp), 0)
    prior = previous.initial(old.geometry(64), 0, 'parameter_scale', optimizer)
    expected, _ = previous.chunk(64, 'parameter_scale', 20, batched=False,
                                optimizer=optimizer, native_epsilon=1e-15)(prior, 1e-4, 0)
    z = np.r_[expected['z'], expected['lam']]
    np.testing.assert_allclose(actual['z'][0], z, rtol=2e-12, atol=2e-14)


def test_ema_active_formula_and_gain_control():
    c=case(ema_strength=2., ema_alpha=.98)
    st=core.initialize(c);hp=core.hyperparameters(c);g=jnp.arange(len(st['z']))*.001
    direction, hist, filtered, _=core.optimizer_direction(st,g,hp,'gd')
    np.testing.assert_allclose(filtered,3*g)
    st.update(hist)
    _,_,f,_=core.optimizer_direction(st,-g,hp,'gd')
    np.testing.assert_allclose(f,-g+2*(.98*g+.02*(-g)))
    hp['ema_alpha']=jnp.array(0.)
    hp['ema_normalized']=jnp.array(True)
    d,_,_,_=core.optimizer_direction(st,g,hp,'gd')
    np.testing.assert_allclose(d,-g,atol=1e-16)


def test_tiny_gradient_cosine_and_zero():
    a=jnp.array([1e-250,2e-250,-1e-250])
    assert float(core.cosine(a,a))==pytest.approx(1.)
    assert float(core.cosine(a,-a))==pytest.approx(-1.)
    assert np.isnan(core.cosine(a,a*0))


def test_polynomial_target_does_not_change_with_grid():
    x=np.linspace(-1,1,137)
    for name in ('moment3','moment5','moment9'):
        expected=core.moments.values(name,x,core.polynomial_mapping())
        np.testing.assert_allclose(core.target(x,name),expected,atol=5e-13,rtol=2e-12)


def test_runner_resume_preserves_adam_and_ema(tmp_path):
    from experiments.expD35_optimization_exploration import run
    c=run.case(n=64, ema_strength=2., eta=1e-5)
    run.advance(tmp_path/'split',[c],10)
    run.advance(tmp_path/'split',[c],20)
    run.advance(tmp_path/'whole',[c],20)
    a,_=run.restore(tmp_path/'split'/run.key(c)/'state.npz')
    b,_=run.restore(tmp_path/'whole'/run.key(c)/'state.npz')
    for name in a:
        np.testing.assert_array_equal(a[name],b[name])
