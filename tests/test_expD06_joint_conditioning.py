import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from experiments.expD06_fixed_center_scales import core, difference_training as dt, run


def test_parameter_difference_coordinates_and_jacobian():
    g = core.geometry(128)
    c, gamma = core.initial_physical(g, 0, "xavier_a_reference")
    z = dt.encode(c, g, "parameter_differences")
    np.testing.assert_allclose(dt.decode(z, g, "parameter_differences"), c, atol=3e-17)
    x = np.linspace(-1, 1, 129)
    phi = np.tanh((x[:, None]-g.centers)*gamma)
    q = z[1:]*np.cumsum(g.alpha[1:])
    alternate = g.alpha[0]*z[0]+(phi[:, :-1]-phi[:, 1:])@q[:-1]+phi[:, -1]*q[-1]
    np.testing.assert_allclose(alternate, c[0]+phi@c[1:], atol=2e-16)
    gc = jnp.asarray(np.random.default_rng(7).normal(size=c.size))
    auto = jax.grad(lambda a: dt.decode(a,g,"parameter_differences")@gc)(jnp.asarray(z))
    np.testing.assert_allclose(auto, dt.pullback(gc,g,"parameter_differences"), atol=2e-14)
    lam = jnp.asarray(g.h*gamma)
    y = core.target(jnp.asarray(x), "sine")
    loss = lambda a: dt.physical_loss(dt.decode(a,g,"parameter_differences"),lam,jnp.asarray(x),y,g)
    direction = jnp.asarray(np.random.default_rng(2).normal(size=z.size))
    direction /= jnp.linalg.norm(direction)
    eps = 1e-6
    np.testing.assert_allclose((loss(z+eps*direction)-loss(z-eps*direction))/(2*eps),
                               jax.grad(loss)(z)@direction, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("optimizer", ["gd", "adam"])
@pytest.mark.parametrize("coordinate", ["parameter_scale", "parameter_differences"])
def test_joint_first_order_updates_and_resume(optimizer,coordinate,tmp_path):
    g = core.geometry(128); eta = 1e-5; epsilon = 1e-12
    state = dt.initial(g,0,coordinate,optimizer)
    c = np.asarray(dt.decode(state["z"],g,coordinate)); lam = np.asarray(state["lam"])
    basis = np.eye(g.width+1)
    transform = np.stack([dt.decode(jnp.asarray(v),g,coordinate) for v in basis],axis=1)
    x=jnp.linspace(-1,1,129); y=core.target(x,"sine")
    gc,gl=jax.grad(dt.physical_loss,argnums=(0,1))(jnp.asarray(c),jnp.asarray(lam),x,y,g)
    gz=transform.T@np.asarray(gc)
    dz=-eta*gz if optimizer=="gd" else -eta*gz/(np.abs(gz)+epsilon)
    dl=-eta*np.asarray(gl) if optimizer=="gd" else -eta*np.asarray(gl)/(np.abs(gl)+epsilon)
    fn=dt.chunk(128,coordinate,1,True,1,False,optimizer,epsilon)
    next_state,(_,dense)=fn(state,eta,0)
    np.testing.assert_allclose(dense["delta_c"][0],transform@dz,atol=3e-16,rtol=1e-10)
    np.testing.assert_allclose(dense["delta_lambda"][0],dl,atol=3e-16,rtol=1e-10)
    whole,_=dt.chunk(128,coordinate,4,False,1,False,optimizer,epsilon)(state,eta,0)
    run.save_state(tmp_path/'state.pkl',next_state,1)
    resumed,start=run.load_state(tmp_path/'state.pkl')
    resumed,_=dt.chunk(128,coordinate,3,False,1,False,optimizer,epsilon)(resumed,eta,start)
    for a,b in zip(jax.tree.leaves(whole),jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a,b)


def test_explicit_epsilon_required_for_difference_adam():
    with pytest.raises(ValueError,match="explicit native epsilon"):
        dt.chunk(128,"parameter_differences",1,optimizer="adam")
