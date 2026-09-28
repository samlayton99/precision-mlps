"""Flow integrator order and exact effective-gradient identities."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.expD34_readout_race.population_feedback_flow import diagnostics,rk4


def test_rk4_on_linear_decay_has_fourth_order_global_error():
    errors=[]
    for dt in (.1,.05,.025):
        q=1.
        for _ in range(round(1/dt)): q=rk4(q,dt,lambda p:p)
        errors.append(abs(q-math.exp(-1)))
    assert errors[0]/errors[1] == pytest.approx(16,rel=.06)
    assert errors[1]/errors[2] == pytest.approx(16,rel=.03)


def test_effective_force_identity_and_coarse_tangency():
    from experiments.expD34_readout_race import mechanism_persistence_kernel as kernel
    p=jnp.array([.15,-.22,.12,.04,.3,-.12,.07])
    x=jnp.linspace(-1,1,31); y=jnp.sin(3*x)+.1*x*x
    state=kernel.decomposition(p,x,y)
    assert np.linalg.norm(np.asarray(state['JC']@state['F'])) < 1e-14
    result=diagnostics(p,x,y)
    assert float(result['identity_relative']) < 1e-12
    derivative=jax.jvp(lambda q:jnp.mean(kernel.decomposition(q,x,y)['eH']**2),(p,),(-state['F'],))[1]
    assert float(derivative) == pytest.approx(float(-2*state['F']@state['F']),rel=1e-12)
