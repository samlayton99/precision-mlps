"""Independent operator and interval checks for the initial-state certificate."""
import math

from flint import arb, ctx
import numpy as np
import pytest
from scipy.integrate import quad

from experiments.expD34_readout_race.population_initial_feedback import (
    blocks_float, certify, constants, initial_arb, initial_float,
)


def fixture():
    x=np.linspace(-1,1,31)
    p=np.array([.15,-.22,.12,.04,.3,-.12,.07])
    y=np.sin(3*x)+.1*x*x
    return p,x,y


def test_loaded_hessian_is_second_derivative_with_residual_held_fixed():
    p,x,_=fixture(); psi=np.sin(2*x)+x*x
    v=np.array([.2,.1,-.4,.3,.15,-.12,.01])
    blocks=blocks_float(p,x,psi)
    particles=v[:-1].reshape(3,-1).T
    expected=np.einsum('wi,wij,wj->',particles,blocks,particles)
    def loaded(q):
        a,b,c=q[:-1].reshape(3,-1)
        return np.mean(psi*(np.tanh(x[:,None]*a+b)@c+q[-1]))
    errors=[]
    for eps in (.04,.02,.01):
        second=(loaded(p+eps*v)-2*loaded(p)+loaded(p-eps*v))/eps**2
        errors.append(abs(second-expected))
    assert errors[0]/errors[1] == pytest.approx(4,rel=.01)
    assert errors[1]/errors[2] == pytest.approx(4,rel=.01)


def test_arb_initial_data_agrees_with_independent_vectorized_calculation():
    p,x,y=fixture(); ctx.prec=128
    exact,_=initial_arb(p,x,y)
    floating=initial_float(p,x,y)
    for key in floating:
        assert float(exact[key]) == pytest.approx(floating[key],rel=2e-11,abs=1e-14)
    ctx.prec=192
    higher,_=initial_arb(p,x,y)
    for key in exact:
        assert float(higher[key]) == pytest.approx(float(exact[key]),rel=1e-14,abs=1e-16)


def test_right_riemann_time_is_below_independent_integral_and_converges():
    state=dict(f0=arb(.0001),Y0=arb(.8),Y0_lower=arb(.8),target_norm=arb(1),
               sigma0=arb(2),P0=arb(.4),B0=arb(.15),H0=arb(.002))
    radius=.001
    const=constants(state,arb(radius),lambda v:arb(v).sqrt())
    expected=quad(lambda a:1/(.0001+.002*a+float(const['L'])*a*a/2),0,radius)[0]
    low=certify(state,radius,intervals=64)
    high=certify(state,radius,intervals=256)
    assert 0 < low['time_lower'] < high['time_lower'] < expected
    assert expected-high['time_lower'] < .3*(expected-low['time_lower'])
    assert 0 < high['energy_relative_floor_lower'] <= .8


def test_rank_loss_is_rejected():
    state=dict(P0=1.,B0=.1,sigma0=.01,Y0=.8)
    assert constants(state,.1,math.sqrt) is None
