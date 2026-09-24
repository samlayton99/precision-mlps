"""Analytical fixtures for the accumulated-feedback theorem implementation."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from experiments.expD34_readout_race.population_feedback_budget import (
    bounds, constant_budget_integral, exp_budget_integral,
)


@pytest.mark.parametrize('rate',[0.,1e-9,.01,1.])
def test_constant_feedback_recovers_exact_exponential_energy(rate):
    t=np.array([0.,.01,.2,2.])
    B,H=exp_budget_integral(t,np.full_like(t,rate))
    assert B==pytest.approx(t*rate)
    assert H==pytest.approx(constant_budget_integral(t,rate),rel=2e-13,abs=1e-14)


def test_varying_feedback_matches_independent_quadrature():
    t=np.array([0.,.3,2.])
    rate=np.array([.1,.8,.2])
    B,H=exp_budget_integral(t,rate)
    first=lambda s:.1*s+.5*(.7/.3)*s*s
    second=lambda s:B[1]+.8*(s-.3)-.5*(.6/1.7)*(s-.3)**2
    reference=quad(lambda s:math.exp(2*first(s)),0.,.3,epsabs=1e-12)[0]
    reference+=quad(lambda s:math.exp(2*second(s)),.3,2.,epsabs=1e-12)[0]
    assert H[-1]==pytest.approx(reference,rel=1e-12)


def test_constant_comparison_saturates_force_transport_and_energy():
    first=dict(F_norm=.01,fine_norm=.8,M4=40.,width=705,target_norm=1.,target_fine_norm=.9)
    t=np.array([0.,.5,2.])
    B,H=exp_budget_integral(t,np.full_like(t,.2))
    result=bounds(first,B,H,3*t)
    exact_energy=.01**2*np.expm1(.4*t)/.4
    exact_travel=.01*3**.5*np.expm1(.2*t)/.2
    assert result['dissipation_envelope']==pytest.approx(exact_energy)
    assert np.all(result['weighted_travel_envelope']>=exact_travel-1e-14)
    assert result['energy_relative_floor']**2==pytest.approx(.8**2-2*exact_energy)
    # A factor ten smaller initial force gives a factor 100 less dissipation.
    smaller=bounds(dict(first,F_norm=.001),B,H,3*t)
    assert smaller['dissipation_envelope']*100==pytest.approx(result['dissipation_envelope'])


def test_large_feedback_is_vacuous_not_a_false_positive():
    first=dict(F_norm=.01,fine_norm=.8,M4=40.,width=705,target_norm=1.,target_fine_norm=.9)
    B,H=exp_budget_integral([0.,1.],[1e6,1e6])
    result=bounds(first,B,H,[0.,3.])
    assert math.isinf(H[-1])
    assert result['combined_relative_floor'][-1]==0


def test_terminal_allowance_dominates_prefix_energy():
    t=np.array([0.,.3,2.])
    B,H=exp_budget_integral(t,[1.,.1,.5])
    assert np.all(H<=t*np.exp(2*B[-1])+1e-14)
    with pytest.raises(ValueError,match='nonnegative'):
        exp_budget_integral(t,[1.,-.1,.5])
