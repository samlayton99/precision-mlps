"""Independent soluble case for the nonlinear scalar comparison and GD transfer."""
import numpy as np
from experiments.expD34_readout_race.population_balance_analysis import scalar_comparison, cumulative


def rows(kind,dt):
    common=dict(q3=0.,q5=0.,n3=0.,n5=0.,j3=0.,j5=0.,g3=.1,g5=0.,C8=0.,C14=0.,
                target_fine=1.,width=1.,M=1.,R_norm=0.,kind=kind,dt=dt)
    return [dict(common,time=float(t)) for t in np.linspace(0,1,11)]


def test_comparison_matches_solvable_positive_growth_ode():
    t,r,complete=scalar_comparison(rows('effective',.01))
    assert complete
    np.testing.assert_allclose(r,(1-.2*t)**-.5,rtol=2e-9)


def test_gd_uses_actual_step_recurrence_and_refines():
    _,coarse,complete=scalar_comparison(rows('gd',.01))
    assert complete
    _,fine,complete=scalar_comparison(rows('gd',.005))
    assert complete
    direct=1.
    for _ in range(100): direct+=.01*.1*direct**3
    np.testing.assert_allclose(coarse[-1],direct,atol=1e-14)
    exact=.8**-.5
    assert 0<exact-fine[-1]<.51*(exact-coarse[-1])
    np.testing.assert_allclose(cumulative(np.array([1.,3.,5.]),np.array([0.,1.,2.])),[0.,2.,6.])
