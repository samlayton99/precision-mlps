import numpy as np
import pytest

from experiments.expD34_readout_race.transport_analyze import weighted_distance, frozen_forecast, marginal_w2, diagnostics
from experiments.expD34_readout_race.recovery import acquisition_distance
from experiments.expD34_readout_race import transport as tr, targets


def test_weighted_population_distance_matches_integer_counts():
    a=np.array([.1,.2,1.2,3.]); w=np.ones(4)/4
    for population in (.25,.5,.75,1.):
        assert weighted_distance(a,w,4,2.,population)==pytest.approx(acquisition_distance(a,2.,population))
    assert weighted_distance(a,w,4,2.,.375)==pytest.approx(np.sqrt(.5*.8**2))


def test_weighted_quantile_coupling():
    assert marginal_w2(np.array([0.,2.]),np.array([.25,.75]),np.array([1.]),np.array([1.]))==pytest.approx(1.)
    a=np.array([3.,1.,2.]); b=np.array([1.1,3.2,2.3]); w=np.ones(3)/3
    assert marginal_w2(a,w,b,w)==pytest.approx(np.sqrt(np.mean((np.sort(a)-np.sort(b))**2)))


def test_frozen_forecast_matches_initial_gradient_and_linearized_loss():
    rng=np.random.default_rng(7); z=rng.normal(size=(3,5))*.3; d=.1
    x=targets.grid(128); y=np.sin(2*np.pi*x); w=np.ones(5)/5
    scalar,arrays=tr.modal_diagnostics(z,d,x,y,w,5,17,.1)
    e=arrays['residual_modes']; dt=1e-6
    zn,dn,loss=frozen_forecast(z,d,x,y,5,dt,17,.1)
    np.testing.assert_allclose((z-zn)/dt,np.stack([arrays['J_'+k].T @ e*r for k,r in [('a',1),('b',1),('c',.1)]]),rtol=2e-6,atol=1e-8)
    K=sum(arrays['K_'+k] for k in ('a','b','c','d'))
    assert (.5*(e @ e)-loss)/dt==pytest.approx(e @ K @ e,rel=2e-6)
    measured,exact=diagnostics(z,d,x,y,w,5,.1)
    assert measured['effective_kernel_accounting_error']<1e-14
    assert sum(measured['effective_'+k+'_share'] for k in 'abcd')==pytest.approx(1.,abs=1e-12)
    K=sum(exact['K_'+k] for k in 'abcd'); C,Q=K[:2,:2],K[:2,2:]
    B=np.linalg.solve(C,Q); S=K[2:,2:]-Q.T @ B
    e=exact['residual_modes']; edot=exact['residual_velocity']; Cdot=exact['K_dot'][:2,:2]
    Bdot=np.linalg.solve(C,exact['K_dot'][:2,2:]-Cdot @ B)
    tracking=e[:2]+B @ e[2:]
    tracking_dot=edot[:2]+Bdot @ e[2:]+B @ edot[2:]
    omitted=edot+K @ e
    forcing=(Bdot-B @ S) @ e[2:]+omitted[:2]+B @ omitted[2:]
    direct=tracking @ C @ tracking_dot+.5*tracking @ Cdot @ tracking
    identity=-tracking @ (C @ C+Q @ Q.T) @ tracking+.5*tracking @ Cdot @ tracking+tracking @ C @ forcing
    assert direct==pytest.approx(identity,abs=1e-14)
    norm2=tracking @ C @ tracking
    upper=-measured['tracking_metric_decay_lower']*norm2+np.sqrt(norm2*(forcing @ C @ forcing))
    assert direct<=upper+1e-14
