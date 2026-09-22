import numpy as np
import pytest

from experiments.expD34_readout_race import mechanism as mech, transport as tr, targets


@pytest.mark.parametrize('arm',list(mech.ARMS))
def test_force_drivers_follow_the_actual_intervention(arm):
    rng=np.random.default_rng(81)
    z=rng.normal(size=(3,7))*.6; d=.2
    x=targets.grid(128); y=np.sin(2*np.pi*x)
    rates=mech.ARMS[arm]
    row,arr=mech.force_metrics(z,d,x,y,rates,degree=17)
    grad=arr['gradient']; velocity=-np.array(rates[:3])[:,None]*grad
    vd=-rates[3]*arr['residual'].mean()
    errors=[]
    for dt in (1e-3,5e-4,2.5e-4):
        _,plus=mech.force_metrics(z+dt*velocity,d+dt*vd,x,y,rates,degree=17)
        _,minus=mech.force_metrics(z-dt*velocity,d-dt*vd,x,y,rates,degree=17)
        numerical=(plus['effective_force']-minus['effective_force'])/(2*dt)
        errors.append(np.linalg.norm(numerical-arr['effective_force_dot']))
        np.testing.assert_allclose((plus['gradient'][0]-minus['gradient'][0])/(2*dt),
                                   arr['full_force_dot'],atol=2e-6,rtol=2e-5)
    assert errors[-1] < errors[0]/10
    assert row['effective_derivative_identity_error'] < 1e-13
    assert row['full_outward']==pytest.approx(sum(row[k+'_outward'] for k in ('effective','tracking','omitted')),abs=1e-14)
    assert sum(row['effective_'+k+'_share'] for k in 'abcd')==pytest.approx(1.,abs=1e-11)
    assert row['actual_slope_speed']==pytest.approx(rates[0]*np.linalg.norm(grad[0]))


def test_existing_equal_rate_diagnostics_are_preserved():
    z,_=targets.initial(128,24,0); x=targets.grid(128); y=np.sin(2*np.pi*x)
    old,arr=tr.modal_diagnostics(z,0.,x,y,np.ones(177)/177,177,17)
    new,now=mech.force_metrics(z,0.,x,y,degree=17)
    for key,value in old.items():
        assert new[key]==pytest.approx(value,abs=1e-15)
    np.testing.assert_array_equal(arr['K_dot'],now['K_dot'])


def test_frozen_curve_matches_explicit_raw_readout_gd():
    rng=np.random.default_rng(91); a=rng.normal(size=9); b=rng.normal(size=9)
    x=targets.grid(128); xe=targets.grid(257)
    y=np.sin(2*np.pi*x); ye=np.sin(2*np.pi*xe)
    steps=(0,1,10,1000)
    rows,capacity,coeff=mech.frozen_curves(a,b,x,y,xe,ye,steps)
    A=mech.design(a,b,x); c=np.zeros(10); found=[c.copy()]
    for n in range(1,1001):
        c-=.002*A.T @ (A @ c-y)/len(x)
        if n in steps: found.append(c.copy())
    np.testing.assert_allclose(coeff,found,atol=2e-14,rtol=2e-12)
    assert rows[0]['relative_train_mse']==pytest.approx(1.)
    assert rows[-1]['train_mse'] < rows[0]['train_mse']
    assert all(row['readout_l2']>=0 for row in capacity)


def test_frozen_diagnostic_retains_tiny_finite_time_coupling():
    x=targets.grid(128); a=np.array([1e-6]); b=np.array([0.])
    rows,_,coeff=mech.frozen_curves(a,b,x,x,x,x,(1,100000))
    # 1-(1-eta*s²)^n would round to zero for the weakest direction here.
    expected=.002*100000*np.mean(x*x)*a[0]
    assert coeff[-1,0]==pytest.approx(expected,rel=1e-8)
    assert rows[-1]['train_mse']<=rows[0]['train_mse']


def test_continuations_apply_old_state_gradients_and_exact_freezes():
    import jax.numpy as jnp
    from experiments.expD34_readout_race.mechanism_run import initial,advance_factory
    rng=np.random.default_rng(8); z=rng.normal(size=(6,3,5))*.3; d=rng.normal(size=6)*.1
    x=targets.grid(64); y=np.broadcast_to(np.sin(2*np.pi*x),(6,64)).copy()
    rates=np.array(list(mech.ARMS.values()))
    state=initial(z,d)
    result=advance_factory(64,.002)(state,jnp.asarray(y),jnp.asarray(rates),1)
    for i,rate in enumerate(rates):
        _,arr=mech.force_metrics(z[i],d[i],x,y[i],rate,degree=17)
        expected=z[i]-.002*rate[:3,None]*arr['gradient']
        np.testing.assert_allclose(result['z'][i],expected,rtol=1e-13,atol=1e-15)
        assert float(result['d'][i])==pytest.approx(d[i]-.002*rate[3]*arr['residual'].mean(),abs=1e-15)
        np.testing.assert_allclose(result['positive'][i]-result['negative'][i],abs(expected[0])-abs(z[i,0]),atol=1e-15)
        assert float(result['path'][i])==pytest.approx(.002*rate[0]*np.linalg.norm(arr['gradient'][0]))
        for block in range(3):
            if rate[block]==0: np.testing.assert_array_equal(result['z'][i,block],z[i,block])
