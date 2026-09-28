"""Comparison accuracy and separation of calibration from future evaluation."""
import numpy as np

from experiments.expD34_readout_race.population_window_analysis import calibration,comparison,evaluate


def test_comparison_on_exact_constant_log_growth():
    t=np.linspace(0,30,101); q0=.003; k=.02
    a,q,e=comparison(t,q0,k,0.,0.)
    np.testing.assert_allclose(q,q0*np.exp(k*t),rtol=2e-9,atol=1e-13)
    np.testing.assert_allclose(a,q0*np.expm1(k*t)/k,rtol=2e-9,atol=1e-13)
    np.testing.assert_allclose(e,q0*q0*np.expm1(2*k*t)/(2*k),rtol=2e-9,atol=1e-13)


def test_calibration_cannot_see_future_records():
    rows=[dict(time=.002*i,offset=i,rotation=1e-5+i*1e-9,
               coefficient=.01+i*1e-7,q=.001,kappa=-.01) for i in range(0,11000,1000)]
    expected=calibration(rows)
    for row in rows:
        if row['offset']>5000:
            for k in ('rotation','coefficient','q','kappa'):row[k]=1e20
    assert calibration(rows)==expected


def test_temporal_variation_can_allow_delayed_sign_change_without_zero_floor():
    rows=[dict(time=i,offset=1000*i,rotation=0.,coefficient=-.02+.004*i,
               q=.001,kappa=-.01) for i in range(6)]
    assert calibration(rows,method='prefix')['K']==0
    assert calibration(rows,method='variation')['K']>0
    for row in rows: row['coefficient']=0.
    assert calibration(rows,method='variation')['K']==0


def test_negative_initial_rate_and_zero_reinforcement_allow_decay():
    t=np.linspace(0,100,101)
    a,q,e=comparison(t,.002,-.01,0.,0.)
    assert np.all(np.diff(q)<0)
    assert np.all(np.diff(a)>0)
    np.testing.assert_allclose(q,.002*np.exp(-.01*t),rtol=2e-9)


def test_motion_calibration_preserves_a_measured_coefficient_drift():
    rows=[dict(time=i,offset=1000*i,rotation=1e-5+2e-6*i,
               coefficient=.01+.003*i,q=.001,kappa=-.01) for i in range(6)]
    spec=calibration(rows,method='motion',factor=2.)
    np.testing.assert_allclose([spec['Lr'],spec['Lc']],[.004,6.])
    expected=spec.copy()
    rows.append(dict(time=6,offset=6000,rotation=1e10,coefficient=1e10,q=1e10,kappa=1e10))
    assert calibration(rows,method='motion',factor=2.)==expected


def test_motion_allowance_retains_decreasing_channel_variation():
    rows=[dict(time=i,offset=1000*i,rotation=1e-5-1e-6*i,
               coefficient=.03-.003*i,q=.001,kappa=-.01) for i in range(6)]
    spec=calibration(rows,method='motion',factor=2.)
    np.testing.assert_allclose([spec['Lr'],spec['Lc']],[.002,6.])


def test_positive_feedback_matches_exact_riccati_solution():
    t=np.linspace(0,20,101); q0=.003; K=.1
    a,q,_=comparison(t,q0,0.,0.,K)
    omega=np.sqrt(q0*K/2)
    np.testing.assert_allclose(a,np.sqrt(2*q0/K)*np.tan(omega*t),rtol=3e-9,atol=1e-13)
    np.testing.assert_allclose(q,q0/np.cos(omega*t)**2,rtol=3e-9,atol=1e-13)


def test_quadratic_movement_feedback_matches_separable_travel_integral():
    from scipy.integrate import quad
    t=np.linspace(0,20,101); q0=.003; Lc=2.
    a,q,_=comparison(t,q0,0.,0.,0.,Lc=Lc)
    np.testing.assert_allclose(q,q0+Lc*a**3/6,rtol=3e-9)
    implicit=quad(lambda s:1/(q0+Lc*s**3/6),0,a[-1],epsabs=1e-11)[0]
    np.testing.assert_allclose(implicit,t[-1],rtol=3e-9)


def test_aggregate_condition_does_not_require_separate_channel_bounds():
    rows=[dict(study='synthetic',target='cancellation',seed=0,width=705,nref=512,
               kind='effective',dt=.002,time=.002*i,offset=i,q=1.,kappa=0.,rotation=1.,
               coefficient=-1.,state_change=-1.,Y2=1000.,target_norm=1.,lambda_rms=.001,
               h=2/512,relative_eval_error=30.) for i in range(0,110001,1000)]
    spec=dict(offset=5000,method='synthetic',factor=0.,q0=1.,kappa0=0.,alpha=0.,K=0.,Lr=0.,Lc=0.)
    result,_=evaluate(rows,spec)
    assert result['premise_horizon']==100000
    assert result['split_premise_horizon']==0
