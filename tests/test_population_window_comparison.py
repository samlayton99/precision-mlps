"""Comparison accuracy and separation of calibration from future evaluation."""
import numpy as np

from experiments.expD34_readout_race.population_window_analysis import calibration,comparison


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
