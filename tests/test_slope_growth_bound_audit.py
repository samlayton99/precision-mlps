import math
import numpy as np
from experiments.expD56_scale_acquisition.audit_slope_growth import constants,force_terms,integrate


def test_normalized_recurrence_matches_equations_in_B():
    x=-1+(np.arange(1024)+.5)*2/1024;k=constants(x)
    w=196;B=13.;chi=np.array([1.5,2.3,6.,20.]);e=.7;t3=.1;t5=.2
    S={p:c*B**p/w**(p/2-1) for p,c in zip((4,6,10,14),chi)}
    Q=min(k['A3']*S[4],k['B3']*S[4]+k['A5']*S[6])
    expected=[k['C3']*e*math.sqrt(S[6]),
              e*(k['D3']*math.sqrt(S[6])+k['C5']*math.sqrt(S[10])),
              k['D3']*math.sqrt(S[6])*(t3+Q)+k['C5']*e*math.sqrt(S[10]),
              k['D3']*math.sqrt(S[6])*t3+k['C5']*math.sqrt(S[10])*t5+
              (k['D3']*math.sqrt(S[6])+k['C5']*math.sqrt(S[10]))*Q+k['C7']*e*math.sqrt(S[14])]
    np.testing.assert_allclose(np.array(force_terms(B/math.sqrt(w),w,chi,e,t3,t5,k))*math.sqrt(w),expected,rtol=1e-13)


def test_allowances_monotone_and_floor_is_optimistic():
    k=constants(np.linspace(-1,1,1024))
    for q in (.02,.2,1.,4.,18.):
        lower=np.array(force_terms(q,196,np.ones(4),.1,.2,.3,k))
        upper=np.array(force_terms(q*1.2,196,np.array([2,3,4,5]),.5,.2,.3,k))
        assert np.all(lower<=upper)


def test_recurrence_accumulates_each_step_without_resetting():
    h={"step":np.array([0.,5.,10.]),"parameter_energy":np.array([1.,1.,1.]),
       "gamma_rms":np.array([.05,.05,.05]),"tracking_norm":np.array([.2,.2,.2]),
       "nonaffine_residual_rms":np.zeros(3),"tau_3":np.zeros(3),"tau_5":np.zeros(3)}
    for p in (4,6,10,14):h[f"chi_{p}"]=np.ones(3)
    status={"width":100,"learning_rate":.1}
    k=constants(np.linspace(-1,1,100))
    r,_=integrate(h,status,0,"interpolated",k)
    # Zero residual => generic force zero. Tracking adds .1*.2/sqrt(100) each step.
    np.testing.assert_allclose(r["end_ceiling"],.1+10*.1*.2/10,rtol=1e-13)
    assert r['recorded_parameter_norm_violations']==0
