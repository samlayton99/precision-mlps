"""Comparison checks with independently solved scalar dynamics."""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from experiments.expD34_readout_race.population_energy_analysis import cap_clock
from experiments.expD34_readout_race import population_concentration as pc


def test_accumulated_cap_handles_delayed_and_early_coefficients():
    # Positive polynomial vector fields exercise time ordering without replacing
    # time-dependent coefficients by their averages in the ODE.
    for reverse in (False,True):
        time=np.linspace(0,2,2001);w=30.;b0=.7;cap=1.4;e0=.8
        target=dict(D3=.18,tau3=.2)
        coefficients=lambda t:np.array([1+.2*np.sin(t)**2,2+(2-t if reverse else t)**3,3+.2*t])
        def rhs(t,y):
            a,b,c=coefficients(t)
            return [min(pc.C3*e0*a/w*y[0]**3,
                        target['D3']*e0*a/w*y[0]**3+pc.C5*e0*b/w**2*y[0]**5,
                        target['D3']*target['tau3']*a/w*y[0]**3+pc.C5*e0*b/w**2*y[0]**5
                        +target['D3']*pc.A3*c/w**2*y[0]**7)]
        actual=solve_ivp(rhs,(0,2),[b0],t_eval=time,rtol=2e-12,atol=1e-14).y[0]
        values=np.array([coefficients(t) for t in time])
        clocks=cumulative_trapezoid(values,time,axis=0,initial=0)
        budget=cap_clock(cap,clocks,target,e0,w)
        upper=b0/np.sqrt(1-2*b0*b0*budget)
        assert np.all(upper<cap)
        assert np.all(actual<=upper+1e-12)


def test_coupled_generic_bound_dominates_saturating_mass_shape_system():
    # b'=a z b^3 and z'=3 s a z^2 b^2 imply (b^2 z)'=(2+3s)a(b^2 z)^2.
    a,s=.03,.8;b0,z0=.7,1.4
    t=np.linspace(0,1,101)
    solution=solve_ivp(lambda _,v:[a*v[1]*v[0]**3,3*s*a*v[1]**2*v[0]**2],
                       (0,1),[b0,z0],t_eval=t,rtol=1e-12,atol=1e-14)
    product=solution.y[0]**2*solution.y[1]
    initial=b0*b0*z0
    exact=initial/(1-(2+3*s)*a*initial*t)
    np.testing.assert_allclose(product,exact,rtol=1e-10,atol=1e-12)
