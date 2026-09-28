import importlib.util
from pathlib import Path

import mpmath as mp
import numpy as np

spec=importlib.util.spec_from_file_location('c10_comparison',Path(__file__).with_name('run.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_frequency_estimator_includes_dc_and_omits_repeated_endpoint():
    x=np.linspace(-1,1,2049)
    # Total Fourier amplitude: 3 at DC, 1 in each of the +-2pi modes.
    y=3+2*np.cos(2*np.pi*x)
    assert abs(r.fft_frequency(y)-(4*np.pi/5))<1e-9


def test_selector_solves_the_actual_note_equation():
    cfg=r.config()
    selected=r.select_lambda(256,2*np.pi,cfg)
    lam=selected['lambda']
    value=2*np.exp(-np.pi**2/lam)*np.sinh(2*np.pi*(2*np.pi)/(256*lam))
    np.testing.assert_allclose(value,cfg['effective_aliasing_budget'],rtol=1e-12)
    assert r.select_lambda(64,400,cfg)['lambda'] is None


def test_log_envelope_matches_high_precision_note_products():
    n,halo,lam,delta=128,24,.3,.25
    actual=r.envelopes(n,halo,lam,delta)
    with mp.workdps(70):
        h=mp.mpf(2)/n;l=mp.mpf(lam);de=mp.mpf(delta)
        zeta=mp.exp(-2*l);m=(halo+1)//2
        pp=[mp.mpf(1)]
        for k in range(1,m+1):pp.append(pp[-1]*(1-zeta**k))
        a=[h/(2*(de-mp.pi*h/(2*l)))]*(n+1+2*halo)
        for i in range(1,m+1):
            li=zeta**(mp.mpf(i*(i+1)-1)/2)/(pp[i-1]*pp[m-i])
            li*=mp.fprod(1+zeta**(mp.mpf(j)-mp.mpf('.5')) for j in range(1,m+1) if j!=i)
            correction=h*(mp.pi/(2*l)+4*mp.log(2)/mp.pi)*li/(2*de)
            a[i-1]+=correction;a[-i]+=correction
        expected=np.array([float(1+sum(a))]+list(map(float,a)))
    np.testing.assert_allclose(actual,expected,rtol=5e-14)
    assert r.envelopes(64,24,.1,.25) is None


def test_coordinate_change_preserves_untruncated_fit_on_well_conditioned_dictionary():
    rng=np.random.default_rng(22)
    A=rng.normal(size=(40,7));truth=rng.normal(size=(7,3));Y=A@truth
    scales=np.geomspace(.1,10,7)
    c,a,rank,_=r.scaled_solve(A,Y,scales,1e-14)
    np.testing.assert_allclose(c,truth,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(A@c,(A*scales)@a,atol=1e-13)
    assert rank==7
