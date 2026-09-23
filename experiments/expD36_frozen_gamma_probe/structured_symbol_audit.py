"""Verify gamma-cap quadratic forms without an eigensolve or training.

Analytic lattice/alias tails accompany FP64 quadrature diagnostics; numerical
outputs are not outward-rounded certificates. This module emits evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (ROOT/'results/checkpoint_D_optimizers/expD36_frozen_gamma_probe'
                  '/full_sweep/refinements/structured_gamma/symbol_audit.json')


def center_normalize(values):
    """Return unit columns after projecting off the sample constant."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError('directions must be a finite vector or matrix')
    centered = values-values.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    if np.any(norms == 0):
        raise ValueError('constant directions have no mean-zero component')
    return centered/norms


def _directions(v, n, q, gamma):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if (n < 1 or q < 1 or gamma <= 0 or v.ndim != 2
            or len(v) != q*n+1 or not np.all(np.isfinite(v))):
        raise ValueError('require positive geometry and finite compatible directions')
    norms = np.linalg.norm(v, axis=0)
    if np.any(norms == 0) or np.any(np.abs(v.sum(axis=0)) > 1e-12*np.sqrt(len(v))*norms):
        raise ValueError('infinite-center form requires nonzero mean-zero directions')
    # Remove only accepted roundoff-sized means; a genuine constant is rejected.
    return v-v.mean(axis=0)


def probe_directions(x):
    names = ['sine_mix_2_6_10', 'exp_sine_3', 'runge', 'quadratic', 'sine_2']
    columns = [np.sin(2*np.pi*x)+.5*np.sin(6*np.pi*x)+.25*np.sin(10*np.pi*x),
               np.exp(np.sin(3*np.pi*x)), 1/(1+25*x*x), np.sqrt(5)*x*x,
               np.sqrt(2)*np.sin(2*np.pi*x)]
    polynomial = np.polynomial.legendre.legvander(x, 7)
    basis, _ = np.linalg.qr(polynomial)
    for k in [2, 6, 10, 20]:
        wave = np.sin(k*np.pi*x)
        names += [f'sine_{k}_pi', f'gaussian_sine_{k}_pi', f'moment7_sine_{k}_pi']
        columns += [wave, np.exp(-.5*(x/.25)**2)*wave, wave-basis@(basis.T@wave)]
    return names, center_normalize(np.column_stack(columns))

def csch(z):
    a = np.abs(z)
    return np.sign(z)*2*np.exp(-a)/(-np.expm1(-2*a))

def symbol_integral(v, n, q, gamma, points=8192, aliases=3):
    """Return exact-symbol quadrature, q1 cap, general cap, alias L2 allowance.

    The same positive gamma argument defines the actual symbol and the cap
    Gamma in the two envelopes. Cross-gamma comparisons evaluate the envelope
    at Gamma and the actual dictionary at any gamma<=Gamma. Every return is a
    per-direction array, except the unavailable q1 envelope is None for q>1.
    The alias allowance does not bound quadrature or floating-point error.
    """
    v = _directions(v, n, q, gamma)
    if points <= n or aliases < 0:
        raise ValueError('FFT grid must exceed n; alias count must be nonnegative')
    h=2/n; lam=gamma*h; m=len(v)
    theta=2*np.pi*(np.arange(points)+.5)/points
    theta=np.where(theta>np.pi,theta-2*np.pi,theta)
    vhat=[]; symbols=[]
    for s in range(q):
        a=v[s::q]
        phase=np.exp(-1j*np.pi*np.arange(len(a))/points)
        vhat.append(np.fft.fft(a*phase[:,None],n=points,axis=0))
        t=np.zeros(points,dtype=complex)
        for k in range(-aliases, aliases+1):
            shifted=theta+2*np.pi*k
            t += csch(np.pi*shifted/(2*lam))*np.exp(1j*shifted*s/q)
        symbols.append(np.pi/(1j*lam)*t)
    vhat=np.array(vhat); symbols=np.array(symbols)
    fhat=np.einsum('sp,spv->pv',symbols.conj(),vhat)
    exact=np.mean(np.abs(fhat)**2,axis=0)/m
    cap_amplitude=np.zeros_like(fhat.real)
    for k in range(-aliases, aliases+1):
        shifted=theta+2*np.pi*k
        phase=np.exp(-1j*np.arange(q)[:,None]*shifted[None,:]/q)
        b=np.einsum('sp,spv->pv',phase,vhat)
        cap_amplitude+=np.pi/lam*np.abs(csch(np.pi*shifted/(2*lam)))[:,None]*np.abs(b)
    general_envelope=np.mean(cap_amplitude**2,axis=0)/m
    if q==1:
        cap=(np.pi/lam)**2*csch(np.pi*theta/(2*lam))**2
        envelope=np.mean(cap[:,None]*np.abs(vhat[0])**2,axis=0)/m
    else:
        envelope=None
    minimum=np.pi*np.pi*(2*aliases+1)/(2*lam)
    alias_l2=4*np.pi/lam*np.exp(-minimum)/((-np.expm1(-np.pi*np.pi/lam))*(-np.expm1(-2*minimum)))
    return exact,envelope,general_envelope,alias_l2*np.linalg.norm(v, axis=0)


def direct(v,n,q,gamma,halo,tail_tolerance=1e-26):
    """Independent finite/extended-center forms and an analytic omitted tail."""
    v = _directions(v, n, q, gamma)
    if halo < 0 or tail_tolerance <= 0:
        raise ValueError('require nonnegative halo and positive tail tolerance')
    h=2/n;x=np.linspace(-1,1,q*n+1);m=len(x)
    ar=2/np.sqrt(m)*np.sum(np.abs(v)*np.exp(-2*gamma*(1-x))[:,None],axis=0)
    al=2/np.sqrt(m)*np.sum(np.abs(v)*np.exp(-2*gamma*(x+1))[:,None],axis=0)
    margin=halo
    denominator=-np.expm1(-4*gamma*h)
    while np.max((ar*ar+al*al)*np.exp(-4*gamma*(margin+1)*h)/denominator)>tail_tolerance:
        margin+=1
    all_energy=np.zeros(v.shape[1]); finite=np.zeros(v.shape[1])
    for start in range(-margin,n+margin+1,96):
        j=np.arange(start,min(start+96,n+margin+1))
        c=-1+h*j
        phi=np.tanh(gamma*(x[:,None]-c[None,:]))
        phi-=phi.mean(axis=0,keepdims=True)
        corr=phi.T@v/np.sqrt(m)
        energies=corr*corr
        all_energy+=energies.sum(axis=0)
        finite+=energies[(j>=-halo)&(j<=n+halo)].sum(axis=0)
    tail=(ar*ar+al*al)*np.exp(-4*gamma*(margin+1)*h)/denominator
    return finite,all_energy,tail,margin


def necessary_time(rate, threshold=.01):
    """Jensen bound for unit target: None for zero rate, zero if vacuous."""
    if rate < 0 or not 0 < threshold < 1:
        raise ValueError('require nonnegative rate and threshold in (0,1)')
    if rate == 0:
        return None
    if rate >= 1:
        return 0
    return int(np.ceil(np.log(1/threshold)/(-np.log1p(-rate))))


def select_case(data, n, q, gamma, name):
    """Select an explicitly named case for an evidence figure."""
    row = next(r for r in data['records'] if (r['n'],r['q'],r['gamma']) == (n,q,gamma))
    return row, next(c for c in row['cases'] if c['name'] == name)


def collect(geometries=((512,16),(512,1),(128,1)), gammas=(8,12,16,64)):
    records=[]
    for n,q in geometries:
        x=np.linspace(-1,1,q*n+1);names,v=probe_directions(x)
        for gamma in gammas:
            finite,total,tail,margin=direct(v,n,q,gamma,int(np.ceil(np.sqrt(n))))
            spectral,cap,general_cap,alias_l2=symbol_integral(v,n,q,gamma,8192)
            fine,cap_fine,general_fine,_=symbol_integral(v,n,q,gamma,16384)
            relative=np.abs(spectral-total)/np.maximum(total,1e-300)
            assert np.all(total+1e-12>=finite)
            assert np.max(relative)<2e-6, (n,q,gamma,np.max(relative))
            if cap is not None:
                assert np.all(cap+1e-12>=spectral)
            cases=[]
            for j,name in enumerate(names):
                cases.append(dict(name=name,finite_quadratic=float(finite[j]),
                    infinite_direct_lower=float(total[j]),infinite_direct_upper=float(total[j]+tail[j]),
                    omitted_tail_bound=float(tail[j]),symbol_quadrature=float(spectral[j]),
                    symbol_quadrature_refined=float(fine[j]),
                    direct_symbol_relative_error=float(relative[j]),
                    infinite_to_finite_ratio=float(total[j]/finite[j]),
                    general_cap_envelope=float(general_cap[j]),
                    general_cap_refined=float(general_fine[j]),
                    general_cap_alias_energy_allowance=float(2*np.sqrt(general_cap[j])*alias_l2[j]+alias_l2[j]**2),
                    general_cap_to_infinite_ratio=float(general_cap[j]/total[j]),
                    scalar_cap_envelope=None if cap is None else float(cap[j]),
                    scalar_cap_refined=None if cap is None else float(cap_fine[j]),
                    cap_to_infinite_ratio=None if cap is None else float(cap[j]/total[j])))
            row=dict(n=n,q=q,m=len(x),gamma=gamma,lambda_dimensionless=gamma*2/n,
                original_halo=int(np.ceil(np.sqrt(n))),extended_halo=margin,cases=cases)
            records.append(row)
            print(json.dumps(dict(n=n,q=q,gamma=gamma,primary_ratio=cases[0]['infinite_to_finite_ratio'],
                max_ratio=max(c['infinite_to_finite_ratio'] for c in cases),max_symbol_error=float(max(relative)))),flush=True)
    for row in records:
        row['cross_gamma_cap_checks']=[]
        for cap_gamma in gammas:
            if cap_gamma<row['gamma']:continue
            bound=next(r for r in records if r['n']==row['n'] and r['q']==row['q'] and r['gamma']==cap_gamma)
            ratios=[]
            for actual,b in zip(row['cases'],bound['cases']):
                assert actual['name']==b['name']
                assert actual['finite_quadratic'] <= b['general_cap_refined']*(1+1e-10)
                ratios.append(b['general_cap_refined']/actual['finite_quadratic'])
            row['cross_gamma_cap_checks'].append(dict(Gamma=cap_gamma,
                minimum_bound_to_actual_ratio=float(min(ratios)),maximum_bound_to_actual_ratio=float(max(ratios))))
        if row['q']==1:
            W=row['n']+2*row['original_halo']+1
            eta=.5/(W+1)
            row['representability']='W>=m; bias plus m distinct tanh centers span R^m by Cauchy determinant for every gamma>0.'
            row['common_analytic_step']=eta
            for c in row['cases']:
                rate=eta*c['scalar_cap_refined']
                c['gamma_cap_directional_rate']=rate
                c['gamma_cap_jensen_necessary_1pct']=necessary_time(rate)
    result=dict(status='FP64 verification, not outward-rounded interval certification',
        formula='For meanzero v: v*K_finite*v <= (2*pi*m)^-1 integral |T(theta)^* V(theta)|^2 dtheta; T_s=pi/(i*lambda) sum_k csch(pi*(theta+2*pi*k)/(2*lambda))*exp(i*(theta+2*pi*k)*s/q).',
        scalar_cap='q=1: sigma_lambda(theta)<=4*M_gamma(theta/h)^2/theta^2 for 0<abs(theta)<pi; proved by signed alias pairing.',
        general_cap='For gamma<=Gamma, the exact integrand is bounded by (sum_k 2*M_Gamma((theta+2*pi*k)/h)/abs(theta+2*pi*k)*abs(B_k(theta)))^2, B_k=sum_s V_s exp(-i*(theta+2*pi*k)*s/q). Audit uses Gamma=gamma, 7 aliases; remaining aliases exponentially negligible but not interval certified.',
        tail_formula='A_right=2/sqrt(m)*sum_i |v_i| exp(-2gamma*(xmax-x_i)), similarly left; omitted center tail <=(A_right^2+A_left^2)*exp(-4gamma*(extended_halo+1)*h)/(1-exp(-4gamma*h)).',
        inputs='All 5 analytic targets are projected off sample constants and normalized. Added sine, Gaussian-windowed sine (sigma=.25), and degree7-Legendre-complement sine tests at k=2,6,10,20.',
        reference='Direct feature correlations with centered columns, uniform extended centers, analytic bound on omitted tails below1e-26; numerics not interval certified.',
        quadrature='8192 and16384 midpoint trapezoid nodes; 7 explicit aliases k=-3..3; no eigensolve and no training.',
        jensen='For normalized meanzero target and eta<=1/(W+1): E_n >= (1-eta*y*K*y)^n >= (1-eta*Q_Gamma(y))^n providedetaQ<1. q1 grids here have W>=m and exact full row rank by Cauchy formula; numerical necessarytimes use common eta=.5/(W+1) and are formula evaluations, not executed hits or interval certificates.',
        source_hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__).resolve(), Path(__file__).with_name('structured_gamma.py'))},
        numpy_version=np.__version__, records=records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__=='__main__':
    main()
