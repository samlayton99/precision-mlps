#!/usr/bin/env python3
"""Check the Fourier-ratio bridge and a few corrected Radon scalar profiles.

These are scalar-profile comparisons, not a full angular reconstruction or
a benchmark of the proposed lambda selector. No coefficients are fitted.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp1f1
from construction import geometry, encode

ROOT = Path(__file__).resolve().parent


def alias_ratio(lam):
    x = np.pi**2/lam
    return 2*x*np.exp(-x)/(-np.expm1(-2*x))


def main():
    root = brentq(lambda lam: np.log(alias_ratio(lam))-np.log(2.**-52), .1, .4,
                  xtol=1e-15)
    bridge_errors = []
    for lam in np.geomspace(.05,1.,80):
        x = np.pi**2/lam
        original = 64*np.pi/(128*lam)*np.exp(-x)
        rewritten = 32/(128*np.pi)*(-np.expm1(-2*x))*alias_ratio(lam)
        bridge_errors.append(abs(original-rewritten)/original)
    assert max(bridge_errors) < 1e-14
    # Compare the new optional argument with the unchanged default schedule.
    x = np.linspace(-1,1,4097)
    profile = lambda z: np.stack([hyp1f1(d/2,.5,-3*np.asarray(z)**2)
                                  for d in [1,2,3,4]],axis=-1)
    y = np.real(profile(x))
    weights_default, _ = encode(profile,128)
    weights_explicit, _ = encode(profile,128,lam=geometry(128)['lam'])
    assert np.array_equal(weights_default,weights_explicit)
    rows = []
    with (ROOT/'findings/scratchpad.md').open('a') as f:
        f.write('\n## Fourier-ratio consistency check — 2026-10-09\n\n'
                'Before running: compare the existing corrected encoder on the canonical scalar profiles '
                'of exp(-3*||x||^2), d=1..4, n=128 and 512, R=n/4. Test 2*pi/R, '
                'the basic Fourier-ratio root, 0.25, 0.30 and 0.40. All are admissible '
                'with delta=1. This isolates scalar behavior; no angular-network claim. '
                'The optional lambda argument preserves the existing default geometry.\n')
    for n in [128,512]:
        halo = int(n/4)
        candidates = [('halo_balance',2*np.pi/halo),('kernel_ratio',root),
                      ('quarter',.25),('0.30',.30),('0.40',.40)]
        for rule, lam in candidates:
            geo = geometry(n,lam=lam)
            weights, _ = encode(profile,n,lam=lam)
            bias = y[len(x)//2]-np.tanh(-geo['gamma']*geo['centers'])@weights
            pred = np.tanh(geo['gamma']*(x[:,None]-geo['centers']))@weights+bias
            relative = np.linalg.norm(pred-y,axis=0)/np.linalg.norm(y,axis=0)
            sup = np.max(abs(pred-y),axis=0)
            for i,d in enumerate([1,2,3,4]):
                rows.append(dict(d=d,n_cells=n,halo=halo,rule=rule,lam=float(lam),
                                 relative_l2=float(relative[i]),sampled_linf=float(sup[i]),
                                 readout_l1=float(np.sum(abs(weights[:,i]))),
                                 quadrature_order=geo['quadrature_order']))
            message=f'n={n}, R={halo}, {rule}, lambda={lam:.6g}: scalar relative errors '+np.array2string(relative,precision=3)
            print(message,flush=True)
            with (ROOT/'findings/scratchpad.md').open('a') as f:
                f.write('\n- '+message+'\n')
    data=dict(root_at_2_pow_minus_52=root,ratio_at_quarter=alias_ratio(.25),
              largest_bridge_relative_difference=max(bridge_errors),
              scope='Corrected scalar Radon profiles only; 4097 independent evaluation points; float64.',
              rows=rows)
    (ROOT/'results/lambda_consistency.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Root:',root,'A(0.25):',alias_ratio(.25),flush=True)


if __name__=='__main__':
    main()
