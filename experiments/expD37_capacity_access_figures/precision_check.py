"""80/120-digit access audit at gamma=4 and the mixed-target certificate maximum.

Rebuild targets, polynomials, and tanh features in mpmath. This audits the
directional certificate, not the full high-precision singular-value spectrum.
"""
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_D_optimizers/expD37_capacity_access_figures/data'
SOURCE = ROOT/'results/checkpoint_D_optimizers/expD36_frozen_lambda_sweep/data'


def compute(digits, x_double, centers_double, k, gamma):
    started = time.perf_counter()
    with mp.workdps(digits):
        x = [mp.mpf(float(t)) for t in x_double]
        centers = [mp.mpf(float(t)) for t in centers_double]
        y = [(mp.sin(2*mp.pi*t)+mp.mpf('.1')*mp.sin(20*mp.pi*t))/mp.sqrt(mp.mpf('.505')) for t in x]
        dot = lambda a, b: mp.fsum(u*v for u, v in zip(a, b))
        norm = lambda a: mp.sqrt(dot(a, a))
        p = [1/mp.sqrt(len(x))]*len(x)
        previous = [mp.mpf(0)]*len(x)
        previous_beta = mp.mpf(0)
        tail = list(y)
        basis = []
        for degree in range(k+1):
            basis.append(p)
            coefficient = dot(p, y)
            tail = [t-coefficient*v for t, v in zip(tail, p)]
            alpha = mp.fsum(t*v*v for t, v in zip(x, p))
            residual = [(t-alpha)*v-previous_beta*w for t,v,w in zip(x,p,previous)]
            beta = norm(residual)
            previous, p, previous_beta = p, [v/beta for v in residual], beta
        tailnorm = norm(tail)
        q = [v/tailnorm for v in tail]
        normy = norm(y)
        # Include the physical output bias in B.
        inner = [mp.fsum(q)/mp.sqrt(len(x))]
        for center in centers:
            inner.append(mp.fsum(v*mp.tanh(gamma*(t-center)) for t,v in zip(x,q))/mp.sqrt(len(x)))
        value = mp.fsum(v*v for v in inner)
        # Polynomial orthogonality, including the highest degree used.
        leakage = max(abs(dot(p, q)) for p in basis)
        return {'digits': digits, 'gamma': gamma, 'degree': k,
                'squared_access': mp.nstr(value, digits),
                'relative_target_tail': mp.nstr(tailnorm/normy, digits),
                'tail_polynomial_leakage': mp.nstr(leakage, 8),
                'seconds': time.perf_counter()-started}


def main():
    z=np.load(OUT/'theorem_diagnostics.npz')
    ref=np.load(SOURCE/'reference.npz')
    k=int(z['directional_maximizing_k'][0, 2])
    results=[]
    for digits in [80,120]:
        result=compute(digits,ref['x_train'],ref['centers'],k,4)
        print(json.dumps(result),flush=True)
        results.append(result)
    fp64=float(z['access'][0,k,2]*z['squared_operator_norm'][0])
    value=float(results[-1]['squared_access'])
    with mp.workdps(130):
        agreement=abs(mp.mpf(results[0]['squared_access'])/mp.mpf(results[1]['squared_access'])-1)
    report={'cells':results,'fp64_squared_access':fp64,
            'fp64_relative_access_error':abs(fp64/value-1),
            'relative_agreement_80_vs_120_digits':str(agreement),
            'scope':'Full high-precision target, polynomial projection and directional access; SVD hitting times remain FP64 with driver/cutoff audits.'}
    (OUT/'precision_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['fp64_relative_access_error']>.01:
        raise AssertionError('Measured certificate needs high-precision correction')


if __name__=='__main__':
    main()
