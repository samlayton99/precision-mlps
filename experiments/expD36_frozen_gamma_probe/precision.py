"""Selected full-grid nominal-tanh witnesses at 80 and 120 decimal digits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.linalg import svd

from . import core


def witness(n, m, gamma, k, digits, deadline=float('inf')):
    """Rebuild the exact endpoint grid and odd target; stream feature columns.

    Uniform-grid orthogonal polynomials avoid a dense arbitrary-precision QR.
    Symmetric centers have equal directional gradients for an odd target tail.
    Exponentials advance by the exact constant center-spacing ratio, so no
    rounded FP64 features are promoted to arbitrary precision.
    """
    start = time.monotonic()
    with mp.workdps(digits):
        x = [mp.mpf(-1)+mp.mpf(2)*i/(m-1) for i in range(m)]
        y = [mp.sin(2*mp.pi*z)+mp.sin(6*mp.pi*z)/2+mp.sin(10*mp.pi*z)/4 for z in x]
        previous = [mp.mpf(0)]*m
        p = [mp.mpf(1)]*m
        tail = y.copy()
        previous_a = mp.mpf(0)
        orthogonality_error = mp.mpf(0)
        for degree in range(k+1):
            norm = mp.fsum(v*v for v in p)/m
            cross = mp.fsum(a*b for a, b in zip(p, previous))/m
            orthogonality_error = max(orthogonality_error, abs(norm-1), abs(cross))
            coefficient = mp.fsum(a*b for a, b in zip(p, y))/m
            tail = [a-coefficient*b for a, b in zip(tail, p)]
            if degree < k:
                d = degree+1
                a = mp.sqrt(mp.mpf(d*d)*(m*m-d*d)/((4*d*d-1)*(m-1)**2))
                pn = [(z*v-previous_a*w)/a for z, v, w in zip(x, p, previous)]
                previous, p, previous_a = p, pn, a
        tail_sq = mp.fsum(v*v for v in tail)/m
        target_sq = mp.fsum(v*v for v in y)/m
        e = mp.sqrt(tail_sq/target_sq)
        bias = mp.fsum(tail)/m
        mu = bias*bias/tail_sq
        radius = int(np.ceil(np.sqrt(n)))
        slope = mp.mpf(gamma)
        exp_values = [mp.exp(2*slope*z) for z in x]
        ratio = mp.exp(-4*slope/n)
        # Center 0 is unpaired; positive centers are paired with their negatives.
        for center_index in range(n//2+radius+1):
            if time.monotonic() >= deadline:
                raise TimeoutError('Precision witness exceeded the reserved CPU deadline')
            correlation = mp.fsum(t*(v-1)/(v+1) for t, v in zip(tail, exp_values))/m
            mu += (1 if center_index == 0 else 2)*correlation**2/tail_sq
            exp_values = [v*ratio for v in exp_values]
        return dict(n=n, m=m, gamma=gamma, k=k, digits=digits,
            E=mp.nstr(e, 65), mu=mp.nstr(mu, 65),
            polynomial_norm_neighbor_error=mp.nstr(orthogonality_error, 12),
            seconds=time.monotonic()-start, model='nominal_real_tanh_exact_grid',
            status='high_precision_evaluation_not_interval_certified')


def factorization_check(root, gamma):
    bank = dict(np.load(root/f'data/raw_g{gamma}.npz'))
    y = np.load(root/'data/common.npz')['y']
    u, s, _ = svd(bank['J'], full_matrices=False, lapack_driver='gesvd')
    keep = s > 1e-14*s[0]
    alpha = u[:, keep].T@y
    residual = y-u[:, keep]@alpha
    floor = np.sum(residual*residual, axis=0)
    curves = []
    for step in [0, 20000, 100000]:
        reference = core.spectral_error(step, bank['singular'][bank['keep']], bank['loadings'],
            bank['floor_sq'], bank['norm_y'], .5/float(bank['L']))
        alternate = core.spectral_error(step, s[keep], alpha, floor, bank['norm_y'], .5/float(bank['L']))
        curves.append(dict(step=step, gesdd=reference.tolist(), gesvd=alternate.tolist(),
                           max_abs_difference=float(np.max(np.abs(reference-alternate)))))
    return dict(gamma=gamma, relative_L_difference=float(abs(s[0]**2/float(bank['L'])-1)),
                curves=curves)


def run(root, seconds):
    deadline = time.monotonic()+seconds
    cfg = core.config()
    certificates = json.loads((root/'certificates.json').read_text())
    destination = root/'validation/precision.json'
    result = dict(witnesses=[], comparisons=[], factorizations=[], complete=False)
    for name in cfg['maps']:
        for gamma in cfg['gammas']:
            with np.load(root/f'data/{name}_g{gamma}.npz') as bank:
                core.save_arrays(root/f'analysis/{name}_g{gamma}.npz',
                    **{k: bank[k] for k in bank.files if not k.startswith('J')})
    for gamma in [4, 16]:
        result['factorizations'].append(factorization_check(root, gamma))
        core.write_json(destination, result)
    try:
        for gamma in [4, 1]:
            row = next(c for c in certificates if c['map'] == 'raw' and c['gamma'] == gamma
                       and c['target'] == 'sine_mix_2_6_10' and c['kind'] == 'directional'
                       and c['epsilon'] == .01)
            values = []
            for digits in [80, 120]:
                value = witness(cfg['n'], cfg['samples_per_cell']*cfg['n']+1, gamma, row['k'], digits, deadline)
                result['witnesses'].append(value)
                values.append(value)
                core.write_json(destination, result)
                print(f'PRECISION gamma={gamma} k={row["k"]} digits={digits} E={value["E"]} mu={value["mu"]}', flush=True)
            with mp.workdps(120):
                a, b = (mp.mpf(v['mu']) for v in values)
                relative = abs(a/b-1)
                accuracy = float(-mp.log10(relative)) if relative else 65.
            high = float(values[-1]['mu'])
            e = float(values[-1]['E'])
            high_bound = core.bound(np.array([e]), np.log(np.array([high])), .01, row['L'])
            result['comparisons'].append(dict(gamma=gamma, k=row['k'], mu_fp64=row['mu'], mu_high=high,
                fp64_relative_difference=abs(row['mu']/high-1), agreeing_decimal_digits=accuracy,
                fp64_log10_bound=row['log10_bound'], high_log10_bound=high_bound['log10_bound'],
                caveat='Nominal high-precision features; stored FP64 features and L are separate approximations'))
            core.write_json(destination, result)
        result['complete'] = True
    except TimeoutError as error:
        result['stopped'] = str(error)
    core.write_json(destination, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=2100)
    args = parser.parse_args()
    run(args.root, args.seconds)


if __name__ == '__main__':
    main()
