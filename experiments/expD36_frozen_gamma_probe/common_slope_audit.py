"""Independent Arb audit of readout timing endpoints on nominal real tanh.

Uses the common-slope Gram identity and integer matrix powers. Neither a GD
trajectory nor an eigendecomposition enters this checker. It certifies the
reported endpoint statements, not the FP64 error-envelope implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from flint import arb, arb_mat, ctx

from . import core, common_slope_analysis as analysis


def finite_gram(x, centers, gamma, y):
    """Nominal real features on exact archived binary inputs, including bias."""
    x = [arb(float(t)) for t in x]
    centers = [arb(float(t)) for t in centers]
    y = [arb(float(t)) for t in y]
    m, width = len(x), len(centers)
    root_m = arb(m).sqrt()
    means, diagonal, corr = [], [], [sum(y, arb(0))/root_m]
    for center in centers:
        phi = [(arb(gamma)*(t-center)).tanh() for t in x]
        means.append(sum(phi, arb(0))/m)
        diagonal.append(sum((a*a for a in phi), arb(0))/m)
        corr.append(sum((a*b for a,b in zip(phi,y)), arb(0))/root_m)
    entries = [[arb(1)]+means]
    denominators = {}
    for i in range(width):
        row = [means[i]]
        for j in range(width):
            if i == j:
                row.append(diagonal[i])
            else:
                offset = centers[j]-centers[i]
                key = str(offset)
                if key not in denominators:
                    denominators[key] = (arb(gamma)*offset).tanh()
                row.append(1-(means[i]-means[j])/denominators[key])
        entries.append(row)
    return arb_mat(entries), arb_mat([[v] for v in corr]), sum((v*v for v in y),arb(0))


def augmented(gram, corr, eta):
    size = gram.nrows()
    eta = arb(float(eta))
    entries = [[arb(int(i==j))-eta*gram[i,j] for j in range(size)]
               +[eta*corr[i,0]] for i in range(size)]
    entries.append([arb(0)]*size+[arb(1)])
    return arb_mat(entries)


def residual_squared(gram, corr, norm_sq, evolution, step):
    power = evolution**int(step)
    w = arb_mat([[power[i,gram.nrows()]] for i in range(gram.nrows())])
    return (norm_sq-2*(w.transpose()*corr)[0,0]+(w.transpose()*gram*w)[0,0])/norm_sq


def endpoints(value):
    return [float(np.nextafter(float(value.lower()),-np.inf)),
            float(np.nextafter(float(value.upper()),np.inf))]


def run(root, output, precision=192):
    ctx.prec = precision
    arrays = np.load(root/'common/N512/arrays.npz')
    summary = analysis.read(output/'summary.json')
    all_results = []
    for dictionary in summary['dictionaries']:
        tick = time.monotonic()
        gamma, eta = dictionary['gamma'], dictionary['eta']
        gram, corr, norm_sq = finite_gram(arrays['x_train'],arrays['centers'],gamma,arrays['y_train'][:,0])
        eta_L_upper = max((arb(float(eta))*sum((abs(gram[i,j]) for j in range(gram.ncols())),arb(0))).upper()
                          for i in range(gram.nrows()))
        if not eta_L_upper < 1:
            raise ValueError('Interval Gershgorin check does not establish contraction')
        evolution = augmented(gram,corr,eta)
        evaluated = {}
        methods = {}
        for method in ['analytic','combined']:
            bracket = analysis.best(summary['rows'],gamma,method=method,
                resolution_key=summary.get('resolution_key','degree'))
            n_low, n_high = bracket['necessary']-1, bracket['sufficient']
            for n in [n_low,n_high]:
                if n is not None and n not in evaluated:
                    evaluated[n] = residual_squared(gram,corr,norm_sq,evolution,n)
            lower = evaluated[n_low]
            upper = evaluated[n_high] if n_high is not None else None
            # A decimal 1/100 is the mathematical threshold, enclosed by Arb.
            threshold = (arb(1)/100)**2
            certified = bool(lower > threshold and upper is not None and upper <= threshold)
            methods[method] = dict(**bracket, excluded_step=n_low,
                excluded_error_squared=endpoints(lower),
                sufficient_error_squared=endpoints(upper) if upper is not None else None,
                status='interval_certified_endpoints' if certified else 'unresolved')
        result = dict(gamma=gamma, target=analysis.TARGETS[0], precision_bits=precision,
            eta=eta, eta_L_gershgorin_upper=float(eta_L_upper), methods=methods,
            seconds=time.monotonic()-tick,
            statement='Nominal real tanh; exact archived binary grid, centers, target, and step',
            computation='Common-slope Gram identity and Arb integer matrix powers; no GD or SVD')
        all_results.append(result)
        core.write_json(output/f'interval_g{gamma}.json',result)
        print(gamma, {k:v['status'] for k,v in methods.items()}, result['seconds'],flush=True)
    core.write_json(output/'interval_audit.json',dict(results=all_results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/checkpoint_D_optimizers/expD36_frozen_gamma_probe/full_sweep'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--precision',type=int,default=192)
    args = parser.parse_args()
    run(args.root,args.output or args.root/'refinements/common_slope_polynomial',args.precision)


if __name__ == '__main__':
    main()
