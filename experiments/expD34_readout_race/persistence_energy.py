"""Predictive GD path bounds from a locally inaccessible hard residual.

Analytical inequalities evaluated in ordinary FP64, without directed rounding.
"""
from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from . import persistence as pe, persistence_theory as pt, persistence_analyze as pa, transport


def hard_cap(p, radius, degree=9):
    """Uniform |<q_degree,f>| on a parameter ball, by Chebyshev tails.

q_degree has empirical norm one and is orthogonal to lower polynomials.
Each ellipse is inside tanh's strip of analyticity. Choosing the smallest
of valid analytic upper bounds does not fit any future trajectory.
"""
    w = (len(p)-1)//3; slopes = abs(p[:w])+radius
    angle = np.linspace(.5, 1.55, 256)
    ratio = angle[None, :]/slopes[:, None]
    rho = ratio+np.sqrt(1+ratio*ratio)
    tail = 2*np.maximum(1, np.tan(angle))[None, :]*rho**(-degree)/(1-1/rho)
    per_neuron = tail.min(axis=1)
    return abs(p[2*w:3*w]) @ per_neuron+radius*np.linalg.norm(per_neuron)


def candidates(p, x, y, eta=.002):
    state = pt.tensors(p, x, y); g0 = np.linalg.norm(state['g'])
    initial_loss = np.mean(state['r']**2)/2
    target_hard = abs(transport.basis(x, 9)[:, 9] @ y/len(x))
    w = (len(p)-1)//3; rows = []
    for outer in np.geomspace(.001, .5, 301):
        constants = pt.ball_constants(p, state, outer, eta)
        upper = constants['upper']; descent = 1-eta*upper/2
        inner = (outer-eta*g0)/(1+eta*upper)
        cap = hard_cap(p, outer); floor = .5*max(target_hard-cap, 0.)**2
        gap = initial_loss-floor
        if descent <= 0 or inner <= 0 or gap <= 0: continue
        limit = inner*inner*descent/(eta*gap)
        updates = max(0, int(np.ceil(limit))-1)
        rows.append(dict(outer_radius=outer, inner_radius=inner, hard_output_cap=cap,
            initial_loss=initial_loss, loss_floor=floor, available_loss=gap,
            hessian_upper=upper, initial_gradient=g0, descent_factor=descent,
            updates=updates, slope_cap=np.max(abs(p[:w]))+inner,
            path_bound=np.sqrt(eta*updates*gap/descent)))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); all_rows = []; best_rows = []
    for start in (100000, 600000):
        pp, x, y, _ = pe.load_inputs(args.source, start, range(5))
        for seed, p in enumerate(pp):
            values = candidates(p, x, y)
            all_rows.extend(dict(seed=seed, start=start, **v) for v in values)
            best = dict(seed=seed, start=start, **max(values, key=lambda v: v['updates']))
            best_rows.append(best); print(best, flush=True)
    pa.table(args.root/'energy_candidates.csv.gz', all_rows)
    pa.table(args.root/'energy_bounds.csv', best_rows)
