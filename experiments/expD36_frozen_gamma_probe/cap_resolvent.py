"""Sharper conversion of a uniform Rayleigh certificate into learning delay.

Cauchy in H+tI gives yhat^T(H+tI)^-1 yhat >= delta^2/(beta+t).
If a <= min_{0<=s<=1} [(1-chi*s)^(2n)-z/(t+s)], z,t>0,
then E(n)^2 >= a+z*delta^2/(beta+t). The scalar minimum is
independently enclosed with Arb, so optimization is only a candidate search.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize

from . import core


def scalar_candidate(steps, beta, delta, chi=.5):
    if steps < 1 or beta <= 0 or not 0 < delta <= 1:
        raise ValueError('Require positive step, beta and overlap')

    def value(parameters):
        t, z = beta*np.exp(parameters)
        def objective(s):
            return np.exp(2*steps*np.log1p(-chi*s))-z/(t+s)
        def derivative_sign(s):
            return (np.log(z)-2*np.log(t+s)-np.log(2*steps*chi)
                    -(2*steps-1)*np.log1p(-chi*s))
        # The log derivative ratio is strictly convex, with this unique minimum.
        turning = float(np.clip((2-(2*steps-1)*chi*t)/(chi*(2*steps+1)), 0, 1))
        candidates = [objective(0.), objective(1.)]
        if turning < 1 and derivative_sign(turning) < 0 < derivative_sign(1.):
            root = brentq(derivative_sign, turning, 1., xtol=5e-324, rtol=1e-12)
            candidates.append(objective(root))
        return min(candidates)+delta*delta*z/(beta+t)

    best = None
    for start in [[-1., -1.], [-4., -4.], [-8., -8.]]:
        result = minimize(lambda p:-value(p), start, method='Nelder-Mead',
            bounds=[(-30., 12.), (-40., 12.)],
            options=dict(maxiter=250, xatol=1e-8, fatol=1e-12))
        if best is None or result.fun < best.fun:
            best = result
    t, z = beta*np.exp(best.x)
    return dict(t=float(t), z=float(z), estimated_error_squared=float(-best.fun))


def enclose(steps, beta, delta, t, z, chi=.5, epsilon=.01, max_intervals=16384):
    """Cover [0,1] completely; early stopping never discards an interval."""
    from flint import arb, ctx
    old = ctx.prec
    precision = max(128, int(math.ceil(math.log2(max(steps, 1))))+96)
    ctx.prec = precision
    try:
        aa, zz, bb, dd, cc = map(arb, [t, z, beta, delta, chi])
        if t <= 0 or z < 0:
            raise ValueError('Require t>0,z>=0')
        offset = zz*dd**2/(bb+aa)
        tolerance = arb(float(epsilon))**2
        def bound(left, right):
            # Each term is monotone in the opposite direction; use its minimum.
            value = (1-cc*arb(right))**(2*steps)-zz/(aa+arb(left))
            return float(np.nextafter(float(value.lower()), -np.inf))
        smallest = max(min(t, 1/max(steps, 1))*1e-4, 1e-300)
        edges = np.unique(np.r_[0., np.geomspace(min(smallest, 1.), 1., 64)])
        heap = [(bound(float(l), float(r)), float(l), float(r)) for l, r in zip(edges[:-1], edges[1:])]
        heapq.heapify(heap)
        splits = 0
        while heap and not arb(heap[0][0])+offset > tolerance and splits < max_intervals:
            value, left, right = heapq.heappop(heap)
            mid = (left+right)/2
            if mid == left or mid == right:
                heapq.heappush(heap, (value, left, right))
                break
            for lo, hi in [(left, mid), (mid, right)]:
                heapq.heappush(heap, (bound(lo, hi), lo, hi))
            splits += 1
        lower = arb(heap[0][0])+offset
        return dict(status='interval_certified', excluded=bool(lower > tolerance),
            steps=int(steps), t=t, z=z, scalar_minimum_lower=heap[0][0],
            error_squared_lower=float(np.nextafter(float(lower.lower()), -np.inf)),
            subdivisions=splits, covered_interval=[0, 1], precision_bits=precision)
    finally:
        ctx.prec = old


def improve(certificate, baseline, epsilon=.01, chi=.5):
    beta, delta = certificate['beta'], certificate['delta']
    if certificate['status'] != 'interval_certified' or beta <= 0 or delta <= epsilon or beta >= delta**2:
        return dict(bound=int(baseline), status='no_improvement')
    # A feasible abstract H=(beta/delta^2)yhat*yhat^T bounds what this
    # information alone could prove. This is not a tanh-family upper witness.
    hi = math.ceil(math.log(epsilon)/math.log1p(-chi*beta/(delta*delta)))
    lo = max(1, int(baseline)-1)
    if hi <= lo:
        return dict(bound=int(baseline), status='no_improvement')
    chosen = None
    while hi-lo > 1:
        mid = (lo+hi)//2
        proposal = scalar_candidate(mid, beta, delta, chi)
        if proposal['estimated_error_squared'] > epsilon**2:
            lo, chosen = mid, proposal
        else:
            hi = mid
    if chosen is None:
        return dict(bound=int(baseline), status='no_improvement')
    # Retreat if the last candidate is too close to the threshold to enclose.
    for excluded_step in [lo, max(int(baseline), lo-1), max(int(baseline), int(.99*lo))]:
        proposal = scalar_candidate(excluded_step, beta, delta, chi)
        proof = enclose(excluded_step, beta, delta, proposal['t'], proposal['z'], chi, epsilon)
        if proof['excluded']:
            return dict(bound=max(int(baseline), excluded_step+1),
                        status='interval_certified', proof=proof,
                        beta=beta, delta=delta, witness=certificate.get('label'))
    return dict(bound=int(baseline), status='no_certified_improvement')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    for path in sorted((a.root/'certificates').glob('*/result.json')):
        result = json.loads(path.read_text())
        baseline = result['bounds']['0.01']['bound']
        rows = [improve(c, baseline) for c in result['certificates']]
        best = max(rows, key=lambda r:r['bound'])
        core.write_json(path.parent/'resolvent_refinement.json', dict(n=result['n'], cap=result['cap'],
            target=result['target'], cdf_bound=baseline, best=best, candidates=rows))
        print('RESOLVENT', path.parent.name, baseline, best['bound'], flush=True)


if __name__ == '__main__':
    main()
