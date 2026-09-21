"""Uniform heterogeneous-slope certificates, independent of any GD trajectory.

Optimization proposes X=P P^T. Only the Arb checker establishes the continuous
slope constraint; it repairs remaining slack with a bias projector. Inputs
x, centers, witness, factor and target are interpreted as exact binary floats.
The tanh features and all normalization factors are enclosed over the reals.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np
from scipy.linalg import eigh


def slow_mass(delta, beta, thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    ratio = np.minimum(np.divide(beta, thresholds, out=np.ones_like(thresholds),
                                 where=thresholds > 0), 1.)
    if beta == 0:
        ratio[:] = 0.
    return np.maximum(delta*np.sqrt(1-ratio)-np.sqrt(max(0., 1-delta*delta))*np.sqrt(ratio), 0.)**2


def time_bound(certificates, epsilon=.01, chi=.5, cap=10**18):
    """Combine guaranteed CDFs by maximum, never by adding their masses."""
    from flint import arb, ctx
    if not 0 < epsilon < 1 or not 0 < chi < 1:
        raise ValueError('Require 0<epsilon,chi<1')
    valid = [c for c in certificates if c['status'] == 'interval_certified' and c['beta'] < 1]
    if not valid:
        return dict(bound=0, thresholds=[], mass=[], status='vacuous')
    previous = ctx.prec
    ctx.prec = 96
    try:
        smallest = min(max(c['beta'], 1e-30) for c in valid)
        thresholds = np.unique(np.r_[0., np.geomspace(smallest, 1., 512), 1.])
        mass = np.zeros(len(thresholds))
        for c in valid:
            delta, beta = arb(c['delta']), arb(c['beta'])
            for i, threshold in enumerate(thresholds):
                if beta == 0:
                    p = delta**2
                elif threshold <= c['beta']:
                    continue
                else:
                    ratio = beta/arb(float(threshold))
                    overlap = delta*(1-ratio).sqrt()-(1-delta**2).sqrt()*ratio.sqrt()
                    p = overlap**2 if overlap > 0 else arb(0)
                mass[i] = max(mass[i], max(0., float(np.nextafter(float(p.lower()), -np.inf))))
        mass = np.maximum.accumulate(mass)
        mass[-1] = 1.
        atoms = [arb(float(p))-arb(float(q)) for p, q in zip(mass, np.r_[0., mass[:-1]])]
        decay = [1-arb(float(chi))*arb(float(s)) for s in thresholds]
        direct = [1-arb(float(chi))*arb(c['beta']) for c in valid if c.get('target_witness')]
        tolerance = arb(float(epsilon))**2

        def still_above(n):
            total = sum((p*d**(2*n) for p, d in zip(atoms, decay)), arb(0))
            return total > tolerance or any(d**(2*n) > tolerance for d in direct)

        lo, hi = 0, 1
        while hi < cap and still_above(hi):
            lo, hi = hi, min(2*hi, cap)
        if still_above(hi):
            bound, status = int(hi)+1, 'interval_certified_above_forecast_cap'
        else:
            while hi-lo > 1:
                mid = (lo+hi)//2
                if still_above(mid):
                    lo = mid
                else:
                    hi = mid
            # Every n<=lo is excluded rigorously; equality never overclaims.
            bound, status = int(lo)+1, 'interval_certified'
        return dict(bound=bound, log10_bound=math.log10(bound), status=status,
                    thresholds=thresholds.tolist(), mass=mass.tolist())
    finally:
        ctx.prec = previous


def optimize_candidate(x, centers, gamma_cap, witness, basis, grid_size=25):
    """Finite-grid convex relaxation, explicitly not a continuous certificate."""
    import cvxpy as cp
    x, centers = np.asarray(x), np.asarray(centers)
    v = np.asarray(witness)/np.linalg.norm(witness)
    b = np.ones(len(x))/np.sqrt(len(x))
    basis, _ = np.linalg.qr(np.column_stack([b, basis]))
    slopes = np.unique(np.r_[0., np.geomspace(gamma_cap/4096, gamma_cap, grid_size),
                             np.linspace(0, gamma_cap, grid_size)])
    r, w = basis.shape[1], len(centers)
    q = cp.Variable((r, r), PSD=True)
    z = cp.Variable(w, nonneg=True)
    constraints = []
    # Scaling resolves small target correlations without changing the SDP.
    features = [np.tanh(g*(x[:, None]-centers))/np.sqrt(len(x)) for g in slopes]
    energy_scale = max(max(float(np.max((v@a)**2)) for a in features), 1e-30)
    for a in features:
        av, ab = v@a, basis.T@a
        constraints.append(av*av/energy_scale-cp.sum(cp.multiply(ab, q@ab), axis=0) <= z)
    bb = basis.T@b
    constraints.append(float(v@b)**2/energy_scale-cp.sum(cp.multiply(bb, q@bb))+cp.sum(z) <= 0)
    problem = cp.Problem(cp.Minimize(cp.trace(q)), constraints)
    problem.solve(solver='CLARABEL', max_iter=150,
                  tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-8)
    if q.value is None:
        return dict(status=str(problem.status), factor=None)
    values, vectors = eigh((q.value+q.value.T)/2)
    keep = values > max(np.finfo(float).tiny, np.max(values)*1e-12)
    factor = basis@(vectors[:, keep]*np.sqrt(energy_scale*np.maximum(values[keep], 0.)))
    if (np.array_equal(x, -x[::-1]) and np.array_equal(centers, -centers[::-1])
            and (np.array_equal(v, v[::-1]) or np.array_equal(v, -v[::-1]))):
        # Reflection averaging preserves feasibility and trace on this geometry.
        factor = np.column_stack([.5*(factor+factor[::-1]), .5*(factor-factor[::-1])])
    return dict(status='grid_candidate', solver_status=str(problem.status),
                factor=factor, beta=float(np.sum(factor**2)), grid=slopes,
                solver_objective=float(energy_scale*problem.value), energy_scale=energy_scale)


def optimize_joint_candidate(x, centers, gamma_cap, target, basis, t, grid_size=13):
    """Joint convex witness/certificate search for a resolvent lower bound.

    In scaled variables maximize 2*yhat^T*v-||v||^2-tr(Q), with
    X=t*UQU^T satisfying the unnormalized directional certificate.
    Dividing the optimum by t gives a candidate resolvent lower bound.
    Only the subsequent independent interval checker certifies the output.
    """
    import cvxpy as cp
    if t <= 0:
        raise ValueError('The resolvent shift must be positive')
    x, centers = np.asarray(x), np.asarray(centers)
    y = np.asarray(target)/np.linalg.norm(target)
    b = np.ones(len(x))/np.sqrt(len(x))
    basis, _ = np.linalg.qr(np.column_stack([b, basis, y]))
    r, w = basis.shape[1], len(centers)
    coefficients = cp.Variable(r)
    q = cp.Variable((r, r), PSD=True)
    slack = cp.Variable(w, nonneg=True)
    grid = np.unique(np.r_[0., np.geomspace(gamma_cap/4096, gamma_cap, grid_size),
                          np.linspace(0, gamma_cap, grid_size)])
    constraints = []
    for slope in grid:
        a = np.tanh(slope*(x[:, None]-centers))/np.sqrt(len(x))
        ab = basis.T@a
        constraints.append(cp.square(coefficients@ab)/t-cp.sum(cp.multiply(ab, q@ab), axis=0) <= slack)
    bb = basis.T@b
    constraints.append(cp.square(coefficients@bb)/t-cp.sum(cp.multiply(bb, q@bb))+cp.sum(slack) <= 0)
    objective = 2*(basis.T@y)@coefficients-cp.sum_squares(coefficients)-cp.trace(q)
    problem = cp.Problem(cp.Maximize(objective), constraints)
    try:
        problem.solve(solver='CLARABEL', max_iter=200,
                      tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-8)
    except cp.error.SolverError as exc:
        return dict(status='solver_failed', detail=str(exc), factor=None)
    if coefficients.value is None or q.value is None:
        return dict(status=str(problem.status), factor=None)
    witness = basis@coefficients.value
    norm = np.linalg.norm(witness)
    if norm < 1e-12:
        return dict(status='unresolved_zero_witness', factor=None)
    values, vectors = eigh((q.value+q.value.T)/2)
    keep = values > max(1e-16, np.max(values)*1e-12)
    factor = (basis@(vectors[:, keep]*np.sqrt(t*np.maximum(values[keep], 0.))))/norm
    return dict(status='grid_candidate', solver_status=str(problem.status),
                witness=witness, factor=factor, beta=float(np.sum(factor**2)),
                delta=abs(float(witness@y))/norm, grid=grid,
                resolvent_candidate=float(problem.value/t), shift=t)


def optimize_overlap_candidate(x, centers, gamma_cap, target, basis, budget, grid_size=17):
    """Maximize target overlap at a fixed curvature budget, then normalize.

    Scaled variables avoid the very small witnesses in the joint resolvent
    problem. This remains a proposal until independently interval certified.
    """
    import cvxpy as cp
    if budget <= 0:
        raise ValueError('Require a positive curvature budget')
    x, centers = np.asarray(x), np.asarray(centers)
    y = np.asarray(target)/np.linalg.norm(target)
    b = np.ones(len(x))/np.sqrt(len(x))
    basis, _ = np.linalg.qr(np.column_stack([b, basis, y]))
    r = basis.shape[1]
    coefficients = cp.Variable(r)
    q = cp.Variable((r, r), PSD=True)
    slack = cp.Variable(len(centers), nonneg=True)
    bb = basis.T@b
    constraints = [cp.sum_squares(coefficients) <= 1/budget, cp.trace(q) <= 1,
        cp.square(coefficients@bb)-cp.sum(cp.multiply(bb, q@bb))+cp.sum(slack) <= 0]
    grid = np.unique(np.r_[0., np.geomspace(gamma_cap/4096, gamma_cap, grid_size),
                          np.linspace(0, gamma_cap, grid_size)])
    for slope in grid:
        a = basis.T@(np.tanh(slope*(x[:, None]-centers))/np.sqrt(len(x)))
        constraints.append(cp.square(coefficients@a)-cp.sum(cp.multiply(a, q@a), axis=0) <= slack)
    problem = cp.Problem(cp.Maximize((basis.T@y)@coefficients), constraints)
    try:
        problem.solve(solver='CLARABEL', max_iter=200,
                      tol_gap_abs=1e-8, tol_feas=1e-9, tol_gap_rel=1e-8)
    except cp.error.SolverError as exc:
        return dict(status='solver_failed', detail=str(exc), factor=None)
    if coefficients.value is None or q.value is None:
        return dict(status=str(problem.status), factor=None)
    witness = basis@coefficients.value
    norm = np.linalg.norm(witness)
    if norm < 1e-12:
        return dict(status='unresolved_zero_witness', factor=None)
    values, vectors = eigh((q.value+q.value.T)/2)
    keep = values > max(1e-16, np.max(values)*1e-12)
    factor = basis@(vectors[:, keep]*np.sqrt(np.maximum(values[keep], 0.)))/norm
    return dict(status='grid_candidate', solver_status=str(problem.status),
                witness=witness, factor=factor, beta=float(np.sum(factor**2)),
                delta=abs(float(witness@y))/norm, grid=grid, curvature_budget=budget)


def certify(x, centers, gamma_cap, witness, factor, target, *,
            precision=96, max_intervals=256, relative_slack=.01, target_witness=False,
            progress=False):
    """Rigorous interval certificate, including conservative bias repair.

    Subdivision may stop early without invalidating the result: every retained
    interval upper bound is included in the repair. This can make beta vacuous.
    The resulting theorem uses beta=tr(PP^T)+repair and the enclosed overlap.
    """
    from flint import arb, arb_mat, ctx
    if gamma_cap < 0 or max_intervals < 1:
        raise ValueError('Require a nonnegative cap and positive interval limit')
    previous_precision = ctx.prec
    ctx.prec = precision
    started = time.monotonic()
    try:
        x, centers, witness, factor, target = map(np.asarray, (x, centers, witness, factor, target))
        if factor.ndim != 2 or factor.shape[0] != len(x) or len(witness) != len(x) or len(target) != len(x):
            raise ValueError('Certificate arrays have incompatible sample dimensions')
        if not all(np.all(np.isfinite(a)) for a in [x, centers, witness, factor, target]):
            raise ValueError('Certificate inputs must be finite')
        if target_witness and not np.array_equal(witness, target):
            raise ValueError('Direct Jensen requires exactly the target witness')
        root_m = arb(len(x)).sqrt()
        vnorm = sum((arb(float(t))**2 for t in witness), arb(0)).sqrt()
        ynorm = sum((arb(float(t))**2 for t in target), arb(0)).sqrt()
        if vnorm.contains(0) or ynorm.contains(0):
            raise ValueError('Witness and target must be nonzero')
        v = [arb(float(t))/vnorm for t in witness]
        rows = [v]+[[arb(float(t)) for t in col] for col in factor.T]
        matrix = arb_mat([[t/root_m for t in row] for row in rows])
        beta_base = sum((arb(float(t))**2 for t in factor.flat), arb(0))
        bias_values = [sum(row, arb(0))/root_m for row in rows]
        bias_q = bias_values[0]**2-sum((t*t for t in bias_values[1:]), arb(0))
        delta = abs(sum((v[i]*arb(float(target[i])) for i in range(len(x))), arb(0)))/ynorm

        def upper(value):
            return float(np.nextafter(float(value.upper()), np.inf))

        def lower(value):
            return float(np.nextafter(float(value.lower()), -np.inf))

        def quadratic(values):
            return values[0]**2-sum((a*a for a in values[1:]), arb(0))

        all_upper, details = [], []
        symmetry = (np.array_equal(x, -x[::-1])
            and (np.array_equal(witness, witness[::-1]) or np.array_equal(witness, -witness[::-1]))
            and np.all(np.all(factor == factor[::-1], axis=0)|np.all(factor == -factor[::-1], axis=0)))
        reflected = {}
        tolerance = relative_slack*max(upper(beta_base), 1e-12)/max(len(centers), 1)
        for center in centers:
            key = abs(float(center))
            if symmetry and key in reflected:
                entry = dict(reflected[key], center=float(center), reflection_reuse=True)
                all_upper.append(entry['upper']); details.append(entry)
                continue
            offsets = [arb(float(t))-arb(float(center)) for t in x]
            cache = {}
            point_cache = {}

            def point(g):
                if g not in point_cache:
                    phi = [(arb(g)*d).tanh() for d in offsets]
                    projected = matrix*arb_mat([[a, d*(1-a*a)] for d, a in zip(offsets, phi)])
                    q = quadratic([projected[i, 0] for i in range(len(rows))])
                    derivative = 2*(projected[0, 0]*projected[0, 1]
                        -sum((projected[i, 0]*projected[i, 1] for i in range(1, len(rows))), arb(0)))
                    point_cache[g] = phi, q, derivative
                return point_cache[g]

            def evaluate(lo, hi):
                key = (lo, hi)
                if key in cache:
                    return cache[key]
                if lo == hi:
                    cache[key] = point(lo)[1]
                    return cache[key]
                # tanh(g*d) is monotone in g for each fixed real d. Reuse
                # endpoint enclosures instead of reevaluating interval tanh.
                phi = [a.union(b) for a, b in zip(point(lo)[0], point(hi)[0])]
                projected = matrix*arb_mat([[a, d*(1-a*a), -2*d*d*a*(1-a*a)]
                                           for d, a in zip(offsets, phi)])
                values = [projected[i, 0] for i in range(len(rows))]
                direct = quadratic(values)
                derivative = 2*(values[0]*projected[0, 1]
                    -sum((values[i]*projected[i, 1] for i in range(1, len(rows))), arb(0)))
                if derivative > 0:
                    result = evaluate(hi, hi)
                elif derivative < 0:
                    result = evaluate(lo, lo)
                else:
                    curvature = 2*(projected[0, 1]**2+values[0]*projected[0, 2]
                        -sum((projected[i, 1]**2+values[i]*projected[i, 2] for i in range(1, len(rows))), arb(0)))
                    mid = (lo+hi)/2
                    midpoint = arb(mid)
                    _, midpoint_value, midpoint_derivative = point(mid)
                    radius = max(abs(arb(lo)-midpoint).upper(), abs(arb(hi)-midpoint).upper())
                    taylor = midpoint_value+abs(midpoint_derivative)*radius+abs(curvature)*radius**2/2
                    # Both are upper enclosures; keep the sharper endpoint.
                    result = arb(min(upper(direct), upper(taylor)))
                cache[key] = result
                return result

            hi = float(gamma_cap)
            best = max(0., lower(evaluate(hi, hi)))
            heap = [(-upper(evaluate(0., hi)), 0., hi)]
            splits = 0
            while heap and -heap[0][0] > best+tolerance and splits < max_intervals:
                _, lo, hi = heapq.heappop(heap)
                mid = (lo+hi)/2
                if mid == lo or mid == hi:
                    heapq.heappush(heap, (-upper(evaluate(lo, hi)), lo, hi))
                    break
                best = max(best, lower(evaluate(mid, mid)))
                for left, right in [(lo, mid), (mid, hi)]:
                    heapq.heappush(heap, (-upper(evaluate(left, right)), left, right))
                splits += 1
            bound = max(0., -heap[0][0])
            all_upper.append(bound)
            details.append(dict(center=float(center), upper=bound, sample_lower=best,
                                subdivisions=splits, remaining_intervals=len(heap)))
            if symmetry:
                reflected[key] = details[-1]
            if progress and len(details) % 32 == 0:
                print('CERTIFICATE_INTERVALS', gamma_cap, len(details), len(centers),
                      'seconds', round(time.monotonic()-started, 1), flush=True)
        slack = bias_q+sum((arb(t) for t in all_upper), arb(0))
        repair = max(0., upper(slack))
        beta = upper(beta_base+arb(repair))
        return dict(status='interval_certified', beta=beta, beta_base=upper(beta_base),
                    bias_repair=repair, constraint_upper_before_repair=upper(slack),
                    delta=max(0., min(1., lower(delta))), target_witness=target_witness,
                    gamma_cap=float(gamma_cap), precision_bits=precision,
                    columns=details, seconds=time.monotonic()-started,
                    input_semantics='binary input data; real tanh; real unit witness; X=PP^T+repair*bb^T')
    finally:
        ctx.prec = previous_precision


def analytic_small_cap(x, width, gamma_cap, target):
    """Explicit mean-zero witness baseline, evaluated with Arb."""
    from flint import arb, ctx
    previous = ctx.prec
    ctx.prec = 96
    try:
        xx = [arb(float(t)) for t in x]
        yy = [arb(float(t)) for t in target]
        mean_x = sum(xx, arb(0))/len(xx)
        mean_y = sum(yy, arb(0))/len(yy)
        variance = sum(((a-mean_x)**2 for a in xx), arb(0))/len(xx)
        energy = sum((a*a for a in yy), arb(0))
        tail = sum(((a-mean_y)**2 for a in yy), arb(0))
        beta = arb(width)*arb(float(gamma_cap))**2*variance
        return dict(status='interval_certified', beta=float(np.nextafter(float(beta.upper()), np.inf)),
                    delta=max(0., float(np.nextafter(float((tail/energy).sqrt().lower()), -np.inf))),
                    gamma_cap=float(gamma_cap), kind='analytic_lipschitz', target_witness=False)
    finally:
        ctx.prec = previous
