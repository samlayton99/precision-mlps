"""FP64 usefulness audit of the uniform width constants, not a certificate.

Evaluate predeclared outer margins and zero tracking on issued natural forks.
No future trajectory, outcome fit, or empirical maximum replaces a uniform
analytic constant. All numerical execution belongs in a Slurm CPU allocation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MARGINS = (.10, .25, .50)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def ratio(value, denominator):
    return finite(value/denominator) if denominator > 0 else None


def initial_geometry(p, x, y):
    """Analytic coarse Jacobian and full-complement force, without autodiff."""
    w = (len(p)-1)//3
    a, b, c = p[:-1].reshape(3, w)
    centered = x-x.mean()
    variance = np.mean(centered**2)
    if variance <= 0 or np.max(np.abs(x)) > 1:
        raise ValueError('The width theorem requires nondegenerate inputs in [-1,1]')
    basis = np.column_stack((np.ones_like(x), centered/np.sqrt(variance)))
    feature = np.tanh(x[:, None]*a+b)
    derivative = 1-feature**2
    jac_coarse = np.concatenate(((basis.T@(derivative*x[:, None]))*c/len(x),
        (basis.T@derivative)*c/len(x), basis.T@feature/len(x),
        basis.mean(axis=0)[:, None]), axis=1)
    gram = jac_coarse@jac_coarse.T
    eigen = np.linalg.eigvalsh(gram)
    if eigen[0] <= 64*np.finfo(float).eps*max(1., eigen[-1]):
        raise ValueError('Numerically unresolved initial coarse conditioning')
    target_fine = y-basis@(basis.T@y/len(x))
    residual = feature@c+p[-1]-y
    residual_fine = residual-basis@(basis.T@residual/len(x))
    weighted = derivative*residual_fine[:, None]
    raw = np.r_[c*(x@weighted)/len(x), c*weighted.mean(axis=0),
                feature.T@residual_fine/len(x), residual_fine.mean()]
    force = raw-jac_coarse.T@np.linalg.solve(gram, jac_coarse@raw)
    return dict(initial_A=np.sqrt(w)*np.max(np.abs(p[:-1].reshape(3, w)), axis=1),
        kappa0=float(eigen[0]), target_fine_norm=float(np.sqrt(np.mean(target_fine**2))),
        q0=float(np.linalg.norm(force)), initial_force=force,
        initial_coarse_jacobian=jac_coarse)


def constants(initial, width, eta, h, margin):
    """Equations28,29,33 with delta_a=delta_b=delta_c=0, in FP64."""
    inner = initial['initial_A']
    outer = (1+margin)*inner
    aa, ab, ac = outer
    kappa0 = initial['kappa0']; kappa = kappa0/2
    u = aa+ab
    j = np.sqrt(2*ac**2*u**4+u**6/9)
    e = initial['target_fine_norm']+ac*u**3/(3*width)
    fine_hessian = 4*ac*u+np.sqrt(2)*u*u
    coarse_hessian = np.sqrt(2)+4*ac*u/width
    cstar = e*(fine_hessian+j*coarse_hessian/np.sqrt(kappa))
    lstar = e*(fine_hessian+2*j*coarse_hessian/np.sqrt(kappa))+j*j/width
    ka = e*ac*(u*u+j/np.sqrt(kappa))
    kc = e*(u**3/3+u*j/np.sqrt(kappa))
    components = np.array([ka, ka, kc])
    g = np.linalg.norm(components)
    jc = np.sqrt(1+2*ac**2+u*u)
    limits = []
    for delta, speed in zip(outer-inner, components):
        limits.append(0. if delta <= 0 else width*delta/(eta*speed) if speed > 0 else np.inf)
    limits.append(width*(kappa0-kappa)/(2*jc*coarse_hessian*eta*g) if g > 0 else np.inf)
    limit = min(limits)
    integer = max(0, int(np.ceil(limit))-1) if np.isfinite(limit) else None
    result = dict(margin=margin, coarse_kappa0=kappa0, coarse_kappa=kappa,
        J_star=j, E_star=e, H_star=fine_hessian, M_star=coarse_hessian,
        C_star=cstar, L_star=lstar, J_C_star=jc, G=g,
        q_bound=j*e/width, q0=initial['q0'],
        q_bound_over_q0=ratio(j*e/width, initial['q0']),
        curvature_rate_bound=cstar/width, derivative_rate_bound=lstar/width,
        relaxation_rate_bound=j*j/(width*width),
        component_a_updates_limit=finite(limits[0]), component_b_updates_limit=finite(limits[1]),
        component_c_updates_limit=finite(limits[2]), conditioning_updates_limit=finite(limits[3]),
        strict_updates_limit=finite(limit), sufficient_integer_updates=integer,
        bottleneck=('a', 'b', 'c', 'conditioning')[int(np.argmin(limits))],
        normalized_rate_bound_per_update=eta*h*ka/width**1.5,
        tracking_delta_a=0., tracking_delta_b=0., tracking_delta_c=0.,
        best_case_zero_tracking=True, ordinary_gd_certificate=False,
        has_positive_initial_box_margin=bool(np.all(outer > inner)))
    for index, label in enumerate(('a', 'b', 'c')):
        result['initial_A_'+label] = finite(inner[index])
        result['outer_A_'+label] = finite(outer[index])
        result['K_'+label] = finite(components[index])
    return {key: finite(value) if isinstance(value, (float, np.floating)) else value
            for key, value in result.items()}


def audit(predictions, output, expected_cases=36):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, provenance, natural_count = [], [], 0
    for directory in map(Path, predictions):
        manifest = json.loads((directory/'manifest.json').read_text())
        with np.load(directory/'inputs.npz') as data:
            pp, x, yy, hs = (data[key].copy() for key in ('p', 'x', 'y', 'h'))
            cases = json.loads(str(data['cases']))
        with (directory/'fork_diagnostics.csv').open() as stream:
            diagnostics = {int(row['source_index']): row for row in csv.DictReader(stream)
                           if row['arm'] == 'natural'}
        provenance.append(dict(directory=str(directory),
            inputs_sha256=digest(directory/'inputs.npz'),
            manifest_sha256=digest(directory/'manifest.json'),
            diagnostics_sha256=digest(directory/'fork_diagnostics.csv')))
        eta = float(manifest['eta'])
        for index, case in enumerate(cases):
            if case['arm'] != 'natural':
                continue
            natural_count += 1
            width = (len(pp[index])-1)//3
            issued = diagnostics[int(case['source_index'])]
            actual_k = float(issued['k']) if issued['k'] else None
            pure_k = float(issued['k_pure']) if issued['k_pure'] else None
            try:
                initial = initial_geometry(pp[index], x, yy[index])
            except ValueError as error:
                for margin in MARGINS:
                    rows.append(dict(case, margin=margin, status='unresolved', reason=str(error),
                        best_case_zero_tracking=True, ordinary_gd_certificate=False))
                continue
            for margin in MARGINS:
                result = constants(initial, width, eta, float(hs[index]), margin)
                bound = result['curvature_rate_bound']
                q_issued, q_squared = float(issued['q']), float(issued['q_squared'])
                c_issued, d_issued = float(issued['C']), float(issued['D'])
                q_slack = result['q_bound']-q_issued
                c_slack = bound*q_squared-abs(c_issued)
                d_slack = result['relaxation_rate_bound']*q_squared-d_issued
                rows.append(dict(case, **result, status='evaluated', eta=eta,
                    fine_target_norm=initial['target_fine_norm'],
                    issued_q0=q_issued,
                    independently_reconstructed_q0_difference=initial['q0']-q_issued,
                    actual_C=c_issued, actual_D=d_issued,
                    pointwise_force_bound_holds=bool(q_slack >= 0),
                    pointwise_curvature_bound_holds=bool(c_slack >= 0),
                    pointwise_relaxation_bound_holds=bool(d_issued >= 0 and d_slack >= 0),
                    pointwise_force_slack=q_slack, pointwise_curvature_slack=c_slack,
                    pointwise_relaxation_slack=d_slack,
                    actual_k0=actual_k, intrinsic_k0=pure_k,
                    curvature_bound_over_abs_actual_k=ratio(bound, abs(actual_k)) if actual_k is not None else None,
                    curvature_bound_over_abs_intrinsic_k=ratio(bound, abs(pure_k)) if pure_k is not None else None))
    if natural_count != expected_cases:
        raise ValueError(f'Expected {expected_cases} natural forks, found {natural_count}')
    output.mkdir(parents=True)
    with (output/'constants.csv').open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for width in sorted({int(row['width']) for row in rows}):
        for margin in MARGINS:
            selected = [row for row in rows if int(row['width']) == width and row['margin'] == margin]
            good = [row for row in selected if row['status'] == 'evaluated']
            record = dict(width=width, margin=margin, cases=len(selected), evaluated=len(good),
                positive_integer_horizon=sum((row['sufficient_integer_updates'] or 0) > 0 for row in good),
                pointwise_force_failures=sum(not row['pointwise_force_bound_holds'] for row in good),
                pointwise_curvature_failures=sum(not row['pointwise_curvature_bound_holds'] for row in good),
                pointwise_relaxation_failures=sum(not row['pointwise_relaxation_bound_holds'] for row in good),
                bottlenecks={key: sum(row['bottleneck'] == key for row in good)
                             for key in ('a', 'b', 'c', 'conditioning')})
            for key in ('sufficient_integer_updates', 'strict_updates_limit', 'q_bound_over_q0',
                        'curvature_bound_over_abs_actual_k', 'curvature_bound_over_abs_intrinsic_k',
                        'normalized_rate_bound_per_update'):
                values = [row[key] for row in good if row[key] is not None]
                record[key] = dict(minimum=finite(min(values)) if values else None,
                    median=finite(np.median(values)) if values else None,
                    maximum=finite(max(values)) if values else None)
            groups.append(record)
    result = dict(source_sha256=digest(__file__), inputs=provenance, natural_cases=natural_count,
        rows=len(rows), margins=MARGINS, arithmetic='FP64, without rounding control',
        scope='Best-case zero-tracking evaluation of Section7 constants; not an ordinary-GD certificate.',
        conditioning='kappa=half the initial analytic coarse-Gram minimum eigenvalue',
        pointwise_checks='Strict FP64 comparisons at the fork check implementation only, not uniform validity.',
        horizon='ceil(minimum strict Eq33 limit)-1, clipped at zero',
        outcome_usage='None; only checkpoint inputs and preissued fork diagnostics are read.',
        groups=groups)
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-cases', type=int, default=36)
    args = parser.parse_args()
    audit(args.predictions, args.output, args.expected_cases)


if __name__ == '__main__':
    main()
