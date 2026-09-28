"""Exact population/output diagnostics; numerical execution belongs in Slurm.

The archive audit is retrospective. Algebraic bounds are distinct from sampled
quadrature estimates and from prospective exact-flow certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af
from . import population_coverage as pc
from . import mechanism_widths as widths
from .mechanism_persistence import validate_backend
from .population_sensitivity_certificate import initial_sensitivity_certificate


def moment_values(p):
    a, b, c = p[:-1].reshape(3, -1)
    w = len(a)
    square = a*a+b*b+c*c
    m = jnp.sum(square)
    h = w*w*jnp.sum(square**3)
    q = jnp.sum(jnp.abs(c)*a*a*(jnp.abs(b)+jnp.abs(a)/3))
    return jnp.array([m, h, h/m**3, jnp.sum(a*a+c*c), q])


def projected_output(p, x):
    a, b, c = p[:-1].reshape(3, -1)
    f = jnp.tanh(x[:, None]*a+b)@c+p[-1]
    q = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))), axis=1)
    return f-q@(q.T@f/len(x))


def observables(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    w = len(a)
    g, r, jc, ec = af.field(p, x, y)
    split, info = af.split(g, jc, ec)
    force, tracking = split[:2]
    q = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))), axis=1)
    eh = r-q@ec
    yh = y-q@(q.T@y/len(x))
    fh = eh+yh
    u = x[:, None]*a+b
    exp = jnp.exp(-2*jnp.abs(u))
    derivative = 4*exp/(1+exp)**2
    hs_squared = jnp.array(0., dtype=p.dtype)
    generated_blocks = []
    for columns in (derivative*c*x[:, None], derivative*c, jnp.tanh(u)):
        remainder = columns-q@(q.T@columns/len(x))
        hs_squared += jnp.sum(jnp.mean(remainder**2, axis=0))
        generated_blocks.append(remainder.T@fh/len(x))
    generated_raw = jnp.concatenate((*generated_blocks, jnp.zeros(1, dtype=p.dtype)))
    safe_gram = jnp.where(info['resolved'], jc@jc.T, jnp.eye(2, dtype=p.dtype))
    generated_force = generated_raw-jc.T@jnp.linalg.solve(safe_gram, jc@generated_raw)
    target_force = force-generated_force
    target = jnp.sqrt(jnp.mean(y*y))
    fine = jnp.sqrt(jnp.mean(eh*eh))
    m, h, concentration, es, capacity = moment_values(p)
    bnorm = (h/(w*w))**(1/6)
    square = a*a+b*b+c*c
    moment4 = w*jnp.sum(square**2)
    row = dict(width=jnp.array(w), M=m, M6=h, C6=concentration, Es=es,
               Q=capacity, B=bnorm, M4=moment4, target_norm=target,
               relative_l2=jnp.linalg.norm(r)/jnp.linalg.norm(y),
               fine_norm=fine, target_fine_norm=jnp.sqrt(jnp.mean(yh*yh)),
               output_fine_norm=jnp.sqrt(jnp.mean(fh*fh)),
               Q_error_floor=jnp.maximum(jnp.sqrt(jnp.mean(yh*yh))-capacity, 0)/target,
               isotropic_capacity=(2*jnp.sqrt(2.)/3)*moment4/w,
               sensitivity_bound=9*h/(w*w),
               fine_jacobian_hs_norm=jnp.sqrt(hs_squared),
               residual_sensitivity=jnp.sum(force*force)/jnp.maximum(fine*fine, 1e-300),
               F_norm=jnp.linalg.norm(force), R_norm=jnp.linalg.norm(tracking),
               coarse_kappa=info['min_eigenvalue'], resolved=info['resolved'],
               coarse_singular_lower=jnp.sqrt(jnp.mean(x*x)*es/(1+2*m))-3*jnp.sqrt(h)/w,
               mean_gamma=jnp.mean(jnp.abs(a)), rms_gamma=jnp.sqrt(jnp.mean(a*a)),
               max_gamma=jnp.max(jnp.abs(a)),
               readout_rms=jnp.sqrt(jnp.mean(c*c)),
               loss=jnp.mean(r*r)/2, tracking_z_norm=jnp.linalg.norm(info['z']),
               activation_rms=jnp.sqrt(jnp.mean((x[:, None]*a+b)**2)),
               force_identity=jnp.linalg.norm(g-split.sum(axis=0)))
    particle_capacity = jnp.abs(c)*a*a*(jnp.abs(b)+jnp.abs(a)/3)
    top = max(1, int(np.ceil(.1*w)))
    row['capacity_top_decile_share'] = jnp.sum(jnp.sort(particle_capacity)[-top:])/jnp.maximum(capacity, 1e-300)
    for order in (8, 12):
        row[f'M{order}'] = w**(order/2-1)*jnp.sum(square**(order/2))
    # Differentiate the current exact observables along each current velocity.
    velocities = (-force, -tracking, -g, -info['fine'], -info['balanced'], -generated_force, -target_force)
    for name, velocity in zip(('effective', 'tracking', 'full', 'direct', 'compensation', 'generated', 'target'), velocities):
        values, derivative = jax.jvp(moment_values, (p,), (velocity,))
        for key, value in zip(('M', 'M6', 'C6', 'Es', 'Q'), derivative):
            row[key+'_dot_'+name] = value
        _, output_velocity = jax.jvp(projected_output, (p, x), (velocity, jnp.zeros_like(x)))
        row['output_speed_'+name] = jnp.sqrt(jnp.mean(output_velocity**2))
        row['fine_loss_dot_'+name] = jnp.mean(eh*output_velocity)
        va, vb, vc = velocity[:-1].reshape(3, w)
        for order in (8, 12):
            row[f'M{order}_dot_'+name] = order*w**(order/2-1)*jnp.sum(
                square**(order/2-1)*(a*va+b*vb+c*vc))
        for degree in (3, 5):
            for power in range(2, degree+1):
                bp = degree-power
                raw = jnp.sum(c*a**power*b**bp)
                deriv = jnp.sum(vc*a**power*b**bp+power*c*a**(power-1)*b**bp*va)
                if bp:
                    deriv += jnp.sum(bp*c*a**power*b**(bp-1)*vb)
                label = f'mixed_a{power}b{bp}c'
                row[label] = raw
                row[label+'_dot_'+name] = deriv
    row['energy_identity'] = jnp.abs(row['fine_loss_dot_effective']+jnp.sum(force*force))
    return row


def initial_certificate(row):
    """Initial-data effective-flow enclosure, never a discrete-GD certificate."""
    B, Y = row['B'], row['fine_norm']
    if not all(np.isfinite(row[k]) for k in ('B', 'fine_norm', 'M', 'Es', 'coarse_kappa', 'target_norm', 'M8', 'M12')):
        return dict(certificate_status='nonfinite_initial_diagnostic')
    P = np.sqrt(row['M']); es = np.sqrt(row['Es'])
    initial_sigma = np.sqrt(max(0., row['coarse_kappa']))
    if not row['resolved'] or B <= 0 or Y <= 0:
        return dict(certificate_status='degenerate_or_unresolved')
    variance = row['input_variance']
    def rank_margin(factor):
        travel = B*(factor-1)
        moment_margin = np.sqrt(variance*max(es-travel, 0)**2/(1+2*(P+travel)**2))-3*(factor*B)**3
        initial_margin = initial_sigma-(np.sqrt(2)+4*(P+travel))*travel
        return max(moment_margin, initial_margin)
    if rank_margin(1.) <= 0:
        return dict(certificate_status='initial_collective_rank_bound_nonpositive')
    low, high = 1., 2.
    for _ in range(60):
        middle = (low+high)/2
        if rank_margin(middle) > 0:
            low = middle
        else:
            high = middle
    factor = 1+.9*(low-1)  # strict interior, no equality certificate
    rate = 6*Y*B*B
    time = (1-factor**-2)/rate
    exponent = 3*B**4/(4*Y)*(factor**4-1)
    result = dict(certificate_status='effective_flow_initial_data_fp64_audit',
                certificate_factor=factor, certificate_flow_time=time,
                certificate_eta002_time_units=time/.002,
                certificate_relative_fine_floor=np.exp(-exponent),
                certificate_absolute_relative_floor=Y*np.exp(-exponent)/row['target_norm'],
                certificate_path=B*(factor-1), certificate_rank_margin=rank_margin(factor),
                B_blowup_comparison_time=1/rate)
    for order in (8, 12):
        width = row['width']
        ell = row[f'M{order}']**(1/order)*width**(1/order-.5)
        D = width**(.5-3/order)
        baseline = max(np.sqrt(variance*row['Es']/(1+2*row['M']))-3*D*ell**3, initial_sigma)
        prefix = f'p{order}_'
        if baseline <= 0:
            result[prefix+'status'] = 'initial_collective_rank_bound_nonpositive'
            continue
        sigma = baseline/2
        def constants(q):
            c = 1+np.sqrt(2)*D*q*ell/sigma
            k = 6*Y*c*ell**2
            A = D*(q-1)*ell/c
            moment_margin = np.sqrt(variance*max(es-A, 0)**2/(1+2*(P+A)**2))-3*D*(q*ell)**3
            initial_margin = initial_sigma-(np.sqrt(2)+4*(P+A))*A
            margin = max(moment_margin, initial_margin)-sigma
            return c, k, A, margin
        low, high = 1., 2.
        for _ in range(60):
            mid = (low+high)/2
            if constants(mid)[-1] > 0:
                low = mid
            else:
                high = mid
        q = 1+.9*(low-1)
        c, k, A, margin = constants(q)
        exponent = 9*D**2*ell**6*(q**4-1)/(2*k)
        result.update({prefix+'status': 'effective_flow_initial_data_fp64_audit',
                       prefix+'flow_time': (1-q**-2)/k,
                       prefix+'path': A, prefix+'factor': q,
                       prefix+'relative_fine_floor': np.exp(-exponent),
                       prefix+'rank_margin': margin, prefix+'sigma': sigma})
    return result


def empirical_basis(x, degree=9):
    q, r = np.linalg.qr(np.polynomial.polynomial.polyvander(x, degree))
    return q*np.sign(np.diag(r))[None, :]*np.sqrt(len(x))


def tail_bounds(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    f = np.tanh(x[:, None]*a+b)@c+p[-1]
    q = empirical_basis(x)
    row = {}
    # Rigorous derivative upper bounds: tanh' = 1-tanh², recursively polynomial.
    polynomial = np.polynomial.Polynomial([0., 1.])
    derivative_bounds = {}
    for k in range(1, 10):
        polynomial = polynomial.deriv()*np.polynomial.Polynomial([1., 0., -1.])
        derivative_bounds[k] = float(np.sum(abs(polynomial.coef)))
    import math
    for k in (2, 3, 5, 9):
        yh = y-q[:, :k]@(q[:, :k].T@y/len(x))
        fh = f-q[:, :k]@(q[:, :k].T@f/len(x))
        cap = derivative_bounds[k]/math.factorial(k)*np.sum(abs(c)*abs(a)**k)
        row.update({f'tail{k}_target': np.linalg.norm(yh)/np.sqrt(len(x)),
                    f'tail{k}_output': np.linalg.norm(fh)/np.sqrt(len(x)),
                    f'tail{k}_capacity': cap,
                    f'tail{k}_relative_floor': max(0., np.linalg.norm(yh)/np.sqrt(len(x))-cap)/(np.linalg.norm(y)/np.sqrt(len(x)))})
    return row


@lru_cache(maxsize=None)
def evaluation_data(target):
    x, y, _, _ = widths.data(target, 8192)
    return x, y, empirical_basis(x)


def audit(args):
    if not jax.config.x64_enabled:
        raise ValueError('Set JAX_ENABLE_X64=true')
    validate_backend(args.backend)
    args.output.mkdir(parents=True, exist_ok=False)
    measure = jax.jit(observables)
    rows, sources, duplicates, seen = [], {}, [], set()

    def add(p, x, y, case, source, index, horizon=0, **extra):
        key = hashlib.sha256(p.tobytes()+x.tobytes()+y.tobytes()).hexdigest()
        identity = ((key, 'static') if extra.get('role') == 'static' else
                    (key, extra.get('role'), extra.get('panel'), horizon, index))
        if identity in seen:
            duplicates.append(dict(source=str(source), index=index)); return
        seen.add(identity)
        meta = {k: v for k, v in case.items() if isinstance(v, (str, int, float, bool))}
        row = dict(meta, source=str(source), index=index, horizon=horizon, state_sha256=key, **extra)
        if extra.get('failed', False) or not np.all(np.isfinite(p)):
            rows.append(dict(row, status='failed_or_nonfinite_state')); return
        if abs(np.mean(x)) > 1e-13 or np.max(abs(x)) > 1+1e-14:
            raise ValueError('Expected centered empirical grid in [-1,1]')
        values = {k: np.asarray(v).item() for k, v in measure(p, x, y).items()}
        if not values['resolved']:
            for key in values:
                if any(label in key for label in ('_effective', '_tracking', '_compensation', '_generated', '_target')) or key in ('F_norm', 'R_norm', 'residual_sensitivity', 'energy_identity'):
                    values[key] = np.nan
        row.update(values, status='finite', input_variance=float(np.mean(x*x)))
        row.update(initial_certificate(row), **tail_bounds(p, x, y))
        row.update(initial_sensitivity_certificate(row))
        if 'target' in case:
            xe, ye, polynomial_basis = evaluation_data(case['target'])
            a, b, c = p[:-1].reshape(3, -1)
            fe = np.tanh(xe[:, None]*a+b)@c+p[-1]
            qe = np.column_stack((np.ones_like(xe), xe/np.sqrt(np.mean(xe*xe))))
            yhe = ye-qe@(qe.T@ye/len(xe))
            norm_e = np.linalg.norm(ye)/np.sqrt(len(xe))
            row.update(relative_eval_l2=float(np.linalg.norm(fe-ye)/np.linalg.norm(ye)),
                       eval_target_fine_norm=float(np.linalg.norm(yhe)/np.sqrt(len(xe))),
                       eval_Q_error_floor=max(0., np.linalg.norm(yhe)/np.sqrt(len(xe))-row['Q'])/norm_e,
                       evaluation_role='8192 midpoints; original target normalization, no checkpoint selection')
            for degree in (2, 3, 5, 9):
                basis = polynomial_basis[:, :degree]
                tail = ye-basis@(basis.T@ye/len(xe))
                tail_norm = np.linalg.norm(tail)/np.sqrt(len(xe))
                row[f'eval_tail{degree}_target'] = float(tail_norm)
                row[f'eval_tail{degree}_relative_floor'] = max(
                    0., tail_norm-row[f'tail{degree}_capacity'])/norm_e
        rows.append(row)

    e = args.base/'evidence'
    static = [e/'inputs'/f'{c}.npz' for c in ('development', 'confirmation')]
    static += [e/'width_inputs'/f'N{n}_fork20000.npz' for n in (128, 512, 1024)]
    static += [e/'polynomial_confirmation/fork_inputs'/f'N{n}.npz' for n in (128, 512, 1024)]
    static += list((e/'population_coverage_inputs').glob('*.npz'))
    static += args.extra_input
    if args.only_trajectories:
        static = []
    for path in static:
        if not path.exists():
            continue
        pack, cases = pc.load(path); sources[str(path)] = pc.digest(path)
        for i, case in enumerate(cases):
            add(pack['p'][i], pack['x'], pack['y'][i], case, path, i, role='static', panel=path.stem)
        pc.write_csv(args.output/'states.csv', rows)
        print(json.dumps(dict(source=str(path), rows=len(rows))), flush=True)

    runs = []
    if args.natural:
        for pred in sorted((e/'persistence_1bf7138').glob('feedback_*')):
            if (pred/'inputs.npz').exists() and pred.with_name(pred.name+'_run').exists():
                runs.append((pred.with_name(pred.name+'_run'), pred/'inputs.npz', True))
    for root in args.dilation_root:
        for manifest_path in sorted(root.glob('*/manifest.json')):
            manifest = json.loads(manifest_path.read_text())
            if 'input' in manifest and 'cases' in manifest:
                runs.append((manifest_path.parent, Path(manifest['input']), False))
    for run, source, natural in runs:
        pack, original_cases = pc.load(source)
        manifest = json.loads((run/'manifest.json').read_text())
        sources[str(run/'manifest.json')] = pc.digest(run/'manifest.json')
        cases = original_cases if natural else manifest['cases']
        snapshots = sorted((run/'snapshots' if natural else run).glob('[0-9]*.npz'))
        sources[str(source)] = pc.digest(source)
        for snapshot in snapshots:
            if args.snapshot_steps and int(snapshot.stem) not in args.snapshot_steps:
                continue
            sources[str(snapshot)] = pc.digest(snapshot)
            with np.load(snapshot) as data:
                pp = data['p'].copy()
                failed = data['failed'].copy() if 'failed' in data else np.zeros(len(pp), dtype=bool)
                counts = data['count'].copy() if 'count' in data else np.full(len(pp), int(snapshot.stem))
            for i, case in enumerate(cases):
                if natural and case.get('arm') != 'natural':
                    continue
                index = i if natural else case['input_index']
                add(pp[i], pack['x'], pack['y'][index], case, snapshot, i,
                    horizon=int(snapshot.stem), role='trajectory', panel=run.name,
                    eta=float(manifest.get('eta', .002)), failed=bool(failed[i]), actual_updates=int(counts[i]))
        pc.write_csv(args.output/'states.csv', rows)
        print(json.dumps(dict(run=str(run), rows=len(rows))), flush=True)
    summary = dict(source_sha256=pc.digest(__file__), sources=sources, rows=len(rows),
                   duplicates=duplicates, role='retrospective exact-state audit; flow certificate is separate from GD',
                   backend=jax.default_backend())
    (args.output/'manifest.json').write_text(json.dumps(pc.clean(summary), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dilation-root', type=Path, action='append', default=[])
    parser.add_argument('--natural', action='store_true')
    parser.add_argument('--only-trajectories', action='store_true')
    parser.add_argument('--extra-input', type=Path, action='append', default=[])
    parser.add_argument('--snapshot-steps', type=int, nargs='+')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    audit(parser.parse_args())
