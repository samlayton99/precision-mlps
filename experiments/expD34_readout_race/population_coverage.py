"""Retrospective FP64 population-sector audit; run numerical work under Slurm."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return float(a/b) if b > 0 else None


def basis(x):
    centered = x-np.mean(x)
    return np.column_stack((np.ones_like(x), centered/np.sqrt(np.mean(centered**2))))


def geometry(p, x, y):
    """Exact sample-metric fine force."""
    w = (len(p)-1)//3
    a, b, c = p[:-1].reshape(3, w)
    u = x[:, None]*a+b
    feature = np.tanh(u)
    derivative = 1-feature**2
    q = basis(x)
    n = len(x)
    jc = np.concatenate(((q.T@(derivative*x[:, None]))*c/n,
        (q.T@derivative)*c/n, q.T@feature/n, q.mean(axis=0)[:, None]), axis=1)
    k = jc@jc.T
    eig = np.linalg.eigvalsh(k)
    if eig[0] <= 64*np.finfo(float).eps*max(1., eig[-1]):
        raise ValueError('Unresolved coarse conditioning')
    residual = feature@c+p[-1]-y
    ec = q.T@residual/n
    eh = residual-q@ec
    weighted = derivative*eh[:, None]
    raw = np.r_[c*(x@weighted)/n, c*weighted.mean(axis=0), feature.T@eh/n, eh.mean()]
    ell = np.linalg.solve(k, jc@raw)
    force = raw-jc.T@ell
    z = ec+ell
    tracking = jc.T@z
    return dict(J=jc, K=k, raw=raw, ell=ell, F=force, R=tracking,
                z=z, kappa=eig[0], residual=residual)


def canonical(p):
    w = (len(p)-1)//3
    a, b, c = np.sqrt(w)*p[:-1].reshape(3, w)
    output_sign = 1. if np.mean(a*c) >= 0 else -1.
    sign = np.where(a != 0, np.sign(a), np.where(c != 0, np.sign(c*output_sign), 1.))
    return a*sign, b*sign, c*sign*output_sign, output_sign


def project_cone(alpha, zeta):
    """Euclidean projection onto 0 <= alpha <= zeta."""
    pooled = np.maximum((alpha+zeta)/2, 0.)
    inside = zeta >= np.maximum(alpha, 0.)
    return np.where(inside, np.maximum(alpha, 0.), pooled), np.where(inside, zeta, pooled)


def classes(p):
    a, _, c, _ = canonical(p)
    negative = c < 0
    dominance = (~negative) & (c < a)
    return dict(negative=negative, dominance=dominance, compliant=~(negative | dominance))


def structural(p, x, y, h):
    w = (len(p)-1)//3
    a, b, c, sign = canonical(p)
    q = basis(x)
    yh = sign*(y-q@(q.T@y/len(x)))
    phi3 = x**3-q@(q.T@(x**3)/len(x))
    coarse = float(np.mean(a*c))
    m3 = -float(np.mean(yh*x**3))
    mu = float(np.mean(yh*x**5))
    t3 = float(np.mean(phi3**2))
    z = float(np.mean(c**2)); moment = float(np.mean(c*a**3)); radius = float(a.max())
    pa, pc = project_cone(a, c)
    na, nc = project_cone(-a, -c)
    flip = (na+a)**2+(nc+c)**2 < (pa-a)**2+(pc-c)**2
    pa, pc = np.where(flip, -na, pa), np.where(flip, -nc, pc)
    displacement = np.column_stack((pa-a, -b, pc-c))
    target_mean = float(np.mean(y)); centered_y = y-target_mean
    centered_d = float(p[-1]-target_mean)
    target_norm = np.sqrt(np.mean(centered_y**2))
    symmetric = np.max(np.abs(x+x[::-1])) <= 1e-12
    odd_error = np.sqrt(np.mean((centered_y+centered_y[::-1])**2)) if symmetric else np.nan
    negative = a*c < 0
    dominance = c < a
    row = dict(width=w, coarse_p=coarse, Z=z, M3=moment, R=radius, t3=t3,
        zero_slope_count=int(np.count_nonzero(p[:w] == 0)),
        m3=m3, m3_times_p=m3*coarse, fifth_mu=mu,
        generated_margin=t3*moment-4*max(mu, 0.)*radius**2,
        initial_moment_margin=t3*coarse**3-4*max(mu, 0.)*radius**2*z,
        negative_product_fraction=float(negative.mean()),
        negative_product_mass=ratio(np.sum(np.abs((a*c)[negative])), np.sum(np.abs(a*c))),
        readout_dominance_violation_fraction=float(dominance.mean()),
        dominance_defect_rms=float(np.sqrt(np.mean(np.maximum(a-c, 0.)**2))),
        beta_rms=float(np.sqrt(np.mean(b*b))), beta_max=float(np.max(abs(b))),
        output_bias=float(p[-1]), target_mean=target_mean, centered_output_bias=centered_d,
        cone_distance_rms=float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))+centered_d**2)),
        cone_distance_max=float(max(np.max(np.linalg.norm(displacement, axis=1)), abs(centered_d))),
        target_odd_error=odd_error, target_odd_relative=ratio(odd_error, target_norm),
        cubic_cancellation_relative=ratio(abs(m3), target_norm*np.sqrt(t3)),
        grid_symmetric=symmetric, h=h, output_sign=sign)
    for quantile, label in ((.5, 'median'), (.9, 'p90'), (.99, 'p99'), (1., 'max')):
        row['lambda_'+label] = float(h*np.quantile(a/np.sqrt(w), quantile))
    # Eligibility cannot be inferred from rounded small cancellations.
    row['exact_bias_zero'] = bool(np.all(b == 0) and centered_d == 0)
    row['cone_exact'] = bool(np.all(c >= a))
    row['cubic_sign_eligible'] = bool(m3*coarse < 0)
    return row


def attribution(p, state, masks, initial_p, h):
    w = (len(p)-1)//3
    alpha, _, zeta, output_sign = canonical(p)
    # At zero slope use the actual velocity direction, so the sum equals the
    # right derivative of |a|. Fine/tracking share that direction, rather than
    # each taking an independent absolute value.
    radial = np.where(p[:w] != 0, np.sign(p[:w]), -np.sign((state['F']+state['R'])[:w]))
    fine = -radial*state['F'][:w]*h
    tracking = -radial*state['R'][:w]*h
    total = fine+tracking
    upper = abs(p[:w]) >= np.quantile(abs(p[:w]), .9)
    initial_upper = abs(initial_p[:w]) >= np.quantile(abs(initial_p[:w]), .9)
    compensation = np.zeros_like(state['ell'])
    rows = []
    for name, mask in masks.items():
        columns = np.r_[mask, mask, mask, False]
        ell = np.linalg.solve(state['K'], state['J'][:, columns]@state['raw'][columns])
        compensation += ell
        # Contribution to radial velocity on ALL particles, retaining the full Gram.
        velocity = radial*(state['J'][:, :w].T@ell)*h
        direct = -radial*state['raw'][:w]*h*mask
        row = dict(group=name, fraction=float(mask.mean()), ell0=float(ell[0]), ell1=float(ell[1]),
            output_sign=output_sign,
            moment_p=float(np.sum((alpha*zeta)[mask])/w),
            moment_M3=float(np.sum((zeta*alpha**3)[mask])/w),
            moment_M5=float(np.sum((zeta*alpha**5)[mask])/w),
            compensation_slope_norm=float(np.linalg.norm(velocity)),
            compensation_signed_mean=float(np.mean(velocity)),
            compensation_upper_signed_mean=float(np.mean(velocity[upper])),
            compensation_initial_upper_signed_mean=float(np.mean(velocity[initial_upper])))
        for label, receiver in (('all', np.ones(w, dtype=bool)), ('upper', upper), ('initial_upper', initial_upper)):
            for component, values in (('compensation', velocity), ('direct_raw', direct)):
                row[component+'_'+label+'_signed_sum_over_width'] = float(np.sum(values[receiver])/w)
                row[component+'_'+label+'_absolute_sum_over_width'] = float(np.sum(abs(values[receiver]))/w)
                row[component+'_'+label+'_outward_sum_over_width'] = float(np.sum(np.maximum(values[receiver], 0))/w)
        for receiver, receiver_mask in masks.items():
            row['compensation_to_'+receiver+'_signed_sum_over_width'] = float(np.sum(velocity[receiver_mask])/w)
        for label, value in (('fine', fine), ('tracking', tracking), ('total', total)):
            row[label+'_signed_sum_over_width'] = float(np.sum(value[mask])/w)
            row[label+'_outward_sum_over_width'] = float(np.sum(np.maximum(value[mask], 0))/w)
        rows.append(row)
    bias_ell = np.linalg.solve(state['K'], state['J'][:, -1]*state['raw'][-1])
    reconstruction = float(np.linalg.norm(compensation+bias_ell-state['ell']))
    for row in rows:
        row['compensation_reconstruction_error'] = reconstruction
        row['bias_compensation_norm'] = float(np.linalg.norm(bias_ell))
    return rows


def force_metrics(p, x, y, exact):
    result = dict(coarse_kappa=float(exact['kappa']), tracking_z=float(np.linalg.norm(exact['z'])),
        fine_norm=float(np.linalg.norm(exact['F'])), tracking_norm=float(np.linalg.norm(exact['R'])),
        coarse_orthogonality_error=float(np.linalg.norm(exact['J']@exact['F'])))
    return result


def load(path):
    with np.load(path) as data:
        pack = {key: data[key].copy() for key in data.files}
    cases = json.loads(str(pack['cases']))
    return pack, cases


def spacing(pack, case, index):
    if 'h' in pack:
        return float(np.asarray(pack['h']).reshape(-1)[index if np.size(pack['h']) > 1 else 0])
    if 'h' in case:
        return float(case['h'])
    if 'nref' in case:
        return 2/float(case['nref'])
    return np.nan  # No substitution of physical width for construction spacing.


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(clean(rows))


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sources, missing, duplicates = {}, [], []
    static, paths, grouped, seen = [], [], [], {}
    evidence = args.base/'evidence'
    inputs = [evidence/'inputs'/f'{cohort}.npz' for cohort in ('development', 'confirmation')]
    inputs += [evidence/'width_inputs'/f'N{n}_fork20000.npz' for n in (128, 512, 1024)]
    inputs += [evidence/'polynomial_confirmation/fork_inputs'/f'N{n}.npz' for n in (128, 512, 1024)]
    inputs += list(args.extra_input)
    for source in inputs:
        if not source.exists():
            missing.append(str(source)); continue
        sources[str(source)] = digest(source)
        pack, cases = load(source)
        for i, case in enumerate(cases):
            if source in args.extra_input and args.extra_nref is not None and 'h' not in pack and 'h' not in case and 'nref' not in case:
                case = dict(case, nref=args.extra_nref, spacing_source='explicit_extra_nref_from_archive_creator')
            p, x, y = pack['p'][i], pack['x'], pack['y'][i]
            h = spacing(pack, case, i)
            key = hashlib.sha256(p.tobytes()+x.tobytes()+y.tobytes()+np.asarray(h, dtype=np.float64).tobytes()).hexdigest()
            if key in seen:
                duplicates.append(dict(source=str(source), index=i, h=h, duplicate_of=seen[key])); continue
            seen[key] = f'{source}:{i}'
            row = dict(case, source=str(source), index=i, state_sha256=key)
            row.update(structural(p, x, y, h))
            try:
                exact = geometry(p, x, y)
                row.update(force_metrics(p, x, y, exact))
            except ValueError as error:
                row['force_status'] = str(error)
            static.append(row)
        print(json.dumps(dict(source=str(source), static_rows=len(static))), flush=True)
        write_csv(args.output/'static.csv', static)
    natural_root = evidence/'persistence_1bf7138'
    for prediction in sorted(natural_root.glob('feedback_*')):
        source = prediction/'inputs.npz'
        run = prediction.with_name(prediction.name+'_run')
        if not source.exists() or not run.exists():
            continue
        prediction_manifest = prediction/'manifest.json'
        run_manifest = run/'manifest.json'
        issued = json.loads(prediction_manifest.read_text())
        if issued['input_sha256'] != digest(source):
            raise ValueError(f'Changed issued input: {source}')
        completed = json.loads(run_manifest.read_text())
        if completed['prediction_sha256'] != digest(prediction_manifest):
            raise ValueError(f'Run refers to different predictions: {run}')
        sources[str(prediction_manifest)] = digest(prediction_manifest)
        sources[str(run_manifest)] = digest(run_manifest)
        sources[str(source)] = digest(source)
        pack, cases = load(source)
        snapshots = sorted((run/'snapshots').glob('*.npz'))
        previous, positive = {}, {}
        for snapshot in snapshots:
            sources[str(snapshot)] = digest(snapshot)
            with np.load(snapshot) as data:
                pp = data['p'].copy()
                counters = {key: data[key].copy() for key in ('positive', 'negative', 'signed_effective', 'signed_tracking', 'crossing', 'first_hit') if key in data}
                counts = data['count'].copy() if 'count' in data else np.full(len(cases), int(snapshot.stem))
                failed = data['failed'].copy() if 'failed' in data else np.zeros(len(cases), dtype=bool)
            for i, case in enumerate(cases):
                if case.get('arm') != 'natural':
                    continue
                p, p0, x, y = pp[i], pack['p'][i], pack['x'], pack['y'][i]
                meta = dict(case, panel=prediction.name, index=i, horizon=int(snapshot.stem), actual_updates=int(counts[i]), failed=bool(failed[i]))
                if failed[i] or not np.all(np.isfinite(p)):
                    paths.append(dict(meta, status='failed')); continue
                w = (len(p)-1)//3; h = spacing(pack, case, i)
                row = dict(meta, **structural(p, x, y, h))
                masks = classes(p0)
                old = previous.get(i, p0[:w])
                positive[i] = positive.get(i, np.zeros(w))+h*np.maximum(abs(p[:w])-abs(old), 0)
                previous[i] = p[:w].copy()
                row['sampled_positive_lambda_mean'] = float(np.mean(positive[i]))
                for key, values in counters.items():
                    row['all_step_'+key+'_mean'] = float(np.mean(values[i])) if key != 'first_hit' else float(np.mean(values[i] >= 0))
                if all(key in counters for key in ('positive', 'negative', 'signed_effective', 'signed_tracking', 'crossing')):
                    gap = counters['positive'][i]-counters['negative'][i]-counters['signed_effective'][i]-counters['signed_tracking'][i]-counters['crossing'][i]
                    row['all_step_identity_max_error'] = float(np.max(abs(gap)))
                    endpoint_gap = counters['positive'][i]-counters['negative'][i]-h*(abs(p[:w])-abs(p0[:w]))
                    row['all_step_endpoint_identity_max_error'] = float(np.max(abs(endpoint_gap)))
                try:
                    exact = geometry(p, x, y)
                    row.update(force_metrics(p, x, y, exact))
                    for group in attribution(p, exact, masks, p0, h):
                        mask = masks[group['group']]
                        group['endpoint_lambda_sum_over_width'] = float(h*np.sum((abs(p[:w])-abs(p0[:w]))[mask])/w)
                        group['sampled_positive_lambda_sum_over_width'] = float(np.sum(positive[i][mask])/w)
                        for key, values in counters.items():
                            group['all_step_'+key+'_sum_over_width'] = float(np.sum(values[i][mask])/w) if key != 'first_hit' else float(np.sum(values[i][mask] >= 0)/w)
                        grouped.append(dict(meta, **group))
                except ValueError as error:
                    row['force_status'] = str(error)
                paths.append(row)
        print(json.dumps(dict(panel=prediction.name, path_rows=len(paths))), flush=True)
        write_csv(args.output/'paths.csv', paths)
        write_csv(args.output/'classes.csv', grouped)
    write_csv(args.output/'static.csv', static)
    write_csv(args.output/'paths.csv', paths)
    write_csv(args.output/'classes.csv', grouped)
    manifest = dict(source_sha256=digest(__file__), sources=sources, missing_inputs=missing,
        duplicates=duplicates, static_rows=len(static), path_rows=len(paths), class_rows=len(grouped),
        missing_spacing_states=sum(not np.isfinite(row['h']) for row in static),
        feedback_base=str(args.feedback_base), role='Retrospective descriptive audit; no certificates',
        extra_nref_override=args.extra_nref,
        extra_nref_reason='Explicit archive-creator metadata: effective_feedback.py prepare and effective_feedback_holdout.py prepare initialize nref=128; only extras missing h and nref use the supplied value',
        normalization='lambda=h*abs(a); missing construction h remains null',
        cancellation_policy='Numeric smallness is not exact cancellation',
        zero_slope_policy='Instantaneous radial attribution uses actual full velocity direction at a=0, shared by fine/tracking channels',
        sampling='sampled_positive fields use saved states only; all_step fields use archived per-update counters')
    (args.output/'manifest.json').write_text(json.dumps(clean(manifest), indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--feedback-base', type=Path, required=True)
    parser.add_argument('--extra-input', type=Path, action='append', default=[])
    parser.add_argument('--extra-nref', type=int, help='Verified construction nref for extra archives missing spacing metadata')
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
