"""Matched outward geometry pulses and checkpoint-only response forecasts.

The nullspace constraints use the actual Hessian, including residual curvature.
Frozen-map and evolving-map predictors share the derivative of the remainder
and are anchored at each pulse's measured initial gradient. Training is ordinary
GD; only initial states differ. Launch all numerical work through Slurm.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, adam_run as ar, effective_feedback as ef
from . import effective_feedback_kernel as kernel, persistence_theory as pt, transport
from .effective_feedback_predict import affine_forecast, discrete_at
from .run import write_json

AMPLITUDES = (0., .01, -.01, .005, -.005, .0025, -.0025)
HORIZONS = (0, 1, 2, 10, 100, 1000, 10000, 20000)


def forecast_family(points, gradients, derivative, eta, horizons):
    """Reuse one derivative's geometric matrix sums across all seven pulses."""
    power = np.eye(len(derivative))-eta*derivative
    total = np.eye(len(derivative))
    powers, sums = [], []
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(int(max(horizons)).bit_length()):
            powers.append(power); sums.append(total)
            total = total+power@total
            power = power@power
        result, supported = [], []
        for n in horizons:
            current = np.asarray(gradients).T.copy()
            integral = np.zeros_like(current)
            bit = 0
            while n:
                if n & 1:
                    integral += sums[bit]@current
                    current = powers[bit]@current
                n >>= 1; bit += 1
            state = np.asarray(points)-eta*integral.T
            result.append(state); supported.append(np.all(np.isfinite(state), axis=1))
    return np.swapaxes(result, 0, 1), np.asarray(supported).T


def output_curvature(p, x, weight):
    """Hessian of <weight,f>; weight is fixed, not differentiated."""
    w = (len(p)-1)//3
    a, b, c = p[:-1].reshape(3, w)
    u = x[:, None]*a+b
    h = np.tanh(u)
    exp = np.exp(-2*np.abs(u)); s = 4*exp/(1+exp)**2
    t = -2*h*s
    blocks = np.zeros((w, 3, 3))
    blocks[:, 0, 0] = c*((weight*x*x)@t)/len(x)
    blocks[:, 0, 1] = blocks[:, 1, 0] = c*((weight*x)@t)/len(x)
    blocks[:, 1, 1] = c*(weight@t)/len(x)
    blocks[:, 0, 2] = blocks[:, 2, 0] = (weight*x)@s/len(x)
    blocks[:, 1, 2] = blocks[:, 2, 1] = weight@s/len(x)
    result = np.zeros((len(p), len(p)))
    indices = np.arange(w)[:, None]+w*np.arange(3)[None, :]
    result[indices[:, :, None], indices[:, None, :]] = blocks
    return result


@jax.jit
def coarse_derivative(p, x, y, q):
    def z(point):
        context = dict(x=x, y=y, q=q, p0=p)
        return kernel.field(point, context)[1]['zC']
    return z(p), jax.jacrev(z)(p)


def local_system(p, x, y, q):
    state = pt.tensors(p, x, y)
    J = q.T@state['J']/len(x)
    JC, JH = J[:2], J[2:]
    C = JC@JC.T
    eig = np.linalg.eigvalsh(C)
    if eig[0] <= 64*np.finfo(float).eps*max(1., eig[-1]):
        raise ValueError('Unresolved coarse Gram matrix')
    B = np.linalg.solve(C, JC@JH.T)
    T = JH.T-JC.T@B
    e = q.T@state['r']/len(x)
    z, Dz = map(np.asarray, coarse_derivative(jnp.asarray(p), jnp.asarray(x),
                                             jnp.asarray(y), jnp.asarray(q)))
    Dtracking = output_curvature(p, x, q[:, :2]@z)+JC.T@Dz
    Domitted = state['hessian']-J.T@J-output_curvature(p, x, q@e)
    DR = Dtracking+Domitted
    return state | dict(JC=JC, JH=JH, T=T, e=e, z=z, Dz=Dz,
        DR=DR, Dtracking=Dtracking, Domitted=Domitted,
        frozen_derivative=T@JH+DR,
        map_derivative=state['hessian']-DR-T@JH)


def matched_direction(p, system):
    """Maximize outward linear functional on a block-scaled unit sphere."""
    w = (len(p)-1)//3
    fallback = np.sqrt(np.mean(p*p))
    if fallback == 0:
        raise ValueError('Zero state has no defined relative pulse scale')
    scales = np.empty_like(p)
    for indices in (slice(0, w), slice(w, 2*w), slice(2*w, 3*w), slice(3*w, None)):
        rms = np.sqrt(np.mean(p[indices]**2))
        scales[indices] = rms if rms > 0 else fallback
    A = np.vstack((system['JC'], system['Dz'], system['hessian'][:w]))
    scaled = A*scales[None, :]
    norms = np.linalg.norm(scaled, axis=1)
    nonzero = norms > 0
    normalized = scaled[nonzero]/norms[nonzero, None]
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    tolerance = np.finfo(float).eps*max(normalized.shape)*singular[0]
    rank = np.count_nonzero(singular > tolerance)
    outward = np.r_[np.sign(p[:w])/w, np.zeros(len(p)-w)]
    u = scales*outward
    projected = u-vt[:rank].T@(vt[:rank]@u)
    gain = np.linalg.norm(projected)
    if gain <= 64*np.finfo(float).eps*np.linalg.norm(u):
        raise ValueError('No numerically resolved outward pulse in constraint nullspace')
    v = projected/gain*np.sqrt(len(p))
    direction = scales*v
    return direction, dict(rank=int(rank), nullity=int(len(p)-rank),
        singular_min_retained=float(singular[rank-1]), rank_tolerance=float(tolerance),
        normalized_constraint_residual=float(np.linalg.norm(normalized@v)),
        outward_derivative=float(outward@direction), scaled_rms=float(np.sqrt(np.mean(v*v))),
        slope_direction_norm=float(np.linalg.norm(direction[:w])),
        remainder_response_norm=float(np.linalg.norm(system['DR']@direction)),
        tracking_response_norm=float(np.linalg.norm(system['Dtracking']@direction)),
        map_response_norm=float(np.linalg.norm(system['map_derivative']@direction)))


def reduced_response(p, x, y, system, direction, eta):
    """Identify a two-observable closure from the fork and one virtual step."""
    w = (len(p)-1)//3
    weight = direction[:w]/(direction[:w]@direction[:w])
    qrow = np.r_[weight, np.zeros(len(p)-w)]
    prow = weight@system['hessian'][:w]
    constraints = np.vstack((system['JC'], system['Dz'], qrow, prow))
    rownorm = np.linalg.norm(constraints, axis=1)
    if np.any(rownorm == 0):
        return np.full(len(HORIZONS), np.nan), dict(reduced_identified=False)
    matrix = constraints/rownorm[:, None]
    rhs = np.zeros(len(matrix)); rhs[-1] = 1/rownorm[-1]
    velocity_direction, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank != len(matrix):
        return np.full(len(HORIZONS), np.nan), dict(reduced_identified=False, reduced_rank=int(rank))
    position_direction = direction-(prow@direction)*velocity_direction
    position_direction /= qrow@position_direction
    next_H = pt.tensors(p-eta*system['g'], x, y)['hessian']
    next_prow = (weight@next_H[:w])@(np.eye(len(p))-eta*system['hessian'])
    bottom = np.array([next_prow@position_direction, next_prow@velocity_direction])
    transition = np.array([[1., -eta], bottom])
    initial = np.array([qrow@direction, prow@direction])
    values, supported = discrete_at(transition, initial, HORIZONS)
    kappa, decay = bottom[0]/eta, (1-bottom[1])/eta
    closure_row = next_prow-bottom[0]*qrow-bottom[1]*prow
    return values[:, 0], dict(reduced_identified=True, reduced_rank=int(rank),
        reduced_condition=float(singular[0]/singular[-1]),
        reduced_velocity_direction_norm=float(np.linalg.norm(velocity_direction)),
        reduced_kappa=float(kappa), reduced_decay=float(decay),
        reduced_spectral_radius=float(max(abs(np.linalg.eigvals(transition)))),
        reduced_fork_closure_operator_norm=float(np.linalg.norm(closure_row)),
        reduced_supported=supported.tolist())


def prepare(args):
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    q = transport.basis(x, args.degree)
    records, pulses, labels, forecasts, supported, directions, diagnostics = [], [], [], [], [], [], []
    reduced = []
    for ci, (p, y, case) in enumerate(zip(pp, yy, cases)):
        try:
            system = local_system(p, x, y, q)
            direction, diagnostic = matched_direction(p, system)
        except ValueError as error:
            diagnostics.append(dict(**case, status='unresolved', reason=str(error)))
            continue
        diagnostic.update(case, status='resolved', source_index=ci,
            slope_tracking=float(np.linalg.norm((system['JC'].T@system['z'])[:(len(p)-1)//3])),
            slope_effective=float(np.linalg.norm((system['T']@system['e'][2:])[:(len(p)-1)//3])))
        response, reduced_info = reduced_response(p, x, y, system, direction, args.eta)
        diagnostic.update(reduced_info)
        diagnostics.append(diagnostic)
        family = p+np.asarray(AMPLITUDES)[:, None]*direction
        gradients = np.asarray([pt.tensors(pulse, x, y)['g'] for pulse in family])
        predictions, valid = [], []
        for derivative in (system['frozen_derivative'], system['hessian']):
            prediction, good = forecast_family(family, gradients, derivative, args.eta, HORIZONS)
            predictions.append(prediction); valid.append(good)
        for ai, (amplitude, pulse) in enumerate(zip(AMPLITUDES, family)):
            records.append(dict(case, source_index=ci, amplitude=amplitude))
            pulses.append(pulse); labels.append(y); forecasts.append(np.asarray(predictions)[:, ai])
            supported.append(np.asarray(valid)[:, ai]); directions.append(direction)
            reduced.append(response)
        print(json.dumps(diagnostic), flush=True)
    if not pulses:
        write_json(args.output/'diagnostics.json', diagnostics)
        raise RuntimeError('No resolved pulse direction')
    ar.atomic_npz(args.output/'inputs.npz', p=np.asarray(pulses), x=x, y=np.asarray(labels),
        cases=np.array(json.dumps(records)), sources=np.array(json.dumps({str(args.inputs): ef.digest(args.inputs)})))
    ar.atomic_npz(args.output/'predictions.npz', p=np.asarray(forecasts),
        direction=np.asarray(directions), horizons=np.asarray(HORIZONS), supported=np.asarray(supported),
        reduced_q=np.asarray(reduced))
    write_json(args.output/'diagnostics.json', diagnostics)
    write_json(args.output/'manifest.json', dict(input_sha256=ef.digest(args.inputs),
        pulse_input_sha256=ef.digest(args.output/'inputs.npz'), source_sha256=ef.digest(__file__),
        forecasts_sha256=ef.digest(args.output/'predictions.npz'), eta=args.eta, degree=args.degree,
        amplitudes=AMPLITUDES, horizons=HORIZONS, models=['frozen_map_with_DR', 'full_Hessian'],
        issued_utc=datetime.now(timezone.utc).isoformat(),
        anchoring='Each pulsed state uses its measured initial full gradient; derivatives fixed at unpulsed fork.',
        amplitude_units='RMS of coordinate perturbation divided by its fork parameter-block RMS',
        h=1/64, nref=128, width=(pp.shape[1]-1)//3, coordinate_order='a,b,c,d', numerical_certificate=False))


def verify(args):
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    q = transport.basis(x, args.degree)
    rows = []
    for p, y, case in zip(pp, yy, cases):
        try:
            s = local_system(p, x, y, q)
            direction, info = matched_direction(p, s)
        except ValueError as error:
            rows.append(dict(case=case, error=str(error))); continue
        w = (len(p)-1)//3
        values = []
        for amplitude in AMPLITUDES[1:]:
            pulse = p+amplitude*direction
            g, residual, _, ec = af.field(jnp.asarray(pulse), jnp.asarray(x), jnp.asarray(y))
            z, _ = coarse_derivative(jnp.asarray(pulse), jnp.asarray(x), jnp.asarray(y), jnp.asarray(q))
            values.append(dict(amplitude=amplitude,
                coarse_change=float(np.linalg.norm(np.asarray(ec)-s['e'][:2])),
                z_change=float(np.linalg.norm(np.asarray(z)-s['z'])),
                slope_force_change=float(np.linalg.norm(np.asarray(g)[:w]-s['g'][:w])),
                actual_mean_lambda_kick=float(np.mean(abs(pulse[:w])-abs(p[:w]))/64)))
        rows.append(dict(case=case, direction=info, pulses=values))
    write_json(args.output, rows)


def analyze(args):
    pack = args.predictions
    pp, _, _, cases = ef.load_inputs(pack/'inputs.npz')
    with np.load(pack/'predictions.npz') as saved:
        pred = {key: saved[key] for key in saved.files}
    rows = []
    w = (pp.shape[1]-1)//3
    for hi, horizon in enumerate(pred['horizons']):
        path = args.source/'snapshots'/f'{horizon:09d}.npz'
        if not path.exists():
            continue
        with np.load(path) as snapshot:
            actual = snapshot['p']
            for base in range(0, len(cases), len(AMPLITUDES)):
                direction = pred['direction'][base, :w]
                den = direction@direction
                for offset in (1, 3, 5):
                    plus, minus = base+offset, base+offset+1
                    amplitude = cases[plus]['amplitude']
                    response = (actual[plus, :w]-actual[minus, :w])/(2*amplitude)
                    row = dict(**cases[plus], horizon=int(horizon),
                        response_projection=float(response@direction/den),
                        response_norm_ratio=float(np.linalg.norm(response)/np.sqrt(den)),
                        mean_lambda=float(np.mean(abs(actual[plus, :w]))/64),
                        pulse_lambda_contrast=float(np.mean(abs(actual[plus, :w])-abs(actual[base, :w]))/64),
                        central_lambda_response=float(np.mean(abs(actual[plus, :w])-abs(actual[minus, :w]))/(128*amplitude)),
                        initial_central_lambda_response=float(np.mean(abs(pp[plus, :w])-abs(pp[minus, :w]))/(128*amplitude)),
                        reduced_projection=float(pred['reduced_q'][plus, hi]),
                        failed=int(snapshot['failed'][plus] or snapshot['failed'][minus]))
                    for model, name in enumerate(('frozen_map', 'full_Hessian')):
                        forecast = (pred['p'][plus, model, hi, :w]-pred['p'][minus, model, hi, :w])/(2*amplitude)
                        row[name+'_projection'] = float(forecast@direction/den)
                        row[name+'_response_error'] = float(np.linalg.norm(forecast-response)/np.sqrt(den))
                        row[name+'_supported'] = bool(pred['supported'][plus, model, hi] and pred['supported'][minus, model, hi])
                    row['retained_offset_error'] = float(np.linalg.norm(direction-response)/np.sqrt(den))
                    rows.append(row)
    if not rows:
        raise ValueError('No paired snapshots to analyze')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def audit_matching(args):
    """Curate amplitude ratios as data, without inventing a pass threshold."""
    rows = json.loads(args.inputs.read_text())
    summary = []
    for row in rows:
        if 'error' in row:
            summary.append(dict(case=row['case'], unresolved=row['error']))
            continue
        positive = [p for p in row['pulses'] if p['amplitude'] > 0]
        item = dict(case=row['case'], direction=row['direction'])
        for key in ('coarse_change', 'z_change', 'slope_force_change'):
            values = [p[key] for p in positive]
            item[key] = values
            item[key+'_halving_ratios'] = [a/b if b > 0 else None for a, b in zip(values, values[1:])]
        item['mean_lambda_kicks'] = [p['actual_mean_lambda_kick'] for p in positive]
        summary.append(item)
    write_json(args.output, summary)


def summarize(args):
    """Numerical tables and a figure only; scientific prose is authored separately."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {}
    summary = {}
    for cohort in ('development', 'confirmation'):
        with (args.source/f'analysis_{cohort}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        smallest = [row for row in rows if int(row['horizon']) == 20000
                    and float(row['amplitude']) == .0025]
        if len(smallest) != 23:
            raise ValueError('Expected all23 fixed-cohort smallest-amplitude responses')
        tables[cohort] = smallest
        projection = np.array([float(row['response_projection']) for row in smallest])
        retained = np.array([float(row['retained_offset_error']) for row in smallest])
        item = dict(count=len(smallest), projection_min=float(projection.min()),
                    projection_max=float(projection.max()),
                    projection_median=float(np.median(projection)),
                    retained_error_median=float(np.median(retained)),
                    retained_error_max=float(retained.max()),
                    failed=sum(int(row['failed']) for row in smallest), models={})
        for name in ('frozen_map', 'full_Hessian'):
            error = np.array([float(row[name+'_response_error']) for row in smallest])
            item['models'][name] = dict(
                median_error=float(np.median(error)), max_error=float(error.max()),
                beats_retained=int(np.count_nonzero(error < retained)),
                supported=sum(row[name+'_supported'] == 'True' for row in smallest))
        item['reduced_projection'] = {row['target']: float(row['reduced_projection']) for row in smallest}
        summary[cohort] = item
    write_json(args.output/'summary.json', summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, layout='constrained')
    targets = [row['target'] for row in tables['development']]
    positions = np.arange(len(targets))
    for cohort, color, shift in (('development', '#126caa', -.13), ('confirmation', '#d17a16', .13)):
        rows = tables[cohort]
        axes[0].scatter(100*(np.array([float(row['response_projection']) for row in rows])-1),
                        positions+shift, color=color, s=22, label=cohort)
        for name, marker in (('retained_offset', 'o'), ('frozen_map', 'x'), ('full_Hessian', '+')):
            key = 'retained_offset_error' if name == 'retained_offset' else name+'_response_error'
            axes[1].scatter([float(row[key]) for row in rows], positions+shift, color=color,
                            marker=marker, s=25, label=name.replace('_', ' ') if cohort == 'development' else None)
    axes[0].set_yticks(positions, targets); axes[0].invert_yaxis()
    axes[0].axvline(0, color='0.7', lw=1)
    axes[0].set_xlabel('Change in retained kick projection (%)')
    axes[0].set_title('Matched outward kicks mostly persist')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[1].set_xscale('log'); axes[1].set_xlabel('Vector error / initial slope-kick norm')
    axes[1].set_title('Forecast error versus a retained offset')
    axes[1].legend(loc='lower right', fontsize=8)
    for axis in axes:
        axis.grid(axis='x', alpha=.2)
    fig.suptitle('20,000 additional GD updates; smallest predeclared pulse amplitude 0.0025')
    fig.savefig(args.output/'pulse_responses.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'analyze', 'audit_matching', 'summarize'))
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--degree', type=int, default=65)
    parser.add_argument('--eta', type=float, default=.002)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('FP64 required')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
