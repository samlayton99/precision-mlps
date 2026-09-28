"""Exact neuron splitting through its replica-symmetry quotient.

The quotient trains original hidden coordinates and aggregated readouts. It is
identical to an expanded cloned network under simultaneous GD, not a width
approximation. Independent-width experiments are deliberately separate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, adam_run as ar, effective_feedback as ef, transport
from .run import write_json

POLICIES = ('none', 'hidden', 'readout', 'full')
ARMS = ('original', *(f'k{k}_{p}' for k in (2, 4) for p in POLICIES))
LAMBDA_THRESHOLDS = (.05, .25, 1.)


def settings(arm):
    if arm == 'original':
        return 1, 1., 1.
    if arm not in ARMS:
        raise ValueError(arm)
    k, policy = arm.split('_')
    k = int(k[1:])
    hidden = float(k) if policy in ('hidden', 'full') else 1.
    readout = 1./k if policy in ('readout', 'full') else 1.
    return k, hidden, readout


def mobility(width, arm):
    k, hidden, readout = settings(arm)
    return np.r_[np.full(2*width, hidden/k), np.full(width, k*readout), 1.]


def expand(p, k):
    z, d = af.unpack(p)
    return jnp.r_[jnp.repeat(z[0], k), jnp.repeat(z[1], k),
                  jnp.repeat(z[2]/k, k), d]


def collapse(p, k):
    z, d = af.unpack(p)
    return jnp.r_[z[0].reshape(-1, k)[:, 0], z[1].reshape(-1, k)[:, 0],
                  z[2].reshape(-1, k).sum(axis=1), d]


def force(p, x, y, mass):
    g, r, jc, ec = af.field(p, x, y)
    channels, info = af.split(g, jc, ec, mass)
    return mass*g, channels*mass[None, :], info, r


def fine_residual_forcing(p, x, channels):
    """Norms of full-complement J v; channels already include mobility."""
    (a, b, c), _ = af.unpack(p)
    u = x[:, None]*a+b
    exponential = jnp.exp(-2*jnp.abs(u)); s = 4*exponential/(1+exponential)**2
    h = jnp.tanh(u)
    w = len(a)
    q = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))), axis=1)
    def one(v):
        value = s@(c*v[w:2*w]) + x*(s@(c*v[:w])) + h@v[2*w:3*w]+v[-1]
        fine = value-q@(q.T@value/len(x))
        return jnp.sqrt(jnp.mean(fine*fine))
    norms = jax.vmap(one)(channels)
    return norms, jnp.where(norms[0] > 0, norms[1]/norms[0], jnp.nan)


def initial(p, h):
    w = (len(p)-1)//3
    z = jnp.zeros(w, dtype=p.dtype)
    occupied = h*jnp.abs(p[:w])[None, :] >= jnp.asarray(LAMBDA_THRESHOLDS)[:, None]
    return dict(p=p, positive=z, negative=z, effective=z, tracking=z, unresolved=z,
                crossing=z, count=jnp.array(0, dtype=jnp.int64),
                first_hit=jnp.where(occupied, 0, -1), failed=jnp.array(0))


def advance_factory(x, mass, h, eta):
    def one(old, y, count):
        w = (len(old['p'])-1)//3
        def step(_, state):
            direction, channels, info, _ = force(state['p'], x, y, mass)
            delta = -eta*direction
            pn = state['p']+delta
            change = h*(jnp.abs(pn[:w])-jnp.abs(state['p'][:w]))
            signed = -eta*h*jnp.sign(state['p'][:w])[None, :]*channels[:, :w]
            n = state['count']+1
            occupied = h*jnp.abs(pn[:w])[None, :] >= jnp.asarray(LAMBDA_THRESHOLDS)[:, None]
            new = dict(p=pn, positive=state['positive']+jnp.maximum(change, 0.),
                       negative=state['negative']+jnp.maximum(-change, 0.),
                       effective=state['effective']+signed[0], tracking=state['tracking']+signed[1],
                       unresolved=state['unresolved']+signed[2],
                       crossing=state['crossing']+change-signed.sum(axis=0), count=n,
                       first_hit=jnp.where((state['first_hit'] < 0)&occupied, n, state['first_hit']),
                       failed=state['failed'])
            valid = jnp.all(jnp.isfinite(pn)) & info['resolved'] & (state['failed'] == 0)
            new = jax.tree.map(lambda a, b: jnp.where(valid, a, b), new, state)
            new['failed'] = jnp.where(valid, 0, 1)
            return new
        return jax.lax.fori_loop(0, count, step, old)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def prepare(args):
    rows, pp, yy, sources = [], [], [], {}
    shared_x = None
    specifications = [(args.original, 0 if args.cohort == 'development' else 20),
                      (args.new_targets, 22 if args.cohort == 'development' else 23)]
    for path, seed in specifications:
        p, x, y, cases = ef.load_inputs(path)
        sources[str(path)] = ef.digest(path)
        if shared_x is None:
            shared_x = x
        np.testing.assert_array_equal(shared_x, x)
        for i, c in enumerate(cases):
            if c['seed'] == seed and c['start'] == 600000:
                rows.append(dict(c, cohort=args.cohort, nref=128))
                pp.append(p[i]); yy.append(y[i])
    if len(rows) != 23 or len({c['target'] for c in rows}) != 23:
        raise ValueError('Expected exactly the locked 23 target cases')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ar.atomic_npz(args.output, p=np.asarray(pp), x=shared_x, y=np.asarray(yy),
                  cases=np.array(json.dumps(rows)), sources=np.array(json.dumps(sources)))


def forecast(p, x, y, mass, eta, horizons):
    """Frozen full-feature Jacobian; diagonal mobility retained exactly.

    This is a finite-time tangent-model forecast, not an acquisition theorem.
    The baseline remainder, parameter-space coupling and all residual samples
    remain in the forecast. No future trajectory data enter it.
    """
    z, _ = af.unpack(p)
    a, b, c = np.asarray(z)
    u = x[:, None]*a+b
    h = np.tanh(u)
    exponential = np.exp(-2*np.abs(u)); s = 4*exponential/(1+exponential)**2
    J = np.concatenate((x[:, None]*s*c, s*c, h, np.ones((len(x), 1))), axis=1)
    r = h@c+float(p[-1])-y
    root = np.sqrt(mass)
    A = J*root[None, :]/np.sqrt(len(x))
    vals, vec = np.linalg.eigh(A.T@A)
    # The Gram operator is analytically PSD. Reject material numerical defects;
    # only negative eigenvalues within a scale-dependent roundoff allowance
    # are projected to zero. Every positive eigenvalue, however small, is kept.
    allowance = 64*np.finfo(float).eps*len(p)*max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
    if vals.min() < -allowance:
        raise ValueError('Gram spectrum has a negative eigenvalue beyond roundoff allowance')
    vals = np.maximum(vals, 0.)
    factors = 1-eta*vals
    projections = vec.T@(root*(J.T@r/len(x)))
    answer = []
    for n in horizons:
        # Near zero, expm1 avoids cancellation in the geometric sum.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            summed = np.where(vals > 0.,
                              -np.expm1(n*np.log1p(-eta*vals))/vals, eta*n)
            negative = factors < 0
            summed[negative] = (1-factors[negative]**n)/vals[negative]
            pn = np.asarray(p)-root*(vec@(summed*projections))
        answer.append(pn)
    return np.asarray(answer), float(np.max(np.abs(factors)))


def predict(args):
    from .effective_feedback_predict import frozen_effective_forecast
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    horizons = [int(v) for v in args.horizons.split(',')]
    predictions, radii = [], []
    effective_pure, effective_remainder, remainders, couplings = [], [], [], []
    q = transport.basis(x, 65)
    for arm in ARMS:
        d = mobility((pp.shape[1]-1)//3, arm)
        aa, rr = zip(*(forecast(p, x, y, d, args.eta, horizons) for p, y in zip(pp, yy)))
        predictions.append(aa); radii.append(rr)
        pure, corrected, rem, coupling = [], [], [], []
        for p, y in zip(pp, yy):
            z, bias = af.unpack(p)
            a, b, c = np.asarray(z)
            features = np.tanh(x[:, None]*a+b)
            exponential = np.exp(-2*np.abs(x[:, None]*a+b))
            s = 4*exponential/(1+exponential)**2
            J = np.concatenate((x[:, None]*s*c, s*c, features, np.ones((len(x), 1))), axis=1)
            modal = q.T@J/len(x)
            JC, JH = modal[:2], modal[2:]
            C = (JC*d)@JC.T
            if np.linalg.eigvalsh(C)[0] <= 64*np.finfo(float).eps*max(1., np.linalg.eigvalsh(C)[-1]):
                raise ValueError('Unresolved metric coarse balance at prediction fork')
            B = np.linalg.solve(C, (JC*d)@JH.T)
            T = d[:, None]*(JH.T-JC.T@B)
            residual = features@c+float(bias)-y
            eH = q[:, 2:].T@residual/len(x)
            result = frozen_effective_forecast(p, d*(J.T@residual/len(x)), T, JH,
                                               eH, args.eta, horizons)
            pure.append(result['pure']['p']); corrected.append(result['with_remainder']['p'])
            rem.append(result['R0']); coupling.append(result['S0'])
        effective_pure.append(pure); effective_remainder.append(corrected)
        remainders.append(rem); couplings.append(coupling)
    ar.atomic_npz(args.output/'predictions.npz', p=np.asarray(predictions),
                  spectral_radius=np.asarray(radii), horizons=np.asarray(horizons),
                  effective_pure_p=np.asarray(effective_pure),
                  effective_remainder_p=np.asarray(effective_remainder),
                  R0=np.asarray(remainders), S0=np.asarray(couplings))
    write_json(args.output/'manifest.json', dict(cases=cases, arms=ARMS, eta=args.eta,
        input_sha256=ef.digest(args.inputs), source_sha256=ef.digest(__file__),
        issued_utc=datetime.now(timezone.utc).isoformat(), horizons=horizons,
        prediction_sha256=ef.digest(args.output/'predictions.npz'),
        model='fork-only full-feature tangent GD with exact quotient mobility',
        tangent_limitation='Gauss-Newton frozen feature model; residual-weighted Hessian curvature is omitted',
        spectral_rule='retain every positive eigenvalue; negative Gram eigenvalues projected only within64*eps*dimension*spectral_scale',
        effective_model='frozen metric-dependent T and Schur S; pure and constant-remainder forecasts separate',
        h=2/128, degree=65, cumulative_travel_prediction='not supplied; endpoint displacement is not travel'))


def analyze(args):
    prediction = np.load(args.predictions/'predictions.npz')
    manifest = json.loads((args.predictions/'manifest.json').read_text())
    pp, _, _, cases = ef.load_inputs(args.inputs)
    w = (pp.shape[1]-1)//3
    rows = []
    for ai, arm in enumerate(ARMS):
        for hi, horizon in enumerate(prediction['horizons']):
            path = args.source/arm/'snapshots'/f'{int(horizon):09d}.npz'
            if not path.exists():
                continue
            with np.load(path) as data:
                for i, case in enumerate(cases):
                    actual = data['p'][i, :w]-pp[i, :w]
                    row = dict(**case, arm=arm, horizon=int(horizon), failed=int(data['failed'][i]),
                        relative_mse=float(data['relative_mse'][i]),
                        mean_lambda=float(np.mean(abs(data['p'][i, :w]))/64),
                        positive=float(data['positive'][i].mean()), negative=float(data['negative'][i].mean()),
                        effective=float(data['effective'][i].mean()), tracking=float(data['tracking'][i].mean()))
                    for label, key in [('tangent', 'p'), ('effective_pure', 'effective_pure_p'),
                                       ('effective_remainder', 'effective_remainder_p')]:
                        predicted = prediction[key][ai, i, hi, :w]-pp[i, :w]
                        row[label+'_displacement_error'] = float(np.linalg.norm(predicted-actual))
                        row[label+'_mean_lambda_change'] = float(np.mean(abs(pp[i, :w]+predicted)-abs(pp[i, :w]))/64)
                    rows.append(row)
    if not rows:
        raise ValueError('No completed prediction horizons found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(args):
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    ef.verify_backend(out, args.backend)
    prediction = json.loads((args.predictions/'manifest.json').read_text())
    if prediction['input_sha256'] != ef.digest(args.inputs) or prediction['eta'] != args.eta:
        raise ValueError('Predictions do not match the run inputs/rate')
    if prediction['prediction_sha256'] != ef.digest(args.predictions/'predictions.npz'):
        raise ValueError('Prediction arrays changed after issuance')
    w = (pp.shape[1]-1)//3
    h = 2/128
    mass = jnp.asarray(mobility(w, args.arm))
    manifest = dict(cases=cases, arm=args.arm, eta=args.eta, nref=128, quotient_width=w,
        expanded_width=w*settings(args.arm)[0], mobility=np.asarray(mass).tolist(),
        input_sha256=ef.digest(args.inputs), source_sha256=ef.digest(__file__),
        prediction_manifest_sha256=ef.digest(args.predictions/'manifest.json'),
        lambda_thresholds=LAMBDA_THRESHOLDS, units='travel in lambda=h*abs(a); actual GD updates')
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Changed run on resume')
    write_json(mp, manifest)
    state = jax.vmap(lambda p: initial(p, h))(jnp.asarray(pp)); offset = 0
    if (out/'state.npz').exists():
        with np.load(out/'state.npz') as old:
            offset = int(old['offset']); state = {k: jnp.asarray(old[k]) for k in state}
    advance = advance_factory(jnp.asarray(x), mass, h, args.eta)
    def measure_one(p, y):
        direction, channels, info, residual = force(p, jnp.asarray(x), y, mass)
        norms, ratio = fine_residual_forcing(p, jnp.asarray(x), channels)
        return direction, channels, info, residual, norms, ratio
    measure = jax.jit(jax.vmap(measure_one))
    begun = time.monotonic()
    (out/'snapshots').mkdir(exist_ok=True)
    def save():
        host = jax.device_get(state)
        direction, channels, info, residual, residual_forcing, forcing_ratio = jax.device_get(measure(state['p'], jnp.asarray(yy)))
        displacement = h*(np.abs(host['p'][:, :w])-np.abs(pp[:, :w]))
        ar.atomic_npz(out/'state.npz', offset=np.array(offset), **host)
        ar.atomic_npz(out/'snapshots'/f'{offset:09d}.npz', offset=np.array(offset), **host,
            direction=direction, channels=channels[:, :, :w], coarse_min=info['min_eigenvalue'],
            full_complement_residual_forcing_norms=residual_forcing,
            full_complement_tracking_to_effective_residual_forcing=forcing_ratio,
            relative_mse=np.mean(residual**2, axis=1)/np.mean(yy**2, axis=1),
            population=np.mean(h*np.abs(host['p'][:, None, :w]) >= np.asarray(LAMBDA_THRESHOLDS)[None, :, None], axis=2))
        status = dict(offset=offset, horizon=args.horizon, complete=offset == args.horizon,
            failed=host['failed'].tolist(), seconds=time.monotonic()-begun,
            travel_error=float(np.max(np.abs(host['positive']-host['negative']-displacement))),
            channel_error=float(np.max(np.abs(host['effective']+host['tracking']+host['unresolved']+host['crossing']-displacement))))
        write_json(out/'status.json', status)
        return status
    if offset == 0:
        save()
    for end in sorted({v for v in (1, 2, 10, 100, 1000, args.horizon) if v <= args.horizon}
                      | set(range(1000, args.horizon+1, args.stride))):
        if end <= offset:
            continue
        state = advance(state, jnp.asarray(yy), end-offset); jax.block_until_ready(state)
        offset = end; status = save()
        if end <= 1000 or end % 10000 == 0:
            print(json.dumps(status), flush=True)
        if any(status['failed']) or time.monotonic()-begun >= args.max_seconds:
            break
    if np.any(np.asarray(state['failed'])):
        raise RuntimeError('Failed numerical branch; evidence retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--new-targets', type=Path, required=True)
    p.add_argument('--cohort', choices=('development', 'confirmation'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('predict')
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--horizons', default='1,10,1000,10000,50000,200000')
    p.add_argument('--eta', type=float, default=.002)
    p = sub.add_parser('run')
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--eta', type=float, default=.002); p.add_argument('--horizon', type=int, default=10000)
    p.add_argument('--stride', type=int, default=1000); p.add_argument('--max-seconds', type=float, default=1600)
    p.add_argument('--backend', choices=('slurm', 'cpu'), default='slurm')
    p = sub.add_parser('analyze')
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--predictions', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('FP64 is required')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
