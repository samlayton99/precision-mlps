"""Ordinary-GD continuations of prepared finite dilations; sparse exact accounting.

Run as a module under Slurm with JAX_ENABLE_X64=true. No repairs are performed
here. Invalid prepared branches remain in the manifest and are not advanced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af
from . import mechanism_persistence as persistence
from . import effective_feedback_holdout as holdout, mechanism_widths as widths
from .mechanism_persistence import validate_backend, finite_json


HORIZONS = (0, 1, 10, 100, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000, 100000)
ERROR_THRESHOLDS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
CHANNELS = ('direct_fine', 'compensation', 'tracking', 'unresolved')


def channels(p, x, y):
    gradient, residual, jc, ec = af.field(p, x, y)
    split, info = af.split(gradient, jc, ec)
    parts = jnp.stack((jnp.where(info['resolved'], info['fine'], 0.),
                       jnp.where(info['resolved'], info['balanced'], 0.), split[1], split[2]))
    return gradient, parts, split[0], info['resolved'], residual, jc


def initial_state(p):
    w = (p.shape[-1]-1)//3
    return dict(p=p, signed=jnp.zeros((4, w), dtype=p.dtype),
                positive=jnp.zeros(w, dtype=p.dtype), negative=jnp.zeros(w, dtype=p.dtype),
                crossing=jnp.zeros(w, dtype=p.dtype), norm_integral=jnp.zeros(4, dtype=p.dtype),
                absolute_radial=jnp.zeros((4, w), dtype=p.dtype),
                effective_absolute_radial=jnp.zeros(w, dtype=p.dtype),
                effective_norm_integral=jnp.array(0., dtype=p.dtype),
                first_hit=jnp.full(w, -1, dtype=jnp.int64),
                error_first_hit=jnp.full(len(ERROR_THRESHOLDS), -1, dtype=jnp.int64),
                diagnostic_unresolved_steps=jnp.array(0, dtype=jnp.int64),
                count=jnp.array(0, dtype=jnp.int64), failed=jnp.array(False))


def step(old, x, y, h, eta, freeze_geometry=False):
    p = old['p']; w = (p.shape[0]-1)//3
    gradient, parts, fine, resolved, residual, _ = channels(p, x, y)
    if freeze_geometry:
        # The gradient remains ordinary on c,d. Channel motion is masked by the
        # same mobility, so nonexistent slope motion is never attributed to F/R.
        gradient = gradient.at[:2*w].set(0.)
        parts = parts.at[:, :2*w].set(0.)
        fine = fine.at[:2*w].set(0.)
    delta = -eta*gradient
    pn = p+delta
    change = h*(jnp.abs(pn[:w])-jnp.abs(p[:w]))
    signed = -eta*h*jnp.sign(p[:w])[None, :]*parts[:, :w]
    relative_error = jnp.linalg.norm(residual)/jnp.linalg.norm(y)
    hits = jnp.where((old['error_first_hit'] < 0)&(relative_error <= jnp.array(ERROR_THRESHOLDS)),
                     old['count'], old['error_first_hit'])
    new = dict(p=pn, signed=old['signed']+signed,
               positive=old['positive']+jnp.maximum(change, 0),
               negative=old['negative']+jnp.maximum(-change, 0),
               crossing=old['crossing']+change-jnp.sum(signed, axis=0),
               norm_integral=old['norm_integral']+eta*h*jnp.linalg.norm(parts[:, :w], axis=1),
               absolute_radial=old['absolute_radial']+jnp.abs(signed),
               effective_absolute_radial=old['effective_absolute_radial']+eta*h*jnp.abs(jnp.sign(p[:w])*fine[:w]),
               effective_norm_integral=old['effective_norm_integral']+eta*h*jnp.linalg.norm(fine[:w]),
               first_hit=jnp.where((old['first_hit'] < 0)&(h*jnp.abs(pn[:w]) >= .25),
                                   old['count']+1, old['first_hit']),
               error_first_hit=hits,
               diagnostic_unresolved_steps=old['diagnostic_unresolved_steps']+(~resolved).astype(jnp.int64),
               count=old['count']+1, failed=old['failed'])
    good = ~old['failed'] & jnp.all(jnp.isfinite(pn)) & jnp.all(jnp.isfinite(gradient))
    new = jax.tree.map(lambda a, b: jnp.where(good, a, b), new, old)
    new['failed'] = ~good
    return new


def advance_factory(x, eta, freeze_geometry=False):
    def advance(state, y, h, length):
        return jax.lax.fori_loop(0, length, lambda _, s: step(s, x, y, h, eta, freeze_geometry), state)
    return jax.jit(jax.vmap(advance, in_axes=(0, 0, 0, None)))


def diagnostic(p, x, y, h, q):
    gradient, parts, fine, resolved, residual, jc = channels(p, x, y)
    w = (p.shape[0]-1)//3
    a, b, c = p[:-1].reshape(3, w)
    u = x[:, None]*a+b
    activation = jnp.tanh(u)
    exponential = jnp.exp(-2*jnp.abs(u))
    derivative = 4*exponential/(1+exponential)**2
    jac = jnp.concatenate((derivative*c*x[:, None], derivative*c, activation,
                           jnp.ones((len(x), 1), dtype=p.dtype)), axis=1)
    output = residual+y
    coeff = q.T@output/len(x); target = q.T@y/len(x)
    modal_jac = q.T@jac/len(x)
    gram = jc@jc.T
    safe_gram = jnp.where(resolved, gram, jnp.eye(2, dtype=p.dtype))
    projected = modal_jac-modal_jac@jc.T@jnp.linalg.solve(safe_gram, jc)
    radial = jnp.sign(a)
    modal_generated = -h*jnp.mean(projected[:, :w]*radial, axis=1)*coeff
    modal_target = h*jnp.mean(projected[:, :w]*radial, axis=1)*target
    velocities = -h*radial[None, :]*jnp.vstack((fine[:w], parts[:, :w]))
    slope_rms = jnp.sqrt(jnp.mean(a*a))
    rms_rates = jnp.where(slope_rms > 0,
                          -h*jnp.mean(a[None, :]*jnp.vstack((fine[:w], parts[:, :w])), axis=1)/slope_rms,
                          h*jnp.sqrt(jnp.mean(jnp.vstack((fine[:w], parts[:, :w]))**2, axis=1)))
    radii2 = w*(a*a+b*b+c*c)
    M = jnp.mean(radii2); M6 = jnp.mean(radii2**3)
    return dict(relative_l2=jnp.linalg.norm(residual)/jnp.linalg.norm(y),
                M=M, M6=M6, C6=M6/M**3,
                Q=jnp.sum(jnp.abs(c)*a*a*(jnp.abs(b)+jnp.abs(a)/3)),
                loss=jnp.mean(residual**2)/2, F_norm=jnp.linalg.norm(fine),
                Fa_norm=jnp.linalg.norm(fine[:w]), Ra_norm=jnp.linalg.norm(parts[2, :w]),
                channel_norms=jnp.linalg.norm(parts[:, :w], axis=1),
                signed_mean_rates=jnp.mean(velocities, axis=1),
                radial_rms_rates=jnp.sqrt(jnp.mean(velocities**2, axis=1)),
                lambda_rms_rates=rms_rates,
                outward_mean_rates=jnp.mean(jnp.maximum(velocities, 0), axis=1),
                lambda_mean=jnp.mean(h*jnp.abs(a)),
                gamma_mean=jnp.mean(jnp.abs(a)),
                activation_argument_rms=jnp.sqrt(jnp.mean(u*u)),
                activation_argument_max=jnp.max(jnp.abs(u)),
                coarse_gram_eigenvalues=jnp.linalg.eigvalsh(gram),
                lambda_rms=jnp.sqrt(jnp.mean((h*a)**2)),
                lambda_quantiles=jnp.quantile(h*jnp.abs(a), jnp.array([.5, .9, .99, 1.])),
                q23_generated=coeff, q23_target=target,
                q23_generated_signed_rate=modal_generated, q23_target_signed_rate=modal_target,
                resolved=resolved, force_identity_error=jnp.linalg.norm(parts.sum(axis=0)-gradient),
                Fa=fine[:w], ga=gradient[:w])


def diagnostic_factory(batch_size=16):
    """Bound the diagnostic Jacobian batch without changing training batches."""
    def measure(pp, x, yy, hh, q):
        return jax.lax.map(lambda item: diagnostic(item[0], x, item[1], item[2], q),
                           (pp, yy, hh), batch_size=batch_size)
    return jax.jit(measure)


def evaluation_relative_l2(p, x, y):
    a, b, c = p[:-1].reshape(3, -1)
    return jnp.linalg.norm(jnp.tanh(x[:, None]*a+b)@c+p[-1]-y)/jnp.linalg.norm(y)


def evaluation_factory(batch_size=16):
    def evaluate(pp, x, yy):
        return jax.lax.map(lambda item: evaluation_relative_l2(item[0], x, item[1]),
                           (pp, yy), batch_size=batch_size)
    return jax.jit(evaluate)


def run(args):
    if not jax.config.x64_enabled:
        raise ValueError('FP64 requires JAX_ENABLE_X64=true')
    validate_backend(args.backend)
    diagnostic_batch = getattr(args, 'diagnostic_batch', 16)
    freeze_geometry = getattr(args, 'freeze_geometry', False)
    if diagnostic_batch <= 0:
        raise ValueError('diagnostic_batch must be positive')
    with np.load(args.input) as data:
        pack = {k: data[k] for k in data.files}
    cases = json.loads(str(pack['cases']))
    chosen = []
    for i, case in enumerate(cases):
        if any(value and str(case.get(key)) not in value.split(',')
               for key, value in [('target', args.targets), ('cohort', args.cohorts), ('arm', args.arms), ('seed', getattr(args, 'seeds', None))]):
            continue
        chosen.append((i, case))
    valid = [(i, c) for i, c in chosen if c.get('valid', True)]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    code_sources={Path(name).name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                  for name in (__file__, af.__file__, persistence.__file__)},
                    versions=dict(jax=jax.__version__, numpy=np.__version__),
                    backend=jax.default_backend(), devices=[str(device) for device in jax.devices()],
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'), source_commit=os.environ.get('D34_SOURCE_COMMIT'),
                    eta=args.eta, steps=args.steps, channels=CHANNELS, error_thresholds=ERROR_THRESHOLDS,
                    diagnostic_batch=diagnostic_batch,
                    freeze_geometry=freeze_geometry,
                    diagnostic_force_scope='unmasked full-GD reference gradient at the current state',
                    motion_rate_and_integral_scope='actual parameter mobility; zero geometry motion when frozen',
                    failure_rule='nonfinite ordinary GD; unresolved decomposition continues in unknown channel',
                    rate_channels=('effective', *CHANNELS),
                    rate_time='physical flow time t=eta*n; normalized slopes lambda=h*abs(a)',
                    quantiles=(.5, .9, .99, 1.), cases=[dict(c, input_index=i) for i, c in valid],
                    skipped_invalid=[dict(c, input_index=i) for i, c in chosen if not c.get('valid', True)])
    (args.output/'manifest.json').write_text(json.dumps(finite_json(manifest), indent=2)+'\n')
    if not valid:
        return
    indices = [i for i, _ in valid]
    p0 = jnp.asarray(pack['p'][indices]); y = jnp.asarray(pack['y'][indices]); x = jnp.asarray(pack['x'])
    if abs(float(jnp.mean(x))) > 1e-14*float(jnp.sqrt(jnp.mean(x*x))):
        raise ValueError('Force decomposition requires the supplied centered symmetric grid')
    h = jnp.asarray([float(c['h']) if 'h' in c else 2/float(c.get('nref', c.get('Nref')))
                     for _, c in valid])
    assert p0.dtype == jnp.float64 and y.dtype == jnp.float64 and x.dtype == jnp.float64
    # Positive-leading empirical orthonormal degrees 2,3 on the actual grid.
    basis, triangular = np.linalg.qr(np.polynomial.polynomial.polyvander(np.asarray(x), 3))
    basis *= np.sign(np.diag(triangular))[None, :]*np.sqrt(len(x))
    q = jnp.asarray(basis[:, 2:4])
    state = jax.vmap(initial_state)(p0)
    state['first_hit'] = jnp.where(h[:, None]*jnp.abs(p0[:, :(p0.shape[1]-1)//3]) >= .25, 0, -1)
    measure = diagnostic_factory(diagnostic_batch)
    initial = measure(p0, x, y, h, q)
    eval_x = eval_y = None
    if 'x_eval' in pack and 'y_eval' in pack:
        eval_x, eval_y = jnp.asarray(pack['x_eval']), jnp.asarray(pack['y_eval'][indices])
    elif all(c['target'] in (*af.TARGETS, *holdout.TARGETS) for _, c in valid):
        evaluation = [widths.data(c['target'], 8192) for _, c in valid]
        eval_x = jnp.asarray(evaluation[0][0])
        eval_y = jnp.asarray(np.stack([record[1] for record in evaluation]))
    evaluate = evaluation_factory(diagnostic_batch) if eval_x is not None else None
    advance = advance_factory(x, args.eta, freeze_geometry)
    schedule = sorted({0, args.steps, *(round(n*.002/args.eta) for n in HORIZONS
                                        if round(n*.002/args.eta) <= args.steps)})
    w = (p0.shape[1]-1)//3
    started = time.monotonic(); previous = 0
    for count in schedule:
        if count > previous:
            state = advance(state, y, h, count-previous)
        measured = measure(state['p'], x, y, h, q)
        if freeze_geometry:
            for name in ('signed_mean_rates', 'radial_rms_rates', 'lambda_rms_rates',
                         'outward_mean_rates', 'q23_generated_signed_rate', 'q23_target_signed_rate'):
                measured[name] = jnp.zeros_like(measured[name])
        state['error_first_hit'] = jnp.where(
            (state['error_first_hit'] < 0)&(measured['relative_l2'][:, None] <= jnp.array(ERROR_THRESHOLDS))
            &~state['failed'][:, None], state['count'][:, None], state['error_first_hit'])
        measured['relative_eval_l2'] = (evaluate(state['p'], eval_x, eval_y) if evaluate is not None
                                         else jnp.full(len(valid), jnp.nan))
        forecast = p0[:, :w]-args.eta*count*initial['Fa']*(not freeze_geometry)
        full_forecast = p0[:, :w]-args.eta*count*initial['ga']*(not freeze_geometry)
        displacement = state['p'][:, :w]-p0[:, :w]
        error = jnp.linalg.norm(state['p'][:, :w]-forecast, axis=1)
        norm = jnp.linalg.norm(displacement, axis=1)
        saved = dict(**state, **{k: v for k, v in measured.items() if k not in ('Fa', 'ga')},
                     forecast_a=forecast, initial_Fa=initial['Fa'],
                     forecast_lambda=h[:, None]*jnp.abs(forecast),
                     full_forecast_a=full_forecast,
                     full_forecast_lambda=h[:, None]*jnp.abs(full_forecast),
                     forecast_absolute_error=error,
                     forecast_relative_error=jnp.where(norm > 0, error/norm, jnp.nan),
                     closure_error=jnp.max(jnp.abs(state['signed'].sum(axis=1)+state['crossing']
                                                  -h[:, None]*(jnp.abs(state['p'][:, :w])-jnp.abs(p0[:, :w]))), axis=1))
        host = jax.tree.map(np.asarray, saved)
        np.savez_compressed(args.output/f'{count:09d}.npz', **host)
        print(json.dumps(dict(step=count, seconds=time.monotonic()-started,
                              failed=int(host['failed'].sum()), max_closure=float(host['closure_error'].max()))), flush=True)
        previous = count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eta', type=float, default=.002)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--diagnostic-batch', type=int, default=16)
    parser.add_argument('--freeze-geometry', action='store_true',
                        help='Hold a,b exactly fixed; update only c,d by ordinary GD')
    parser.add_argument('--targets'); parser.add_argument('--cohorts'); parser.add_argument('--arms'); parser.add_argument('--seeds')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    args = parser.parse_args()
    if args.eta <= 0 or args.steps < 0:
        parser.error('eta must be positive and steps nonnegative')
    run(args)


if __name__ == '__main__':
    main()
