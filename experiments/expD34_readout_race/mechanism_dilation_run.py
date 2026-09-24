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
from .mechanism_persistence import validate_backend, finite_json


HORIZONS = (0, 1, 10, 100, 1000, 2000, 5000, 10000, 20000)
CHANNELS = ('direct_fine', 'compensation', 'tracking')


def channels(p, x, y):
    gradient, residual, jc, ec = af.field(p, x, y)
    split, info = af.split(gradient, jc, ec)
    parts = jnp.stack((info['fine'], info['balanced'], split[1]))
    return gradient, parts, split[0], info['resolved'], residual, jc


def initial_state(p):
    w = (p.shape[-1]-1)//3
    return dict(p=p, signed=jnp.zeros((3, w), dtype=p.dtype),
                positive=jnp.zeros(w, dtype=p.dtype), negative=jnp.zeros(w, dtype=p.dtype),
                crossing=jnp.zeros(w, dtype=p.dtype), norm_integral=jnp.zeros(3, dtype=p.dtype),
                absolute_radial=jnp.zeros((3, w), dtype=p.dtype),
                effective_absolute_radial=jnp.zeros(w, dtype=p.dtype),
                effective_norm_integral=jnp.array(0., dtype=p.dtype),
                first_hit=jnp.full(w, -1, dtype=jnp.int64),
                count=jnp.array(0, dtype=jnp.int64), failed=jnp.array(False))


def step(old, x, y, h, eta):
    p = old['p']; w = (p.shape[0]-1)//3
    gradient, parts, fine, resolved, _, _ = channels(p, x, y)
    delta = -eta*gradient
    pn = p+delta
    change = h*(jnp.abs(pn[:w])-jnp.abs(p[:w]))
    signed = -eta*h*jnp.sign(p[:w])[None, :]*parts[:, :w]
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
               count=old['count']+1, failed=old['failed'])
    good = ~old['failed'] & resolved & jnp.all(jnp.isfinite(pn)) & jnp.all(jnp.isfinite(parts))
    new = jax.tree.map(lambda a, b: jnp.where(good, a, b), new, old)
    new['failed'] = ~good
    return new


def advance_factory(x, eta):
    def advance(state, y, h, length):
        return jax.lax.fori_loop(0, length, lambda _, s: step(s, x, y, h, eta), state)
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
    return dict(loss=jnp.mean(residual**2)/2, F_norm=jnp.linalg.norm(fine),
                Fa_norm=jnp.linalg.norm(fine[:w]), Ra_norm=jnp.linalg.norm(parts[2, :w]),
                channel_norms=jnp.linalg.norm(parts[:, :w], axis=1),
                signed_mean_rates=jnp.mean(velocities, axis=1),
                radial_rms_rates=jnp.sqrt(jnp.mean(velocities**2, axis=1)),
                lambda_rms_rates=rms_rates,
                outward_mean_rates=jnp.mean(jnp.maximum(velocities, 0), axis=1),
                lambda_mean=jnp.mean(h*jnp.abs(a)),
                lambda_rms=jnp.sqrt(jnp.mean((h*a)**2)),
                lambda_quantiles=jnp.quantile(h*jnp.abs(a), jnp.array([.5, .9, .99, 1.])),
                q23_generated=coeff, q23_target=target,
                q23_generated_signed_rate=modal_generated, q23_target_signed_rate=modal_target,
                resolved=resolved, force_identity_error=jnp.linalg.norm(parts.sum(axis=0)-gradient),
                Fa=fine[:w], ga=gradient[:w])


def run(args):
    if not jax.config.x64_enabled:
        raise ValueError('FP64 requires JAX_ENABLE_X64=true')
    validate_backend(args.backend)
    with np.load(args.input) as data:
        pack = {k: data[k] for k in data.files}
    cases = json.loads(str(pack['cases']))
    chosen = []
    for i, case in enumerate(cases):
        if any(value and str(case.get(key)) not in value.split(',')
               for key, value in [('target', args.targets), ('cohort', args.cohorts), ('arm', args.arms)]):
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
                    eta=args.eta, steps=args.steps, channels=CHANNELS,
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
    measure = jax.jit(jax.vmap(diagnostic, in_axes=(0, None, 0, 0, None)))
    initial = measure(p0, x, y, h, q)
    advance = advance_factory(x, args.eta)
    schedule = sorted({0, args.steps, *(round(n*.002/args.eta) for n in HORIZONS
                                        if round(n*.002/args.eta) <= args.steps)})
    w = (p0.shape[1]-1)//3
    started = time.monotonic(); previous = 0
    for count in schedule:
        if count > previous:
            state = advance(state, y, h, count-previous)
        measured = measure(state['p'], x, y, h, q)
        forecast = p0[:, :w]-args.eta*count*initial['Fa']
        full_forecast = p0[:, :w]-args.eta*count*initial['ga']
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
    parser.add_argument('--targets'); parser.add_argument('--cohorts'); parser.add_argument('--arms')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='gpu')
    args = parser.parse_args()
    if args.eta <= 0 or args.steps < 0:
        parser.error('eta must be positive and steps nonnegative')
    run(args)


if __name__ == '__main__':
    main()
