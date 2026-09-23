"""Independent-width feasibility panel; original full-batch GD without tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, adam_run as ar, effective_feedback as ef
from . import effective_feedback_holdout as holdout, targets
from .run import write_json
from .mechanism_splitting import fine_residual_forcing

TARGETS = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
WIDTHS = ((128, 24), (512, 96), (1024, 192))
SEEDS = (30, 31)
ETA = .002


def data(name, samples=2048):
    return holdout.data(name, samples) if name in holdout.TARGETS else af.data(name, samples)


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for nref, halo in WIDTHS:
        path = args.output/f'N{nref}.npz'
        if path.exists():
            raise FileExistsError(path)
        pp, yy, y_eval, cases = [], [], [], []
        for seed in SEEDS:
            z, d = targets.initial(nref, halo, seed)
            for name in TARGETS:
                x, y, _, scale = data(name)
                xe, ye, _, _ = data(name, 8192)
                pp.append(np.r_[z.ravel(), d]); yy.append(y); y_eval.append(ye)
                cases.append(dict(target=name, seed=seed, start=0, eta=ETA,
                    width=nref+2*halo+1, nref=nref, halo=halo, target_scale=scale))
        ar.atomic_npz(path, p=np.asarray(pp), x=x, y=np.asarray(yy),
                      x_eval=xe, y_eval=np.asarray(y_eval), cases=np.array(json.dumps(cases)),
                      sources=np.array(json.dumps(dict(runner_sha256=ef.digest(__file__),
                          target_sha256=ef.digest(targets.__file__),
                          initialization='unchanged independent D28 Xavier; same seed shared across targets only'))))


def initial(p, h):
    w = (len(p)-1)//3
    zero = jnp.zeros(w, dtype=p.dtype)
    occupied = h*jnp.abs(p[:w]) >= .25
    return dict(p=p, positive=zero, negative=zero, first_hit=jnp.where(occupied, 0, -1),
                count=jnp.array(0, dtype=jnp.int64), failed=jnp.array(False))


def gradient(p, x, y):
    """Plain full empirical gradient; no coarse solve during training."""
    (a, b, c), d = af.unpack(p)
    u = x[:, None]*a+b
    feature = jnp.tanh(u)
    exponential = jnp.exp(-2*jnp.abs(u)); derivative = 4*exponential/(1+exponential)**2
    residual = feature@c+d-y
    weighted = residual[:, None]*derivative
    return jnp.r_[c*(x@weighted)/len(x), c*jnp.mean(weighted, axis=0),
                  feature.T@residual/len(x), jnp.mean(residual)]


def advance_factory(x, h):
    def one(old, y, length):
        w = (len(old['p'])-1)//3
        def step(_, state):
            pn = state['p']-ETA*gradient(state['p'], x, y)
            change = h*(jnp.abs(pn[:w])-jnp.abs(state['p'][:w]))
            count = state['count']+1
            occupied = h*jnp.abs(pn[:w]) >= .25
            new = dict(p=pn, positive=state['positive']+jnp.maximum(change, 0),
                negative=state['negative']+jnp.maximum(-change, 0), count=count,
                first_hit=jnp.where((state['first_hit'] < 0)&occupied, count, state['first_hit']),
                failed=state['failed'])
            good = jnp.all(jnp.isfinite(pn)) & ~state['failed']
            new = jax.tree.map(lambda a, b: jnp.where(good, a, b), new, state)
            new['failed'] = ~good
            return new
        return jax.lax.fori_loop(0, length, step, old)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def diagnostic(p, x, y, xe, ye, h):
    g, r, jc, ec = af.field(p, x, y)
    channels, info = af.split(g, jc, ec)
    (a, b, c), d = af.unpack(p)
    w = len(a)
    direct_coarse = jc.T@ec
    er = jnp.tanh(xe[:, None]*a+b)@c+d-ye
    effective_norm = jnp.linalg.norm(channels[0, :w])
    tracking_norm = jnp.linalg.norm(channels[1, :w])
    residual_forcing, forcing_ratio = fine_residual_forcing(p, x, channels)
    return dict(mean_lambda=jnp.mean(abs(a))*h, max_lambda=jnp.max(abs(a))*h,
        mean_lambda_rate=-h*jnp.mean(jnp.sign(a)*g[:w]),
        mean_lambda_rate_per_update=-ETA*h*jnp.mean(jnp.sign(a)*g[:w]),
        effective_lambda_rate=-h*jnp.mean(jnp.sign(a)*channels[0, :w]),
        tracking_lambda_rate=-h*jnp.mean(jnp.sign(a)*channels[1, :w]),
        lambda_speed=h*jnp.linalg.norm(g[:w])/jnp.sqrt(w),
        readout_rms_over_h=jnp.sqrt(jnp.mean(c*c))/h,
        relative_mse=jnp.mean(r*r)/jnp.mean(y*y), relative_eval_mse=jnp.mean(er*er)/jnp.mean(ye*ye),
        coarse_residual_norm=jnp.linalg.norm(ec), coarse_tracking_residual_norm=jnp.linalg.norm(info['z']),
        direct_coarse_slope_norm=jnp.linalg.norm(direct_coarse[:w]),
        direct_coarse_readout_norm=jnp.linalg.norm(direct_coarse[2*w:3*w]),
        effective_slope_norm=effective_norm, tracking_slope_norm=tracking_norm,
        tracking_readout_norm=jnp.linalg.norm(channels[1, 2*w:3*w]),
        effective_readout_norm=jnp.linalg.norm(channels[0, 2*w:3*w]),
        full_complement_effective_residual_forcing=residual_forcing[0],
        full_complement_tracking_residual_forcing=residual_forcing[1],
        full_complement_tracking_to_effective_residual_forcing=forcing_ratio,
        tracking_to_effective_slope=jnp.where(effective_norm > 0, tracking_norm/effective_norm, jnp.nan),
        coarse_resolved=info['resolved'], coarse_min=info['min_eigenvalue'],
        population_lambda025=jnp.mean(h*abs(a) >= .25))


def run(args):
    pp, x, yy, cases = ef.load_inputs(args.inputs)
    with np.load(args.inputs) as source:
        xe, ye = source['x_eval'], source['y_eval']
    nrefs = {c['nref'] for c in cases}
    if len(nrefs) != 1:
        raise ValueError('Group runs by width')
    nref = nrefs.pop(); h = 2/nref
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    ef.verify_backend(out, args.backend)
    manifest = dict(cases=cases, eta=ETA, nref=nref, h=h, input_sha256=ef.digest(args.inputs),
        source_sha256=ef.digest(__file__), threshold_lambda=.25,
        clock='actual updates; physical time=.002*updates',
        rate_units='all rates per physical GD time except explicit rate_per_update',
        eligibility='Report coarse ratios at every retained state; no post-hoc regime selection or automatic learning-rate tuning')
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Changed run on resume')
    write_json(mp, manifest)
    state = jax.vmap(lambda p: initial(p, h))(jnp.asarray(pp)); offset = 0
    if (out/'state.npz').exists():
        with np.load(out/'state.npz') as old:
            offset = int(old['offset']); state = {k: jnp.asarray(old[k]) for k in state}
    advance = advance_factory(jnp.asarray(x), h)
    measure = jax.jit(jax.vmap(lambda p, y, e: diagnostic(p, jnp.asarray(x), y,
                                 jnp.asarray(xe), e, h)))
    begun = time.monotonic(); (out/'snapshots').mkdir(exist_ok=True)
    def save():
        host = jax.device_get(state)
        metrics = jax.device_get(measure(state['p'], jnp.asarray(yy), jnp.asarray(ye)))
        w = (pp.shape[1]-1)//3
        displacement = h*(np.abs(host['p'][:, :w])-np.abs(pp[:, :w]))
        ar.atomic_npz(out/'state.npz', offset=np.array(offset), **host)
        ar.atomic_npz(out/'snapshots'/f'{offset:09d}.npz', offset=np.array(offset), **host,
                      **{'metric_'+key: value for key, value in metrics.items()})
        status = dict(offset=offset, horizon=args.horizon, complete=offset == args.horizon,
            failed=host['failed'].tolist(), seconds=time.monotonic()-begun,
            unresolved_coarse=bool(np.any(~metrics['coarse_resolved'])),
            travel_error=float(np.max(abs(host['positive']-host['negative']-displacement))),
            everhit_fraction=np.mean(host['first_hit'] >= 0, axis=1).tolist())
        write_json(out/'status.json', status)
        return status
    if offset == 0:
        save()
    schedule = {v for v in (1, 2, 10, 100, 1000, args.horizon) if v <= args.horizon}
    schedule |= set(range(2000, args.horizon+1, 2000))
    for end in sorted(schedule):
        if end <= offset:
            continue
        state = advance(state, jnp.asarray(yy), end-offset); jax.block_until_ready(state)
        offset = end; status = save(); print(json.dumps(status), flush=True)
        if any(status['failed']) or time.monotonic()-begun >= args.max_seconds:
            break
    if np.any(np.asarray(state['failed'])):
        raise RuntimeError('Nonfinite ordinary GD case retained explicitly')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--inputs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--horizon', type=int, choices=(20000, 100000), default=20000)
    p.add_argument('--max-seconds', type=float, default=600)
    p.add_argument('--backend', choices=('slurm', 'cpu'), default='slurm')
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError('FP64 is required')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
