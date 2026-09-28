"""Checkpoint-issued force-growth forecasts and matched GD feedback tests.

Run preparation, numerical checks, continuations, and analysis under Slurm.
Only sparse states and cumulative motion budgets are saved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, mechanism_adam as io
from . import mechanism_persistence_kernel as kernel

HORIZONS = (0, 1, 2, 10, 100, 1000, 10000, 20000)
ETA = .002


def hashes():
    return {Path(p).name: io.digest(p) for p in (__file__, kernel.__file__, af.__file__)}


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(finite_json(rows))


def exponential_sum(rate, steps):
    if steps == 0:
        return 0.
    z = ETA*rate
    with np.errstate(over='ignore', invalid='ignore'):
        return float(steps) if z == 0 else np.expm1(steps*z)/np.expm1(z)


def selected(pack, args):
    cases = json.loads(str(pack['cases']))
    targets = set(args.targets.split(',')) if args.targets else None
    return [(i, c) for i, c in enumerate(cases)
            if (args.seed is None or int(c['seed']) == args.seed)
            and (targets is None or c['target'] in targets)
            and (not args.unpulsed or float(c.get('amplitude', 0)) == 0)]


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    pack = dict(np.load(args.inputs))
    x = jnp.asarray(pack['x'])
    fork = jax.jit(kernel.fork)
    rows, records, points, labels, refs_f, refs_e, arms, hs = [], [], [], [], [], [], [], []
    forecasts = {name: [] for name in ('constant', 'amplification', 'q', 'valid', 'first_step')}
    for source_index, case in selected(pack, args):
        p, y = jnp.asarray(pack['p'][source_index]), jnp.asarray(pack['y'][source_index])
        item = jax.device_get(fork(p, x, y))
        reference = item['reference']
        f0 = np.asarray(reference['F0']); q0 = np.linalg.norm(f0)
        g0 = np.asarray(af.field(p, x, y)[0]); r0 = g0-f0
        chosen = (0,) if args.physical else range(len(kernel.ARMS))
        for ai in chosen:
            arm = kernel.ARMS[ai]
            rate = float(item['arm_diagnostics']['k_actual'][ai])
            width = (len(p)-1)//3
            h = 2/args.nref
            rec = dict(case, parent_source_index=case.get('source_index', source_index),
                       source_index=source_index, cohort=args.cohort,
                       arm=kernel.ARM_NAMES[ai], kappa=arm[0], nu=arm[1],
                       nref=args.nref, width=width, h=h)
            record = dict(rec, **{k: float(v) for k, v in item['diagnostics'].items()},
                          k_actual_arm=rate,
                          k_pure_arm=float(item['arm_diagnostics']['k_pure'][ai]))
            rows.append(record); records.append(rec)
            points.append(np.asarray(p)); labels.append(np.asarray(y)); arms.append(arm); hs.append(h)
            refs_f.append(f0); refs_e.append(np.asarray(reference['eH0']))
            constant, amplified, speeds, valid = [], [], [], []
            for n in HORIZONS:
                constant.append(np.asarray(p)-ETA*n*g0)
                with np.errstate(over='ignore', invalid='ignore'):
                    prediction = np.asarray(p)-ETA*exponential_sum(rate, n)*f0-ETA*n*r0
                    speed = q0 if n == 0 else q0*np.exp(ETA*n*rate)
                amplified.append(prediction); speeds.append(speed)
                valid.append(bool(np.isfinite(prediction).all() and np.isfinite(speed)))
            forecasts['constant'].append(constant); forecasts['amplification'].append(amplified)
            forecasts['q'].append(speeds); forecasts['valid'].append(valid)
            forecasts['first_step'].append(np.asarray(p)-ETA*g0)
        print(json.dumps(finite_json(dict(case=case, diagnostics=item['diagnostics']))), flush=True)
    if not records:
        raise ValueError('No cases selected')
    io.save(args.output/'inputs.npz', p=np.asarray(points), x=np.asarray(x), y=np.asarray(labels),
            reference_F0=np.asarray(refs_f), reference_eH0=np.asarray(refs_e),
            arms=np.asarray(arms), h=np.asarray(hs), cases=np.array(json.dumps(records)))
    io.save(args.output/'forecasts.npz', **{k: np.asarray(v) for k, v in forecasts.items()},
            horizons=np.asarray(HORIZONS))
    write_csv(args.output/'fork_diagnostics.csv', rows)
    io.write_json(args.output/'manifest.json', dict(source_hashes=hashes(), eta=ETA,
        input_source=str(args.inputs), input_source_sha256=io.digest(args.inputs),
        input_sha256=io.digest(args.output/'inputs.npz'),
        forecast_sha256=io.digest(args.output/'forecasts.npz'), cases=records,
        issued_utc=datetime.now(timezone.utc).isoformat(), horizons=HORIZONS,
        model='F_n=exp(eta*k_actual*n)F0; R_n=R0; direction fixed; no future fitting',
        numerical_certificate=False, kind='physical' if args.physical else 'feedback'))


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    pack = dict(np.load(args.inputs)); x = jnp.asarray(pack['x'])
    diag = jax.jit(kernel.diagnostics)
    rows = []
    for i, case in selected(pack, args):
        value = jax.device_get(diag(jnp.asarray(pack['p'][i]), x, jnp.asarray(pack['y'][i])))
        rows.append(dict(case, **{k: float(v) for k, v in value.items()}))
    write_csv(args.output/'diagnostics.csv', rows)
    io.write_json(args.output/'manifest.json', dict(input_sha256=io.digest(args.inputs),
        input_path=str(args.inputs), source_hashes=hashes(), cases=len(rows),
        scope='Checkpoint diagnostics only, not a uniform neighborhood certificate'))


def validate_backend(backend):
    if not jax.config.x64_enabled:
        raise RuntimeError('FP64 required')
    if backend == 'gpu':
        for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'CUDA_VISIBLE_DEVICES'):
            if not os.environ.get(key):
                raise RuntimeError(f'Missing allocated GPU context: {key}')
        for kind, identifier in (('job', os.environ['SLURM_JOB_ID']),
                                 ('step', os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])):
            subprocess.run(['scontrol', 'show', kind, identifier], check=True)
        if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
            raise RuntimeError('Expected exactly one allocated GPU')
    elif any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('CPU operation requested')


def run(args):
    if args.steps not in HORIZONS[1:]:
        raise ValueError('Requested horizon has no issued prediction')
    validate_backend(args.backend)
    manifest = json.loads((args.predictions/'manifest.json').read_text())
    if manifest['source_hashes'] != hashes():
        raise ValueError('Source differs from issued prediction')
    path = args.predictions/'inputs.npz'
    if io.digest(path) != manifest['input_sha256'] or io.digest(args.predictions/'forecasts.npz') != manifest['forecast_sha256']:
        raise ValueError('Issued input/prediction changed')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'snapshots').mkdir()
    pack = dict(np.load(path)); x = jnp.asarray(pack['x'])
    y, arm, h = map(jnp.asarray, (pack['y'], pack['arms'], pack['h']))
    ref = dict(F0=jnp.asarray(pack['reference_F0']), eH0=jnp.asarray(pack['reference_eH0']))
    p = jnp.asarray(pack['p']); batch, size = p.shape; width = (size-1)//3
    zero = jnp.zeros((batch, width))
    state = dict(p=p, positive=zero, negative=zero, crossing=zero,
                 signed_effective=zero, signed_tracking=zero,
                 first_hit=jnp.where(h[:, None]*jnp.abs(p[:, :width]) >= .25, 0, -1),
                 count=jnp.zeros(batch, dtype=jnp.int64), failed=jnp.zeros(batch, dtype=bool),
                 travel=jnp.zeros(batch), effective_travel=jnp.zeros(batch),
                 tracking_travel=jnp.zeros(batch), action=jnp.zeros(batch))
    def one(old, target, reference, controls, scale):
        gradient, applied, natural, remainder = kernel.step_components(old['p'], x, target, reference, controls)
        pn = old['p']-ETA*gradient
        change = scale*(jnp.abs(pn[:width])-jnp.abs(old['p'][:width]))
        sign = jnp.sign(old['p'][:width])
        eff = -ETA*scale*sign*applied[:width]
        rem = -ETA*scale*sign*remainder[:width]
        count = old['count']+1
        new = dict(p=pn, positive=old['positive']+jnp.maximum(change, 0),
                   negative=old['negative']+jnp.maximum(-change, 0),
                   crossing=old['crossing']+change-eff-rem,
                   signed_effective=old['signed_effective']+eff,
                   signed_tracking=old['signed_tracking']+rem,
                   first_hit=jnp.where((old['first_hit'] < 0)&(scale*jnp.abs(pn[:width]) >= .25), count, old['first_hit']),
                   count=count, failed=old['failed'],
                   travel=old['travel']+ETA*jnp.linalg.norm(gradient),
                   effective_travel=old['effective_travel']+ETA*jnp.linalg.norm(applied),
                   tracking_travel=old['tracking_travel']+ETA*jnp.linalg.norm(remainder),
                   action=old['action']+ETA*jnp.sum(applied*applied))
        good = ~old['failed']&jnp.all(jnp.isfinite(pn))
        for value in new.values():
            good = good&jnp.all(jnp.isfinite(value))
        new = jax.tree.map(lambda a,b: jnp.where(good, a,b), new, old)
        new['failed'] = ~good
        return new
    batched = jax.vmap(one)
    @jax.jit
    def advance(old, steps):
        return jax.lax.fori_loop(0, steps, lambda _, s: batched(s, y, ref, arm, h), old)
    def sampled(p):
        def at(pp, yy, rr, aa):
            gradient, applied, natural, rem = kernel.step_components(pp, x, yy, rr, aa)
            return jnp.stack((jnp.linalg.norm(applied), jnp.linalg.norm(natural), jnp.linalg.norm(rem),
                              jnp.linalg.norm(applied[:width]), jnp.linalg.norm(rem[:width])))
        return jax.vmap(at)(p, y, ref, arm)
    sampled = jax.jit(sampled)
    start = 0; begin = time.perf_counter()
    for end in (n for n in HORIZONS if n <= args.steps):
        state = advance(state, end-start)
        values = jax.device_get(state)
        io.save(args.output/'snapshots'/f'{end:09d}.npz', **values,
                sampled=np.asarray(sampled(state['p'])), offset=np.array(end))
        print(json.dumps(dict(horizon=end, failed=int(values['failed'].sum()),
              elapsed_seconds=time.perf_counter()-begin)), flush=True)
        start = end
    io.write_json(args.output/'manifest.json', dict(prediction_sha256=io.digest(args.predictions/'manifest.json'),
        source_hashes=hashes(), cases=manifest['cases'], steps=args.steps,
        elapsed_seconds=time.perf_counter()-begin, backend=args.backend,
        sampled_columns=['applied_force_norm', 'natural_force_norm', 'tracking_norm', 'applied_slope_norm', 'tracking_slope_norm']))


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    pack = dict(np.load(args.predictions/'inputs.npz'))
    forecast = dict(np.load(args.predictions/'forecasts.npz'))
    cases = json.loads(str(pack['cases'])); width = (pack['p'].shape[-1]-1)//3
    base = {c['source_index']: i for i,c in enumerate(cases) if c['arm'] == 'natural'}
    rows, contrasts = [], []
    for path in sorted((args.run/'snapshots').glob('*.npz')):
        n = int(path.stem)
        if n == 0:
            continue
        hi = list(forecast['horizons']).index(n)
        actual = dict(np.load(path)); states = actual['p']
        for i, case in enumerate(cases):
            motion = states[i, :width]-pack['p'][i, :width]
            norm = np.linalg.norm(motion)
            row = dict(case, horizon=n, failed=bool(actual['failed'][i]),
                       q_actual=float(actual['sampled'][i, 0]), q_predicted=float(forecast['q'][i, hi]),
                       motion_norm=float(norm), mean_lambda_change=float(case['h']*np.mean(np.abs(states[i,:width])-np.abs(pack['p'][i,:width]))),
                       mean_positive_travel=float(actual['positive'][i].mean()),
                       ever_acquired=int(np.sum(actual['first_hit'][i]>=0)),
                       initially_above=int(np.sum(actual['first_hit'][i] == 0)),
                       newly_ever_hit=int(np.sum(actual['first_hit'][i]>0)),
                       effective_travel=float(actual['effective_travel'][i]),
                       tracking_travel=float(actual['tracking_travel'][i]),
                       mean_effective_signed=float(actual['signed_effective'][i].mean()),
                       mean_tracking_signed=float(actual['signed_tracking'][i].mean()),
                       mean_crossing=float(actual['crossing'][i].mean()))
            for model in ('constant', 'amplification'):
                error = np.linalg.norm(forecast[model][i, hi, :width]-states[i, :width])
                row[model+'_error'] = float(error)
                row[model+'_relative_error'] = float(error/norm) if norm else None
            rows.append(row)
            b = base[case['source_index']]
            if i == b:
                continue
            observed = case['h']*(np.abs(states[i,:width])-np.abs(states[b,:width]))
            predicted = case['h']*(np.abs(forecast['amplification'][i,hi,:width])-np.abs(forecast['amplification'][b,hi,:width]))
            den = np.linalg.norm(observed); err = np.linalg.norm(predicted-observed)
            contrasts.append(dict(case, horizon=n, failed=bool(actual['failed'][i] or actual['failed'][b]),
                actual_mean=float(observed.mean()), predicted_mean=float(predicted.mean()),
                actual_norm=float(den), error_norm=float(err),
                skill=float(1-err/den) if den else None,
                sign_match=bool(np.sign(observed.mean()) == np.sign(predicted.mean())),
                q_ratio_to_natural=float(actual['sampled'][i,0]/actual['sampled'][b,0]) if actual['sampled'][b,0] else None,
                positive_ratio_to_natural=float(actual['positive'][i].mean()/actual['positive'][b].mean()) if actual['positive'][b].mean() else None))
    write_csv(args.output/'states.csv', rows); write_csv(args.output/'contrasts.csv', contrasts)
    io.write_json(args.output/'manifest.json', dict(predictions=io.digest(args.predictions/'manifest.json'),
        run=io.digest(args.run/'manifest.json'), source_hashes=hashes(), states=len(rows), contrasts=len(contrasts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'audit', 'run', 'analyze'))
    parser.add_argument('--inputs', type=Path); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path); parser.add_argument('--run', type=Path)
    parser.add_argument('--seed', type=int); parser.add_argument('--targets')
    parser.add_argument('--nref', type=int, default=128)
    parser.add_argument('--cohort', default='development'); parser.add_argument('--unpulsed', action='store_true')
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--steps', type=int, choices=HORIZONS[1:], default=20000)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    if args.command != 'run':
        validate_backend('cpu')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
