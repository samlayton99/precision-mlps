"""Bounded matched replays with exact per-step moment accounting; no trace arrays."""
import argparse
import csv
import json
from pathlib import Path
import time
import zipfile

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from . import population_balance_dynamics as model


def run(args):
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Numerical replays require the Modal GPU')
    with zipfile.ZipFile(args.inputs) as archive:
        if sum(p.file_size for p in archive.infolist()) > 16*1024**2:
            raise ValueError('Expected only the small prepared input capsule')
    with np.load(args.inputs, allow_pickle=False) as data:
        ps, x, ys = data['p'], data['x'], data['y']
        cases = json.loads(str(data['cases']))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    x = jnp.asarray(x)
    samples = round(args.horizon/args.sample_interval)
    if abs(samples*args.sample_interval-args.horizon) > 1e-12 or samples % 2:
        raise ValueError('Require an even number of equally spaced samples')
    all_rows, endpoints, labels, summaries = [], [], [], []
    for index, (p0, y, case) in enumerate(zip(ps, ys, cases, strict=True)):
        y = jnp.asarray(y)
        target = case['target']
        for kind, dt in (('effective', .02), ('effective', .01), ('gd', .002), ('gd', .001)):
            if abs(round(args.sample_interval/dt)*dt-args.sample_interval) > 1e-12:
                raise ValueError('Sample interval must be divisible by every time step')
            p = jnp.asarray(p0)
            ledger = jnp.zeros(len(model.LEDGER))
            initial = {k: float(v) for k,v in jax.device_get(model.moments(p)).items()}
            rows = []
            for sample in range(samples+1):
                d = {k: float(v) for k,v in jax.device_get(model.diagnostics(p,x,y)).items()}
                acc = dict(zip(model.LEDGER, map(float,jax.device_get(ledger)), strict=True))
                delta_change = sum(acc[k] for k in model.LEDGER[:5])+acc['Delta_quadratic']
                defects = [d['Delta']-initial['Delta']-delta_change,
                           d['A']-initial['A']-acc['A_linear']-acc['A_quadratic'],
                           d['C']-initial['C']-acc['C_linear']-acc['C_quadratic'],
                           d['m']-initial['m']-acc['m_linear']-acc['m_quadratic'],
                           d['M']-initial['M']-acc['M_linear']-acc['M_quadratic']]
                h = 2/case.get('nref', case.get('Nref', 512))
                row = dict(target=target, seed=case.get('seed'), start=case.get('start'),
                           case_index=index, kind=kind, dt=dt, time=sample*args.sample_interval,
                           h=h, lambda_rms=h*d['slope_rms'], **d,
                           moment_balance_error=max(map(abs,defects)),
                           **{'int_'+k:v for k,v in acc.items()})
                rows.append(row)
                if sample < samples:
                    p, ledger = model.advance(p,ledger,x,y,dt,kind,round(args.sample_interval/dt))
                if time.monotonic()-started > args.max_seconds:
                    raise RuntimeError('Reserved replay time exhausted')
            all_rows.extend(rows)
            endpoints.append(np.asarray(p)); labels.append(str((target,kind,dt)))
            summary = dict(target=target,kind=kind,dt=dt,
                           max_balance_error=max(r['moment_balance_error'] for r in rows),
                           max_bound_violation=max(-r['delta_bound_slack'] for r in rows),
                           min_abs_alignment=min(abs(r['rho']) for r in rows),
                           initial_M=initial['M'], final_M=rows[-1]['M'],
                           initial_Delta=initial['Delta'], final_Delta=rows[-1]['Delta'],
                           final_lambda_rms=rows[-1]['lambda_rms'], relative_error=rows[-1]['relative_error'])
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    with (args.output/'states.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(args.output/'endpoints.npz',p0=ps,endpoints=np.stack(endpoints),labels=np.array(labels))
    (args.output/'facts.json').write_text(json.dumps(dict(
        scope='FP64 matched replays, exact GD moment ledgers; sampled structural bounds are not interval certificates',
        cases=cases, horizon=args.horizon, sample_interval=args.sample_interval,
        seconds=time.monotonic()-started, trajectories=summaries),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--horizon',type=float,default=200.)
    parser.add_argument('--sample-interval',type=float,default=1.)
    parser.add_argument('--max-seconds',type=float,default=1500.)
    run(parser.parse_args())
