"""Replay post-transient GD states; retain scalars and six sparse states only."""
import argparse
import csv
import json
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, effective_feedback_holdout as holdout, targets
from . import population_window as model, population_balance_dynamics as balance
from . import mechanism_persistence_kernel as kernel

SIX = ('moment5', 'mixed_sine', 'gauss_left', 'bump_right', 'step_right', 'kink_abs')
ALL = (*af.TARGETS, *holdout.TARGETS)
INPUT = Path('results/checkpoint_D_optimizers/expD34_readout_race/mechanism_refinement/widths/inputs')
SHAPES = ('q3','q5','n3','n5','j3','j5','g3','g5','C4','C6','C8','C10','C14','target_fine')


def independent_initial(nref, halo, seed):
    """Exact C05/D24 Xavier streams, checked bitwise against archived inputs.

    Avoid importing the unrelated Torch models and plotting modules. The two
    streams and draw ordering are those in C05 xavier_draw and D24 initial_arrays.
    """
    width = nref+2*halo+1
    bound = float(np.sqrt(6./(width+1)))
    geometry = np.random.default_rng([seed, nref])
    readout = np.random.default_rng([seed, nref, 24])
    return np.r_[geometry.uniform(-bound,bound,width), geometry.uniform(-bound,bound,width),
                 readout.uniform(-bound,bound,width), 0.]


def prepare(study):
    if study == 'development': requested = [(512,30,t) for t in SIX]
    elif study == 'panel30': requested = [(512,30,t) for t in ALL if t not in SIX]
    elif study == 'panel31': requested = [(512,31,t) for t in ALL]
    elif study == 'confirmation': requested = [(512,33,t) for t in SIX]
    elif study == 'widths': requested = [(n,31,t) for n in (128,1024) for t in SIX]
    elif study == 'refinement': requested = [(512,30,t) for t in ('moment5','gauss_left','step_right','kink_abs')]
    else: raise ValueError(study)
    cached = {}
    for nref in sorted({n for n,_,_ in requested}):
        halo = {128:24,512:96,1024:192}[nref]
        with np.load(INPUT/f'N{nref}.npz', allow_pickle=False) as data:
            cases = json.loads(str(data['cases']))
            for p, case in zip(data['p'], cases, strict=True):
                np.testing.assert_array_equal(p, independent_initial(nref,halo,case['seed']))
        with np.load(INPUT/f'N{nref}_fork20000.npz', allow_pickle=False) as data:
            for p, case in zip(data['p'], json.loads(str(data['cases'])), strict=True):
                cached[(nref,case['seed'],case['target'])] = p.copy()
    for nref, seed, target in requested:
        halo = {128:24,512:96,1024:192}[nref]
        data = holdout.data if target in holdout.TARGETS else af.data
        x,y,_,scale = data(target)
        xe,ye,_,_ = data(target,8192)
        p = cached.get((nref,seed,target))
        cold = p is None
        if cold: p = independent_initial(nref,halo,seed)
        yield dict(nref=nref,width=nref+2*halo+1,seed=seed,target=target,start=20000,
                   target_scale=scale,eta=.002,cold=cold), p,x,y,xe,ye


@jax.jit
def evaluation(p, x, y):
    return jnp.sqrt(jnp.mean((kernel.output(p,x)-y)**2)/jnp.mean(y*y))


def run(args):
    if jax.default_backend() != 'gpu': raise RuntimeError('Use Modal GPU')
    begun = time.monotonic()
    rows, summaries, saved = [], [], {}
    offsets = sorted(set(range(0,10001,100)) | set(range(10500,110001,500)))
    for case_index, (case,p0,x,y,xe,ye) in enumerate(prepare(args.study)):
        x,y,xe,ye = map(jnp.asarray,(x,y,xe,ye))
        p0 = jnp.asarray(p0)
        if case['cold']:
            p0 = model.advance(p0,x,y,.002,20000,'gd')
        variants = [('gd',.002)]
        if args.study == 'development': variants += [('effective',.02)]
        if args.study == 'refinement': variants = [('gd',.001),('effective',.01)]
        for kind,dt in variants:
            p = p0
            start = time.monotonic()
            case_rows = []
            for index,offset in enumerate(offsets):
                if index:
                    steps = round(.002*(offset-offsets[index-1])/dt)
                    p = model.advance(p,x,y,dt,steps,kind)
                d = {k:float(v) for k,v in jax.device_get(model.diagnostics(p,x,y)).items()}
                if not all(np.isfinite(v) for v in d.values()) or not d['force_resolved']:
                    raise RuntimeError(f'Unresolved force/geometry: {case}, {offset}, {d}')
                structural = jax.device_get(balance.diagnostics(p,x,y))
                d.update({k:float(structural[k]) for k in SHAPES})
                row = dict(study=args.study,case_index=case_index,**case,kind=kind,dt=dt,
                           offset=offset,time=.002*offset,h=2/case['nref'],
                           lambda_rms=2/case['nref']*d['slope_rms'],
                           relative_eval_error=float(evaluation(p,xe,ye)),**d)
                rows.append(row); case_rows.append(row)
                if offset in (0,2000,5000,10000,60000,110000):
                    saved[f'case{case_index}_{kind}_{offset}'] = np.asarray(p)
                if time.monotonic()-begun > args.max_seconds:
                    raise RuntimeError('Reserved GPU budget exhausted')
            summary = dict(**case,kind=kind,dt=dt,seconds=time.monotonic()-start,
                           q_initial=case_rows[0]['q'],q_final=case_rows[-1]['q'],
                           kappa_initial=case_rows[0]['kappa'],kappa_final=case_rows[-1]['kappa'],
                           final_lambda_rms=case_rows[-1]['lambda_rms'],
                           final_relative_error=case_rows[-1]['relative_eval_error'],
                           maximum_rate_identity_error=max(abs(r['rate_identity_error']) for r in case_rows))
            summaries.append(summary)
            print(json.dumps(summary),flush=True)
    args.output.mkdir(exist_ok=True,parents=True)
    with (args.output/'states.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output/'sparse_states.npz',**saved)
    (args.output/'facts.json').write_text(json.dumps(dict(study=args.study,
        clock='physical time=.002*offset; refined GD has twice as many actual updates',
        calibration_offsets=[2000,5000,10000],primary_calibration=5000,
        calibration_rule='Primary: endpoint levels plus twice window total variation per force travel; baseline: fixed prefix allowances; no future fitting',
        final_offset=110000,sample_offsets=offsets,device=str(jax.devices()[0]),
        target_normalization='original 2048 midpoint grid; unchanged at 8192 evaluation points',
        initialization='bitwise verified C05/D24 streams; fresh confirmation seed 33',
        seconds=time.monotonic()-begun,trajectories=summaries),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    run(parser.parse_args())
