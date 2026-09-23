"""Post-development, checkpoint-derived Adam feedback models and phase audit.

No future state enters either prediction. The optional phase audit reads actual
saved development states only to diagnose the earlier imposed-phase closure.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import adam_forces as af, mechanism_adam as ma
from .mechanism_adam_analyze import skill, ratio, write_csv
from .mechanism_pulses import output_curvature
from . import persistence_theory as pt


@jax.jit
def z_derivative(p, x, y):
    def z(point):
        g, _, jc, ec = af.field(point, x, y)
        return af.split(g, jc, ec)[1]['z']
    return z(p), jax.jacrev(z)(p)


def derivatives(p, x, y):
    tensors = pt.tensors(p, x, y)
    g, _, jc, ec = af.field(jnp.asarray(p), jnp.asarray(x), jnp.asarray(y))
    channels, info = af.split(g, jc, ec)
    if not bool(info['resolved']):
        raise ValueError('Unresolved full-complement coarse balance')
    z, dz = map(np.asarray, z_derivative(jnp.asarray(p), jnp.asarray(x), jnp.asarray(y)))
    q = np.column_stack((np.ones_like(x), x/np.sqrt(np.mean(x*x))))
    dq = np.asarray(jc).T@dz + output_curvature(p, x, q@z)
    return dict(F=np.asarray(channels[0]), Q=np.asarray(channels[1]),
                DQ=dq, DF=tensors['hessian']-dq, H=tensors['hessian'], Dz=dz)


def predict(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Do not overwrite issued feedback forecasts')
    pack = dict(np.load(args.inputs)); cases = json.loads(str(pack['cases']))
    representatives = [i for i, c in enumerate(cases) if c['alpha_m'] == c['alpha_v'] == 1]
    systems = {}
    for i in representatives:
        systems[cases[i]['target']] = derivatives(pack['p'][i], pack['x'], pack['y'][i])
    arrays = {key: np.stack([systems[c['target']][key] for c in cases])
              for key in ('F', 'Q', 'DQ', 'DF', 'H', 'Dz')}
    ma.save(args.output/'derivatives.npz', **arrays)
    f0, q0, dq, df, p0, alpha = map(jnp.asarray,
        (arrays['F'], arrays['Q'], arrays['DQ'], arrays['DF'], pack['p'], pack['alpha']))
    width = (p0.shape[-1]-1)//3
    state0 = {k: jnp.asarray(pack[k]) for k in ('p', 'm', 'v', 'count')}
    state0['valid'] = jnp.ones(len(cases), dtype=bool)
    def advance(state, effective_feedback, start, end):
        def body(_, old):
            displacement = old['p']-p0
            q = q0+jnp.einsum('bij,bj->bi', dq, displacement)
            f = f0+jnp.where(effective_feedback,
                jnp.einsum('bij,bj->bi', df, displacement), jnp.zeros_like(f0))
            gm = (f+q).at[:, :width].add((alpha[:, 0, None]-1)*q[:, :width])
            gv = (f+q).at[:, :width].add((alpha[:, 1, None]-1)*q[:, :width])
            count = old['count']+1
            m = .9*old['m']+.1*gm
            v = .999*old['v']+.001*gv**2
            p = old['p']-.002*(m/(1-.9**count[:, None]))/(jnp.sqrt(v/(1-.999**count[:, None]))+1e-8)
            valid = old['valid'] & jnp.all(jnp.isfinite(p)&jnp.isfinite(v), axis=1)
            return dict(p=p, m=m, v=v, count=count, valid=valid)
        return jax.lax.fori_loop(start, end, body, state)
    advance = jax.jit(advance)
    forecasts, validity = {}, {}
    for model, feedback in (('tracking_closed', False), ('both_closed', True)):
        state = state0; start = 0
        for end in sorted({k for k in ma.SAVED if 0 < k <= args.horizon} | {args.horizon}):
            state = advance(state, feedback, start, end)
            forecasts[f'{model}_p_{end}'] = np.asarray(state['p'])
            forecasts[f'{model}_valid_{end}'] = np.asarray(state['valid'])
            validity[f'{model}_{end}'] = int(np.asarray(state['valid']).sum())
            start = end
    ma.save(args.output/'forecasts.npz', **forecasts)
    ma.write_json(args.output/'manifest.json', dict(input_sha256=ma.digest(args.inputs),
        horizon=args.horizon, issued_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=ma.source_hashes() | {Path(__file__).name: ma.digest(__file__),
            'mechanism_pulses.py': ma.digest(Path(__file__).with_name('mechanism_pulses.py')),
            'persistence_theory.py': ma.digest(Path(pt.__file__))},
        derivatives_sha256=ma.digest(args.output/'derivatives.npz'),
        forecasts_sha256=ma.digest(args.output/'forecasts.npz'), valid_counts=validity,
        scope='Model introduced after seed0 outcomes; derivatives and predictions use checkpoint only. No fitting to future states.'))


def compare(args):
    pack = dict(np.load(args.inputs)); cases = json.loads(str(pack['cases']))
    predictions = np.load(args.predictions/'forecasts.npz')
    baseline = {c['target']: i for i, c in enumerate(cases) if c['alpha_m'] == c['alpha_v'] == 1}
    width = (pack['p'].shape[-1]-1)//3
    rows = []
    for path in sorted((args.run/'snapshots').glob('*.npz')):
        horizon = int(path.stem)
        if not horizon:
            continue
        actual = np.load(path)['p']
        for model in ('tracking_closed', 'both_closed'):
            key = f'{model}_p_{horizon}'
            if key not in predictions:
                continue
            pred = predictions[key]
            for i, c in enumerate(cases):
                b = baseline[c['target']]
                if i == b:
                    continue
                valid = bool(predictions[f'{model}_valid_{horizon}'][i] and predictions[f'{model}_valid_{horizon}'][b])
                a = ma.H*(abs(actual[i, :width])-abs(actual[b, :width]))
                v = ma.H*(abs(pred[i, :width])-abs(pred[b, :width]))
                rows.append(dict(target=c['target'], seed=c['seed'], alpha_m=c['alpha_m'],
                    alpha_v=c['alpha_v'], horizon=horizon, model=model, valid=valid,
                    actual_mean_effect=float(a.mean()), predicted_mean_effect=float(v.mean()) if valid else None,
                    actual_effect_norm=float(np.linalg.norm(a)),
                    error_norm=float(np.linalg.norm(v-a)) if valid else None,
                    vector_skill=skill(a, v) if valid else None,
                    mean_sign_match=bool(np.sign(a.mean()) == np.sign(v.mean())) if valid else None))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'contrasts.csv', rows)
    summaries = []
    for h in sorted({r['horizon'] for r in rows}):
        for model in ('tracking_closed', 'both_closed'):
            for am, av in ma.ARMS[1:]:
                group = [r for r in rows if (r['horizon'], r['model'], r['alpha_m'], r['alpha_v']) == (h, model, am, av)]
                values = [r['vector_skill'] for r in group if r['vector_skill'] is not None]
                summaries.append(dict(horizon=h, model=model, alpha_m=am, alpha_v=av,
                    cases=len(group), valid=sum(r['valid'] for r in group),
                    median_skill=float(np.median(values)) if values else None,
                    beating_zero=sum(v > 0 for v in values),
                    sign_matches=sum(r['mean_sign_match'] is True for r in group)))
    ma.write_json(args.output/'summary.json', dict(rows=len(rows), summaries=summaries))


def phase_audit(args):
    """Retrospective sparse phase diagnostics; never supplied to predict()."""
    pack = dict(np.load(args.inputs)); cases = json.loads(str(pack['cases']))
    original = np.load(args.predictions/'driver_phases.npz')
    x = jnp.asarray(pack['x']); width = (pack['p'].shape[-1]-1)//3
    def pair(state, y, alpha):
        g, _, jc, ec = af.field(state['p'], x, y)
        ch = af.split(g, jc, ec)[0]
        next_state = ma.moment_step(state, ch, alpha, gradient=g)[0]
        gg, _, jj, ee = af.field(next_state['p'], x, y)
        return ch, af.split(gg, jj, ee)[0]
    pair = jax.jit(jax.vmap(pair))
    rows = []
    for path in sorted((args.run/'snapshots').glob('*.npz')):
        state = {k: jnp.asarray(v) for k, v in dict(np.load(path)).items()}
        ch0, ch1 = map(np.asarray, pair(state, jnp.asarray(pack['y']), jnp.asarray(pack['alpha'])))
        for i, c in enumerate(cases):
            for channel, label in enumerate(('effective', 'tracking')):
                v0, v1 = ch0[i, channel, :width], ch1[i, channel, :width]
                ref0, ref1 = original['phase_zero'][i, channel, :width], original['phase_one'][i, channel, :width]
                rows.append(dict(target=c['target'], seed=c['seed'], alpha_m=c['alpha_m'], alpha_v=c['alpha_v'],
                    horizon=int(path.stem), channel=label,
                    phase_cosine=ratio(v0@v1, np.linalg.norm(v0)*np.linalg.norm(v1)),
                    actual_even_norm=float(np.linalg.norm(v0)),
                    fork_even_norm=float(np.linalg.norm(ref0)),
                    phase_pair_drift=float(np.linalg.norm(np.stack((v0-ref0, v1-ref1)))),
                    mean_squared_driver=float(np.mean(np.stack((v0, v1))**2)),
                    mean_FQ_cross_term=float(np.mean(ch0[i, 0, :width]*ch0[i, 1, :width]
                        +ch1[i, 0, :width]*ch1[i, 1, :width])),
                    next_phase='virtual_same_arm_update'))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'sampled_phase_drift.csv', rows)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('predict', 'compare', 'phase_audit'))
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--horizon', type=int, default=2000)
    args = p.parse_args()
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('Use FP64 CPU Slurm')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
