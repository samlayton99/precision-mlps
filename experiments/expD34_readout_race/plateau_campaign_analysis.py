"""Tables and figures for the completed campaign; never generate report prose."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import adam_analyze as aa, adam_forces as af, adam_run as ar
from . import plateau, plateau_dense, plateau_followup as follow, plateau_results
from .run import write_json


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def group_summary(data, keys, metrics):
    groups = sorted({tuple(r[k] for k in keys) for r in data})
    result = []
    for group in groups:
        selected = [r for r in data if tuple(r[k] for k in keys) == group]
        record = dict(zip(keys, group), cases=len(selected))
        for metric in metrics:
            values = np.array([float(r[metric]) for r in selected])
            record[metric+'_median'] = float(np.median(values))
            record[metric+'_min'] = float(np.min(values))
            record[metric+'_max'] = float(np.max(values))
        result.append(record)
    return result


def summarize(analysis, output):
    output.mkdir(parents=True, exist_ok=True)
    endpoints = rows(analysis/'endpoints.csv')
    contrasts = rows(analysis/'contrasts.csv')
    for r in contrasts:
        r['control'] = r['bundle'].split('_', 1)[0] if r['stage']=='checks' else 'none'
    tables = dict(
        long=group_summary([r for r in endpoints if r['stage']=='long'], ['optimizer','target'],
            ['eval_relative_mse','mean_gamma','gamma_change','fraction_gamma_ge_1',
             'effective_force_ratio','effective_force_cosine','signed_effective','signed_tracking']),
        probes=group_summary([r for r in endpoints if r['stage'] in ('probes','confirm-probes')],
            ['stage','target','fork','arm','weight'],
            ['eval_relative_mse','gamma_change','reference_effective_norm','force_coherence']),
        contrasts=group_summary(contrasts,
            ['stage','control','target','fork','arm','weight'],
            ['gamma_change_difference','eval_relative_mse_difference','reference_effective_norm_difference']),
    )
    if (analysis/'forecasts.csv').exists():
        tables['forecasts'] = group_summary(rows(analysis/'forecasts.csv'),
            ['target','issued','method'], ['relative_vector_error','slope_vector_error'])
    for name, data in tables.items():
        if data:
            aa.write_csv(output/f'{name}_summary.csv', data)
    write_json(output/'summary.json', tables)
    print(json.dumps(dict(cases=len(endpoints), failed=sum(r['failed']=='True' for r in endpoints),
              incomplete=sum(r['complete']!='True' for r in endpoints),
              maximum_motion_identity=max(float(r['motion_identity']) for r in endpoints))), flush=True)


def dense(source, output):
    starts = (600000, 1100000, 3000000, 6000000)
    adapted = output/'inputs'
    for seed in range(3):
        folder = adapted/f'primary_{seed}'
        folder.mkdir(parents=True, exist_ok=False)
        states = []
        cases = []
        for step in starts:
            pair = [follow.extract(source/'long'/optimizer, seed, af.TARGETS, step)
                    for optimizer in ('gd','adam')]
            cases = pair[0][0]+pair[1][0]
            states.append({k:np.concatenate([p[1][k] for p in pair]) for k in pair[0][1]})
        write_json(folder/'manifest.json', dict(cases=cases, source=str(source),
                   source_hashes={o:follow.sha(source/'long'/o/'snapshots.npz') for o in ('gd','adam')}))
        ar.atomic_npz(folder/'snapshots.npz', steps=np.array(starts),
                      **{k:np.stack([s[k] for s in states],axis=1) for k in states[0]})
    plateau_dense.run(adapted, output, 512, 3, starts)
    data = rows(output/'windows.csv')
    aa.write_csv(output/'summary.csv', group_summary(data,['optimizer','channel','start'],
                 ['lag1_cosine','lag2_cosine','vector_coherence','norm_cv']))


def score_late(root, output):
    output.mkdir(parents=True, exist_ok=True)
    plateau_results.run(root, output)
    predictions = []
    for seed in range(3):
        folder = root/'probes'/f'seed{seed}_{follow.START}'
        if not (folder/'state.npz').exists():
            continue
        status = json.loads((folder/'status.json').read_text())
        if not status['complete']:
            continue
        manifest = json.loads((folder/'manifest.json').read_text())
        parent = root/'inputs'/f'primary_{seed}'
        source_cases = json.loads((parent/'manifest.json').read_text())['cases']
        with np.load(parent/f'forecast_{follow.START}.npz') as forecast, np.load(folder/'state.npz') as state:
            for j, index in enumerate(forecast['indices']):
                case = source_cases[int(index)]
                i = next(i for i,c in enumerate(manifest['cases']) if c['target']==case['target'] and c['arm']=='joint')
                p = state['p'][i]; x,y,_,_ = af.data(case['target'])
                residual = plateau.prediction(jnp.asarray(p),jnp.asarray(x))-y
                actual = np.asarray(plateau.effective(jnp.asarray(p),residual,jnp.asarray(x)))[:177]
                for method in ('frozen','constant'):
                    pred = forecast[method+'_force'][j]
                    slope = forecast['frozen_p'][j,:177] if method=='frozen' else forecast['constant_a'][j]
                    predictions.append(dict(target=case['target'],seed=seed,issued=follow.START,
                        end=follow.START+follow.HORIZON,method=method,failed=bool(state['failed'][i]),
                        relative_vector_error=float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)),
                        slope_vector_error=float(np.linalg.norm(slope-p[:177]))))
    if predictions:
        aa.write_csv(output/'forecasts.csv', predictions)
    summarize(output, output/'summary')


def figures(source, analysis, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for suffix in ('png','pdf'):
            fig.savefig(output/f'{name}.{suffix}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(3,5,figsize=(16,9),layout='constrained')
    for optimizer, color in (('gd','C0'),('adam','C1')):
        folder=source/'long'/optimizer
        cases=json.loads((folder/'manifest.json').read_text())['cases']
        with np.load(folder/'trace.npz') as trace:
            for ax,target in zip(axes.flat,af.TARGETS):
                for i,c in enumerate(cases):
                    if c['target']==target:
                        ax.semilogy(trace['ends']/1e6,trace['values'][i,:,ar.METRICS.index('raw_effective_norm')],
                                    color=color,alpha=.8 if optimizer=='gd' else .3,
                                    linewidth=1.1 if optimizer=='gd' else .5,zorder=3 if optimizer=='gd' else 1,
                                    label=optimizer if c['seed']==0 else None)
                ax.set(title=target,xlabel='Updates (millions)',ylabel='Effective slope-force norm')
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(af.TARGETS):]: ax.set_visible(False)
    fig.suptitle('600k to 6 million updates: all 13 targets and seeds 0–2')
    save(fig,'05_long_force_trajectories')
    data=rows(analysis/'contrasts.csv')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    arms=('freeze_readout','clamp_fine','no_slope_tracking')
    titles=('Freeze readout','Clamp fine residual','Remove slope tracking')
    labels=('Degree 3','Degree 4','Degree 5','Degree 9','Mixed sine','Localized sine','Chirp')
    for col,arm in enumerate(arms):
        for row,metric,label in ((0,'gamma_change_difference','Change in mean |slope| versus joint GD'),
                                  (1,'eval_relative_mse_difference','Evaluation relative MSE versus joint GD')):
            ax=axes[row,col]
            for shift,stage,color in ((-.13,'probes','C0'),(.13,'confirm-probes','C1')):
                for k,target in enumerate(follow.pp.ANCHORS):
                    selected=[r for r in data if r['stage']==stage and r['target']==target and r['arm']==arm and int(r['fork'])==600000]
                    ax.scatter(np.full(len(selected),k+shift),[float(r[metric]) for r in selected],
                               color=color,alpha=.65,s=22,
                               label=('Discovery (3 seeds)' if stage=='probes' else 'Confirmation (5 seeds)') if k==0 else None)
            ax.axhline(0,color='gray',lw=.8)
            if col==2:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            else:ax.set_yscale('symlog',linthresh=1e-5)
            ax.set_xticks(range(7),labels,rotation=45,ha='right')
            ax.set(title=titles[col],ylabel=label.replace(' versus ','\nversus '))
    axes[0,0].legend();fig.suptitle('Paired interventions: fork at 600k, evaluate at 1.1m')
    save(fig,'06_intervention_contrasts')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('summary','dense','late','figures'))
    p.add_argument('--source',type=Path)
    p.add_argument('--analysis',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.phase=='summary':summarize(args.analysis,args.output)
    elif args.phase=='dense':dense(args.source,args.output)
    elif args.phase=='late':score_late(args.source,args.output)
    else:figures(args.source,args.analysis,args.output)


if __name__=='__main__': main()
