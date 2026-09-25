"""Compare sampled structural budgets with the proved first-exit condition."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .population_balance_dynamics import comparison, C3


def cumulative(values, times):
    return np.r_[0., np.cumsum(np.diff(times)*(values[1:]+values[:-1])/2)]


def budget(rows, factor, stride=1):
    rows = rows[::stride]
    t = np.array([r['time'] for r in rows])
    m0, width, h = (rows[0][k] for k in ('M','width','h'))
    mass = m0*factor
    parts = [comparison(r, mass) for r in rows]
    force = np.array([p['force'] for p in parts])
    track = np.array([r['int_tracking_path'] for r in rows])
    path = cumulative(force,t)+track
    margin = np.sqrt(mass)-np.sqrt(m0)
    accepted = path < margin
    last = np.flatnonzero(accepted)[-1]
    # Sampled quadrature only, even for GD; exact tracking integral comes from its step ledger.
    R = np.array([r['R_norm'] if r['kind']=='gd' else 0. for r in rows])
    jn = np.array([C3*np.sqrt(r['C6'])*mass**1.5/width for r in rows])
    # The input measure is read from the matching capsule.
    bm = (np.sqrt(1+2*mass)*track+cumulative(jn*(force+R),t))/np.sqrt(rows[0]['input_variance'])
    if rows[0]['kind']=='gd':
        bm += .5*rows[0]['dt']*cumulative((force+R)**2,t)
    alignment = np.maximum(2*(abs(rows[0]['m'])-bm)/mass,0.)
    return dict(factor=factor, time=t, path=path, margin=margin, alignment=alignment,
                last_time=float(t[last]), final_ratio=float(path[-1]/margin),
                lambda_bound=float(h*np.sqrt(mass/width)),
                force=force, target=np.array([p['target'] for p in parts]),
                generated=np.array([p['force']-p['target'] for p in parts]))


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    groups = defaultdict(list)
    with args.source.open() as stream:
        for row in csv.DictReader(stream):
            parsed={k:(v if k in ('target','kind') else float(v) if v else None) for k,v in row.items()}
            groups[(row['target'],row['kind'],float(row['dt']))].append(parsed)
    with np.load(args.capsule,allow_pickle=False) as pack:
        variance=float(np.mean(pack['x']**2))
    summaries=[]; curves={}; bounds=[]
    factors=(1.02,1.05,1.1,1.2,1.5,2.,3.,4.,6.,8.)
    for key,rows in groups.items():
        rows.sort(key=lambda r:r['time'])
        for row in rows: row['input_variance']=variance
        trials=[budget(rows,f) for f in factors]
        successes=[p for p in trials if p['last_time']==rows[-1]['time']]
        chosen=min(successes,key=lambda p:p['factor']) if successes else max(trials,key=lambda p:p['last_time'])
        curves[key]=chosen
        coarse=budget(rows,chosen['factor'],2)
        t=chosen['time']; m0=rows[0]['M']
        upper=cumulative(np.array([r['Delta_upper_full' if key[1]=='gd' else 'Delta_upper_effective'] for r in rows]),t)
        if key[1]=='gd': upper += np.array([r['int_Delta_quadratic'] for r in rows])
        full_ratio=min(p['final_ratio'] for p in trials)
        summary=dict(target=key[0],kind=key[1],dt=key[2],factor=chosen['factor'],
                     valid_sampled_duration=chosen['last_time'],final_budget_ratio=chosen['final_ratio'],
                     best_full_interval_ratio=full_ratio,structural_slack_multiplier=1/full_ratio,
                     lambda_bound=chosen['lambda_bound'],lambda_final=rows[-1]['lambda_rms'],
                     mass_ratio=rows[-1]['M']/m0,
                     alignment_lower_at_valid_duration=float(chosen['alignment'][np.flatnonzero(t<=chosen['last_time'])[-1]]),
                     min_abs_alignment=min(abs(r['rho']) for r in rows),
                     Delta_change=rows[-1]['Delta']-rows[0]['Delta'],Delta_upper_change=float(upper[-1]),
                     budget_quadrature_relative_difference=abs(coarse['path'][-1]-chosen['path'][-1])/max(chosen['path'][-1],1e-300),
                     force_bound_ratio_initial=float(chosen['force'][0]/rows[0]['F_norm']),
                     target_share_initial=float(chosen['target'][0]/chosen['force'][0]),
                     max_balance_error=max(r['moment_balance_error'] for r in rows),
                     min_Delta_bound_slack=min(r['delta_bound_slack'] for r in rows),
                     relative_error=rows[-1]['relative_error'])
        summaries.append(summary)
        for p in trials:
            bounds.append(dict(target=key[0],kind=key[1],dt=key[2],factor=p['factor'],
                               valid_sampled_duration=p['last_time'],final_budget_ratio=p['final_ratio']))
    with (args.output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    with (args.output/'radius_scan.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bounds[0])); writer.writeheader(); writer.writerows(bounds)
    refinements=[]
    for target in sorted({k[0] for k in groups}):
        for kind,dt1,dt2 in (('gd',.002,.001),('effective',.02,.01)):
            if (target,kind,dt1) not in groups: continue
            a,b=groups[(target,kind,dt1)],groups[(target,kind,dt2)]
            refinements.append(dict(target=target,kind=kind,
                max_relative_M_difference=max(abs(r['M']-s['M'])/s['M'] for r,s in zip(a,b,strict=True)),
                max_relative_lambda_difference=max(abs(r['lambda_rms']-s['lambda_rms'])/s['lambda_rms'] for r,s in zip(a,b,strict=True))))
    facts=dict(scope='Sampled conditional-budget verification, not validated interval arithmetic or initial-data prediction',
               factors=list(factors),input_variance=variance,summaries=summaries,step_refinement=refinements)
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    principal=[s for s in summaries if s['kind']=='gd' and s['dt']==.002]
    fig,axes=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    colors=plt.cm.tab10(np.arange(len(principal)))
    for color,s in zip(colors,principal,strict=True):
        key=(s['target'],'gd',.002); rows=groups[key]; p=curves[key]; t=p['time']/.002/1000
        axes[0,0].plot(t,p['path']/p['margin'],color=color,label=s['target'])
        axes[0,1].plot(t,[r['lambda_rms'] for r in rows],color=color)
        axes[0,1].hlines(p['lambda_bound'],0,p['last_time']/.002/1000,color=color,linestyle='--',alpha=.65)
        axes[1,0].plot(t,[r['M']/rows[0]['M'] for r in rows],color=color)
        axes[1,1].plot(t,[abs(r['rho']) for r in rows],color=color)
        mask=p['time']<=p['last_time']
        axes[1,1].plot(t[mask],p['alignment'][mask],color=color,linestyle='--',alpha=.7)
    axes[0,0].axhline(1,color='black',linestyle=':',label='First-exit threshold')
    axes[0,0].set(ylabel='Accumulated bound / allowed radius increase',title='A. Does the structural budget close?')
    axes[0,0].legend(fontsize=8,ncol=2)
    axes[0,1].set(ylabel='Normalized slope RMS',title='B. Actual scale; conditional bound dashed')
    axes[1,0].set(ylabel='Total hidden squared norm / starting value',title='C. Growth is allowed')
    axes[1,1].set(ylabel='Absolute slope–readout alignment',title='D. Actual alignment; derived lower bound dashed')
    for ax in axes.flat: ax.set_xlabel('Additional GD updates (thousands)'); ax.grid(alpha=.2)
    fig.savefig(args.output/'population_comparison.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    labels=[s['target'] for s in principal]; x=np.arange(len(labels)); bottom=np.zeros(len(labels))
    # Keep each signed contribution visible; do not stack unlike signs on a shared baseline.
    for i,name in enumerate(('generated','target','higher','compensation','tracking')):
        values=[groups[(s['target'],'gd',.002)][-1]['int_'+name] for s in principal]
        axes[0].bar(x+(i-2)*.15,values,width=.15,label=name)
    axes[0].axhline(0,color='black',linewidth=.7); axes[0].set_xticks(x,labels,rotation=20)
    axes[0].set(ylabel='Contribution to imbalance change',title='Signed balance over 100k additional GD updates')
    axes[0].legend(fontsize=7,ncol=2)
    axes[1].bar(x-.17,[s['Delta_change'] for s in principal],.34,label='Actual')
    axes[1].bar(x+.17,[s['Delta_upper_change'] for s in principal],.34,label='Analytic upper allowance')
    axes[1].set_xticks(x,labels,rotation=20); axes[1].set(ylabel='Imbalance change',title='Positive-growth allowance includes compensation')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'signed_growth.png',dpi=180); plt.close(fig)
    print(json.dumps(dict(principal=principal,refinement=refinements),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','capsule','output'): parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
