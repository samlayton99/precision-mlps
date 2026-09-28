"""Curated scalar comparisons and figures; never write report prose."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .population_accumulated_audit import stats,clean


def read_csv(path):
    with path.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            try: row[key]=float(value)
            except ValueError: pass
    return rows


def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    archive=read_csv(root/'feedback_budget_final/trajectories.csv')
    initial=json.loads((root/'initial_feedback/initial.json').read_text())['results']
    fresh={}
    for name in ('feedback_flow','feedback_flow_100k'):
        fresh[name]=dict(facts=json.loads((root/name/'facts.json').read_text()),
                        states=read_csv(root/name/'states.csv'))
    base=[r for r in archive if r['width']==705 and r['arm']=='original' and r['level']=='directional']
    facts=dict(baseline={k:stats([r[k] for r in base]) for k in (
        'initial_force','initial_relative_error','final_relative_error','budget',
        'energy_floor_over_observed','spare_feedback_budget_for_one_percent',
        'prefix_budget_over_initial_rate','final_M4_envelope_ratio','final_M4_ratio')})
    facts['interventions']=[]
    for arm in ('original','s3.2_primary','s10_primary','s32_primary','s100_primary',
                's3.2_inverse','s10_inverse','s32_inverse','s100_inverse'):
        rows=[r for r in archive if r['width']==705 and r['arm']==arm and r['eta']==.002 and r['horizon']==20000 and r['level']=='directional']
        facts['interventions'].append(dict(arm=arm,count=len(rows),
            floor_above_one_percent=sum(r['energy_relative_floor']>.01 for r in rows),
            initially_above_one_percent=sum(r['initial_relative_error']>.01 for r in rows),
            finally_above_one_percent=sum(r['final_relative_error']>.01 for r in rows),
            initial_force=stats([r['initial_force'] for r in rows]),
            budget=stats([r['budget'] for r in rows])))

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    labels=[r['case']['target'] for r in initial]
    positions=np.arange(len(labels))
    times=[r['certificate']['time_lower']/.002 for r in initial]
    floors=[r['certificate']['energy_relative_floor_lower'] for r in initial]
    axes[0].bar(positions,times,color='#167a81')
    axes[0].axhline(20000,color='black',ls='--',label='20k comparison interval')
    axes[0].axhline(100000,color='#b26028',ls=':',label='100k comparison interval')
    axes[0].set(yscale='log',ylabel='Certified flow time / 0.002')
    axes[0].legend(fontsize=8)
    axes[1].bar(positions,floors,color='#167a81')
    axes[1].axhline(.01,color='black',ls='--',label='1% requirement')
    axes[1].set(ylabel='Certified relative error floor',ylim=(0,1)); axes[1].legend(fontsize=8)
    for ax in axes: ax.set_xticks(positions,labels,rotation=22)
    fig.suptitle('Initial-state certificates: exact effective ODE, width 705, seed 30\nTime divided by learning rate is a comparison unit, not a GD certificate')
    fig.savefig(output/'initial_certificates.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for reference,color in (('primary','#167a81'),('inverse','#ad6b25')):
        groups=[facts['interventions'][0]]+[next(r for r in facts['interventions'] if r['arm']==f's{s}_{reference}') for s in ('3.2','10','32','100')]
        scales=np.array([1,3.2,10,32,100])
        axes[0].plot(scales,[r['floor_above_one_percent']/r['count'] for r in groups],'.-',color=color,label=reference)
        axes[1].plot(scales,[r['initial_force']['median'] for r in groups],'.-',color=color,label=reference)
    for ax in axes: ax.set(xscale='log',xlabel='Injected geometry multiplier'); ax.legend()
    axes[0].set(ylabel='Fraction with a >1% conditional floor',ylim=(-.05,1.05))
    axes[1].set(yscale='log',ylabel='Median initial effective-force norm')
    fig.suptitle('Large injections leave the weak-force regime\nWidth 705, 23 targets, two seeds; 20k continuation, two readout repairs')
    fig.savefig(output/'intervention_limits.png',dpi=180); plt.close(fig)

    facts['fresh']=[]
    for name,entry in fresh.items():
        summaries=entry['facts']['trajectories']; comparisons=entry['facts']['comparisons']
        eff=[r for r in summaries if r['kind']=='effective' and r['dt']==.01]
        gd=[r for r in summaries if r['kind']=='gd' and r['dt']==.002]
        item=dict(name=name,horizon=entry['facts']['flow_horizon'],
            effective={key:stats([r[key] for r in eff]) for key in ('maximum_identity_relative','maximum_coarse_drift','maximum_effective_energy_defect','max_prefix_rate_ratio','maximum_force_excess','maximum_energy_floor_excess')},
            comparisons={key:stats([r[key] for r in comparisons]) for key in comparisons[0] if key!='target'},
            tracking_floor_loss=stats([r['energy_floor']-r['tracked_energy_floor'] for r in gd]),
            tracking_derivative_over_initial_force=stats([r['total_tracking_derivative']/r['initial_force'] for r in gd]))
        facts['fresh'].append(item)
    states=fresh['feedback_flow_100k']['states']
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    for ax,target in zip(axes.flat,labels):
        eff=[r for r in states if r['target']==target and r['kind']=='effective' and r['dt']==.01]
        gd=[r for r in states if r['target']==target and r['kind']=='gd' and r['dt']==.002]
        t=np.array([r['time'] for r in eff])/.002
        initial_force=eff[0]['f']
        ax.plot(t,[r['f']/initial_force for r in eff],color='#167a81',label='Effective ODE')
        ax.plot(t,[r['f']/initial_force for r in gd],color='black',ls='--',label='GD')
        ax.plot(t,[r['force_envelope']/initial_force for r in eff],color='#ad6b25',label='Feedback envelope')
        ax.set(title=target,xlabel='Equivalent updates at learning rate 0.002',ylabel='Force / initial force')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Does force reinforce faster than the conditional envelope?\n100k comparison; the envelope uses accumulated evolving feedback, not a frozen Jacobian')
    fig.savefig(output/'fresh_force_envelopes.png',dpi=180); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    summaries=fresh['feedback_flow_100k']['facts']['trajectories']
    for kind,dt,marker,color,label in (('effective',.01,'o','#167a81','Effective ODE'),('gd',.002,'x','black','GD')):
        group=[next(r for r in summaries if r['target']==target and r['kind']==kind and r['dt']==dt) for target in labels]
        axes[0].scatter(positions,[r['relative_error'] for r in group],marker=marker,color=color,label=label)
        axes[1].scatter(positions,[r['final_M4']/r['initial_M4'] for r in group],marker=marker,color=color,label=label)
    group=[next(r for r in summaries if r['target']==target and r['kind']=='effective' and r['dt']==.01) for target in labels]
    axes[0].scatter(positions,[r['energy_floor'] for r in group],marker='_',s=120,color='#ad6b25',label='Conditional error floor')
    axes[1].scatter(positions,[r['final_M4_envelope']/r['initial_M4'] for r in group],marker='_',s=120,color='#ad6b25',label='Conditional moment bound')
    axes[0].axhline(.01,color='black',ls='--',lw=1)
    axes[0].set(ylabel='Raw relative error',ylim=(0,1))
    axes[1].set(ylabel='Fourth moment / initial fourth moment',yscale='log')
    for ax in axes: ax.set_xticks(positions,labels,rotation=22); ax.legend(fontsize=8)
    fig.suptitle('Output failure and population motion over 100k equivalent updates\nWidth 705, six targets, seed 30; no readout refitting')
    fig.savefig(output/'fresh_output_population.png',dpi=180); plt.close(fig)
    (output/'facts.json').write_text(json.dumps(clean(facts),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(args.root,args.output)
