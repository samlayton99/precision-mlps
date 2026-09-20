"""Small, traceable result tables and intervention figures, without report prose."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);args=p.parse_args()
    records=json.loads((args.analysis/'summary.json').read_text())
    primary=[r for r in records if r['config'].get('implementation')=='primed_guard_v2' and 'parent' not in r['config']]
    comparisons=[]
    for n in sorted({r['config']['n'] for r in primary}):
        for coord in ('individual','neighbor'):
            for target in ('sine','mixed'):
                chosen=[r for r in primary if r['config']['n']==n and r['config']['coordinates']==coord and r['config']['target']==target and r['config']['seed']>=2]
                by={(r['config']['policy'],r['config']['seed']):r for r in chosen}
                for policy in ('adaptive','periodic'):
                    pairs=[]
                    for seed in (2,3,4):
                        if ('baseline',seed) not in by or (policy,seed) not in by:continue
                        base,arm=by['baseline',seed],by[policy,seed]
                        pairs.append(dict(seed=seed,baseline=base['id'],arm=arm['id'],
                            equal_20k=base['latest']['step']==arm['latest']['step']==20000,
                            baseline_step=base['latest']['step'],arm_step=arm['latest']['step'],
                            baseline_status=base['latest']['status'],arm_status=arm['latest']['status'],
                            baseline_mse=base['latest']['train_mse'],arm_mse=arm['latest']['train_mse'],
                            mse_ratio=arm['latest']['train_mse']/base['latest']['train_mse'],
                            validation_ratio=arm['latest']['validation_mse']/base['latest']['validation_mse'],
                            window_ratio=arm['last_2048_mean_mse']/base['last_2048_mean_mse'] if arm['last_2048_mean_mse'] and base['last_2048_mean_mse'] else None))
                    complete=[q for q in pairs if q['equal_20k']]
                    comparisons.append(dict(n=n,coordinates=coord,target=target,policy=policy,pairs=pairs,
                        complete_pairs=len(complete),geometric_mean_ratio=float(np.exp(np.mean(np.log([q['mse_ratio'] for q in complete])))) if complete else None))
    probes=[]
    for path in sorted(args.root.glob('*/probe_curvature_*.json')):
        row=json.loads(path.read_text());config=json.loads((path.parent/'case.json').read_text())
        row.update(case=path.parent.name,config=config);probes.append(row)
    compact=[{k:v for k,v in r.items() if k!='curve'} for r in records]
    memory=[]
    for path in sorted(args.root.glob('*/memory_*.json')):
        row=json.loads(path.read_text());valid=min(row['replay_valid_steps'])
        samples=row.pop('cosine_at_1_10_100_512')
        row['valid_cosine_at_1_10_100_512']=[float(v) if position<=valid else None for position,v in zip((1,10,100,512),samples)]
        memory.append(dict(case=path.parent.name,**row))
    out=dict(horizon=20000,cases=compact,confirmation=comparisons,curvature_probes=probes,memory=memory)
    (args.analysis/'findings.json').write_text(json.dumps(out,indent=2)+'\n')
    if probes:
        fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
        labels=[]
        for i,row in enumerate(probes):
            arms={a['arm']:a for a in row['arms']};base,mix=arms['history_only'],arms['metric_mix']
            labels.append(f"{row['config']['target']}, s{row['config']['seed']}, {row['step']:,}")
            axes[0].scatter(mix['native_step_norm']/base['native_step_norm'],i,color='#0072b2',s=22)
            axes[0].scatter(mix['geometry_step_norm']/base['geometry_step_norm'],i,color='#d55e00',marker='x',s=24)
            if base['unit_true_curvature']>0 and mix['unit_true_curvature']>0:
                axes[1].scatter(mix['unit_true_curvature']/base['unit_true_curvature'],i,color='#555555',s=22)
        for ax in axes:ax.set_xscale('log');ax.axvline(1,color='gray',ls='--',lw=.8);ax.grid(axis='x',alpha=.2)
        axes[0].set_yticks(range(len(labels)),labels,fontsize=8);axes[0].set_xlabel('Accepted movement: mixture / unchanged metric')
        axes[0].scatter([],[],color='#0072b2',label='Whole native step');axes[0].scatter([],[],color='#d55e00',marker='x',label='Bandwidth step');axes[0].legend(fontsize=8)
        axes[1].set_xlabel('Unit-direction curvature: mixture / unchanged metric')
        fig.suptitle('Same-state interventions at the 17 selection switching points')
        fig.tight_layout();fig.savefig(args.analysis/'intervention_curvature.png',dpi=180);plt.close(fig)
    reset=[r for r in records if 'reinitialization' in r['config']]
    if reset:
        modes=['none','state_only','zero_physical','scaled_physical','scaled_bandwidth']
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
        for r in reset:
            c=r['config'];j=['sine','mixed'].index(c['target']);i=modes.index(c['reinitialization'])
            baseline=next(q for q in reset if q['config']['parent']==c['parent'] and q['config']['reinitialization']=='none')
            axes[j].scatter(i+(.08 if c['seed'] else -.08),r['latest']['train_mse']/baseline['latest']['train_mse'],marker='o' if c['seed']==0 else 's',color='#0072b2' if c['seed']==0 else '#d55e00')
        for ax,title in zip(axes,('Sine','Mixed sine')):
            ax.set_yscale('log');ax.axhline(1,color='gray',ls='--');ax.set_title(title);ax.set_ylabel('Final MSE / continued-parent MSE')
            ax.set_xticks(range(5),['Continue','State only','Zero w\nsmall γ','Scaled w\nsmall γ','Scaled w\nbroader λ']);ax.grid(axis='y',alpha=.2)
        axes[0].scatter([],[],marker='o',color='#0072b2',label='Seed 0');axes[0].scatter([],[],marker='s',color='#d55e00',label='Seed 1');axes[0].legend()
        fig.suptitle('One replacement event after 5k updates; 20k continuation or numerical failure')
        fig.tight_layout();fig.savefig(args.analysis/'neuron_reset.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
