"""Analyze the denominator intervention and its matched scalar-gain control."""
import argparse
import json
from pathlib import Path

import numpy as np

from .population_adam_motion_analysis import read,stats,LABELS
from .population_adam_run import write_csv,SIX


def analyze(args):
    rows=read(args.inputs/'states.csv');runs=read(args.inputs/'runs.csv')
    keys=('target','width','seed','arm')
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),{})[r['offset']]=r
    complete={tuple(r[k] for k in keys) for r in runs if r['complete']=='True'}
    native=[];contrasts=[]
    for key,lookup in groups.items():
        if key not in complete:continue
        first=lookup[0]
        if key[-1]=='native':
            native.append(dict(**{k:first[k] for k in keys[:-1]},initial_gain=first['instant_variance_gain'],
                mean_gain=lookup[10000]['mean_variance_gain'],mean_capped_gain=lookup[10000]['mean_capped_gain'],
                cap_fraction=lookup[10000]['gain_cap_fraction']))
            continue
        baseline=groups.get((*key[:-1],'native'))
        if baseline is None or (*key[:-1],'native') not in complete:continue
        for off in (10000,20000):
            r,b=lookup[off],baseline[off]
            contrasts.append(dict(**{k:first[k] for k in keys},offset=off,
                slope_effect_percent=100*(r['slope_rms']/b['slope_rms']-1),lambda_rms=r['lambda_rms'],
                lambda_difference=r['lambda_rms']-b['lambda_rms'],
                error_effect_percent=100*(r['relative_error']/b['relative_error']-1),
                error=r['relative_error'],native_error=b['relative_error'],
                fine_path_ratio=r['fine_slope_path']/b['fine_slope_path'],
                fine_A_over_start=r['A_fine']/first['A'],native_fine_A_over_start=b['A_fine']/first['A'],
                tracking_A_over_start=r['A_tracking']/first['A'],native_tracking_A_over_start=b['A_tracking']/first['A'],
                fine_slope_coherence=r['slope_coherence_10000'],native_fine_slope_coherence=b['slope_coherence_10000'],
                fine_radial_cancellation=r['fine_radial_cancellation'],native_fine_radial_cancellation=b['fine_radial_cancellation'],
                fine_A_change_over_start=(r['A_fine']-b['A_fine'])/first['A'],
                tracking_A_change_over_start=(r['A_tracking']-b['A_tracking'])/first['A'],
                mean_gain=r['mean_variance_gain'],mean_capped_gain=r['mean_capped_gain'],cap_fraction=r['gain_cap_fraction']))
    write_csv(args.output/'native_denominator_gains.csv',native)
    write_csv(args.output/'denominator_contrasts.csv',contrasts)
    facts=dict(runs=len(runs),complete_runs=len(complete),native_cases=len(native),
        native={k:stats([r[k] for r in native]) for k in ('initial_gain','mean_gain','mean_capped_gain','cap_fraction')},arms={})
    facts['native_by_target']={t:{k:stats([r[k] for r in native if r['target']==t]) for k in
        ('initial_gain','mean_gain','mean_capped_gain','cap_fraction')} for t in SIX}
    for arm in ('gain_only','tracking_attenuated_variance'):
        for off in (10000,20000):
            rr=[r for r in contrasts if r['arm']==arm and r['offset']==off]
            facts['arms'][f'{arm}_{off}']={k:stats([r[k] for r in rr]) for k in
                ('slope_effect_percent','lambda_rms','error_effect_percent','fine_path_ratio','fine_A_change_over_start','tracking_A_change_over_start','cap_fraction')}
            facts['arms'][f'{arm}_{off}'].update(slope_positive=sum(r['slope_effect_percent']>0 for r in rr),
                error_better=sum(r['error_effect_percent']<0 for r in rr),
                both_better=sum(r['slope_effect_percent']>0 and r['error_effect_percent']<0 for r in rr))
    facts['verification']={k:stats([abs(r[k]) for r in rows]) for k in
        ('component_error','norm_error','bias_error','A_closure','bias_closure','readout_closure','unresolved','zero_candidate')}
    facts['verification']['nonfinite_state_rows']=sum(not all(np.isfinite(r[k]) for k in ('M','A','C6','relative_error','fine_slope_path')) for r in rows)
    facts['verification']['grid_error_difference']=stats([abs(r['relative_eval_error']-r['relative_error']) for r in rows if np.isfinite(r.get('relative_eval_error',np.nan))])
    facts['execution']=json.loads((args.inputs/'execution.json').read_text())
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,grid=plt.subplots(2,2,figsize=(11,8),layout='constrained');axes=grid.ravel()
    for i,target in enumerate(SIX):
        rr=[r for r in native if r['target']==target]
        for r in rr:axes[0].scatter(i+(-.08 if r['width']==705 else .08),r['mean_capped_gain'],color='#0072B2',s=24)
    axes[0].axhline(1,color='gray',lw=.8);axes[0].set_ylabel('Available mean fine-norm gain\n(native trajectory; cap 10)')
    for ax,key in zip(axes[1:],('fine_path_ratio','slope_effect_percent','error_effect_percent')):
        for ci,arm in enumerate(('gain_only','tracking_attenuated_variance')):
            color=plt.get_cmap('tab10')(ci)
            for i,target in enumerate(SIX):
                for off,shift,filled in ((10000,-.04,True),(20000,.04,False)):
                    v=[r[key] for r in contrasts if r['target']==target and r['arm']==arm and r['offset']==off]
                    if not v:continue
                    xx=i+(ci-.5)*.28+shift;ax.vlines(xx,min(v),max(v),color=color,alpha=.35)
                    ax.scatter(xx,np.median(v),facecolors=color if filled else 'white',edgecolors=color,s=28,zorder=3)
        ax.axhline(1 if key=='fine_path_ratio' else 0,color='gray',lw=.8)
        if key=='fine_path_ratio':ax.set_yscale('log')
        else:ax.set_yscale('symlog',linthresh=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
    axes[1].set_ylabel('Accumulated fine slope path / native')
    axes[2].set_ylabel('Slope RMS change vs native (%)');axes[3].set_ylabel('Raw error change vs native (%)')
    axes[3].set_ylim(bottom=-100)
    for ax in axes:ax.set_xticks(range(6),LABELS);ax.set_xlim(-.6,5.6)
    handles=[Line2D([],[],color=plt.get_cmap('tab10')(i),marker='o',ls='',label=label) for i,label in enumerate(('Gain only','Tracking attenuation in fine denominator'))]
    handles += [Line2D([],[],color='black',marker='o',ls='',label='10k pulse'),Line2D([],[],color='black',marker='o',mfc='white',ls='',label='After 10k native release')]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Does tracking indirectly restrict the fine update through Adam variance?')
    fig.savefig(args.output/'tracking_denominator.png',dpi=180,bbox_inches='tight')
    fig.savefig(args.output/'tracking_denominator.svg',bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(runs=len(runs),complete_runs=len(complete),native_cases=len(native),native=facts['native'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=300);analyze(parser.parse_args())
