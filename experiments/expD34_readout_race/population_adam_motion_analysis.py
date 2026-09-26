"""Scalar evidence and three figures for Adam motion; no generated report prose."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .population_adam_run import SIX, write_csv

LABELS=('Degree 5','Mixed sine','Gaussian','Bump','Step','Kink')
POLICIES=('current_fine','fixed_scaling','both')
POLICY_LABELS=('Current fine gradient','Fixed fine scaling','Both')


def read(path):
    with path.open() as f: rows=list(csv.DictReader(f))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v) if v!='' else np.nan
            except ValueError:pass
    return rows


def stats(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    return dict(n=len(v),min=float(v.min()),median=float(np.median(v)),max=float(v.max())) if len(v) else dict(n=0)


def identity(r):return tuple(r[k] for k in ('target','width','seed','age','arm'))


def analyze(args):
    rows=read(args.inputs/'states.csv');runs=read(args.inputs/'runs.csv');crossed=read(args.inputs/'crossed.csv')
    groups={}
    for r in rows:groups.setdefault(identity(r),{})[r['offset']]=r
    complete={identity(r) for r in runs if r['complete']=='True'}
    native=[];contrasts=[];force_changes=[]
    for key,lookup in groups.items():
        if key not in complete:continue
        first=lookup[0];last=lookup[max(lookup)]
        if first['arm']=='native' and first['age']==25000 and last['offset']==100000:
            s={k:first[k] for k in ('target','width','seed','age')}
            s.update(M_start=first['M'],M_end=last['M'],lambda_start=first['lambda_rms'],lambda_end=last['lambda_rms'],
                M_ratio=last['M']/first['M'],
                slope_ratio=last['slope_rms']/first['slope_rms'],error_start=first['relative_error'],error_end=last['relative_error'],
                energy_overlap=last['energy_overlap'],effective_count_start=first['energy_effective_count'],
                effective_count_end=last['energy_effective_count'],fine_global_coherence=last['fine_slope_net_norm']/last['fine_slope_path'],
                fine_radial_cancellation=last['fine_radial_cancellation'],
                actual_displacement_per_total_path=last['actual_slope_displacement']/last['total_slope_path'],
                top5_energy_start=first['group0_energy_share']+first['group1_energy_share'],
                top5_energy_end=last['group0_energy_share']+last['group1_energy_share'],
                top5_fine_activity=last['group0_fine_activity_share']+last['group1_fine_activity_share'],
                top5_total_activity=last['group0_total_activity_share']+last['group1_total_activity_share'],
                top5_fine_A_fraction=(last['group0_fine_A']+last['group1_fine_A'])/last['A_fine'],
                fine_activity_energy_overlap=last['fine_activity_energy_overlap'],
                total_activity_energy_overlap=last['total_activity_energy_overlap'])
            for end,r in (('start',first),('end',last)):
                for name,k in (('slope','A'),('bias','bias2'),('readout','readout2')):s[f'{name}_fraction_{end}']=r[k]/r['M']
            for window in (100,1000,10000):
                for k in ('slope_coherence','hidden_coherence','endpoint_alignment','adjacent_alignment'):
                    s[f'{k}_{window}']=last[f'{k}_{window}']
            for proposal in ('raw','scaled_current','processed'):
                for measure in ('hidden_norm','slope_fraction','outward_cosine','A_rate'):
                    k=proposal+'_'+measure;s[k+'_median']=float(np.median([r[k] for off,r in lookup.items() if off>0]))
            s['processed_current_alignment_median']=float(np.median([r['processed_current_alignment'] for off,r in lookup.items() if off>0]))
            for component in ('fine','tracking','step'):s['A_'+component+'_over_start']=last['A_'+component]/first['A']
            s['fine_path_late_over_early']=(last['fine_slope_path']-lookup[90000]['fine_slope_path'])/lookup[10000]['fine_slope_path']
            early=lookup[10000]['A_fine'];late=last['A_fine']-lookup[90000]['A_fine']
            s['fine_A_early_over_start']=early/first['A'];s['fine_A_late_over_start']=late/first['A']
            s['fine_A_late_over_early']=late/early if early>0 else np.nan
            # Geometric means preserve the multiplicative identity exactly.
            # The error is total relative error, not an operator sensitivity.
            def amplitude_factors(offsets):
                values=[]
                for off in offsets:
                    r=lookup[off];e=r['relative_error']
                    raw,scaled,processed=[r[p+'_hidden_norm']*r[p+'_slope_fraction'] for p in ('raw','scaled_current','processed')]
                    if min(e,raw,scaled,processed)<=0:continue
                    values.append([e,raw/e,scaled/raw,processed/scaled,processed])
                return np.exp(np.mean(np.log(values),axis=0)),len(values)
            before,nb=amplitude_factors(range(1000,10001,1000))
            after,na=amplitude_factors(range(91000,100001,1000))
            factors=after/before
            for name,value in zip(('error','raw_fine_per_error','adaptive_gain','momentum_gain','processed_slope_norm'),factors):
                s['amplitude_'+name+'_late_over_early']=float(value)
            s['amplitude_samples_early']=nb;s['amplitude_samples_late']=na
            s['amplitude_product_relative_error']=float(abs(np.prod(factors[:4])/factors[4]-1))
            native.append(s)
        if first['arm']=='native':continue
        baseline_key=(*key[:-1],'native')
        if baseline_key not in complete:continue
        baseline=groups[baseline_key]
        for offset in (10000,20000):
            if offset not in lookup or offset not in baseline:continue
            r=lookup[offset];b=baseline[offset]
            q={k:first[k] for k in ('target','width','seed','age','arm')}
            q.update(offset=offset,slope_effect_percent=100*(r['slope_rms']/b['slope_rms']-1),
                lambda_difference=r['lambda_rms']-b['lambda_rms'],error_effect_percent=100*(r['relative_error']/b['relative_error']-1),
                error_difference=r['relative_error']-b['relative_error'],
                fine_path_ratio=r['fine_slope_path']/b['fine_slope_path'],
                fine_A_change_over_start=(r['A_fine']-b['A_fine'])/first['A'],
                tracking_A_change_over_start=(r['A_tracking']-b['A_tracking'])/first['A'],
                fine_A_over_start=r['A_fine']/first['A'],tracking_A_over_start=r['A_tracking']/first['A'],
                step_A_over_start=r['A_step']/first['A'])
            common=min(r['fine_slope_path'],b['fine_slope_path'])
            def at_path(series):
                rr=[series[o] for o in sorted(series) if o<=offset]
                return float(np.interp(common,[v['fine_slope_path'] for v in rr],[v['slope_rms'] for v in rr]))
            q['equal_path_slope_effect_percent']=100*(at_path(lookup)/at_path(baseline)-1)
            contrasts.append(q)
    for key in sorted({identity(r) for r in crossed}):
        rr=[r for r in crossed if identity(r)==key]
        if key not in complete:continue
        s=dict(zip(('target','width','seed','age','arm'),key))
        g=np.array([r['geometry_A_rate_change'] for r in rr]);e=np.array([r['residual_A_rate_change'] for r in rr])
        s.update(intervals=len(rr),geometry_positive_fraction=float(np.mean(g>0)),
            residual_negative_fraction=float(np.mean(e<0)),opposite_sign_fraction=float(np.mean(g*e<0)),
            geometry_positive_residual_negative_fraction=float(np.mean((g>0)&(e<0))),
            absolute_geometry_sum=float(np.abs(g).sum()),absolute_residual_sum=float(np.abs(e).sum()),
            geometry_signed_sum=float(g.sum()),residual_signed_sum=float(e.sum()),
            remaining_change_fraction=float(np.abs(g+e).sum()/(np.abs(g).sum()+np.abs(e).sum())))
        force_changes.append(s)
    write_csv(args.output/'native_motion.csv',native);write_csv(args.output/'intervention_contrasts.csv',contrasts)
    write_csv(args.output/'force_changes.csv',force_changes)
    fields=('slope_ratio','M_ratio','fine_path_late_over_early','fine_A_late_over_early','energy_overlap','effective_count_start','effective_count_end','fine_global_coherence',
        'fine_radial_cancellation','actual_displacement_per_total_path','top5_energy_start','top5_energy_end',
        'top5_fine_activity','top5_total_activity','top5_fine_A_fraction',
        'slope_fraction_end','bias_fraction_end','readout_fraction_end',
        'slope_coherence_100','slope_coherence_1000','slope_coherence_10000','adjacent_alignment_1000',
        'raw_outward_cosine_median','scaled_current_outward_cosine_median','processed_outward_cosine_median',
        'processed_current_alignment_median','raw_slope_fraction_median','processed_slope_fraction_median',
        'amplitude_error_late_over_early','amplitude_raw_fine_per_error_late_over_early',
        'amplitude_adaptive_gain_late_over_early','amplitude_momentum_gain_late_over_early',
        'amplitude_processed_slope_norm_late_over_early','amplitude_product_relative_error')
    facts=dict(native_cases=len(native),runs=len(runs),complete_runs=len(complete),
        native={k:stats([r[k] for r in native]) for k in fields},interventions={},force_changes={})
    for age in (25000,125000):
        for arm in POLICIES:
            for offset in (10000,20000):
                selected=[r for r in contrasts if r['age']==age and r['arm']==arm and r['offset']==offset]
                key=f'{age}_{arm}_{offset}'
                facts['interventions'][key]={k:stats([r[k] for r in selected]) for k in
                    ('slope_effect_percent','error_effect_percent','equal_path_slope_effect_percent','fine_path_ratio',
                     'fine_A_change_over_start','tracking_A_change_over_start')}
                facts['interventions'][key].update(slope_positive=sum(r['slope_effect_percent']>0 for r in selected),
                    error_better=sum(r['error_effect_percent']<0 for r in selected),
                    both_better=sum(r['slope_effect_percent']>0 and r['error_effect_percent']<0 for r in selected))
        fc=[r for r in force_changes if r['age']==age]
        facts['force_changes'][str(age)]={k:stats([r[k] for r in fc]) for k in
            ('geometry_positive_fraction','residual_negative_fraction','opposite_sign_fraction',
             'geometry_positive_residual_negative_fraction','remaining_change_fraction')}
    facts['verification']={k:max(abs(r[k]) for r in rows) for k in
        ('component_error','norm_error','bias_error','A_closure','bias_closure','readout_closure','unresolved','zero_candidate')}
    facts['verification'].update({k:max(abs(r[k]) for r in crossed) for k in ('crossed_closure','current_force_error')})
    facts['verification']['crossed_unresolved']=sum(r['crossed_resolved']!=1 for r in crossed)
    facts['verification']['archive_relative_difference']=stats([r['archive_relative_difference'] for r in runs if np.isfinite(r.get('archive_relative_difference',np.nan))])
    facts['verification']['max_grid_error_difference']=max(abs(r['relative_eval_error']-r['relative_error']) for r in rows if np.isfinite(r.get('relative_eval_error',np.nan)))
    facts['execution']=json.loads((args.inputs/'execution.json').read_text())
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    plot(args.output,native,contrasts)
    print(json.dumps(dict(native_cases=len(native),complete_runs=len(complete),contrasts=len(contrasts),verification=facts['verification'])),flush=True)


def plot(output,native,contrasts):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    def vals(width,target,key):return [r[key] for r in native if r['width']==width and r['target']==target]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for j,w in enumerate((705,1409)):
        bottom=np.zeros(6)
        for k,c,label in [('slope_fraction_end','#0072B2','Slopes'),('bias_fraction_end','#E69F00','Biases'),('readout_fraction_end','#009E73','Readouts')]:
            v=np.array([np.mean(vals(w,t,k)) for t in SIX]);axes[0,j].bar(range(6),v,bottom=bottom,color=c,label=label);bottom+=v
        for k,c,marker,label in [('top5_energy_end','#0072B2','o','Energy in initial top 5%'),('top5_fine_activity','#D55E00','s','Fine activity in initial top 5%'),('top5_total_activity','#CC79A7','x','Total activity in initial top 5%'),('energy_overlap','#009E73','^','Energy-distribution overlap')]:
            for i,t in enumerate(SIX):
                vv=vals(w,t,k);axes[1,j].scatter([i]*len(vv),vv,color=c,marker=marker,s=23,label=label if i==0 else None)
        axes[0,j].set_title(f'Width {w}: parameter energy at 125k')
        for ax in axes[:,j]:ax.set_ylim(0,1.04);ax.set_xticks(range(6),LABELS,rotation=25);ax.set_xlim(-.6,5.6)
        axes[0,j].set_ylim(0,1.22);axes[0,j].set_yticks([0,.25,.5,.75,1])
    axes[0,0].set_ylabel('Fraction of parameter energy');axes[1,0].set_ylabel('Population share / overlap')
    axes[0,1].legend(fontsize=8,loc='upper center',ncol=3)
    handles,labels=axes[1,1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='outside lower center',ncol=2)
    fig.suptitle('Concentrated energy, active groups, and redistribution: 25k–125k')
    save(fig,'energy_and_activity')
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for j,w in enumerate((705,1409)):
        for ci,n in enumerate((100,1000,10000)):
            for i,t in enumerate(SIX):
                vv=vals(w,t,f'slope_coherence_{n}');axes[0,j].scatter([i+(ci-1)*.16]*len(vv),vv,s=24,color=plt.get_cmap('tab10')(ci),label=f'{n:,} updates' if i==0 else None)
        for ci,(proposal,label) in enumerate(zip(('raw','scaled_current','processed'),('Raw fine','Scaled current fine','Momentum-processed fine'))):
            for i,t in enumerate(SIX):
                vv=vals(w,t,proposal+'_outward_cosine_median');axes[1,j].scatter([i+(ci-1)*.16]*len(vv),vv,s=24,color=plt.get_cmap('tab10')(ci),label=label if i==0 else None)
        for ci,(key,label) in enumerate((('M_ratio','Energy: end / start'),('fine_path_late_over_early','Fine slope activity: late / early'))):
            for i,t in enumerate(SIX):
                vv=vals(w,t,key);axes[2,j].scatter([i+(ci-.5)*.16]*len(vv),vv,s=24,color=plt.get_cmap('tab10')(ci),label=label if i==0 else None)
        axes[0,j].set_title(f'Width {w}');axes[0,j].set_ylim(0,1.04);axes[1,j].set_ylim(-1.04,1.04)
        axes[1,j].axhline(0,color='gray',lw=.8)
        axes[2,j].set_yscale('log');axes[2,j].axhline(1,color='gray',lw=.8)
        axes[2,j].yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        axes[2,j].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        axes[2,j].yaxis.set_minor_formatter(NullFormatter())
        for ax in axes[:,j]:ax.set_xticks(range(6),LABELS,rotation=25);ax.set_xlim(-.6,5.6)
    axes[0,0].set_ylabel('Net fine slope displacement / path');axes[1,0].set_ylabel('Cosine with outward slope direction')
    axes[2,0].set_ylabel('Ratio (activity: final / first 10k windows)')
    axes[0,1].legend(fontsize=7.5,loc='lower center',ncol=3)
    axes[1,1].legend(fontsize=7,loc='lower center',ncol=3)
    axes[2,1].legend(fontsize=8)
    fig.suptitle('Directional persistence and outward alignment: each seed shown')
    save(fig,'fine_direction')
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    for j,age in enumerate((25000,125000)):
        for i,key in enumerate(('slope_effect_percent','error_effect_percent')):
            ax=axes[i,j]
            for ci,arm in enumerate(POLICIES):
                color=plt.get_cmap('tab10')(ci)
                for ti,target in enumerate(SIX):
                    for offset,shift,filled in ((10000,-.035,True),(20000,.035,False)):
                        values=[r[key] for r in contrasts if r['age']==age and r['arm']==arm and r['target']==target and r['offset']==offset]
                        if not values:continue
                        mid=np.median(values);xx=ti+(ci-1)*.22+shift
                        ax.vlines(xx,min(values),max(values),color=color,alpha=.35,lw=1)
                        ax.scatter(xx,mid,s=25,facecolors=color if filled else 'white',edgecolors=color,zorder=3)
            ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(6),LABELS,rotation=25);ax.set_xlim(-.6,5.6)
            ax.set_yscale('symlog',linthresh=1)
            ticks=np.array([-1000,-300,-100,-30,-10,-3,-1,-.3,-.1,0,.1,.3,1,3,10,30,100,300,1000])
            lo,hi=ax.get_ylim()
            if max(abs(lo),abs(hi))>=1:ticks=ticks[(np.abs(ticks)>=1)|(ticks==0)]
            ax.set_yticks(ticks[(ticks>=lo)&(ticks<=hi)])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
        axes[0,j].set_title(f'Fork at {age//1000}k; 10k pulse + 10k release')
    axes[0,0].set_ylabel('Slope RMS change vs native (%)');axes[1,0].set_ylabel('Raw error change vs native (%)')
    handles=[Line2D([],[],color=plt.get_cmap('tab10')(i),marker='o',ls='',label=label) for i,label in enumerate(POLICY_LABELS)]
    handles += [Line2D([],[],color='black',marker='o',ls='',label='Pulse endpoint'),Line2D([],[],color='black',marker='o',mfc='white',ls='',label='After release')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Fine-update interventions: medians and ranges across widths and seeds')
    save(fig,'intervention_responses')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=300)
    analyze(parser.parse_args())
