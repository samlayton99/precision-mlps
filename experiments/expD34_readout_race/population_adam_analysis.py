"""Curated scalar comparisons and figures, without generating report prose."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .population_adam_run import SIX, write_csv
from .population_adam import COMPONENTS

LABELS=dict(moment5='Degree 5',mixed_sine='Mixed sine',gauss_left='Gaussian',
            bump_right='Bump',step_right='Step',kink_abs='Kink')


def read(path):
    if not path.exists():return []
    with path.open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            try:row[k]=float(v) if v!='' else np.nan
            except (TypeError,ValueError):pass
    return rows


def identity(row):
    return tuple(row.get(k,'') for k in ('cohort','optimizer','width','seed','target','eta','recipe'))


def groups(rows):
    grouped={}
    for r in rows:grouped.setdefault(identity(r),{})[r['step']]=r
    return [sorted(v.values(),key=lambda r:r['step']) for v in grouped.values()]


def stats(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    return dict(n=len(v),median=float(np.median(v)),min=float(v.min()),max=float(v.max())) if len(v) else dict(n=0)


def summarize(rows):
    lookup={r['step']:r for r in rows}
    if 25000 not in lookup or max(lookup)<25000:return None
    first=lookup[25000];last=rows[-1]
    result={k:first[k] for k in ('cohort','optimizer','width','seed','target')}
    result.update(start=25000,end=last['step'],complete=last['step']==125000,
        M_start=first['M'],M_end=last['M'],M_ratio=last['M']/first['M'],
        C6_start=first['C6'],C6_end=last['C6'],C6_ratio=last['C6']/first['C6'],
        effective_count_start=first['width']/np.sqrt(first['C6']),
        effective_count_end=last['width']/np.sqrt(last['C6']),
        lambda_start=first['lambda_rms'],lambda_end=last['lambda_rms'],
        slope_ratio=last['slope_rms']/first['slope_rms'],
        error_start=first['relative_error'],error_end=last['relative_error'],
        eval_error_end=last.get('relative_eval_error',np.nan),
        error_ratio=last['relative_error']/first['relative_error'],
        K_start=first['K'],K_end=last['K'],
        logC6_change=float(np.log(last['C6']/first['C6'])),
        M_change=last['M']-first['M'],A_change=last.get('A',np.nan)-first.get('A',np.nan))
    if 'sum_rootC6_sum' not in first:return result
    future=[r for r in rows if r['step']>25000]
    clocks=[r['accumulated_concentration_ratio'] for r in future]
    result.update(clock_ratio_end=clocks[-1],clock_ratio_max=max(clocks),
        clock_factor_two_all_saved_prefixes=max(clocks)<=2,
        minimum_training_error=min(last['minimum_error'],last['relative_error']),
        any_1pct=last['hits_1pct']>0 or last['relative_error']<.01,
        any_0p1pct=last['hits_0p1pct']>0 or last['relative_error']<.001,
        any_1e4=last['hits_1e4']>0 or last['relative_error']<.0001,
        unresolved=last['unresolved'],identity_max=last['identity_max'])
    for q in ('M','A','logC6'):
        parts=[]
        for c in (*COMPONENTS,'defect'):
            value=last['sum_'+q+'_'+c]-first['sum_'+q+'_'+c]
            result[q+'_'+c]=value;parts.append(value)
        result[q+'_fine']=sum(result[q+'_'+c] for c in ('generated','target','compensation','inherited'))
        result[q+'_direct_fine']=result[q+'_generated']+result[q+'_target']
        result[q+'_closure_max']=max(abs(r[q+'_closure']) for r in future)
        result[q+'_signed_budget']=sum(abs(v) for v in parts)
        result[q+'_closure_relative']=result[q+'_closure_max']/max(1.,result[q+'_signed_budget'])
    for key in ('fine_slope_path','tracking_slope_path','total_slope_path'):
        result[key]=last['sum_'+key]-first['sum_'+key]
    total=result['fine_slope_path']+result['tracking_slope_path']
    result['tracking_slope_activity_share']=result['tracking_slope_path']/total if total else np.nan
    for key in ('raw_access','balanced_raw_access','adaptive_access','balanced_adaptive_access',
                'jacobian_fine_hs','jacobian_adaptive_fine_hs'):
        result[key+'_start']=first[key];result[key+'_end']=last[key]
        result[key+'_ratio']=last[key]/first[key] if first[key] else np.nan
        result[key+'_median']=float(np.nanmedian([r[key] for r in future]))
    result['virtual_next_loss_increase_share']=float(np.mean([r['actual_loss_change']>0 for r in future]))
    replay=[r.get('archive_parameter_relative_difference',np.nan) for r in future]
    result['archive_relative_difference_max']=max((v for v in replay if np.isfinite(v)),default=np.nan)
    return result


def aggregate(summaries):
    result={}
    for cohort,optimizer,width in sorted({(s['cohort'],s['optimizer'],s['width']) for s in summaries}):
        rr=[s for s in summaries if (s['cohort'],s['optimizer'],s['width'])==(cohort,optimizer,width) and s['complete']]
        key=f'{cohort}_{optimizer}_W{int(width)}'
        fields=('M_start','M_ratio','C6_start','C6_ratio','effective_count_start','effective_count_end',
                'lambda_end','slope_ratio','error_end','eval_error_end','clock_ratio_max',
                'tracking_slope_activity_share','balanced_adaptive_access_start','balanced_adaptive_access_ratio',
                'balanced_adaptive_access_median','balanced_raw_access_median',
                'M_closure_relative','A_closure_relative','logC6_closure_relative','archive_relative_difference_max')
        result[key]=dict(cases=len(rr),**{f:stats([r.get(f,np.nan) for r in rr]) for f in fields})
        for field in ('clock_factor_two_all_saved_prefixes','any_1pct','any_0p1pct','any_1e4'):
            observed=[r for r in rr if field in r]
            result[key][field]=dict(measured=len(observed),count=sum(bool(r[field]) for r in observed))
        for q in ('logC6','M','A'):
            for c in ('generated','target','direct_fine','compensation','tracking','fine','defect'):
                field=q+'_'+c;observed=[r for r in rr if field in r]
                result[key][field]=dict(measured=len(observed),positive=sum(r[field]>0 for r in observed),
                                       negative=sum(r[field]<0 for r in observed),**stats([r[field] for r in observed]))
    return result


def archive_comparisons(rows):
    output=[]
    for rr in groups(rows):
        lookup={r['step']:r for r in rr}
        if 20000 not in lookup:continue
        first=lookup[20000]
        for end in (100000,120000,600000,2000000):
            if end not in lookup:continue
            last=lookup[end]
            output.append(dict(**{k:first[k] for k in ('cohort','optimizer','width','seed','target','eta','recipe')},
                start=20000,end=end,M_start=first['M'],M_ratio=last['M']/first['M'],
                C6_start=first['C6'],C6_ratio=last['C6']/first['C6'],
                effective_count_start=first['width']/np.sqrt(first['C6']),
                effective_count_end=first['width']/np.sqrt(last['C6']),
                slope_ratio=last['slope_rms']/first['slope_rms'],
                lambda_end=last['lambda_rms'],error_end=last['relative_error']))
    return output


def plot(output,all_groups,summaries):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'svg.fonttype':'none'})
    colors=dict(zip(SIX,plt.get_cmap('tab10').colors))
    widths=(705,1409)
    selected=[r for r in all_groups if r[0]['cohort'] in ('wide','gd_reference')]
    if not selected:return
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    def curves(ax,w,key,normalize=False,adam_only=False):
        for opt in (('adam',) if adam_only else ('adam','gd')):
            for target in SIX:
                rr=[[s for s in r if 25000<=s['step']<=125000] for r in selected
                    if r[0]['width']==w and r[0]['optimizer']==opt and r[0]['target']==target]
                rr=[r for r in rr if r and key in r[-1]]
                if not rr:continue
                common=sorted(set.intersection(*[{s['step'] for s in r if np.isfinite(s.get(key,np.nan))} for r in rr]))
                if not common:continue
                values=[]
                for r in rr:
                    lookup={s['step']:s for s in r}
                    divisor=r[0][key] if normalize else 1.
                    values.append([lookup[t][key]/divisor for t in common])
                v=np.asarray(values);t=(np.asarray(common)-25000)/1000
                ax.plot(t,np.median(v,axis=0),color=colors[target],ls='-' if opt=='adam' else '--',lw=1.3)
                if opt=='adam':ax.fill_between(t,v.min(axis=0),v.max(axis=0),color=colors[target],alpha=.10)
    legend=[Line2D([],[],color=colors[t],label=LABELS[t]) for t in SIX]
    styles=[Line2D([],[],color='black',label='Adam'),Line2D([],[],color='black',ls='--',label='GD')]
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for j,w in enumerate(widths):
        curves(axes[0,j],w,'relative_error');curves(axes[1,j],w,'lambda_rms')
        axes[0,j].axhline(.01,color='gray',ls=':',lw=1);axes[1,j].axhline(.25,color='gray',ls=':',lw=1)
        axes[0,j].set_title(f'Width {w}');axes[0,j].set_yscale('log');axes[1,j].set_yscale('log')
        axes[1,j].set_xlabel('Updates after 25k (thousands)')
    axes[0,0].set_ylabel('Raw relative output error');axes[1,0].set_ylabel('Normalized slope RMS')
    fig.legend(handles=legend+styles,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Same targets and initializations; shading spans available Adam seeds')
    save(fig,'output_and_scale')
    for rr in selected:
        for r in rr:r['effective_count']=r['width']/np.sqrt(r['C6'])
    fig,axes=plt.subplots(3,2,figsize=(10,8),layout='constrained')
    for j,w in enumerate(widths):
        curves(axes[0,j],w,'M');curves(axes[1,j],w,'effective_count')
        curves(axes[2,j],w,'accumulated_concentration_ratio',adam_only=True)
        axes[0,j].set_title(f'Width {w}');axes[0,j].set_yscale('log');axes[1,j].set_yscale('log')
        axes[2,j].axhline(2,color='black',ls=':',label='GD reference allowance')
        axes[2,j].axhline(1,color='gray',ls=':',lw=.7)
        axes[2,j].set_xlabel('Updates after 25k (thousands)')
    axes[0,0].set_ylabel('Total hidden parameter energy')
    axes[1,0].set_ylabel('Effective energy-sharing count')
    axes[2,0].set_ylabel('Accumulation / window reference')
    fig.legend(handles=legend+styles,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('A stable concentration history need not mean broadly shared, modest energy')
    save(fig,'population_structure')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    components=('direct_fine','compensation','tracking','defect')
    component_labels=('Direct fine residual','Coarse compensation','Coarse tracking','Finite-step remainder')
    for j,w in enumerate(widths):
        for i,q in enumerate(('logC6','A')):
            ax=axes[i,j];positive=np.zeros(6);negative=np.zeros(6);net=[]
            denominator=lambda r: 1. if q=='logC6' else (r['lambda_start']/ (2/(512 if w==705 else 1024)))**2*w
            for ci,c in enumerate(components):
                values=[]
                for target in SIX:
                    rr=[r for r in summaries if r['cohort']=='wide' and r['optimizer']=='adam' and r['width']==w and r['target']==target and r['complete']]
                    values.append(np.mean([r[q+'_'+c]/denominator(r) for r in rr]) if rr else np.nan)
                values=np.asarray(values)
                ax.bar(np.arange(6),values,bottom=np.where(values>=0,positive,negative),label=component_labels[ci])
                positive+=np.maximum(values,0);negative+=np.minimum(values,0)
            for target in SIX:
                rr=[r for r in summaries if r['cohort']=='wide' and r['optimizer']=='adam' and r['width']==w and r['target']==target and r['complete']]
                net.append(np.mean([r[q+'_change']/denominator(r) for r in rr]) if rr else np.nan)
            ax.plot(range(6),net,'kD',ms=4,label='Actual change')
            ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(6),[LABELS[t] for t in SIX],rotation=25)
            ax.set_title(f'Width {w}');ax.set_yscale('symlog',linthresh=.02)
    axes[0,0].set_ylabel('Change in log concentration');axes[1,0].set_ylabel('Slope energy change / initial slope energy')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Signed actual-update contributions, 25k–125k; averages across available seeds')
    save(fig,'signed_growth')
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    for j,w in enumerate(widths):
        for target in SIX:
            rr=[[s for s in r if 25000<=s['step']<=125000] for r in selected if r[0]['width']==w and r[0]['optimizer']=='adam' and r[0]['target']==target]
            if not rr:continue
            common=sorted(set.intersection(*[{s['step'] for s in r} for r in rr]));t=(np.asarray(common)-25000)/1000
            for key,style in [('balanced_raw_access','--'),('balanced_adaptive_access','-')]:
                values=np.asarray([[{s['step']:s for s in r}[step][key]*.002 for step in common] for r in rr])
                axes[0,j].plot(t,np.median(values,axis=0),color=colors[target],ls=style,lw=1.)
            x=[];y=[]
            for r in rr:
                for s in r:
                    if s['step']%10000==0:
                        norm=.5*s['fine_residual_norm']**2
                        if norm>0:x.append(s['current_gradient_descent']/norm);y.append(-s['actual_loss_change']/norm)
            axes[1,j].scatter(x,y,s=10,color=colors[target],alpha=.45)
        axes[0,j].set_title(f'Width {w}');axes[0,j].set_yscale('log');axes[0,j].set_xlabel('Updates after 25k (thousands)')
        axes[1,j].set_xscale('symlog',linthresh=1e-8);axes[1,j].set_yscale('symlog',linthresh=1e-8)
        axes[1,j].axhline(0,color='gray',lw=.7);axes[1,j].set_xlabel('Current-gradient descent / fine residual loss')
    axes[0,0].set_ylabel('Instantaneous balanced fine access × step size')
    axes[1,0].set_ylabel('Actual next-step loss reduction / fine residual loss')
    fig.legend(handles=legend+[Line2D([],[],color='black',ls='--',label='Raw metric'),Line2D([],[],color='black',label='Adaptive metric')],loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Adaptive access and realized progress; checkpoint diagnostics, not hitting times')
    save(fig,'adaptive_access')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage');parser.add_argument('--seconds');args=parser.parse_args()
    states=[];archives=[];interventions=[];execution=[];cases=[]
    for folder in sorted(args.inputs.glob('part*')):
        states+=read(folder/'states.csv');states+=read(folder/'gd_reference.csv')
        archives+=read(folder/'archive_states.csv');interventions+=read(folder/'archived_interventions.csv')
        cases+=read(folder/'cases.csv')
        if (folder/'execution.json').exists():execution.append(json.loads((folder/'execution.json').read_text()))
    grouped=groups(states);summaries=[s for rr in grouped if (s:=summarize(rr)) is not None]
    archived=archive_comparisons(archives)
    write_csv(args.output/'population_comparisons.csv',summaries)
    write_csv(args.output/'archive_comparisons.csv',archived)
    intervention_contrasts=[]
    for r in interventions:
        if r['offset']!=20000:continue
        baseline=next(s for s in interventions if s['target']==r['target'] and s['cohort']==r['cohort'] and s['offset']==r['offset'] and s['alpha_m']==1 and s['alpha_v']==1)
        intervention_contrasts.append(dict(target=r['target'],cohort=r['cohort'],alpha_m=r['alpha_m'],alpha_v=r['alpha_v'],
            C6_ratio=r['C6']/baseline['C6'],M_ratio=r['M']/baseline['M'],slope_ratio=r['slope_rms']/baseline['slope_rms']))
    write_csv(args.output/'archived_intervention_contrasts.csv',intervention_contrasts)
    archive_facts={}
    for cohort in ('archive177','archive512'):
        for optimizer in ('adam','gd'):
            for end in (100000,120000,600000):
                rr=[r for r in archived if r['cohort']==cohort and r['optimizer']==optimizer and r['end']==end and r['eta']==.002 and r['recipe']=='constant']
                if rr:archive_facts[f'{cohort}_{optimizer}_{end}']=dict(cases=len(rr),**{k:stats([r[k] for r in rr]) for k in ('M_start','M_ratio','C6_start','C6_ratio','effective_count_start','effective_count_end','lambda_end','error_end')})
    facts=dict(aggregate=aggregate(summaries),archive=archive_facts,
        gpu_seconds=sum(r['seconds'] for r in execution if r['platform']=='Modal GPU'),
        completed_cases=sum(r['status']=='complete' for r in cases),
        incomplete_cases=[r for r in cases if r['status']!='complete'],
        input_execution_records=execution,
        intervention_snapshot_cohorts=sorted({r['cohort'] for r in interventions}))
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    plot(args.output,grouped,summaries)
    print(json.dumps(dict(completed_cases=facts['completed_cases'],gpu_seconds=facts['gpu_seconds'],groups=list(facts['aggregate']))),flush=True)


if __name__=='__main__':main()
