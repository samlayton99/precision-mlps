"""Audit structural population bounds using saved wide-network trajectories.

Run on Modal CPU. Writes numerical evidence and plots, never report prose.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from . import adam_forces as af, effective_feedback_holdout as holdout
from . import population_concentration as pc

MODELS = ('generic', 'projected', 'target', 'reference')
LABELS = dict(generic='Generic concentration', projected='Affine projection',
              target='Projection + target moments', reference='Earlier polynomial-shape bound')
STRUCTURAL = ('width','C4','C6','C8','C10','C14','q3','q5','j3','j5','g3','g5')


def cumulative(t, v):
    return cumulative_trapezoid(v, t, initial=0.)


def target_data(name):
    data = holdout.data if name in holdout.TARGETS else af.data
    x,y,_,_ = data(name)
    return x,y,pc.target_structure(x,y)


def load(paths):
    groups = defaultdict(list)
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                for key in row:
                    if key not in ('study','target','kind','cold'): row[key]=float(row[key])
                if row['width'] < 705: continue
                row['source'] = str(path.with_suffix('.npz'))
                key = tuple(row[k] for k in ('study','target','width','seed','kind','dt'))
                groups[key].append(row)
    return [sorted(rows,key=lambda r:r['offset']) for rows in groups.values()]


def first_horizon(t, valid):
    bad = np.flatnonzero(~valid)
    return float(t[max(0,bad[0]-1)]/.002 if len(bad) else t[-1]/.002)


def evaluate(rows, target, offset=5000):
    future = [r for r in rows if offset <= r['offset'] <= offset+100000]
    t = np.array([r['time'] for r in future]); t -= t[0]
    get = lambda k:np.array([r[k] for r in future])
    initial = future[0]
    is_gd = initial['kind']=='gd'
    e0 = initial['relative_error']*initial['target_norm'] if is_gd else math.sqrt(initial['Y2'])
    r0 = math.sqrt(initial['M'])
    tracking = get('tracking_norm') if is_gd else np.zeros_like(t)
    path = cumulative(t,tracking)
    clock = cumulative(t,np.sqrt(get('C6')))
    c6integral = cumulative(t,get('C6'))
    window = [r for r in rows if r['offset']<=offset]
    wt = np.array([r['time'] for r in window])
    window_mean = float(cumulative(wt,np.sqrt([r['C6'] for r in window]))[-1]/(wt[-1]-wt[0]))
    ratio = np.divide(clock,window_mean*t,out=np.ones_like(t),where=t>0)
    shape_arrays = np.array([[r[k] for k in STRUCTURAL] for r in future])
    def shape_at(s):
        index = min(np.searchsorted(t,s,side='right')-1,len(t)-2)
        index = max(0,index)
        fraction = (s-t[index])/(t[index+1]-t[index])
        return dict(zip(STRUCTURAL,shape_arrays[index]+fraction*(shape_arrays[index+1]-shape_arrays[index])))
    def normalized_floor(radius,row):
        cap = pc.bounds(radius,row,target,e0)['capacity']
        return max(0.,target['fine_norm']-cap)/target['target_norm']
    measured = [pc.bounds(math.sqrt(r['M']),r,target,e0) for r in future]
    summaries,curves = [],{}
    for model in MODELS:
        if model=='generic':
            radius = pc.concentration_envelope(r0,clock,e0,initial['width'],path)
            status = 0
        else:
            def rhs(s,z):
                b = r0*np.exp(min(z[0],30.))
                speed = pc.bounds(b,shape_at(s),target,e0)[model]+np.interp(s,t,tracking)
                return [speed/b]
            def stop(s,z):
                b = r0*np.exp(min(z[0],30.))
                return min(normalized_floor(b,shape_at(s))-.01,
                           .25-initial['h']*b/math.sqrt(initial['width']))
            stop.terminal=True; stop.direction=-1
            solution = solve_ivp(rhs,(0,float(t[-1])),[0.],t_eval=t,events=stop,
                                  rtol=2e-9,atol=1e-11,max_step=1.)
            radius = np.full_like(t,np.nan)
            radius[:len(solution.t)] = r0*np.exp(solution.y[0])
            status = int(solution.status)
            if status < 0: raise RuntimeError(f'Comparison integration failed: {initial}, {model}')
        cap_floor = np.full_like(t,np.nan)
        speed = np.full_like(t,np.nan)
        for i,(b,row) in enumerate(zip(radius,future,strict=True)):
            if np.isfinite(b):
                bound = pc.bounds(b,row,target,e0)
                cap_floor[i] = max(0.,target['fine_norm']-bound['capacity'])/target['target_norm']
                speed[i] = bound[model]
        lam = initial['h']*radius/math.sqrt(initial['width'])
        valid = np.isfinite(radius)&(lam < .25)&(cap_floor > .01)
        horizon = first_horizon(t,valid)
        valid_prefix = np.logical_and.accumulate(valid)
        slack = np.array([b[model] for b in measured])/get('q')
        item = {k:initial[k] for k in ('study','target','width','seed','kind','dt')}
        item.update(model=model,offset=offset,useful_horizon=horizon,integration_status=status,
            E0=e0,M0=initial['M'],tau3=target['tau3'],tau5=target['tau5'],
            D3=target['D3'],B3=target['B3'],
            initial_force_slack=float(slack[0]),minimum_force_slack=float(np.min(slack)),
            median_force_slack=float(np.median(slack)),
            final_lambda_upper=float(lam[-1]) if np.isfinite(lam[-1]) else None,
            final_error_floor=float(cap_floor[-1]) if np.isfinite(cap_floor[-1]) else None,
            final_mass_ratio=float(radius[-1]**2/initial['M']) if np.isfinite(radius[-1]) else None,
            actual_mass_ratio=float(future[-1]['M']/initial['M']),
            maximum_actual_mass_over_bound=float(np.nanmax(get('M')/radius**2)),
            final_concentration_clock=float(clock[-1]),average_sqrt_C6=float(clock[-1]/t[-1]),
            maximum_clock_over_window_mean=float(np.max(ratio)),
            clock_factor1_horizon=first_horizon(t,ratio<=1+64*np.finfo(float).eps),
            clock_factor2_horizon=first_horizon(t,ratio<=2+64*np.finfo(float).eps),
            clock_factor4_horizon=first_horizon(t,ratio<=4+64*np.finfo(float).eps),
            max_sampled_residual_over_initial=float(np.max(get('relative_error')*get('target_norm'))/e0)
            if is_gd else float(np.sqrt(np.max(get('Y2')))/e0),
            final_tracking_path=float(path[-1]),
            cubic_target_term_initial=measured[0]['target_cubic'],
            cubic_generated_term_initial=measured[0]['generated_cubic'],
            cubic_remainder_initial=measured[0]['cubic_remainder'])
        summaries.append(item)
        curves[model]=dict(t=t,radius=radius,lambda_upper=lam,error_floor=cap_floor,
            lambda_actual=get('lambda_rms'),error_actual=get('relative_eval_error'),
            mass_actual=get('M'),q=get('q'),force_upper=speed,slack=slack,
            valid=valid_prefix,clock_ratio=ratio,concentration=get('C6'),
            c6_integral=c6integral)
    return summaries,curves


def snapshot_audit(rows, target, data):
    """Read one sparse state at a time; no dense Hessians or projectors."""
    x,y = data
    basis = np.stack((np.ones_like(x),x/np.sqrt(np.mean(x*x))),axis=1)
    audit=[]
    with np.load(rows[0]['source'],allow_pickle=False) as saved:
        for offset in (0,5000,110000):
            row = next(r for r in rows if r['offset']==offset)
            p = saved[f"case{int(row['case_index'])}_{row['kind']}_{offset}"]
            a,b,c=p[:-1].reshape(3,-1)
            hidden=np.tanh(x[:,None]*a+b); derivative=1-hidden*hidden
            J=np.concatenate((derivative*c*x[:,None],derivative*c,hidden,np.ones((len(x),1))),axis=1)
            jc=basis.T@J/len(x); gram=jc@jc.T
            fine=lambda v:v-basis@(basis.T@v/len(x))
            project=lambda v:v-jc.T@np.linalg.solve(gram,jc@v)
            f=hidden@c+p[-1]; fh=fine(f); g=fine(y); eh=fh-g
            jh=J-basis@jc
            jacnorm=np.sqrt(np.sum(jh*jh)/len(x))
            force=project(J.T@eh/len(x))
            e0=np.linalg.norm(eh)/np.sqrt(len(x))
            bound=pc.bounds(np.sqrt(row['M']),row,target,e0)
            v=np.r_[a,b,-c,0.]
            gen=-v@project(J.T@fh/len(x))
            tar=v@project(J.T@g/len(x))
            direct_j3=(target['s2']*np.sum(4*c*c*a*a*b*b+c*c*a**4+a**4*b*b)
                       +target['s3']*np.sum(c*c*a**4+a**6/9))**.5
            saved_j3=row['j3']*row['M']**1.5/row['width']
            audit.append(dict(target=row['target'],study=row['study'],width=row['width'],seed=row['seed'],
                kind=row['kind'],dt=row['dt'],offset=offset,
                force_reconstruction_relative=abs(np.linalg.norm(force)-row['q'])/row['q'],
                jacobian_generic_slack=bound['jacobian_generic']/jacnorm,
                jacobian_projected_slack=bound['jacobian_projected']/jacnorm,
                capacity_slack=bound['capacity']/(np.linalg.norm(fh)/np.sqrt(len(x))),
                projected_cubic_identity_relative=abs(direct_j3-saved_j3)/saved_j3,
                generated_imbalance=gen,target_imbalance=tar,
                net_imbalance=-v@force,
                signed_split_error=gen+tar+v@force))
    return audit


def plots(summaries,curves,output):
    import matplotlib.pyplot as plt
    dev=[(key,values) for key,values in curves.items() if key[0]=='development' and key[4]=='gd']
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for ax,(key,values) in zip(axes.flat,dev):
        for model in MODELS:
            curve=values[model]; use=curve['valid']
            ax.plot(curve['t'][use]/.002/1000,curve['radius'][use]**2/curve['mass_actual'][0],label=LABELS[model])
        curve=values['generic']
        ax.plot(curve['t']/.002/1000,curve['mass_actual']/curve['mass_actual'][0],'k--',label='Observed population')
        ax.set(title=key[1],xlabel='Further updates (thousands)',ylabel='Hidden energy / starting energy',yscale='log')
    axes.flat[0].legend(fontsize=7)
    fig.savefig(output/'population_bounds.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for key,values in dev:
        c=values['target']; use=c['valid']
        line,=axes[0].plot(c['t']/.002/1000,c['lambda_actual'],label=key[1])
        axes[0].plot(c['t'][use]/.002/1000,c['lambda_upper'][use],'--',color=line.get_color())
        axes[1].plot(c['t']/.002/1000,c['error_actual'],color=line.get_color())
        axes[1].plot(c['t'][use]/.002/1000,c['error_floor'][use],'--',color=line.get_color())
    axes[0].axhline(.25,color='k',ls=':',label='Construction scale 0.25')
    axes[1].axhline(.01,color='k',ls=':')
    axes[0].set(yscale='log',ylabel='Normalized slope RMS',ylim=(1e-4,1.))
    axes[1].set(ylabel='Relative output error',ylim=(0,1.))
    for ax in axes:ax.set_xlabel('Further updates (thousands)')
    axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(output/'population_output.png',dpi=170);plt.close(fig)
    native=[s for s in summaries if s['kind']=='gd' and s['dt']==.002]
    names=sorted({s['target'] for s in native})
    columns=[(705,30),(705,31),(705,33),(1409,31)]
    fig,axes=plt.subplots(1,4,figsize=(13,max(4,len(names)*.25)),layout='constrained',sharey=True)
    for ax,model in zip(axes,MODELS):
        values=np.full((len(names),len(columns)),np.nan)
        for s in native:
            if s['model']==model:
                values[names.index(s['target']),columns.index((s['width'],s['seed']))]=s['useful_horizon']/1000
        im=ax.imshow(values,vmin=0,vmax=100,aspect='auto',cmap='viridis')
        ax.set_title(LABELS[model],fontsize=10)
        ax.set_yticks(range(len(names)),names)
        ax.set_xticks(range(len(columns)),[f'{w}\nseed {seed}' for w,seed in columns],fontsize=8)
    fig.colorbar(im,ax=axes,label='Thousands of further updates with both informative bounds')
    fig.savefig(output/'coverage.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for key,values in dev:
        c=values['generic']; time=c['t']/.002/1000
        axes[0].plot(time,np.sqrt(c['concentration']),label=key[1])
        axes[1].plot(time,c['clock_ratio'])
    axes[0].set(ylabel=r'$\sqrt{\chi_6}$: concentration factor')
    axes[1].set(ylabel='Accumulated concentration\nrelative to window average')
    axes[1].axhline(2.,color='k',ls=':',label='Factor-two diagnostic')
    for ax in axes:ax.set_xlabel('Further updates (thousands)')
    axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(output/'concentration.png',dpi=170);plt.close(fig)


def run(inputs,output):
    output.mkdir(parents=True,exist_ok=False)
    trajectories=load(inputs)
    target_cache={name:target_data(name) for name in {r[0]['target'] for r in trajectories}}
    summaries=[];curve_store={};audits=[];sampling=[]
    for rows in trajectories:
        x,y,target=target_cache[rows[0]['target']]
        results,curves=evaluate(rows,target)
        summaries.extend(results)
        key=tuple(rows[0][k] for k in ('study','target','width','seed','kind','dt'))
        curve_store[key]=curves
        audits.extend(snapshot_audit(rows,target,(x,y)))
        if rows[0]['study']=='development':
            thin,_=evaluate(rows[::2],target)
            for full,half in zip(results,thin,strict=True):
                sampling.append(dict(target=full['target'],kind=full['kind'],model=full['model'],
                    full_horizon=full['useful_horizon'],thin_horizon=half['useful_horizon'],
                    final_lambda_relative_change=abs(full['final_lambda_upper']-half['final_lambda_upper'])/full['final_lambda_upper']
                    if full['final_lambda_upper'] is not None and half['final_lambda_upper'] is not None else None))
        print(json.dumps(dict(case=key,horizons={s['model']:s['useful_horizon'] for s in results})),flush=True)
    cohorts={
        'W705_original':lambda s:s['width']==705 and s['seed'] in (30,31) and s['kind']=='gd' and s['dt']==.002,
        'W705_seed33':lambda s:s['width']==705 and s['seed']==33,
        'W1409':lambda s:s['width']==1409,
        'effective':lambda s:s['study']=='development' and s['kind']=='effective'}
    overview=[]
    for name,select in cohorts.items():
        for model in MODELS:
            rows=[s for s in summaries if select(s) and s['model']==model]
            if not rows:continue
            full=[s for s in rows if s['useful_horizon']==100000]
            overview.append(dict(cohort=name,model=model,count=len(rows),full_useful=len(full),
                minimum_horizon=min(s['useful_horizon'] for s in rows),
                median_horizon=float(np.median([s['useful_horizon'] for s in rows])),
                median_initial_force_slack=float(np.median([s['initial_force_slack'] for s in rows])),
                maximum_final_lambda=max((s['final_lambda_upper'] for s in full),default=None),
                minimum_final_error_floor=min((s['final_error_floor'] for s in full),default=None),
                factor2_clock_passes=sum(s['clock_factor2_horizon']==100000 for s in rows),
                maximum_clock_ratio=max(s['maximum_clock_over_window_mean'] for s in rows)))
    refinement=[]
    for fine in summaries:
        if fine['study']!='refinement':continue
        coarse=next(s for s in summaries if s['study']=='development' and
                    all(s[k]==fine[k] for k in ('target','kind','model')))
        refinement.append(dict(target=fine['target'],kind=fine['kind'],model=fine['model'],
            coarse_horizon=coarse['useful_horizon'],fine_horizon=fine['useful_horizon'],
            final_lambda_relative_change=abs(coarse['final_lambda_upper']-fine['final_lambda_upper'])/fine['final_lambda_upper']
            if coarse['final_lambda_upper'] is not None and fine['final_lambda_upper'] is not None else None))
    for name,rows in [('comparisons',summaries),('snapshot_audit',audits)]:
        with (output/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checks=dict(minimum_force_bound_ratio=min(s['minimum_force_slack'] for s in summaries),
        maximum_mass_over_bound=max(s['maximum_actual_mass_over_bound'] for s in summaries),
        maximum_residual_over_initial=max(s['max_sampled_residual_over_initial'] for s in summaries),
        maximum_force_reconstruction=max(s['force_reconstruction_relative'] for s in audits),
        maximum_cubic_identity_error=max(s['projected_cubic_identity_relative'] for s in audits),
        minimum_projected_jacobian_slack=min(s['jacobian_projected_slack'] for s in audits),
        maximum_signed_split_error=max(abs(s['signed_split_error']) for s in audits),
        generated_nonpositive_snapshots=int(sum(s['generated_imbalance']<=0 for s in audits)),
        target_exceeds_generated_snapshots=int(sum(s['target_imbalance']>abs(s['generated_imbalance']) for s in audits)),
        snapshot_count=len(audits))
    checks['maximum_sampling_endpoint_relative_change']=max(
        s['final_lambda_relative_change'] for s in sampling if s['final_lambda_relative_change'] is not None)
    checks['maximum_sampling_horizon_difference']=max(abs(s['full_horizon']-s['thin_horizon']) for s in sampling)
    checks['maximum_step_endpoint_relative_change']=max(
        s['final_lambda_relative_change'] for s in refinement if s['final_lambda_relative_change'] is not None)
    checks['maximum_step_horizon_difference']=max(abs(s['coarse_horizon']-s['fine_horizon']) for s in refinement)
    facts=dict(scope='Post-hoc structural validation of known trajectories; no held-out claims or new training',
        offset=5000,further_updates=100000,trajectories=len(trajectories),cohorts=overview,
        targets={k:v[2] for k,v in target_cache.items()},checks=checks,
        sampling_refinement=sampling,step_refinement=refinement,comparisons=summaries)
    (output/'facts.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    plots(summaries,curve_store,output)
    print(json.dumps(dict(cohorts=overview,checks=checks)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.inputs,args.output)
