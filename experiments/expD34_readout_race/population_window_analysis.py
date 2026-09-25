"""Short-window calibration and future-premise audits; run on Modal CPU.

Writes numerical evidence and figures only. Future diagnostics assess premises
and separately account for tracking; they never fit window coefficients.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp


def cumulative(t, y):
    return cumulative_trapezoid(y,t,initial=0.)


def calibration(rows, offset=5000, method='variation', factor=2.):
    window = [r for r in rows if r['offset'] <= offset]
    if not window or window[-1]['offset'] != offset:
        raise ValueError('Calibration endpoint is missing')
    t = np.array([r['time'] for r in window]); t -= t[0]
    get = lambda key: np.array([r[key] for r in window])
    r,c,q = get('rotation'),get('coefficient'),get('q')
    Lr,Lc=0.,0.
    if method == 'motion':
        alpha,K=max(0,r[-1]),max(0,c[-1])
        travel=cumulative(t,q)[-1]
        # A decreasing channel can turn upward later. Its measured variation
        # remains information; a positive-part slope would erase it.
        Lr=factor*np.sum(np.abs(np.diff(r)))/travel
        Lc=factor*np.sum(np.abs(np.diff(c)))/travel
    elif method == 'instant':
        alpha,K = max(0,r[-1]),max(0,c[-1])
    else:
        alpha = max(0,float(np.max(cumulative(t,r)[1:]/t[1:])))
        K = max(0,float(np.max(cumulative(t,q*c)[1:]/cumulative(t,q)[1:])))
        if method == 'variation':
            alpha += factor*np.sum(np.abs(np.diff(r)))
            K += factor*np.sum(np.abs(np.diff(c)))
        elif method != 'prefix': raise ValueError(method)
    return dict(alpha=float(alpha),K=float(K),Lr=float(Lr),Lc=float(Lc),q0=float(q[-1]),
                kappa0=float(window[-1]['kappa']),offset=offset,method=method,factor=factor)


def comparison(t, q0, kappa0, alpha, K, zeta=None, delta=None, Lr=0., Lc=0.):
    """Integrate scaled travel, log-force ratio, and scaled dissipated energy."""
    t = np.asarray(t)
    zeta = np.zeros_like(t) if zeta is None else np.asarray(zeta)
    delta = np.zeros_like(t) if delta is None else np.asarray(delta)
    integrated_delta = cumulative(t,delta)
    def rhs(s,v):
        gain = np.exp(min(v[1],40.))
        a=q0*v[0]
        return [gain,kappa0+alpha*s+K*a+Lr*q0*v[3]+.5*Lc*a*a
                +np.interp(s,t,zeta+integrated_delta),gain*gain,v[0]]
    def stop(s,v): return 30.-v[1]
    stop.terminal = True
    answer = solve_ivp(rhs,(0,float(t[-1])),[0.,0.,0.,0.],rtol=2e-10,
                       atol=1e-12,t_eval=t,events=stop,max_step=2.)
    n = len(answer.t)
    travel,force,energy = (np.full(len(t),np.nan) for _ in range(3))
    travel[:n] = q0*answer.y[0]
    force[:n] = q0*np.exp(answer.y[1])
    energy[:n] = q0*q0*answer.y[2]
    return travel,force,energy


def evaluate(rows, spec):
    future = [r for r in rows if spec['offset'] <= r['offset'] <= spec['offset']+100000]
    get = lambda k: np.array([r[k] for r in future])
    t=get('time'); t-=t[0]
    q=get('q'); actual_travel=cumulative(t,q)
    ir=cumulative(t,get('rotation')); ic=cumulative(t,get('state_change'))
    rbudget=spec['alpha']*t+spec['Lr']*cumulative(t,actual_travel)
    cbudget=spec['K']*actual_travel+.5*spec['Lc']*actual_travel**2
    dr=ir-rbudget; dc=ic-cbudget
    numerical_scale = np.maximum(np.abs(ir)+np.abs(ic)+spec['alpha']*t+spec['K']*actual_travel,1e-300)
    split_valid=(dr <= 128*np.finfo(float).eps*numerical_scale)&(dc <= 128*np.finfo(float).eps*numerical_scale)
    # The proof needs the sum. Separate conditions are useful diagnostics but
    # needlessly reject compensation between their available budgets.
    valid=dr+dc <= 128*np.finfo(float).eps*numerical_scale
    is_gd = future[0]['kind']=='gd'
    delta = get('tracking_kappa_drift') if is_gd else np.zeros_like(t)
    zeta = get('tracking_log_rate') if is_gd else np.zeros_like(t)
    travel,force,energy = comparison(t,spec['q0'],spec['kappa0'],spec['alpha'],spec['K'],zeta,delta,spec['Lr'],spec['Lc'])
    pure_travel,pure_force,_ = comparison(t,spec['q0'],spec['kappa0'],spec['alpha'],spec['K'],Lr=spec['Lr'],Lc=spec['Lc'])
    tracking_path = cumulative(t,get('tracking_norm')) if is_gd else np.zeros_like(t)
    adverse_energy = cumulative(t,np.maximum(0,get('tracking_energy'))) if is_gd else np.zeros_like(t)
    floor = np.sqrt(np.maximum(0,future[0]['Y2']-2*energy-2*adverse_energy))/future[0]['target_norm']
    lamb = future[0]['lambda_rms']+future[0]['h']*(travel+tracking_path)/np.sqrt(future[0]['width'])
    bad = np.flatnonzero(~valid)
    horizon = t[bad[0]-1]/.002 if len(bad) else t[-1]/.002
    split_bad=np.flatnonzero(~split_valid)
    split_horizon=t[split_bad[0]-1]/.002 if len(split_bad) else t[-1]/.002
    reconstructed_k = spec['kappa0']+ir+ic+cumulative(t,delta)
    reconstructed_log = cumulative(t,get('kappa')+zeta)
    numerical_log_allowance = (np.maximum.accumulate(abs(np.log(q/q[0])-reconstructed_log))
        +cumulative(t,np.maximum.accumulate(abs(get('kappa')-reconstructed_k))))
    upper = (force >= q*np.exp(-numerical_log_allowance-128*np.finfo(float).eps))&np.isfinite(force)
    good = valid & upper & (lamb < .25) & (floor > .01)
    failed = np.flatnonzero(~good)
    useful = t[failed[0]-1]/.002 if len(failed) else t[-1]/.002
    summary = {k:future[0][k] for k in ('study','target','seed','width','nref','kind','dt')}
    summary.update(**spec,premise_horizon=horizon,split_premise_horizon=split_horizon,useful_horizon=useful,
        final_force_ratio=float(q[-1]/q[0]),final_force_upper_ratio=float(force[-1]/q[0]),
        maximum_force_excess_ratio=float(np.nanmax(q/force)),
        final_pure_force_upper_ratio=float(pure_force[-1]/q[0]),
        final_lambda_upper=float(lamb[-1]),final_lambda_actual=float(get('lambda_rms')[-1]),
        final_error_floor=float(floor[-1]),final_error_actual=float(get('relative_eval_error')[-1]),
        final_rotation_integral=float(ir[-1]),final_state_integral=float(ic[-1]),
        final_tracking_log_budget=float(cumulative(t,zeta+cumulative(t,delta))[-1]),
        max_rotation_excess=float(max(dr)),max_state_excess=float(max(dc)),
        kappa_reconstruction_error=float(max(abs(get('kappa')-reconstructed_k))),
        logq_reconstruction_error=float(max(abs(np.log(q/q[0])-reconstructed_log))),
        numerical_log_allowance=float(numerical_log_allowance[-1]),
        calibration_to_forecast_ratio=100000/spec['offset'])
    curves = dict(time=t,offset=get('offset'),q=q,force=force,pure_force=pure_force,
        lambda_upper=lamb,lambda_actual=get('lambda_rms'),error_floor=floor,
        error_actual=get('relative_eval_error'),rotation_integral=ir,state_integral=ic,
        kappa=get('kappa'),kappa_reconstructed=reconstructed_k,
        valid_prefix=np.logical_and.accumulate(valid),actual_travel=actual_travel,travel_upper=travel,
        window_time=np.array([r['time']-future[0]['time'] for r in rows if r['offset']<=spec['offset']]),
        window_q=np.array([r['q'] for r in rows if r['offset']<=spec['offset']]))
    return summary,curves


def load(paths):
    groups = defaultdict(list)
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                for key in row:
                    if key not in ('study','target','kind','cold'): row[key]=float(row[key])
                key=tuple(row[k] for k in ('study','target','seed','width','kind','dt'))
                groups[key].append(row)
    return [sorted(rows,key=lambda r:r['offset']) for rows in groups.values()]


def plots(primary, output):
    import matplotlib.pyplot as plt
    dev=[(s,c) for s,c in primary if s['study']=='development' and s['kind']=='gd']
    if not dev: return
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for ax,(s,c) in zip(axes.flat,dev):
        ax.axvspan(-5,0,color='0.9',label='Calibration')
        ax.plot(c['window_time']/.002/1000,c['window_q']/s['q0'],color='0.4')
        ax.plot(c['time']/.002/1000,c['q']/s['q0'],label='Observed GD')
        ax.plot(c['time']/.002/1000,c['force']/s['q0'],'--',label='Window bound + tracking')
        ax.set(title=s['target'],xlabel='Updates after 5k window (thousands)',ylabel='Effective force / window endpoint')
        ax.set_ylim(bottom=0,top=min(10,max(1.1,1.15*np.nanmax(c['q']/s['q0']))))
        if s['premise_horizon']<100000: ax.axvline(s['premise_horizon']/1000,color='red',ls=':',label='Premise first fails')
    axes.flat[0].legend(fontsize=8)
    fig.savefig(output/'window_forecasts.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for ax,(s,c) in zip(axes.flat,dev):
        ax.plot(c['time']/.002/1000,c['rotation_integral'],label='Rotation')
        ax.plot(c['time']/.002/1000,c['state_integral'],label='Loaded operator change')
        ax.plot(c['time']/.002/1000,c['kappa']-s['kappa0'],color='black',label='Net reinforcement change')
        ax.set(title=s['target'],xlabel='Updates after window (thousands)',ylabel='Accumulated change in growth rate')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes.flat[0].legend(fontsize=8)
    fig.savefig(output/'reinforcement_sources.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for s,c in dev:
        line,=axes[0].plot(c['time']/.002/1000,c['lambda_actual'],label=s['target'])
        axes[0].plot(c['time'][c['valid_prefix']]/.002/1000,c['lambda_upper'][c['valid_prefix']],
                     '--',color=line.get_color())
        axes[1].plot(c['time']/.002/1000,c['error_actual'],color=line.get_color())
        axes[1].plot(c['time'][c['valid_prefix']]/.002/1000,c['error_floor'][c['valid_prefix']],
                     '--',color=line.get_color())
    axes[0].axhline(.25,color='black',ls=':',label='Desired scale 0.25')
    axes[0].set(yscale='log',ylim=(1e-4,1),ylabel='Normalized slope RMS')
    axes[1].axhline(.01,color='black',ls=':',label='1% error')
    axes[1].set(ylabel='Raw relative output error',ylim=(0,1.05))
    for ax in axes: ax.set_xlabel('Updates after window (thousands)')
    axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(output/'population_output.png',dpi=170); plt.close(fig)
    gd=[s for s,_ in primary if s['kind']=='gd' and s['dt']==.002]
    names=sorted({s['target'] for s in gd})
    columns=[(705,30),(705,31),(705,33),(177,31),(1409,31)]
    values=np.full((len(names),len(columns)),np.nan)
    for s in gd:
        if (s['width'],s['seed']) in columns:
            values[names.index(s['target']),columns.index((s['width'],s['seed']))]=s['premise_horizon']/1000
    fig,ax=plt.subplots(figsize=(7,max(4,len(names)*.25)),layout='constrained')
    im=ax.imshow(values,vmin=0,vmax=100,aspect='auto',cmap='viridis')
    ax.set_yticks(range(len(names)),names)
    ax.set_xticks(range(len(columns)),[f'W={w}\nseed {s}' for w,s in columns])
    ax.set_title('Duration of the accumulated premise after the 5k window')
    fig.colorbar(im,ax=ax,label='Thousands of additional updates (100 = full interval)')
    fig.savefig(output/'coverage.png',dpi=170); plt.close(fig)


def run(inputs, output):
    output.mkdir(parents=True,exist_ok=True)
    summaries=[]; primary=[]
    for rows in load(inputs):
        for offset in (2000,5000,10000):
            for method in ('instant','prefix','variation','motion'):
                factors=(1.,2.,4.) if method in ('variation','motion') else (0.,)
                for factor in factors:
                    s,c=evaluate(rows,calibration(rows,offset,method,factor))
                    summaries.append(s)
                    if offset==5000 and method=='motion' and factor==2.:
                        primary.append((s,c))
    with (output/'comparisons.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    facts=dict(primary=[s for s,_ in primary],
        scope='Window-calibrated structural conditions; GD uses separately measured future tracking; sampled evidence, not interval certificates',
        full_premise=int(sum(s['premise_horizon']==100000 for s,_ in primary)),
        full_useful=int(sum(s['useful_horizon']==100000 for s,_ in primary)),count=len(primary))
    (output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    plots(primary,output)
    print(json.dumps(facts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.inputs,args.output)
