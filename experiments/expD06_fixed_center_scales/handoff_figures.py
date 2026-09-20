"""Line plots for the controlled Newton, SSBroyden, and Adam handoff comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import core, diagnostics, joint_conditioning as campaign, run


COLORS={'newton':'#762a83','gn':'#1b7837','ssb_small':'#2166ac','ssb_default':'#d6604d','adam':'.45'}
NAMES={'newton':'Full-Hessian Newton','gn':'Gauss–Newton','ssb_small':'SSBroyden, guard 1e−30',
       'ssb_default':'SSBroyden, conservative guard','adam':'Continue Adam'}


def tag(case):
    if case['optimizer']=='ssbroyden':return 'ssb_small' if case['curvature_epsilon']<1e-20 else 'ssb_default'
    return case['optimizer']


def save(fig,path):
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def curves(record,root):
    with np.load(root/record['key']/'window_mse.npz') as a:steps=a['step'];mse=a['mean']
    with np.load(root/record['key']/'history.npz') as a:history=dict(a)
    return steps,mse,history


def style(ax,xlabel='Accepted updates after initialization',ylabel=None,log=True):
    ax.set_xscale('symlog',linthresh=1);ax.set_xlim(left=0)
    if log:ax.set_yscale('log')
    if ylabel:ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel);ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)


def plot_record(axes,record,root,label,color,linestyle='-'):
    steps,mse,h=curves(record,root)
    axes[0].plot(h['step'],h['train_mse'],label=label,color=color,ls=linestyle)
    axes[1].plot(h['step'],h['lambda_quantiles'][:,1],label=label,color=color,ls=linestyle)
    if record['status']['status']!='continuing':
        axes[0].plot(record['end'],record['status']['train_mse'],'x',color=color,ms=8)


def smooth(values):
    # Preserve early transients; place subsequent window medians at window ends.
    prefix=min(100,len(values));size=max(1,len(values)//200)
    starts=np.arange(prefix,len(values),size)
    return (np.r_[np.arange(1,prefix+1),np.minimum(starts+size,len(values))],
            np.r_[values[:prefix],[np.median(values[i:i+size]) for i in starts]])


def figures(output,reference,construction):
    records=json.loads((output/'summary.json').read_text());records=[r for r in records if r['end']]
    old=json.loads((reference/'summary.json').read_text()) if reference else []
    extras=json.loads((output/'optimizer_summary.json').read_text())
    dest=output/'figures';dest.mkdir(exist_ok=True)
    construction_mse=None
    if construction:
        with np.load(construction) as a:cc,cg,centers=a['c'],a['gamma'],a['centers']
        x=diagnostics.midpoint_grid(32768)
        construction_mse=float(np.mean((diagnostics.prediction(x,centers,cc,cg)-core.target(x,'sine',np))**2))
        run.write_json(output/'construction_check.json',dict(midpoint_points=len(x),mse=construction_mse,
            l2re=float(np.sqrt(construction_mse/np.mean(core.target(x,'sine',np)**2))),
            coefficient_source_sha256=hashlib.sha256(construction.read_bytes()).hexdigest(),evaluation='NumPy FP64, stored physical coefficients'))
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for col,kind in enumerate(('newton','gn','ssb_small')):
        pool=[(r,output) for r in records if tag(r['case'])==kind and r['case']['coordinates']=='parameter_scale']
        if kind=='gn':pool += [(r,reference) for r in old if r['case']['optimizer']=='gn' and r['case']['coordinates']=='parameter_scale' and r['case']['n']==512]
        for r,root in pool:
            warm=bool(r['case'].get('warm_start'));seed=r['case']['seed']
            plot_record(axes[:,col],r,root,f"{'Adam endpoint' if warm else 'Xavier'}, seed {seed}",
                        '#2166ac' if warm else '#d6604d','-' if seed==0 else '--')
        axes[0,col].set_title(NAMES[kind]);axes[1,col].axhline(.25,color='.5',ls=':')
        style(axes[0,col],ylabel='Training MSE at saved updates');style(axes[1,col],ylabel='Median core |lambda|',log=False)
    fig.suptitle('Initialization comparison, N=512; × = explicit numerical stop\nAdam endpoints follow 5.3 million updates; dotted bandwidth = construction reference')
    save(fig,dest/'initialization.png')

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for seed in (0,1):
        for r in records:
            c=r['case']
            if c['seed']!=seed or not c.get('warm_start'):continue
            kind=tag(c);plot_record(axes[:,seed],r,output,NAMES[kind],COLORS[kind])
        control=output/f'adam_seed_{seed}.npz'
        if control.exists():
            with np.load(control) as a:
                h=a['history'];trace=a['trace'];starts=np.arange(0,len(trace),1000)
                axes[0,seed].plot(starts+1000,[2*trace[i:i+1000,0].mean() for i in starts],color=COLORS['adam'],label=NAMES['adam'])
                axes[1,seed].plot(h[:,0],h[:,4],color=COLORS['adam'],label=NAMES['adam'])
        if construction_mse:axes[0,seed].axhline(construction_mse,color='.25',ls=':',label='Construction midpoint MSE')
        axes[0,seed].set_title(f'Seed {seed}');axes[1,seed].axhline(.25,color='.5',ls=':')
        style(axes[0,seed],ylabel='Training MSE');style(axes[1,seed],ylabel='Median core |lambda|',log=False)
    if construction_mse:
        upper=max(ax.get_ylim()[1] for ax in axes[0])
        for ax in axes[0]:ax.set_ylim(construction_mse*.3,upper)
    fig.suptitle('Same Adam physical endpoint, fresh higher-order optimizer states\nHigher-order curves: saved states; Adam: means over 1000 updates. Post-handoff counts have different costs')
    save(fig,dest/'handoffs.png')

    extended=[r for r in records if r['end']>20000 and r['case']['optimizer'] in ('newton','gn')]
    if extended:
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for r in extended:
            c=r['case'];seed=c['seed'];warm=bool(c.get('warm_start'))
            _,_,h=curves(r,output);baseline=h['train_mse'][h['step']==20000]
            if len(baseline)!=1:raise ValueError('Continuation plot requires the fixed 20k checkpoint')
            keep=h['step']>=20000
            label=f"{'Newton' if c['optimizer']=='newton' else 'GN'}, {'Adam start' if warm else 'Xavier start'}"
            color=COLORS[c['optimizer']] if warm else '#d6604d'
            axes[0,seed].plot(h['step'][keep],h['train_mse'][keep]/baseline[0],label=label,color=color)
            axes[1,seed].plot(h['step'][keep],h['lambda_quantiles'][keep,1],label=label,color=color)
        for seed in (0,1):
            axes[0,seed].set_title(f'Seed {seed}');axes[0,seed].set_yscale('log')
            axes[0,seed].axhline(1,color='.5',ls=':')
            axes[1,seed].axhline(.25,color='.5',ls=':',label='Construction bandwidth')
            for row,ylabel in enumerate(('MSE / same run at 20k','Median core |lambda|')):
                axes[row,seed].set(xlabel='Accepted updates after initialization',ylabel=ylabel,xlim=(20000,None))
                axes[row,seed].grid(alpha=.2);axes[row,seed].legend(fontsize=8)
        fig.suptitle('Continuation with optimizer state preserved; linear update axis\nRelative errors use each run\'s own 20k MSE; final endpoints may have unequal update counts')
        save(fig,dest/'continuation.png')

    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for r in records:
        c=r['case']
        if not c.get('warm_start'):continue
        with np.load(output/r['key']/'optimizer_trace.npz') as a:
            trace=a['trace'];ix=np.unique(np.linspace(0,len(trace)-1,min(len(trace),1000),dtype=int))
            axes[c['seed']].plot(trace[ix,campaign.HIGHER_COLUMNS.index('function_evaluations')],2*trace[ix,0],
                                 color=COLORS[tag(c)],label=NAMES[tag(c)])
    for seed in (0,1):
        axes[seed].set_title(f'Seed {seed}');style(axes[seed],xlabel='Post-handoff function evaluations',ylabel='Training MSE at sampled updates')
    fig.suptitle('Evaluation cost includes rejected proposals and explicit diagnostic evaluations\nAdam pretraining cost is 5.3 million full-batch updates before these axes begin')
    save(fig,dest/'evaluation_cost.png')

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for r in records:
        c=r['case']
        if tag(c)!='ssb_default' or c.get('warm_start'):continue
        physical=c['coordinates']=='physical';seed=c['seed']
        plot_record(axes[:,seed],r,output,'Unscaled physical parameters' if physical else 'Individual scales',
                    '#d6604d' if physical else '#2166ac')
    for seed in (0,1):
        axes[0,seed].set_title(f'Seed {seed}');axes[1,seed].axhline(.25,color='.5',ls=':')
        style(axes[0,seed],ylabel='Training MSE at saved updates');style(axes[1,seed],ylabel='Median core |lambda|',log=False)
    fig.suptitle('SSBroyden scaling control: identical physical Xavier initialization\nIdentity initial inverse Hessian in each coordinate system; same conservative guard and line search')
    save(fig,dest/'ssb_scaling.png')

    for filename,fields,ylabels in (
        ('gradients',('gradient_native_readout_norm','gradient_native_slope_norm'),('Readout gradient norm in prescribed units','Lambda gradient norm')),
        ('motion',('delta_c_over_alpha_rms','delta_lambda_rms'),('RMS readout update / alpha','RMS lambda update'))):
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for r in records:
            c=r['case'];kind=tag(c)
            if not c.get('warm_start'):continue
            with np.load(output/r['key']/'optimizer_trace.npz') as a:
                for row,field in enumerate(fields):
                    steps,values=smooth(a['details'][:,list(a['detail_columns']).index(field)])
                    axes[row,c['seed']].plot(steps,np.maximum(values,1e-300),label=NAMES[kind],color=COLORS[kind])
        for row in (0,1):
            for seed in (0,1):style(axes[row,seed],ylabel=ylabels[row]);axes[row,seed].set_title(f'Seed {seed}')
        fig.suptitle('Adam handoffs: first 100 updates, then consecutive-window medians; all methods use individual scales\nGradient norms use the same coordinates; motion is measured from accepted physical changes')
        save(fig,dest/f'{filename}.png')

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for r in records:
        c=r['case']
        if c['optimizer']!='newton':continue
        warm=bool(c.get('warm_start'));label='Adam endpoint' if warm else 'Xavier';color='#2166ac' if warm else '#d6604d';seed=c['seed']
        with np.load(output/r['key']/'optimizer_trace.npz') as a:
            for ax,values in ((axes[0,seed],a['trace'][:,campaign.HIGHER_COLUMNS.index('damping')]),
                              (axes[1,seed],a['details'][:,campaign.DETAIL_COLUMNS.index('radius')])):
                step,values=smooth(values);ax.plot(step,np.maximum(values,1e-300),color=color,label=label)
    for seed in (0,1):
        axes[0,seed].set_title(f'Seed {seed}')
        style(axes[0,seed],ylabel='Trust-region shift');style(axes[1,seed],ylabel='Trust radius in prescribed units')
    fig.suptitle('Full-Hessian Newton: numerical safeguards throughout training\nConsecutive-window medians; the zero-shift fraction is recorded separately')
    save(fig,dest/'newton_trust_region.png')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for r in records:
        c=r['case'];path=output/r['key']/'curvature_profile.npz'
        if c['optimizer']!='newton' or not path.exists():continue
        ax=axes[int(bool(c.get('warm_start'))),c['seed']]
        with np.load(path) as a:
            amplitude=a['amplitudes'];baseline=float(a['initial_mse'])
            for sign,color in ((1,'#2166ac'),(-1,'#d6604d')):
                mask=amplitude*sign>0;order=np.argsort(np.abs(amplitude[mask]));steps=np.abs(amplitude[mask])[order]
                ax.loglog(steps,(a['actual_mse'][mask]/baseline)[order],color=color,label=f'Actual, sign {sign:+d}')
                ax.loglog(steps,(a['residual_model_mse'][mask]/baseline)[order],color=color,ls='--',label=f'Residual model, sign {sign:+d}')
            ax.axvline(float(a['radius']),color='.4',ls=':',label='Stored next trust radius')
        ax.set(xlabel='Step length along minimum-curvature eigenvector',ylabel='Trial MSE / current MSE',
               title=f"{'Adam endpoint' if c.get('warm_start') else 'Xavier'} start, seed {c['seed']}")
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Detached Newton endpoint profiles; eigenvectors use prescribed parameter units\nDashed curves evaluate r + t Jv + ½t² r″[v,v]; neither curves nor sampled minima are training updates')
    save(fig,dest/'newton_curvature_profiles.png')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for extra in extras:
        c=extra['case']
        if tag(c)!='ssb_small' or c['coordinates']!='parameter_scale':continue
        rows=[s for s in extra['states'] if 'metric' in s]
        ax=axes[int(bool(c.get('warm_start'))),c['seed']]
        for k,label in enumerate(('Readout ← readout gradient','Readout ← gamma gradient','Gamma ← readout gradient','Gamma ← gamma gradient')):
            ax.plot([r['step'] for r in rows],[r['metric']['function_direction_norm'][k] for r in rows],label=label)
        style(ax,ylabel='Function-space norm of direction component')
        ax.set_title(f"{'Adam endpoint' if c.get('warm_start') else 'Xavier'}, seed {c['seed']}")
    fig.suptitle('SSBroyden coupling, curvature guard 1e−30\nUnit search directions before line search; component magnitudes do not measure net descent')
    save(fig,dest/'ssb_coupling.png')

    fig,axes=plt.subplots(3,3,figsize=(17,11),layout='constrained')
    for col,kind in enumerate(('newton','gn','ssb_small')):
        r=next((r for r in records if r['case']['seed']==0 and r['case'].get('warm_start') and tag(r['case'])==kind),None)
        if r is None:continue
        with np.load(output/r['key']/f"dense_{r['end']}.npz") as a:
            bounds=a['band_bounds'];labels=['DC' if lo==0 else str(lo) if hi==lo+1 else f'{lo}–{hi-1}' for lo,hi in bounds]
            energy=a['band_mse'].mean(axis=0)
            values=[100*energy/max(energy.sum(),1e-300),-a['band_readout_linear_mse_change'].mean(axis=0),-a['band_geometry_linear_mse_change'].mean(axis=0)]
            for row,v in enumerate(values):
                ax=axes[row,col];ax.bar(np.arange(len(v)),v,color=COLORS[kind]);ax.set_xticks(np.arange(len(v)),labels,rotation=60,ha='right',fontsize=8)
                ax.set_xlabel('DFT index magnitude; both signs combined');ax.grid(axis='y',alpha=.2)
                if row:
                    peak=max(np.max(np.abs(v)),1e-300);decade=10.**np.floor(np.log10(peak))
                    ax.set_yscale('symlog',linthresh=peak*1e-5);ax.set_ylim(-peak*1.15,peak*1.15)
                    ax.set_yticks([-decade,0,decade],[f'−{decade:.0e}','0',f'{decade:.0e}'])
                    ax.axhline(0,color='.4',lw=.7)
            axes[0,col].set_title(NAMES[kind])
    for ax in axes[0]:ax.set_ylabel('Share of mean residual MSE (%)')
    for ax in axes[1]:ax.set_ylabel('Readout predicted MSE reduction')
    for ax in axes[2]:ax.set_ylabel('Geometry predicted MSE reduction')
    fig.suptitle('Fourier evidence, Adam handoffs, seed 0, sampled final 2048 accepted updates\nPositive descent bars predict improvement; linear terms omit quadratic and joint interaction costs')
    save(fig,dest/'fourier_seed_0.png')

    for seed in (0,1):
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for col,kind in enumerate(('newton','gn','ssb_small')):
            r=next((r for r in records if r['case']['seed']==seed and r['case'].get('warm_start') and tag(r['case'])==kind),None)
            if r is None:continue
            with np.load(output/r['key']/'history.npz') as h:
                centers=h['centers']
                for ax,field in ((axes[0,col],'c'),(axes[1,col],'gamma')):
                    initial,final=h[field][0],h[field][-1]
                    if field=='c':initial,final=initial[1:],final[1:]
                    ax.scatter(centers,initial,color='.6',s=7,alpha=.6,label='Adam handoff')
                    ax.scatter(centers,final,color=COLORS[kind],s=6,alpha=.9,label='Optimizer endpoint')
                    for boundary in (-1,1):ax.axvline(boundary,color='.75',ls=':',lw=.7)
                    ax.set_yscale('symlog',linthresh=1e-3 if field=='c' else 1.)
                    ax.set(xlabel='Physical center',ylabel='Physical readout w' if field=='c' else 'Signed physical gamma')
                    ax.grid(alpha=.2);ax.legend(fontsize=8)
                if construction:
                    axes[0,col].plot(centers,cc[1:],'k:',lw=1,label='Construction, different geometry')
                    axes[1,col].plot(centers,cg,'k:',lw=1,label='Construction')
                    axes[0,col].legend(fontsize=7);axes[1,col].legend(fontsize=7)
                axes[0,col].set_title(NAMES[kind])
        fig.suptitle(f'Parameters remain attached to their physical centers, seed {seed}\nConstruction coefficients belong to its uniform geometry; no coefficient-matching objective was trained')
        save(fig,dest/f'parameters_seed_{seed}.png')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--construction',type=Path)
    a=p.parse_args();figures(a.output,a.reference,a.construction)


if __name__=='__main__':main()
