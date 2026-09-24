"""Paper figure: theorem lower bounds, joint output error, and acquired slopes."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from .section34_analyze import binned_traces, style
except ImportError:
    from section34_analyze import binned_traces, style


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--bounds',type=Path,required=True)
    p.add_argument('--gd',type=Path,required=True)
    p.add_argument('--base',type=Path)
    p.add_argument('--frozen-adam',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.analysis/'summary.json').read_text())
    trace=np.load(args.analysis/'selected_traces.npz')
    actual=np.load(args.gd/'raw_error.npy',mmap_mode='r')
    horizon=summary['horizon']
    if len(actual)!=horizon+1:raise ValueError('Panel A and joint horizons must match')
    style();plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':7})
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.45),layout='constrained')
    bandwidths=[.03125,.0625,.125,.25]
    palette=['#332288','#0072B2','#009E73','#CC6677']
    indices=np.unique(np.r_[0,np.geomspace(1,horizon,2300).astype(int),np.linspace(0,horizon,1800).astype(int)])
    for i,(lam,color) in enumerate(zip(bandwidths,palette)):
        values=np.load(args.bounds/f'lambda{lam:g}_q16_p24.npz')
        if values['steps'][-1]!=horizon:raise ValueError('Bound horizon mismatch')
        axes[0].plot(indices,actual[indices,i],color=color,lw=1.45)
        axes[0].plot(values['steps'],values['lower_error'],color=color,ls='--',lw=1.25)
    handles=[Line2D([],[],color=c,label=f'λ = {lam:g}') for c,lam in zip(palette,bandwidths)]
    legend=axes[0].legend(handles=handles,loc='lower left',ncol=2,handlelength=1.5,columnspacing=.8)
    axes[0].add_artist(legend)
    axes[0].legend(handles=[Line2D([],[],color='black',label='Executed GD'),Line2D([],[],color='black',ls='--',label='Theorem lower bound')],loc='upper right',handlelength=2.3)
    axes[0].set_xscale('symlog',linthresh=10);axes[0].set_xlabel('Readout updates');axes[0].set_ylabel('Relative output L2 error')
    colors={'adam':'#0072B2','gd':'#D55E00'}
    for optimizer in summary['selected']:
        color=colors.get(optimizer,'#009E73');label='Adam' if optimizer=='adam' else optimizer.upper()
        t=trace[f'{optimizer}_error_steps']/1e6
        y=trace[f'{optimizer}_error_median']
        axes[1].plot(t,np.median(y,axis=1),color=color,lw=1.6,label=f'Joint {label}')
        axes[1].fill_between(t,np.min(trace[f'{optimizer}_error_low'],axis=1),np.max(trace[f'{optimizer}_error_high'],axis=1),color=color,alpha=.12,lw=0)
        for name,ls in [('rms','-'),('q99','--')]:
            if name=='rms':ts=trace[f'{optimizer}_rms_steps'];ys=trace[f'{optimizer}_rms_median']
            else:ts=trace[f'{optimizer}_checkpoint_steps'];ys=trace[f'{optimizer}_lambda_q99']
            axes[2].plot(ts/1e6,np.median(ys,axis=1),color=color,ls=ls,lw=1.5,label=f'{label} '+('RMS' if name=='rms' else '99th pct.'))
            axes[2].fill_between(ts/1e6,np.min(ys,axis=1),np.max(ys,axis=1),color=color,alpha=.08,lw=0)
    reference=None
    if args.frozen_adam:
        if args.base is None:p.error('--frozen-adam requires --base')
        base=np.load(args.base/'input.npz');manifest=json.loads((args.base/'manifest.json').read_text())
        meta=json.loads((args.frozen_adam/'metadata.json').read_text())
        weights=np.load(args.frozen_adam/'state.npz')['w']
        gi=next(i for i,row in enumerate(manifest['geometries']) if row['family']=='uniform' and np.isclose(row['lambda_rms'],.25))
        x=base['validation_x'];y=base['validation_target']
        phi=np.column_stack((np.tanh(x[:,None]*base['a'][gi]+base['b'][gi]),np.ones(len(x))))
        errors=np.linalg.norm(phi@weights[gi]-y[:,None],axis=0)/np.linalg.norm(y)
        ri=int(np.argmin(np.where(np.isfinite(errors),errors,np.inf)))
        if not np.isfinite(errors[ri]):raise ValueError('No finite uniform Adam reference')
        arr=np.load(args.frozen_adam/'relative_error.npy',mmap_mode='r')
        if len(arr)!=horizon+1:raise ValueError('Frozen Adam horizon mismatch')
        values=binned_traces(arr,ri)
        axes[1].plot(values['steps']/1e6,values['median'][:,gi],color='#333333',ls='--',lw=1.2,label='Frozen Adam, λ = 0.25')
        axes[1].fill_between(values['steps']/1e6,values['low'][:,gi],values['high'][:,gi],color='#333333',alpha=.08,lw=0)
        reference=dict(geometry_index=gi,recipe_index=ri,recipe=meta['recipes'][ri],validation_error=float(errors[ri]))
    axes[1].plot(indices/1e6,actual[indices,-1],color='#777777',ls=':',lw=1.2,label='Frozen GD, λ = 0.25')
    axes[2].axhline(.25,color='#333333',ls=':',lw=1,label='Uniform reference, λ = 0.25')
    axes[1].set_ylabel('Relative output L2 error');axes[2].set_ylabel('Slope × reference spacing, λ')
    for ax in axes:
        ax.set_yscale('log');ax.grid(alpha=.13,which='major');ax.tick_params(labelsize=8)
    for ax in axes[1:]:
        ax.set_xlim(0,horizon/1e6);ax.set_xlabel('Joint/readout updates (millions)' if ax==axes[1] else 'Joint-training updates (millions)');ax.legend(loc='best')
    for ax,title in zip(axes,['A  Frozen-readout learning','B  Joint-training output error','C  Acquired slope scale']):ax.set_title(title,loc='left',pad=10)
    axes[0].set_xlim(0,horizon)
    for suffix in ['pdf','png','svg']:fig.savefig(args.output/f'section34_three_panel.{suffix}',dpi=300)
    plt.close(fig)
    (args.output/'figure_provenance.json').write_text(json.dumps(dict(joint_analysis=str(args.analysis),bound_source=str(args.bounds),executed_gd_source=str(args.gd),frozen_adam_reference=reference,horizon=horizon,display='Seedwise within-bin medians; shading retains all seedwise extrema in every displayed bin. Slopes use fixed reference spacing 2/467.'),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
