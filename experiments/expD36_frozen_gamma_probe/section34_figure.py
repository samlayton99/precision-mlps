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
    reference_source=p.add_mutually_exclusive_group()
    reference_source.add_argument('--frozen-adam',type=Path)
    reference_source.add_argument('--frozen-analysis',type=Path,
        help='Use validated compact uniform-access analysis instead of copying full raw Adam traces')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((args.analysis/'summary.json').read_text())
    trace=np.load(args.analysis/'selected_traces.npz')
    actual=np.load(args.gd/'raw_error.npy',mmap_mode='r')
    horizon=summary['horizon']
    if len(actual)!=horizon+1:raise ValueError('Panel A and joint horizons must match')
    style();plt.rcParams.update({'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7})
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.9),layout='constrained')
    bandwidths=[.03125,.0625,.125,.25]
    palette=['#332288','#0072B2','#009E73','#CC6677']
    indices=np.unique(np.r_[0,np.geomspace(1,horizon,2300).astype(int),np.linspace(0,horizon,1800).astype(int)])
    for i,(lam,color) in enumerate(zip(bandwidths,palette)):
        values=np.load(args.bounds/f'lambda{lam:g}_q16_p24.npz')
        if values['steps'][-1]!=horizon:raise ValueError('Bound horizon mismatch')
        axes[0].plot(indices,actual[indices,i],color=color,lw=1.45)
        axes[0].plot(values['steps'],values['lower_error'],color=color,ls='--',lw=1.25)
    handles=[Line2D([],[],color=c,label=label) for c,label in zip(palette,['1/32','1/16','1/8','1/4'])]
    legend=axes[0].legend(handles=handles,title='Bandwidth λ',title_fontsize=7,loc='lower left',ncol=2,handlelength=1.3,columnspacing=.6)
    axes[0].add_artist(legend)
    axes[0].legend(handles=[Line2D([],[],color='black',label='GD'),Line2D([],[],color='black',ls='--',label='Bound')],loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,handlelength=1.5,columnspacing=.7)
    axes[0].set_xscale('symlog',linthresh=10);axes[0].set_xlabel('Readout updates');axes[0].set_ylabel('Relative output error')
    colors={'adam':'#0072B2','gd':'#D55E00'}
    for optimizer in summary['selected']:
        color=colors.get(optimizer,'#009E73');label='Adam' if optimizer=='adam' else optimizer.upper()
        t=trace[f'{optimizer}_error_steps']/1e6
        y=trace[f'{optimizer}_error_median']
        axes[1].plot(t,np.median(y,axis=1),color=color,lw=1.6,label=f'Joint {label}')
        axes[1].fill_between(t,np.min(trace[f'{optimizer}_error_low'],axis=1),np.max(trace[f'{optimizer}_error_high'],axis=1),color=color,alpha=.12,lw=0)
        for name,ls in [('rms','-'),('q99','--')]:
            if name=='rms':
                ts=trace[f'{optimizer}_rms_steps'];ys=trace[f'{optimizer}_rms_median']
                low=trace[f'{optimizer}_rms_low'];high=trace[f'{optimizer}_rms_high']
            else:
                ts=trace[f'{optimizer}_checkpoint_steps'];ys=trace[f'{optimizer}_lambda_q99']
                low=high=ys
            axes[2].plot(ts/1e6,np.median(ys,axis=1),color=color,ls=ls,lw=1.5,label=f'{label} '+('RMS' if name=='rms' else '99th'))
            axes[2].fill_between(ts/1e6,np.min(low,axis=1),np.max(high,axis=1),color=color,alpha=.08,lw=0)
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
        axes[1].plot(values['steps']/1e6,values['median'][:,gi],color='#333333',ls='--',lw=1.2,label='Fixed Adam')
        axes[1].fill_between(values['steps']/1e6,values['low'][:,gi],values['high'][:,gi],color='#333333',alpha=.08,lw=0)
        reference=dict(source=str(args.frozen_adam),geometry_index=gi,recipe_index=ri,recipe=meta['recipes'][ri],validation_error=float(errors[ri]))
    if args.frozen_analysis:
        access=json.loads((args.frozen_analysis/'summary.json').read_text())
        if access['assay_steps']!=horizon:raise ValueError('Frozen Adam analysis horizon mismatch')
        if args.base is not None:
            manifest=json.loads((args.base/'manifest.json').read_text())
            if manifest['input_sha256']!=access['input_sha256']:raise ValueError('Frozen Adam analysis uses different inputs')
        gi=next(i for i,row in enumerate(access['geometries']) if row['family']=='uniform' and np.isclose(row['lambda_rms'],.25))
        row=access['geometries'][gi];selected=row['selected']
        if not np.isfinite(selected['validation_error']):raise ValueError('No finite uniform Adam reference')
        curves=np.load(args.frozen_analysis/'assay_traces.npz')
        t=curves[f'g{gi}_steps']
        if t[-1]!=horizon:raise ValueError('Frozen Adam display trace is incomplete')
        axes[1].plot(t/1e6,curves[f'g{gi}_median'],color='#333333',ls='--',lw=1.2,label='Fixed Adam')
        axes[1].fill_between(t/1e6,curves[f'g{gi}_low'],curves[f'g{gi}_high'],color='#333333',alpha=.08,lw=0)
        reference=dict(source=str(args.frozen_analysis),geometry_index=gi,recipe_index=selected['recipe_index'],
            recipe={k:selected[k] for k in ['schedule','learning_rate']},validation_error=selected['validation_error'])
    axes[1].plot(indices/1e6,actual[indices,-1],color='#777777',ls=':',lw=1.2,label='Fixed GD')
    axes[2].axhline(.25,color='#333333',ls=':',lw=1,label='Reference 1/4')
    axes[1].set_ylabel('Relative output error');axes[2].set_ylabel(r'Scaled slope, $h|a_j|$')
    for ax in axes:
        ax.set_yscale('log');ax.grid(alpha=.13,which='major');ax.tick_params(labelsize=7)
    for ax in axes[1:]:
        ax.set_xlim(0,horizon/1e6);ax.set_xlabel('Updates (millions)');ax.legend(loc='upper right' if ax==axes[1] else 'lower right')
    for ax,title in zip(axes,['A  Frozen readout','B  Joint training','C  Slope acquisition']):ax.set_title(title,loc='left',pad=24)
    axes[0].set_xlim(0,horizon)
    axes[0].set_xticks([0,100,10000,1000000],['0',r'$10^2$',r'$10^4$',r'$10^6$'])
    for suffix in ['pdf','png','svg']:fig.savefig(args.output/f'section34_three_panel.{suffix}',dpi=300)
    plt.close(fig)
    (args.output/'figure_provenance.json').write_text(json.dumps(dict(joint_analysis=str(args.analysis),bound_source=str(args.bounds),executed_gd_source=str(args.gd),frozen_adam_reference=reference,horizon=horizon,display='Error and RMS lines use seedwise within-bin medians; their shading retains all raw seedwise extrema in each bin. The 99th-percentile slopes use saved parameter checkpoints and show their seed range. Slopes use fixed reference spacing 2/467.'),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
