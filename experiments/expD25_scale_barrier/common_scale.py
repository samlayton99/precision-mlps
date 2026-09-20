"""One-factor control: independent scales versus their uniform projection."""
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD25_scale_barrier import run as exp


def main():
    cfg=exp.configuration();torch.set_num_threads(2);cases={}
    with threadpool_limits(limits=2):
        for target in exp.old.TARGETS:
            for arm in ('gamma_1','gamma_4','gamma_16'):
                for mode in ('scale_only','shared_scale'):
                    key=f'{target}__{arm}__10000__{mode}';path=exp.RESULTS/'data'/f'{key}.npz'
                    if path.exists():
                        c=exp.load_case(path,cfg,geometry_mode=mode)
                    else:
                        c=exp.evaluate(exp.train(target,arm,10000,cfg,geometry_mode=mode),cfg)
                        exp.arrays_save(path,c,cfg)
                    cases[key]=c
                    print('COMMON',key,'GD/refit',c['evaluation'][-1,:2],'mean gamma',c['gamma'][-1,0],flush=True)
        plot(cases,cfg)


def plot(cases,cfg):
    plt=exp.graphics();from matplotlib.lines import Line2D
    for view in ('errors','scales'):
        fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True,sharey=(view=='errors'))
        for row,target in enumerate(exp.old.TARGETS):
            for col,arm in enumerate(('gamma_1','gamma_4','gamma_16')):
                ax=axes[row,col]
                for mode,color in [('scale_only','#2274b5'),('shared_scale','#e07826')]:
                    c=cases[f'{target}__{arm}__10000__{mode}']
                    if view=='errors':
                        ax.plot(c['steps'],np.maximum(c['evaluation'][:,0],1e-16),color=color,lw=1.6)
                        ax.plot(c['steps'],np.maximum(c['evaluation'][:,1],1e-16),color=color,lw=1.6,ls='--')
                    else:
                        s=np.arange(len(c['gamma']));ax.plot(s,c['gamma'][:,0],color=color,lw=1.6)
                        if mode=='scale_only':ax.fill_between(s,c['gamma'][:,2],c['gamma'][:,3],color=color,alpha=.15)
                if view=='errors':
                    ax.set(yscale='log',ylim=(1e-16,3))
                    ind=cases[f'{target}__{arm}__10000__scale_only']['evaluation'][-1,1]
                    shared=cases[f'{target}__{arm}__10000__shared_scale']['evaluation'][-1,1]
                    ax.text(.03,.05,f'Final refit: independent {ind:.2g}\nshared {shared:.2g}',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
                else:
                    lo,hi=(0,8) if arm=='gamma_1' else ((0,12) if arm=='gamma_4' else (12,20));ax.set_ylim(lo,hi)
                ax.set_xscale('symlog',linthresh=2);ax.set_xlim(0,cfg['steps']);ax.grid(alpha=.18)
                ax.set_xticks([0,2,10,100,2000],labels=['0','2','10','100','2000'])
                if row==0:ax.set_title('Initial γ = '+arm.split('_')[1])
                if col==0:ax.set_ylabel(exp.old.matched.rc.LABELS[target]+('\nRelative L₂ error' if view=='errors' else '\nMean gamma; shading = 10–90%'))
                if row==3:ax.set_xlabel('GD step (log spacing after 2)')
        fig.suptitle('Does preserving a common scale protect the approximation?',fontsize=19,y=.985)
        h=[Line2D([],[],color=c,lw=2,label=l) for c,l in [('#2274b5','Independent scales; fixed centers'),('#e07826','One shared scale; fixed centers')]]
        if view=='errors':h += [Line2D([],[],color='black',lw=1.5,label='Ordinary evaluation'),Line2D([],[],color='black',lw=1.5,ls='--',label='LS evaluation only')]
        fig.legend(handles=h,loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False)
        fig.subplots_adjust(top=.87,bottom=.12,left=.08,right=.98,wspace=.18,hspace=.23)
        fig.text(.5,.025,'Same centers, initial scales, readouts, samples, and loss. Geometry rate 20; readout rate 0.002. Only the permitted scale directions change.\n'
                 'Shared update = mean of the individual scale gradients, preserving equal slopes. Equivalent scalar-gamma learning rate is 20/177, not 20.\n'
                 'LS solves are offline evaluation only (SVD cutoff 10⁻¹³). All four functions train and evaluate on [−1,1].',ha='center',fontsize=9)
        fig.savefig(exp.FIGURES/f'common_scale_{view}.png');plt.close(fig)


if __name__=='__main__':main()
