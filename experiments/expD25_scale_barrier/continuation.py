"""Extend the same common-scale control without changing its optimizer."""
from pathlib import Path
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD25_scale_barrier import run as exp


def main():
    cfg=exp.configuration() | dict(steps=10000);torch.set_num_threads(2);cases={}
    with threadpool_limits(limits=2):
        for target in exp.old.TARGETS:
            for multiplier in (1,10000):
                path=exp.RESULTS/'data'/f'{target}__gamma_1__{multiplier}__shared_scale__10000steps.npz'
                if path.exists():c=exp.load_case(path,cfg,geometry_mode='shared_scale')
                else:
                    c=exp.evaluate(exp.train(target,'gamma_1',multiplier,cfg,geometry_mode='shared_scale'),cfg)
                    exp.arrays_save(path,c,cfg)
                if multiplier==10000:
                    prior=exp.load_case(exp.RESULTS/'data'/f'{target}__gamma_1__10000__shared_scale.npz',exp.configuration(),geometry_mode='shared_scale')
                    index=np.flatnonzero(c['steps']==2000)[0]
                    for k in ('a','b','v'):np.testing.assert_array_equal(c[k][index],prior[k][-1])
                cases[target+'__'+str(multiplier)]=c
                print('CONTINUED',target,multiplier,c['evaluation'][-1,:2],c['gamma'][-1,0],flush=True)
        plot(cases,cfg)


def plot(cases,cfg):
    plt=exp.graphics();from matplotlib.lines import Line2D
    fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True)
    colors=['#482878','#21918c']
    for row,target in enumerate(exp.old.TARGETS):
        y=exp.old.matched.target_values(target,exp.old.midpoint_grid(cfg['n_train']),cfg);energy=np.mean(y*y)
        for multiplier,color in zip((1,10000),colors):
            c=cases[target+'__'+str(multiplier)];steps=c['steps'];ev=c['evaluation']
            axes[row,0].plot(np.arange(len(c['gamma'])),c['gamma'][:,0],color=color,lw=1.7)
            # Both components use the same training norm, so their squares add.
            approximation=np.sqrt(2*ev[:,4]/energy);gap=np.sqrt(2*ev[:,5]/energy)
            axes[row,1].plot(steps,np.maximum(approximation,1e-16),color=color,lw=1.7)
            axes[row,2].plot(steps,np.maximum(gap,1e-16),color=color,lw=1.7)
        for ax in axes[row]:
            ax.set_xscale('symlog',linthresh=2);ax.set_xlim(0,cfg['steps']);ax.axvline(2000,color='#777777',ls=':',lw=1);ax.grid(alpha=.18)
            ax.set_xticks([0,2,10,100,1000,10000],labels=['0','2','10','100','1k','10k'])
            if row==3:ax.set_xlabel('GD step (log spacing after 2)')
        axes[row,0].set(ylim=(0,16),ylabel=exp.old.matched.rc.LABELS[target]+'\nCommon gamma')
        axes[row,1].set(yscale='log',ylim=(1e-16,2),ylabel='Approximation error / ‖y‖')
        axes[row,2].set(yscale='log',ylim=(1e-3,2),ylabel='Readout optimization gap / ‖y‖')
        fast=cases[target+'__10000'];end=fast['evaluation'][-1]
        axes[row,0].text(.03,.92,f'Final γ: {fast["gamma"][-1,0]:.3f}',transform=axes[row,0].transAxes,va='top',fontsize=10)
        axes[row,1].text(.03,.08,f'Final dense-grid LS error: {end[1]:.3g}',transform=axes[row,1].transAxes,fontsize=9)
        axes[row,2].text(.03,.08,f'Final ordinary error: {end[0]:.3g}',transform=axes[row,2].transAxes,fontsize=9)
    for ax,title in zip(axes[0],['Does gamma keep moving?','Error the geometry cannot numerically fit','Error left because readout GD is unfinished']):ax.set_title(title,pad=13,fontsize=11)
    fig.suptitle('Continue the same controlled escape; keep the two errors separate',fontsize=19,y=.987)
    fig.legend([Line2D([],[],color=c,lw=2) for c in colors],['Original per-neuron scale rate: 0.002','Larger per-neuron scale rate: 20'],loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False)
    fig.subplots_adjust(top=.88,bottom=.12,left=.08,right=.98,hspace=.24,wspace=.28)
    fig.text(.5,.025,'All runs start at γ=1 with fixed uniform centers and zero readout. Shared-scale update is the mean per-neuron gradient; effective scalar rate is shown rate / 177.\n'
             'Readout rate remains 0.002; no solves enter training. The dotted line marks 2,000 steps, whose parameters exactly reproduce the preceding fast runs.\n'
             'Middle/right use one training-sample SVD projector: ordinary error² = approximation component² + readout gap², verified to roundoff. Independent-grid LS errors are annotated.',ha='center',fontsize=9)
    fig.savefig(exp.FIGURES/'continuation.png');plt.close(fig)


if __name__=='__main__':main()
