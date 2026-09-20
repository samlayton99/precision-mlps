"""Classical control: only the shared-scale optimizer changes to Adam."""
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
            for arm in ('gamma_1','gamma_16'):
                for optimizer in ('sgd','adam'):
                    name=f'{target}__{arm}__1__shared_scale__10000steps'+('__adam' if optimizer=='adam' else '')
                    path=exp.RESULTS/'data'/f'{name}.npz'
                    if path.exists():c=exp.load_case(path,cfg,geometry_mode='shared_scale',geometry_optimizer=optimizer)
                    else:
                        c=exp.evaluate(exp.train(target,arm,1,cfg,geometry_mode='shared_scale',geometry_optimizer=optimizer),cfg)
                        exp.arrays_save(path,c,cfg)
                    cases[target+'__'+arm+'__'+optimizer]=c
                    print('ADAPTIVE',target,arm,optimizer,'GD/refit',c['evaluation'][-1,:2],'gamma',c['gamma'][-1,0],flush=True)
        plot(cases,cfg)


def plot(cases,cfg):
    plt=exp.graphics();from matplotlib.lines import Line2D
    for arm in ('gamma_1','gamma_16'):
        fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True)
        for row,target in enumerate(exp.old.TARGETS):
            for optimizer,color in [('sgd','#482878'),('adam','#de7525')]:
                c=cases[target+'__'+arm+'__'+optimizer]
                axes[row,0].plot(np.arange(len(c['gamma'])),c['gamma'][:,0],color=color,lw=1.7)
                for col,field in [(1,1),(2,0)]:axes[row,col].plot(c['steps'],np.maximum(c['evaluation'][:,field],1e-16),color=color,lw=1.7)
            a=cases[target+'__'+arm+'__adam']
            for ax in axes[row]:
                ax.set_xscale('symlog',linthresh=2);ax.set_xlim(0,cfg['steps']);ax.grid(alpha=.18)
                ax.set_xticks([0,2,10,100,1000,10000],labels=['0','2','10','100','1k','10k'])
                if row==3:ax.set_xlabel('Training step (log spacing after 2)')
            axes[row,0].set(ylim=(0,40),ylabel=exp.old.matched.rc.LABELS[target]+'\nCommon gamma')
            axes[row,1].set(yscale='log',ylim=(1e-16,2),ylabel='Refitted relative L₂')
            axes[row,2].set(yscale='log',ylim=(1e-4,2),ylabel='Actual relative L₂')
            axes[row,0].text(.03,.92,f'Final Adam γ: {a["gamma"][-1,0]:.3f}',transform=axes[row,0].transAxes,va='top',fontsize=10)
            for col,field in [(1,1),(2,0)]:axes[row,col].text(.03,.08,f'Final Adam error: {a["evaluation"][-1,field]:.3g}',transform=axes[row,col].transAxes,fontsize=9)
        for ax,title in zip(axes[0],['Can an adaptive step move gamma?','Does the geometry improve or deteriorate?','What error does ordinary readout GD leave?']):ax.set_title(title,pad=12,fontsize=11)
        fig.suptitle('Change only the shared-scale optimizer — initial γ='+arm.split('_')[1],fontsize=19,y=.987)
        fig.legend([Line2D([],[],color=c,lw=2) for c in ['#482878','#de7525']],['Shared scale: SGD, rate 0.002','Shared scale: Adam, rate 0.002'],loc='upper center',bbox_to_anchor=(.5,.953),ncol=2,frameon=False)
        fig.subplots_adjust(top=.88,bottom=.12,left=.08,right=.98,hspace=.24,wspace=.28)
        fig.text(.5,.025,'Same initial parameters, fixed uniform centers, zero readout, samples, and loss. Readout optimizer stays SGD at 0.002 in both runs.\n'
                 'The shared geometry gradient is the mean per-neuron gradient. Adam uses standard betas (0.9, 0.999) and epsilon 10⁻⁸; no large learning-rate multiplier.\n'
                 'LS solves are evaluation only, including step 0; same finite samples, SVD cutoff 10⁻¹³. No solved coefficients enter training.',ha='center',fontsize=9)
        fig.savefig(exp.FIGURES/f'adaptive_{arm}.png');plt.close(fig)


if __name__=='__main__':main()
