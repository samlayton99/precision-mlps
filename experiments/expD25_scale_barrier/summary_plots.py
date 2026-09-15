"""Focused presentation of the two sequential, center-fixed interventions."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD25_scale_barrier import run as exp


def main():
    plt=exp.graphics();from matplotlib.lines import Line2D
    arms=[(1,'scale_only','#482878','Independent scales, original rate 0.002'),
          (10000,'scale_only','#21918c','Independent scales, larger rate 20'),
          (10000,'shared_scale','#e07826','Shared scale, same larger rate 20')]
    cfg=exp.configuration();fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True)
    for row,target in enumerate(exp.old.TARGETS):
        for multiplier,mode,color,label in arms:
            case=exp.load_case(exp.RESULTS/'data'/f'{target}__gamma_1__{multiplier}__{mode}.npz',cfg,geometry_mode=mode)
            axes[row,0].plot(np.arange(len(case['gamma'])),case['gamma'][:,0],color=color,lw=1.8)
            axes[row,1].plot(case['steps'],np.maximum(case['evaluation'][:,1],1e-16),color=color,lw=1.8)
            axes[row,2].plot(case['steps'],case['evaluation'][:,0],color=color,lw=1.8)
        for ax in axes[row]:
            ax.set_xscale('symlog',linthresh=2);ax.set_xlim(0,2000);ax.grid(alpha=.18)
            ax.set_xticks([0,2,10,100,2000],labels=['0','2','10','100','2000'])
            if row==3:ax.set_xlabel('GD step (log spacing after 2)')
        axes[row,0].set(ylim=(0,6),ylabel=exp.old.matched.rc.LABELS[target]+'\nMean gamma')
        axes[row,1].set(yscale='log',ylim=(1e-16,2),ylabel='Relative L₂ with solved readout')
        axes[row,2].set(yscale='log',ylim=(.03,1.2),ylabel='Actual trained-readout relative L₂')
    for ax,title in zip(axes[0],['Can gamma move with centers fixed?','Does the available approximation improve?','What does the current readout achieve?']):ax.set_title(title,pad=12,fontsize=11)
    fig.suptitle('Small gamma is escapable: separate mobility, geometry quality, and readout fitting',fontsize=18,y=.988)
    fig.legend([Line2D([],[],color=a[2],lw=2) for a in arms],[a[3] for a in arms],loc='upper center',bbox_to_anchor=(.5,.951),ncol=1,frameon=False,fontsize=10)
    fig.subplots_adjust(top=.85,bottom=.12,left=.08,right=.98,wspace=.25,hspace=.24)
    fig.text(.5,.025,'Every run starts at γ=1 with the same fixed uniform centers and zero readout. Purple→teal changes only scale learning rate; teal→orange only constrains scales to remain equal.\n'
             'All readouts train by GD at 0.002; no readout solves enter training. Shared update is the mean per-neuron gradient, not its sum.\n'
             'Middle: independent readout solve on 1,024 training midpoints, evaluated on 8,192 other midpoints in [−1,1]. SVD cutoff 10⁻¹³; coefficient and cutoff audits are separate.',ha='center',fontsize=9)
    fig.savefig(exp.FIGURES/'scale_escape.png');plt.close(fig)


if __name__=='__main__':main()
