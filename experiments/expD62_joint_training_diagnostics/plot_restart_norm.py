"""Measured B0 at each possible GD restart: three functions, two learning rates."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MaxNLocator
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.expD56_scale_acquisition.plot import apply_style,style_axis,COLORS
from experiments.expD56_scale_acquisition.run import case_id
BASE=ROOT/'results/checkpoint_D_optimizers/expD62_joint_training_diagnostics'
OUT=BASE/'restart_norm';WIDTH=196
TARGETS=[('square',r'$x^2$'),('sine_4pi',r'$\sin(4\pi x)$'),('runge_100',r'$1/(1+100x^2)$')]
RATES=[.001,.0001]

def main():
    OUT.mkdir(exist_ok=True);apply_style()
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.77,wspace=.16)
    fig.suptitle(r'GD: Starting Bound $B_0$',fontsize=13,fontweight='bold',y=.98)
    fig.text(.5,.865,'Width 196 · Default initialization',ha='center',fontsize=8,color='#555555')
    report={'width':WIDTH,'optimizer':'GD','definition':'B0(t)=sqrt(sum_j(gamma_j(t)^2+beta_j(t)^2+w_j(t)^2)); excludes output bias',
            'axes':'Linear x and y; shared y limits zoomed to observed range. Each point is a fresh restart norm, not a propagated ceiling.',
            'curves':[]};ys=[];handles=[]
    for ax,(name,label) in zip(axes,TARGETS):
        style_axis(ax);ax.set_title(label,pad=9,fontsize=10)
        for lr in RATES:
            path=BASE/'width196'/'runs'/case_id(name,1.,lr)/'checkpoint.npz'
            with np.load(path) as d:
                keys=d['keys'].tolist();steps=d['history'][:,keys.index('step')]
                values=np.linalg.norm(d['snapshots'][:,:-1],axis=1)
                np.testing.assert_allclose(values,np.sqrt(d['history'][:,keys.index('parameter_energy')]),rtol=1e-14)
            assert steps[0]==0 and steps[-1]==50000;ys.extend(values)
            line,=ax.plot(steps,values,color=COLORS[lr],lw=1.25,ls='-' if lr==.001 else (0,(4,2)))
            if name=='square':handles.append(line)
            report['curves'].append({'function':name,'learning_rate':lr,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'step':steps.tolist(),'B0':values.tolist()})
        ax.set_xlim(0,50000);ax.set_xticks([0,10000,20000,30000,40000,50000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda n,_:'0' if n==0 else f'{n/1000:g}k'))
        ax.set_xlabel('Restart step',labelpad=6)
    span=max(ys)-min(ys);axes[0].set_ylim(min(ys)-.08*span,max(ys)+.10*span)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5));axes[0].yaxis.set_major_formatter(FuncFormatter(lambda n,_:f'{n:.2f}'))
    axes[0].set_ylabel(r'Starting bound $B_0$',labelpad=7)
    fig.legend(handles,[f'LR = {lr:g}' for lr in RATES],ncol=2,frameon=False,loc='lower center',
               bbox_to_anchor=(.5,.07),handlelength=3,columnspacing=3,fontsize=8)
    fig.text(.5,.015,r'$B_0(t)=\sqrt{\sum_j[\gamma_j(t)^2+\beta_j(t)^2+w_j(t)^2]}$',ha='center',fontsize=9,color='#444444')
    fig.savefig(OUT/'gd_restart_norm.png',dpi=240)
    fig.savefig(OUT/'gd_restart_norm.svg')
    plt.close(fig)
    (OUT/'gd_restart_norm.json').write_text(json.dumps(report,indent=2)+'\n')
    print(OUT/'gd_restart_norm.png')
if __name__=='__main__':main()
