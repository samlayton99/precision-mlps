"""Summarize readout roundoff and complete-model rescue measurements."""
import os,json,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/precisionmlps-mpl')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.expC12_mhaskar_comparison import robust,pbit


def main():
    data=robust.OUT/'data'
    audit=json.loads((data/'readout_error_audit.json').read_text())['rows']
    original=json.loads((pbit.OUT/'data/summary.json').read_text())
    rescue=json.loads((data/'summary.json').read_text())
    extrap=json.loads((data/'extrapolation.json').read_text())
    plt.rcParams.update({'font.size':11,'axes.labelsize':13,'axes.titlesize':15})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4))
    fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.75,wspace=.28)
    for mode,color,label in [('sequential','#482475','Sequential'),('neumaier','#21918c','Neumaier'),('dot2','#7ad151','Compensated dot')]:
        axes[0].semilogy([r['p'] for r in audit],[r['readout_arithmetic_error'][mode] for r in audit],
                         lw=2,color=color,label=label)
    axes[0].set(title='Readout arithmetic becomes much more accurate',ylim=(1e-18,1),
                ylabel='Relative error against reference dot product')
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False,fontsize=10)
    axes[1].plot([r['p'] for r in original],[r['mhaskar_error'] for r in original],color='#482475',lw=2,label='Original Mhaskar')
    axes[1].plot([r['p'] for r in rescue],[r['rescued_error'] for r in rescue],'o-',color='#21918c',lw=2,
                 label='Symmetric weights + compensation')
    axes[1].plot([r['p'] for r in extrap],[r['error'] for r in extrap],'D',color='#7ad151',mec='#507729',ms=8,
                 label='Also cancel leading step errors (two checks)')
    axes[1].set(title='Total chirp approximation improves modestly',ylim=(.93,1),ylabel=r'Relative $L^2$ error')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.13),ncol=1,frameon=False,fontsize=9)
    for ax in axes:
        ax.set(xlim=(8,53),xlabel=r'Working precision $p$ (bits)')
        ax.set_xticks([8,16,24,32,40,48,53]);ax.grid(alpha=.2)
    fig.savefig(robust.OUT/'figures/rescue_summary.png',dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
