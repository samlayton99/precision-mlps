"""Reference-only audit of readout roundoff; never used to build models."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/precisionmlps-mpl')
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import gmpy2 as g
import numpy as np
from experiments.expC12_mhaskar_comparison import pbit,robust
from experiments.expC12_mhaskar_comparison.rescue import load_model
from experiments.expC12_mhaskar_comparison.construction import chirp
from experiments.expC12_mhaskar_comparison.run import json_write,relative_l2


def main():
    x=np.linspace(-1,1,257);truth=chirp(x)
    rows=[]
    for p in range(8,54):
        model=load_model(pbit.OUT/f'models/mhaskar_p{p}.npz')
        phi=pbit.features(x,model.slope,model.bias,p)
        predictions={mode:robust.readout(phi,model.readout,model.offset,p,mode)
                     for mode in ['sequential','neumaier','dot2']}
        # Higher precision is exclusively an independent diagnostic reference.
        # It is never used by the construction, selection, or deployed evaluator.
        with g.context(precision=256):
            a=[g.mpfr(v) for v in model.readout]
            exact=[g.mpfr(model.offset)+g.fsum(g.mpfr(v)*w for v,w in zip(row,a)) for row in phi]
            denom=g.fsum(v*v for v in exact)
            errors={mode:float(g.sqrt(g.fsum((g.mpfr(v)-z)**2 for v,z in zip(pred,exact))/denom))
                    for mode,pred in predictions.items()}
        rows.append({'p':p,'readout_arithmetic_error':errors,
                     'function_error':{mode:relative_l2(pred,truth) for mode,pred in predictions.items()},
                     'exact_dot_function_error':relative_l2(np.array(exact,dtype=float),truth)})
    json_write(robust.OUT/'data/readout_error_audit.json',{
        'reference_precision':256,'reference_used_for_model_or_selection':False,'points':257,'rows':rows})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.labelsize':13,'axes.titlesize':15,'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))
    fig.subplots_adjust(left=.09,right=.985,bottom=.17,top=.76,wspace=.3)
    colors=['#482475','#21918c','#7ad151']
    for mode,color in zip(['sequential','neumaier','dot2'],colors):
        axes[0].semilogy([r['p'] for r in rows],[r['readout_arithmetic_error'][mode] for r in rows],
                         lw=2,color=color,label=mode.capitalize())
    axes[0].set(title='Error introduced by the readout arithmetic',ylim=(1e-18,1))
    axes[0].set_ylabel('Relative error against reference dot product')
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=3,frameon=False)
    axes[1].plot([r['p'] for r in rows],[r['function_error']['sequential'] for r in rows],
                 color=colors[0],lw=2,label='Sequential readout')
    axes[1].plot([r['p'] for r in rows],[r['function_error']['dot2'] for r in rows],
                 color=colors[2],lw=2,label='Compensated dot product')
    axes[1].plot([r['p'] for r in rows],[r['exact_dot_function_error'] for r in rows],
                 '--',color='.3',lw=1.5,label='Reference dot product')
    axes[1].set(title='Total error against the chirp',ylim=(.94,1.01),ylabel=r'Relative $L^2$ error')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=1,frameon=False,fontsize=9)
    for ax in axes:
        ax.set(xlim=(8,53),xlabel=r'Working precision $p$ (bits)')
        ax.set_xticks([8,16,24,32,40,48,53]);ax.grid(alpha=.2)
    fig.savefig(robust.OUT/'figures/readout_error_diagnosis.png',dpi=240)
    plt.close(fig)


if __name__=='__main__':main()
