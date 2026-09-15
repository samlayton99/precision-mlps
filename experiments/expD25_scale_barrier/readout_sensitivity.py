"""Check geometry gains across numerical LS cutoffs and independent SVD drivers."""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD25_scale_barrier import run as exp

STATES=('initial','independent_2000','shared_2000','shared_10000','adaptive_10000')
LABELS=('Initial γ=1','Independent scales, step 2,000','Shared scale, step 2,000','Shared scale, step 10,000','Adam shared scale, step 10,000')


def main():
    cfg=exp.configuration();x=exp.old.midpoint_grid(cfg['n_train']);xx=exp.old.midpoint_grid(32768)
    cutoffs=np.array([1e-15,1e-14,1e-13,1e-12,1e-11]);data={'cutoffs':cutoffs}
    with threadpool_limits(limits=2):
        for target in exp.old.TARGETS:
            y=exp.old.matched.target_values(target,x,cfg);yy=exp.old.matched.target_values(target,xx,cfg)
            for state in STATES:
                if state=='initial':
                    parameters=exp.old.initial_state('gamma_1',cfg)
                else:
                    mode='scale_only' if state.startswith('independent') else 'shared_scale'
                    suffix='__10000steps' if state.endswith('10000') else ''
                    multiplier=1 if state.startswith('adaptive') else 10000
                    if state.startswith('adaptive'):suffix+='__adam'
                    path=exp.RESULTS/'data'/f'{target}__gamma_1__{multiplier}__{mode}{suffix}.npz'
                    with np.load(path) as f:parameters={k:f[k][-1] for k in ('a','b','v')}
                A=exp.old.design(x,parameters['a'],parameters['b'])/np.sqrt(len(x));rhs=y/np.sqrt(len(x))
                Ae=exp.old.design(xx,parameters['a'],parameters['b']);records=[]
                for driver in ('gesdd','gesvd'):
                    U,s,Vh=svd(A,full_matrices=False,check_finite=False,lapack_driver=driver)
                    for cutoff in cutoffs:
                        keep=s>cutoff*s[0];v=Vh[keep].T@((U[:,keep].T@rhs)/s[keep])
                        error=np.linalg.norm(Ae@v-yy)/np.linalg.norm(yy)
                        records.append([error,np.linalg.norm(v),keep.sum()])
                data[target+'__'+state]=np.asarray(records).reshape(2,len(cutoffs),3)
            print('SVD AUDIT',target,'cutoff 1e-13 initial/independent/shared2k/shared10k/Adam10k',
                  [data[target+'__'+s][0,2,0] for s in STATES],flush=True)
    exp.arrays_save(exp.RESULTS/'data/readout_sensitivity.npz',data,cfg | dict(audit_n_eval=32768))
    plot(data)


def plot(data):
    plt=exp.graphics();from matplotlib.lines import Line2D
    colors=['#444444','#2274b5','#e07826','#803c94','#20966b'];fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True)
    for row,target in enumerate(exp.old.TARGETS):
        for state,color in zip(STATES,colors):
            values=data[target+'__'+state]
            for col in range(3):
                for driver in range(2):
                    axes[row,col].plot(data['cutoffs'],np.maximum(values[driver,:,col],1e-16),color=color,
                                       ls='-' if driver==0 else ':',lw=1.6,marker='o' if driver==0 else None,ms=3)
        for ax in axes[row]:
            ax.set_xscale('log');ax.set_xlim(1e-15,1e-11);ax.axvline(1e-13,color='#888888',ls='--',lw=.8);ax.grid(alpha=.18)
            if row==3:ax.set_xlabel('Relative SVD cutoff')
        axes[row,0].set(yscale='log',ylim=(1e-16,1),ylabel=exp.old.matched.rc.LABELS[target]+'\nIndependent-grid relative L₂')
        axes[row,1].set(yscale='log',ylim=(1e-2,1e13),ylabel='Solved coefficient norm')
        axes[row,2].set(ylim=(0,185),ylabel='Retained numerical rank')
    for ax,title in zip(axes[0],['Does the approximation gain survive the cutoff?','What coefficient sizes does it require?','Which numerical directions are retained?']):ax.set_title(title,pad=12,fontsize=11)
    fig.suptitle('Audit the readout solves before interpreting geometry improvements',fontsize=19,y=.987)
    fig.legend([Line2D([],[],color=c,lw=2) for c in colors],LABELS,loc='upper center',bbox_to_anchor=(.5,.953),ncol=2,frameon=False)
    fig.subplots_adjust(top=.855,bottom=.12,left=.08,right=.98,wspace=.26,hspace=.24)
    fig.text(.5,.025,'Same saved geometries and 1,024 training midpoints; independent evaluation increased to 32,768 midpoints, all on [−1,1]. No training repeated.\n'
             'Solid: divide-and-conquer SVD (gesdd). Dotted: QR-iteration SVD (gesvd). The vertical line marks the original 10⁻¹³ cutoff.\n'
             'These are numerical readout solves, not an arbitrary-precision unrestricted-span theorem. Large coefficients and floor-level errors are shown explicitly.',ha='center',fontsize=9)
    fig.savefig(exp.FIGURES/'readout_sensitivity.png');plt.close(fig)


if __name__=='__main__':main()
