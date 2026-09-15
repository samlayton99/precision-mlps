"""Offline derivative and curvature measurements on unchanged baseline states."""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import svd, eigvalsh
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.expD25_scale_barrier import run as exp


def measure(a,b,v,x,y):
    h=np.tanh(x[:,None]*a+b);s=1-h*h;n=len(x)
    A=np.column_stack((h,np.ones(n)));r=A@v-y
    Jb=s*v[:-1];Ja=Jb*x[:,None];J=np.column_stack((Ja,Jb))
    U,singular,V=svd(A/np.sqrt(n),full_matrices=False,check_finite=False)
    keep=singular>exp.old.matched.rc.RCOND*singular[0]
    uk=U[:,keep];perp=r-uk@(uk.T@r)
    gr=A.T@r/n;gg=J.T@r/n;gp=J.T@perp/n
    # Exact geometry Hessian with readout held fixed, including residual curvature.
    H=J.T@J/n;k=len(a)
    diagonal=r[:,None]*v[:-1]*(-2*h*s)
    haa=np.mean(x[:,None]**2*diagonal,axis=0);hab=np.mean(x[:,None]*diagonal,axis=0);hbb=np.mean(diagonal,axis=0)
    indices=np.arange(k)
    H[indices,indices]+=haa;H[indices,indices+k]+=hab
    H[indices+k,indices]+=hab;H[indices+k,indices+k]+=hbb
    eig=eigvalsh(H,check_finite=False)
    geometry_norm=np.linalg.norm(gg)
    fraction=np.linalg.norm(gp)/geometry_norm if geometry_norm else np.nan
    return np.array([singular[0]**2,max(abs(eig[0]),abs(eig[-1])),
                     np.sqrt(np.mean(gr*gr)),np.sqrt(np.mean(gg*gg)),fraction,eig[0],eig[-1],
                     np.linalg.norm(v),np.sqrt(np.mean(r*r)),np.linalg.norm(perp)/np.linalg.norm(r)]),H


def main():
    cfg=exp.configuration();x=exp.old.midpoint_grid(cfg['n_train'])
    out={}
    with threadpool_limits(limits=2):
        for target in exp.old.TARGETS:
            y=exp.old.matched.target_values(target,x,cfg)
            for arm in ('gamma_1','gamma_4','gamma_16'):
                with np.load(exp.RESULTS/'data'/f'{target}__{arm}__1.npz') as f:
                    ids=np.unique([np.argmin(abs(f['steps']-s)) for s in (2,10,50,200,500,1000,2000)])
                    records=[measure(f['a'][i],f['b'][i],f['v'][i],x,y)[0] for i in ids]
                    out[target+'__'+arm]=np.column_stack((f['steps'][ids],records))
                    print(target,arm,'last [step, readout curvature, geometry curvature, readout RMSg, geometry RMSg, out-of-span ratio]',out[target+'__'+arm][-1,:6],flush=True)
    exp.arrays_save(exp.RESULTS/'data/derivatives.npz',out,cfg)
    plot(out)


def plot(data):
    plt=exp.graphics();from matplotlib.lines import Line2D
    fig,axes=plt.subplots(4,3,figsize=(16,13),dpi=170,sharex=True)
    colors=plt.cm.viridis([.12,.5,.86])
    for row,target in enumerate(exp.old.TARGETS):
        for arm,color in zip(('gamma_1','gamma_4','gamma_16'),colors):
            v=data[target+'__'+arm];steps=v[:,0]
            axes[row,0].plot(steps,v[:,1],color=color,lw=1.7);axes[row,0].plot(steps,v[:,2],'--',color=color,lw=1.7)
            axes[row,1].plot(steps,v[:,3],color=color,lw=1.7);axes[row,1].plot(steps,v[:,4],'--',color=color,lw=1.7)
            axes[row,2].plot(steps,np.maximum(v[:,5],1e-14),color=color,lw=1.7)
        for ax in axes[row]:
            ax.set_xscale('log');ax.set_xlim(2,2000);ax.set_yscale('log');ax.grid(alpha=.18)
            if row==3:ax.set_xlabel('GD step')
        axes[row,0].set(ylim=(1e-9,1e3),ylabel=exp.old.matched.rc.LABELS[target]+'\nBlock curvature norm')
        axes[row,1].set(ylim=(1e-10,1),ylabel='RMS gradient per parameter')
        axes[row,2].set(ylim=(1e-14,1),ylabel='‖Jᵀr⊥‖ / ‖Jᵀr‖')
    for ax,title in zip(axes[0],['Very different local curvature scales','One learning rate gives very different steps','How much geometry signal remains out of span?']):ax.set_title(title,fontsize=11,pad=13)
    fig.suptitle('Why the baseline step is weak, and which error drives it',fontsize=19,y=.986)
    handles=[Line2D([],[],color=c,lw=2,label=f'Initial γ = {g}') for c,g in zip(colors,[1,4,16])]
    handles += [Line2D([],[],color='black',lw=2,label='Readout block'),Line2D([],[],color='black',lw=2,ls='--',label='Raw geometry block')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.953),ncol=5,frameon=False)
    fig.subplots_adjust(top=.885,bottom=.12,left=.08,right=.98,hspace=.22,wspace=.28)
    fig.text(.5,.025,'Unchanged ordinary-GD trajectories, learning rate 0.002 for both blocks. Left: exact block Hessians with the other block fixed; geometry may have negative curvature.\n'
             'Right: current-readout geometry Jacobian and projection outside the SVD-retained feature span (cutoff 10⁻¹³). This is a gradient decomposition, not the gradient after refitting.\n'
             'Offline dense matrices are measurement oracles, not optimizer state. All functions use the same finite samples. Ratios below 10⁻¹⁴ are shown at the display floor.',ha='center',fontsize=9)
    fig.savefig(exp.FIGURES/'derivatives.png');plt.close(fig)


if __name__=='__main__':main()
