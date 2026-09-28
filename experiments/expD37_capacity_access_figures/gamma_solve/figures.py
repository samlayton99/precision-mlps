"""PNG snapshots of the same arrays used by the interactive viewer."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def make_figures(output,kernel,result,kernel_only=False):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained',gridspec_kw={'width_ratios':[1,1,1]})
    vmin=min(kernel['direct'].min(),kernel['continuous'].min())
    vmax=max(kernel['direct'].max(),kernel['continuous'].max())
    x=kernel['x'];meta=kernel['meta']
    for ax,array,title in zip(axes[:2],[kernel['direct'],kernel['continuous']],['Finite neuron sum','Uniform-center integral']):
        im=ax.pcolormesh(x,x,array,cmap='viridis',vmin=vmin,vmax=vmax,shading='nearest',rasterized=True)
        ax.add_patch(Rectangle((-1,-1),2,2,fill=False,color='#df2435',lw=1.8))
        ax.set(xlabel='Column position $x_j$',ylabel='Row position $x_i$',title=title,aspect='equal',xlim=meta['bounds'],ylim=meta['bounds'][::-1])
    fig.colorbar(im,ax=list(axes[:2]),label='$K=k/m$',shrink=.75,pad=.02)
    for v,name,style in [(kernel['eigenvalues'],'Finite sum','-'),(kernel['continuous_eigenvalues'],'Integral','--')]:
        axes[2].semilogy(np.arange(1,len(v)+1),v/v[0],style,label=name)
    axes[2].set(xlabel='Eigenvalue rank $i$',ylabel=r'$\lambda_i/\lambda_{\max}$',ylim=(1e-16,1.3),title='Training-kernel spectrum')
    axes[2].legend(loc='lower center',bbox_to_anchor=(.5,1.08),ncol=2,frameon=False)
    fig.suptitle(f"N={meta['N']} · m={meta['m']} · γ={meta['gamma']:g} · λ={meta['lambda']:g} · {meta['halo']} halo centers per side")
    fig.savefig(output/'kernel_comparison.png',dpi=180);plt.close(fig)
    if kernel_only:
        return

    fig,axes=plt.subplots(3,2,figsize=(15,13),layout='constrained')
    g=np.asarray(result['gammas']);p=np.asarray(result['p']);r=np.asarray(result['rates'])
    colors=np.broadcast_to(g[:,None],p.shape)
    mask=(p>1e-16)&(r>1e-18)
    ax=axes[0,0]
    sc=ax.scatter(r[mask],p[mask],c=colors[mask],cmap='viridis',norm=LogNorm(g.min(),g.max()),s=3,alpha=.7,rasterized=True)
    ax.set(xscale='log',yscale='log',xlabel=r'$\eta_\gamma\lambda_i=\frac{1}{2}\lambda_i/\lambda_{\max}$',ylabel=r'$p_i(\gamma)=|u_i^Ty|^2/\|y\|^2$',xlim=(1e-18,.6),ylim=(1e-16,1.3),title='a · Target weights and per-step rates')
    fig.colorbar(sc,ax=ax,label=r'$\gamma$',pad=.02)
    axes[0,1].axis('off')
    ref=result['reference_index'];n=result['n']
    text=(f"Mixed sine: sin(2πx) + ½ sin(6πx) + ¼ sin(10πx)\n\n"
          f"N={result['geometry']['N']}, m={result['geometry']['m']}\n"
          f"Reference γ₀={g[ref]:g}; fixed n={n:,}\n"
          f"Matching: {result['matching']}\n\n"
          "Black: actual GD prediction\nBlue: fixed target weights pᵢ(γ₀)\nOrange: fixed normalized eigenvalues\n\n"
          "Shading: uncertainty from unresolved eigendirections.\n"
          "Crosses at 10¹⁵: not reached within the displayed budget.\n"
          "1% relative L₂ corresponds to 10⁻⁴ squared relative error.\n\n"
          "These are calculated curves, not executed training runs.")
    axes[0,1].text(.02,.97,text,va='top',transform=axes[0,1].transAxes,linespacing=1.65)
    for row,timing in [(1,False),(2,True)]:
        for col,key in [(0,'fixed_p'),(1,'fixed_rates')]:
            ax=axes[row,col];prefix='steps_' if timing else '';color='#1768a5' if col==0 else '#ce6518'
            title=('b' if col==0 else 'c') if not timing else ('d' if col==0 else 'e')
            title+=' · '+('Change eigenvalue ratios' if col==0 else 'Change target weights')
            lo=np.array([result['max_steps'] if z is None else z for z in result[prefix+key+'_lower']])
            hi=np.array([result['max_steps'] if z is None else z for z in result[prefix+key+'_upper']])
            ax.fill_between(g,lo,hi,color=color,alpha=.13,lw=0)
            for name,label,cc in [('actual','Actual GD prediction','#172432'),(key,'Fixed pᵢ(γ₀)' if col==0 else 'Fixed λᵢ/λₘₐₓ at γ₀',color)]:
                v=np.array([np.nan if z is None else z for z in result[prefix+name]])
                ax.plot(g,v,color=cc,lw=1.8,label=label)
                if timing:
                    ax.scatter(g[np.isnan(v)],np.full(np.isnan(v).sum(),result['max_steps']),marker='x',s=20,color=cc)
            ax.axvline(g[ref],color='.6',ls=':',lw=1)
            if not timing:ax.axhline(1e-4,color='.6',ls=':',lw=1)
            ax.set(xscale='log',yscale='log',xlabel=r'Common slope $\gamma$',ylabel='First step to 1% relative L₂' if timing else r'Squared relative error $E_\gamma(n)^2$',ylim=(1e2,2e15) if timing else (1e-16,1.5),title=title)
            ax.legend(loc='lower center',bbox_to_anchor=(.5,1.12),ncol=2,frameon=False,fontsize=9)
            ax.grid(alpha=.15)
    fig.savefig(output/'spectrum_alignment.png',dpi=180);plt.close(fig)
