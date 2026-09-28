"""Eigenvalue-ratio diagnostics: fixed geometry; no optimization trajectories.

Exact matrix derivatives are evaluated using the rectangular feature SVD.
Fourier identities are checked by quadrature, separately from their proof.
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/precisionmlps-mpl')
import numpy as np
from scipy import linalg as la
from scipy.integrate import simpson
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import Geometry, features, continuum_kernel

ROOT = HERE.parents[3]
OUT = ROOT/'results/checkpoint_D_optimizers/expD37_capacity_access_figures/gamma_solve/spectrum_mechanism'
FLOOR = 1e-14


def design(x, c, gamma, weights=None):
    d = x[:, None]-c
    t = np.tanh(gamma*d)
    # sech^2 computed without subtraction from one in the saturated tails.
    z = np.exp(-2*np.abs(gamma*d))
    dt = d*4*z/(1+z)**2
    scale = np.ones(len(c)) if weights is None else np.sqrt(weights)
    return (np.column_stack((np.ones(len(x)), t*scale))/np.sqrt(len(x)),
            np.column_stack((np.zeros(len(x)), dt*scale))/np.sqrt(len(x)))


def eigen_data(J, Jp, gamma):
    u, s, vt = la.svd(J, full_matrices=False, lapack_driver='gesvd', check_finite=False)
    coupling = np.einsum('ij,ij->j', u, Jp@vt.T)
    growth = np.divide(2*gamma*coupling, s, out=np.full_like(s,np.nan), where=s>0)
    return u, s*s, growth


def center_quadrature(geom, order):
    z, w = np.polynomial.legendre.leggauss(order)
    _, centers = geom.arrays()
    nodes = (centers[:, None]+geom.h*z/2).ravel()
    # Includes center density 1/h, so weights per cell sum to one.
    weights = np.tile(w/2, len(centers))
    return nodes, weights


def modal_integral(q, J, Jp, gamma):
    a, b = q.T@J, q.T@Jp
    return np.sum(a*a,axis=1), 2*gamma*np.sum(a*b,axis=1)


def fourier_modal(geom, gamma, q, points, odd=True):
    """Positive-frequency whole-center-line quadratic form for odd q.

    q has unit sample norm and exact antisymmetric construction. Use its
    paired sine sum to avoid spurious nonzero sums near omega=0.
    """
    x, _ = geom.arrays()
    omega = np.linspace(0,30*gamma,points)
    z = np.pi*omega/(2*gamma)
    M = np.ones_like(z)
    H = np.zeros_like(z)
    M[1:] = z[1:]/np.sinh(z[1:])
    H[1:] = 2*(z[1:]/np.tanh(z[1:])-1)
    if odd:
        left = len(x)//2
        # sum_i q_i exp(-i omega x_i) / omega = -2i sum_left q_i sin(omega x_i)/omega.
        oscillation = np.sinc(omega[:,None]*x[None,:left]/np.pi)*x[None,:left]
        squared_amplitude = (2*oscillation@q[:left])**2
    else:
        # For general zero-mean vectors, subtract 1 in the exponential before
        # dividing by omega. Both real and imaginary limits are then stable.
        phase=omega[:,None]*x
        re=(-x[None,:]*np.sin(phase/2)*np.sinc(phase/(2*np.pi)))@q
        im=(-x[None,:]*np.sinc(phase/np.pi))@q
        squared_amplitude=re**2+im**2
    density = 4/(np.pi*geom.h*geom.m)*M[:,None]**2*squared_amplitude
    energy = simpson(density,x=omega,axis=0)
    derivative = simpson(H[:,None]*density,x=omega,axis=0)
    # An analytic bound on omitted positive-frequency energy for any unit q.
    tail_bound = 4/(geom.h*gamma*np.expm1(30*np.pi))
    return omega, density, energy, derivative, tail_bound


def centered_main(geom, gamma):
    x, _ = geom.arrays()
    d = x[:,None]-x
    f = np.full_like(d,1/gamma)
    np.divide(d,np.tanh(gamma*d),out=f,where=d!=0)
    Q = np.eye(len(x))-np.ones((len(x),len(x)))/len(x)
    return -2/(geom.h*geom.m)*Q@f@Q


def run():
    OUT.joinpath('data').mkdir(parents=True,exist_ok=True)
    geom = Geometry()
    x,c = geom.arrays()
    gammas = np.unique(np.r_[np.geomspace(.25,128,73),[1,2,4,8,16,32,64]])
    values=[];growth=[];fd_error=[]
    for gamma in gammas:
        J,Jp=design(x,c,gamma)
        _,ev,elasticity=eigen_data(J,Jp,gamma)
        values.append(ev);growth.append(elasticity)
        if gamma in [1,2,4,8,16,32,64,128]:
            ds=2e-5
            sp=la.svdvals(design(x,c,gamma*np.exp(ds))[0])
            sm=la.svdvals(design(x,c,gamma*np.exp(-ds))[0])
            fd=(np.log(sp)-np.log(sm))/ds
            keep=ev/ev[0]>1e-12
            fd_error.append({'gamma':gamma,'modes':int(keep.sum()),
                             'max_absolute_error':float(np.max(abs(fd[keep]-elasticity[keep])))})
    values=np.array(values);growth=np.array(growth)
    ratios=values/values[:,:1]
    relative_growth=growth-growth[:,:1]
    resolved=ratios>FLOOR
    np.savez_compressed(OUT/'data/ratios.npz',gammas=gammas,eigenvalues=values,
                        ratios=ratios,growth=growth,ratio_growth=relative_growth,resolved=resolved)

    # Odd symmetry makes the constant-feature contribution exactly zero.
    # These are eigenvectors of the actual discrete kernel, not Fourier waves.
    O=np.zeros((geom.m,geom.m//2))
    j=np.arange(geom.m//2)
    O[j,j]=1/np.sqrt(2);O[geom.m-1-j,j]=-1/np.sqrt(2)
    odd_indices=np.array([1,3,6,10,16])-1
    nodes8,weights8=center_quadrature(geom,8)
    nodes16,weights16=center_quadrature(geom,16)
    records=[];modal_arrays={}
    for gamma in [8.,16.,32.,64.,128.]:
        J,Jp=design(x,c,gamma)
        _,full_ev,full_growth=eigen_data(J,Jp,gamma)
        u,ev,a=eigen_data(O.T@J,O.T@Jp,gamma)
        q=O@u[:,odd_indices]
        lam=ev[odd_indices]
        omega,density,bulk,bulk_g,tail=fourier_modal(geom,gamma,q,16385)
        _,_,bulk8,bulk_g8,_=fourier_modal(geom,gamma,q,8193)
        Jc,Jcp=design(x,nodes16,gamma,weights16)
        cont,cont_g=modal_integral(q,Jc,Jcp,gamma)
        Jc8,Jcp8=design(x,nodes8,gamma,weights8)
        cont8,cont_g8=modal_integral(q,Jc8,Jcp8,gamma)
        analytic=continuum_kernel(x,gamma,*geom.bounds,1/geom.h)/geom.m
        matrix_error=float(la.norm(Jc@Jc.T-analytic,2)/full_ev[0])
        main=centered_main(geom,gamma)
        main_energy=np.einsum('ij,ij->j',q,main@q)
        actual_g=lam*a[odd_indices]
        components=np.column_stack((bulk_g/lam,(cont_g-bulk_g)/lam,
                                    (actual_g-cont_g)/lam,
                                    np.full(len(lam),-full_growth[0])))
        # Select normalized frequency distributions, not just carrier peaks.
        probability=density/bulk[None,:]
        cumulative=np.vstack((np.zeros(len(lam)),np.cumsum((probability[:-1]+probability[1:])*.5*np.diff(omega)[:,None],axis=0)))
        means=simpson(omega[:,None]*probability,x=omega,axis=0)
        # A fixed frequency threshold supplies a sufficient lower bound;
        # do not infer ordering from only the mean or the largest peak.
        threshold=8*np.pi
        tail_mass=np.array([1-np.interp(threshold,omega,cumulative[:,k]) for k in range(len(lam))])
        threshold_z=np.pi*threshold/(2*gamma)
        threshold_H=2*(threshold_z/np.tanh(threshold_z)-1)
        derivative_correction=components[:,1]+components[:,2]
        lower_bound=(bulk/lam)*tail_mass*threshold_H-abs(derivative_correction)-full_growth[0]
        record={'gamma':gamma,'odd_ranks':(odd_indices+1).tolist(),
                'normalized_eigenvalues':(lam/full_ev[0]).tolist(),
                'full_top_growth':float(full_growth[0]),
                'mode_growth':a[odd_indices].tolist(),
                'ratio_growth':(a[odd_indices]-full_growth[0]).tolist(),
                'bulk_fourier_growth':(bulk_g/bulk).tolist(),
                'bulk_to_actual_energy':(bulk/lam).tolist(),
                'finite_integral_to_actual_energy':(cont/lam).tolist(),
                'growth_components':components.tolist(),
                'mean_angular_frequency':means.tolist(),
                'frequency_threshold_over_pi':8.,'mass_above_threshold':tail_mass.tolist(),
                'sufficient_ratio_derivative_lower_bound':lower_bound.tolist(),
                'cdf_dominance_max_violations':[float(np.max(cumulative[:,i]-cumulative[:,0])) for i in range(len(lam))],
                'fourier_energy_refinement_relative':float(np.max(abs(bulk-bulk8)/bulk)),
                'fourier_derivative_refinement_relative':float(np.max(abs(bulk_g-bulk_g8)/bulk_g)),
                'center_energy_refinement_relative':float(np.max(abs(cont-cont8)/cont)),
                'center_derivative_refinement_relative':float(np.max(abs(cont_g-cont_g8)/np.maximum(abs(cont_g),1e-30))),
                'closed_integral_matrix_relative_error':matrix_error,
                'closed_main_relative_modal_error':float(np.max(abs(main_energy-bulk)/bulk)),
                'fourier_omitted_energy_upper_bound':tail}
        records.append(record)
        modal_arrays[f'omega_{int(gamma)}']=omega
        modal_arrays[f'probability_{int(gamma)}']=probability
        modal_arrays[f'cdf_{int(gamma)}']=cumulative
        print('modal gamma',gamma,'ratio growth',np.round(record['ratio_growth'],4),flush=True)
    np.savez_compressed(OUT/'data/fourier_modes.npz',**modal_arrays)

    # The same accounting for actual odd AND even eigenvectors. Projection
    # onto zero-mean vectors introduces an explicit mean-coupling term.
    full_modes=[]
    indices=np.arange(1,40)
    for gamma in [8.,16.,64.]:
        J,Jp=design(x,c,gamma)
        u,ev,elasticity=eigen_data(J,Jp,gamma)
        U=u[:,indices];q=U-U.mean(axis=0)
        _,_,D,bulk_g,_=fourier_modal(geom,gamma,q,16385,odd=False)
        Jc,Jcp=design(x,nodes16,gamma,weights16)
        _,center_g=modal_integral(q,Jc,Jcp,gamma)
        _,full_g=modal_integral(U,Jc,Jcp,gamma)
        lam=ev[indices]
        parts=np.column_stack((bulk_g/lam,(center_g-bulk_g)/lam,
                               (full_g-center_g)/lam,elasticity[indices]-full_g/lam))
        full_modes.append({'gamma':gamma,'ranks':(indices+1).tolist(),
                           'actual_growth':elasticity[indices].tolist(),
                           'largest_growth':float(elasticity[0]),
                           'components':parts.tolist(),
                           'component_names':['Fourier','finite halo','mean coupling','center grid'],
                           'normalized_eigenvalues':(lam/ev[0]).tolist(),
                           'relative_centered_energy':(D/lam).tolist()})

    # Test the previously suggested sharp-step-minus-identity approximation.
    kinf=(1+len(c)-2/geom.h*np.abs(x[:,None]-x))/geom.m
    step_ev=la.eigvalsh(kinf)[::-1]
    approximation=[]
    for gamma in [8.,16.,32.,64.,128.]:
        J,_=design(x,c,gamma)
        actual=la.svdvals(J)**2
        Jc,_=design(x,nodes16,gamma,weights16)
        ec=la.svdvals(Jc)**2
        shift=2/(geom.h*gamma*geom.m)
        predicted=step_ev-shift
        # Finite matrix dimension m is used in the continuous model; the
        # discrete model has at most W+1 nonzero eigenvalues.
        approximation.append({'gamma':gamma,'gamma_sample_spacing':gamma*(x[1]-x[0]),
                              'negative_predicted_eigenvalues':int(np.sum(predicted<0)),
                              'actual':actual.tolist(),'continuous':ec.tolist(),
                              'shift_prediction':predicted.tolist(),
                              'shift_matrix_error_relative':float(la.norm(Jc@Jc.T-(kinf-shift*np.eye(geom.m)),2)/ec[0])})

    summary={'geometry':{'N':geom.N,'m':geom.m,'width':len(c),'h':geom.h,'bounds':geom.bounds},
             'resolved_ratio_floor':FLOOR,'derivative_checks':fd_error,'modal':records,'full_modes':full_modes,
             'sharp_step_approximation':approximation,
             'minimum_resolved_ratio_growth':float(np.min(relative_growth[:,1:][resolved[:,1:]]))}
    (OUT/'data/summary.json').write_text(json.dumps(summary,indent=2))
    plot(gammas,values,growth,resolved,records,modal_arrays,approximation,full_modes)
    return summary


def style(ax,xlabel,ylabel,logx=False,logy=False):
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    if logx:ax.set_xscale('log')
    if logy:ax.set_yscale('log')
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)


def legend(ax,columns=2):
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=columns,frameon=False,fontsize=9)


def plot(gammas,values,growth,resolved,records,modal_arrays,approximation,full_modes):
    plt.rcParams.update({'font.size':11,'figure.dpi':160,'axes.titlepad':57})
    colors=plt.cm.viridis(np.linspace(.08,.9,5))
    fig,axs=plt.subplots(2,2,figsize=(13.6,10),layout='constrained')
    ranks=[2,4,8,16,32,131]
    for rank,col in zip(ranks,[*colors,'#444444']):
        keep=resolved[:,rank-1]
        ls='--' if rank==131 else '-'
        axs[0,0].plot(gammas,np.where(keep,values[:,rank-1]/values[:,0],np.nan),color=col,label=f'Rank {rank}',linestyle=ls)
        axs[0,1].plot(gammas,np.where(keep,growth[:,rank-1]-growth[:,0],np.nan),color=col,label=f'Rank {rank}',linestyle=ls)
    axs[0,0].set_title('Actual finite kernel: normalized eigenvalues')
    style(axs[0,0],r'Common slope $\gamma$',r'$\lambda_i/\lambda_1$',True,True)
    axs[0,0].set_ylim(FLOOR,1.1);legend(axs[0,0],3)
    axs[0,1].set_title('Exact local change in each ratio')
    style(axs[0,1],r'Common slope $\gamma$',r'$d\log(\lambda_i/\lambda_1)/d\log\gamma$',True)
    axs[0,1].axhline(0,color='black',linewidth=.8);legend(axs[0,1],3)
    gi=np.where(gammas==16)[0][0]
    ii=np.arange(1,len(values[gi])+1)
    keep=resolved[gi]
    axs[1,0].plot(ii[keep],growth[gi,keep],label=r'Each eigenvalue: $d\log\lambda_i/d\log\gamma$',color='#1768a5')
    full=next(r for r in full_modes if r['gamma']==16.)
    parts=np.array(full['components'])
    axs[1,0].plot(full['ranks'],parts[:,0]+parts[:,2],'x',markevery=3,markersize=4,
                  label='Fourier contribution + mean coupling (odd and even modes)',color='#ce6518')
    axs[1,0].axhline(growth[gi,0],label=r'Largest eigenvalue: $d\log\lambda_1/d\log\gamma$',color='black',linestyle='--')
    axs[1,0].set_title(r'Why ratios improve at $\gamma=16$')
    style(axs[1,0],'Eigenvalue rank i','Fractional eigenvalue growth / fractional γ growth');legend(axs[1,0],1)
    records_at={r['gamma']:r for r in records}
    r=records_at[16.]
    components=np.array(r['growth_components'])
    locations=np.arange(len(r['odd_ranks']));positive=np.zeros(len(locations));negative=positive.copy()
    for j,(label,col) in enumerate([('Whole-line Fourier term','#1768a5'),('Finite-halo correction','#ce6518'),('Discrete-center correction','#23856c'),('Subtract largest-eigenvalue growth','#555555')]):
        vals=components[:,j];bottom=np.where(vals>=0,positive,negative)
        axs[1,1].bar(locations,vals,bottom=bottom,color=col,label=label,width=.65)
        positive+=np.maximum(vals,0);negative+=np.minimum(vals,0)
    axs[1,1].plot(locations,r['ratio_growth'],'ko',markersize=4,label='Actual ratio derivative')
    axs[1,1].set_xticks(locations,[str(k) for k in r['odd_ranks']])
    axs[1,1].set_title(r'Account for the actual derivative, $\gamma=16$')
    style(axs[1,1],'Eigenvalue rank within odd subspace',r'$d\log(\lambda_i/\lambda_1)/d\log\gamma$');legend(axs[1,1],2)
    fig.suptitle('Fixed tanh geometry · N=128, m=263 · no training\nRatios below 10⁻¹⁴ are omitted; derivatives use the feature SVD',fontsize=14)
    fig.savefig(OUT/'eigenvalue_ratio_growth.png');plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for axcol,gamma in zip(axs.T,[8,16,64]):
        omega=modal_arrays[f'omega_{gamma}'];probability=modal_arrays[f'probability_{gamma}']
        r=records_at[float(gamma)]
        for j,(rank,col) in enumerate(zip(r['odd_ranks'],colors)):
            axcol[0].plot(omega/np.pi,probability[:,j]*np.pi,color=col,label=f'Odd rank {rank}')
        axcol[0].set_xlim(0,22)
        axcol[0].set_ylim(0,1.85)
        axcol[0].set_title(f'γ={gamma}: frequencies in the quadratic form')
        style(axcol[0],r'Angular frequency $\omega/\pi$',r'Normalized density (area = 1)');legend(axcol[0],3)
        axcol[1].plot(r['odd_ranks'],r['bulk_fourier_growth'],'o-',color='#1768a5',label='Fourier-weighted growth')
        axcol[1].plot(r['odd_ranks'],r['mode_growth'],'x--',color='#ce6518',label='Actual finite-kernel growth')
        axcol[1].axhline(r['full_top_growth'],color='black',linestyle=':',label='Largest-eigenvalue growth')
        style(axcol[1],'Eigenvalue rank within odd subspace',r'$d\log\lambda_i/d\log\gamma$',logy=True);legend(axcol[1],1)
        axcol[1].set_ylim(1e-4,20)
        axcol[1].set_title('Integral prediction versus actual derivative')
    fig.suptitle('Actual odd eigenvectors, not assumed Fourier waves\nWeight ∝ Mγ(ω)² |Σᵢ uᵢ exp(−iωxᵢ)|² / ω²; growth = weighted mean of 2(z coth z − 1)',fontsize=13)
    fig.savefig(OUT/'fourier_eigenvalue_growth.png');plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,gamma in zip(axs,[8.,16.,64.]):
        rec=next(row for row in approximation if row['gamma']==gamma)
        for key,label,col,ls in [('actual','Actual discrete kernel','#172432','-'),('continuous','Finite center integral','#1768a5','--'),('shift_prediction','Step kernel − 2/(hmγ) I','#ce6518',':')]:
            a=np.array(rec[key]);a=a/a[0]
            ax.plot(np.arange(1,len(a)+1),np.where(a>FLOOR,a,np.nan),label=label,color=col,linestyle=ls)
        ax.set_xlim(1,100);ax.set_ylim(FLOOR,1.1)
        ax.set_title(f'γ={gamma:g}: γΔx={rec["gamma_sample_spacing"]:.3g}\nShift formula gives {rec["negative_predicted_eigenvalues"]} negative eigenvalues')
        style(ax,'Eigenvalue rank i',r'$\lambda_i/\lambda_1$',False,True);legend(ax,1)
    fig.suptitle('Check the previous approximation before using it to explain the data\nThe finite integral is accurate over much of the spectrum; the sharp-step diagonal shift is not valid here',fontsize=13)
    fig.savefig(OUT/'approximation_check.png');plt.close(fig)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        summary=run()
    print(OUT)
    print('minimum resolved ratio growth:',summary['minimum_resolved_ratio_growth'])
