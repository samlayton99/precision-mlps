"""Integrate the Fourier modal explanation along actual eigendirections.

This is mechanism accounting, not an eigenvalue predictor that avoids knowing
the eigendirections. No trajectory-derived fit parameters are used.
"""
import json
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits
import run as mechanism


def calculate(lo,hi,order):
    geom=mechanism.Geometry();x,c=geom.arrays()
    z,w=np.polynomial.legendre.leggauss(order)
    t=(np.log(lo)+np.log(hi))/2+(np.log(hi)-np.log(lo))*z/2
    weights=w*(np.log(hi)-np.log(lo))/2
    O=np.zeros((geom.m,geom.m//2));ii=np.arange(geom.m//2)
    O[ii,ii]=1/np.sqrt(2);O[geom.m-1-ii,ii]=-1/np.sqrt(2)
    indices=np.array([1,3,6,10,16])-1
    full=np.zeros(len(indices));fourier=np.zeros(len(indices));correction=np.zeros(len(indices))
    for gamma,weight in zip(np.exp(t),weights):
        J,Jp=mechanism.design(x,c,gamma)
        _,_,top_growth=mechanism.eigen_data(J,Jp,gamma)
        u,ev,growth=mechanism.eigen_data(O.T@J,O.T@Jp,gamma)
        q=O@u[:,indices]
        _,_,D,H_integral,_=mechanism.fourier_modal(geom,gamma,q,8193)
        actual=growth[indices]-top_growth[0]
        predicted=H_integral/D-top_growth[0]
        full+=weight*actual;fourier+=weight*predicted;correction+=weight*(actual-predicted)
    endpoints=[]
    for gamma in [lo,hi]:
        J,Jp=mechanism.design(x,c,gamma)
        leading=la.svdvals(J)[0]**2
        odds=la.svdvals(O.T@J)**2
        endpoints.append(odds[indices]/leading)
    return {'gamma_interval':[lo,hi],'quadrature_order':order,'odd_ranks':(indices+1).tolist(),
            'actual_ratio_gain':(endpoints[1]/endpoints[0]).tolist(),
            'integrated_exact_growth_gain':np.exp(full).tolist(),
            'integrated_fourier_growth_gain':np.exp(fourier).tolist(),
            'integrated_correction_in_log_gain':correction.tolist()}


if __name__=='__main__':
    with threadpool_limits(limits=1):
        results=[calculate(lo,hi,order) for lo,hi in [(8,16),(16,64)] for order in [16,32]]
    (mechanism.OUT/'data/integrated_growth.json').write_text(json.dumps(results,indent=2))
    plt=mechanism.plt
    plt.rcParams.update({'font.size':11,'figure.dpi':160})
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,row in zip(axs,[r for r in results if r['quadrature_order']==32]):
        ax.plot(row['odd_ranks'],row['actual_ratio_gain'],'o-',color='#172432',label='Actual endpoint ratio change')
        ax.plot(row['odd_ranks'],row['integrated_fourier_growth_gain'],'x--',color='#1768a5',label='Integrated Fourier growth; omit corrections')
        ax.set_title(f'γ: {row["gamma_interval"][0]} → {row["gamma_interval"][1]}',pad=50)
        mechanism.style(ax,'Eigenvalue rank within odd subspace',r'$(\lambda_i/\lambda_1)_{\rm end}/(\lambda_i/\lambda_1)_{\rm start}$',logy=True)
        ax.set_ylim(.8,2000)
        mechanism.legend(ax,1)
    fig.suptitle('How much of the finite change does the Fourier term account for?\nActual eigendirections are used along the path; this does not assume they stay fixed.',fontsize=12)
    fig.savefig(mechanism.OUT/'integrated_ratio_growth.png');plt.close(fig)
    for row in results:
        print(row['gamma_interval'],row['quadrature_order'])
        print('actual ',row['actual_ratio_gain'])
        print('Fourier',row['integrated_fourier_growth_gain'])
