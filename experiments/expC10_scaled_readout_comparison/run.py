"""Compare raw and note-scaled truncated least squares on identical dictionaries.

Only the readout map differs along each curve. Mark the raw baseline lambda=.25
and the note's pre-fit frequency-based choice. Keep the user's 24 halo per side.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'results/checkpoint_C_geometry/expC10_scaled_readout_comparison'
TITLES = {'sine':'Sine', 'mixed':'Mixed sine', 'quadratic':'Quadratic',
          'runge':'Runge', 'gaussian_envelope':'Gaussian envelope', 'exp':'Exponential'}
EQUATIONS = {
    'sine':r'$\sqrt{2}\sin(2\pi x)$',
    'mixed':r'$\left[\sin(2\pi x)+0.1\sin(20\pi x)\right]/\sqrt{0.505}$',
    'quadratic':r'$\sqrt{5}\,x^2$',
    'runge':r'$1/(1+25x^2)$',
    'gaussian_envelope':r'$e^{-x^2/(2\cdot0.4^2)}f_{\rm mixed}(x)$',
    'exp':r'$e^x$',
}


def config():
    return yaml.safe_load((HERE/'config.yaml').read_text())


def target(x, name):
    if name=='sine': return np.sqrt(2)*np.sin(2*np.pi*x)
    if name=='quadratic': return np.sqrt(5)*x*x
    if name=='runge': return 1/(1+25*x*x)
    if name=='exp': return np.exp(x)
    mixed=(np.sin(2*np.pi*x)+.1*np.sin(20*np.pi*x))/np.sqrt(.505)
    if name=='mixed': return mixed
    if name=='gaussian_envelope': return np.exp(-.5*(x/.4)**2)*mixed
    raise ValueError(name)


def fft_frequency(y):
    """The note's period-two diagnostic; omit repeated right endpoint, retain DC."""
    values=np.asarray(y)[:-1]
    mass=np.abs(np.fft.fft(values))/len(values)
    omega=2*np.pi*np.fft.fftfreq(len(values),d=2/len(values))
    total=mass.sum()
    return float(np.dot(mass,abs(omega))/total) if total>0 else 0.


def log_alias_score(lam, n, omega):
    # log(2 exp(-pi^2/lam) sinh(2*pi*omega/(N*lam))), without overflow.
    if omega==0: return -np.inf
    u=2*np.pi*omega/(n*lam)
    return -np.pi**2/lam+u+np.log(-np.expm1(-2*u))


def select_lambda(n, omega, cfg):
    lo,hi=cfg['lambda_min'],cfg['lambda_max']
    if omega<=0:
        return {'lambda':None,'status':'constant_target_no_positive_budget_root'}
    if 2*omega/n>=np.pi:
        return {'lambda':None,'status':'no_high_precision_branch_at_estimated_frequency'}
    budget=np.log(cfg['effective_aliasing_budget'])
    fun=lambda lam: log_alias_score(lam,n,omega)-budget
    if fun(lo)>0 or fun(hi)<0:
        return {'lambda':None,'status':'root_outside_prescribed_bracket'}
    lam=float(brentq(fun,lo,hi,xtol=1e-15))
    return {'lambda':lam,'status':'root','log_budget_residual':float(fun(lam))}


def envelopes(n, halo, lam, delta):
    """Note Section 2 at the CURRENT lambda, not the old fixed reference lambda."""
    h=2/n
    pole=np.pi*h/(2*lam)
    if pole>=delta:
        return None
    width=n+1+2*halo
    alpha=np.full(width,h/(2*(delta-pole)))
    m=(halo+1)//2
    ell=np.arange(1,m+1)
    logp=np.r_[0.,np.cumsum(np.log(-np.expm1(-2*lam*ell)))]
    d_lambda=np.pi/(2*lam)+4*np.log(2)/np.pi
    for i in range(1,m+1):
        others=ell[ell!=i]
        log_li=-lam*(i*(i+1)-1)-logp[i-1]-logp[m-i]
        log_li+=np.log1p(np.exp(-2*lam*(others-.5))).sum()
        correction=h*d_lambda*np.exp(log_li)/(2*delta)
        alpha[i-1]+=correction
        alpha[-i]+=correction
    return np.r_[1+alpha.sum(),alpha]


def scaled_solve(A,Y,scales,rcond):
    # Both arms use this same call and normalization.
    F=A*scales/np.sqrt(len(A))
    a,_,rank,singular=lstsq(F,Y/np.sqrt(len(A)),cond=rcond,lapack_driver='gelsd',
                            check_finite=False)
    return scales[:,None]*a, a, int(rank), singular


def save(path, **values):
    path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**values)
    temp.replace(path)


def run(cfg):
    data=OUT/'data';data.mkdir(parents=True,exist_ok=True)
    x=np.linspace(-1,1,cfg['train_points'])
    xe=-1+2*(np.arange(cfg['eval_points'])+.5)/cfg['eval_points']
    names=cfg['targets'];nf=len(names)
    Y=np.stack([target(x,name) for name in names],axis=1)
    Ye=np.stack([target(xe,name) for name in names],axis=1)
    frequency=np.array([fft_frequency(Y[:,f]) for f in range(nf)])
    # Use analytic frequencies where explicitly known, as directed by the note.
    frequency[0]=2*np.pi
    frequency[1]=(2*np.pi+.1*20*np.pi)/1.1
    delta=np.array([cfg['delta_by_target'].get(name,cfg['delta_default']) for name in names])
    selections=[]
    for n in cfg['interior_resolutions']:
        row=[]
        for f,name in enumerate(names):
            item=dict(target=name,omega=float(frequency[f]),delta=float(delta[f]),
                      **select_lambda(n,frequency[f],cfg))
            item['envelope_valid_at_prediction']=(item['lambda'] is not None and
                                                  envelopes(n,cfg['halo_per_side'],item['lambda'],delta[f]) is not None)
            row.append(item)
        selections.append(row)
    metadata={'config':cfg,'targets':EQUATIONS,'frequency':frequency.tolist(),'selections':selections,
              'frequency_sources':['analytic','analytic','FFT','FFT','FFT','FFT'],
              'width_convention':'N is interior intervals; W=N+1+2*24 is actual tanh count',
              'primary_comparison':'Same centers, samples, loss normalization, LAPACK gelsd and relative cutoff. Raw vs c=D a; D recomputed at each lambda.',
              'note_adaptations':'Fixed halo 24 per side per user preference. Effective aliasing budget=FP64 epsilon; SVD rcond=1e-14 as D36. Delta=.25 for entire targets and .19 below the Runge poles at distance .2. These are declared practical choices, not a complete floating-point certificate.',
              'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
    (data/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    save(data/'samples.npz',x_train=x,x_eval=xe,y_train=Y,y_eval=Ye)
    for wi,n in enumerate(cfg['interior_resolutions']):
        path=data/f'N{n}.npz'
        if path.exists():
            with np.load(path) as old:
                if str(old['config_sha256'])!=metadata['config_sha256']:raise ValueError('Existing config differs')
            print(f'Reusing completed N={n}',flush=True)
            continue
        started=time.perf_counter()
        halo=cfg['halo_per_side'];centers=-1+2/n*np.arange(-halo,n+halo+1)
        lambdas=np.unique(np.r_[np.geomspace(cfg['lambda_min'],cfg['lambda_max'],cfg['lambda_count']),
                                  cfg['standard_lambda'],
                                  [p['lambda'] for p in selections[wi] if p['lambda'] is not None]])
        nk=len(lambdas);p=len(centers)+1
        error=np.full((nk,2,nf),np.nan);train=np.full_like(error,np.nan)
        linf=np.full_like(error,np.nan);ranks=np.full_like(error,np.nan)
        coeff=np.full((nk,2,p,nf),np.nan);native=np.full_like(coeff,np.nan)
        scales=np.full((nk,p,nf),np.nan)
        spectra=np.full((nk,2,p,nf),np.nan)
        elapsed=np.zeros(nk);map_discrepancy=np.full((nk,nf),np.nan)
        for li,lam in enumerate(lambdas):
            begin=time.perf_counter()
            gamma=lam*n/2
            A=np.c_[np.ones(len(x)),np.tanh(gamma*(x[:,None]-centers))]
            E=np.c_[np.ones(len(xe)),np.tanh(gamma*(xe[:,None]-centers))]
            c,a,rank,s=scaled_solve(A,Y,np.ones(p),cfg['relative_svd_cutoff'])
            coeff[li,0]=c;native[li,0]=a;ranks[li,0]=rank;spectra[li,0]=s[:,None]
            for value in np.unique(delta):
                ids=np.flatnonzero(delta==value)
                alpha=envelopes(n,halo,lam,value)
                if alpha is None: continue
                # Upward-rounded square roots; not a claim of interval-certified alpha.
                d=np.nextafter(np.sqrt(alpha),np.inf)
                c,a,rank,s=scaled_solve(A,Y[:,ids],d,cfg['relative_svd_cutoff'])
                for column,f in enumerate(ids):
                    coeff[li,1,:,f]=c[:,column];native[li,1,:,f]=a[:,column]
                    scales[li,:,f]=d;ranks[li,1,f]=rank;spectra[li,1,:,f]=s
                    if np.isclose(lam,selections[wi][f]['lambda'] or -1,rtol=0,atol=1e-15):
                        diff=E@c[:,column]-(E*d)@a[:,column]
                        map_discrepancy[li,f]=np.linalg.norm(diff)/np.linalg.norm(Ye[:,f])
            for arm in [0,1]:
                residual=E@coeff[li,arm]-Ye
                error[li,arm]=np.linalg.norm(residual,axis=0)/np.linalg.norm(Ye,axis=0)
                linf[li,arm]=abs(residual).max(axis=0)
                train[li,arm]=np.linalg.norm(A@coeff[li,arm]-Y,axis=0)/np.linalg.norm(Y,axis=0)
            elapsed[li]=time.perf_counter()-begin
        save(path,lambdas=lambdas,gammas=lambdas*n/2,centers=centers,
             eval_rel_l2=error,train_rel_l2=train,eval_linf=linf,ranks=ranks,
             physical_coefficients=coeff,native_coefficients=native,scales=scales,
             singular_values=spectra,seconds_per_lambda=elapsed,
             map_prediction_discrepancy=map_discrepancy,config_sha256=metadata['config_sha256'])
        print(f'Completed N={n}, W={len(centers)}, {nk} lambdas; {time.perf_counter()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    cfg=config()
    with threadpool_limits(limits=cfg['threads']):
        if not args.plot_only:run(cfg)
        from plot import draw
        draw(cfg,OUT)


if __name__=='__main__':main()
