"""Separate reference scaling from cutoff selection; retain original results.

The practical-cutoff scaled arm is NOT the original A.65 sufficient recipe.
All cutoffs below are declared before measurements and never selected by
evaluation errors. Primary comparison uses the same single-target GELSD path.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.expC14_theory_solver_comparison import run as r
from experiments.expC14_theory_solver_comparison.plot import COLORS,LABELS,NAMES

OUT=r.OUT/'scaling_ablation'


def features(x,centers,slope,lam):
    if lam==.25:
        if len(x)==1:
            with r.gp.context(r.gp.get_context(),precision=53):
                vals=[float(r.gp.tanh(r.gp.mpfr(float(slope*(x[0]-c))))) for c in centers]
            return np.array([vals+[1.]])
        return r.feature_matrix(x,centers,slope)
    return np.column_stack((np.tanh(slope*(x[:,None]-centers)),np.ones(len(x))))


def predictions(x,centers,slope,lam,coefficients):
    result=np.empty((len(x),coefficients.shape[1]))
    for k in range(0,len(x),512):
        phi=features(x[k:k+512],centers,slope,lam)
        for j in range(coefficients.shape[1]):
            result[k:k+len(phi),j]=r.pairwise_sum(phi*coefficients[:,j],axis=1)
    return result


def measure(N,cfg,rc,xe,ye,xv,yv):
    started=time.perf_counter()
    _,R,slope,centers,checks=r.geometry(N,cfg)
    x=np.linspace(-1,1,N*cfg['sampling_density']+1)
    y=r.target_values(x)
    A=features(x,centers,slope,cfg['lambda'])
    alpha,D,L,E=r.envelopes(N,cfg)
    B=r.analytic_B(cfg)
    outputs=[];ranks=[];minimum_singular=[]
    for scaled in [False,True]:
        F=A*D/math.sqrt(len(x)) if scaled else A
        yy=y/math.sqrt(len(x)) if scaled else y
        for j in range(6):
            a,_,rank,singular=lstsq(F,yy[:,j],cond=rc,lapack_driver='gelsd',check_finite=False)
            outputs.append(a*D if scaled else a)
            ranks.append(int(rank));minimum_singular.append(float(singular[rank-1]))
    coefficients=np.column_stack(outputs)
    train=predictions(x,centers,slope,cfg['lambda'],coefficients)
    # Release the dense fit matrix before evaluating.
    train_residual=train-np.tile(y,(1,2))
    train_err=np.linalg.norm(train_residual,axis=0)/np.linalg.norm(np.tile(y,(1,2)),axis=0)
    del A,F,train
    combined=np.column_stack((coefficients,coefficients[:,6:]-coefficients[:,:6]))
    pred=predictions(xe,centers,slope,cfg['lambda'],combined)
    residual=pred[:,:12]-np.tile(ye,(1,2))
    errors=np.linalg.norm(residual,axis=0)/np.linalg.norm(np.tile(ye,(1,2)),axis=0)
    maxima=np.max(abs(residual),axis=0)
    residual_v=predictions(xv,centers,slope,cfg['lambda'],coefficients)-np.tile(yv,(1,2))
    validation=np.linalg.norm(residual_v,axis=0)/np.linalg.norm(np.tile(yv,(1,2)),axis=0)
    # Output differences are evaluated as one network, avoiding separate
    # rounded predictions when measuring the difference between procedures.
    diffs=np.linalg.norm(pred[:,12:],axis=0)/np.linalg.norm(ye,axis=0)
    rows=[]
    for j,name in enumerate(NAMES):
        row=dict(N=N,W=len(centers),R=R,sampling_density=cfg['sampling_density'],
                 lambda_value=cfg['lambda'],relative_cutoff=rc,target=name,
                 function_difference_relative_l2=float(diffs[j]),B=float(B[j]),L=L,
                 feature_evaluator='MPFR tanh 53-bit' if cfg['lambda']==.25 else 'numpy tanh')
        for arm,offset in [('raw',0),('scaled',6)]:
            k=j+offset;c=coefficients[:,k]
            row[arm]=dict(relative_l2=float(errors[k]),validation_l2=float(validation[k]),
                          max_abs=float(maxima[k]),train_relative_l2=float(train_err[k]),
                          rank=ranks[k],coefficient_l1=float(np.sum(abs(c))),
                          coefficient_l2=r.norm(c),scaled_coefficient_norm=r.norm(c/D),
                          scaled_norm_budget_ratio=r.norm(c/D)/(2*B[j]*L),
                          smallest_retained_singular=minimum_singular[k])
        rows.append(row)
    key=f'N{N}_lambda{cfg["lambda"]:g}_s{cfg["sampling_density"]}_rc{rc:.4e}'
    np.savez_compressed(OUT/f'{key}.npz',centers=centers,gamma=slope,
                        raw=coefficients[:,:6],scaled=coefficients[:,6:],D=D)
    (OUT/f'{key}.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'{key} ({time.perf_counter()-started:.1f}s): '+', '.join(
        f"{x['target']} {x['raw']['relative_l2']:.2e}/{x['scaled']['relative_l2']:.2e}" for x in rows),flush=True)
    return rows


def draw(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
    base=[x for x in rows if x['lambda_value']==.25 and x['sampling_density']==8 and x['relative_cutoff']==2**-52]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':18,
                         'xtick.labelsize':11.5,'ytick.labelsize':11.5,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(11,7.4))
    fig.subplots_adjust(left=.105,right=.975,bottom=.13,top=.73)
    for name,color in zip(NAMES,COLORS):
        rr=sorted([x for x in base if x['target']==name],key=lambda x:x['N'])
        for arm,style,marker in [('raw','-','o'),('scaled','--','s')]:
            ax.plot([x['N'] for x in rr],[x[arm]['relative_l2'] for x in rr],
                    color=color,lw=2.4,ls=style,marker=marker,ms=4,
                    mfc=color if arm=='raw' else 'white')
    fig.suptitle('Unscaled and reference-scaled SVD',fontsize=20,fontweight='bold',y=.98)
    fig.legend(handles=[Line2D([],[],color=c,lw=2.5,label=l) for c,l in zip(COLORS,LABELS)],
               loc='upper center',bbox_to_anchor=(.54,.91),ncol=3,frameon=False,fontsize=11.5)
    fig.legend(handles=[Line2D([],[],color='.2',lw=2.4,ls='-',label='Unscaled SVD (existing recipe)'),
                        Line2D([],[],color='.2',lw=2.4,ls='--',label='Reference-scaled SVD (same cutoff)')],
               loc='upper center',bbox_to_anchor=(.54,.80),ncol=2,frameon=False,handlelength=3.)
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    ns=sorted({x['N'] for x in base})
    ax.xaxis.set_major_locator(FixedLocator(ns));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r'Interior intervals $N$');ax.set_ylabel(r'Relative $L^2$ error')
    ax.set_ylim(3e-16,1e-9)
    ax.grid(axis='y',which='major',alpha=.13)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    for side in ('bottom','left'):ax.spines[side].set_linewidth(1.5)
    ax.tick_params(which='major',direction='out',length=5,width=1.25,pad=7)
    fig.text(.54,.025,r'FP64; same GELSD driver, geometry, samples, and relative cutoff $2^{-52}$; no test-error tuning.',ha='center',fontsize=10,color='.3')
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'scaling_only.{suffix}',dpi=220)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--robustness',action='store_true');p.add_argument('--plot-only',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((r.HERE/'config.yaml').read_text())
    jobs=[(N,.25,8,2**-52) for N in cfg['N']]
    if args.robustness:
        # Declared robustness grid, not hyperparameter selection.
        jobs += [(N,lam,8,2**-52) for N in [256,512,1024] for lam in [.2,.3,.4]]
        jobs += [(N,.25,s,rc) for N in [256,512,1024] for s,rc in [(16,2**-52),(8,2**-50),(8,2**-48)]]
    if not args.plot_only:
        (OUT/'protocol.json').write_text(json.dumps(dict(jobs=jobs,primary='same relative cutoff, not A.65',
            reason='isolate scaling from the conservative sufficient cutoff',test_error_tuning=False),indent=2))
        xe=-1+2*(np.arange(16384)+.5)/16384
        xv=-1+2*(np.arange(32768)+.5)/32768
        ye,yv=r.target_values(xe),r.target_values(xv)
        with threadpool_limits(limits=4):
            for N,lam,s,rc in jobs:
                key=f'N{N}_lambda{lam:g}_s{s}_rc{rc:.4e}'
                if (OUT/f'{key}.json').exists():continue
                cc=dict(cfg);cc['lambda']=lam;cc['sampling_density']=s
                measure(N,cc,rc,xe,ye,xv,yv)
    rows=[r for path in OUT.glob('N*.json') for r in json.loads(path.read_text())]
    draw(rows)


if __name__=='__main__':main()
