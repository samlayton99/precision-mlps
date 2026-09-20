"""Line plots from compact evidence; no training or model-state mutation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import targets

LABELS=dict(sine='Sine',runge='Runge',moment3='Coarse + degree 3',moment5='Coarse + degree 5',moment9='Coarse + degree 9')
COLORS=plt.colormaps['viridis'](np.linspace(.05,.95,7))
PCOLORS={0:'black',1:'#999999',3:'#2b70b6',5:'#de7a22',7:'#139579'}


def load(root):
    result=[]
    for path in sorted(root.glob('core_*_curves.npz')):
        with np.load(path) as data:
            row={k:data[k] for k in data.files}
        row['config']=json.loads(str(row['configuration']))
        row['case_list']=json.loads(str(row['cases']))
        result.append(row)
    return result


def trace(bundle,target,kappa,degree=0):
    index=next((i for i,c in enumerate(bundle['case_list']) if c['target']==target and c['kappa']==kappa),None)
    if index is None or f'p{degree}_trace' not in bundle: return None,None
    return bundle[f'p{degree}_steps']*bundle['config']['eta'],bundle[f'p{degree}_trace'][index]


def finish(fig,path,title):
    fig.suptitle(title,fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    for ax in fig.axes:
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(path,dpi=150);plt.close(fig)


def scale_curves(bundles,root):
    fig,axes=plt.subplots(3,5,figsize=(19,10),squeeze=False)
    for row,n in enumerate((64,128,256)):
        for col,target in enumerate(targets.TARGETS):
            ax=axes[row,col]
            for ki,kappa in enumerate(targets.RATIOS):
                for bundle in bundles:
                    if bundle['config']['n']!=n: continue
                    tau,tr=trace(bundle,target,kappa)
                    if tr is None: continue
                    label=f'κ={kappa:g}' if bundle['config']['seed']==0 else None
                    ax.plot(tau,tr[:,1]-bundle['mean_gamma0'],color=COLORS[ki],alpha=.65,lw=1,label=label)
            ax.axhline(0,color='.6',lw=.6)
            ax.set_title(f"{LABELS[target]} · W={n+2*(3*n//16)+1}")
            if col==0: ax.set_ylabel('Signed change in mean |a|')
            if row==2: ax.set_xlabel('Geometry time τ = updates × η')
    axes[0,-1].legend(fontsize=8,ncol=2)
    finish(fig,root/'scale_acquisition.png','Scale acquisition: each line is one paired seed; all readout rates shown')


def signal_curves(bundles,root):
    bundle=next((b for b in bundles if b['config']['n']==128 and b['config']['seed']==0),None)
    if bundle is None: return
    fig,axes=plt.subplots(4,5,figsize=(19,12),squeeze=False)
    for col,target in enumerate(targets.TARGETS):
        for ki in (0,4,6):
            kappa=targets.RATIOS[ki];tau,tr=trace(bundle,target,kappa)
            if tr is None: continue
            R=np.sqrt(np.where(tr[:,0]>0,2*tr[:,0],np.nan))
            sigma=np.sqrt(np.mean(targets.grid(bundle['config']['m'])**2))
            coarse=np.linalg.norm(tr[:,31:33]/[1,sigma],axis=1)
            color=COLORS[ki]
            axes[0,col].semilogy(tau,2*tr[:,0],color=color,label=f'κ={kappa:g}')
            axes[1,col].semilogy(tau,coarse,color=color)
            axes[2,col].semilogy(tau,tr[:,12]/R,color=color)
            axes[3,col].plot(tau,tr[:,16],color=color)
            axes[3,col].plot(tau,tr[:,17],color=color,ls='--')
        axes[0,col].set_title(LABELS[target]);axes[3,col].set_xlabel('Geometry time τ')
        axes[3,col].set_yscale('symlog',linthresh=1e-7)
    for row,label in enumerate(('Training MSE','Coarse residual norm','Slope gradient norm / residual RMS','Signed scale force\nsolid: coarse; dashed: remainder')):
        axes[row,0].set_ylabel(label)
    axes[0,-1].legend(fontsize=9)
    finish(fig,root/'signal_evolution.png','Residual signal and signed forces · W=177, seed 0 · fixed rate anchors')


def rate_contrasts(rows,root):
    fig,axes=plt.subplots(3,5,figsize=(19,10),squeeze=False)
    for ni,n in enumerate((64,128,256)):
        for ti,target in enumerate(targets.TARGETS):
            ax=axes[ni,ti]
            for degree in (0,1,3,5,7):
                for seed in range(5):
                    chosen=[r for r in rows if r['n']==n and r['target']==target and r['degree']==degree
                            and r['seed']==seed and r['complete'] and r['bundle'].startswith('core_')]
                    anchor=next((r for r in chosen if r['kappa']==1),None)
                    if anchor is None: continue
                    chosen=sorted(chosen,key=lambda r:r['kappa'])
                    ax.plot([r['kappa'] for r in chosen],[r['signed_mean_change']-anchor['signed_mean_change'] for r in chosen],
                        color=PCOLORS[degree],alpha=.65,ls='-' if degree==0 else '--',lw=1,
                        label=('tanh' if degree==0 else f'degree {degree}') if seed==0 else None)
            ax.set_xscale('log');ax.axhline(0,color='.6',lw=.6);ax.set_title(f'{LABELS[target]} · N={n}')
            if ti==0: ax.set_ylabel('Mean-|a| change relative to κ=1')
            if ni==2: ax.set_xlabel('Readout / geometry learning rate κ')
    axes[0,-1].legend(fontsize=8)
    finish(fig,root/'predicted_rate_contrasts.png','Does the independent moment model predict the effect of changing readout speed?')


def matched_curves(bundles,root):
    chosen=[b for b in bundles if b['config']['n']==128]
    fig,axes=plt.subplots(3,2,figsize=(13,10),squeeze=False)
    for row,kappa in enumerate((1e-4,1.,100.)):
        for col,left in enumerate(('moment3','moment5')):
            ax=axes[row,col]
            for degree in (0,1,3,5,7):
                for bundle in chosen:
                    tau,a=trace(bundle,left,kappa,degree);_,b=trace(bundle,'moment9',kappa,degree)
                    if a is None or b is None: continue
                    ax.plot(tau,a[:,1]-b[:,1],color=PCOLORS[degree],alpha=.6,ls='-' if degree==0 else '--',
                        label=('tanh' if degree==0 else f'degree {degree}') if bundle['config']['seed']==0 else None)
            ax.set_title(f'{left} − moment9 · κ={kappa:g}')
            ax.set_ylabel('Difference in mean |a|');ax.set_xlabel('Geometry time τ')
            ax.axhline(0,color='.6',lw=.6)
    axes[0,-1].legend(fontsize=9)
    finish(fig,root/'matched_targets.png','Same coarse target and tail energy; different unresolved moments · W=177')


def reference_errors(bundles,root):
    fig,axes=plt.subplots(3,5,figsize=(19,10),squeeze=False)
    for ri,kappa in enumerate((1e-4,1.,100.)):
        for ti,target in enumerate(targets.TARGETS):
            ax=axes[ri,ti]
            for b in bundles:
                if b['config']['n']!=128: continue
                index=next(i for i,c in enumerate(b['case_list']) if c['target']==target and c['kappa']==kappa)
                for degree in (1,3,5,7):
                    key=f'p{degree}_gradient_error'
                    if key not in b: continue
                    ax.semilogy(b[f'p{degree}_error_steps']*b['config']['eta'],b[key][index],
                        color=PCOLORS[degree],alpha=.6,lw=1,label=f'degree {degree}' if b['config']['seed']==0 else None)
            ax.set_title(f'{LABELS[target]} · κ={kappa:g}')
            if ti==0: ax.set_ylabel('Relative slope-gradient vector error')
            if ri==2: ax.set_xlabel('Geometry time τ')
    axes[0,-1].legend(fontsize=8)
    finish(fig,root/'reference_errors.png','Independent reference validity · W=177 · each line is one seed')


def distribution_curves(bundles,root):
    fig,axes=plt.subplots(3,5,figsize=(19,10),squeeze=False)
    selected=[b for b in bundles if b['config']['n']==128 and b['config']['seed']==0]
    if not selected: plt.close(fig);return
    bundle=selected[0]
    for col,target in enumerate(targets.TARGETS):
        for ki,kappa in enumerate(targets.RATIOS):
            tau,tr=trace(bundle,target,kappa)
            axes[0,col].plot(tau,tr[:,4],color=COLORS[ki],label=f'κ={kappa:g}')
            axes[0,col].fill_between(tau,tr[:,3],tr[:,5],color=COLORS[ki],alpha=.08)
            axes[1,col].plot(tau,tr[:,25],color=COLORS[ki])
            axes[2,col].plot(tau,tr[:,28],color=COLORS[ki])
        axes[0,col].set_title(LABELS[target]);axes[2,col].set_xlabel('Geometry time τ')
    axes[0,0].set_ylabel('Median |a|; shading: quartiles')
    axes[1,0].set_ylabel('Fraction with |a| ≥ 1')
    axes[2,0].set_ylabel('Fraction with λ ≥ 0.05')
    axes[0,-1].legend(fontsize=8,ncol=2)
    finish(fig,root/'scale_distribution.png','Population-scale acquisition · W=177, seed 0 · all rates')


def coarse_velocities(root):
    path=root/'sample_probes.csv'
    if not path.exists(): return
    with path.open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['bundle']=='core_N128_s0' and r['degree']=='0']
    if not rows or 'Vbias_0' not in rows[0]: return
    fig,axes=plt.subplots(3,5,figsize=(19,10),squeeze=False)
    for col,target in enumerate(targets.TARGETS):
        for ki in (0,4,6):
            kappa=targets.RATIOS[ki]
            chosen=sorted([r for r in rows if r['target']==target and float(r['kappa'])==kappa],key=lambda r:int(r['step']))
            tau=[int(r['step'])*.002 for r in chosen]
            for j,name in enumerate(('Vv','Vq','Vbias')):
                axes[j,col].plot(tau,[float(r[name+'_coarse_energy_removal']) for r in chosen],color=COLORS[ki],label=f'κ={kappa:g}')
                axes[j,col].set_yscale('symlog',linthresh=1e-8)
        axes[0,col].set_title(LABELS[target]);axes[2,col].set_xlabel('Geometry time τ')
    for j,label in enumerate(('Readout block: −mᵀVᵥ','Hidden block: −mᵀVq','Output bias within readout: −mᵀVbias')):
        axes[j,0].set_ylabel(label)
    axes[0,-1].legend(fontsize=8)
    finish(fig,root/'coarse_velocities.png','Instantaneous coarse-energy removal · positive removes, negative adds · W=177, seed 0')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    bundles=load(args.root);rows=json.loads((args.root/'summary.json').read_text())
    scale_curves(bundles,args.root);signal_curves(bundles,args.root)
    rate_contrasts(rows,args.root);matched_curves(bundles,args.root);reference_errors(bundles,args.root)
    distribution_curves(bundles,args.root);coarse_velocities(args.root)


if __name__=='__main__':main()
