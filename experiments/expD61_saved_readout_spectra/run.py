"""Readout spectra of every saved D56/D58 checkpoint; no training."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/precisionmlps-mpl')
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import svd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/checkpoint_D_optimizers'
OUT=BASE/'expD61_saved_readout_spectra'
SWEEPS=[BASE/'expD56_scale_acquisition/sweep',BASE/'expD58_extra_hard_scale_acquisition/sweep']
PERCENTILES=np.arange(0,101,10)

def spectrum(theta,x):
    width=(len(theta)-1)//3
    a=np.column_stack((np.tanh(x[:,None]*theta[:width]+theta[width:2*width]),np.ones(len(x))))
    a/=np.sqrt(len(x))
    s=svd(a,compute_uv=False,check_finite=True,lapack_driver='gesvd')
    rates=(s/s[0])**2
    floor=(np.finfo(float).eps*max(a.shape))**2
    return rates, s[0]**2, floor

def process(path):
    status=json.loads((path.parent/'status.json').read_text())
    assert status['state']=='complete' and status['step']==200000
    destination=OUT/'runs'/path.parent.name
    destination.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    initial=path.parent/'initial.npz'
    initial_hash=hashlib.sha256(initial.read_bytes()).hexdigest()
    meta={k:status[k] for k in ('target','width','initialization_multiplier','learning_rate','seed')}
    meta.update(source=str(path.relative_to(ROOT)),source_sha256=digest,initial_sha256=initial_hash)
    if (destination/'metadata.json').exists():
        if json.loads((destination/'metadata.json').read_text())==meta:
            return path.parent.name
    with np.load(path) as z, np.load(initial) as init:
        theta=z['snapshots']; steps=z['history'][:,z['keys'].tolist().index('step')].astype(int)
        x=init['train_x'].reshape(-1)
    assert len(theta)==len(steps) and steps[0]==0 and steps[-1]==200000
    assert np.all(np.diff(steps)>0)
    rates=[]; maxima=[]
    for t in theta:
        r,m,floor=spectrum(t,x);rates.append(r);maxima.append(m)
    rates=np.array(rates)
    quantiles=np.quantile(rates,PERCENTILES/100,axis=1).T
    assert np.all(np.diff(quantiles,axis=1)>=0) and np.all(quantiles[:,-1]==1)
    np.savez_compressed(destination/'spectra.npz',steps=steps,rates=rates,
        percentiles=PERCENTILES,quantiles=quantiles,mu_max=maxima,resolution_floor=floor,
        unresolved_fraction=np.mean(rates<=floor,axis=1))
    np.savetxt(destination/'percentiles.csv',np.column_stack((steps,quantiles,
        np.mean(rates<=floor,axis=1))),delimiter=',',
        header='step,'+','.join(f'p{p:02}' for p in PERCENTILES)+',unresolved_fraction',comments='')
    (destination/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return path.parent.name

def load():
    records=[]
    for path in sorted((OUT/'runs').glob('*/spectra.npz')):
        meta=json.loads((path.parent/'metadata.json').read_text())
        with np.load(path) as z: record={k:z[k].copy() for k in z.files}
        records.append({**meta,**record})
    return records

def plot(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter,NullLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':9,
        'axes.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    colors=plt.colormaps['plasma'](np.linspace(.08,.80,9))
    floor=float(records[0]['resolution_floor'])
    titles=['a) Default Initialization','b) 2x Initialization','c) 4x Initialization']
    def draw(rows,title,individuals=False):
        fig,axes=plt.subplots(3,3,figsize=(9.4,7.4),sharex=True,sharey=True)
        fig.subplots_adjust(left=.11,right=.985,top=.90,bottom=.15,hspace=.23,wspace=.16)
        for j,init in enumerate([1,2,4]):
            axes[0,j].set_title(titles[j],fontweight='bold',pad=8)
            for i,lr in enumerate([.01,.001,.0001]):
                ax=axes[i,j]; selected=[r for r in rows if r['initialization_multiplier']==init and r['learning_rate']==lr]
                steps=selected[0]['steps'];q=np.array([r['quantiles'] for r in selected])
                for r in selected:np.testing.assert_array_equal(steps,r['steps'])
                for k,color in enumerate(colors,1):
                    if individuals:
                        for v in q:ax.plot(steps,np.maximum(v[:,k],floor),color=color,lw=.3,alpha=.13)
                    ax.plot(steps,np.maximum(np.median(q[:,:,k],axis=0),floor),color=color,
                            lw=1.1 if k==5 else .7,zorder=4)
                ax.axhspan(floor/6,floor,color='#EEEEEE',zorder=0)
                ax.axhline(floor,color='#777777',lw=.6,ls=':')
                ax.text(.97,.94,f'LR = {lr:g}',ha='right',va='top',transform=ax.transAxes,fontsize=7)
                for side in ['top','right']:ax.spines[side].set_visible(False)
                ax.grid(axis='y',color='#DDDDDD',lw=.45);ax.set_axisbelow(True)
                ax.set(xlim=(0,200000),yscale='log',ylim=(floor/6,2))
                ax.set_xticks([0,50000,100000,150000,200000])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}k' if x else '0'))
                ax.set_yticks([1,1e-5,1e-10,1e-15,1e-20,floor])
                ax.set_yticklabels(['$10^0$','$10^{-5}$','$10^{-10}$','$10^{-15}$','$10^{-20}$','Unresolved'])
                ax.yaxis.set_minor_locator(NullLocator())
                if j==0:ax.set_ylabel(r'Normalized rate $\rho_i$')
                if i==2:ax.set_xlabel('GD step')
        fig.suptitle(title,fontsize=12,fontweight='bold',y=.97)
        fig.legend([Line2D([],[],color=c,lw=1.2) for c in colors],
                   [f'{p}th' if p!=50 else '50th (median)' for p in range(10,100,10)],
                   loc='lower center',bbox_to_anchor=(.5,.065),ncol=9,frameon=False,
                   handlelength=1.5,columnspacing=1.)
        note='Percentiles across 197 readout modes; rates below the numerical resolution are placed on the bottom line.'
        if individuals:note+='\nSolid: median across 24 functions at each percentile. Faint: individual functions.'
        fig.text(.5,.025,note,ha='center',fontsize=7,color='#444444')
        return fig
    fig=draw(records,'Readout Spectrum: 24 functions',True)
    fig.savefig(figures/'spectrum_overview.png',dpi=200);fig.savefig(figures/'spectrum_overview.pdf');plt.close(fig)
    with PdfPages(figures/'all_24_functions.pdf') as pdf:
        for target in dict.fromkeys(r['target'] for r in records):
            fig=draw([r for r in records if r['target']==target],f'Readout Spectrum: {target.replace("_"," ")}')
            fig.savefig(figures/f'{target}.png',dpi=170);pdf.savefig(fig);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plot-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    paths=sorted(p for root in SWEEPS for p in (root/'runs').glob('*/checkpoint.npz'))
    assert len(paths)==216
    start=time.time()
    if not args.plot_only:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(process,p) for p in paths]
            for count,f in enumerate(as_completed(futures),1):
                name=f.result()
                if count%12==0 or count==1:
                    print(f'{count}/216 runs; {time.time()-start:.1f}s; {name}',flush=True)
    records=load();assert len(records)==216
    summary={'runs':len(records),'functions':24,'checkpoints':sum(len(r['steps']) for r in records),
        'modes_per_checkpoint':197,'percentiles':PERCENTILES.tolist(),
        'definition':'A=[tanh(x*a+b),1]/sqrt(1024); rho=(svd(A)/sigma_max)^2',
        'resolution_floor':float(records[0]['resolution_floor']),
        'resolution_rule':'(max(A.shape)*machine_epsilon)^2: conservative numerical rank convention, not a certified error bound',
        'unresolved_fraction_range_all_checkpoints':[float(min(r['unresolved_fraction'].min() for r in records)),float(max(r['unresolved_fraction'].max() for r in records))],
        'final_p90_range':[float(min(r['quantiles'][-1,9] for r in records)),float(max(r['quantiles'][-1,9] for r in records))],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    plot(records)
    print(f'Complete: {OUT}',flush=True)

if __name__=='__main__':main()
