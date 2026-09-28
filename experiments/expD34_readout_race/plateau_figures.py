"""Curate completed checkpoint and adjacent-step evidence; generate no prose."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import adam_analyze as aa, adam_forces as af, plateau


def select(rows,**values):
    keep=np.ones(len(rows),dtype=bool)
    for key,value in values.items():keep &= rows[key]==value
    return rows[keep]


def medians(rows,keys):
    records=[]
    for group in sorted(set(tuple(r[k] for k in keys) for r in rows)):
        r=select(rows,**dict(zip(keys,group)))
        records.append(dict(zip(keys,group))|{k:float(np.median(r[k])) for k in rows.dtype.names
            if rows.dtype[k].kind in 'iuf' and k not in keys})
    return records


def run(root):
    out=root/'figures';out.mkdir(exist_ok=True)
    read=lambda path:np.genfromtxt(path,delimiter=',',names=True,dtype=None,encoding='utf-8')
    states=read(root/'diagnostics/states.csv')
    windows=read(root/'diagnostics/windows.csv')
    forecasts=read(root/'diagnostics/forecasts.csv')
    dense=read(root/'dense/windows.csv')
    targets=af.TARGETS
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(3,5,figsize=(16,9),layout='constrained')
    for ax,t in zip(axes.flat,targets):
        for opt,color in [('gd','C0'),('adam','C1')]:
            for seed in range(5):
                rows=np.sort(select(states,target=t,optimizer=opt,seed=seed),order='step')
                ax.semilogy(rows['step']/1000,rows['force_norm'],color=color,alpha=.45,label=opt if seed==0 else None)
        ax.set_title(t);ax.set_xlabel('update (thousands)');ax.set_ylabel('effective slope-force norm')
    axes.flat[0].legend()
    for ax in list(axes.flat)[len(targets):]:ax.set_visible(False)
    fig.suptitle('Sparse force norms: five seeds, unchanged training')
    save(fig,'01_force_norms')

    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    metrics=[('lag1_cosine','Adjacent-vector cosine'),('lag2_cosine','Two-step vector cosine'),
             ('vector_coherence','Net vector / total path over 512 steps'),('norm_cv','Norm standard deviation / mean')]
    sub=dense[(dense['start']==600000)&np.isin(dense['channel'],['raw_effective','step_effective'])]
    for ax,(metric,label) in zip(axes.flat,metrics):
        for opt,channel,color,marker,offset in [('gd','raw_effective','C0','o',-.2),
            ('adam','raw_effective','C1','x',0),('adam','step_effective','C2','+',.2)]:
            for i,t in enumerate(targets):
                r=select(sub,target=t,optimizer=opt,channel=channel)[metric]
                ax.scatter(np.full(len(r),i+offset),r,s=18,color=color,marker=marker,alpha=.65,
                    label=f'{opt}: {channel}' if i==0 else None)
        ax.set_xticks(range(len(targets)),targets,rotation=65,ha='right');ax.set_ylabel(label);ax.grid(alpha=.2)
        if metric in ('vector_coherence','norm_cv'):ax.set_yscale('log')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Dense windows starting at 600k: each point is one seed')
    save(fig,'02_dense_vectors')

    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for metric,color,offset,label in [('frozen_relative_vector_error','C0',-.12,'Frozen Jacobian'),
        ('constant_relative_vector_error','C1',.12,'Constant effective force')]:
        for i,t in enumerate(targets):
            r=select(forecasts,target=t,end=600000)[metric]
            axes[0].scatter(np.full(len(r),i+offset),r,color=color,alpha=.7,s=20,label=label if i==0 else None)
    axes[0].set_yscale('log');axes[0].axhline(1,color='gray',ls=':');axes[0].legend()
    axes[0].set_ylabel('Relative force-vector prediction error at 600k')
    for i,t in enumerate(targets):
        r=select(windows,target=t,optimizer='gd',start=400000)
        axes[1].scatter(np.full(len(r),i),r['end_alignment'],color='C0',s=20)
    axes[1].axhline(0,color='gray',ls=':');axes[1].set_ylabel('Signed outward alignment at 600k (GD)')
    for ax in axes:ax.set_xticks(range(len(targets)),targets,rotation=65,ha='right');ax.grid(alpha=.2)
    fig.suptitle('Checkpoint predictions from 100k, and direction of the surviving force')
    save(fig,'03_forecasts_alignment')

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,t in zip(axes.flat,('moment9','moment5','sine','moment3')):
        rows=np.sort(select(states,target=t,optimizer='gd',step=600000),order='seed')
        for k,name in enumerate(plateau.DRIVERS):
            ax.plot(rows['seed'],rows['rate_'+name],marker='o',label=name)
        ax.plot(rows['seed'],rows['total_log_rate'],'k--',label='sum')
        ax.set_title(t);ax.set_xlabel('seed');ax.set_ylabel('Contribution to d log ||F|| / dt');ax.axhline(0,color='gray',lw=.5)
        ax.set_yscale('symlog',linthresh=1e-7)
    axes[0,0].legend(fontsize=8);fig.suptitle('GD force-amplitude drivers at 600k; t = 0.002 × update')
    save(fig,'04_force_drivers')
    # Tables and JSON are evidence artifacts; reports are authored separately.
    gd=select(windows,optimizer='gd',start=400000)
    aa.write_csv(out/'late_gd_medians.csv',medians(gd,['target']))
    aa.write_csv(out/'forecast_medians.csv',medians(select(forecasts,end=600000),['target']))
    aa.write_csv(out/'dense_medians.csv',medians(sub,['optimizer','channel']))
    summary={}
    for opt in ('gd','adam'):
        for ch in ('raw_effective','step_effective','raw_tracking','step_tracking'):
            r=select(dense,optimizer=opt,channel=ch,start=600000)
            summary[f'{opt}_{ch}']=dict(windows=len(r),negative_lag1=int((r['lag1_cosine']<0).sum()),
                lag1_quantiles=np.quantile(r['lag1_cosine'],[0,.25,.5,.75,1]).tolist(),
                lag2_median=float(np.median(r['lag2_cosine'])),coherence_median=float(np.median(r['vector_coherence'])),
                norm_cv_median=float(np.median(r['norm_cv'])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
