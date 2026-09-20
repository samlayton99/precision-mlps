"""Plot curated cases; numerical artifacts only, no generated report prose."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import core,history,run

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':160})


def curves(root,entries,out):
    groups=sorted({(e['optimizer'],e['target']) for e in entries})
    fig,axes=plt.subplots(len(groups),2,figsize=(12,3*len(groups)),squeeze=False)
    labels=list(dict.fromkeys(e['label'] for e in entries))
    colors=dict(zip(labels,plt.cm.tab10(np.arange(len(labels))%10)))
    for row,group in enumerate(groups):
        for entry in entries:
            if (entry['optimizer'],entry['target'])!=group:continue
            values=history.evaluations(root/entry['id'])
            if not values:continue
            x=np.array([v['step'] for v in values]);color=colors[entry['label']]
            style='-' if entry['seed']%2==0 else '--'
            label=f"{entry['label']}, seed {entry['seed']}"
            axes[row,0].semilogy(x,[v['validation_mse'] for v in values],style,color=color,label=label,lw=1.1)
            axes[row,1].plot(x,[v['lambda_median'] for v in values],style,color=color,lw=1.1)
        axes[row,0].set(title=' / '.join(group),ylabel='Validation MSE')
        axes[row,1].set(title='Median absolute bandwidth',ylabel=r'$|\lambda|=h|\gamma|$')
        axes[row,0].legend(fontsize=8,ncol=2)
        for ax in axes[row]:ax.set_xlabel('Updates');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def parameters(folder,out):
    config=json.loads((folder/'case.json').read_text());g=core.old.geometry(config['n'])
    saved=list(history.arrays(folder,'snapshot_*.npz'))
    if not saved:return
    steps=np.array([int(a['step']) for _,a in saved])
    desired=[20,20000,100000,int(steps[-1])]
    chosen=sorted(set(int(np.argmin(abs(steps-s))) for s in desired))
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for index in chosen:
        data=saved[index][1];c,gamma=map(np.asarray,core.physical(data['z'],g,config['coordinates']))
        label=f"Update {int(data['step']):,}"
        axes[0].plot(g.centers,c[1:],'.-',ms=2,lw=.6,label=label)
        axes[1].plot(g.centers,gamma,'.-',ms=2,lw=.6,label=label)
    axes[0].set_ylabel('Physical readout w');axes[1].set_ylabel('Physical slope gamma')
    axes[1].set_xlabel('Fixed physical center')
    for ax in axes:ax.grid(alpha=.2);ax.legend(ncol=len(chosen),fontsize=8)
    fig.suptitle(f"{config['optimizer']} / {config['coordinates']} / {config['target']} / N={config['n']} / seed {config['seed']}")
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def mechanism(path,dense_path,out):
    with np.load(path) as d: data={k:d[k] for k in d.files}
    with np.load(dense_path) as d: dense={k:d[k] for k in d.files}
    frequency_labels=['DC','1','2–3','4–7','8–15','16–31','32–63','64–127','128–255','256+']
    mode_labels=['<1e−8','1e−8–1e−6','1e−6–1e−4','1e−4–1e−2','1e−2–0.1','0.1–1']
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    residual=np.mean(dense['residual'],axis=0)
    axes[0,0].bar(frequency_labels,residual)
    axes[0,0].set(yscale='log',ylabel='Residual MSE in band',title='Final window: spatial Fourier residual')
    axes[0,1].bar(frequency_labels,100*residual/residual.sum())
    axes[0,1].set(ylabel='Percent of residual MSE',title='Same residual, normalized to 100%')
    position=np.arange(len(mode_labels));width=.26
    for offset,(key,label) in enumerate([('mode_residual','Residual'),('mode_readout_update','Readout update'),('mode_geometry_update','Geometry update')]):
        value=np.mean(dense[key],axis=0)
        outside=float(np.mean(dense[{'mode_residual':'outside_fixed_span','mode_readout_update':'readout_update_outside_span','mode_geometry_update':'geometry_update_outside_span'}[key]]))
        total=value.sum()+outside
        axes[1,0].bar(position+(offset-1)*width,100*value/total,width,label=f'{label} ({100*outside/total:.2g}% outside span)')
    axes[1,0].set_xticks(position,mode_labels)
    axes[1,0].set(ylabel='Percent of respective energy',title=r'Fixed window-start SVD bands: $\sigma/\sigma_{max}$')
    axes[1,0].legend(fontsize=8)
    for key,label in [('readout_descent','Readout'),('geometry_descent','Geometry')]:
        axes[1,1].plot(frequency_labels,np.mean(dense[key],axis=0),'o-',label=label)
    axes[1,1].axhline(0,color='black',lw=.6)
    axes[1,1].set(yscale='symlog',ylabel=r'Mean $-2\langle r,\Delta f\rangle$ per update',title='Signed linear descent: positive reduces MSE')
    axes[1,1].set_yscale('symlog',linthresh=max(1e-30,np.max(np.abs(np.mean(dense['readout_descent'],axis=0)))*1e-4))
    axes[1,1].legend()
    for ax in axes.flat:ax.tick_params(axis='x',rotation=40);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    entries=json.loads(args.selection.read_text())
    curves(args.root,entries,args.out/'learning_curves.png')
    for e in entries:
        if e.get('parameters'):parameters(args.root/e['id'],args.out/(e['id']+'_parameters.png'))


if __name__=='__main__':main()
