"""Plot curated cases; numerical artifacts only, no generated report prose."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import core,history,run,diagnose

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


def training_comparison(root,entries,out,offset=0):
    """Compare complete 5k-update loss means, with no validation-cadence aliasing."""
    targets=list(dict.fromkeys(e['target'] for e in entries))
    labels=list(dict.fromkeys(e['label'] for e in entries));colors=dict(zip(labels,plt.cm.tab10(range(len(labels)))))
    fig,axes=plt.subplots(len(targets),2,figsize=(12,3*len(targets)),squeeze=False)
    saved={}
    for row,target in enumerate(targets):
        for entry in entries:
            if entry['target']!=target:continue
            folder=root/entry['id'];sums=np.zeros(100);counts=np.zeros(100)
            for _,data in history.arrays(folder,'trace_*.npz'):
                start=int(data['start']);loss=data['trace'][:,0];index=(start+np.arange(len(loss)))//5000
                assert index.max()<len(sums)
                sums+=np.bincount(index,weights=loss,minlength=len(sums));counts+=np.bincount(index,minlength=len(sums))
            keep=counts==5000;steps=(np.flatnonzero(keep)+1)*5000+offset;values=sums[keep]/counts[keep]
            saved[entry['id']+'_step']=steps;saved[entry['id']+'_train_mse']=values
            style=['-', '--', ':'][entry['seed']%3];color=colors[entry['label']]
            axes[row,0].semilogy(steps,values,style,color=color,lw=1,label=f"{entry['label']}, seed {entry['seed']}")
            evaluations=[v for v in history.evaluations(folder) if v['step']%5000==0]
            axes[row,1].plot([v['step']+offset for v in evaluations],[v['lambda_median'] for v in evaluations],style,color=color,lw=1)
        axes[row,0].set(title=target,ylabel='Mean training MSE per 5k updates')
        axes[row,1].set(title='Geometry evolution',ylabel=r'Median $|\lambda|$')
        axes[row,0].legend(fontsize=8,ncol=2)
        for ax in axes[row]:ax.set_xlabel('Total updates');ax.grid(alpha=.2)
    run.save(out.with_suffix('.npz'),**saved)
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


def motion(folder,out):
    """Show physical gradient signal and actual motion in consecutive trace blocks."""
    steps=[];values=[]
    columns=['mse','physical_readout_gradient','physical_gamma_gradient',
             'delta_readout_rms','delta_gamma_rms','eta']
    indices=[core.TRACE.index(k) for k in columns]
    for _,data in history.arrays(folder,'trace_*.npz'):
        a=data['trace'];start=int(data['start'])
        for offset in range(0,len(a),1000):
            block=a[offset:offset+1000]
            if not np.all(np.isfinite(block[:,indices])):continue
            steps.append(start+offset+len(block))
            values.append(np.mean(block[:,indices],axis=0))
    if not steps:return
    a=np.asarray(values);fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].semilogy(steps,a[:,0]);axes[0].set_ylabel('Training MSE')
    axes[1].semilogy(steps,a[:,1],label='Readouts and output bias')
    axes[1].semilogy(steps,a[:,2],label='Slopes gamma')
    axes[1].set_ylabel('Physical gradient norm')
    axes[2].semilogy(steps,a[:,3],label='Readouts and output bias')
    axes[2].semilogy(steps,a[:,4],label='Slopes gamma')
    axes[2].set(ylabel='RMS physical step',xlabel='Updates')
    for ax in axes:ax.grid(alpha=.2)
    axes[1].legend();axes[2].legend()
    fig.suptitle('Training signal and motion; means over at most 1,000 updates')
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def frequency_evolution(folder,out,grid_size=8192):
    config=json.loads((folder/'case.json').read_text());g=core.old.geometry(config['n'])
    x=-1+2*(np.arange(grid_size)+.5)/grid_size;y=core.target(x,config['target'],np)
    steps=[];energies=[]
    for _,data in history.arrays(folder,'snapshot_*.npz'):
        step=int(data['step'])
        if step>20000 and step%20000:continue
        residual=diagnose.forward(data['z'],g,config['coordinates'],x)-y
        steps.append(step);energies.append(diagnose.band_energy(residual))
    if not steps:return
    energy=np.asarray(energies)
    labels=['DC (spatial mean)','1','2–3','4–7','8–15','16–31','32–63','64–127','128–255','256+']
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for i,label in enumerate(labels):
        color=plt.cm.tab10(i)
        axes[0].semilogy(steps,energy[:,i],label=label,color=color)
        axes[1].plot(steps,100*energy[:,i]/energy.sum(axis=1),color=color)
    axes[0].set_ylabel('Residual MSE in Fourier band')
    axes[1].set(ylabel='Percent of residual MSE',xlabel='Updates',ylim=(0,100))
    axes[0].legend(ncol=5,fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f'Spatial frequency indices; {grid_size:,} midpoint samples on [-1, 1]')
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def construction(folder,reference,out):
    config=json.loads((folder/'case.json').read_text());g=core.old.geometry(config['n'])
    assert config['target']=='sine' and config.get('architecture','fixed')=='fixed'
    with np.load(reference) as d:ref={k:d[k] for k in d.files}
    np.testing.assert_allclose(g.centers,ref['centers'],atol=0,rtol=0)
    with np.load(sorted(folder.glob('snapshot_*.npz'))[-1]) as d:z=d['z'];step=int(d['step'])
    c,gamma=map(np.asarray,core.physical(z,g,config['coordinates']))
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    axes[0].plot(g.centers,ref['c'][1:],label='Construction at lambda = 0.25',lw=1.5)
    axes[0].plot(g.centers,c[1:],'.',label=f'Trained, update {step:,}',ms=3)
    axes[1].plot(g.centers,ref['gamma'],label='Construction',lw=1.5)
    axes[1].plot(g.centers,gamma,'.',label='Trained',ms=3)
    axes[0].set_ylabel('Physical readout w');axes[1].set(ylabel='Physical slope gamma',xlabel='Fixed physical center')
    for ax in axes:
        ax.axvline(-1,color='grey',ls=':',lw=.8);ax.axvline(1,color='grey',ls=':',lw=.8)
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Detached construction reference; coefficients need not be unique')
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def geometry_signal(path,out):
    with np.load(path) as d:a={k:d[k] for k in d.files}
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    groups=[ [('physical_gamma_gradient','Total'),('coarse_gamma_gradient','From constant + linear residual'),
              ('remainder_gamma_gradient','From remaining residual')],
             [('physical_gamma_gradient','Total'),('readout_span_gamma_gradient','From retained readout span'),
              ('orthogonal_gamma_gradient','From orthogonal residual')] ]
    for ax,group in zip(axes,groups):
        scale=max(np.max(abs(a[k])) for k,_ in group)
        for k,label in group:
            ax.plot(a['centers'],a[k],'.',ms=3,label=f'{label} (norm {np.linalg.norm(a[k]):.2g})')
        ax.set_yscale('symlog',linthresh=max(scale*1e-5,1e-30));ax.grid(alpha=.2)
        exponent=int(np.ceil(np.log10(max(scale,1e-30))))
        positive=10.**np.arange(exponent-4,exponent+1,2)
        ax.set_yticks(np.r_[-positive[::-1],0.,positive])
        ax.set_ylabel('Physical slope gradient');ax.legend(fontsize=8)
    axes[1].set_xlabel('Fixed physical center')
    fig.suptitle('Residual contributions to gamma gradients; readout-span cutoff = 1e-12')
    fig.tight_layout();fig.savefig(out);plt.close(fig)


def frequency_signal(path,out):
    with np.load(path) as d:a={k:d[k] for k in d.files}
    labels=['DC','1','2–3','4–7','8–15','16–31','32–63','64–127','128–255','256+']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    axes[0].bar(labels,a['band_energy']);axes[0].set(ylabel='Residual MSE in band',title='Residual spectrum',yscale='log')
    for ax,parts,total,title in zip(axes[1:],
        ['frequency_readout_gradient','frequency_gamma_gradient'],
        ['physical_readout_gradient','physical_gamma_gradient'],['Physical readout gradient','Physical slope gradient']):
        ax.bar(labels,np.linalg.norm(a[parts],axis=1))
        ax.axhline(np.linalg.norm(a[total]),color='black',ls='--',label='Norm after summing vectors')
        ax.set(ylabel='Norm contributed by residual band',title=title,yscale='log');ax.legend(fontsize=8)
    for ax in axes:ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Residual Fourier bands passed through the same Jacobian transpose; gradient norms do not add')
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
    scale=max(np.max(np.abs(np.mean(dense[k],axis=0))) for k in ('readout_descent','geometry_descent'))
    axes[1,1].set_yscale('symlog',linthresh=max(1e-30,scale*1e-3))
    exponent=int(np.ceil(np.log10(max(scale,1e-30))))
    positive=10.**np.arange(exponent-2,exponent+1,2)
    axes[1,1].set_yticks(np.r_[-positive[::-1],0.,positive])
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
