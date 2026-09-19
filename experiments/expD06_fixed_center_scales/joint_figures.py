"""Readable curves and per-seed physical-parameter movies from detached evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import joint_analysis as analysis, joint_conditioning as campaign, run


def save(fig,path):
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def figures(output):
    records=json.loads((output/'summary.json').read_text());dest=output/'figures';dest.mkdir(exist_ok=True)
    if (output/'sweep_summary.json').exists():
        sweep=json.loads((output/'sweep_summary.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,opt in zip(axes,('gd','adam')):
            for coord in campaign.MAPS:
                rows=sorted([r for r in sweep if r['case']['optimizer']==opt and r['case']['coordinates']==coord
                             and r['case']['n']==512 and r['case']['seed']==0 and r['eligible']
                             and r['case'].get('native_epsilon')==1e-12],key=lambda r:r['case']['eta'])
                ax.loglog([r['case']['eta'] for r in rows],[r['window_mean_mse'] for r in rows],
                          'o-',color=analysis.COLORS[coord],label=analysis.LABELS[coord])
            ax.set(title=opt.upper(),xlabel='One constant shared native rate eta',ylabel='Mean MSE over updates 80k–100k')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Matched-rate comparison, N=512 seed 0; divergent trials omitted from finite curves')
        save(fig,dest/'rate_search.png')
    for n in sorted({r['case']['n'] for r in records}):
        fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
        for col,opt in enumerate(campaign.OPTIMIZERS):
            for record in records:
                c=record['case']
                if c['n']!=n or c['optimizer']!=opt or not record['end']:continue
                label=f'{analysis.LABELS[c["coordinates"]]}, seed {c["seed"]}'
                color=analysis.COLORS[c['coordinates']];style='-' if c['seed']==0 else '--'
                with np.load(output/record['key']/'window_mse.npz') as a:
                    axes[0,col].loglog(a['step'],a['mean'],color=color,ls=style,label=label)
                    axes[0,col].fill_between(a['step'],a['quantiles'][1],a['quantiles'][3],color=color,alpha=.08)
                with np.load(output/record['key']/'history.npz') as h:
                    axes[1,col].semilogx(np.maximum(h['step'],1),h['lambda_quantiles'][:,1],color=color,ls=style,label=label)
                if record['status']['status']!='continuing':
                    axes[0,col].plot(record['end'],record['status']['train_mse'],'x',color=color,ms=9)
            axes[0,col].set(title=opt.upper(),ylabel='Window mean training MSE',xlabel='Accepted updates (higher order) / updates')
            axes[1,col].set(ylabel='Median core |lambda|',xlabel='Updates')
            axes[1,col].axhline(.25,color='.4',ls=':',label='Construction reference 0.25')
            for row in (0,1):axes[row,col].grid(alpha=.2)
            if axes[0,col].lines:axes[0,col].legend(fontsize=7)
        fig.suptitle(f'N={n}; constant shared GD/Adam rates; × = numerical/search stop; shading = 10–90% within window')
        save(fig,dest/f'learning_N{n}.png')
    for record in records:
        if not record['end']:continue
        folder=output/record['key'];c=record['case'];end=record['end']
        with np.load(folder/'history.npz') as h:
            fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
            bands=h['band_bounds'];total=np.maximum(h['band_mse'].sum(axis=1),1e-300)
            groups=((0,1,'DC (0)'),(1,8,'1–7'),(8,64,'8–63'),(64,128,'64–127'),(128,256,'128–255'),(256,np.inf,'256+'))
            for lo,hi,label in groups:
                mask=(bands[:,0]>=lo)&(bands[:,0]<hi);energy=h['band_mse'][:,mask].sum(axis=1)
                axes[0].loglog(np.maximum(h['step'],1),energy,label=label)
                axes[1].semilogx(np.maximum(h['step'],1),100*energy/total,label=label)
            axes[0].set(ylabel='Residual MSE in frequency group',xlabel='Saved update',title='Absolute energy')
            axes[1].set(ylabel='Share of residual MSE (%)',xlabel='Saved update',title='Same groups, normalized at each checkpoint')
            for ax in axes:ax.grid(alpha=.2);ax.legend(title='DFT index magnitude',fontsize=8,ncol=2)
            fig.suptitle(f'{c["optimizer"].upper()} · {analysis.LABELS[c["coordinates"]]} · N={c["n"]}, seed={c["seed"]}')
            save(fig,dest/f'{record["key"]}_frequency_evolution.png')
        with np.load(folder/f'dense_{end}.npz') as a:
            fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
            bands=a['band_bounds'];labels=['DC (0)' if lo==0 else str(lo) if hi==lo+1 else f'{lo}–{hi-1}' for lo,hi in bands]
            energy=a['band_mse'].mean(axis=0);idx=np.arange(len(labels));total=max(energy.sum(),1e-300)
            axes[0,0].bar(idx,energy,color='#2166ac');axes[0,0].set(yscale='log',ylabel='Mean residual MSE in band',title='Absolute Fourier energy')
            axes[0,1].bar(idx,100*energy/total,color='#2166ac');axes[0,1].set(ylabel='Share of mean residual MSE (%)',title='Same samples and bands, normalized')
            for ax in axes[0]:ax.set_xticks(idx,labels,rotation=60,ha='right');ax.set_xlabel('DFT index magnitude; both signs combined')
            axes[1,0].plot(idx,a['band_readout_linear_mse_change'].mean(axis=0),'o-',label='Readout')
            axes[1,0].plot(idx,a['band_geometry_linear_mse_change'].mean(axis=0),'s-',label='Geometry')
            axes[1,0].set(yscale='symlog',ylabel='Mean first-order MSE change',title='Negative values predict descent')
            scale=max(float(np.max(np.abs(np.r_[a['band_readout_linear_mse_change'],a['band_geometry_linear_mse_change']]))),1e-30)
            axes[1,0].set_yscale('symlog',linthresh=scale*1e-4)
            tick_scale=10.**np.floor(np.log10(scale))
            axes[1,0].set_yticks(np.r_[-tick_scale*np.array([1.,.01,.0001]),0.,tick_scale*np.array([.0001,.01,1.])])
            axes[1,0].set_xticks(idx,labels,rotation=60,ha='right');axes[1,0].legend()
            s=a['singular_readout'];rel=s/s[0];edges=np.array([0,1e-14,1e-10,1e-6,1e-4,1e-3,.1,1.01])
            tick=['<1e−14','1e−14–1e−10','1e−10–1e−6','1e−6–1e−4','1e−4–1e−3','1e−3–0.1','≥0.1','Outside basis']
            for offset,field,denominator,label in ((-.2,'readout_residual_modal',a['residual_mse'].mean(),'Residual'),
                                                  (.2,'readout_update_modal',a['readout_update_energy'].mean(),'Readout update')):
                modal=np.mean(a[field]**2,axis=0)
                fractions=[modal[(rel>=lo)&(rel<hi)].sum()/max(denominator,1e-300) for lo,hi in zip(edges[:-1],edges[1:])]
                fractions.append(max(0.,1-sum(fractions)))
                axes[1,1].bar(np.arange(len(tick))+offset,100*np.asarray(fractions),width=.4,label=label)
            axes[1,1].set_xticks(np.arange(len(tick)),tick,rotation=55,ha='right')
            axes[1,1].set(ylabel='Share of function-space energy (%)',title='Fixed window-start readout basis',xlabel='Relative singular value sigma / sigma_max')
            axes[1,1].legend()
            for ax in axes.flat:ax.grid(alpha=.15)
            fig.suptitle(f'{c["optimizer"].upper()} · {analysis.LABELS[c["coordinates"]]} · N={c["n"]}, seed={c["seed"]}\n'
                         f'{len(a["step"])} sampled actual updates from the final {min(2048,end):,}; endpoint {end:,}')
            save(fig,dest/f'{record["key"]}_mechanism.png')
        with np.load(folder/f'spectrum_{end}.npz') as a, np.load(folder/'history.npz') as h:
            fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
            for coord in campaign.MAPS:
                for row,kind in enumerate(('readout','joint')):
                    s=a[f'singular_{kind}_{coord}']
                    axes[row,0].semilogy(np.arange(1,len(s)+1),s/s[0],label=analysis.LABELS[coord])
                    axes[row,0].set(ylabel=f'Relative {kind} singular value',xlabel='Sorted mode index',title='Both maps at this same physical geometry')
                    axes[row,0].axhline(1e-14,color='.5',ls=':',lw=1)
                    axes[row,0].legend(fontsize=8)
            axes[0,1].plot(h['centers'],a['c'][1:],'.',ms=3,label='Trained physical w')
            axes[0,1].plot(h['centers'],a['readout_refit'][1:],'.',ms=2,label='Detached LS refit, same slopes')
            reference=output/f'construction_N{c["n"]}.npz'
            if reference.exists():
                with np.load(reference) as construction:
                    axes[0,1].plot(construction['centers'],construction['c'][1:],'.',ms=2,label='Construction at lambda=0.25')
            axes[0,1].set(yscale='symlog',ylabel='Physical readout w',xlabel='Fixed physical center');axes[0,1].legend(fontsize=8)
            axes[1,1].plot(h['centers'],a['gamma'],'.',ms=3);axes[1,1].set(ylabel='Signed physical gamma',xlabel='Fixed physical center')
            fig.suptitle(f'{c["optimizer"].upper()} · {analysis.LABELS[c["coordinates"]]} · N={c["n"]}, seed={c["seed"]}, update={end:,}')
            save(fig,dest/f'{record["key"]}_geometry.png')


def movies(output):
    from matplotlib.animation import FuncAnimation,FFMpegWriter
    records=json.loads((output/'summary.json').read_text())
    for opt in campaign.OPTIMIZERS:
        for seed in (0,1):
            pair=[r for coord in campaign.MAPS for r in records if r['case']['optimizer']==opt and r['case']['seed']==seed and r['case']['n']==512 and r['case']['coordinates']==coord and r['end']]
            if len(pair)!=2:continue
            histories=[]
            for r in pair:
                with np.load(output/r['key']/'history.npz') as h:histories.append(dict(h))
            # Use union times. A stopped arm visibly holds its last actual saved state.
            times=sorted(set(np.concatenate([h['step'] for h in histories]).tolist()))
            times=[s for s in times if s<=300000 or s%100000==0 or any(s==r['end'] for r in pair)]
            frames=np.repeat(times,[5 if s<=300000 else 1 for s in times]);fps=5
            fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True,layout='constrained');lines=[];notes=[]
            for col,(record,h) in enumerate(zip(pair,histories)):
                for row,field in enumerate(('c','gamma')):
                    values=h[field][:,1:] if row==0 else h[field]
                    bound=max(float(np.max(np.abs(values)))*1.1,1e-9)
                    line,=axes[row,col].plot(h['centers'],values[0],'.',ms=3);lines.append((line,row,col,field))
                    axes[row,col].set(ylim=(-bound,bound),ylabel='Physical w' if row==0 else 'Physical gamma')
                    axes[row,col].set_yscale('symlog',linthresh=.01 if row==0 else .1)
                    axes[row,col].axvspan(h['centers'][0],-1,color='.93');axes[row,col].axvspan(1,h['centers'][-1],color='.93')
                    axes[row,col].grid(alpha=.2)
                axes[0,col].set_title(analysis.LABELS[record['case']['coordinates']]);axes[1,col].set_xlabel('Fixed physical center')
                notes.append(axes[0,col].text(.02,.98,'',transform=axes[0,col].transAxes,va='top'))
            title=fig.suptitle('')
            def update(index):
                requested=int(frames[index])
                for line,row,col,field in lines:
                    h=histories[col];k=max(0,np.searchsorted(h['step'],requested,side='right')-1)
                    line.set_ydata(h[field][k,1:] if row==0 else h[field][k])
                    if row==0:
                        status=pair[col]['status']['status'] if requested>=pair[col]['end'] else 'saved state'
                        notes[col].set_text(f'Update {h["step"][k]:,}; bias={h["c"][k,0]:+.3g}\n{status}')
                title.set_text(f'{opt.upper()} · seed {seed} · timeline update {requested:,}\nActual saved states; first 300k shown at one checkpoint/second')
            movie=FuncAnimation(fig,update,frames=len(frames),interval=1000/fps,repeat=False)
            name=f'{opt}_seed_{seed}'
            movie.save(output/f'{name}.mp4',writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=1400,extra_args=['-threads','1','-pix_fmt','yuv420p','-movflags','+faststart']),dpi=100)
            update(len(frames)//2);fig.savefig(output/f'{name}_middle.png',dpi=100);plt.close(fig)
            run.write_json(output/f'{name}_animation.json',dict(cases=[r['key'] for r in pair],steps=frames.tolist(),fps=fps))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--movies',action='store_true');args=parser.parse_args()
    movies(args.output) if args.movies else figures(args.output)


if __name__=='__main__':main()
