"""Curated numerical evidence and figures; report prose is authored separately."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text())


def curvature_error(row):
    c=row['curvature'][0]
    pred=c.get('predicted_curvature',row['direction_cosine']/row['scale'])
    return abs(c['curvature']-pred)/max(abs(c['curvature']),abs(pred),1e-300)


def collect(root):
    records=[]
    for path in sorted(root.glob('*/case.json')):
        folder=path.parent
        if not (folder/'latest.json').exists():continue
        config=read(path);last=read(folder/'latest.json')
        rows=[read(p) for p in sorted(folder.glob('diagnostic_*.json'))]
        rows=[r for r in rows if 'exposure' in r]
        traces=[]
        for p in sorted(folder.glob('ssb_trace_*.npz')):
            a=np.load(p);t=a['trace'];traces.append(t[np.isfinite(t[:,0])])
        trace=np.concatenate(traces) if traces else np.empty((0,18))
        events=[r for r in rows if r.get('event')]
        summary=dict(id=folder.name,config=config,latest=last,
            event_steps=[r['step'] for r in events],event_betas=[r.get('beta') for r in events],
            candidate_steps=[r['step'] for r in rows if r.get('beta_candidate') is not None],
            event_curvature_discrepancy=[curvature_error(r) for r in events],
            reliable_candidates={str(t):sum(curvature_error(r)<=t and r['curvature'][0]['curvature']>0 and
                r['curvature'][0]['hvp_fd_relative']<=.1 for r in rows if r.get('beta_candidate') is not None) for t in (.1,.25,.5)},
            last_2048_mean_mse=float(np.mean(trace[-2048:,0])) if len(trace) else None,
            accepted_trace_rows=len(trace),emergency_resets=int(np.nansum(trace[:,14])) if len(trace) else None,
            gamma_rms_path_length=float(np.sum(trace[:,11])) if len(trace) else None,
            readout_rms_path_length=float(np.sum(trace[:,10])) if len(trace) else None,
            initial_lambda=rows[0]['lambda_quantiles'] if rows else None,
            final_lambda=rows[-1]['lambda_quantiles'] if rows else None)
        if (folder/'replacement.json').exists():summary['replacement']=read(folder/'replacement.json')
        summary['curve']=[dict(step=r['step'],mse=r['mse'],lambda_median=r['lambda_quantiles'][2],
            gain=r['gain_svd'],geometry_gradient=r['geometry_gradient'],readout_gradient=r['readout_gradient'],
            readout_norm=r['readout_norm'],exposure=r['exposure'],uncertainty=r['uncertainty'],
            curvature_discrepancy=curvature_error(r)) for r in rows]
        records.append(summary)
    return records


COLORS=dict(baseline='#222222',adaptive='#0072b2',periodic='#d55e00',sham='#009e73')
LABELS=dict(baseline='SSBroyden',adaptive='Adaptive metric mixture',periodic='Periodic metric mixture',sham='Search-history restart')


def figures(records,out):
    primary=[r for r in records if r['config'].get('implementation')=='primed_guard_v2' and 'parent' not in r['config']]
    for n in sorted({r['config']['n'] for r in primary}):
        for cohort,seeds in [('selection',(0,1)),('confirmation',(2,3,4))]:
            selected=[r for r in primary if r['config']['n']==n and r['config']['seed'] in seeds]
            if not selected:continue
            fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
            for r in selected:
                c=r['config'];ax=axes[['sine','mixed'].index(c['target']),['individual','neighbor'].index(c['coordinates'])]
                points=r['curve'];step=[q['step'] for q in points];mse=[q['mse'] for q in points]
                ax.semilogy(step,mse,color=COLORS[c['policy']],ls=['-','--',':'][seeds.index(c['seed'])],lw=1.2,
                    label=f"{LABELS[c['policy']]}, seed {c['seed']}")
                if r['latest']['status']!='continuing':ax.scatter(step[-1],mse[-1],marker='x',color=COLORS[c['policy']],s=35)
            for i,target in enumerate(('Sine','Mixed sine')):
                for j,coord in enumerate(('Individual scales','Neighbor differences')):
                    axes[i,j].set_title(f'{target}; {coord}');axes[i,j].set_ylabel('Training MSE')
                    axes[i,j].grid(alpha=.2);axes[i,j].set_xlabel('Accepted updates')
            axes[0,1].legend(fontsize=7,ncol=2)
            fig.suptitle(f'N = {n}; {cohort} seeds; crosses mark numerical failures')
            fig.tight_layout();fig.savefig(out/f'loss_N{n}_{cohort}.png',dpi=180);plt.close(fig)
    selected=[r for r in primary if r['config']['n']==128 and r['config']['seed']==0 and r['config']['coordinates']=='individual']
    if selected:
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for r in selected:
            c=r['config'];j=['sine','mixed'].index(c['target']);q=r['curve'];x=[p['step'] for p in q]
            axes[0,j].plot(x,[p['lambda_median'] for p in q],color=COLORS[c['policy']],label=LABELS[c['policy']])
            axes[1,j].semilogy(x,[p['readout_gradient'] for p in q],color=COLORS[c['policy']])
            axes[2,j].semilogy(x,[p['geometry_gradient'] for p in q],color=COLORS[c['policy']])
        for j,title in enumerate(('Sine','Mixed sine')):
            axes[0,j].set_title(title);axes[0,j].set_ylabel('Median |λ|')
            axes[1,j].set_ylabel('Native readout gradient norm');axes[2,j].set_ylabel('Native bandwidth gradient norm')
            axes[2,j].set_xlabel('Accepted updates')
            for ax in axes[:,j]:ax.grid(alpha=.2)
        axes[0,1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'geometry_and_signal.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=collect(args.root)
    (args.output/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    figures(records,args.output)
    for r in records:
        c=r['config']
        if c.get('implementation')!='primed_guard_v2' and 'parent' not in c:continue
        print(c['n'],c['coordinates'],c['target'],c['seed'],c.get('reinitialization',c['policy']),
              r['latest']['step'],f"{r['latest']['train_mse']:.3e}",r['latest']['status'],len(r['event_steps']))


if __name__=='__main__':main()
