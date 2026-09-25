"""One-state-at-a-time audit of saved target/width/dilation panels on Modal CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .population_balance_dynamics import diagnostics, comparison

PANELS=('late_development','late_confirmation','wide_N512','wide_N1024')


def checked_load(path):
    with zipfile.ZipFile(path) as archive:
        if sum(p.file_size for p in archive.infolist())>96*1024**2:
            raise ValueError(f'Unexpected decompressed archive size: {path}')
    return np.load(path,allow_pickle=False)


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]; candidates=[]; skipped=[]; seen=set()
    for panel in PANELS:
        source=args.base/(panel+'.npz')
        manifest=json.loads((args.base/(panel+'_run/manifest.json')).read_text())
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=manifest['input_sha256']: raise ValueError('Input provenance mismatch')
        with checked_load(source) as pack:
            x,ys=pack['x'],pack['y']
        xx=jnp.asarray(x)
        for count in (0,20000):
            snapshot=args.base/f'{panel}_run/{count:09d}.npz'
            with checked_load(snapshot) as pack:
                ps=pack['p']; failed=pack['failed'] if 'failed' in pack.files else np.zeros(len(ps),bool)
            for i,case in enumerate(manifest['cases']):
                source_index=case['input_index']; p=ps[i]; y=ys[source_index]
                if failed[i] or not np.isfinite(p).all():
                    skipped.append(dict(panel=panel,count=count,index=i,reason='failed/nonfinite')); continue
                identity=hashlib.sha256(p.tobytes()+x.tobytes()+y.tobytes()).hexdigest()
                # Retain duplicates with their intervention labels, but report unique-state coverage too.
                seen.add(identity)
                d={k:float(v) for k,v in jax.device_get(diagnostics(jnp.asarray(p),xx,jnp.asarray(y))).items()}
                bound=comparison(d,d['M'])
                trials=[]
                for factor in (1.02,1.05,1.1,1.2,1.5,2.,3.,4.,6.,8.):
                    speed=comparison(d,d['M']*factor)['force']
                    trials.append((np.sqrt(d['M']*factor)-np.sqrt(d['M']))/(speed+d['R_norm']))
                row=dict(panel=panel,target=case['target'],seed=case['seed'],start=case['start'],
                         nref=case.get('nref',case.get('Nref')),arm=case['arm'],scale=case['scale'],
                         count=count,input_index=source_index,snapshot_index=i,state_sha256=identity,
                         source_sha256=digest,**d,force_bound=bound['force'],
                         force_bound_ratio=bound['force']/max(d['F_norm'],1e-300),
                         tracking_ratio=d['R_norm']/max(d['F_norm'],1e-300),
                         static_budget_time=max(trials))
                rows.append(row)
                if panel=='wide_N512' and count==0 and case['arm']=='original' and case['seed']==31:
                    candidates.append((row,p.copy(),y.copy(),dict(case),x.copy()))
            print(f'{panel} count={count}: {len(rows)} rows, {len(skipped)} skipped',flush=True)
        del ps,ys
    # Fixed rule: four distinct target families with the shortest baseline comparison durations.
    families={'moment5':'polynomial','mixed_sine':'oscillatory','gauss_left':'gaussian',
              'bump_right':'bump','step_right':'step','kink_abs':'kink'}
    selected=[]; selected_families=set()
    for item in sorted(candidates,key=lambda c:c[0]['static_budget_time']):
        family=families[item[0]['target']]
        if family in selected_families: continue
        selected.append(item); selected_families.add(family)
        if len(selected)==4: break
    if len(selected)!=4: raise ValueError('Expected four distinct-family stress cases')
    if any(not np.array_equal(item[4],selected[0][4]) for item in selected): raise ValueError('Incompatible grids')
    np.savez_compressed(args.output/'stress_inputs.npz',p=np.stack([c[1] for c in selected]),
                        y=np.stack([c[2] for c in selected]),x=selected[0][4],
                        cases=np.array(json.dumps([c[3] for c in selected])))
    with (args.output/'states.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    subsets={name:[r for r in rows if r['arm']=='original' and r['panel']==name] for name in PANELS}
    facts=dict(rows=len(rows),unique_states=len(seen),targets=sorted({r['target'] for r in rows}),
               widths=sorted({int(r['width']) for r in rows}),seeds=sorted({r['seed'] for r in rows}),
               scales=sorted({r['scale'] for r in rows}),skipped=skipped,
               max_delta_bound_violation=max(-r['delta_bound_slack'] for r in rows),
               max_relative_force_bound_violation=max((r['F_norm']-r['force_bound'])/max(r['F_norm'],1e-300) for r in rows),
               natural_panels={name:dict(rows=len(rs),targets=len({r['target'] for r in rs}),
                   static_budget_time_min=min(r['static_budget_time'] for r in rs),
                   static_budget_time_median=float(np.median([r['static_budget_time'] for r in rs])),
                   alignment_min=min(abs(r['rho']) for r in rs)) for name,rs in subsets.items()},
               stress_selection_rule='Shortest static comparison duration among width705 original seed31 checkpoints; four distinct families. Diagnostic selection, not independent held-out evaluation.',
               selected=[{k:c[0][k] for k in ('target','seed','width','start','static_budget_time','tracking_ratio')} for c in selected],
               scope='Initial and 20k-additional-update snapshots only; static durations freeze shape for diagnosis and are not persistence guarantees')
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,rs in subsets.items():
        axes[0].scatter([r['width'] for r in rs],[r['static_budget_time']/.002 for r in rs],s=22,alpha=.65,label=name.replace('_',' '))
    axes[0].set(yscale='log',xlabel='Width',ylabel='Static-shape comparison duration (GD updates)',title='Diagnostic durations across natural checkpoints')
    axes[0].legend(fontsize=7)
    for arm in sorted({r['arm'] for r in rows}):
        rs=[r for r in rows if r['arm']==arm and r['count']==0]
        axes[1].scatter([r['M'] for r in rs],[r['force_bound_ratio'] for r in rs],s=12,alpha=.4,label=arm)
    axes[1].set(xscale='log',yscale='log',xlabel='Total hidden squared norm',ylabel='Structural force bound / exact effective force',title='Dilation exposes the limit of broad-feature bounds')
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(args.output/'archive_coverage.png',dpi=180); plt.close(fig)
    print(json.dumps(facts,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
