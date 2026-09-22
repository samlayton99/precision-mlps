"""Independent-grid, replay, basis, and matched-time checks for useful slopes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import targets
from .mechanism import ARMS, design, force_metrics, frozen_curves
from .recovery import clean,write_table


def load(path):
    f=dict(np.load(path)); cases=json.loads(str(f['cases']))
    cases=[dict(seed=c[0],target=c[1]) if isinstance(c,list) else c for c in cases]
    return f,cases


def verify(root,primary):
    output=root/'verification'; output.mkdir(exist_ok=True)
    baselines={}
    for path in (primary,root/'missing/compact_states.npz'):
        f,cases=load(path)
        for i,c in enumerate(cases): baselines[c['seed'],c['target']]=(f,i)
    replay=[]; freeze=[]; refinement=[]; basis=[]; grids=[]
    x=targets.grid(2048); xe=targets.grid(8192); mapping=targets.polynomial_map(x)
    for path in sorted(root.glob('seed*_fork*/compact_states.npz'))+sorted(root.glob('dense_seed*/compact_states.npz')):
        f,cases=load(path)
        status=json.loads((path.parent/'status.json').read_text())
        if not status['complete'] or f['steps'][-1]!=600000: raise ValueError(f'Incomplete {path}')
        for i,c in enumerate(cases):
            rates=ARMS[c['arm']]
            for block in range(3):
                if rates[block]==0:
                    error=float(np.max(abs(f['z'][i,:,block]-f['z'][i,0,block])))
                    freeze.append(dict(**c,block=block,error=error)); assert error==0
            if rates[3]==0:
                error=float(np.max(abs(f['d'][i]-f['d'][i,0])))
                freeze.append(dict(**c,block=3,error=error)); assert error==0
            if c['arm']=='joint':
                original,oi=baselines[c['seed'],c['target']]
                for j,step in enumerate(f['steps']):
                    js=np.flatnonzero(original['steps']==step)
                    if not len(js): continue
                    oj=int(js[0]); diff=f['z'][i,j]-original['z'][oi,oj]
                    replay.append(dict(**c,step=int(step),parameter_l2=np.linalg.norm(diff),
                        parameter_max=np.max(abs(diff)),bias_error=abs(f['d'][i,j]-original['d'][oi,oj])))
            if c['seed']==0:
                # Endpoints from every intervention exercise their own actual vector fields.
                z,d,y=f['z'][i,-1],f['d'][i,-1],f['y'][i]
                low,la=force_metrics(z,d,x,y,rates,degree=65)
                high,ha=force_metrics(z,d,x,y,rates,degree=129)
                basis.append(dict(**c,step=600000,full_force=low['full_slope_norm'],
                    omitted65=low['omitted_slope_norm'],omitted129=high['omitted_slope_norm'],
                    effective_force_difference=np.linalg.norm(la['effective_force']-ha['effective_force'])))
        if path.parent.name.startswith('dense_') or cases[0]['seed']!=0: continue
        fork=cases[0]['fork_step']; half,hcases=load(root/f'half_fork{fork}/compact_states.npz')
        assert half['steps'][-1]==f['steps'][-1]
        assert cases==[dict(c,training_eta=.002) for c in hcases]
        np.testing.assert_array_equal(f['z'][:,0],half['z'][:,0])
        for i,c in enumerate(cases):
            z,d=f['z'][i,-1],f['d'][i,-1]; hz,hd=half['z'][i,-1],half['d'][i,-1]
            ye=targets.values(c['target'],xe,mapping)
            r=design(z[0],z[1],xe) @ np.r_[z[2],d]-ye
            hr=design(hz[0],hz[1],xe) @ np.r_[hz[2],hd]-ye
            fc,_,_=frozen_curves(z[0],z[1],x,f['y'][i],xe,ye,(600000,))
            hc,_,_=frozen_curves(hz[0],hz[1],x,f['y'][i],xe,ye,(600000,))
            refinement.append(dict(**c,full_step=.002,half_step=.001,
                parameter_l2=np.linalg.norm(z-hz),mean_gamma_difference=np.mean(abs(hz[0]))-np.mean(abs(z[0])),
                heldout_mse_difference=np.mean(hr*hr)-np.mean(r*r),
                frozen_relative_mse_difference=hc[0]['relative_heldout_mse']-fc[0]['relative_heldout_mse']))
    centers=-1+2*np.arange(-24,153)/128
    for target in targets.TARGETS:
        for gamma in (1.,3.2,16.):
            a=np.full(177,gamma); b=-gamma*centers
            xx=targets.grid(4096); xee=targets.grid(16384)
            yy=targets.values(target,x,mapping); ye=targets.values(target,xe,mapping)
            yy2=targets.values(target,xx,mapping); ye2=targets.values(target,xee,mapping)
            rr,_,_=frozen_curves(a,b,x,yy,xe,ye,(600000,))
            fine,_,_=frozen_curves(a,b,xx,yy2,xee,ye2,(600000,))
            grids.append(dict(target=target,gamma=gamma,coarse=rr[0]['relative_heldout_mse'],
                              fine=fine[0]['relative_heldout_mse'],difference=fine[0]['relative_heldout_mse']-rr[0]['relative_heldout_mse']))
    for name,rows in [('unchanged_gd_replay',replay),('frozen_blocks',freeze),('halfstep',refinement),('modal_basis',basis),('frozen_grid',grids)]:
        write_table(output/(name+'.csv'),rows)
    result=dict(replay_comparisons=len(replay),replay_max_parameter_error=max(r['parameter_max'] for r in replay),
        frozen_blocks_checked=len(freeze),freeze_max_error=max(r['error'] for r in freeze),
        halfstep_cases=len(refinement),halfstep_max_mean_gamma_difference=max(abs(r['mean_gamma_difference']) for r in refinement),
        halfstep_max_mse_difference=max(abs(r['heldout_mse_difference']) for r in refinement),
        halfstep_max_frozen_relative_mse_difference=max(abs(r['frozen_relative_mse_difference']) for r in refinement),
        basis_max_effective_force_difference=max(r['effective_force_difference'] for r in basis),
        grid_max_relative_mse_difference=max(abs(r['difference']) for r in grids))
    (output/'numerical_checks.json').write_text(json.dumps(clean(result),indent=2)+'\n')
    print(json.dumps(clean(result)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--primary',type=Path,required=True)
    args=parser.parse_args(); verify(args.root,args.primary)


if __name__=='__main__': main()
