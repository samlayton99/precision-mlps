"""Read-only optimizer-state and precision evidence for the Newton handoffs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh, solve

from . import core, diagnostics, higher_order as higher, full_newton as newton
from . import joint_analysis as analysis, joint_conditioning as campaign, joint_mechanism_probes as probes
from . import difference_training as first, newton_handoffs as protocol, run


def metric_blocks(matrix,gradient,jac,scales,reference_scales,split):
    """Attribute native inverse-Hessian action, then compare in physical/function units."""
    parts=np.zeros((4,len(gradient)))
    parts[0,:split]=-matrix[:split,:split]@gradient[:split]
    parts[1,:split]=-matrix[:split,split:]@gradient[split:]
    parts[2,split:]=-matrix[split:,:split]@gradient[:split]
    parts[3,split:]=-matrix[split:,split:]@gradient[split:]
    direction=parts.sum(axis=0);physical_gradient=gradient/scales
    directions=np.stack((direction,-physical_gradient/scales,-reference_scales**2*physical_gradient/scales))
    function=parts@jac.T;combined=direction@jac.T
    readout=function[:2].sum(axis=0);geometry=function[2:].sum(axis=0)
    denom=np.linalg.norm(readout)*np.linalg.norm(geometry)
    relative=scales/reference_scales
    common=relative[:,None]*matrix*relative[None,:]
    eig=np.linalg.eigvalsh((common+common.T)/2)
    return dict(component_order=['readout_from_readout','readout_from_geometry','geometry_from_readout','geometry_from_geometry'],
                physical_direction_rms=np.sqrt(np.mean((parts*scales)**2,axis=1)).tolist(),
                function_direction_norm=np.linalg.norm(function,axis=1).tolist(),
                directional_derivatives=(parts@gradient).tolist(),
                readout_geometry_function_cosine=float(readout@geometry/denom) if denom else None,
                direction_closure=float(np.linalg.norm(direction+matrix@gradient)),
                function_closure=float(np.linalg.norm(function.sum(axis=0)-combined)),
                comparison_order=['learned','physical_identity','prescribed_initial'],
                comparison_function_norm=np.linalg.norm(directions@jac.T,axis=1).tolist(),
                comparison_directional_derivative=(directions@gradient).tolist(),
                common_metric_eigen_quantiles=np.quantile(eig,[0,.1,.5,.9,1]).tolist(),
                common_metric_asymmetry=float(np.linalg.norm(common-common.T)/max(np.linalg.norm(common),1e-300)))


def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def extra_case(task):
    root,output,case=task;key=first.case_key(case);folder=root/key;dest=output/key;dest.mkdir(exist_ok=True)
    latest=json.loads((folder/'latest.json').read_text());end=latest['completed_updates']
    g=core.geometry(case['n']);split=g.width+1;coordinate=case['coordinates']
    scales=np.r_[np.diag(higher.readout_map(g,coordinate)),np.full(g.width,1. if coordinate=='physical' else 1/g.h)]
    reference=np.r_[g.alpha,np.full(g.width,1/g.h)]
    rows=[]
    for step in sorted({0,1,10,100,1000,2000,5000,10000,20000,end}):
        cp_path=folder/f'checkpoint_{step:09d}.npz';state_path=folder/f'state_{step:09d}.pkl'
        if step>end or not cp_path.exists() or not state_path.exists():continue
        with np.load(cp_path) as cp:c,gamma=cp['c'],cp['gamma']
        x,y,a,r,j=analysis.linearize(g,c,gamma)
        jac=np.column_stack((analysis.native_features(a,g,coordinate),j*(g.h if coordinate=='physical' else 1.)))
        gradient=jac.T@r
        row=dict(step=step,checkpoint_sha256=hashlib.sha256(cp_path.read_bytes()).hexdigest(),
                 state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                 gradient_physical_readout_norm=float(np.linalg.norm(gradient[:split]/scales[:split])),
                 gradient_physical_gamma_norm=float(np.linalg.norm(gradient[split:]/scales[split:])),
                 gradient_reference_readout_norm=float(np.linalg.norm(gradient[:split]/scales[:split]*reference[:split])),
                 gradient_lambda_norm=float(np.linalg.norm(j.T@r)))
        if case['optimizer']=='ssbroyden':
            leaves,_=run.load_state(state_path)
            matrices=[v for v in leaves if np.shape(v)==(len(gradient),len(gradient))]
            if len(matrices)!=1:raise ValueError('Ambiguous inverse-Hessian checkpoint')
            row['metric']=metric_blocks(np.asarray(matrices[0]),gradient,jac,scales,reference,split)
        if case['optimizer']=='newton' and step in (0,1000,end):
            h,gn,gradient=probes.native_hessians(g,c,gamma,coordinate)
            values,vectors=eigh(h);gn_values,gn_vectors=eigh(gn)
            row['curvature']=dict(minimum=float(values[0]),maximum=float(values[-1]),
                negative_eigenvalues=int(np.sum(values<0)),residual_curvature_norm=float(np.linalg.norm(h-gn)),
                gn_norm=float(np.linalg.norm(gn)),eigen_backward_error=float(np.linalg.norm(h@vectors-vectors*values)/max(np.linalg.norm(h),1e-300)))
            # Use the recorded next radius. The freshly initialized template
            # reconstructs the optimizer state's static tree structure.
            z=higher.encode_physical(c,gamma,g,coordinate)
            state,_=higher.load_state(state_path,newton.initial(z));radius=float(state['radius'])
            trials=[]
            for label,spectrum,basis in [('full_hessian',values,vectors),('gauss_newton',gn_values,gn_vectors)]:
                delta,shift,hard=newton.eigen_step(jnp.asarray(spectrum),jnp.asarray(basis),jnp.asarray(gradient),radius)
                delta=np.asarray(delta);dc=scales[:split]*delta[:split];dg=scales[split:]*delta[split:]
                trial=(diagnostics.prediction(x,g.centers,c+dc,gamma+dg)-y)/np.sqrt(len(x))
                used=h if label=='full_hessian' else gn
                trials.append(dict(model=label,radius=radius,shift=float(shift),hard_case=bool(hard),
                    predicted_reduction=float(-gradient@delta-.5*delta@(used@delta)),actual_mse=float(trial@trial),
                    readout_step_norm=float(np.linalg.norm(dc)),lambda_step_norm=float(np.linalg.norm(g.h*dg))))
            row['same_radius_trials']=trials
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');delta=solve(h,-gradient,assume_a='sym')
                trial=(diagnostics.prediction(x,g.centers,c+scales[:split]*delta[:split],gamma+scales[split:]*delta[split:])-y)/np.sqrt(len(x))
                denom=np.linalg.norm(h)*np.linalg.norm(delta)+np.linalg.norm(gradient)
                row['undamped']=dict(warnings=[str(w.message) for w in caught],directional_derivative=float(gradient@delta),
                    actual_mse=float(trial@trial),step_norm=float(np.linalg.norm(delta)),
                    backward_error=float(np.linalg.norm(h@delta+gradient)/max(denom,1e-300)))
            except np.linalg.LinAlgError as error:row['undamped']=dict(failure=str(error))
        rows.append(row)
    trace=campaign.read_trace(folder,end,case['optimizer']) if end else np.empty((0,len(campaign.HIGHER_COLUMNS)))
    details=[]
    for path in sorted(folder.glob('details_*.npz')):
        with np.load(path) as a:details.append(a['trace'])
    detail=np.concatenate(details) if details else np.empty((0,len(campaign.DETAIL_COLUMNS)))
    if len(detail)!=end:raise ValueError(f'Incomplete diagnostic trace at {key}')
    run.save_arrays(dest/'optimizer_trace.npz',trace=trace,columns=campaign.HIGHER_COLUMNS,
                    details=detail,detail_columns=campaign.DETAIL_COLUMNS)
    result=dict(key=key,case=case,end=end,states=rows)
    if case['optimizer']=='newton' and end:
        shift=trace[:,campaign.HIGHER_COLUMNS.index('damping')]
        result['unshifted_fraction']=float(np.mean(shift==0))
        result['maximum_solve_backward_error']=float(np.max(trace[:,campaign.HIGHER_COLUMNS.index('linear_residual')]))
        result['shift_quantiles']=np.quantile(shift,[0,.1,.5,.9,1]).tolist()
        result['radius_quantiles']=np.quantile(detail[:,campaign.DETAIL_COLUMNS.index('radius')],[0,.1,.5,.9,1]).tolist()
    result=finite(result);run.write_json(dest/'optimizer_mechanism.json',result)
    return result


def adam_control(root,output,seed):
    case=protocol.adam_case(seed);key=first.case_key(case);folder=root/'adam_continuation'/key
    if not (folder/'latest.json').exists():return None
    end=json.loads((folder/'latest.json').read_text())['step'];traces=[];history=[]
    for path in sorted(folder.glob('trace_*.npz')):
        with np.load(path) as a:traces.append(a['trace'])
    g=core.geometry(case['n'])
    for path in sorted(folder.glob('checkpoint_*.npz')):
        with np.load(path) as a:
            history.append([int(path.stem.split('_')[-1])-5300000,float(a['train_mse']),float(a['validation_mse']),
                            *np.quantile(np.abs(a['lambda'][g.core]),[.1,.5,.9])])
    trace=np.concatenate(traces) if traces else np.empty((0,len(first.TRACE_COLUMNS)))
    run.save_arrays(output/f'adam_seed_{seed}.npz',history=history,trace=trace,columns=first.TRACE_COLUMNS)
    row=dict(seed=seed,start=5300000,end=end,completed_additional=end-5300000)
    if len(trace):row.update(mean_mse=float(2*trace[-20000:,0].mean()),median_mse=float(np.median(2*trace[-20000:,0])),max_mse=float(2*trace[-20000:,0].max()))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--extra-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=json.loads((args.root/'cases.json').read_text())
    cases=[c for c in cases if (args.root/first.case_key(c)/'latest.json').exists()]
    tasks=[(args.root,args.output,c) for c in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if not args.extra_only:
            records=list(pool.map(analysis.analyze_case,tasks));run.write_json(args.output/'summary.json',records)
        extras=list(pool.map(extra_case,tasks));run.write_json(args.output/'optimizer_summary.json',extras)
    controls=[adam_control(args.root,args.output,s) for s in (0,1)]
    run.write_json(args.output/'adam_controls.json',controls)


if __name__=='__main__':main()
