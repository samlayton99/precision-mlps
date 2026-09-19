"""Detached evidence for the paired joint-conditioning campaign; no model updates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

from . import core, diagnostics, difference_analysis as old, difference_training as first
from . import higher_order as higher, joint_conditioning as campaign, ratio_analysis, run

LABELS={'parameter_scale':'Individual scales','parameter_differences':'Individual scales + neighbors'}
COLORS={'parameter_scale':'#2166ac','parameter_differences':'#b2182b'}
CUTOFFS=(1e-10,1e-12,1e-14)


def linearize(g,c,gamma,samples=16):
    x=np.linspace(-1,1,samples*g.n+1);root=np.sqrt(len(x));y=core.target(x,'sine',np)
    a=diagnostics.features(x,g.centers,gamma)/root
    r=a@c-y/root;distance=x[:,None]-g.centers
    e=np.exp(-2*np.abs(distance*gamma))
    j=c[1:]*distance*4*e/(1+e)**2/(g.h*root)
    return x,y,a,r,j


def native_features(a,g,coordinate):
    if coordinate=='parameter_scale':return a*g.alpha
    b=np.column_stack((a[:,0]*g.alpha[0],a[:,1:-1]-a[:,2:],a[:,-1]))
    b[:,1:]*=np.cumsum(g.alpha[1:])
    return b


def probe(g,c,gamma,coordinate,samples=16):
    x,y,a,r,j=linearize(g,c,gamma,samples);root=np.sqrt(len(x))
    spectra={};solutions={};arrays={};stats={}
    for coord in campaign.MAPS:
        b=native_features(a,g,coord)
        u,s,v=svd(b,full_matrices=False,lapack_driver='gesdd')
        arrays[f'singular_readout_{coord}']=s
        arrays[f'residual_modal_{coord}']=u.T@r
        spectra[coord]=(u,s,v)
        stats[coord]=dict(sigma_max=float(s[0]),ranks={str(t):int(np.sum(s>t*s[0])) for t in CUTOFFS},
                          frozen_gd_stability_ceiling=float(2/s[0]**2))
    u,s,v=spectra['parameter_scale'];fits=[]
    for cutoff in CUTOFFS:
        keep=s>cutoff*s[0];retained=u[:,keep]
        cf=g.alpha*(v[keep].T@((retained.T@(y/root))/s[keep]))
        fitted=a@cf-y/root
        xv=diagnostics.midpoint_grid(32768)
        validation=diagnostics.prediction(xv,g.centers,cf,gamma)-core.target(xv,'sine',np)
        gp,gn,leak,closure=old.projected_forces(j,r,retained)
        fits.append(dict(cutoff=cutoff,rank=int(keep.sum()),mse=float(fitted@fitted),
                         midpoint_mse=float(np.mean(validation**2)),coefficient_l1=float(np.abs(cf).sum()),
                         parallel_force=float(np.linalg.norm(gp)),perpendicular_force=float(np.linalg.norm(gn)),
                         projection_leakage=float(leak),gradient_closure=float(closure)))
        if cutoff==1e-12:
            arrays.update(readout_refit=cf,gradient_parallel=gp,gradient_perpendicular=gn,
                          residual_parallel=retained@(retained.T@r),residual_perpendicular=r-retained@(retained.T@r))
    for coord in campaign.MAPS:
        jac=np.column_stack((native_features(a,g,coord),j))
        uj,sj,vj=svd(jac,full_matrices=False,lapack_driver='gesdd')
        arrays[f'singular_joint_{coord}']=sj
        arrays[f'joint_residual_modal_{coord}']=uj.T@r
        rows=[]
        for cutoff in CUTOFFS:
            keep=sj>cutoff*sj[0]
            dz=-vj[keep].T@((uj[:,keep].T@r)/sj[keep])
            dc=higher.readout_map(g,coord)@dz[:g.width+1];dl=dz[g.width+1:]
            motion=jac@dz
            rows.append(dict(cutoff=cutoff,rank=int(keep.sum()),linearized_mse=float(np.sum((r+motion)**2)),
                             delta_c_norm=float(np.linalg.norm(dc)),delta_lambda_norm=float(np.linalg.norm(dl))))
            if cutoff==1e-12:solutions[coord]=(dc,dl,motion)
        stats[coord]['undamped_shadow']=rows
    p0,p1=(solutions[k] for k in campaign.MAPS)
    native_u,native_s,_=spectra[coordinate]
    bounds,rb,_=diagnostics.band_residuals(r)
    regions={}
    for name,mask in g.masks.items():
        direction=np.sign(gamma)*mask;direction/=max(np.linalg.norm(direction),1.)
        regions[name]=dict(lambda_quantiles=np.quantile(np.abs(g.h*gamma[mask]),[0,.1,.5,.9,1]).tolist(),
                          signed_lambda_growth_force=float(-(j.T@r)@direction),
                          parallel_growth_force=float(-arrays['gradient_parallel']@direction),
                          perpendicular_growth_force=float(-arrays['gradient_perpendicular']@direction),
                          weight_over_allowance_rms=float(np.sqrt(np.mean((c[1:][mask]/g.alpha[1:][mask])**2))))
    stats.update(mse=float(r@r),refits=fits,regions=regions,
                 opposite_slope_neighbor_fraction=float(np.mean(gamma[:-1]*gamma[1:]<0)),
                 neighbor_feature_norm_quantiles=np.quantile(np.linalg.norm(a[:,1:-1]-a[:,2:],axis=0),[0,.1,.5,.9,1]).tolist(),
                 undamped_shadow_map_difference=dict(delta_c_norm=float(np.linalg.norm(p0[0]-p1[0])),
                      delta_lambda_norm=float(np.linalg.norm(p0[1]-p1[1])),function_norm=float(np.linalg.norm(p0[2]-p1[2]))),
                 fourier_closure=float(abs(np.sum(rb**2)-r@r)))
    arrays.update(c=c,gamma=gamma,residual=r,band_bounds=bounds,band_mse=np.sum(rb**2,axis=1),
                  band_gradient_c=rb@a,band_gradient_lambda=rb@j,
                  band_gradient_native_readout=rb@native_features(a,g,coordinate),
                  gradient_c=a.T@r,gradient_lambda=j.T@r)
    return stats,arrays


def dense_window(folder,end,optimizer):
    if optimizer in ('gd','adam'):return ratio_analysis.read_dense(folder,end)
    path=folder/f'dense_{end:09d}.npz'
    with np.load(path if path.exists() else folder/'dense_latest.npz') as a:result=dict(a)
    np.testing.assert_array_equal(result['step'],np.arange(max(0,end-2048),end))
    result['gradient_lambda']=result['gradient_native'][:,result['c'].shape[1]:]
    return result


def dense_audit(folder,g,case,end,dest):
    dense=dense_window(folder,end,case['optimizer']);count=len(dense['step'])
    if not count:return {}
    x,y,a,r,j=linearize(g,dense['c'][0],dense['gamma'][0])
    b=native_features(a,g,case['coordinates']);u,s,_=svd(b,full_matrices=False)
    uj,sj,_=svd(np.column_stack((b,j)),full_matrices=False)
    up,sp,_=svd(a*g.alpha,full_matrices=False);retained=up[:,sp>1e-12*sp[0]]
    rng=np.random.default_rng(391)
    boundaries=np.linspace(0,count,min(16,count)+1,dtype=int)
    indices=np.array([rng.integers(lo,hi) for lo,hi in zip(boundaries[:-1],boundaries[1:])])
    root=np.sqrt(len(x));rows=[]
    for index in indices:
        c,gamma,dc,dl=(dense[k][index] for k in ('c','gamma','delta_c','delta_lambda'))
        raw,pieces,budget=ratio_analysis.update_budget(x,y,g.centers,g.h,c,gamma,dc,dl)
        _,_,a,r,j=linearize(g,c,gamma);bounds,rb,_=diagnostics.band_residuals(r)
        gc,gl=rb@a,rb@j;gp,gn,leak,closure=old.projected_forces(j,r,retained)
        rows.append(dict(step=dense['step'][index],**budget,residual_mse=r@r,band_mse=np.sum(rb**2,axis=1),
                         band_readout_linear_mse_change=2*gc@dc,band_geometry_linear_mse_change=2*gl@dl,
                         band_gradient_lambda=gl,band_gradient_c=gc,
                         band_gradient_native_readout=rb@native_features(a,g,case['coordinates']),
                         readout_linear_by_region=np.array([2*gc[:,0].sum()*dc[0]]+
                             [2*gc[:,1:][:,mask].sum(axis=0)@dc[1:][mask] for mask in g.masks.values()]),
                         readout_residual_modal=u.T@r,readout_update_modal=u.T@(pieces[0]/root),
                         joint_residual_modal=uj.T@r,joint_update_modal=uj.T@(pieces.sum(axis=0)/root),
                         readout_update_energy=np.mean(pieces[0]**2),joint_update_energy=np.mean(pieces.sum(axis=0)**2),
                         readout_geometry_cosine=pieces[0]@pieces[1]/max(np.linalg.norm(pieces[0])*np.linalg.norm(pieces[1]),1e-300),
                         gradient_parallel=gp,gradient_perpendicular=gn,projection_leakage=leak,
                         gradient_closure=max(closure,np.linalg.norm(gl.sum(axis=0)-dense['gradient_lambda'][index]))))
    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    run.save_arrays(dest/f'dense_{end}.npz',**arrays,singular_readout=s,singular_joint=sj,band_bounds=bounds)
    run.save_arrays(dest/f'dense_parameters_{end}.npz',**{k:dense[k] for k in ('step','c','gamma')})
    motion={}
    for label,delta in (('w',dense['delta_c'][:,1:]),('lambda',dense['delta_lambda'])):
        norms=np.linalg.norm(delta,axis=1)
        cos=np.sum(delta[:-1]*delta[1:],axis=1)/np.maximum(norms[:-1]*norms[1:],1e-300)
        motion[label]=dict(adjacent_cosine_median=float(np.median(cos)) if len(cos) else None,
                           mean_step_rms=float(np.sqrt(np.mean(delta**2,axis=1)).mean()),
                           net_change_rms=float(np.sqrt(np.mean(delta.sum(axis=0)**2))))
    fractions={}
    for name,spectrum in (('readout',s),('joint',sj)):
        fractions[name]=[dict(threshold=t,
            residual_below=float(np.mean(np.sum(arrays[f'{name}_residual_modal'][:,spectrum<t*spectrum[0]]**2,axis=1))/max(arrays['residual_mse'].mean(),1e-300)),
            update_below=float(np.mean(np.sum(arrays[f'{name}_update_modal'][:,spectrum<t*spectrum[0]]**2,axis=1))/max(arrays[f'{name}_update_energy'].mean(),1e-300)))
            for t in (.1,1e-3,1e-4,1e-6)]
    return dict(end=end,sampled_steps=dense['step'][indices].tolist(),motion=motion,modal_fractions=fractions,
                mean_mse_change=arrays['mse_change'].mean(axis=0).tolist(),
                mean_linear_mse_change=arrays['linear_mse_change'].mean(axis=0).tolist(),
                mean_quadratic_cost=arrays['quadratic_mse_cost'].mean(axis=0).tolist(),
                mean_readout_geometry_cosine=float(arrays['readout_geometry_cosine'].mean()),
                mean_readout_linear_by_region=dict(zip(['bias',*g.masks],arrays['readout_linear_by_region'].mean(axis=0).tolist())),
                maximum_prediction_closure=float(arrays['closure_max'].max()),maximum_gradient_closure=float(arrays['gradient_closure'].max()))


def precision_check(g,cp,coordinate=None):
    import mpmath as mp
    x=diagnostics.midpoint_grid(19);c,gamma=cp['c'],cp['gamma']
    p64=diagnostics.prediction(x,g.centers,c,gamma);y64=core.target(x,'sine',np)
    with mp.workdps(80):
        xx=list(map(mp.mpf,x));cc=list(map(mp.mpf,c));gg=list(map(mp.mpf,gamma));tt=list(map(mp.mpf,g.centers))
        pred=[cc[0]+mp.fsum(w*mp.tanh(s*(z-t)) for w,s,t in zip(cc[1:],gg,tt)) for z in xx]
        yy=[mp.sqrt(2)*mp.sin(2*mp.pi*z) for z in xx]
        rms=mp.sqrt(mp.fsum((p-y)**2 for p,y in zip(pred,yy))/len(xx))
        mismatch=max(abs(mp.mpf(p)-q) for p,q in zip(p64,pred))
        target_error=max(abs(mp.mpf(p)-q) for p,q in zip(y64,yy))
        native_error=None
        if coordinate is not None:
            z=list(map(mp.mpf,cp['z']));alpha=list(map(mp.mpf,g.alpha))
            if coordinate=='parameter_scale':native_c=[u*v for u,v in zip(z,alpha)]
            else:
                # The recorded map uses the FP64 reference cumulative allowances.
                aa=list(map(mp.mpf,np.cumsum(g.alpha[1:])))
                q=[u*v for u,v in zip(z[1:],aa)]
                native_c=[z[0]*alpha[0]]+[q[i]-(q[i-1] if i else 0) for i in range(len(q))]
            native_g=[mp.mpf(v)/mp.mpf(g.h) for v in cp['lambda']]
            native_pred=[native_c[0]+mp.fsum(w*mp.tanh(s*(z-t)) for w,s,t in zip(native_c[1:],native_g,tt)) for z in xx]
            native_error=float(max(abs(a-b) for a,b in zip(pred,native_pred)))
    return dict(points=len(x),digits=80,physical_parameters_held_fixed=True,native_decode_max_difference=native_error,
                rms_fp64=float(np.sqrt(np.mean((p64-y64)**2))),rms_mp80=float(rms),
                prediction_max_difference=float(mismatch),target_max_difference=float(target_error))


def inverse_hessian_direction(matrix,gradient):
    """Audit the stored FP64 metric/gradient; this does not recompute the loss gradient."""
    import mpmath as mp
    direction=-matrix@gradient
    with mp.workdps(80):
        gg=list(map(mp.mpf,gradient))
        slope=-mp.fsum(gi*mp.fsum(mp.mpf(v)*gj for v,gj in zip(row,gg)) for gi,row in zip(gg,matrix))
    return dict(stored_metric_directional_derivative_fp64=float(gradient@direction),
                stored_metric_directional_derivative_mp80=float(slope),
                direction_norm=float(np.linalg.norm(direction)),gradient_norm=float(np.linalg.norm(gradient)),
                cosine=float(gradient@direction/max(np.linalg.norm(gradient)*np.linalg.norm(direction),1e-300)))


def analyze_case(task):
    root,output,case,*options=task;key=first.case_key(case);folder=root/key;dest=output/key;dest.mkdir(parents=True,exist_ok=True)
    source_status=json.loads((folder/'latest.json').read_text());status=dict(source_status)
    end=status['completed_updates'];g=core.geometry(case['n'])
    if options and options[0]:
        limit=100000 if case['optimizer'] in ('gd','adam') else 20000
        end=min(end,limit)
        if end<status['completed_updates']:
            with np.load(folder/f'checkpoint_{end:09d}.npz') as cp:
                status.update(step=end,completed_updates=end,status='continuing',train_mse=float(cp['train_mse']),validation_mse=float(cp['validation_mse']))
            if 'failed_update' in status:status['failed_update']=None
    if not end:return dict(case=case,key=key,status=status,end=0)
    trace=campaign.read_trace(folder,end,case['optimizer'])
    if case.get('restart'):
        run.save_arrays(dest/'restart_trace.npz',trace=trace,columns=campaign.HIGHER_COLUMNS)
    if end<source_status['completed_updates'] and case['optimizer'] in ('gn','ssbroyden'):
        for counter in ('function_evaluations','gradient_evaluations','jacobian_evaluations'):
            status[counter]=int(trace[-1,campaign.HIGHER_COLUMNS.index(counter)])
        status['training_seconds']=float(trace[-1,campaign.HIGHER_COLUMNS.index('elapsed_seconds')])
    steps=[];history=[]
    for path in sorted(folder.glob('checkpoint_*.npz')):
        step=int(path.stem.split('_')[-1])
        if step>end:continue
        with np.load(path) as cp:
            c,gamma=cp['c'],cp['gamma'];x,y,a,r,j=linearize(g,c,gamma)
            bounds,rb,_=diagnostics.band_residuals(r)
            history.append(dict(c=c,gamma=gamma,train_mse=float(cp['train_mse']),validation_mse=float(cp['validation_mse']),
                                band_mse=np.sum(rb**2,axis=1),lambda_quantiles=np.quantile(np.abs(g.h*gamma[g.core]),[.1,.5,.9]),
                                opposite_slope_neighbor_fraction=np.mean(gamma[:-1]*gamma[1:]<0)))
            steps.append(step)
    run.save_arrays(dest/'history.npz',step=steps,centers=g.centers,alpha=g.alpha,band_bounds=bounds,
                    **{k:np.stack([r[k] for r in history]) for k in history[0]})
    block=min(20000,max(1,end//20));starts=np.arange(0,end,block)
    run.save_arrays(dest/'window_mse.npz',step=np.minimum(starts+block,end),
                    mean=[2*trace[i:min(i+block,end),0].mean() for i in starts],
                    quantiles=np.array([np.quantile(2*trace[i:min(i+block,end),0],[0,.1,.5,.9,1]) for i in starts]).T)
    record=dict(case=case,key=key,end=end,status=status,source_latest=source_status,checkpoints={},source_hashes={},dense=[])
    probes=sorted({0,*[s for s in (100,1000,20000,100000) if s<=end],end})
    for step in probes:
        path=folder/f'checkpoint_{step:09d}.npz'
        if not path.exists():continue
        with np.load(path) as cp:stats,arrays=probe(g,cp['c'],cp['gamma'],case['coordinates'])
        record['checkpoints'][str(step)]=stats;record['source_hashes'][str(step)]=hashlib.sha256(path.read_bytes()).hexdigest()
        run.save_arrays(dest/f'spectrum_{step}.npz',**arrays)
    for de in sorted({min(2048,end),end}):
        record['dense'].append(dense_audit(folder,g,case,de,dest))
    with np.load(folder/f'checkpoint_{end:09d}.npz') as cp:
        record['precision']=precision_check(g,cp,case['coordinates'])
        x=np.linspace(-1,1,32*g.n+1)
        rr=diagnostics.prediction(x,g.centers,cp['c'],cp['gamma'])-core.target(x,'sine',np)
        record['doubled_grid_mse']=float(np.mean(rr**2))
    last=min(end,20000 if case['optimizer'] in ('gd','adam') else 4000)
    window=2*trace[-last:,0]
    record.update(late_window_updates=last,late_mean_mse=float(window.mean()),late_median_mse=float(np.median(window)),
                  late_max_mse=float(window.max()),minimum_observed_mse=float(2*trace[:,0].min()))
    if case['optimizer'] in ('gn','ssbroyden'):
        columns={name:trace[:,i] for i,name in enumerate(campaign.HIGHER_COLUMNS)}
        record['numerics']=dict(guard_active_updates=int(columns['guard_active'].sum()),
             attempts_quantiles=np.quantile(columns['attempts'],[0,.5,.9,1]).tolist(),
             curvature_quantiles=np.quantile(columns['curvature'],[0,.1,.5,.9,1]).tolist(),
             final_damping=float(columns['damping'][-1]),maximum_linear_residual=float(columns['linear_residual'].max()))
        if case['optimizer']=='ssbroyden':
            leaves,_=run.load_state(folder/f'state_{end:09d}.pkl')
            matrices=[np.asarray(a) for a in leaves if a.ndim==2 and a.shape==(2*g.width+1,2*g.width+1)]
            if len(matrices)!=1:raise ValueError('Expected exactly one full inverse-Hessian array')
            matrix=matrices[0];eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2)
            run.save_arrays(dest/'inverse_hessian_spectrum.npz',eigenvalues=eigenvalues)
            record['numerics'].update(inverse_hessian_eigenvalue_quantiles=np.quantile(eigenvalues,[0,.1,.5,.9,1]).tolist(),
                 inverse_hessian_asymmetry=float(np.linalg.norm(matrix-matrix.T)/max(np.linalg.norm(matrix),1e-300)))
    elif case['optimizer']=='adam':
        with np.load(folder/f'checkpoint_{end:09d}.npz') as cp:
            record['numerics']={f'{block}_epsilon_dominated_fraction':float(np.mean(cp[f'adam_{block}_sqrt_v_over_epsilon']<1)) for block in ('readout','slope')}
    if end==source_status['completed_updates'] and (folder/'failure.npz').exists():
        with np.load(folder/'failure.npz') as a:
            record['failure']={k:float(a[k]) if np.isfinite(a[k]) else None for k in ('attempts','trial_mse','actual','predicted','ratio','curvature','step_size')}
            if case['optimizer']=='ssbroyden' and np.all(np.isfinite(matrix)) and np.all(np.isfinite(a['gradient'])):
                record['failure'].update(inverse_hessian_direction(matrix,a['gradient']))
    run.write_json(dest/'mechanism.json',record)
    print(json.dumps(dict(analyzed=key,end=end,status=status['status'])),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--minimum-horizon',action='store_true',help='Analyze the fixed 100k/20k comparison even when training has continued')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=json.loads(args.cases.read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(analyze_case,[(args.root,args.output,c,args.minimum_horizon) for c in cases]))
    run.write_json(args.output/'summary.json',records)


if __name__=='__main__':main()
