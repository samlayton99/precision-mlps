"""Detached CPU dictionary screen for the full campaign; preserves the probe."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.linalg import svd, svdvals
from . import core, full_core as f


def dictionary_id(n, name, gamma):
    return f'N{n}_{name}_g{gamma:g}'


def dictionary_matrix(cfg):
    cases=[(cfg['n'],name,gamma) for name in cfg['maps'] for gamma in cfg['gammas']]
    cases += [(cfg['n'],name,gamma) for name in cfg['coordinate_controls'] for gamma in cfg['robust_gammas']]
    cases += [(n,name,gamma) for n in cfg['widths'] if n != cfg['n']
              for name in cfg['width_maps'] for gamma in [1,4,n/8]]
    return cases


def common(root,n,cfg):
    folder=root/f'common/N{n}'; folder.mkdir(parents=True,exist_ok=True)
    g=core.geometry(n); x=np.linspace(-1,1,cfg['samples_per_cell']*n+1)
    grids=dict(train=x,validation=-1+2*(np.arange(cfg['n_validation'])+.37)/cfg['n_validation'],
               evaluation=core.grid(cfg['n_eval']))
    arrays=dict(centers=g.centers,alpha=g.alpha)
    for label,grid in grids.items():
        arrays['x_'+label]=grid
        arrays['y_'+label]=np.column_stack([f.target(grid,t) for t in cfg['targets']])/np.sqrt(len(grid))
    degree=min(cfg['k_max'],len(x)-2)
    qr=core.polynomial_transform(x,degree)
    arrays['y_hat']=core.transform(qr,arrays['y_train'])
    q=core.discrete_polynomials(x,degree)
    orth=float(np.linalg.norm(q.T@q-np.eye(degree+1),2))
    coefficient=float(np.max(np.abs(np.abs(q.T@arrays['y_train'])-np.abs(arrays['y_hat'][:degree+1]))))
    core.save_arrays(folder/'arrays.npz',**arrays)
    core.write_json(folder/'meta.json',dict(n=n,width=g.width,halo=g.radius,k_max=degree,
        grid_hash=core.array_hash(x),target_hash=core.array_hash(arrays['y_train']),
        geometry_hash=core.array_hash(g.centers),orthogonality=orth,coefficient_abs_difference=coefficient,
        target_norms={label:np.linalg.norm(arrays['y_'+label],axis=0).tolist() for label in grids}))
    return g,arrays,qr


def coefficients_norms(theta,r):
    return {label:{norm:np.linalg.norm(value,ord=order,axis=0).tolist()
        for norm,order in [('l1',1),('l2',2),('linf',np.inf)]}
        for label,value in [('native',theta),('physical',r@theta)]}


def scan_dictionary(root,cfg,n,name,gamma,g,arrays,qr,deadline):
    tag=dictionary_id(n,name,gamma); folder=root/'dictionaries'/tag
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'meta.json').exists():
        previous=json.loads((folder/'meta.json').read_text())
        if previous.get('complete'):
            return previous
    start=time.monotonic(); scale,neighbor=f.map_spec(g,name); r=f.map_matrix(g,name)
    x=arrays['x_train']; y=arrays['y_train']; yh=arrays['y_hat']
    a=core.design(x,g.centers,gamma); j=f.design_from_physical(a,scale,neighbor)
    jh=core.transform(qr,j)
    u,s,vh=svd(j,full_matrices=False,lapack_driver='gesdd'); curvature=float(s[0]**2)
    norm_y=np.linalg.norm(y,axis=0)
    degrees=np.arange(cfg['k_max']+1)
    measured_e,mu,frobenius=core.access(yh,jh,cfg['k_max'])
    e=measured_e.copy(); e[2:,cfg['targets'].index('quadratic')]=0.
    envelopes=f.envelopes(gamma,degrees,r)
    noise=(64*np.finfo(float).eps*np.linalg.norm(j)/np.maximum(measured_e,np.finfo(float).tiny))**2
    selected=set(range(0,cfg['k_max']+1,16)); certificates=[]
    for ti,target in enumerate(cfg['targets']):
        for epsilon in cfg['tolerances']:
            with np.errstate(divide='ignore'):
                choices=dict(analytic=envelopes['used'],cap=envelopes['cap'],
                             abbreviated=envelopes['abbreviated'],directional=np.log(mu[:,ti]))
            for kind,denominator in choices.items():
                value=f.bound(e[:,ti],denominator,epsilon,curvature)
                k=value['k']
                value.update(target=target,epsilon=epsilon,kind=kind,chi=.5,L=curvature)
                if k is not None:
                    value.update(E=float(e[k,ti]),mu=float(mu[k,ti]),log_B=float(envelopes['used'][k]),
                        resolution='fp64_estimate' if mu[k,ti]>noise[k,ti] else 'unresolved',
                        access_noise=float(noise[k,ti]))
                    if epsilon==.01:
                        selected.add(k)
                certificates.append(value)
    b=np.full(len(degrees),np.nan)
    for degree in sorted(selected):
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU screen deadline reached')
        b[degree]=svdvals(jh[degree+1:])[0]**2
    for ti,target in enumerate(cfg['targets']):
        for epsilon in cfg['tolerances']:
            with np.errstate(divide='ignore'):
                denominator=np.where(np.isfinite(b),np.log(b),np.inf)
            value=f.bound(e[:,ti],denominator,epsilon,curvature)
            value.update(target=target,epsilon=epsilon,kind='subspace_sampled',chi=.5,L=curvature)
            certificates.append(value)
    capacity=[]
    for cutoff in cfg['cutoffs']:
        keep=s>cutoff*s[0]; uk=u[:,keep]; sk=s[keep]; vk=vh[keep]
        loading=uk.T@y; perpendicular=y-uk@loading
        floor=np.sum(perpendicular**2,axis=0); theta=vk.T@(loading/sk[:,None])
        direct=np.linalg.norm(j@theta-y,axis=0)/norm_y
        norms=coefficients_norms(theta,r)
        xe=arrays['x_evaluation']
        je=f.design_from_physical(core.design(xe,g.centers,gamma),scale,neighbor)
        evaluation=np.linalg.norm(je@theta-arrays['y_evaluation'],axis=0)/np.linalg.norm(arrays['y_evaluation'],axis=0)
        for ti,target in enumerate(cfg['targets']):
            capacity.append(dict(target=target,cutoff=cutoff,retained_rank=int(keep.sum()),
                projection_error=float(np.sqrt(floor[ti])/norm_y[ti]),train_refit=float(direct[ti]),
                eval_refit=float(evaluation[ti]),
                norms={label:{key:values[ti] for key,values in values.items()} for label,values in norms.items()},
                predictions=[dict(epsilon=eps,**core.spectral_hit(sk,loading[:,ti],floor[ti],norm_y[ti],.5/curvature,eps))
                             for eps in cfg['tolerances']]))
        if cutoff==min(cfg['cutoffs']):
            np.save(folder/'U.npy',uk)
            core.save_arrays(folder/'spectrum.npz',singular=sk,Vh=vk,loadings=loading,floor_sq=floor,
                norm_y=norm_y,refit_theta=theta,L=curvature)
    core.save_arrays(folder/'access.npz',E=e,E_measured=measured_e,mu=mu,b=b,frobenius=frobenius,
        noise=noise,**{f'log_B_{key}':value for key,value in envelopes.items()})
    np.save(folder/'J.npy',j); np.save(folder/'QJ.npy',jh)
    core.write_json(folder/'certificates.json',certificates); core.write_json(folder/'capacity.json',capacity)
    boundary=[c for c in certificates if c['k']==cfg['k_max'] and c.get('E',0)>c['epsilon']
              and c.get('resolution')=='fp64_estimate']
    if boundary and cfg['k_max']<cfg['k_max_extension']:
        expanded=dict(cfg,k_max=cfg['k_max_extension'])
        expanded_qr=core.polynomial_transform(x,expanded['k_max'])
        expanded_arrays=dict(arrays,y_hat=core.transform(expanded_qr,y))
        return scan_dictionary(root,expanded,n,name,gamma,g,expanded_arrays,expanded_qr,deadline)
    meta=dict(dictionary_id=tag,n=n,width=g.width,halo=g.radius,map=name,gamma=gamma,
        lambda_value=gamma*g.h,neighbor=neighbor,scales=scale.tolist(),L=curvature,
        matrix_hash=core.array_hash(j),map_hash=core.array_hash(r),geometry_hash=core.array_hash(g.centers),
        source_commit=os.environ.get('PROBE_SOURCE_COMMIT','local'),seconds=time.monotonic()-start,
        k_max=cfg['k_max'],boundary_witnesses=boundary,
        complete=True)
    core.write_json(folder/'meta.json',meta)
    print(f'SCREEN {tag} seconds={meta["seconds"]:.2f} L={curvature:.6g}',flush=True)
    return meta


def run(root,cfg,seconds):
    root.mkdir(parents=True,exist_ok=True); deadline=time.monotonic()+seconds
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text()); assert manifest['config']==cfg,'Campaign configuration changed'
    else:
        manifest=dict(config=cfg,source_commit=os.environ.get('PROBE_SOURCE_COMMIT','local'),
            versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','jax','optax','mpmath']},
            dictionaries=[],complete=False)
        core.write_json(manifest_path,manifest)
    cases=dictionary_matrix(cfg)
    for n in sorted(set(row[0] for row in cases),key=lambda value:(value!=cfg['n'],value)):
        g,arrays,qr=common(root,n,cfg)
        for _,name,gamma in [row for row in cases if row[0]==n]:
            if time.monotonic()>=deadline:
                return
            meta=scan_dictionary(root,cfg,n,name,gamma,g,arrays,qr,deadline)
            if meta['dictionary_id'] not in manifest['dictionaries']:
                manifest['dictionaries'].append(meta['dictionary_id'])
            core.write_json(manifest_path,manifest)
    manifest['complete']=True
    core.write_json(manifest_path,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=core.HERE/'full_config.yaml')
    parser.add_argument('--seconds',type=float,default=1500)
    args=parser.parse_args()
    run(args.root,core.config(args.config),args.seconds)


if __name__=='__main__':
    main()
