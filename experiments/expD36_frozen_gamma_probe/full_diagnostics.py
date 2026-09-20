"""Detached residual certificates, norm requirements, and damping measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import svd
from . import core, full_core as f, full_screen as screen, full_train as train


def tail_measurements(jh,rh,uh,singular,eta,kmax):
    tail_sq=np.cumsum((rh*rh)[::-1])[::-1][1:kmax+2]
    gradients=np.cumsum((jh*rh[:,None])[::-1],axis=0)[::-1][1:kmax+2]
    projections=np.cumsum((uh*rh[:,None])[::-1],axis=0)[::-1][1:kmax+2]
    norm=np.linalg.norm(rh)
    with np.errstate(divide='ignore',invalid='ignore'):
        mu=np.sum(gradients**2,axis=1)/tail_sq
        effective=np.sum(projections**2*(-np.log1p(-eta*singular**2)),axis=1)/tail_sq
        unit_gradients=gradients/np.sqrt(tail_sq[:,None])
    return np.sqrt(tail_sq)/norm,mu,effective,unit_gradients


def case_certificates(root,folder,cfg,deadline):
    case=json.loads((folder/'case.json').read_text())
    if case['optimizer']!='gd':
        return []
    final=dict(np.load(folder/'state.npz')); initial=np.load(folder/'initial.npz')['theta']
    assert int(final['count'])==cfg['training_steps'],folder
    arrays,banks=train.load_banks(root,case['n'],case['map'],case['gammas'])
    y=train.column_targets(arrays,case['columns'],'train'); ynorm=np.linalg.norm(y,axis=0)
    out=[]; destination=root/'diagnostics'/case['tag']; destination.mkdir(parents=True,exist_ok=True)
    for bi,(gamma,bank) in enumerate(zip(case['gammas'],banks)):
        if time.monotonic()>deadline:
            raise TimeoutError('Residual certificate deadline')
        meta=bank['meta']; kmax=meta['k_max']; qr=core.polynomial_transform(arrays['x_train'],kmax)
        jh=np.load(root/'dictionaries'/meta['dictionary_id']/'QJ.npy')
        uh=core.transform(qr,bank['U']); singular=bank['spectrum']['singular']
        residual=bank['J']@initial[bi]-y; rh=core.transform(qr,residual)
        rnorm=np.linalg.norm(residual,axis=0)
        envelopes=f.envelopes(gamma,np.arange(kmax+1),f.map_matrix(core.geometry(case['n']),case['map']))
        deltas=[]; mus=[]; effective_values=[]
        for ci,column in enumerate(case['columns']):
            eta=case['rates'][bi][ci]; chi=eta*meta['L']
            delta,mu,effective,gradient=tail_measurements(jh,rh[:,ci],uh,singular,eta,kmax)
            if np.count_nonzero(initial[bi,:,ci])==0:
                degree=column.get('degree',2 if column['target']=='quadratic' else None)
                if degree is not None:
                    delta[degree:]=0.
            deltas.append(delta); mus.append(mu); effective_values.append(effective)
            for ei,epsilon in enumerate(cfg['tolerances']):
                epsilon_res=float(epsilon*ynorm[ci]/rnorm[ci])
                hit=int(final['hits'][bi,ci,ei]); failed=int(final['failed'][bi,ci])
                common=dict(run=case['tag'],gamma=gamma,column=ci,target=column['target'],
                    initialization=column['initialization'],seed=column.get('seed'),
                    epsilon_target=epsilon,epsilon_residual=epsilon_res,
                    initial_target_relative_error=float(rnorm[ci]/ynorm[ci]),
                    initial_function_relative_norm=float(np.linalg.norm(bank['J']@initial[bi,:,ci])/ynorm[ci]),
                    eta=eta,chi=chi,L=meta['L'],executed_steps=int(final['count']),
                    first_hit=hit if hit>=0 else None,
                    hit_status='nonfinite_failure' if failed else ('reached_exact_step' if hit>=0 else 'budget_censored'))
                with np.errstate(divide='ignore',invalid='ignore'):
                    denominators=dict(analytic=envelopes['used'],cap=envelopes['cap'],
                                      directional=np.log(mu),effective_generator=np.log(effective))
                for kind,logden in denominators.items():
                    # C3 is the unit-time auxiliary gradient flow, so L/log(2)=1.
                    value=(f.bound(delta,logden,epsilon_res,np.log(2.)) if kind=='effective_generator'
                           else f.bound(delta,logden,epsilon_res,meta['L'],chi))
                    k=value['k']; resolved=None
                    if k is not None:
                        noise=(64*np.finfo(float).eps*np.linalg.norm(bank['J'])/
                               max(delta[k],np.finfo(float).tiny))**2
                        resolved=bool(mu[k]>noise)
                    out.append(dict(common,kind=kind,**value,access_resolved=resolved,
                        spectral_model='cutoff_1e-14' if kind=='effective_generator' else None))
                if 'degree' in column:
                    mu0=float(np.linalg.norm(bank['J'].T@residual[:,ci])**2/rnorm[ci]**2)
                    pure=0. if epsilon_res>=1 else float(np.ceil(np.log(1/epsilon_res)/(-np.log1p(-eta*mu0))))
                    out.append(dict(common,kind='polynomial_mean_access',bound=pure,k=column['degree']-1,
                                    mu=mu0,status='fp64_estimate'))
                if column['initialization']=='zero' and epsilon_res<1:
                    numerator=np.maximum(delta-epsilon_res,0)*rnorm[ci]
                    requirements={}
                    # R^T g_physical=g_native; the neighboring inverse is a suffix sum.
                    scale,neighbor=f.map_spec(core.geometry(case['n']),case['map'])
                    physical_gradient=gradient/scale
                    if neighbor:
                        physical_gradient[:,1:]=np.cumsum(physical_gradient[:,1:][:,::-1],axis=1)[:,::-1]
                    for basis,grad in [('native',gradient),('physical',physical_gradient)]:
                        requirements[basis]={}
                        for name,dual in [('l1',np.inf),('l2',2),('linf',1)]:
                            denominator=np.linalg.norm(grad,ord=dual,axis=1)
                            ratio=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
                            requirements[basis][name]=float(np.max(ratio))
                    out.append(dict(common,kind='coefficient_norm_requirement',requirements=requirements,
                                    status='fp64_estimate'))
        core.save_arrays(destination/f'gamma_{gamma:g}.npz',delta=np.array(deltas),mu=np.array(mus),
            effective=np.array(effective_values),initial_residual_norm=rnorm,target_norm=ynorm)
    core.write_json(destination/'certificates.json',train.safe_json(out))
    return out


def damping(root,cfg,deadline):
    n=cfg['n']; arrays=dict(np.load(root/f'common/N{n}/arrays.npz')); y=arrays['y_train']
    raw4=root/'dictionaries'/screen.dictionary_id(n,'raw',4)
    primary=json.loads((raw4/'certificates.json').read_text())
    baseline=root/'training'/f'N{n}_raw_gd'
    checkpoint=np.load(baseline/'checkpoint_020000.npz')['theta'][cfg['gammas'].index(16)]
    reference_j=np.load(root/'dictionaries'/screen.dictionary_id(n,'raw',16)/'J.npy')
    shared_residual=reference_j@checkpoint-y
    rows=[]; ridges=[]
    for name in cfg['width_maps']:
        for gamma in cfg['robust_gammas']:
            if time.monotonic()>deadline:
                raise TimeoutError('Damping deadline')
            folder=root/'dictionaries'/screen.dictionary_id(n,name,gamma)
            j=np.load(folder/'J.npy'); meta=json.loads((folder/'meta.json').read_text())
            u,s,vh=svd(j,full_matrices=False); L=s[0]**2
            r=f.map_matrix(core.geometry(n),name)
            for target in cfg['robust_targets']:
                ti=cfg['targets'].index(target)
                witness=next(c for c in primary if c['target']==target and c['epsilon']==.01 and c['kind']=='directional')
                k=witness['k']; qr=core.polynomial_transform(arrays['x_train'],k)
                transformed=core.transform(qr,y[:,ti,None]); transformed[:k+1]=0
                tail=core.transform(qr,transformed,transpose=False)[:,0]
                for label,vector in [('full_target',y[:,ti]),('common_tail',tail),('shared_GD20k_residual',shared_residual[:,ti])]:
                    q=vector/np.linalg.norm(vector); loading=u.T@q; perpendicular=q-u@loading
                    mu=float(np.linalg.norm(j.T@q)**2)
                    logB=f.envelopes(gamma,np.array([k]),r)['used'][0] if label=='common_tail' else None
                    for exponent in cfg['damping_log10_relative']:
                        rho=10.**exponent; penalty=rho*L
                        step,direct,lower=f.damping(j,u,s,vh,q,penalty)
                        stable=perpendicular+u@(penalty/(s*s+penalty)*loading)
                        discrepancy=float(np.linalg.norm(direct-stable))
                        remaining=float(np.linalg.norm(stable))
                        row=dict(map=name,gamma=gamma,target=target,probe=label,k=k if label=='common_tail' else None,
                            rho=rho,penalty=float(penalty),mu=mu,L=float(L),remaining=remaining,
                            direct_remaining=float(np.linalg.norm(direct)),direct_spectral_difference=discrepancy,
                            native_step_norm=float(np.linalg.norm(step)),physical_step_norm=float(np.linalg.norm(r@step)),
                            measured_lower=lower,analytic_lower=float(1/(1+np.exp(logB-np.log(penalty)))) if logB is not None else None,
                            status='fp64_estimate' if discrepancy<=1e-8*max(remaining,1e-12) else 'numerically_unresolved')
                        rows.append(row)
                        if label=='full_target':
                            norm=np.linalg.norm(vector)
                            ridges.append(dict(map=name,gamma=gamma,target=target,rho=rho,error=remaining,
                                native_norm=row['native_step_norm']*norm,physical_norm=row['physical_step_norm']*norm,
                                status=row['status']))
            print(f'DAMPING {name} gamma={gamma}',flush=True)
            core.write_json(root/'diagnostics/damping.json',rows)
            core.write_json(root/'diagnostics/ridge.json',ridges)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=800)
    args=parser.parse_args(); cfg=json.loads((args.root/'manifest.json').read_text())['config']
    deadline=time.monotonic()+args.seconds
    for path in sorted((args.root/'training').glob('*/case.json')):
        case_certificates(args.root,path.parent,cfg,deadline)
        print(f'DIAGNOSTICS {path.parent.name}',flush=True)
    damping(args.root,cfg,deadline)
    core.write_json(args.root/'validation/diagnostics_complete.json',dict(complete=True))


if __name__=='__main__':
    main()
