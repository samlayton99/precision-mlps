"""State-conditioned force drivers and frozen-geometry usefulness for D34.

No fitted model or readout solve enters a training update. Flow derivatives
are evaluated at actual GD states; finite-step motion is measured separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from . import targets, transport as tr
from .recovery import clean, concentration, write_table


ARMS = dict(joint=(1., 1., 1., 1.), freeze_readout=(1., 1., 0., 0.),
            freeze_slopes=(0., 1., 1., 1.), freeze_biases=(1., 0., 1., 1.),
            slow_readout=(1., 1., .1, .1), fast_readout=(1., 1., 10., 10.))
HORIZONS = (0, 2000, 20000, 100000, 600000)
GAMMAS = (.1, .25, .5, 1., 2., 3.2, 4., 8., 16.)


def ratio(a, b):
    return float(a/b) if b > 0 else np.nan


def force_metrics(z, d, x, y, rates=ARMS['joint'], degree=65, modes=None):
    """Return force decomposition and signed d(||effective force||²/2)/dt.

The four driver terms sum as vectors. None is assumed to have a negative
projection on the current effective force, including residual relaxation.
"""
    width = z.shape[1]
    row, arr = tr.modal_diagnostics(z, d, x, y, np.ones(width)/width,
                                    width, degree, rates=rates, modes=modes)
    e, ja, grad, r = (arr[k] for k in ('residual_modes', 'J_a', 'gradient', 'residual'))
    ga = grad[0]
    full2, residual2 = ga @ ga, np.mean(r*r)
    gamma = abs(z[0])
    outward = -np.sign(z[0])/width
    row.update(half_mse=residual2/2, residual_rms=np.sqrt(residual2),
        fine_residual_norm=np.linalg.norm(e[2:]), mean_gamma=gamma.mean(),
        median_gamma=np.median(gamma), q90_gamma=np.quantile(gamma, .9), max_gamma=gamma.max(),
        readout_l2=np.linalg.norm(z[2]), xi=np.sqrt(ratio(full2, residual2)),
        full_outward=outward @ ga, actual_outward=rates[0]*(outward @ ga),
        actual_slope_speed=rates[0]*np.sqrt(full2),
        signed_alignment=ratio(np.sqrt(width)*(outward @ ga), np.sqrt(full2)),
        **{f'fraction_{g:g}': np.mean(gamma >= g) for g in (1., 3.2, 16.)})
    # Direct full-force derivative: readout prefactor, activation shape, residual.
    u=x[:, None]*z[0]+z[1]; h=np.tanh(u)
    exp=np.exp(-2*abs(u)); s=4*exp/(1+exp)**2
    udot=-rates[0]*x[:, None]*grad[0]-rates[1]*grad[1]
    rdot=(s*udot) @ z[2]-rates[2]*h @ grad[2]-rates[3]*r.mean()
    gain=-rates[2]*grad[2]*(x @ (r[:, None]*s))/len(x)
    shape=z[2]*(x @ (r[:, None]*(-2*h*s*udot)))/len(x)
    residual=z[2]*(x @ (rdot[:, None]*s))/len(x)
    for label, vector in [('readout_gain', gain), ('shape_change', shape), ('residual_change', residual)]:
        row['full_force_driver_'+label]=float(ga @ vector)
    arr['full_force_dot']=gain+shape+residual
    if not row['coarse_inverse_resolved']:
        return row, arr
    K=sum(arr['K_'+key] for key in 'abcd'); Kdot=arr['K_dot']
    C,Q=K[:2,:2],K[:2,2:]
    B=np.linalg.solve(C,Q); S=K[2:,2:]-Q.T @ B
    Bdot=np.linalg.solve(C,Kdot[:2,2:]-Kdot[:2,:2] @ B)
    T=ja[2:].T-ja[:2].T @ B
    Tdot=arr['J_a_dot'][2:].T-arr['J_a_dot'][:2].T @ B-ja[:2].T @ Bdot
    zc=e[:2]+B @ e[2:]
    F=T @ e[2:]; tracking=ja[:2].T @ zc
    tail=ga-F-tracking
    omitted=arr['residual_velocity']+K @ e
    drivers=dict(map_drift=Tdot @ e[2:], fine_relaxation=-T @ S @ e[2:],
                 tracking=-T @ Q.T @ zc, omitted=T @ omitted[2:])
    fdot=Tdot @ e[2:]+T @ arr['residual_velocity'][2:]
    row['effective_derivative_identity_error']=np.linalg.norm(sum(drivers.values())-fdot)
    for label, vector in drivers.items():
        row['effective_driver_'+label]=float(F @ vector)
        row['effective_driver_norm_'+label]=float(np.linalg.norm(vector))
    row['effective_force_energy_derivative']=float(F @ fdot)
    for label, force in [('effective', F), ('tracking', tracking), ('omitted', tail)]:
        row[label+'_outward']=float(outward @ force)
    fine2=e[2:] @ e[2:]; eff2=F @ F; dissipation=e[2:] @ S @ e[2:]
    row.update(effective_coupling=ratio(eff2, fine2), effective_fitting_rate=ratio(dissipation, fine2),
        effective_outward_alignment=ratio(np.sqrt(width)*(outward @ F), np.sqrt(eff2)),
        tracking_metric=np.sqrt(max(0., zc @ C @ zc)),
        force_triangle_ratio=ratio(np.sqrt(full2),np.linalg.norm(F)+np.linalg.norm(tracking)+np.linalg.norm(tail)))
    for key, rate in zip('abcd', rates):
        j=arr['J_'+key][:, None] if key=='d' else arr['J_'+key]
        tj=j[2:].T-j[:2].T @ B
        row['effective_'+key+'_share']=ratio(rate*np.linalg.norm(tj @ e[2:])**2,dissipation)
    cv,cu=np.linalg.eigh(C); ir=(cu/np.sqrt(cv)) @ cu.T
    row['tracking_metric_decay_lower']=cv[0]-.5*np.linalg.eigvalsh(ir @ Kdot[:2,:2] @ ir)[-1]
    fz_parts=dict(map_drift=Bdot @ e[2:], fine_relaxation=-B @ S @ e[2:],
                  omitted=omitted[:2]+B @ omitted[2:])
    fz=sum(fz_parts.values())
    row['tracking_forcing_metric']=np.sqrt(max(0.,fz @ C @ fz))
    for label, vector in fz_parts.items():
        row['tracking_forcing_'+label]=np.sqrt(max(0.,vector @ C @ vector))
    modal=T*e[None,2:]
    row['modal_coherence']=ratio(np.linalg.norm(F),np.linalg.norm(modal,axis=0).sum())
    arr.update(effective_force=F, effective_force_dot=fdot, tracking_force=tracking,
        mode_force_norm=np.linalg.norm(modal,axis=0), mode_force_along=F @ modal,
        mode_outward=outward @ modal)
    return row, arr


def design(a, b, x):
    return np.column_stack((np.tanh(x[:,None]*a+b),np.ones(len(x))))


def frozen_curves(a, b, x, y, xe, ye, horizons=HORIZONS, eta=.002):
    """Numerically evaluated exact linear GD from zero, including tiny modes.

Use the stable geometric sum for finite-time coefficients. A truncated solve
is a separate capacity diagnostic and is never substituted for the GD curve.
"""
    A=design(a,b,x); Ae=design(a,b,xe)
    U,s,Vh=svd(A/np.sqrt(len(x)),full_matrices=False,check_finite=False)
    loading=U.T @ (y/np.sqrt(len(x)))
    t=eta*s*s
    if np.max(t) >= 1:
        raise ValueError('This diagnostic requires nonoscillating stable frozen GD')
    rows=[]; coefficients=[]
    for n in horizons:
        factor=-np.expm1(n*np.log1p(-t))
        amp=np.divide(factor,s,out=np.zeros_like(s),where=s>0)*loading
        c=Vh.T @ amp
        rows.append(dict(updates=int(n),train_mse=float(np.mean((A @ c-y)**2)),
            heldout_mse=float(np.mean((Ae @ c-ye)**2)),readout_l2=float(np.linalg.norm(c)),
            relative_train_mse=float(np.mean((A @ c-y)**2)/np.mean(y*y)),
            relative_heldout_mse=float(np.mean((Ae @ c-ye)**2)/np.mean(ye*ye))))
        coefficients.append(c)
    capacity=[]
    for cutoff in (1e-10,1e-12,1e-14):
        keep=s>cutoff*s[0]
        c=Vh[keep].T @ (loading[keep]/s[keep])
        capacity.append(dict(cutoff=cutoff,rank=int(keep.sum()),train_mse=float(np.mean((A @ c-y)**2)),
            heldout_mse=float(np.mean((Ae @ c-ye)**2)),readout_l2=float(np.linalg.norm(c))))
    return rows, capacity, np.stack(coefficients)


def analyze(archives, output, stride=1):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; windows=[]; modal=[]; provenance=[]
    for path in archives:
        f=dict(np.load(path)); x=f['x']; q=tr.basis(x,65)
        cases=json.loads(str(f['cases']))
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for i,case in enumerate(cases):
            if isinstance(case,list): case=dict(seed=case[0],target=case[1])
            case=dict(arm='joint',fork_step=0,eta=.002,**case) if 'arm' not in case else case
            rates=ARMS[case['arm']]; indices=sorted(set(range(0,len(f['steps']),stride))|{len(f['steps'])-1})
            measured=[]
            for j in indices:
                row,arr=force_metrics(f['z'][i,j],f['d'][i,j],x,f['y'][i],rates,modes=q)
                meta=dict(**case,step=int(f['steps'][j]),archive=path.parent.name)
                row=dict(**meta,**row)
                row['gradient_path']=float(f['path'][i,j])
                row['positive_mean']=float(f['positive'][i,j].mean())
                row['negative_mean']=float(f['negative'][i,j].mean())
                rows.append(row); measured.append(row)
                if 'mode_force_norm' in arr:
                    modal.append((len(rows)-1,arr['mode_force_norm'],arr['mode_force_along'],arr['mode_outward'],arr['residual_modes']))
            for lo,hi in ((0,20000),(20000,100000),(100000,600000),(20000,600000)):
                js=np.flatnonzero(f['steps']==lo); je=np.flatnonzero(f['steps']==hi)
                if not len(js) or not len(je): continue
                s,t=int(js[0]),int(je[0]); positive=f['positive'][i,t]-f['positive'][i,s]
                negative=f['negative'][i,t]-f['negative'][i,s]
                rr=[r for r in measured if lo<=r['step']<=hi]
                times=np.array([r['step']*case['eta'] for r in rr]); speeds=np.array([r['actual_slope_speed'] for r in rr])
                exact=float(f['path'][i,t]-f['path'][i,s])
                windows.append(dict(**case,archive=path.parent.name,start=lo,end=hi,path=exact,
                    sampled_path=float(np.trapezoid(speeds,times)),
                    decimated_path=float(np.trapezoid(speeds[::2],times[::2])) if len(rr)%2 else np.nan,
                    positive_mean=positive.mean(),negative_mean=negative.mean(),
                    motion_identity_error=float(np.max(abs(positive-negative-(abs(f['z'][i,t,0])-abs(f['z'][i,s,0]))))),
                    **concentration(positive)))
            print(json.dumps(dict(archive=path.parent.name,case=case,states=len(indices))),flush=True)
    write_table(output/'metrics.csv',rows); write_table(output/'movement_windows.csv',windows)
    np.savez_compressed(output/'modal_forces.npz',row=np.array([r[0] for r in modal]),
        norm=np.array([r[1] for r in modal]),along_effective=np.array([r[2] for r in modal]),
        outward=np.array([r[3] for r in modal]),residual_modes=np.array([r[4] for r in modal]))
    (output/'manifest.json').write_text(json.dumps(clean(dict(archives=provenance,states=len(rows),
        degree=65,stride=stride,force_driver_units='d(norm squared / 2)/d physical time')),indent=2)+'\n')


def geometry_study(archives, output):
    output.mkdir(parents=True,exist_ok=True)
    x=targets.grid(2048); xe=targets.grid(8192); mapping=targets.polynomial_map(x)
    curves=[]; capacities=[]
    def measure(a,b,target,meta):
        y=targets.values(target,x,mapping); ye=targets.values(target,xe,mapping)
        rr,cc,_=frozen_curves(a,b,x,y,xe,ye)
        curves.extend(dict(target=target,**meta,**r) for r in rr)
        capacities.extend(dict(target=target,**meta,**r) for r in cc)
    centers=-1+2*np.arange(-24,128+24+1)/128
    for target in targets.TARGETS:
        for gamma in GAMMAS:
            measure(np.full(177,gamma),-gamma*centers,target,dict(kind='construction_centers',gamma=gamma))
    for path in archives:
        f=dict(np.load(path)); cases=json.loads(str(f['cases']))
        for i,case in enumerate(cases):
            if isinstance(case,list): case=dict(seed=case[0],target=case[1])
            previous=None
            for step in (0,20000,100000,600000):
                indices=np.flatnonzero(f['steps']==step)
                if not len(indices): continue
                j=int(indices[0]); a,b=f['z'][i,j,:2]
                meta=dict(seed=case['seed'],arm=case.get('arm','joint'),fork_step=case.get('fork_step',0),
                          step=step,archive=path.parent.name)
                measure(a,b,case['target'],dict(**meta,kind='learned'))
                if previous is not None and case.get('arm','joint')=='joint':
                    oldstep,olda,oldb=previous
                    measure(a,oldb,case['target'],dict(**meta,kind='new_slopes_old_biases',previous_step=oldstep))
                    measure(olda,b,case['target'],dict(**meta,kind='old_slopes_new_biases',previous_step=oldstep))
                previous=(step,a,b)
            print(json.dumps(dict(geometry=case,archive=path.parent.name)),flush=True)
    write_table(output/'frozen_curves.csv',curves); write_table(output/'capacity.csv',capacities)
    (output/'protocol.json').write_text(json.dumps(dict(eta=.002,horizons=HORIZONS,gammas=GAMMAS,
        initialization='zero readout; diagnostic only',width=177,training_samples=2048,evaluation_samples=8192,
        target_mapping='fitted once on the D34 training grid',archives=[str(p) for p in archives]),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--geometry',action='store_true')
    parser.add_argument('--stride',type=int,default=1)
    args=parser.parse_args()
    if args.geometry: geometry_study(args.archives,args.output)
    else: analyze(args.archives,args.output,args.stride)


if __name__=='__main__': main()
