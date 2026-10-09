"""Preliminary disk PDE solves in prescribed coordinates of a flat tanh bank.

The unknowns are polynomial-profile coordinates a, with explicit readout w=Ea.
Every collocation value/Laplacian comes from actual tanh evaluations. This is
not an independent million-weight optimization and does not establish general
high-dimensional PDE scaling. Run with BLAS/OMP threads set to one.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import lstsq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/expF19_radon_direct_pde'))
from quill_boundary import encode
from solver.ridge_pinn_features import RidgePINNFeatures, chebyshev_u
from local_operator import LocalBank
OUT=ROOT/'results/checkpoint_F_applications/expF20_local_ridge_pde'


def disk_points(n,seed):
    r=np.random.default_rng(seed)
    radius=np.sqrt(r.random(n)); theta=2*np.pi*r.random(n)
    return np.column_stack([radius*np.cos(theta),radius*np.sin(theta)])


def boundary(n,offset=0.):
    theta=2*np.pi*(np.arange(n)+offset)/n
    return np.column_stack([np.cos(theta),np.sin(theta)])


def target(points,frequency):
    x,y=points.T
    base=np.exp(.3*x)*np.cos(frequency*y)
    return base+.1*x*y,(frequency**2-.3**2)*base


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def rel(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))


def build(m,centers,degree,points,cutoff=18.):
    start=time.perf_counter()
    geom=RidgePINNFeatures(degree,directions=m,centers=centers,backend='polynomial')
    enc=encode(lambda z:chebyshev_u(z,degree),centers-1,lam=.2)
    bank=LocalBank(geom.directions,enc.centers,enc.gamma,cutoff)
    plan=bank.bind(points)
    geometry_seconds=time.perf_counter()-start
    features=[]
    errors=[]
    for order in [0,2]:
        began=time.perf_counter()
        F=np.empty((len(points),geom.size)); F[:,0]=1/np.sqrt(np.pi) if order==0 else 0
        col=1
        for n in range(1,degree+1):
            weights=np.broadcast_to(enc.weights[:,n],(m,len(enc.centers)))
            profiles=plan.profiles(weights,order)
            if order==0: profiles+=enc.bias[n]
            F[:,col:col+n+1]=profiles.T@geom.angular_maps[n]
            col+=n+1
        features.append(F)
        # Compare actual neuron derivatives against analytic basis on fixed points.
        q=points[:min(32,len(points))]
        truth=geom.evaluate(q) if order==0 else geom.evaluate(q,(2,0))+geom.evaluate(q,(0,2))
        errors.append(dict(order=order,max_abs=float(np.max(abs(F[:len(q)]-truth))),
                           rms=rms(F[:len(q)]-truth),seconds=time.perf_counter()-began))
    return geom,enc,bank,features,dict(geometry_encoding_seconds=geometry_seconds,
            features_seconds=time.perf_counter()-start-geometry_seconds,profile_checks=errors)


def compile_weights(geom,enc,a):
    coeff=np.zeros((geom.direction_count,geom.degree+1))
    col=1
    for n in range(1,geom.degree+1):
        coeff[:,n]=geom.angular_maps[n]@a[col:col+n+1]
        col+=n+1
    return coeff@enc.weights.T,float(a[0]/np.sqrt(np.pi)+np.sum(coeff@enc.bias))


def dense_network(points,directions,enc,weights,bias,order):
    result=np.zeros(len(points))
    # Bound temporary arrays; loop direction blocks, ordinary dense tanh features.
    for lo in range(0,len(directions),8):
        p=directions[lo:lo+8]@points.T
        z=enc.gamma*(p[:,:,None]-enc.centers[None,None,:])
        t=np.tanh(z)
        if order==2:
            e=np.exp(-2*np.abs(z));t=-2*enc.gamma**2*t*(4*e/(1+e)**2)
        result+=np.einsum('mkn,mn->k',t,weights[lo:lo+8])
    return result+bias if order==0 else result


def solve(F,L,points,ni,nb,nhi,nhb,p,frequency,nonlinear):
    d=(p+1)*(p+2)//2
    V=F[:ni,:d]; A=-L[:ni,:d]; B=F[ni:ni+nb,:d]
    exact,minuslap=target(points,frequency)
    forcing=minuslap+exact**3 if nonlinear else minuslap
    forcing_scale=max(1.,rms(forcing[:ni])); boundary_scale=max(1.,rms(exact[ni:ni+nb]))
    interior_weight=1/(np.sqrt(ni)*forcing_scale)
    boundary_weight=3/(np.sqrt(nb)*boundary_scale)
    a=np.zeros(d); history=[]
    def residual(a):
        pred=V@a
        return np.r_[(A@a+(pred**3 if nonlinear else 0)-forcing[:ni])*interior_weight,
                     (B@a-exact[ni:ni+nb])*boundary_weight]
    began=time.perf_counter()
    for iteration in range(15):
        r=residual(a); objective=float(np.linalg.norm(r))
        history.append(dict(iteration=iteration,weighted_residual=objective))
        J=np.vstack([(A+(3*(V@a)**2)[:,None]*V if nonlinear else A)*interior_weight,
                     B*boundary_weight])
        delta,_,rank,_=lstsq(J,-r,cond=1e-13,lapack_driver='gelsy')
        step=1.
        while step>2**-16 and np.linalg.norm(residual(a+step*delta))>objective*(1-1e-4*step):
            step*=.5
        history[-1].update(step=step,correction_norm=float(np.linalg.norm(delta)),rank=int(rank))
        if step<=2**-16:break
        a+=step*delta
        if np.linalg.norm(delta)<1e-12*(1+np.linalg.norm(a)):break
        if not nonlinear:break
    seconds=time.perf_counter()-began
    history.append(dict(iteration=len(history),weighted_residual=float(np.linalg.norm(residual(a)))))
    i=slice(ni+nb,ni+nb+nhi);b=slice(ni+nb+nhi,None)
    pred=F[:,:d]@a;lp=-L[:,:d]@a
    residual_pde=lp+(pred**3 if nonlinear else 0)-forcing
    row=dict(degree=p,independent_coordinates=d,target_frequency=frequency,
             equation='-laplace(u)+u^3=f' if nonlinear else '-laplace(u)=f',
             initialization='all zero',uses_solution_values_only_on_boundary=True,
             interior_samples=ni,boundary_samples=nb,heldout_interior=nhi,heldout_boundary=nhb,
             solve_seconds=seconds,heldout_field_relative_l2=rel(pred[i],exact[i]),
             heldout_field_max_abs=float(np.max(abs(pred[i]-exact[i]))),
             heldout_pde_relative_l2=float(np.linalg.norm(residual_pde[i])/np.linalg.norm(forcing[i])),
             heldout_pde_rms=rms(residual_pde[i]),heldout_boundary_max_abs=float(np.max(abs(pred[b]-exact[b]))),
             train_pde_relative_l2=float(np.linalg.norm(residual_pde[:ni])/np.linalg.norm(forcing[:ni])),
             history=history,coefficients=a.tolist())
    return row,np.pad(a,(0,F.shape[1]-len(a))),pred


def run(m,centers,label,degrees=(6,10,14),frequencies=(1.3,5.)):
    ni,nb,nhi,nhb=1024,128,2048,401
    points=np.vstack([disk_points(ni,101),boundary(nb),disk_points(nhi,202),boundary(nhb,.37)])
    began=time.perf_counter()
    geom,enc,bank,(F,L),setup=build(m,centers,max(degrees),points)
    print(json.dumps(dict(label=label,stage='features',seconds=time.perf_counter()-began)),flush=True)
    rows=[]; last=None
    for p in degrees:
        for frequency in frequencies:
            for nonlinear in [False,True]:
                row,a,pred=solve(F,L,points,ni,nb,nhi,nhb,p,frequency,nonlinear)
                row.update(label=label,tanh_neurons=bank.size,directions=m,centers_per_direction=len(enc.centers))
                rows.append(row);last=(a,pred)
                print(json.dumps({k:v for k,v in row.items() if k not in ('coefficients','history')}),flush=True)
    a,pred=last
    weights,bias=compile_weights(geom,enc,a)
    audit_points=points[ni+nb:ni+nb+16]
    audit=[]
    for order in [0,2]:
        plan=bank.bind(audit_points)
        start=time.perf_counter();local=plan.forward(weights,order)+(bias if order==0 else 0);local_seconds=time.perf_counter()-start
        start=time.perf_counter();dense=dense_network(audit_points,geom.directions,enc,weights,bias,order);dense_seconds=time.perf_counter()-start
        wider=LocalBank(geom.directions,enc.centers,enc.gamma,26).bind(audit_points).forward(weights,order)+(bias if order==0 else 0)
        reduced=(F if order==0 else L)[ni+nb:ni+nb+16]@a
        audit.append(dict(order=order,points=len(audit_points),local_seconds=local_seconds,dense_seconds=dense_seconds,
                          speedup=dense_seconds/local_seconds,dense_local_max_abs=float(np.max(abs(dense-local))),
                          cutoff18_vs26_max_abs=float(np.max(abs(wider-local))),
                          reduced_vs_compiled_max_abs=float(np.max(abs(local-reduced)))))
    profile_audit=[]
    profile_x=np.linspace(-1.,1.,128)
    profile_plan=LocalBank([[1.,0.]],enc.centers,enc.gamma,18).bind(np.column_stack([profile_x,np.zeros_like(profile_x)]))
    for order in [0,2]:
        start=time.perf_counter()
        dense=enc.evaluate(profile_x,derivative=order)
        dense_seconds=time.perf_counter()-start
        start=time.perf_counter()
        local=np.column_stack([profile_plan.forward(enc.weights[:,n],order) for n in range(geom.degree+1)])
        if order==0:local+=enc.bias
        local_seconds=time.perf_counter()-start
        profile_audit.append(dict(order=order,points=len(profile_x),profile_columns=geom.degree+1,
             existing_dense_seconds=dense_seconds,local_seconds=local_seconds,speedup=dense_seconds/local_seconds,
             max_abs=float(np.max(abs(local-dense)))))
    result=dict(label=label,method='Dense reduced-coordinate collocation; actual local-window tanh derivatives; w=Ea',
        qualification=f'Million-neuron counts stress overresolved representation; only up to {geom.size} coordinates are optimized; this is not a free-weight scalability claim.',
        architecture='flat single-hidden-layer tanh plus output bias',
        seed_train=101,seed_holdout=202,directions=m,interior_centers=centers,centers_per_direction=len(enc.centers),
        neurons=bank.size,gamma=enc.gamma,lambda_gamma_h=enc.lam,cutoff=18.,
        setup=setup,rows=rows,compiled_dense_audit=audit,existing_encoding_audit=profile_audit,total_seconds=time.perf_counter()-began)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'reduced_{label}.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/f'reduced_{label}_snapshot.npz',points=points,field=pred,truth=target(points,frequencies[-1])[0])
    print(json.dumps(dict(label=label,stage='complete',seconds=result['total_seconds'],audit=audit)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['small','million','all','refinement'],default='all')
    args=parser.parse_args()
    if args.case in ['small','all']:run(32,257,'small')
    if args.case in ['million','all']:run(256,4097,'million')
    if args.case=='refinement':run(256,4097,'million_refinement',degrees=(18,22),frequencies=(5.,))
