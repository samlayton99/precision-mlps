"""Reproducible local-operator scaling and independent-readout PDE probes.

No interior solution labels enter the solve. Optional smooth coordinates are
an invertible DCT filter, not dimension reduction; they change the minimum-
norm bias of underdetermined least squares and are reported separately.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import platform
import resource
import time
import numpy as np
from scipy.fft import dct,idct
from scipy.sparse.linalg import LinearOperator,lsmr
from scipy.stats import qmc
from local_operator import LocalBank,OUT


def disk(n,seed):
    u=qmc.Sobol(2,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
    angle=2*np.pi*u[:,1]
    return np.sqrt(u[:,0,None])*np.c_[np.cos(angle),np.sin(angle)]


def boundary(n,shift):
    a=2*np.pi*(np.arange(n)+shift)/n
    return np.c_[np.cos(a),np.sin(a)]


def exact(x):return np.exp(.3*x[:,0])*np.cos(1.3*x[:,1])+.1*x[:,0]*x[:,1]
def source(x):return 1.6*np.exp(.3*x[:,0])*np.cos(1.3*x[:,1])


def geometry(m,n,cutoff=18.):
    a=np.arange(m)*np.pi/m
    c=np.linspace(-1.25,1.25,n)
    return LocalBank(np.c_[np.cos(a),np.sin(a)],c,.25/(c[1]-c[0]),cutoff)


def timed(fun,repeats=3):
    times=[]
    for _ in range(repeats):
        before=time.perf_counter();result=fun();times.append(time.perf_counter()-before)
    return result,float(np.median(times))


def benchmark(m,n,k=1024):
    started=time.perf_counter();bank=geometry(m,n);plan=bank.bind(disk(k,37))
    setup=time.perf_counter()-started
    rng=np.random.default_rng(99);w=rng.normal(size=(m,n))/np.sqrt(m*n);z=rng.normal(size=k)
    rows=[]
    for order in (0,1,2):
        y,fw=timed(lambda:plan.forward(w,order))
        adj,bw=timed(lambda:plan.adjoint(z,order))
        lhs=y@z;rhs=np.sum(w*adj)
        rows.append(dict(order=order,forward_seconds=fw,adjoint_seconds=bw,
                         adjoint_relative_error=float(abs(lhs-rhs)/max(1.,abs(lhs),abs(rhs)))))
    # Dense independent evaluation at a bounded, fixed number of points.
    audit=bank.bind(plan.points[:16]);dense=[]
    for order in (0,1,2):
        def direct():
            result=np.zeros(audit.k)
            for a in range(m):
                zz=bank.gamma*(audit.projection[a,:,None]-bank.centers)
                t=np.tanh(zz);e=np.exp(-2*abs(zz));d=4*e/(1+e)**2
                phi=t if order==0 else bank.gamma*d if order==1 else -2*bank.gamma**2*t*d
                result+=phi@w[a]
            return result
        truth,seconds=timed(direct,1)
        actual,local_seconds=timed(lambda:audit.forward(w,order))
        dense.append(dict(order=order,points=audit.k,dense_seconds=seconds,
            local_seconds=local_seconds,speedup=seconds/local_seconds,
            max_absolute_error=float(np.max(abs(actual-truth))),
            relative_l2_error=float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))))
    row=dict(kind='operator',directions=m,centers=n,neurons=m*n,points=k,gamma=bank.gamma,
        cutoff=bank.cutoff,stencil_width_bound=int(np.ceil(2*bank.cutoff/(bank.gamma*bank.h)))+1,
        setup_seconds=setup,projection_bytes=plan.projection.nbytes,weight_bytes=w.nbytes,
        hypothetical_dense_matrix_bytes=8*k*m*n,products=rows,dense_audit=dense,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        host=platform.node(),machine=platform.machine(),threads=1)
    return row


def solve_free(m,n,k=1024,iterations=200,coordinates='raw',boundary_weight=4.,seed=813,smooth_power=1.,nonlinear=False,restart=50):
    began=time.perf_counter();bank=geometry(m,n);p=bank.size
    train=bank.bind(disk(k,seed));bp=bank.bind(boundary(256,.173))
    test=bank.bind(disk(2048,18973));bt=bank.bind(boundary(401,.5913))
    # Invertible symmetric smoothing in center coordinates; no modes removed.
    frequency=np.pi*np.arange(n)/(bank.h*n)
    multiplier=bank.h/(1+frequency**2)**smooth_power if coordinates=='smooth' else np.full(n,1/bank.gamma**2)
    def transform(v):
        v=np.asarray(v).reshape(m,n)
        return idct(dct(v,type=2,norm='ortho',axis=1)*multiplier,type=2,norm='ortho',axis=1) if coordinates=='smooth' else v*multiplier
    scales=np.r_[np.full(k,1/np.sqrt(k)),np.full(bp.k,boundary_weight/np.sqrt(bp.k))]
    calls=dict(forward=0,adjoint=0)
    sensitivity=np.zeros(k)
    def mv(v):
        calls['forward']+=1;w=transform(v[:-1]);bias=v[-1]
        interior=-train.forward(w,2)
        if nonlinear:interior+=sensitivity*(train.forward(w)+bias)
        return np.r_[interior,bp.forward(w)+bias]*scales
    def rmv(z):
        calls['adjoint']+=1;z=z*scales
        w=-train.adjoint(z[:k],2)+bp.adjoint(z[k:])
        if nonlinear:w+=train.adjoint(z[:k]*sensitivity)
        return np.r_[transform(w).ravel(),np.sum(z[k:])+np.dot(z[:k],sensitivity)]
    op=LinearOperator((k+bp.k,p+1),matvec=mv,rmatvec=rmv,dtype=float)
    forcing=source(train.points)+(exact(train.points)**3 if nonlinear else 0)
    rhs=np.r_[forcing,exact(bp.points)]*scales
    def actual_residual(v):
        weights=transform(v[:-1]);bias=v[-1]
        interior=-train.forward(weights,2)-forcing
        if nonlinear:interior+=(train.forward(weights)+bias)**3
        return np.r_[interior,bp.forward(weights)+bias-exact(bp.points)]*scales
    setup=time.perf_counter()-began
    initial=np.zeros(p+1);history=[];st=time.perf_counter()
    # Restarted LSMR in bounded chunks permits truthful convergence histories.
    total=0;status='iteration_budget_exhausted'
    while total<iterations:
        if nonlinear:
            current=train.forward(transform(initial[:-1]))+initial[-1]
            sensitivity[:]=3*current**2
            rhs=np.r_[forcing+2*current**3,exact(bp.points)]*scales
        result=lsmr(op,rhs,atol=1e-12,btol=1e-12,conlim=0.,maxiter=min(restart,iterations-total),x0=initial)
        candidate=result[0];total+=result[2];step=1.
        if nonlinear:
            previous=np.linalg.norm(actual_residual(initial))
            trial=np.linalg.norm(actual_residual(candidate))
            while step>2**-12 and (not np.isfinite(trial) or trial>previous):
                step*=.5;candidate=initial+step*(result[0]-initial)
                trial=np.linalg.norm(actual_residual(candidate))
            if not np.isfinite(trial) or trial>previous:
                candidate=initial;step=0.;status='line_search_failed'
        initial=candidate
        w=transform(initial[:-1]);bias=initial[-1]
        train_res=actual_residual(initial)
        pred=test.forward(w)+bias;res=-test.forward(w,2)-source(test.points)
        if nonlinear:res+=pred**3-exact(test.points)**3
        bres=bt.forward(w)+bias-exact(bt.points)
        row=dict(iterations=total,seconds=time.perf_counter()-st,lsmr_stop=int(result[1]),step=step,
            weighted_training_residual_l2=float(np.linalg.norm(train_res)),
            train_pde_rms=float(np.linalg.norm(train_res[:k])),
            train_boundary_rms=float(np.linalg.norm(train_res[k:])/boundary_weight),
            fresh_relative_l2=float(np.linalg.norm(pred-exact(test.points))/np.linalg.norm(exact(test.points))),
            fresh_pde_rms=float(np.sqrt(np.mean(res**2))),
            fresh_boundary_rms=float(np.sqrt(np.mean(bres**2))),
            maximum_field_error=float(np.max(abs(pred-exact(test.points)))))
        history.append(row);print(json.dumps(dict(progress=True,coordinates=coordinates,neurons=p,**row)),flush=True)
        if status=='line_search_failed':break
        if np.linalg.norm(train_res)<1e-11:
            status='sampled_residual_tolerance';break
        if ((not nonlinear and result[1] in (0,1,2,4,5)) or result[2]==0):
            status='inner_solver_stopped';break
    elapsed=time.perf_counter()-st
    # Stronger cutoff validates truncation at the solved weights.
    tight=LocalBank(bank.directions,bank.centers,bank.gamma,24.).bind(test.points[:64])
    audit={}
    for order in (0,2):
        audit[str(order)]=float(np.max(abs(tight.forward(w,order)-bank.bind(test.points[:64]).forward(w,order))))
    # Independent all-center evaluation of the solved network on 16 fresh points.
    dense_audit={}
    for order in (0,2):
        ap=bank.bind(test.points[:16]);dense=np.zeros(ap.k)
        for a in range(m):
            zz=bank.gamma*(ap.projection[a,:,None]-bank.centers)
            tt=np.tanh(zz)
            if order==2:
                ee=np.exp(-2*abs(zz));tt=-2*bank.gamma**2*tt*(4*ee/(1+ee)**2)
            dense+=tt@w[a]
        dense_audit[str(order)]=float(np.max(abs(ap.forward(w,order)-dense)))
    suffix=f'_power{smooth_power:g}' if coordinates=='smooth' and smooth_power!=1 else ''
    if nonlinear:suffix+='_nonlinear'
    if restart!=50:suffix+=f'_restart{restart}'
    filename=f'free_{coordinates}_m{m}_n{n}_k{k}_i{iterations}_seed{seed}{suffix}'
    np.savez_compressed(OUT/(filename+'.npz'),weights=w,bias=bias,points=test.points,prediction=pred,truth=exact(test.points))
    return dict(kind='free_solve',coordinates=coordinates,directions=m,centers=n,neurons=p,
        independent_unknowns=p+1,interior_points=k,boundary_points=bp.k,validation_points=test.k,
        seed=seed,boundary_weight=boundary_weight,setup_seconds=setup,solve_and_monitor_seconds=elapsed,
        products=calls,history=history,final=history[-1],cutoff_audit=audit,dense_network_audit=dense_audit,status=status,
        weight_l1=float(np.sum(abs(w))),weight_l2=float(np.linalg.norm(w)),
        smooth_filter_min=float(multiplier.min()),smooth_filter_max=float(multiplier.max()),smooth_power=smooth_power,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        initial='all zero',interior_labels_used=False,pde=('-Delta u+u^3=f' if nonlinear else '-Delta u=f')+' on unit disk, exact Dirichlet data',nonlinear=nonlinear,
        reference='exp(.3*x)*cos(1.3*y)+.1*x*y',
        solver=f'LSMR chunks of {restart} iterations, no damping, scaled constant bias',restart=restart,
        selection_prior='minimum norm in invertibly smoothed DCT coordinates' if coordinates=='smooth' else 'minimum norm in uniformly scaled raw readouts',
        arrays=filename+'.npz')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['benchmark','free'])
    parser.add_argument('--m',type=int,default=32);parser.add_argument('--n',type=int,default=1025)
    parser.add_argument('--k',type=int,default=1024);parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--coordinates',choices=['raw','smooth'],default='raw')
    parser.add_argument('--seed',type=int,default=813)
    parser.add_argument('--smooth-power',type=float,default=1.)
    parser.add_argument('--nonlinear',action='store_true')
    parser.add_argument('--restart',type=int,default=50)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    row=benchmark(args.m,args.n,args.k) if args.mode=='benchmark' else solve_free(args.m,args.n,args.k,args.iterations,args.coordinates,seed=args.seed,smooth_power=args.smooth_power,nonlinear=args.nonlinear,restart=args.restart)
    stem=f'operator_m{args.m}_n{args.n}_k{args.k}' if args.mode=='benchmark' else f'free_{args.coordinates}_m{args.m}_n{args.n}_k{args.k}_i{args.iterations}_seed{args.seed}'
    if args.mode=='free' and args.coordinates=='smooth' and args.smooth_power!=1:stem+=f'_power{args.smooth_power:g}'
    if args.mode=='free' and args.nonlinear:stem+='_nonlinear'
    if args.mode=='free' and args.restart!=50:stem+=f'_restart{args.restart}'
    (OUT/(stem+'.json')).write_text(json.dumps(row,indent=2))
    print(json.dumps(dict(saved=stem,**(row.get('final') or {'neurons':row['neurons'],'products':row['products']}))),flush=True)
