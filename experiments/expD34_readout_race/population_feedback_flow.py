"""Six-target verification of effective flow, GD, and their feedback budgets.

Only initial and final parameter vectors are retained. Scalar diagnostics
are sampled every 0.2 flow-time units, including tracking derivative loading.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from . import mechanism_persistence_kernel as kernel
from .population_feedback_budget import exp_budget_integral
from .population_accumulated_audit import cumulative


def rk4(p,dt,field):
    k1=field(p); k2=field(p-dt*k1/2)
    k3=field(p-dt*k2/2); k4=field(p-dt*k3)
    return p-dt*(k1+2*k2+2*k3+k4)/6


@partial(jax.jit,static_argnames=('steps','kind'))
def advance(p,x,y,dt,steps,kind):
    field=lambda v:kernel.effective(v,x,y)
    def body(_,q):
        if kind=='effective': return rk4(q,dt,field)
        return q-dt*kernel.ordinary_gradient(q,x,y)
    return jax.lax.fori_loop(0,steps,body,p)


@jax.jit
def diagnostics(p,x,y):
    state=kernel.decomposition(p,x,y)
    F,R,J,eh,basis,jc=(state[k] for k in ('F','R','J','eH','basis','JC'))
    f=jnp.linalg.norm(F); v=F/f
    second=kernel._second_output(p,x,v)
    hc=basis.T@second/len(x); hh=kernel._fine(second,basis)
    geometry=-jnp.mean(eh*hh); compensation=state['balance']@hc
    rate=jnp.sqrt(jnp.mean(eh*eh))*jnp.sqrt(jnp.mean(hh*hh))+jnp.linalg.norm(state['balance'])*jnp.linalg.norm(hc)
    dr=jax.jvp(lambda q:kernel.effective(q,x,y),(p,),(R,))[1]
    df=jax.jvp(lambda q:kernel.effective(q,x,y),(p,),(F,))[1]
    jf=kernel._fine(J@F,basis)
    relax=jnp.mean(jf*jf)/f**2
    w=(len(p)-1)//3
    particles=p[:-1].reshape(3,-1)
    forceblocks=F[:-1].reshape(3,-1)
    rblocks=R[:-1].reshape(3,-1)
    r2=jnp.sum(particles**2,axis=0)
    residual=kernel.output(p,x)-y
    ec=basis.T@residual/len(x)
    return dict(f=f,Y=jnp.sqrt(jnp.mean(eh*eh)),target_norm=jnp.sqrt(jnp.mean(y*y)),
                relative_error=jnp.sqrt(jnp.mean(residual*residual)/jnp.mean(y*y)),
                M4=w*jnp.sum(r2*r2),I=w*jnp.sum(jnp.sum(forceblocks**2,axis=0)**2)/f**4,
                rate=rate,signed_rate=jnp.maximum(geometry+compensation,0),
                relaxation=relax,geometry=geometry,compensation=compensation,
                uR=jnp.linalg.norm(dr),rR=(w*jnp.sum(jnp.sum(rblocks**2,axis=0)**2))**.25,
                zR=jnp.maximum(jnp.mean(eh*(J@R)),0),R_norm=jnp.linalg.norm(R),
                identity_relative=jnp.abs(F@df/f**2-(relax-geometry-compensation)),
                coarse0=ec[0],coarse1=ec[1],sigma=jnp.sqrt(jnp.linalg.eigvalsh(state['gram'])[0]))


def summarize(rows):
    t=np.array([r['time'] for r in rows])
    get=lambda key:np.array([r[key] for r in rows])
    B,H=exp_budget_integral(t,get('rate'))
    f0=rows[0]['f']; Y0=rows[0]['Y']; target=rows[0]['target_norm']
    fbar=np.exp(B)*(f0+cumulative(t,np.exp(-B)*get('uR')))
    tracked_dissipation=cumulative(t,fbar*fbar)
    error_floor=np.sqrt(np.maximum(0,Y0**2-2*f0*f0*H))/target
    tracked_floor=np.sqrt(np.maximum(0,Y0**2-2*tracked_dissipation-2*cumulative(t,get('zR'))))/target
    qbound=rows[0]['M4']**.25+np.sqrt(cumulative(t,np.sqrt(get('I')))*f0*f0*H)
    energy_defect=get('Y')**2-Y0**2+2*cumulative(t,get('f')**2)
    coarse_drift=np.hypot(get('coarse0')-rows[0]['coarse0'],get('coarse1')-rows[0]['coarse1'])
    for i,row in enumerate(rows):
        row.update(B=B[i],force_envelope=f0*np.exp(B[i]),energy_floor=error_floor[i],
                   tracked_force_envelope=fbar[i],tracked_energy_floor=tracked_floor[i],
                   M4_envelope=qbound[i]**4)
    return dict(target=rows[0]['target'],kind=rows[0]['kind'],dt=rows[0]['dt'],
                initial_force=f0,final_force=rows[-1]['f'],B=B[-1],
                initial_rate=rows[0]['rate'],max_prefix_rate_ratio=max(B[1:]/(t[1:]*rows[0]['rate'])),
                relative_error=rows[-1]['relative_error'],energy_floor=error_floor[-1],
                tracked_energy_floor=tracked_floor[-1],
                initial_M4=rows[0]['M4'],final_M4=rows[-1]['M4'],final_M4_envelope=qbound[-1]**4,
                maximum_force_excess=max(get('f')-f0*np.exp(B)),
                maximum_tracked_force_excess=max(get('f')-fbar),
                maximum_energy_floor_excess=max(error_floor-get('relative_error')),
                maximum_identity_relative=max(get('identity_relative')),
                maximum_coarse_drift=max(coarse_drift),maximum_effective_energy_defect=max(abs(energy_defect)),
                total_tracking_derivative=cumulative(t,get('uR'))[-1],
                total_tracking_hidden_travel=cumulative(t,get('rR'))[-1],
                total_adverse_tracking_energy=cumulative(t,get('zR'))[-1],
                final_tracked_force_envelope=fbar[-1])


def run(args):
    if jax.default_backend()!='gpu': raise RuntimeError('This continuation requires the allocated GPU')
    started=time.monotonic(); args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.inputs,allow_pickle=False) as data:
        ps,x,ys=data['p'],data['x'],data['y']; cases=json.loads(str(data['cases']))
    x=jnp.asarray(x); rows=[]; summaries=[]; endpoints={}; comparisons=[]
    for p0,y,case in zip(ps,ys,cases):
        y=jnp.asarray(y); target=case['target']; traces={}
        for kind,dt in (('effective',.02),('effective',.01),('gd',.002),('gd',.001)):
            p=jnp.asarray(p0); path=[]
            for k in range(201):
                raw=jax.device_get(diagnostics(p,x,y))
                row=dict(target=target,kind=kind,dt=dt,time=.2*k,**{key:float(value) for key,value in raw.items()})
                path.append(row)
                if k<200: p=advance(p,x,y,dt,steps=round(.2/dt),kind=kind)
            summary=summarize(path); rows.extend(path); summaries.append(summary)
            traces[(kind,dt)]=path; endpoints[(target,kind,dt)]=np.asarray(p)
            print(json.dumps(summary),flush=True)
            if time.monotonic()-started > 3000: raise RuntimeError('Campaign continuation budget exhausted')
        ref=endpoints[(target,'effective',.01)]
        a,b=traces[('effective',.02)],traces[('effective',.01)]
        comparisons.append(dict(target=target,
            effective_step_parameter_difference=float(np.linalg.norm(endpoints[(target,'effective',.02)]-ref)),
            gd_step_parameter_difference=float(np.linalg.norm(endpoints[(target,'gd',.002)]-endpoints[(target,'gd',.001)])),
            gd_vs_effective_parameter_difference=float(np.linalg.norm(endpoints[(target,'gd',.001)]-ref)),
            effective_relative_error_difference=abs(a[-1]['relative_error']-b[-1]['relative_error']),
            gd_vs_effective_relative_error_difference=abs(traces[('gd',.001)][-1]['relative_error']-b[-1]['relative_error']),
            feedback_quadrature_full_vs_half=abs(b[-1]['B']-summarize([dict(v) for v in b[::2]])['B'])))
    with (args.output/'states.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    facts=dict(scope='FP64 integrations and sampled feedback budgets, not trajectory certificates',
               device=str(jax.devices()[0]),flow_horizon=40,sampling_interval=.2,
               cases=cases,trajectories=summaries,comparisons=comparisons,seconds=time.monotonic()-started)
    (args.output/'facts.json').write_text(json.dumps(facts,indent=2)+'\n')
    np.savez_compressed(args.output/'endpoints.npz',p0=ps,
        endpoints=np.stack(list(endpoints.values())),labels=np.array([str(v) for v in endpoints]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
