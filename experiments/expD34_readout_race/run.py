"""Batched independent raw-GD trajectories, complete traces and resumable states."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core, targets

DEGREES = (0, 1, 3, 5, 7)
TRACE = ("half_mse", "mean_gamma", "q10", "q25", "median_gamma", "q75", "q90",
         "max_gamma", "readout_l1", "readout_l2", "bias", "max_preactivation",
         "grad_a_norm", "grad_b_norm", "grad_c_norm", "grad_d",
         "signed_coarse_force", "signed_remainder_force", "delta_mean_gamma",
         "crossing_remainder", "sign_crossings", "delta_a_rms", "delta_b_rms",
         "delta_c_rms", "delta_d", "fraction_gamma_1", "fraction_gamma_4",
         "fraction_gamma_16", "fraction_lambda_005", "fraction_lambda_01",
         "fraction_lambda_025") + tuple(f"residual_moment_{k}" for k in range(11))


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez(f, **arrays)
    temporary.replace(path)


def initialize(z, d, batch):
    return dict(z=jnp.broadcast_to(z,(batch,)+z.shape), d=jnp.full(batch,d),
                failed=jnp.zeros(batch,dtype=jnp.int64),
                event_z=jnp.zeros((batch,2)+z.shape), event_d=jnp.zeros((batch,2)),
                event_step=jnp.full((batch,2),-1,dtype=jnp.int64))


def measurements(z, d, zn, dn, loss, moments, grad, gd, coarse, x, n):
    a,b,c=z
    gamma=jnp.abs(a)
    quantiles=jnp.quantile(gamma,jnp.array([.1,.25,.5,.75,.9]))
    sc=-jnp.mean(jnp.sign(a)*coarse)
    sr=-jnp.mean(jnp.sign(a)*(grad[0]-coarse))
    movement=jnp.mean(jnp.abs(zn[0])-gamma)
    crossed=jnp.sum(a*zn[0]<0)
    thresholds=jnp.array([1.,4.,16.,.05*n/2,.1*n/2,.25*n/2])
    fractions=jnp.mean(gamma[None,:]>=thresholds[:,None],axis=1)
    top=jnp.maximum(jnp.abs(a*x[0]+b),jnp.abs(a*x[-1]+b)).max()
    head=jnp.array([loss,jnp.mean(gamma),*quantiles,jnp.max(gamma),
                    jnp.sum(jnp.abs(c)),jnp.linalg.norm(c),d,top,
                    *jnp.linalg.norm(grad,axis=1),gd,sc,sr,movement,0.,crossed,
                    *jnp.sqrt(jnp.mean((zn-z)**2,axis=1)),dn-d])
    return jnp.concatenate((head,fractions,moments))


@lru_cache(maxsize=64)
def chunk(n, m, degree, blocks=50, stride=20):
    x=jnp.asarray(targets.grid(m));powers=x[:,None]**jnp.arange(11)
    Q=powers.T@powers/m
    sigma=jnp.sqrt(Q[1,1])

    def one(state, y, ym, sy, kappa, coarse0, eta, start):
        inputs=dict(x=x,y=y,powers=powers,Q=Q,ym=ym,sy=sy)
        def step(current,index):
            z,d=current['z'],current['d']
            loss,moments,grad,gd,coarse=core.field(z,d,inputs,degree)
            zn=z-eta*jnp.array([1.,1.,kappa])[:,None]*grad
            dn=d-eta*kappa*gd
            finite=(jnp.isfinite(loss)&jnp.all(jnp.isfinite(grad))&jnp.isfinite(gd)
                    &jnp.all(jnp.isfinite(zn))&jnp.isfinite(dn))
            active=(current['failed']==0)&finite
            trace=measurements(z,d,zn,dn,loss,moments,grad,gd,coarse,x,n)
            trace=trace.at[19].set(trace[18]-eta*(trace[16]+trace[17]))
            event=(jnp.linalg.norm(jnp.array([moments[0],moments[1]/sigma]))
                   <= jnp.array([.1,.01])*coarse0)&(current['event_step']<0)&active
            updated=dict(z=jnp.where(active,zn,z),d=jnp.where(active,dn,d),
                failed=jnp.where((current['failed']==0)&~finite,index+1,current['failed']),
                event_z=jnp.where(event[:,None,None],z[None,:,:],current['event_z']),
                event_d=jnp.where(event,d,current['event_d']),
                event_step=jnp.where(event,index,current['event_step']))
            return updated,jnp.where(active,trace,jnp.nan)

        def block(current,index):
            loss,moments,g,gd,coarse=core.field(current['z'],current['d'],inputs,degree)
            snapshot=dict(z=current['z'],d=current['d'],gradient=g,gradient_d=gd,
                          loss=loss,moments=moments,coarse_gradient_a=coarse,
                          failed=current['failed'])
            updated,trace=jax.lax.scan(step,current,start+index*stride+jnp.arange(stride))
            return updated,(trace,snapshot)
        return jax.lax.scan(block,state,jnp.arange(blocks))

    return jax.jit(jax.vmap(one,in_axes=(0,0,0,0,0,0,None,None)))


def specification(n,halo,seed,m,eta,names,ratios):
    return dict(n=n,halo=halo,width=n+2*halo+1,seed=seed,m=m,eta=eta,
                targets=list(names),ratios=list(ratios),evaluation_m=8192,
                coordinates='raw a,b,c,d',loss='half mean squared error',
                initialization='unchanged D28 xavier',dtype='float64',
                simultaneous=True,trace_columns=list(TRACE))


def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [Path('experiments/expD28_loss_gradient_decomposition/run.py'),
              Path('experiments/expD24_gd_residual_spectrum/run.py'),
              Path('experiments/expC05_geometry_interpolation/common.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(folder,config):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'manifest.json'
    cases=[dict(target=name,kappa=k,seed=config['seed'],n=config['n'],m=config['m'],
                eta_geometry=config['eta'],eta_readout=k*config['eta'])
           for name in config['targets'] for k in config['ratios']]
    z,d=targets.initial(config['n'],config['halo'],config['seed'])
    data=[targets.data(config['m'],case['target']) for case in cases]
    y=np.stack([v['y'] for v in data]);ym=np.stack([v['ym'] for v in data]);sy=np.array([v['sy'] for v in data])
    if path.exists():
        old=json.loads(path.read_text())
        if old['configuration']!=config or old['initial_hash']!=targets.array_hash(z,np.array(d)):
            raise ValueError(f'Incompatible resume: {folder}')
    else:
        for degree in DEGREES:
            for i,case in enumerate(cases):
                case_id=hashlib.sha256(json.dumps(case|dict(degree=degree),sort_keys=True).encode()).hexdigest()[:16]
                cases[i][f'run_id_p{degree}']=case_id
        manifest=dict(configuration=config,cases=cases,
            initial_hash=targets.array_hash(z,np.array(d)),data_hash=targets.array_hash(data[0]['x'],y),
            source_hashes=source_hashes(),commit=(os.environ.get('RACE_SOURCE_COMMIT') or
                subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),
            initial_distribution=dict(bound=float(np.sqrt(6/(z.shape[1]+1))),hidden_rng=[config['seed'],config['n']],
                                      readout_rng=[config['seed'],config['n'],24],bias_zero=True),
            scalar_convention='trace step n is before update n to n+1; snapshots store that old state')
        write_json(path,manifest)
        save(folder/'initial.npz',z=z,d=np.array(d),x=data[0]['x'],y=y,ym=ym,sy=sy,
             mapping=data[0]['mapping'],Q=data[0]['Q'])
    return z,d,y,ym,sy,np.array([c['kappa'] for c in cases])


def advance(folder,config,degree,frontier,deadline):
    z,d,y,ym,sy,kappa=prepare(folder,config)
    out=folder/f'p{degree}';out.mkdir(exist_ok=True)
    checkpoint=out/'state.npz'
    state=initialize(jnp.asarray(z),d,len(kappa));start=0
    if checkpoint.exists():
        with np.load(checkpoint) as data:
            start=int(data['step']);state={key:jnp.asarray(data[key]) for key in state}
    if start>=frontier or np.all(np.asarray(state['failed'])>0):
        return True
    inputs={key:jnp.asarray(value) for key,value in targets.data(config['m'],config['targets'][0]).items()}
    coarse0=[]
    for i in range(len(kappa)):
        _,mom,_,_,_=core.field(jnp.asarray(z),jnp.asarray(d),inputs|dict(y=jnp.asarray(y[i]),ym=jnp.asarray(ym[i]),sy=sy[i]),degree)
        coarse0.append(float(jnp.linalg.norm(jnp.array([mom[0],mom[1]/inputs['sigma']]))))
    batch_args=[jnp.asarray(a) for a in (y,ym,sy,kappa,np.asarray(coarse0))]
    begin=time.monotonic()
    while start<frontier:
        if time.monotonic()>deadline-15:
            return False
        # Every first update is captured; later snapshots are every 20 updates.
        stride=1 if start<20 else 20
        length=min(20-start if start<20 else 1000,frontier-start)
        blocks=length//stride
        if blocks==0:
            raise ValueError('Frontiers must preserve the 20-update snapshot grid')
        tick=time.monotonic()
        fn=chunk(config['n'],config['m'],degree,blocks,stride)
        state,(trace,snap)=fn(state,*batch_args,config['eta'],start)
        host_state,trace,snap=jax.device_get((state,trace,snap))
        stop=start+length
        save(out/f'trace_{start:09d}_{stop:09d}.npz',trace=trace.reshape(len(kappa),length,len(TRACE)),
             start=np.array(start),stop=np.array(stop))
        save(out/f'snapshots_{start:09d}_{stop:09d}.npz',steps=start+stride*np.arange(blocks),**snap)
        save(checkpoint,step=np.array(stop),**host_state)
        failed=host_state['failed']
        write_json(out/'status.json',dict(step=stop,frontier=frontier,failed_steps=failed.tolist(),
            stop_reason='all_failed' if np.all(failed>0) else ('frontier' if stop==frontier else 'continuing'),
            last_chunk_seconds=time.monotonic()-tick,invocation_seconds=time.monotonic()-begin,
            job=os.environ.get('SLURM_JOB_ID'),jax_version=jax.__version__))
        print(json.dumps(dict(bundle=folder.name,degree=degree,step=stop,failed=int(np.sum(failed>0)),
                              seconds=round(time.monotonic()-tick,3))),flush=True)
        start=stop
        if np.all(failed>0):
            break
    return True


def bundles(stage,root):
    if stage=='baseline':
        return [(root/'baseline_N128_s0',specification(128,24,0,1024,.002,('sine','runge'),(1.,)))]
    widths=targets.WIDTHS if stage=='core' else ((128,24),)
    seeds=range(5,10) if stage=='replicate' else ([0] if stage=='refine' else range(5))
    result=[]
    for n,halo in sorted(widths,key=lambda v:(v[0]!=128,v[0])):
        for seed in seeds:
            prefix='refine' if stage=='refine' else 'core'
            eta=.001 if stage=='refine' else .002
            result.append((root/f'{prefix}_N{n}_s{seed}',specification(n,halo,seed,2048,eta,targets.TARGETS,targets.RATIOS)))
    return result


def verify_gpu(root):
    job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID')
    mask=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not job or not step or not mask:
        raise RuntimeError('GPU computation requires a Slurm step and preserved nonempty device mask')
    job_info=subprocess.check_output(['scontrol','show','job',job],text=True)
    step_info=subprocess.check_output(['scontrol','show','step',f'{job}.{step}'],text=True)
    if 'JobState=RUNNING' not in job_info or 'State=RUNNING' not in step_info or 'gpu' not in step_info.lower():
        raise RuntimeError('Missing running GPU allocation')
    devices=jax.devices()
    if len(devices)!=1 or any(d.platform!='gpu' for d in devices):
        raise RuntimeError(f'Expected exactly one allocated GPU; saw {devices}')
    write_json(root/f'environment_{job}_{step}.json',dict(job=job_info,step=step_info,mask=mask,
        devices=[str(d) for d in devices],jax=jax.__version__,numpy=np.__version__,sources=source_hashes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--stage',choices=['baseline','core','refine','continue','replicate'],required=True)
    parser.add_argument('--frontier',type=int,default=20000)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--seconds',type=float,default=3500)
    parser.add_argument('--require-gpu',action='store_true')
    args=parser.parse_args()
    if args.frontier<20000 or args.frontier%20000:
        parser.error('Scientific runs require a multiple of 20,000 updates')
    if not jax.config.x64_enabled:
        parser.error('Set JAX_ENABLE_X64=true for this FP64 experiment')
    args.root.mkdir(parents=True,exist_ok=True)
    if args.require_gpu:
        verify_gpu(args.root)
    deadline=time.monotonic()+args.seconds
    groups=bundles(args.stage,args.root)
    # Stage identity and membership do not depend on measured scientific outcomes.
    for frontier in range(20000,args.frontier+1,20000):
        for i,(folder,config) in enumerate(groups):
            if i%args.workers!=args.worker:
                continue
            for degree in DEGREES:
                if not advance(folder,config,degree,frontier,deadline):
                    return


if __name__=='__main__':
    main()
