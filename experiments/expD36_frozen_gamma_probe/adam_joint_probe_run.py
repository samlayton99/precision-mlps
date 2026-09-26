"""FP64 joint Adam/GD probe with full-horizon schedules and raw output errors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


def recipes(config):
    return [dict(schedule=s, learning_rate=lr) for s in config['schedules'] for lr in config['learning_rates']]


def field(p, x, y):
    (a, b, c), d = p[:-1].reshape(3, -1), p[-1]
    z = x[:, None]*a+b
    h = jnp.tanh(z)
    exp = jnp.exp(-2*jnp.abs(z))
    sech2 = 4*exp/(1+exp)**2
    residual = h @ c+d-y
    weighted = residual[:, None]*sech2
    gradient = jnp.concatenate((c*(x @ weighted)/len(x), c*jnp.mean(weighted, axis=0),
                                h.T @ residual/len(x), jnp.array([jnp.mean(residual)])))
    error = jnp.sqrt(jnp.sum(residual**2)/jnp.sum(y**2))
    slope_rms = jnp.sqrt(jnp.mean(a*a))
    return gradient, error, slope_rms


batch_field = jax.vmap(jax.vmap(field, in_axes=(0,None,None)), in_axes=(0,None,None))


def make_chunk(x, y, config, size):
    x, y = jnp.asarray(x), jnp.asarray(y)
    optimizer = config.get('optimizer', 'adam')
    if optimizer not in {'adam', 'gd'}:
        raise ValueError(f'Unknown optimizer: {optimizer}')
    rec = recipes(config)
    rates = jnp.asarray([r['learning_rate'] for r in rec])[None, :, None]
    cosine = jnp.asarray([r['schedule']=='cosine' for r in rec])[None, :, None]
    def step(state, _):
        p, m, v, gradient, count = state
        t = count+1
        multiplier = jnp.where(cosine, .5*(1+jnp.cos(jnp.pi*count/config['horizon'])), 1.)
        if optimizer == 'adam':
            m = .9*m+.1*gradient
            v = .999*v+.001*gradient**2
            direction = (m/(1-.9**t))/(jnp.sqrt(v/(1-.999**t))+config['epsilon'])
        else:
            direction = gradient
        p = p-rates*multiplier*direction
        gradient, error, rms = batch_field(p, x, y)
        return (p,m,v,gradient,t), (error,rms)
    return jax.jit(lambda state: jax.lax.scan(step, state, None, length=size))


def initial_state(parameters, x, y, config):
    p = jnp.broadcast_to(jnp.asarray(parameters)[:,None,:], (len(parameters),len(recipes(config)),parameters.shape[1]))
    gradient, error, rms = batch_field(p,jnp.asarray(x),jnp.asarray(y))
    return (p,jnp.zeros_like(p),jnp.zeros_like(p),gradient,jnp.asarray(0,dtype=jnp.int64)), error, rms


def self_test(optimizer='adam'):
    rng = np.random.default_rng(891)
    x = np.linspace(-1,1,17)
    y = np.sin(3*x)
    ps = rng.normal(scale=.2,size=(2,13))
    cfg = dict(horizon=31,learning_rates=[.001,.02],epsilon=1e-8,schedules=['constant','cosine'],optimizer=optimizer)
    def numpy_field(p):
        a,b,c = p[:-1].reshape(3,-1)
        h = np.tanh(x[:,None]*a+b)
        exp = np.exp(-2*np.abs(x[:,None]*a+b))
        s = 4*exp/(1+exp)**2
        r = h@c+p[-1]-y
        grad = np.r_[c*(x@(r[:,None]*s))/len(x),c*np.mean(r[:,None]*s,axis=0),h.T@r/len(x),r.mean()]
        return grad, np.linalg.norm(r)/np.linalg.norm(y)
    def loss(p):
        a,b,c = p[:-1].reshape(3,-1)
        return .5*jnp.mean((jnp.tanh(jnp.asarray(x)[:,None]*a+b)@c+p[-1]-y)**2)
    np.testing.assert_allclose(np.asarray(field(jnp.asarray(ps[0]),jnp.asarray(x),jnp.asarray(y))[0]),np.asarray(jax.grad(loss)(jnp.asarray(ps[0]))),rtol=1e-12,atol=1e-14)
    state,_,_ = initial_state(ps,x,y,cfg)
    result,(errors,_) = make_chunk(x,y,cfg,10)(state)
    expected = np.zeros((10,2,4)); expected_p = np.zeros((2,4,13))
    for si in range(2):
        for ri,rec in enumerate(recipes(cfg)):
            p=ps[si].copy(); m=np.zeros_like(p); v=np.zeros_like(p)
            for n in range(10):
                g,_=numpy_field(p)
                factor=.5*(1+np.cos(np.pi*n/cfg['horizon'])) if rec['schedule']=='cosine' else 1.
                if optimizer == 'adam':
                    m=.9*m+.1*g; v=.999*v+.001*g*g
                    direction=(m/(1-.9**(n+1)))/(np.sqrt(v/(1-.999**(n+1)))+cfg['epsilon'])
                else:
                    direction=g
                p-=rec['learning_rate']*factor*direction
                expected[n,si,ri]=numpy_field(p)[1]
            expected_p[si,ri]=p
    np.testing.assert_allclose(np.asarray(result[0]),expected_p,rtol=2e-12,atol=2e-14)
    np.testing.assert_allclose(np.asarray(errors),expected,rtol=2e-12,atol=2e-14)
    part,_=make_chunk(x,y,cfg,4)(state); split,_=make_chunk(x,y,cfg,6)(part)
    np.testing.assert_allclose(np.asarray(split[0]),expected_p,rtol=2e-12,atol=2e-14)
    assert .5*(1+np.cos(np.pi))==0. and .5*(1+np.cos(0))==1.
    print(json.dumps(dict(self_test='passed',optimizer=optimizer,max_error_difference=float(np.max(np.abs(np.asarray(errors)-expected))))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path); parser.add_argument('--config',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--steps',type=int); parser.add_argument('--require-gpu',action='store_true'); parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--export-dir',type=Path)
    parser.add_argument('--export-every',type=int,default=250000)
    parser.add_argument('--backup-timeout',type=int,default=300)
    args=parser.parse_args()
    if args.require_gpu:
        for key in ['SLURM_JOB_ID','SLURM_STEP_ID','CUDA_VISIBLE_DEVICES']:
            if not os.environ.get(key): raise RuntimeError(f'Missing GPU allocation environment: {key}')
        if not any(os.environ.get(k) for k in ['SLURM_STEP_GPUS','SLURM_JOB_GPUS','SLURM_GPUS_ON_NODE']):
            raise RuntimeError('No Slurm GPU allocation reported')
        devices=jax.devices()
        if len(devices)!=1 or devices[0].platform!='gpu': raise RuntimeError(f'Expected one GPU, got {devices}')
    if args.self_test:
        self_test()
        self_test('gd')
        if args.input is None: return
    if not all([args.input,args.config,args.output]): parser.error('Training requires --input --config --output')
    config=json.loads(args.config.read_text()); steps=args.steps if args.steps is not None else config['horizon']
    if args.export_dir and (args.export_every <= 0 or args.export_every % 10000 or steps % 10000):
        raise ValueError('Durable export intervals and trial steps must be multiples of 10000')
    if not 0<steps<=config['horizon'] or set(config['schedules'])-{'constant','cosine'}: raise ValueError('Invalid steps/schedule')
    with np.load(args.input) as data:
        x=np.asarray(data['x'],dtype=np.float64); y=np.asarray(data['target'],dtype=np.float64); parameters=np.asarray(data['initial_parameters'],dtype=np.float64)
    if parameters.ndim!=2 or (parameters.shape[1]-1)%3 or x.shape!=y.shape or not all(np.all(np.isfinite(a)) for a in (x,y,parameters)) or np.linalg.norm(y)==0:
        raise ValueError('Invalid input shapes/values')
    width=(parameters.shape[1]-1)//3
    if width<1 or width&(width-1): raise ValueError('Total hidden width must be a power of two')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'relative_error.npy').exists(): raise FileExistsError('Use a fresh output directory')
    rec=recipes(config); shape=(steps+1,len(parameters),len(rec))
    errors=np.lib.format.open_memmap(args.output/'relative_error.npy',mode='w+',dtype=np.float64,shape=shape)
    rms=np.lib.format.open_memmap(args.output/'slope_rms.npy',mode='w+',dtype=np.float64,shape=shape)
    snapshot_steps=np.unique(np.r_[np.arange(0,steps+1,10000),steps])
    snapshots=np.lib.format.open_memmap(args.output/'parameter_checkpoints.npy',mode='w+',dtype=np.float64,shape=(len(snapshot_steps),len(parameters),len(rec),parameters.shape[1]))
    np.save(args.output/'checkpoint_steps.npy',snapshot_steps)
    metadata=dict(config=config,optimizer=config.get('optimizer','adam'),recipes=rec,planned_horizon=config['horizon'],actual_steps=steps,width=width,
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),parameter_layout='a[W],b[W],c[W],d',
                  trace_axes=['update','seed','recipe'],checkpoint_axes=['checkpoint','seed','recipe','parameter'],
                  metric='raw relative output L2 error',slope_metric='sqrt(mean(a**2)), physical slope not dimensionless bandwidth',
                  loss='0.5 * mean((sum(c*tanh(a*x+b))+d-target)**2)',devices=[str(d) for d in jax.devices()],
                  cosine_formula='eta0 * (1 + cos(pi*n/horizon))/2; n=0,...,horizon-1')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    state,e0,r0=initial_state(parameters,x,y,config); errors[0]=np.asarray(e0); rms[0]=np.asarray(r0); snapshots[0]=np.asarray(state[0])
    chunk_size=min(config.get('chunk_size',100),steps)
    if 10000%chunk_size and steps>chunk_size: raise ValueError('Chunk size must divide 10000')
    t0=time.perf_counter(); chunk=make_chunk(x,y,config,chunk_size).lower(state).compile(); compilation=time.perf_counter()-t0
    t0=time.perf_counter(); compute=0.; exported_through=-1
    for left in range(0,steps,chunk_size):
        size=min(chunk_size,steps-left); fn=chunk if size==chunk_size else make_chunk(x,y,config,size)
        tick=time.perf_counter(); state,(e,r)=fn(state); eh,rh=np.asarray(e),np.asarray(r); compute+=time.perf_counter()-tick
        count=left+size; errors[left+1:count+1]=eh; rms[left+1:count+1]=rh
        if count in snapshot_steps:
            snapshots[int(np.searchsorted(snapshot_steps,count))]=np.asarray(state[0]); errors.flush(); rms.flush(); snapshots.flush()
            p,m,v,gradient,_=state; temporary=args.output/'state.tmp.npz'
            np.savez(temporary,p=np.asarray(p),m=np.asarray(m),v=np.asarray(v),
                     gradient=np.asarray(gradient),count=count); temporary.replace(args.output/'state.npz')
            if args.export_dir and (count % args.export_every == 0 or count == steps):
                from figure4_durable import export_checkpoint
                export_checkpoint(args.output,args.export_dir,exported_through+1,count,
                                  errors,rms,snapshots,snapshot_steps,metadata,args.backup_timeout)
                exported_through=count
        if count%1000==0 or count==steps:
            print(json.dumps(dict(update=count,elapsed_seconds=time.perf_counter()-t0,min_error=float(np.nanmin(eh[-1])),max_error=float(np.nanmax(eh[-1])),finite_cases=int(np.isfinite(eh[-1]).sum()))),flush=True)
    summary=dict(completed_updates=steps,compilation_seconds=compilation,elapsed_seconds=time.perf_counter()-t0,synchronized_compute_seconds=compute,
                 updates_per_second=steps/compute,final_relative_error=np.asarray(errors[-1]).tolist(),final_slope_rms=np.asarray(rms[-1]).tolist())
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
