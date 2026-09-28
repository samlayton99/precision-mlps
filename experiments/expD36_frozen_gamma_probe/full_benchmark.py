"""Measure full campaign batch shapes before committing its compute budget."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import svd
from . import core, full_core as f, full_kernels as k, train


def frozen(n, batch, columns, optimizer, neighbor, root, steps=2000):
    start=time.monotonic()
    g=core.geometry(n); x=np.linspace(-1,1,16*n+1)
    scale, _=f.map_spec(g,'collective_neighbor' if neighbor else 'raw')
    js=np.stack([f.design_from_physical(core.design(x,g.centers,gamma),scale,neighbor)
                 for gamma in np.geomspace(4,64,batch)])
    y=np.broadcast_to(np.column_stack([f.target(x,f.config()['targets'][i%5]) for i in range(columns)])
                      /np.sqrt(len(x)),(batch,len(x),columns)).copy()
    state=k.initialize(np.zeros((batch,g.width+1,columns)),6)
    rates=np.full((batch,columns),1e-3)
    args=(jnp.asarray(js),jnp.asarray(y),jnp.asarray(np.broadcast_to(scale,(batch,len(scale))).copy()),
          jnp.asarray(rates),jnp.full((batch,columns),1e-12),jnp.array(f.config()['tolerances']))
    kernel=k.make_chunk(optimizer,neighbor,500)
    compiled=time.monotonic(); state,trace=kernel(state,*args); jax.block_until_ready(state)
    first=time.monotonic()-compiled
    steady=time.monotonic()
    for _ in range(steps//500):
        state,trace=kernel(state,*args); jax.block_until_ready(state)
        _=np.asarray(trace)
    elapsed=time.monotonic()-steady
    io=time.monotonic()
    core.save_arrays(root/'benchmark_trace.npz',trace=np.asarray(trace))
    io=time.monotonic()-io
    result=dict(kind='frozen',n=n,batch=batch,columns=columns,optimizer=optimizer,neighbor=neighbor,
        first_chunk_seconds=first,steps=steps,steady_seconds=elapsed,seconds_per_update=elapsed/steps,
        chunk_write_seconds=io,total_seconds=time.monotonic()-start)
    print(json.dumps(result),flush=True)
    return result


def joint(n,root,steps=500):
    g=core.geometry(n); x=jnp.linspace(-1,1,16*n+1); y=f.target(x,'sine_mix_2_6_10',jnp)
    state=k.joint_initial(g.width,list(range(5))); kernel=k.make_joint_chunk(100)
    start=time.monotonic(); state,trace=kernel(state,x,y); jax.block_until_ready(state)
    first=time.monotonic()-start; start=time.monotonic()
    for _ in range(steps//100):
        state,trace=kernel(state,x,y); jax.block_until_ready(state)
    result=dict(kind='joint',n=n,seeds=5,first_chunk_seconds=first,steps=steps,
                seconds_per_update=(time.monotonic()-start)/steps)
    print(json.dumps(result),flush=True)
    return result


def cpu(n):
    g=core.geometry(n); x=np.linspace(-1,1,16*n+1)
    j=core.design(x,g.centers,4)
    start=time.monotonic(); raw=core.polynomial_transform(x,256); jh=core.transform(raw,j)
    projection=time.monotonic()-start
    start=time.monotonic(); svd(j,full_matrices=False); factor=time.monotonic()-start
    start=time.monotonic(); svd(jh[33:],compute_uv=False); subspace=time.monotonic()-start
    result=dict(kind='cpu',n=n,projection_seconds=projection,svd_seconds=factor,subspace_seconds=subspace)
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args(); args.root.mkdir(parents=True,exist_ok=True)
    results=[]
    if args.cpu:
        for n in [512,1024]:
            results.append(cpu(n)); core.write_json(args.root/'benchmark_cpu.json',results)
    else:
        train.verify_gpu(args.root,'benchmark')
        for n,b,c,opt,neighbor in [(512,11,5,'gd',False),(512,11,50,'adam',False),
            (512,11,10,'adam',False),(512,3,20,'gd',True),(512,3,20,'adam',True),
            (1024,3,2,'gd',True),(1024,3,20,'adam',True),(1024,3,4,'adam',True),
            (512,3,4,'gd',False),(512,3,1,'adam',False)]:
            results.append(frozen(n,b,c,opt,neighbor,args.root))
            core.write_json(args.root/'benchmark_gpu.json',results)
        for n in [128,256,512,1024]:
            results.append(joint(n,args.root)); core.write_json(args.root/'benchmark_gpu.json',results)


if __name__=='__main__':
    main()
