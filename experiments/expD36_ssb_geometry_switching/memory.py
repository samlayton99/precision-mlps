"""Replay common observed secants under different priors; no live model reset."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD35_optimization_exploration import core, run as old, ssb
from experiments.expD06_fixed_center_scales import higher_order as higher


def update(solver, h, s, y, first):
    """General secant formula: the original line-search shortcut is invalid here."""
    inner=s@y;hy=h@y;yhy=y@hy
    b=(s@jnp.linalg.solve(h,s))/inner
    a=b*yhy/inner-1;hk=yhy/inner
    theta=solver._compute_thetak(a,b,hk,first)
    tau=solver._compute_tauk(theta,a,b,hk,first,len(s))
    proposed=solver._invhessian_update_term(hessian_pytree=h,y_diff=s,grad_diff=y,
        Hy=hy,inner=inner,yHy=yhy,rho=1/inner,thetak=theta,tauk=tau,ak=a,bk=b,hk=hk)
    valid=(inner>1e-30)&(b>0)&(yhy>0)&jnp.isfinite(tau)&(tau>0)&jnp.all(jnp.isfinite(proposed))
    # Record invalid updates, rather than silently manufacturing a new prior.
    return jnp.where(valid,proposed,h),valid


def capture(config,source,state,length):
    _,loss,_=ssb.problem(config)
    one=ssb.kernel(config['n'],config['coordinates'],config['target'],source,1e-30,1e-15,'non_descent',1000,1)
    @eqx.filter_jit
    def run(st):
        def step(current,_):
            before=current['z'];g0=jax.grad(loss)(before)
            nxt,trace=one(current);g1=jax.grad(loss)(nxt['z'])
            return nxt,(nxt['z']-before,g1-g0,g1,trace[0])
        return jax.lax.scan(step,st,None,length=length)
    return run(state)


def replay(solver,priors,steps,differences,gradients,split,first=True):
    @jax.jit
    def run(h0):
        def step(carry,inputs):
            h,initial,active=carry;s,y,g=inputs
            nxt,valid=jax.vmap(lambda hh:update(solver,hh,s,y,initial))(h)
            active=active&valid
            nxt=jnp.where(active[:,None,None],nxt,h)
            directions=-jnp.einsum('aij,j->ai',nxt,g)
            directions=directions/jnp.linalg.norm(directions,axis=1)[:,None]
            cosine=directions[0]@directions[1]
            geometry=jnp.linalg.norm(directions[:,split:],axis=1)
            secant=jnp.linalg.norm(jnp.einsum('aij,j->ai',nxt,y)-s,axis=1)/jnp.maximum(jnp.linalg.norm(s),1e-300)
            return (nxt,jnp.array(False),active),(cosine,geometry,secant,active)
        return jax.lax.scan(step,(h0,jnp.array(first),jnp.ones(2,dtype=bool)),(steps,differences,gradients))
    return run(jnp.stack(priors))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);args=p.parse_args()
    old.verify_gpu(args.root)
    cases=[]
    for path in sorted(args.root.glob('*/case.json')):
        c=json.loads(path.read_text())
        if c['policy']=='baseline' and c['seed']==0 and 'parent' not in c and c['n']==128 and c.get('implementation')=='primed_guard_v2':cases.append((path.parent,c))
    for index,(folder,c) in enumerate(cases):
        if index%args.workers!=args.worker:continue
        g,loss,_=ssb.problem(c);solver=higher.ssb_solver(args.source,1e-30,1e-15,'accepted_step')
        template=higher.ssb_initial(solver,loss,core.initialize(c)['z'])
        for at in (0,5000):
            state=ssb.load_solver(folder/f'solver_{at:09d}.npz',template)
            end,(s,y,grad,trace)=capture(c,args.source,state,512)
            native=state['solver'].f_info.hessian_inv.pytree
            # Both replay priors see exactly the same observed secants.
            # At initialization compare readout/geometry ratios; later restore I.
            other=jnp.diag(jnp.r_[jnp.full(g.width+1,100.),jnp.full(g.width,.01)]) if at==0 else jnp.eye(len(state['z']))
            final,rows=replay(solver,(native,other),s,y,grad,g.width+1,first=at==0)
            cosine,geometry,secant,active=map(np.asarray,rows)
            out=folder/f'memory_{at:09d}.npz'
            old.save(out,steps=np.asarray(s),gradient_differences=np.asarray(y),gradients=np.asarray(grad),
                trace=np.asarray(trace),direction_cosine=cosine,geometry_fraction=geometry,secant_error=secant,valid=active)
            old.write_json(out.with_suffix('.json'),dict(start=at,accepted=int(end['count'])-at,status=int(end['status']),
                replay_valid_steps=active.sum(axis=0).tolist(),cosine_at_1_10_100_512=cosine[[0,9,99,511]].tolist(),
                source_commit=os.environ.get('EXPLORATION_SOURCE_COMMIT'),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                meaning='common-secant diagnostic replay; no counterfactual training trajectory'))
            print(out,flush=True)


if __name__=='__main__':main()
