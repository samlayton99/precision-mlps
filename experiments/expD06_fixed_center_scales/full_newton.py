"""Dense full-Hessian trust-region Newton for the fixed-center experiment."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import core, higher_order as higher


def derivatives(g,coordinates,residual_jacobian,samples=16):
    """Return residual, gradient, exact Hessian, and the GN matrix in native units."""
    x=jnp.linspace(-1,1,samples*g.n+1);root=np.sqrt(x.size)
    slope_scale=1. if coordinates=='physical' else 1/g.h
    distances=(x[:,None]-jnp.asarray(g.centers))*slope_scale
    transform=jnp.asarray(higher.readout_map(g,coordinates));split=g.width+1
    def evaluate(z):
        c,gamma=higher.physical(z,g,coordinates)
        arg=(x[:,None]-jnp.asarray(g.centers))*gamma
        sech=core.sech_squared(arg)
        r,jac=residual_jacobian(z);gn=jac.T@jac
        mixed=(distances*sech/root).T@r
        second=c[1:]*((-2*distances**2*core.tanh(arg)*sech/root).T@r)
        correction=transform[1:].T*mixed
        h=gn.at[:split,split:].add(correction).at[split:,:split].add(correction.T)
        h=h.at[split:,split:].add(jnp.diag(second))
        return r,jac.T@r,(h+h.T)/2,gn
    return evaluate


def eigen_step(values,vectors,gradient,radius):
    """Global quadratic trust-region minimizer, including the indefinite hard case.

    The scalar root uses an offset above the spectral lower bound, avoiding
    cancellation in the smallest shifted eigenvalue. No eigenvalue cutoff or
    fixed denominator epsilon discards weak directions.
    """
    projected=vectors.T@gradient
    lower=jnp.maximum(0.,-values[0]);base=values+lower
    positive=base>0
    particular=-jnp.where(positive,projected/jnp.where(positive,base,1.),0.)
    compatible=jnp.all(jnp.where(positive,0.,projected)==0)
    inside=compatible & (jnp.linalg.norm(particular)<=radius)
    hard=inside & (values[0]<0)
    addition=jnp.sqrt(jnp.maximum(0.,radius**2-jnp.dot(particular,particular)))
    special=particular.at[0].add(jnp.where(hard,addition,0.))
    upper=jnp.maximum(jnp.linalg.norm(projected)/radius,jnp.finfo(gradient.dtype).tiny)
    def bisect(_,bounds):
        lo,hi=bounds;mid=(lo+hi)/2
        p=-projected/(base+mid)
        outside=jnp.linalg.norm(p)>radius
        return jnp.where(outside,mid,lo),jnp.where(outside,hi,mid)
    lo,hi=jax.lax.fori_loop(0,100,bisect,(jnp.array(0.),upper))
    regular=-projected/(base+hi)
    p=jnp.where(inside,special,regular)
    shift=jnp.where(inside,lower,lower+hi)
    return vectors@p,shift,hard


def initial(z):
    state=higher.gn_initial(z)
    state.update(radius=jnp.array(1.),hessian_evaluations=jnp.array(0))
    return state


def step(residual,evaluate,physical_fn,max_trials=40):
    """One accepted full-Hessian step, or explicit failure; never use a GN fallback."""
    def advance(state):
        z=state['z'];r,grad,h,gn=evaluate(z)
        values,vectors=jnp.linalg.eigh(h)
        cp,gp=physical_fn(z)
        finite=jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(grad)) & jnp.all(jnp.isfinite(r))
        trial=dict(attempts=jnp.array(0),radius=state['radius'],accepted=jnp.array(False),
                   status=jnp.where(finite,0,1),z=z,r=r,delta=jnp.zeros_like(z),shift=jnp.array(0.),
                   ratio=jnp.array(0.),predicted=jnp.array(0.),actual=jnp.array(0.),
                   solve_error=jnp.array(0.),hard=jnp.array(False),used_radius=state['radius'])
        def body(t):
            radius=t['radius'];delta,shift,hard=eigen_step(values,vectors,grad,radius)
            candidate=z+delta;rr=residual(candidate)
            predicted=-grad@delta-.5*delta@(h@delta)
            dr=rr-r;actual=-(r@dr+.5*(dr@dr))
            ratio=jnp.where(predicted>0,actual/predicted,-jnp.inf)
            c1,g1=physical_fn(candidate);changed=jnp.any(c1!=cp)|jnp.any(g1!=gp)
            finite=jnp.all(jnp.isfinite(rr)) & jnp.all(jnp.isfinite(delta))
            accepted=finite & changed & (predicted>0) & (ratio>.1)
            radius1=jnp.where(ratio<.25,radius/4,jnp.where((ratio>.75)&(jnp.linalg.norm(delta)>=.9*radius),jnp.minimum(2*radius,1000.),radius))
            normal=h@delta+shift*delta+grad
            denominator=(jnp.max(jnp.abs(values))+shift)*jnp.linalg.norm(delta)+jnp.linalg.norm(grad)
            error=jnp.where(denominator>0,jnp.linalg.norm(normal)/denominator,jnp.linalg.norm(normal))
            return dict(attempts=t['attempts']+1,radius=radius1,accepted=accepted,
                        status=jnp.where(~finite,1,jnp.where(~changed,3,0)),z=candidate,r=rr,delta=delta,
                        shift=shift,ratio=ratio,predicted=predicted,actual=actual,solve_error=error,hard=hard,used_radius=radius)
        trial=jax.lax.while_loop(lambda t:(~t['accepted'])&(t['status']==0)&(t['attempts']<max_trials),body,trial)
        status=jnp.where((~trial['accepted'])&(trial['status']==0),2,trial['status'])
        z1=jnp.where(trial['accepted'],trial['z'],z)
        nxt=dict(z=z1,radius=trial['radius'],damping=trial['shift'],count=state['count']+trial['accepted'],status=status,
                 function_evaluations=state['function_evaluations']+1+trial['attempts'],
                 gradient_evaluations=state['gradient_evaluations']+1,jacobian_evaluations=state['jacobian_evaluations']+1,
                 hessian_evaluations=state['hessian_evaluations']+1)
        ev=dict(loss=.5*(r@r),gradient=grad,delta=z1-z,accepted=trial['accepted'],attempts=trial['attempts'],
                ratio=trial['ratio'],predicted=trial['predicted'],actual=trial['actual'],linear_residual=trial['solve_error'],
                damping=trial['shift'],guard_active=jnp.array(False),trial_z=trial['z'],trial_mse=trial['r']@trial['r'],
                curvature=values[0],step_size=jnp.array(1.),radius=trial['used_radius'],hessian_min=values[0],
                hessian_max=values[-1],negative_eigenvalues=jnp.sum(values<0),hard_case=trial['hard'],
                residual_curvature_norm=jnp.linalg.norm(h-gn),gn_norm=jnp.linalg.norm(gn))
        return nxt,ev
    return jax.jit(advance)
