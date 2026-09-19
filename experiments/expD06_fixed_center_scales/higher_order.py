"""Joint damped GN and pinned SSBroyden; accepted physical steps are explicit."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from . import core, difference_training as training, run

SSB_COMMIT = "4c87785c68f0fec6b09000f474daef76fb181eea"
OPTIMISTIX_COMMIT = "8cd4931713658f8dfe4423ead6f11b348b675540"
SSB_SOURCE_SHA256 = "8487b31d74e4dc78b35519c103d1fb158394f4661b69a3a38c1434bd672e7d05"
STATUS = {0: "continuing", 1: "nonfinite", 2: "search_failed", 3: "no_representable_step"}


def save_state(path,state,step):
    # Pickling Lineax's static operator tags breaks their identity on reload.
    # Save array leaves and reconstruct against the freshly initialized solver.
    run.save_state(path,jax.tree.leaves(state),step)


def load_state(path,template):
    leaves,step=run.load_state(path)
    expected,structure=jax.tree.flatten(template)
    if len(leaves)!=len(expected) or any(a.shape!=b.shape or a.dtype!=b.dtype for a,b in zip(leaves,expected)):
        raise ValueError("Checkpoint does not match this solver's array structure")
    return jax.tree.unflatten(structure,leaves),step


def readout_map(g, coordinates):
    if coordinates == "parameter_scale":
        return np.diag(g.alpha)
    if coordinates != "parameter_differences":
        raise ValueError(coordinates)
    transform = np.zeros((g.width+1, g.width+1))
    transform[0,0] = g.alpha[0]
    transform[1:,1:] = (np.eye(g.width)-np.eye(g.width,k=-1))*np.cumsum(g.alpha[1:])[None,:]
    return transform


def initial_parameters(g, seed, coordinates):
    c,gamma = core.initial_physical(g,seed,"xavier_a_reference")
    return jnp.asarray(np.r_[training.encode(c,g,coordinates),g.h*gamma])


def physical(z,g,coordinates):
    return training.decode(z[:g.width+1],g,coordinates),z[g.width+1:]/g.h


def problem(n,coordinates,samples=16):
    """Same physical forward evaluation in both maps; analytic stable Jacobian."""
    g=core.geometry(n);x=jnp.linspace(-1,1,samples*n+1);y=core.target(x,"sine")
    distances=x[:,None]-jnp.asarray(g.centers);root=np.sqrt(x.size)
    def residual(z):
        c,gamma=physical(z,g,coordinates)
        return (c[0]+core.tanh(distances*gamma)@c[1:]-y)/root
    def residual_jacobian(z):
        c,gamma=physical(z,g,coordinates);arg=distances*gamma;phi=core.tanh(arg)
        r=(c[0]+phi@c[1:]-y)/root
        if coordinates=="parameter_scale":
            readout=phi*jnp.asarray(g.alpha[1:])
        else:
            readout=jnp.concatenate((phi[:,:-1]-phi[:,1:],phi[:,-1:]),axis=1)
            readout=readout*jnp.asarray(np.cumsum(g.alpha[1:]))
        geom=c[1:]*distances*core.sech_squared(arg)/g.h
        jac=jnp.concatenate((jnp.full((x.size,1),g.alpha[0]),readout,geom),axis=1)/root
        return r,jac
    def loss(z):
        r=residual(z)
        return .5*jnp.dot(r,r)
    return g,residual,residual_jacobian,loss


def augmented_qr(jac,residual,damping):
    """Solve the regularized linearized residual without forming J.T @ J."""
    p=jac.shape[1]
    augmented=jnp.concatenate((jac,jnp.sqrt(damping)*jnp.eye(p)),axis=0)
    rhs=jnp.concatenate((-residual,jnp.zeros(p)))
    q,upper=jnp.linalg.qr(augmented,mode="reduced")
    return solve_triangular(upper,q.T@rhs,lower=False)


def gn_step(residual,residual_jacobian,physical_fn,damping_floor=1e-24,max_trials=40):
    """One accepted LM/GN step, or an explicit terminal failure (no fallback)."""
    def step(state):
        z=state["z"];r,jac=residual_jacobian(z);grad=jac.T@r
        scale=jnp.max(jnp.sum(jac*jac,axis=0));floor=damping_floor*scale
        mu=jnp.maximum(jnp.where(state["count"]==0,1e-3*scale,state["damping"]),floor)
        c0,gamma0=physical_fn(z)
        start_bad=~(jnp.all(jnp.isfinite(r)) & jnp.all(jnp.isfinite(jac)) & jnp.isfinite(mu))
        start=dict(attempts=jnp.array(0),accepted=jnp.array(False),status=jnp.where(start_bad,1,0),
                   damping=mu,z=z,r=r,ratio=jnp.array(0.),predicted=jnp.array(0.),
                   actual=jnp.array(0.),linear_residual=jnp.array(0.),floor_hit=jnp.array(False))
        def body(trial):
            mu=jnp.maximum(trial["damping"],floor)
            delta=augmented_qr(jac,r,mu);candidate=z+delta;next_r=residual(candidate)
            jd=jac@delta;predicted=-grad@delta-.5*(jd@jd)
            dr=next_r-r;actual=-(r@dr+.5*(dr@dr))
            ratio=jnp.where(predicted>0,actual/predicted,-jnp.inf)
            cp,gp=physical_fn(candidate)
            changed=jnp.any(cp!=c0)|jnp.any(gp!=gamma0)
            finite=jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(next_r)) & jnp.all(jnp.isfinite(delta))
            accepted=finite & changed & (predicted>0) & (ratio>1e-4)
            status=jnp.where(~finite,1,jnp.where(~changed,3,0))
            normal=jac.T@(jd+r)+mu*delta
            norm=jnp.linalg.norm(grad)
            lin_error=jnp.where(norm>0,jnp.linalg.norm(normal)/norm,jnp.linalg.norm(normal))
            next_mu=jnp.where(accepted,jnp.where(ratio>.75,mu/3,mu),mu*10)
            return dict(attempts=trial["attempts"]+1,accepted=accepted,status=status,
                        damping=next_mu,z=candidate,r=next_r,ratio=ratio,predicted=predicted,
                        actual=actual,linear_residual=lin_error,floor_hit=trial["floor_hit"]|(mu<=floor))
        trial=jax.lax.while_loop(lambda t:(~t["accepted"])&(t["status"]==0)&(t["attempts"]<max_trials),body,start)
        status=jnp.where((~trial["accepted"])&(trial["status"]==0),2,trial["status"])
        z1=jnp.where(trial["accepted"],trial["z"],z)
        next_state=dict(z=z1,damping=trial["damping"],count=state["count"]+trial["accepted"],status=status,
                        function_evaluations=state["function_evaluations"]+1+trial["attempts"],
                        gradient_evaluations=state["gradient_evaluations"]+1,
                        jacobian_evaluations=state["jacobian_evaluations"]+1)
        evidence=dict(loss=.5*(r@r),gradient=grad,delta=z1-z,accepted=trial["accepted"],
                      attempts=trial["attempts"],ratio=trial["ratio"],predicted=trial["predicted"],
                      actual=trial["actual"],linear_residual=trial["linear_residual"],
                      damping=trial["damping"],guard_active=trial["floor_hit"],
                      trial_z=trial["z"],trial_mse=trial["r"]@trial["r"],curvature=jnp.array(0.),step_size=jnp.array(1.))
        return next_state,evidence
    return jax.jit(step)


def gn_initial(z):
    return dict(z=z,damping=jnp.array(0.),count=jnp.array(0),status=jnp.array(0),
                function_evaluations=jnp.array(0),gradient_evaluations=jnp.array(0),jacobian_evaluations=jnp.array(0))


@lru_cache(maxsize=12)
def ssb_module(source,curvature_epsilon):
    """Load the pinned source with one auditable threshold substitution.

    The module name contains the threshold so complete Equinox states can be
    unpickled after recreating the same solver. No upstream file is modified.
    """
    path=Path(source)/"ssbrodyen_family.py"
    code=path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SSB_SOURCE_SHA256:
        raise ValueError("SSBroyden source differs from the pinned implementation")
    needle="inner_nonzero = inner > jnp.finfo(inner.dtype).eps"
    if code.count(needle)!=1:
        raise ValueError("Unexpected SSBroyden source; cannot apply the recorded guard patch")
    code=code.replace(needle,"inner_nonzero = inner > CURVATURE_EPSILON")
    name="precision_ssbroyden_"+hashlib.sha256((code+repr(curvature_epsilon)).encode()).hexdigest()[:12]
    module=types.ModuleType(name);module.__file__=str(path)
    module.CURVATURE_EPSILON=float(curvature_epsilon)
    sys.modules[name]=module
    exec(compile(code,str(path),"exec",dont_inherit=True),module.__dict__)
    return module


def ssb_solver(source,curvature_epsilon=1e-24,search_threshold=1e-15):
    from optimistix._solver.zoom import Zoom
    module=ssb_module(str(source),curvature_epsilon)
    search=Zoom(c1=1e-4,c2=.9,c3=1e-6,max_stepsize=1.,initial_guess_strategy="one",
                min_interval_length=search_threshold,min_stepsize=search_threshold,line_search_max_steps=64)
    return module.SSBroyden(rtol=0.,atol=0.,search=search)


def ssb_initial(solver,loss,z):
    fn=lambda p,args:(loss(p),None)
    state=solver.init(fn,z,None,{},jax.ShapeDtypeStruct((),jnp.float64),None,frozenset())
    return dict(z=z,solver=state,count=jnp.array(0),status=jnp.array(0),
                function_evaluations=jnp.array(0),gradient_evaluations=jnp.array(0),jacobian_evaluations=jnp.array(0))


def ssb_step(solver,loss,physical_fn,curvature_epsilon=1e-24):
    """Count changed accepted steps, not the library's initialization/search steps."""
    import equinox as eqx
    import optimistix as optx
    fn=lambda p,args:(loss(p),None)
    def step(state):
        origin=state["z"];old=state["solver"];c0,g0=physical_fn(origin)
        initial_loss=loss(origin)
        # The first gradient in the library state is a placeholder until priming.
        grad0=jax.grad(loss)(origin)
        init=(origin,old,jnp.array(0),jnp.array(False),jnp.array(0),jnp.array(0.),origin)
        def body(carry):
            p,st,calls,accepted,status,trial_loss,trial_z=carry
            trial_z=st.y_eval
            trial_loss=loss(st.y_eval)
            next_p,next_st,_=solver.step(fn,p,None,{},st,frozenset())
            cp,gp=physical_fn(next_p)
            changed=jnp.any(cp!=c0)|jnp.any(gp!=g0)
            library_accept=next_st.num_accepted_steps>st.num_accepted_steps
            initialized=~st.first_step
            finite=jnp.isfinite(trial_loss)&jnp.all(jnp.isfinite(next_p))&jnp.all(jnp.isfinite(next_st.f_info.grad))
            finite &= jnp.all(jnp.isfinite(next_st.f_info.hessian_inv.pytree))
            search_bad=next_st.search_state.failed|(next_st.result!=optx.RESULTS.successful)
            no_step=initialized & library_accept & (~changed)
            status=jnp.where(~finite,1,jnp.where(search_bad,2,jnp.where(no_step,3,0)))
            accepted=library_accept & changed & (status==0)
            return next_p,next_st,calls+1,accepted,status,trial_loss,trial_z
        p,st,calls,accepted,status,trial_loss,trial_z=jax.lax.while_loop(
            lambda t:(~t[3])&(t[4]==0)&(t[2]<66),body,init)
        status=jnp.where((~accepted)&(status==0),2,status)
        z1=jnp.where(accepted,p,origin)
        curvature=(p-origin)@(st.f_info.grad-grad0)
        # Each search call evaluates loss+gradient in the library plus the explicit
        # loss-only finite check. The initial loss/gradient audit adds two calls.
        nxt=dict(z=z1,solver=st,count=state["count"]+accepted,status=status,
                 function_evaluations=state["function_evaluations"]+2*calls+2,
                 gradient_evaluations=state["gradient_evaluations"]+calls+1,
                 jacobian_evaluations=state["jacobian_evaluations"])
        ev=dict(loss=initial_loss,gradient=grad0,delta=z1-origin,accepted=accepted,attempts=calls,
                ratio=jnp.array(0.),predicted=jnp.array(0.),actual=initial_loss-st.f_info.f,
                linear_residual=jnp.array(0.),damping=jnp.array(0.),
                guard_active=(curvature<=curvature_epsilon)&accepted,curvature=curvature,
                step_size=st.search_state.stepsize,trial_z=trial_z,trial_mse=2*trial_loss)
        return nxt,ev
    return eqx.filter_jit(step)
