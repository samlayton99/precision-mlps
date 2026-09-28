"""Exact aggregate energy-concentration balances for the post-transient flow.

The four signed velocity components retain coarse compensation separately from
coarse disequilibrium. Run numerical work remotely with JAX FP64 enabled.
"""
from functools import partial

import jax
import jax.numpy as jnp

from . import mechanism_persistence_kernel as kernel
from . import population_balance_dynamics as balance

ORDERS = (4, 6, 10, 14)
COMPONENTS = ('generated', 'target', 'compensation', 'tracking')
LEDGER = tuple(f'logC{k}_{c}' for k in ORDERS for c in COMPONENTS)
LEDGER += tuple(f'{q}_{c}' for q in ('M', 'A') for c in COMPONENTS)
LEDGER += ('tracking_path', 'tracking_log6_positive')
LEDGER += tuple(f'discrete_logC{k}' for k in ORDERS)+('discrete_M', 'discrete_A')


def shape(p, k):
    hidden = p[:-1].reshape(3, -1)
    e = jnp.sum(hidden*hidden, axis=0)
    m = jnp.sum(e)
    return len(e)**(k/2-1)*jnp.sum(e**(k/2))/m**(k/2)


def score(p, k):
    """Gradient of log concentration, including its zero output-bias entry."""
    h = p[:-1].reshape(3, -1)
    e = jnp.sum(h*h, axis=0)
    weight = k*(e**(k/2-1)/jnp.sum(e**(k/2))-1/jnp.sum(e))
    return jnp.r_[(h*weight).reshape(-1), 0.]


def velocities(p, x, y, kind):
    state = kernel.decomposition(p, x, y)
    target = state['J'].T@state['yH']/len(x)
    raw = state['J'].T@state['eH']/len(x)
    generated = -raw-target
    compensation = raw-state['F']
    tracking = -state['R'] if kind == 'gd' else jnp.zeros_like(raw)
    return state, jnp.stack((generated, target, compensation, tracking))


def flux(p, x, y, kind):
    state, parts = velocities(p, x, y, kind)
    velocity = jnp.sum(parts, axis=0)
    w = (len(p)-1)//3
    rates = [parts@score(p,k) for k in ORDERS]
    rates += [2*parts[:,:-1]@p[:-1], 2*parts[:,:w]@p[:w]]
    rates += [jnp.array([jnp.linalg.norm(parts[3]),
                        jnp.maximum(parts[3]@score(p,6),0.)]), jnp.zeros(6)]
    return velocity, jnp.concatenate(rates)


@partial(jax.jit, static_argnames=('kind', 'steps'))
def advance(p, ledger, x, y, dt, steps, kind):
    def body(_, carry):
        p, ledger = carry
        v, r = flux(p,x,y,kind)
        if kind == 'gd':
            next_p = p+dt*v
            defects = [jnp.log(shape(next_p,k)/shape(p,k))-
                       dt*jnp.sum(r[4*i:4*i+4]) for i,k in enumerate(ORDERS)]
            w = (len(p)-1)//3
            defects += [dt*dt*jnp.sum(v[:-1]**2),dt*dt*jnp.sum(v[:w]**2)]
            increment = dt*r
            increment = increment.at[-6:].set(jnp.array(defects))
            return next_p, ledger+increment
        v2,r2 = flux(p+dt*v/2,x,y,kind)
        v3,r3 = flux(p+dt*v2/2,x,y,kind)
        v4,r4 = flux(p+dt*v3,x,y,kind)
        return p+dt*(v+2*v2+2*v3+v4)/6, ledger+dt*(r+2*r2+2*r3+r4)/6
    return jax.lax.fori_loop(0,steps,body,(p,ledger))


@partial(jax.jit, static_argnames=('kind',))
def diagnostics(p, x, y, kind='gd'):
    state, parts = velocities(p,x,y,kind)
    v = jnp.sum(parts,axis=0)
    _, rates = flux(p,x,y,kind)
    a,b,c = p[:-1].reshape(3,-1)
    w = len(a)
    e = a*a+b*b+c*c
    m = jnp.sum(e)
    norm = lambda z:jnp.sqrt(jnp.mean(z*z))
    ratio = lambda num,den:jnp.where(den>0,num/den,jnp.nan)
    jh = kernel._fine(state['J'],state['basis'])
    jnorm = jnp.sqrt(jnp.sum(jh*jh)/len(x))
    raw = state['J'].T@state['eH']/len(x)
    q = jnp.linalg.norm(state['F'])
    mu2,mu4,mu6 = (jnp.mean(x**k) for k in (2,4,6))
    s2,s3 = mu4-mu2**2,mu6-mu4**2/mu2
    j3 = jnp.sqrt(s2*jnp.sum(4*c*c*a*a*b*b+c*c*a**4+a**4*b*b)
                  +s3*jnp.sum(c*c*a**4+a**6/9))
    d3 = jnp.sqrt(8*s2/27+3*s3/16)
    root6,root10 = jnp.sqrt(jnp.sum(e**3)),jnp.sqrt(jnp.sum(e**5))
    result = {f'C{k}':shape(p,k) for k in (4,6,10,14,18,26)}
    result.update({f'rate_{key}':rates[i] for i,key in enumerate(LEDGER[:-6])})
    result.update(M=m,A=a@a,bias2=b@b,readout2=c@c,
        relative_error=norm(kernel.output(p,x)-y)/norm(y),
        fine_error=norm(state['eH']), fine_output=norm(state['fH']),target_norm=norm(y),
        q=q,tracking_norm=jnp.linalg.norm(state['R']),slope_rms=jnp.sqrt(a@a/w),
        logC6_rate=score(p,6)@v,
        logC6_acceleration=jax.jvp(lambda z:score(z,6)@jnp.sum(velocities(z,x,y,kind)[1],axis=0),
                                 (p,),(v,))[1],
        K6=shape(p,10)/shape(p,6)**2,
        score_norm2=score(p,6)@score(p,6),
        score_identity_error=score(p,6)@score(p,6)-36/m*(shape(p,10)/shape(p,6)**2-1),
        tangent_radial_error=score(p,6)@p,
        jacobian_exact=jnorm,jacobian_cubic=j3,
        jacobian_generic=balance.C3*root6,
        jacobian_projected=d3*root6+balance.C5*root10,
        jacobian_cubic_concentration=d3*root6,
        polynomial_jacobian_remainder=balance.C5*root10,
        fine_alignment=ratio(jnp.linalg.norm(raw),jnorm*norm(state['eH'])),
        projection_fraction=ratio(q,jnp.linalg.norm(raw)),
        slope_fraction=ratio(jnp.linalg.norm(state['F'][:w]),q),
        mass_radial_fraction=ratio(-p[:-1]@state['F'][:-1],jnp.sqrt(m)*q),
        slope_radial_fraction=ratio(-a@state['F'][:w],jnp.linalg.norm(a)*jnp.linalg.norm(state['F'][:w])),
        coarse_min=jnp.linalg.eigvalsh(state['gram'])[0],
        velocity_identity_error=jnp.linalg.norm(v+(state['g'] if kind=='gd' else state['F'])))
    return result
