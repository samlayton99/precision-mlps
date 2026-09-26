"""Exact coarse-response diagnostics and paired full-batch update interventions."""
from functools import partial

import jax
import jax.numpy as jnp

from . import adam_forces as af, population_adam as pa

ARMS = ('native', 'fine10', 'balanced', 'balanced10', 'frozen10', 'frozen_balanced10')
BURST = ('error', 'A', 'z0', 'z1', 'fine_norm', 'tracking_norm', 'coarse_fine0',
         'coarse_fine1', 'fine_A', 'tracking_A', 'step_A', 'shadow_gain', 'delta_norm')


def divide(a, b):
    return jnp.where(b > 0, a / jnp.where(b > 0, b, 1.), 0.)


def balanced(u, jc, inverse):
    gram = (jc * inverse) @ jc.T
    ev = jnp.linalg.eigvalsh(gram)
    resolved = ev[0] > 64 * jnp.finfo(u.dtype).eps * jnp.maximum(1., ev[-1])
    correction = inverse * (jc.T @ jnp.linalg.solve(jnp.where(resolved, gram, jnp.eye(2)), jc @ u))
    return u - correction, resolved


def proposal(state, x, y, config, arm=0, tracking_factor=1.):
    eta, b1, b2, eps, adaptive = config
    g, r, jc, ec = af.field(state['p'], x, y)
    channels, info = af.split(g, jc, ec)
    count = state['count'] + 1
    m, v, cm, mh, ch, inverse = af.moments(g, channels, state['m'], state['v'],
        state['cm'], count, b1, b2, eps, adaptive)
    native = -eta * inverse * mh
    original = -eta * ch * inverse
    frozen = arm >= 4
    metric = jnp.where(frozen, state['fixed_inverse'], inverse)
    candidate = -eta * metric * ch[0]
    projected, resolved = balanced(candidate, jc, metric)
    use_balance = (arm == 2) | (arm == 3) | (arm == 5) | (arm == 8)
    gain = jnp.where((arm == 1) | (arm == 3) | (arm == 4) | (arm == 5), 10., 1.)
    target = gain * jnp.linalg.norm(candidate[:-1])
    direction = jnp.where(use_balance, projected, candidate)
    norm = jnp.linalg.norm(direction[:-1])
    altered = gain * candidate
    altered = jnp.where(use_balance, direction * divide(target, norm), altered)
    fine = jnp.where(arm == 0, original[0], altered)
    # Secondary controls: no fine motion, or the fork denominator at unit gain.
    fine = jnp.where(arm == 6, jnp.zeros_like(fine), fine)
    target = jnp.where(arm == 6, 0., target)
    tracking = tracking_factor * original[1]
    delta = native + (fine - original[0]) + (tracking_factor - 1.) * original[1]
    delta = jnp.where((arm == 0) & (tracking_factor == 1.), native, delta)
    sv = b2 * state['shadow_v'] + (1 - b2) * (g - .9 * channels[1])**2
    si = 1 / (jnp.sqrt(sv / (1 - b2**count)) + eps)
    shadow_gain = divide(jnp.linalg.norm((-eta * ch[0] * si)[:-1]),
                         jnp.linalg.norm(original[0, :-1]))
    valid = info['resolved'] & (jnp.logical_not(use_balance) | (resolved & ((norm > 0) | (target == 0))))
    return dict(g=g, r=r, jc=jc, ec=ec, z=info['z'], star=ec-info['z'],
        channels=channels, m=m, v=v, cm=cm, count=count, inverse=inverse,
        shadow_v=sv, shadow_gain=shadow_gain, fine=fine, tracking=tracking,
        unknown=original[2], native_fine=original[0], native_tracking=original[1],
        delta=delta, valid=valid, target_norm=jnp.where(arm == 0,
            jnp.linalg.norm(original[0, :-1]), target), balance=use_balance,
        zero_candidate=(target > 0) & (norm == 0) & use_balance)


def initialize(saved, x, y, config):
    p = jnp.asarray(saved['p']); w = (len(p)-1)//3
    cm = jnp.asarray(saved['cm'])
    if cm.shape[0] == 6:
        cm = jnp.stack((cm[0]+cm[1]+cm[2]+cm[5], cm[3], cm[4]))
    state = {k:jnp.asarray(saved[k]) for k in ('p','m','v','count')}
    state.update(cm=cm, fixed_inverse=jnp.ones_like(p), shadow_v=jnp.asarray(saved.get('alternative_v',saved['v'])))
    z = proposal(state,x,y,config)
    state.update(fixed_inverse=z['inverse'],p0=p,offset=jnp.array(0),alive=jnp.array(True),
        failure=jnp.array(0),accounts=jnp.zeros(6),paths=jnp.zeros(4),fine_sum=jnp.zeros(w),
        window_sum=jnp.zeros(w),window_path=jnp.array(0.),last_coherence=jnp.array(0.),
        checks=jnp.zeros(4),stats=jnp.zeros(9),minimum_error=jnp.array(jnp.inf),
        hits=jnp.zeros(4,dtype=jnp.int64),last_error=jnp.array(jnp.inf),loss_increases=jnp.array(0),
        previous_ec=z['ec'],previous_z=z['z'],previous_star=z['star'],previous_linear=jnp.zeros(2))
    return state


def _step(state,x,y,config,arm,tracking_factor):
    z=proposal(state,x,y,config,arm,tracking_factor)
    p=state['p']; w=(len(p)-1)//3; a=p[:w]; delta=z['delta']
    fine,tracking=z['fine'],z['tracking']
    error=jnp.sqrt(jnp.mean(z['r']**2)/jnp.mean(y*y))
    accounts=jnp.array([2*a@fine[:w],2*a@tracking[:w],2*a@z['unknown'][:w],
        delta[:w]@delta[:w],2*a@(fine-z['native_fine'])[:w],
        2*a@(tracking-z['native_tracking'])[:w]])
    fine_path=jnp.linalg.norm(fine[:w]); coarse=z['jc']@fine
    dc=z['ec']-state['previous_ec']; ds=z['star']-state['previous_star']
    remainder=dc-state['previous_linear']; dz=z['z']-state['previous_z']
    closure=dz-(state['previous_linear']+remainder-ds)
    prior=state['offset']>0
    stats=jnp.array([z['z']@z['z'],z['channels'][1]@z['channels'][1],
        coarse@coarse,z['shadow_gain'],jnp.linalg.norm(ds)*prior,
        jnp.linalg.norm(remainder)*prior,z['z']@state['previous_z']*prior,
        z['ec']@z['ec'],jnp.linalg.norm(delta)])
    checks=jnp.array([jnp.linalg.norm(delta-fine-tracking-z['unknown']),
        jnp.abs(jnp.linalg.norm(fine[:-1])-z['target_norm']),
        jnp.linalg.norm(closure)*prior,
        jnp.where(z['balance'],jnp.linalg.norm(coarse),0.)])
    finite=jnp.all(jnp.isfinite(p+delta)) & jnp.isfinite(error)
    valid=z['valid'] & finite
    total=state['window_sum']+fine[:w]; length=state['window_path']+fine_path
    closes=(state['offset']+1)%1000==0
    new=dict(state)
    new.update(p=jnp.where(valid,p+delta,p),m=z['m'],v=z['v'],cm=z['cm'],count=z['count'],
        shadow_v=z['shadow_v'],offset=state['offset']+1,alive=valid,
        failure=jnp.where(~finite,1,jnp.where(~z['valid'],2,0)),
        accounts=state['accounts']+jnp.where(valid,accounts,0.),
        paths=state['paths']+jnp.where(valid,jnp.array([fine_path,jnp.linalg.norm(fine[:-1]),
            jnp.linalg.norm(tracking[:w]),jnp.linalg.norm(delta[:w])]),0.),
        fine_sum=state['fine_sum']+jnp.where(valid,fine[:w],0.),
        window_sum=jnp.where(closes,0.,total),window_path=jnp.where(closes,0.,length),
        last_coherence=jnp.where(closes,divide(jnp.linalg.norm(total),length),state['last_coherence']),
        checks=jnp.maximum(state['checks'],checks),stats=state['stats']+stats,
        minimum_error=jnp.minimum(state['minimum_error'],error),
        hits=state['hits']+(error<jnp.array([.01,.001,.0001,.000001])),
        loss_increases=state['loss_increases']+(error>state['last_error']+1e-14),last_error=error,
        previous_ec=z['ec'],previous_z=z['z'],previous_star=z['star'],previous_linear=z['jc']@delta)
    row=jnp.r_[error,a@a,z['z'],jnp.linalg.norm(fine[:-1]),jnp.linalg.norm(tracking[:-1]),
        coarse,accounts[0],accounts[1],accounts[3],z['shadow_gain'],jnp.linalg.norm(delta)]
    return new,row


@partial(jax.jit,static_argnames=('steps',))
def advance(state,x,y,config,arm=0,tracking_factor=1.,steps=1000):
    def step(s,_):
        return jax.lax.cond(s['alive'],lambda _: _step(s,x,y,config,arm,tracking_factor),
            lambda _: (s,jnp.full(len(BURST),jnp.nan)),None)
    return jax.lax.scan(step,state,None,length=steps)


@partial(jax.jit,static_argnames=('steps',))
def advance_many(states,x,y,configs,arms,tracking_factors,steps=1000):
    return jax.vmap(lambda s,c,a,r:advance(s,x,y,c,a,r,steps=steps))(
        states,configs,arms,tracking_factors)


@jax.jit
def diagnostics(state,x,y,config,arm=0,tracking_factor=1.):
    z=proposal(state,x,y,config,arm,tracking_factor);p=state['p'];w=(len(p)-1)//3
    d=pa.population(p); error=jnp.sqrt(jnp.mean(z['r']**2)/jnp.mean(y*y))
    d.update(relative_error=error,minimum_error=jnp.minimum(state['minimum_error'],error),
        A_closure=p[:w]@p[:w]-state['p0'][:w]@state['p0'][:w]-jnp.sum(state['accounts'][:4]),
        fine_coherence=divide(jnp.linalg.norm(state['fine_sum']),state['paths'][0]),
        last_coherence=state['last_coherence'],z_norm=jnp.linalg.norm(z['z']),
        shadow_gain=z['shadow_gain'],fine_norm=jnp.linalg.norm(z['fine'][:-1]),
        coarse_fine_norm=jnp.linalg.norm(z['jc']@z['fine']),alive=state['alive'],failure=state['failure'])
    for i,k in enumerate(('fine','tracking','unresolved','step','fine_intervention','tracking_intervention')):d['A_'+k]=state['accounts'][i]
    for i,k in enumerate(('fine_slope_path','fine_hidden_path','tracking_slope_path','total_slope_path')):d[k]=state['paths'][i]
    for i,k in enumerate(('component_error','norm_error','coarse_identity_error','balanced_response_error')):d[k]=state['checks'][i]
    for i,k in enumerate(('z2_sum','raw_tracking2_sum','coarse_fine2_sum','shadow_gain_sum','equilibrium_movement','nonlinear_coarse_movement','z_lag_dot','ec2_sum','total_hidden_path')):d[k]=state['stats'][i]
    for i,k in enumerate(('1pct','0p1pct','1e4','1e6')):d['hits_'+k]=state['hits'][i]+(error<jnp.array([.01,.001,.0001,.000001])[i])
    d['loss_increases']=state['loss_increases']
    return d


@jax.jit
def isolated_response(p,u,x,y):
    g,r,j,ec=af.field(p,x,y);_,info=af.split(g,j,ec)
    gn,rn,jn,ecn=af.field(p+u,x,y);_,after=af.split(gn,jn,ecn)
    linear=j@u;dc=ecn-ec;remainder=dc-linear
    star=ec-info['z'];starn=ecn-after['z'];ds=starn-star;dz=after['z']-info['z']
    w=(len(p)-1)//3
    return dict(linear_coarse=jnp.linalg.norm(linear),nonlinear_coarse=jnp.linalg.norm(remainder),
        equilibrium_change=jnp.linalg.norm(ds),z_change=jnp.linalg.norm(dz),
        identity_error=jnp.linalg.norm(dz-linear-remainder+ds),
        linear_z_pairing=linear@dz,equilibrium_z_pairing=-ds@dz,remainder_z_pairing=remainder@dz,
        tracking_change=jnp.linalg.norm(jn.T@after['z']-j.T@info['z']),
        fine_A=2*p[:w]@u[:w],linear_loss=g@u,hidden_norm=jnp.linalg.norm(u[:-1]),
        resolved=info['resolved'] & after['resolved'])


@jax.jit
def hessian_product(p,v,root,x,y):
    return root*jax.jvp(lambda z:af.field(z,x,y)[0],(p,),(root*v,))[1]
