"""Population activity, direction persistence, and norm-matched fine interventions."""
from functools import partial

import jax
import jax.numpy as jnp

from . import population_adam as pa, adam_forces as af

ARMS = ('native', 'current_fine', 'fixed_scaling', 'both')
WINDOWS = jnp.array([100, 1000, 10000])
FINE = jnp.array([1., 1., 1., 0., 0., 1.])


def ratio(a, b):
    return jnp.where(b > 0, a / jnp.where(b > 0, b, 1.), 0.)


def proposals(state, x, y, settings, fixed_inverse, arm):
    eta, b1, b2, eps, adaptive = settings
    g, residual, _, parts, info = pa.field(state['p'], x, y)
    count = state['count'] + 1
    m, v, cm, mh, ch, inverse = af.moments(
        g, parts, state['m'], state['v'], state['cm'], count, b1, b2, eps, adaptive)
    native_components = -eta * ch * inverse
    native_fine = FINE @ native_components
    source = jnp.where((arm & 1) != 0, parts, ch)
    metric = jnp.where((arm & 2) != 0, fixed_inverse, inverse)
    candidate = -eta * source * metric
    candidate_fine = FINE @ candidate
    scale = ratio(jnp.linalg.norm(native_fine[:-1]), jnp.linalg.norm(candidate_fine[:-1]))
    modified = jnp.where(FINE[:, None] > 0, scale * candidate[:, :-1], native_components[:, :-1])
    components = native_components.at[:, :-1].set(jnp.where(arm != 0, modified, native_components[:, :-1]))
    fine = FINE @ components
    native_delta = -eta * inverse * mh
    delta = jnp.where(arm != 0, native_delta + fine - native_fine, native_delta)
    return dict(m=m, v=v, cm=cm, count=count, delta=delta, components=components,
                fine=fine, native_fine=native_fine, inverse=inverse,
                raw_fine=FINE @ parts, raw_tracking=parts[3], gradient=g,
                norm_target=jnp.linalg.norm(native_fine[:-1]),residual=residual, resolved=info['resolved'],
                scale=scale, candidate_norm=jnp.linalg.norm(candidate_fine[:-1]))


def initialize(saved, x, y, settings):
    state = {k: saved[k] for k in ('p', 'm', 'v', 'cm', 'count')}
    p = state['p']; hidden = p[:-1].reshape(3, -1); w = hidden.shape[1]
    energy = jnp.sum(hidden**2, axis=0)
    rank = jnp.argsort(jnp.argsort(-energy))
    cuts = jnp.ceil(jnp.array([.01, .05, .20]) * w)
    category = jnp.sum(rank[:, None] >= cuts, axis=1)
    groups = (jnp.arange(4)[:, None] == category[None, :]).astype(p.dtype)
    inv = proposals(state, x, y, settings, jnp.ones_like(p), 0)['inverse']
    state.update(p0=p, energy0=energy/jnp.sum(energy), groups=groups, fixed_inverse=inv,
                 offset=jnp.array(0, dtype=jnp.int64), accounts=jnp.zeros((4, 3, 4)),
                 activity=jnp.zeros((3, w)), fine_sum=jnp.zeros_like(p[:-1]),
                 paths=jnp.zeros(4), window_sum=jnp.zeros((3, len(p)-1)),
                 window_paths=jnp.zeros((3, 2)), window_start=jnp.zeros((3, len(p)-1)),
                 window_alignment=jnp.zeros(3), previous_fine=jnp.zeros_like(p[:-1]),
                 window_totals=jnp.zeros((3, 4)), window_counts=jnp.zeros(3),
                 radial=jnp.zeros(2), checks=jnp.zeros(3), unresolved=jnp.array(0),
                 zero_candidate=jnp.array(0))
    return state


def step(state, x, y, settings, arm):
    z = proposals(state, x, y, settings, state['fixed_inverse'], arm)
    return apply_proposal(state,z,settings,arm)


def apply_proposal(state,z,settings,arm):
    """Account for a specified update, including an explicitly chosen norm budget."""
    p = state['p']; w = (len(p)-1)//3
    delta = z['delta']; fine = z['fine']; tracking = z['components'][3]
    unresolved = z['components'][4]
    block = p[:-1].reshape(3, w)
    vectors = jnp.stack((fine[:-1], tracking[:-1], unresolved[:-1])).reshape(3, 3, w)
    linear = 2*block[None, :, :]*vectors
    values = jnp.concatenate((linear, delta[:-1].reshape(1, 3, w)**2), axis=0)
    accounts = jnp.einsum('gn,cbn->gbc', state['groups'], values)
    slope_norm = jnp.linalg.norm(fine[:w]); hidden_norm = jnp.linalg.norm(fine[:-1])
    paths = jnp.array([slope_norm, hidden_norm, jnp.linalg.norm(tracking[:w]), jnp.linalg.norm(delta[:w])])
    start = state['offset'] % WINDOWS == 0
    first = jnp.where(start[:, None], fine[:-1], state['window_start'])
    accum = state['window_sum'] + fine[:-1]
    lengths = state['window_paths'] + jnp.array([slope_norm, hidden_norm])
    adjacent = ratio(fine[:-1] @ state['previous_fine'], hidden_norm*jnp.linalg.norm(state['previous_fine']))
    alignment = state['window_alignment'] + jnp.where(start, 0., adjacent)
    closes = (state['offset']+1) % WINDOWS == 0
    window_values = jnp.stack((ratio(jnp.linalg.norm(accum[:, :w], axis=1), lengths[:, 0]),
        ratio(jnp.linalg.norm(accum, axis=1), lengths[:, 1]),
        ratio(jnp.sum(first*fine[:-1], axis=1), jnp.linalg.norm(first, axis=1)*hidden_norm),
        alignment/(WINDOWS-1)), axis=1)
    mismatch = jnp.linalg.norm(z['components'].sum(axis=0)-delta)
    norm_mismatch = jnp.abs(hidden_norm-z['norm_target'])
    native_bias_delta = -settings[0]*z['inverse'][-1]*z['m'][-1]/(1-settings[1]**z['count'])
    new = dict(state)
    new.update(p=p+delta, m=z['m'], v=z['v'], cm=z['cm'], count=z['count'],
        offset=state['offset']+1, accounts=state['accounts']+accounts,
        activity=state['activity']+jnp.sum(jnp.stack((fine[:-1], tracking[:-1], delta[:-1])).reshape(3, 3, w)**2, axis=1),
        fine_sum=state['fine_sum']+fine[:-1], paths=state['paths']+paths,
        window_sum=jnp.where(closes[:, None], 0., accum),
        window_paths=jnp.where(closes[:, None], 0., lengths), window_start=first,
        window_alignment=jnp.where(closes, 0., alignment), previous_fine=fine[:-1],
        window_totals=state['window_totals']+jnp.where(closes[:, None], window_values, 0.),
        window_counts=state['window_counts']+closes,
        radial=state['radial']+jnp.array([2*p[:w]@fine[:w], 2*jnp.abs(p[:w]@fine[:w])]),
        checks=jnp.maximum(state['checks'], jnp.array([mismatch, norm_mismatch, jnp.abs(delta[-1]-native_bias_delta)])),
        unresolved=state['unresolved']+~z['resolved'],
        zero_candidate=state['zero_candidate']+((arm!=0)&(z['candidate_norm']==0)&(jnp.linalg.norm(z['native_fine'][:-1])>0)))
    return new


@partial(jax.jit, static_argnames=('steps',))
def advance(state, x, y, settings, arm, steps=1000):
    return jax.lax.fori_loop(0, steps, lambda _, s: step(s, x, y, settings, arm), state)


@jax.jit
def diagnostics(state, x, y, settings):
    p=state['p']; w=(len(p)-1)//3; hidden=p[:-1].reshape(3,w)
    energy=jnp.sum(hidden**2,axis=0); weights=energy/jnp.sum(energy)
    d=pa.population(p); d['relative_error']=jnp.sqrt(jnp.mean(prediction(p,x,y)**2)/jnp.mean(y*y))
    d['energy_overlap']=jnp.sum(jnp.minimum(weights,state['energy0']))
    d['energy_effective_count']=ratio(1.,jnp.sqrt(jnp.sum(weights**3)))
    accounts=jnp.sum(state['accounts'],axis=0)
    names=('fine','tracking','unresolved','step')
    for block,name in enumerate(('A','bias','readout')):
        for c,channel in enumerate(names):d[f'{name}_{channel}']=accounts[block,c]
        observed=jnp.sum(hidden[block]**2-state['p0'][:-1].reshape(3,w)[block]**2)
        d[name+'_closure']=observed-jnp.sum(accounts[block])
    for g in range(4):
        d[f'group{g}_energy_share']=state['groups'][g]@weights
        d[f'group{g}_slope_energy']=state['groups'][g]@(hidden[0]**2)
        d[f'group{g}_fine_A']=state['accounts'][g,0,0]
        d[f'group{g}_tracking_A']=state['accounts'][g,0,1]
    for i,name in enumerate(('fine','tracking','total')):
        activity=ratio(state['activity'][i],jnp.sum(state['activity'][i]))
        d[name+'_activity_effective_count']=ratio(1.,jnp.sqrt(jnp.sum(activity**3)))
        d[name+'_activity_energy_overlap']=jnp.sum(jnp.minimum(activity,weights))
        for g in range(4):d[f'group{g}_{name}_activity_share']=state['groups'][g]@activity
    for i,n in enumerate((100,1000,10000)):
        for j,name in enumerate(('slope_coherence','hidden_coherence','endpoint_alignment','adjacent_alignment')):
            d[f'{name}_{n}']=ratio(state['window_totals'][i,j],state['window_counts'][i])
        d[f'window_count_{n}']=state['window_counts'][i]
    for i,name in enumerate(('fine_slope_path','fine_hidden_path','tracking_slope_path','total_slope_path')):d[name]=state['paths'][i]
    d['fine_slope_net_norm']=jnp.linalg.norm(state['fine_sum'][:w])
    d['fine_hidden_net_norm']=jnp.linalg.norm(state['fine_sum'])
    d['actual_slope_displacement']=jnp.linalg.norm(p[:w]-state['p0'][:w])
    d['fine_radial_cancellation']=ratio(state['radial'][0],state['radial'][1])
    z=proposals(state,x,y,settings,state['fixed_inverse'],0)
    raw=-settings[0]*z['raw_fine']; scaled=raw*z['inverse']; native=z['fine']
    for name,vec in (('raw',raw),('scaled_current',scaled),('processed',native)):
        norm=jnp.linalg.norm(vec[:-1]); slope=jnp.linalg.norm(vec[:w])
        d[name+'_hidden_norm']=norm; d[name+'_slope_fraction']=ratio(slope,norm)
        d[name+'_outward_cosine']=ratio(p[:w]@vec[:w],jnp.linalg.norm(p[:w])*slope)
        d[name+'_A_rate']=2*p[:w]@vec[:w]
    d['processed_current_alignment']=ratio(native[:-1]@scaled[:-1],jnp.linalg.norm(native[:-1])*jnp.linalg.norm(scaled[:-1]))
    for arm in (1,2,3):
        probe=proposals(state,x,y,settings,state['fixed_inverse'],arm)
        d[f'arm{arm}_normalization']=probe['scale']
        d[f'arm{arm}_A_rate']=2*p[:w]@probe['fine'][:w]
    d.update(component_error=state['checks'][0], norm_error=state['checks'][1],
             bias_error=state['checks'][2], unresolved=state['unresolved'], zero_candidate=state['zero_candidate'])
    return d


def prediction(p,x,y):
    a,b,c=p[:-1].reshape(3,-1)
    return jnp.tanh(x[:,None]*a+b)@c+p[-1]-y


def fine_residual(p,x,y):
    r=prediction(p,x,y); q=x/jnp.sqrt(jnp.mean(x*x))
    return r-jnp.mean(r)-q*jnp.mean(q*r)


def force_with_residual(p,x,r):
    # The chosen residual is a fixed output vector; all geometry is current.
    g,_,jc,ec=af.field(p,x,prediction(p,x,jnp.zeros_like(x))-r)
    channels,info=af.split(g,jc,ec)
    return channels[0],info['resolved']


@jax.jit
def crossed_forces(old,new,x,y,settings):
    r0=fine_residual(old['p'],x,y); r1=fine_residual(new['p'],x,y)
    f00,s00=force_with_residual(old['p'],x,r0)
    f01,s01=force_with_residual(old['p'],x,r1)
    f10,s10=force_with_residual(new['p'],x,r0)
    f11,s11=force_with_residual(new['p'],x,r1)
    geometry=((f10-f00)+(f11-f01))/2
    residual=((f01-f00)+(f11-f10))/2
    z=proposals(new,x,y,settings,new['fixed_inverse'],0)
    w=(len(new['p'])-1)//3
    pairing=lambda f:-2*settings[0]*new['p'][:w]@(z['inverse'][:w]*f[:w])
    return dict(geometry_A_rate_change=pairing(geometry),residual_A_rate_change=pairing(residual),
        total_A_rate_change=pairing(f11-f00),old_force_A_rate=pairing(f00),new_force_A_rate=pairing(f11),
        geometry_force_norm=jnp.linalg.norm(geometry),residual_force_norm=jnp.linalg.norm(residual),
        crossed_closure=jnp.linalg.norm(geometry+residual-(f11-f00)),
        current_force_error=jnp.linalg.norm(f11-z['raw_fine']),crossed_resolved=s00&s01&s10&s11)
