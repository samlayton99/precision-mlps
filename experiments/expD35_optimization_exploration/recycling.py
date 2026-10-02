"""Mature-neuron recycling and matched optimizer-state controls."""
import jax
import jax.numpy as jnp
import numpy as np
from . import core


def encode(c, gamma, g, coordinates):
    if coordinates=='physical': return jnp.r_[c,gamma]
    if coordinates=='neighbor':
        readout=jnp.r_[c[0]/g.alpha[0],jnp.cumsum(c[1:])/jnp.asarray(np.cumsum(g.alpha[1:]))]
    else:
        scale=g.d if coordinates=='collective' else g.alpha
        readout=c/jnp.asarray(scale)
    return jnp.r_[readout,g.h*gamma]


def apply(state, utility, g, coordinates, hp, mode, replay_mask, index, x):
    """Reset after the simultaneous optimizer update; expose the actual jump."""
    smoothed=.99*state['utility']+.01*utility
    age=state['neuron_age']+1
    eligible=age>=hp['maturity']
    active=index<hp['reset_until']
    accumulator=state['replacement_accumulator']+hp['replacement_rate']*jnp.sum(eligible)*active
    count=jnp.minimum(jnp.floor(accumulator).astype(jnp.int64),jnp.sum(eligible))
    if mode in ('replay_state','replay_random'):
        count=jnp.sum(replay_mask).astype(jnp.int64)
    score=smoothed
    if mode=='replay_random':
        score=jax.random.uniform(jax.random.fold_in(state['key'],3109),score.shape,dtype=jnp.float64)
    order=jnp.argsort(jnp.where(eligible,score,jnp.inf))
    ranks=jnp.zeros(g.width,dtype=jnp.int64).at[order].set(jnp.arange(g.width))
    mask=(ranks<count)&eligible&active
    if mode=='replay_state': mask=replay_mask&active
    c,gamma=core.physical(state['z'],g,coordinates)
    replacement=jnp.abs(jax.random.normal(jax.random.fold_in(state['key'],7213),(g.width,),dtype=jnp.float64))
    replacement*=5/3*np.sqrt(2/(g.width+1))*hp['slope_redraw_factor']
    cnew=jnp.r_[c[0],jnp.where(mask,0.,c[1:])]
    gnew=jnp.where(mask,replacement,gamma)
    if mode=='replay_state': cnew,gnew=c,gamma
    # In difference coordinates deleting a physical weight changes the suffix of q.
    affected_readout=jnp.cumsum(mask)>0 if coordinates=='neighbor' else mask
    affected=jnp.r_[False,affected_readout,mask]
    z=jnp.where(jnp.any(mask)&(mode!='replay_state'),encode(cnew,gnew,g,coordinates),state['z'])
    out=dict(state,z=z,utility=jnp.where(mask,0.,smoothed),
             neuron_age=jnp.where(mask,0,age),replacement_accumulator=accumulator-jnp.floor(accumulator))
    for name in ('m','v','ema','post_ema','age'):
        out[name]=jnp.where(affected,0,state[name])
    # Reset weights are zero, so the deleted old contribution gives the exact jump.
    jump=jnp.tanh((x[:,None]-jnp.asarray(g.centers))*gamma) @ jnp.where(mask,c[1:],0.)
    jump=jnp.where(mode=='replay_state',0.,jnp.sqrt(jnp.mean(jump**2)))
    return out,mask,jump
