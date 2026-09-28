"""Test tracking's indirect effect through the fine-update second moment."""
from functools import partial
import jax
import jax.numpy as jnp

from . import population_adam_motion as motion

ARMS=('native','gain_only','tracking_attenuated_variance')


def initialize(saved,x,y,settings):
    state=motion.initialize(saved,x,y,settings)
    # Preserve incoming variance; attenuate only subsequent tracking gradients.
    state.update(alternative_v=saved['v'],gain_sums=jnp.zeros(3),gain_count=jnp.array(0))
    return state


def proposal(state,x,y,settings,arm):
    z=motion.proposals(state,x,y,settings,state['fixed_inverse'],0)
    adjusted=z['gradient']-.9*z['raw_tracking']
    alternative=settings[2]*state['alternative_v']+(1-settings[2])*adjusted**2
    inv=1/(jnp.sqrt(alternative/(1-settings[2]**z['count']))+settings[3])
    candidate=z['fine']*inv/z['inverse']
    n=jnp.linalg.norm(z['fine'][:-1]);raw_gain=motion.ratio(jnp.linalg.norm(candidate[:-1]),n)
    gain=jnp.minimum(raw_gain,10.)
    # Match gain-only and reweighted proposals locally; cap the common norm gain.
    new_components=z['components']*jnp.where(arm==1,gain,inv/z['inverse']*motion.ratio(gain,raw_gain))
    hidden=jnp.where(motion.FINE[:,None]>0,new_components[:,:-1],z['components'][:,:-1])
    components=z['components'].at[:,:-1].set(jnp.where(arm!=0,hidden,z['components'][:,:-1]))
    fine=motion.FINE@components
    z.update(delta=jnp.where(arm!=0,z['delta']+fine-z['fine'],z['delta']),components=components,fine=fine,
             norm_target=jnp.where(arm!=0,gain*n,n),candidate_norm=jnp.linalg.norm(candidate[:-1]))
    return z,alternative,raw_gain,gain


def step(state,x,y,settings,arm):
    z,alternative,raw_gain,gain=proposal(state,x,y,settings,arm)
    new=motion.apply_proposal(state,z,settings,arm)
    new.update(alternative_v=alternative,gain_sums=state['gain_sums']+jnp.array([raw_gain,gain,raw_gain>10]),
               gain_count=state['gain_count']+1)
    return new


@partial(jax.jit,static_argnames=('steps',))
def advance(state,x,y,settings,arm,steps=1000):
    return jax.lax.fori_loop(0,steps,lambda _,s:step(s,x,y,settings,arm),state)


@jax.jit
def diagnostics(state,x,y,settings):
    d=motion.diagnostics(state,x,y,settings)
    _,_,raw,gain=proposal(state,x,y,settings,0)
    d.update(instant_variance_gain=raw,instant_capped_gain=gain,
        mean_variance_gain=motion.ratio(state['gain_sums'][0],state['gain_count']),
        mean_capped_gain=motion.ratio(state['gain_sums'][1],state['gain_count']),
        gain_cap_fraction=motion.ratio(state['gain_sums'][2],state['gain_count']))
    return d
