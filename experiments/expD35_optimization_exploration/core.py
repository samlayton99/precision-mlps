"""Physical-coordinate model and explicit GD/Adam/EMA state transitions."""
from __future__ import annotations

from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD06_fixed_center_scales import core as old
from experiments.expD06_fixed_center_scales import difference_training as maps
from experiments.expD34_readout_race import targets as moments

COORDINATES = ('physical', 'collective', 'individual', 'neighbor')
ALIASES = dict(collective='scaled', individual='parameter_scale', neighbor='parameter_differences')
TRACE = ('mse', 'eta', 'mean_abs_lambda', 'readout_l2', 'native_readout_gradient',
         'native_geometry_gradient', 'physical_readout_gradient', 'physical_gamma_gradient',
         'delta_readout_rms', 'delta_gamma_rms', 'delta_mean_abs_gamma', 'coarse_residual_norm',
         'filtered_gradient_norm', 'raw_filtered_cosine', 'direction_norm', 'zero_physical_step',
         'adam_guard_fraction', 'active', 'neuron_replacements', 'reset_function_rms')


@lru_cache(maxsize=1)
def polynomial_mapping():
    return moments.polynomial_map(moments.grid(2048))


def target(x, name, xp=jnp):
    if name in ('sine', 'mixed', 'quadratic'):
        return old.target(x, name, xp)
    if name == 'runge':
        return 1/(1+25*x*x)
    coefficients = np.zeros(10)
    coefficients[[0, 1, int(name.removeprefix('moment'))]] = [.3, .4, np.sqrt(.75)]
    power = np.polynomial.legendre.leg2poly(polynomial_mapping() @ coefficients)
    return xp.polyval(xp.asarray(power[::-1]), x)


def physical(z, g, coordinates):
    c = z[:g.width+1]
    if coordinates != 'physical':
        c = maps.decode(c, g, ALIASES[coordinates])
    gamma = z[g.width+1:] / (1. if coordinates == 'physical' else g.h)
    return c, gamma


def encode(c, gamma, g, coordinates):
    if coordinates == 'physical':
        return np.r_[c, gamma]
    return np.r_[maps.encode(c, g, ALIASES[coordinates]), g.h*gamma]


def initialize(case):
    g = old.geometry(case['n'])
    c, gamma = old.initial_physical(g, case['seed'], 'xavier_a_reference')
    if case.get('initialization', 'reference_xavier') == 'individual_gaussian':
        c[1:] *= np.sqrt(g.alpha[1:]) / np.sqrt(2/(g.width+1))
    if case.get('slope_initialization', 'physical_xavier') == 'lambda_xavier':
        gamma /= g.h
    z = jnp.asarray(encode(c, gamma, g, case['coordinates']))
    return dict(z=z, m=jnp.zeros_like(z), v=jnp.zeros_like(z), ema=jnp.zeros_like(z),
                post_ema=jnp.zeros_like(z), age=jnp.zeros(z.shape, dtype=jnp.int64),
                failed=jnp.array(0, dtype=jnp.int64), eta=jnp.array(case['eta']),
                agreement_ema=jnp.array(1.),
                utility=jnp.zeros(g.width),neuron_age=jnp.zeros(g.width,dtype=jnp.int64),
                replacement_accumulator=jnp.array(0.),
                key=jax.random.PRNGKey(case['seed']+5701))


def field(z, x, y, g, coordinates):
    c, gamma = physical(z, g, coordinates)
    distance = x[:, None]-jnp.asarray(g.centers)
    pre = distance*gamma
    phi = jnp.tanh(pre)
    e = c[0]+phi @ c[1:]-y
    gc = jnp.concatenate((jnp.mean(e)[None], phi.T @ e / len(x)))
    gg = c[1:]*jnp.mean(e[:, None]*old.sech_squared(pre)*distance, axis=0)
    if coordinates == 'physical':
        grad = jnp.concatenate((gc, gg))
    else:
        grad = jnp.concatenate((maps.pullback(gc, g, ALIASES[coordinates]), gg/g.h))
    # Center x explicitly so this diagnostic also applies to stratified samples.
    xc = x-jnp.mean(x)
    coarse = jnp.sqrt(jnp.mean(e)**2+jnp.mean(e*xc)**2/jnp.mean(xc*xc))
    utility = jnp.abs(c[1:])*jnp.mean(jnp.abs(phi), axis=0)
    return .5*jnp.mean(e*e), grad, gc, gg, coarse, utility


def cosine(a, b):
    """Scale before normalization; no absolute epsilon hides tiny directions."""
    sa, sb = jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))
    aa, bb = a/jnp.where(sa>0, sa, 1.), b/jnp.where(sb>0, sb, 1.)
    den = jnp.linalg.norm(aa)*jnp.linalg.norm(bb)
    return jnp.where(den>0, jnp.dot(aa, bb)/jnp.where(den>0, den, 1.), jnp.nan)


def optimizer_direction(state, grad, hp, optimizer):
    """EMA modifies raw native gradients before Adam; post is a separate arm."""
    ema = jnp.where(state['age']==0, grad, hp['ema_alpha']*state['ema']+(1-hp['ema_alpha'])*grad)
    filtered = (grad+hp['ema_strength']*ema)/jnp.where(hp['ema_normalized'], 1+hp['ema_strength'], 1.)
    filtered = jnp.where(hp['ema_location']==1, filtered, grad)
    age = state['age']+1
    m = hp['beta1']*state['m']+(1-hp['beta1'])*filtered
    v = hp['beta2']*state['v']+(1-hp['beta2'])*filtered**2
    sqrt_v = jnp.sqrt(v/(1-hp['beta2']**age))
    if optimizer == 'adam':
        direction = -(m/(1-hp['beta1']**age))/(sqrt_v+hp['epsilon'])
    else:
        direction = -filtered
    post = jnp.where(state['age']==0, direction,
                     hp['ema_alpha']*state['post_ema']+(1-hp['ema_alpha'])*direction)
    direction = jnp.where(hp['ema_location']==2, direction+hp['ema_strength']*post, direction)
    return direction, dict(m=m, v=v, ema=ema, post_ema=post, age=age), filtered, sqrt_v


def hyperparameters(case):
    defaults = dict(ema_alpha=.98, ema_strength=0., ema_location=1, ema_normalized=False,
                    beta1=.9, beta2=.999, epsilon=1e-15,
                    maturity=5000,replacement_rate=1e-5,reset_until=10**12,slope_redraw_factor=1.)
    if case.get('slope_initialization')=='lambda_xavier': defaults['slope_redraw_factor']=case['n']/2
    return {k:jnp.asarray(case.get(k, v)) for k,v in defaults.items()}


@lru_cache(maxsize=64)
def chunk(n, coordinates, optimizer, name, length, sampling='full', batch_size=1024, reset='none', capture=False):
    g = old.geometry(n)
    grid = jnp.linspace(-1, 1, 16*n+1)
    def one(state, hp, start, replay):
        def step(current, inputs):
            index,replay_mask=inputs
            key, draw = jax.random.split(current['key'])
            if sampling == 'stratified':
                x = -1+2*(jnp.arange(batch_size)+jax.random.uniform(draw, (batch_size,), dtype=jnp.float64))/batch_size
            else:
                x = grid
            y = target(x, name)
            loss, grad, gc, gg, coarse, utility = field(current['z'], x, y, g, coordinates)
            direction, history, filtered, sqrt_v = optimizer_direction(current, grad, hp, optimizer)
            z1 = current['z']+current['eta']*direction
            c0, ga0 = physical(current['z'], g, coordinates)
            c1, ga1 = physical(z1, g, coordinates)
            finite = jnp.isfinite(loss)&jnp.all(jnp.isfinite(z1))&jnp.all(jnp.isfinite(grad))
            finite &= jnp.all(jnp.isfinite(history['v'])) & jnp.all(jnp.isfinite(c1))
            active = (current['failed']==0)&finite
            next_state = dict(current, z=z1, key=key, **history)
            mask=jnp.zeros(g.width,dtype=bool); jump=jnp.array(0.)
            if reset!='none':
                from . import recycling
                next_state,mask,jump=recycling.apply(next_state,utility,g,coordinates,hp,reset,replay_mask,index,x)
            else:
                next_state['neuron_age']=current['neuron_age']+1
            next_state = jax.tree.map(lambda new, old: jnp.where(active, new, old), next_state, current)
            next_state['failed'] = jnp.where((current['failed']==0)&~finite, index+1, current['failed'])
            trace = jnp.array([2*loss, current['eta'], jnp.mean(jnp.abs(g.h*ga0)), jnp.linalg.norm(c0[1:]),
                jnp.linalg.norm(grad[:g.width+1]), jnp.linalg.norm(grad[g.width+1:]),
                jnp.linalg.norm(gc), jnp.linalg.norm(gg), jnp.sqrt(jnp.mean((c1-c0)**2)),
                jnp.sqrt(jnp.mean((ga1-ga0)**2)), jnp.mean(jnp.abs(ga1)-jnp.abs(ga0)), coarse,
                jnp.linalg.norm(filtered), cosine(grad, filtered), jnp.linalg.norm(direction),
                jnp.all(c1==c0)&jnp.all(ga1==ga0), jnp.mean(sqrt_v<=hp['epsilon']), active,
                jnp.sum(mask),jump])
            trace=jnp.where(active, trace, jnp.nan)
            if reset!='none': trace=jnp.r_[trace,mask&active]
            return next_state, (trace,next_state['z']) if capture else trace
        return jax.lax.scan(step, state, (start+jnp.arange(length),replay))
    compiled=jax.jit(jax.vmap(one, in_axes=(0, 0, None, 0)))
    def run(states,hp,start,replay=None):
        if replay is None: replay=jnp.zeros((len(states['z']),length,g.width),dtype=bool)
        return compiled(states,hp,start,replay)
    return run


@lru_cache(maxsize=32)
def evaluate(n, coordinates, name, size=32768):
    g = old.geometry(n)
    x = -1+2*(jnp.arange(size)+.5)/size
    y = target(x, name)
    def one(z):
        c, gamma = physical(z, g, coordinates)
        def block(xx):
            return c[0]+jnp.tanh((xx[:, None]-jnp.asarray(g.centers))*gamma) @ c[1:]
        prediction = jax.lax.map(block, x.reshape(-1, 512)).reshape(-1)
        mse = jnp.mean((prediction-y)**2)
        return jnp.array([mse, mse/jnp.mean(y*y), *jnp.quantile(jnp.abs(g.h*gamma), jnp.array([.1,.5,.9])),
                          jnp.linalg.norm(c[1:]), jnp.max(jnp.abs(gamma))])
    return jax.jit(jax.vmap(one))


@lru_cache(maxsize=32)
def agreement(n, coordinates, name, batch_size=1024):
    """Independent stratified batches evaluated at one unchanged parameter state."""
    g=old.geometry(n)
    grid=jnp.linspace(-1,1,16*n+1)
    def one(z, key):
        draws=jax.random.split(key,8)
        def sample(draw):
            x=-1+2*(jnp.arange(batch_size)+jax.random.uniform(draw,(batch_size,),dtype=jnp.float64))/batch_size
            return field(z,x,target(x,name),g,coordinates)[1]
        gradients=jax.vmap(sample)(draws)
        full=field(z,grid,target(grid,name),g,coordinates)[1]
        rows=[]
        for part in (slice(None),slice(None,g.width+1),slice(g.width+1,None)):
            v=gradients[:,part]; f=full[part]
            pair=jnp.stack([cosine(v[i],v[j]) for i in range(8) for j in range(i)])
            mean=jnp.mean(v,axis=0)
            scale=jnp.maximum(jnp.max(jnp.abs(v)),jnp.max(jnp.abs(f)))
            scale=jnp.where(scale>0,scale,1.)
            variance=jnp.mean(jnp.sum(((v-mean)/scale)**2,axis=1))
            signal=jnp.sum((f/scale)**2)
            rows.append(jnp.array([jnp.mean(pair),cosine(mean,f),variance/jnp.where(signal>0,signal,jnp.nan)]))
        return jnp.stack(rows), gradients, full
    return jax.jit(jax.vmap(one))
