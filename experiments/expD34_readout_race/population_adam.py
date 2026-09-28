"""Population measurements of native Adam, with passive shared-denominator ledgers.

All numerical execution is remote. The parameter update uses the original
full gradient; auxiliary histories never feed back into the optimizer.
"""
from functools import partial

import jax
import jax.numpy as jnp

from . import adam_forces as af

COMPONENTS = ('generated', 'target', 'compensation', 'tracking', 'unresolved', 'inherited')
LEDGER = tuple(f'{q}_{c}' for q in ('M', 'A', 'logC6') for c in COMPONENTS)
LEDGER += ('M_defect', 'A_defect', 'logC6_defect', 'rootC6_sum', 'dispersion_sum',
           'fine_slope_path', 'tracking_slope_path', 'total_slope_path',
           'fine_raw_slope', 'tracking_raw_slope', 'loss_increases')


def population(p):
    a, b, c = p[:-1].reshape(3, -1)
    e = a*a+b*b+c*c
    m = jnp.sum(e)
    c6 = len(a)**2*jnp.sum(e**3)/m**3
    c10 = len(a)**4*jnp.sum(e**5)/m**5
    return dict(M=m, A=a@a, C6=c6, C10=c10, K=c10/c6**2,
                slope_rms=jnp.sqrt(jnp.mean(a*a)), bias2=b@b, readout2=c@c)


def concentration_score(p):
    hidden = p[:-1].reshape(3, -1)
    e = jnp.sum(hidden*hidden, axis=0)
    weight = 6*(e*e/jnp.sum(e**3)-1/jnp.sum(e))
    return jnp.r_[(hidden*weight).ravel(), 0.]


def field(p, x, y):
    """Gradient components, using the unchanged GD-reference coarse balance."""
    g, r, jc, ec = af.field(p, x, y)
    channels, info = af.split(g, jc, ec)
    a, b, c = p[:-1].reshape(3, -1)
    u = x[:, None]*a+b
    h = jnp.tanh(u)
    exp = jnp.exp(-2*jnp.abs(u))
    s = 4*exp/(1+exp)**2
    q1 = x/jnp.sqrt(jnp.mean(x*x))
    yh = y-jnp.mean(y)-q1*jnp.mean(q1*y)
    target = -jnp.r_[c*(x@(yh[:, None]*s))/len(x),
                     c*jnp.mean(yh[:, None]*s, axis=0), h.T@yh/len(x), jnp.mean(yh)]
    generated = info['fine']-target
    parts = jnp.stack((generated, target, info['balanced'], channels[1],
                       channels[2], jnp.zeros_like(g)))
    # Preserve the explicit unresolved channel rather than regularizing a solve.
    parts = parts.at[:3].set(jnp.where(info['resolved'], parts[:3], 0.))
    return g, r, jc, parts, info


def initial(p, m=None, v=None, count=0, channel_m=None):
    p = jnp.asarray(p)
    m = jnp.zeros_like(p) if m is None else jnp.asarray(m)
    v = jnp.zeros_like(p) if v is None else jnp.asarray(v)
    cm = jnp.zeros((len(COMPONENTS), len(p)), dtype=p.dtype)
    if channel_m is None:
        cm = cm.at[5].set(m)
    else:
        old = jnp.asarray(channel_m)
        cm = cm.at[3].set(old[1]).at[4].set(old[2]).at[5].set(old[0])
    return dict(p=p, m=m, v=v, cm=cm, count=jnp.asarray(count, dtype=jnp.int64),
                ledger=jnp.zeros(len(LEDGER)), identity=jnp.zeros(3),
                unresolved=jnp.array(0, dtype=jnp.int64),
                minimum_error=jnp.array(jnp.inf), hits=jnp.zeros(3, dtype=jnp.int64),
                previous_loss=jnp.array(jnp.inf))


def step(old, x, y, settings):
    eta, beta1, beta2, epsilon, adaptive = settings
    p = old['p']
    g, r, _, parts, info = field(p, x, y)
    count = old['count']+1
    m, v, cm, mh, ch, inverse = af.moments(
        g, parts, old['m'], old['v'], old['cm'], count,
        beta1, beta2, epsilon, adaptive)
    delta = -eta*inverse*mh
    components = -eta*inverse*ch
    pn = p+delta
    w = (len(p)-1)//3
    before, after = population(p), population(pn)
    log_parts = components@concentration_score(p)
    mass_parts = 2*components[:, :-1]@p[:-1]
    slope_parts = 2*components[:, :w]@p[:w]
    fine = components[0]+components[1]+components[2]+components[5]
    raw_fine = parts[0]+parts[1]+parts[2]
    loss = .5*jnp.mean(r*r)
    relative = jnp.sqrt(2*loss/jnp.mean(y*y))
    extras = jnp.array([
        delta[:-1]@delta[:-1], delta[:w]@delta[:w],
        jnp.log(after['C6']/before['C6'])-jnp.sum(log_parts),
        (jnp.sqrt(before['C6'])+jnp.sqrt(after['C6']))/2,
        (jnp.sqrt(jnp.maximum(0., 9*before['K']-5))+
         jnp.sqrt(jnp.maximum(0., 9*after['K']-5)))/2,
        jnp.linalg.norm(fine[:w]), jnp.linalg.norm(components[3, :w]),
        jnp.linalg.norm(delta[:w]), jnp.linalg.norm(raw_fine[:w]),
        jnp.linalg.norm(parts[3, :w]), loss>old['previous_loss']+1e-14])
    identity = jnp.array([jnp.linalg.norm(jnp.sum(parts, axis=0)-g),
                          jnp.linalg.norm(jnp.sum(cm, axis=0)-m),
                          jnp.linalg.norm(jnp.sum(components, axis=0)-delta)])
    return dict(p=pn, m=m, v=v, cm=cm, count=count,
                ledger=old['ledger']+jnp.r_[mass_parts, slope_parts, log_parts, extras],
                identity=jnp.maximum(old['identity'], identity),
                unresolved=old['unresolved']+~info['resolved'],
                minimum_error=jnp.minimum(old['minimum_error'], relative),
                hits=old['hits']+(relative<jnp.array([.01,.001,.0001])),
                previous_loss=loss)


@partial(jax.jit, static_argnames=('steps',))
def advance(state, x, y, settings, steps):
    return jax.lax.fori_loop(0, steps, lambda _, old: step(old, x, y, settings), state)


@jax.jit
def diagnostics(state, x, y, settings):
    p = state['p']
    eta, beta1, beta2, epsilon, adaptive = settings
    g, r, jc, parts, info = field(p, x, y)
    _, _, _, mh, ch, inverse = af.moments(g, parts, state['m'], state['v'],
        state['cm'], state['count']+1, beta1, beta2, epsilon, adaptive)
    delta = -eta*inverse*mh
    components = -eta*inverse*ch
    a, b, c = p[:-1].reshape(3, -1)
    u = x[:, None]*a+b
    h = jnp.tanh(u)
    exp = jnp.exp(-2*jnp.abs(u))
    s = 4*exp/(1+exp)**2
    jacobian = jnp.c_[x[:,None]*s*c, s*c, h, jnp.ones_like(x)]
    basis = jnp.stack((jnp.ones_like(x), x/jnp.sqrt(jnp.mean(x*x))), axis=1)
    jh = jacobian-basis@jc
    eh = r-basis@(basis.T@r/len(x))
    raw = jh.T@eh/len(x)
    fine = parts[0]+parts[1]+parts[2]
    C = (jc*inverse)@jc.T
    ev = jnp.linalg.eigvalsh(C)
    resolved = ev[0]>64*jnp.finfo(p.dtype).eps*jnp.maximum(1.,ev[-1])
    load = jc@(inverse*raw)
    projected = raw@ (inverse*raw)-load@jnp.linalg.solve(jnp.where(resolved,C,jnp.eye(2)),load)
    pn = p+delta
    an,bn,cn = pn[:-1].reshape(3,-1)
    rn = jnp.tanh(x[:,None]*an+bn)@cn+pn[-1]-y
    linear = jacobian@delta
    change = .5*jnp.mean(rn*rn-r*r)
    linear_loss = g@delta
    quadratic = .5*jnp.mean(linear*linear)
    residual2 = jnp.mean(eh*eh)
    ratio = lambda value, denominator: jnp.where(denominator>0,value/denominator,jnp.nan)
    result = population(p)
    result.update(relative_error=jnp.sqrt(jnp.mean(r*r)/jnp.mean(y*y)),
        fine_residual_norm=jnp.sqrt(residual2),fine_norm=jnp.linalg.norm(fine),
        tracking_norm=jnp.linalg.norm(parts[3]), coarse_resolved=info['resolved'],
        jacobian_fine_hs=jnp.sqrt(jnp.sum(jh*jh)/len(x)),
        jacobian_adaptive_fine_hs=jnp.sqrt(jnp.sum(jh*jh*inverse)/len(x)),
        raw_access=ratio(raw@raw,residual2), balanced_raw_access=ratio(fine@fine,residual2),
        adaptive_access=ratio(raw@(inverse*raw),residual2),
        balanced_adaptive_access=jnp.where(resolved,ratio(projected,residual2),jnp.nan),
        adaptive_coarse_resolved=resolved, actual_loss_change=change,
        linear_loss_change=linear_loss, quadratic_loss_change=quadratic,
        nonlinear_loss_change=change-linear_loss-quadratic,
        current_gradient_descent=eta*g@(inverse*g),
        fine_velocity_alignment=ratio(-fine@delta,jnp.linalg.norm(fine)*jnp.linalg.norm(delta)),
        processed_logC6=jnp.sum(components@concentration_score(p)),
        discrete_logC6=jnp.log(population(pn)['C6']/result['C6']),
        inherited_step_norm=jnp.linalg.norm(components[5]),
        raw_logC6=concentration_score(p)@(-g))
    for i,name in enumerate(COMPONENTS):
        result[f'next_logC6_{name}']=components[i]@concentration_score(p)
        result[f'next_M_{name}']=2*components[i,:-1]@p[:-1]
        result[f'next_A_{name}']=2*components[i,:len(a)]@a
    return result
