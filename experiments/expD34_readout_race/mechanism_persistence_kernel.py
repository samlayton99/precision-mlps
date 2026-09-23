"""Full-complement persistence diagnostics and matched initial-force surgery.

Coordinates are concatenated (a, b, c, d), with empirical half-MSE and
Euclidean parameter metric. Callers must enable JAX FP64. No dense parameter
Hessian or sample-space projector is formed. The six modified fields are
causal interventions, not gradients of the original loss.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


ARMS = ((1., 1.), (0., 0.), (0., 1.), (2., 1.), (1., 0.), (1., 2.))
ARM_NAMES = ('natural', 'projected_constant', 'no_geometry', 'double_geometry',
             'no_relaxation', 'double_relaxation')


def output(p, x):
    a, b, c = p[:-1].reshape(3, -1)
    return jnp.tanh(x[:, None]*a+b)@c+p[-1]


def _basis(x):
    centered = x-jnp.mean(x)
    return jnp.stack((jnp.ones_like(x), centered/jnp.sqrt(jnp.mean(centered**2))), axis=1)


def _fine(v, basis):
    return v-basis@(basis.T@v/len(basis))


def _project(v, jc, gram):
    return v-jc.T@jnp.linalg.solve(gram, jc@v)


def decomposition(p, x, y):
    """Return full gradient g=F+R and coarse geometry.

    z is coarse disequilibrium, so R=JC.T@z up to roundoff. balance is
    B eH, which enters the loaded-curvature correction; they are distinct.
    F is evaluated from the fine residual to avoid subtracting a large coarse
    gradient. Algebraically it equals Pi g.
    """
    a, b, c = p[:-1].reshape(3, -1)
    h = jnp.tanh(x[:, None]*a+b)
    s = 1-h*h
    jacobian = jnp.concatenate((s*c*x[:, None], s*c, h, jnp.ones((len(x), 1), dtype=p.dtype)), axis=1)
    f = h@c+p[-1]
    residual = f-y
    basis = _basis(x)
    jc = basis.T@jacobian/len(x)
    gram = jc@jc.T
    eh = _fine(residual, basis)
    raw = jacobian.T@eh/len(x)
    balance = jnp.linalg.solve(gram, jc@raw)
    force = raw-jc.T@balance
    g = jacobian.T@residual/len(x)
    return dict(F=force, R=g-force, g=g, eH=eh, JC=jc, gram=gram,
                z=basis.T@residual/len(x)+balance, balance=balance,
                J=jacobian, basis=basis, fH=_fine(f, basis), yH=_fine(y, basis))


def effective(p, x, y):
    return decomposition(p, x, y)['F']


def ordinary_gradient(p, x, y):
    return decomposition(p, x, y)['g']


def _second_output(p, x, v):
    a, b, c = p[:-1].reshape(3, -1)
    va, vb, vc = v[:-1].reshape(3, -1)
    h = jnp.tanh(x[:, None]*a+b)
    s = 1-h*h
    du = x[:, None]*va+vb
    return jnp.sum(-2*c*h*s*du*du+2*vc*s*du, axis=1)


def _ratio(value, q2):
    # Zero force has undefined relative persistence, not zero persistence.
    return jnp.where(q2 > 0, value/jnp.where(q2 > 0, q2, 1.), jnp.nan)


def diagnostics(p, x, y):
    """Scalar diagnostics; q is full effective speed, not slope acquisition.

    k is the log-speed derivative along ordinary velocity -(F+R). The loaded
    term F.T DF[R] is retained explicitly. C=C_generated-C_target includes
    the derivative of the instantaneous coarse projection.
    """
    state = decomposition(p, x, y)
    F, R, J, jc, gram, basis = (state[k] for k in ('F', 'R', 'J', 'JC', 'gram', 'basis'))
    jf = _fine(J@F, basis)
    second = _second_output(p, x, F)
    second_coarse = basis.T@second/len(x)

    def curvature(load):
        balanced = jnp.linalg.solve(gram, jc@(J.T@load/len(x)))
        return jnp.mean(load*second)-balanced@second_coarse

    cg, cy = curvature(state['fH']), curvature(state['yH'])
    c = curvature(state['eH'])
    d = jnp.mean(jf*jf)
    q2 = F@F
    derivative_r = jax.jvp(lambda v: effective(v, x, y), (p,), (R,))[1]
    derivative_g = jax.jvp(lambda v: effective(v, x, y), (p,), (state['g'],))[1]
    derivative_f = jax.jvp(lambda v: effective(v, x, y), (p,), (F,))[1]
    loaded = F@derivative_r
    return dict(q=jnp.sqrt(q2), q_squared=q2, D=d, C=c,
                C_generated=cg, C_target=cy, loaded=loaded,
                k_pure=_ratio(-(d+c), q2), k=_ratio(-(d+c+loaded), q2),
                k_jvp=_ratio(-F@derivative_g, q2),
                identity_error=F@derivative_f-d-c,
                curvature_split_error=c-cg+cy,
                tracking_norm=jnp.linalg.norm(R), gradient_norm=jnp.linalg.norm(state['g']),
                tracking_fine_forcing_norm=jnp.linalg.norm(_fine(J@R, basis))/jnp.sqrt(len(x)),
                effective_fine_forcing_norm=jnp.sqrt(d),
                coarse_min_eigenvalue=jnp.linalg.eigvalsh(gram)[0],
                coarse_disequilibrium_norm=jnp.linalg.norm(state['z']))


def _arm_from_state(state, reference, arm):
    kappa, nu = arm
    c0 = _project(reference['F0'], state['JC'], state['gram'])
    a = _project(state['J'].T@reference['eH0']/len(reference['eH0']), state['JC'], state['gram'])
    b = state['F']-a
    # This form makes the natural arm exactly equal to F (and g below).
    return state['F']+(kappa-1.)*(a-c0)+(nu-1.)*b


def arm_effective(p, x, y, reference, arm):
    return _arm_from_state(decomposition(p, x, y), reference, arm)


def step_components(p, x, y, reference, arm):
    """Return modified gradient, arm effective field, natural F, and R.

    The ordinary tracking field is evaluated at each arm's own state. It is
    not frozen or silently removed. arm may be a two-element JAX array.
    """
    state = decomposition(p, x, y)
    kappa, nu = arm
    c0 = _project(reference['F0'], state['JC'], state['gram'])
    a = _project(state['J'].T@reference['eH0']/len(x), state['JC'], state['gram'])
    correction = (kappa-1.)*(a-c0)+(nu-1.)*(state['F']-a)
    return state['g']+correction, state['F']+correction, state['F'], state['R']


def gradient(p, x, y, reference, arm):
    """Modified full gradient; the natural arm equals ordinary GD exactly."""
    return step_components(p, x, y, reference, arm)[0]


def initial_arm_diagnostics(p, x, y, reference):
    """Six-arm directional predictions, including full-GD slope acceleration.

    The acceleration is D g_arm[g0], so two-step parameter contrasts begin
    with eta**2 times the acceleration contrast. Before sign crossings, the
    signed slope acceleration is sign(a)*acceleration_a. Actual positive
    travel must be computed from discrete |a| increments by the runner.
    """
    state = decomposition(p, x, y)
    f0, g0, r0 = state['F'], state['g'], state['R']
    q2 = f0@f0
    w = (len(p)-1)//3

    def one(arm):
        field = lambda v: arm_effective(v, x, y, reference, arm)
        df_f = jax.jvp(field, (p,), (f0,))[1]
        df_r = jax.jvp(field, (p,), (r0,))[1]
        dg_g = jax.jvp(lambda v: gradient(v, x, y, reference, arm), (p,), (g0,))[1]
        actual = _ratio(-f0@(df_f+df_r), q2)
        return dict(k_pure=_ratio(-f0@df_f, q2), loaded=f0@df_r,
                    k=actual, k_actual=actual,
                    initial_force_mismatch=jnp.linalg.norm(field(p)-f0),
                    slope_acceleration=dg_g[:w],
                    signed_slope_acceleration=jnp.sign(p[:w])*dg_g[:w])

    return jax.vmap(one)(jnp.asarray(ARMS, dtype=p.dtype))


def fork(p, x, y):
    """Return frozen reference, scalar diagnostics, and six-arm predictions."""
    state = decomposition(p, x, y)
    reference = dict(F0=state['F'], eH0=state['eH'])
    return dict(reference=reference, diagnostics=diagnostics(p, x, y),
                arm_diagnostics=initial_arm_diagnostics(p, x, y, reference),
                slope_initial_velocity=-state['g'][:(len(p)-1)//3])
