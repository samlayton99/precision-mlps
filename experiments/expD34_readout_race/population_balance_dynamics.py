"""Exact moment balances and global Taylor-remainder comparison bounds.

All norms use the empirical probability measure. Run numerical work on Modal.
The polynomial quantities are observables of the evolving exact tanh state.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp

from . import mechanism_persistence_kernel as kernel

A3 = math.sqrt(6)/8
A5 = (2/15)*2**2.5*5**2.5/6**3
A7 = (17/315)*2**3.5*7**3.5/8**4
C3, C5, C7 = 4*A3, 6*A5, 8*A7
LEDGER = ('generated', 'target', 'higher', 'compensation', 'tracking',
          'A_linear', 'C_linear', 'm_linear', 'M_linear',
          'tracking_path', 'coarse0', 'coarse1',
          'Delta_quadratic', 'A_quadratic', 'C_quadratic', 'm_quadratic', 'M_quadratic')


def moments(p):
    a, b, c = p[:-1].reshape(3, -1)
    A, C, B = a@a, c@c, b@b
    m = a@c
    valid = (A*C > 0) & (m != 0)
    rho = jnp.where(A*C > 0, m/jnp.sqrt(jnp.where(A*C > 0, A*C, 1.)), jnp.nan)
    return dict(A=A, C=C, bias2=B, M=A+B+C, m=m, Delta=(A+B-C)/2,
                rho=rho, log_alignment_valid=valid, alignment_margin=m*m-.01*A*C)


def flux(p, x, y, kind):
    state = kernel.decomposition(p, x, y)
    F, J, jc, q = (state[k] for k in ('F', 'J', 'JC', 'basis'))
    R = state['R'] if kind == 'gd' else jnp.zeros_like(F)
    V = F+R
    a, b, c = p[:-1].reshape(3, -1)
    va, vb, vc = V[:-1].reshape(3, -1)
    u = x[:, None]*a+b
    phi = jnp.tanh(u)
    kk = (phi-u*(1-phi**2))@c
    ee = (3*phi-u*(1-phi**2)-2*u)@c
    kc = q.T@kk/len(x)
    fh, gh, eh = state['fH'], state['yH'], state['eH']
    generated = -2*jnp.mean(fh**2)
    target = 2*jnp.mean(gh*fh)
    higher = jnp.mean(eh*kernel._fine(ee, q))
    compensation = -state['balance']@kc
    tracking = state['z']@kc if kind == 'gd' else jnp.array(0.)
    coarse = -jc@R
    rates = jnp.array([generated, target, higher, compensation, tracking,
                       -2*a@va, -2*c@vc, -c@va-a@vc, -2*p[:-1]@V[:-1],
                       jnp.linalg.norm(R), coarse[0], coarse[1], 0., 0., 0., 0., 0.])
    quadratic = jnp.array([0.]*12+[(va@va+vb@vb-vc@vc)/2,
                                  va@va, vc@vc, va@vc, V[:-1]@V[:-1]])
    return V, rates, quadratic


@partial(jax.jit, static_argnames=('kind', 'steps'))
def advance(p, ledger, x, y, dt, kind, steps):
    def step(_, carry):
        p, acc = carry
        v1, s1, quadratic = flux(p, x, y, kind)
        if kind == 'gd':
            return p-dt*v1, acc+dt*s1+dt*dt*quadratic
        v2, s2, _ = flux(p-dt*v1/2, x, y, kind)
        v3, s3, _ = flux(p-dt*v2/2, x, y, kind)
        v4, s4, _ = flux(p-dt*v3, x, y, kind)
        return p-dt*(v1+2*v2+2*v3+v4)/6, acc+dt*(s1+2*s2+2*s3+s4)/6
    return jax.lax.fori_loop(0, steps, step, (p, ledger))


@jax.jit
def diagnostics(p, x, y):
    state = kernel.decomposition(p, x, y)
    F, R, J, jc, gram, basis = (state[k] for k in ('F','R','J','JC','gram','basis'))
    result = moments(p)
    M = result['M']
    a, b, c = p[:-1].reshape(3, -1)
    w = len(a)
    u = x[:, None]*a+b
    phi = jnp.tanh(u)
    r2 = a*a+b*b+c*c
    sums = {k: jnp.sum(r2**(k/2)) for k in (4,6,8,10,14)}
    norm = lambda v: jnp.sqrt(jnp.mean(v*v))
    fh, g, eh = state['fH'], state['yH'], state['eH']
    gn = norm(g)
    gp = jnp.zeros_like(p)
    jpoly = jnp.zeros_like(J)
    fpoly = jnp.zeros_like(x)
    for k, coefficient in ((3, -1/3), (5, 2/15)):
        fp = coefficient*(u**k)@c
        jp = coefficient*jnp.concatenate((k*c*x[:,None]*u**(k-1),
                                          k*c*u**(k-1), u**k, jnp.zeros((len(x),1))), axis=1)
        jh = kernel._fine(jp, basis)
        target_gradient = jp.T@g/len(x)
        gp += target_gradient
        jpoly += jh
        fpoly += kernel._fine(fp, basis)
        result[f'q{k}'] = norm(kernel._fine(fp, basis))*w**((k-1)/2)/M**((k+1)/2)
        result[f'n{k}'] = norm(fp)*w**((k-1)/2)/M**((k+1)/2)
        result[f'j{k}'] = jnp.sqrt(jnp.sum(jh*jh)/len(x))*w**((k-1)/2)/M**(k/2)
        result[f'g{k}'] = jnp.linalg.norm(target_gradient)*w**((k-1)/2)/M**(k/2)
    sigma = jnp.sqrt(jnp.linalg.eigvalsh(gram)[0])
    v = jnp.r_[a, b, -c, 0.]
    vp = kernel._project(v, jc, gram)
    K0, E0, J0 = 2*A3*sums[4], 2*A5*sums[6], C3*jnp.sqrt(sums[6])
    remainder_gradient = C7*gn*jnp.sqrt(sums[14])
    target_bound = jnp.linalg.norm(gp)+remainder_gradient
    generated_allowance = (E0+K0*J0/sigma)**2/8
    track_delta = -v@R
    signed_upper = generated_allowance+vp@gp+jnp.linalg.norm(vp)*remainder_gradient
    raw_g = J.T@g/len(x)
    full_v, full_s, _ = flux(p, x, y, 'gd')
    A, C, m = result['A'], result['C'], result['m']
    Ad, Cd, md = full_s[5:8]
    valid = result['log_alignment_valid']
    log_rate = jnp.where(valid, md/jnp.where(m != 0,m,1.)-.5*Ad/jnp.where(A>0,A,1.)
                        -.5*Cd/jnp.where(C>0,C,1.), jnp.nan)
    residual = kernel.output(p,x)-y
    result.update(width=w, loss=jnp.mean(residual**2)/2, relative_error=norm(residual)/norm(y),
                  fine_error=norm(eh), fine_output=norm(fh), target_fine=gn, target_norm=norm(y),
                  F_norm=jnp.linalg.norm(F), R_norm=jnp.linalg.norm(R), sigma=sigma,
                  slope_rms=jnp.sqrt(A/w), coarse0=(basis.T@residual/len(x))[0],
                  coarse1=(basis.T@residual/len(x))[1],
                  nonlinear_coarse1=(basis.T@(kernel.output(p,x)-p[-1]-u@c)/len(x))[1],
                  Delta_dot_effective=-v@F, Delta_dot_full=-v@full_v,
                  generated_allowance=generated_allowance,
                  Delta_upper_effective=signed_upper, Delta_upper_full=signed_upper+track_delta,
                  target_gradient_bound=target_bound, target_gradient_actual=jnp.linalg.norm(raw_g),
                  delta_bound_slack=signed_upper+v@F,
                  log_alignment_rate=log_rate, alignment_margin_rate=2*m*md-.01*(Ad*C+A*Cd),
                  target_tangent_polynomial=vp@gp, target_tangent_remainder=jnp.linalg.norm(vp)*remainder_gradient,
                  remainder_capacity=A7*sums[8], remainder_sensitivity=C7*jnp.sqrt(sums[14]),
                  capacity_remainder_check=norm(fh-fpoly),
                  sensitivity_remainder_check=jnp.sqrt(jnp.sum((kernel._fine(J,basis)-jpoly)**2)/len(x)),
                  moment_delta_identity=full_s[:5].sum()+v@full_v,
                  gradient_poly_error=jnp.linalg.norm(raw_g-gp))
    for k, value in sums.items():
        result[f'C{k}'] = value*w**(k/2-1)/M**(k/2)
    return result


def comparison(row, mass):
    """Proved force bound at an allowed total hidden squared norm, using shape scalars."""
    w = row['width']
    cap = row['q3']*mass**2/w+row['q5']*mass**3/w**2+A7*row['C8']*mass**4/w**3
    jac = row['j3']*mass**1.5/w+row['j5']*mass**2.5/w**2+C7*math.sqrt(row['C14'])*mass**3.5/w**3
    target = row['g3']*mass**1.5/w+row['g5']*mass**2.5/w**2+C7*row['target_fine']*math.sqrt(row['C14'])*mass**3.5/w**3
    nonlinear = row['n3']*mass**2/w+row['n5']*mass**3/w**2+A7*row['C8']*mass**4/w**3
    return dict(capacity=cap, sensitivity=jac, target=target, force=jac*cap+target, nonlinear=nonlinear)
