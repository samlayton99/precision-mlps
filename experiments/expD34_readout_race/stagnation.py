"""Modal-loss interventions and signed coarse-balance diagnostics for actual GD."""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from . import targets, transport as tr

ARMS = ('full', 'five_mode', 'ten_mode', 'remove_lower')
MODES = dict(full=(), five_mode=(0, 1, 2, 3, 9), ten_mode=tuple(range(10)),
             remove_lower=tuple(range(2, 9)))


def mode_matrix(x, arm):
    return tr.basis(x, 9)[:, MODES[arm]]


def coarse_jacobian(z, x, h, s, q):
    """Two output modes by all raw parameters, in a,b,c,d order (equal masses)."""
    m = len(x)
    return jnp.concatenate(((q.T @ (x[:, None]*s)/m)*z[2],
                            (q.T @ s/m)*z[2], q.T @ h/m,
                            jnp.mean(q, axis=0)[:, None]), axis=1)


def initial(z, d):
    state = tr.initialize(z, d)
    return state | dict(tracking_travel=jnp.zeros_like(state['positive']),
        tracking_path=jnp.zeros_like(state['energy']),
        slope_energy_flow=jnp.zeros_like(state['d']),
        slope_energy_discrete=jnp.zeros_like(state['d']),
        min_coarse_eigenvalue=jnp.full_like(state['d'], 1e300))


@lru_cache(maxsize=16)
def advance_factory(m, width, arm, eta):
    x = jnp.asarray(targets.grid(m))
    modes = jnp.asarray(mode_matrix(np.asarray(x), arm))
    qc = jnp.asarray(tr.basis(np.asarray(x), 9)[:, :2])
    weights = jnp.ones(width)/width

    def one(state, y, count):
        def step(_, old):
            z, d = old['z'], old['d']
            g, gd, r, p, h, s = tr.field(z, d, x, y, weights, width, modes,
                                        remove=arm == 'remove_lower')
            jc = coarse_jacobian(z, x, h, s, qc)
            C = jc @ jc.T
            # Exact full-complement coarse projection: no modal tail is dropped.
            rhs = jc @ jnp.r_[g.ravel(), gd]
            det = C[0, 0]*C[1, 1]-C[0, 1]*C[1, 0]
            tracking = jnp.array([C[1, 1]*rhs[0]-C[0, 1]*rhs[1],
                                  C[0, 0]*rhs[1]-C[1, 0]*rhs[0]])/det
            qg = jc.T @ tracking
            qz = qg[:-1].reshape(z.shape)
            eigen = .5*(jnp.trace(C)-jnp.sqrt((C[0, 0]-C[1, 1])**2+4*C[0, 1]**2))
            zn, dn = z-eta*g, d-eta*gd
            delta = jnp.abs(zn[0])-jnp.abs(z[0])
            energy = jnp.sum(g*g, axis=1)
            u = x[:, None]*z[0]+z[1]
            balance_flow = 2*jnp.sum(z[2]*(p @ (h-u*s))/m)
            return dict(z=zn, d=dn,
                positive=old['positive']+jnp.maximum(delta, 0),
                negative=old['negative']+jnp.maximum(-delta, 0),
                path=old['path']+eta*jnp.sqrt(energy[0]),
                energy=old['energy']+eta*jnp.r_[energy, gd*gd],
                balance_flow=old['balance_flow']+eta*balance_flow,
                balance_discrete=old['balance_discrete']+eta**2*(energy[0]+energy[1]-energy[2]),
                crossing=old['crossing']+jnp.mean(delta+eta*jnp.sign(z[0])*g[0]),
                tracking_travel=old['tracking_travel']-eta*jnp.sign(z[0])*qz[0],
                tracking_path=old['tracking_path']+eta*jnp.r_[jnp.linalg.norm(qz, axis=1), jnp.abs(qg[-1])],
                slope_energy_flow=old['slope_energy_flow']-eta*z[0] @ g[0],
                slope_energy_discrete=old['slope_energy_discrete']+.5*eta**2*energy[0],
                min_coarse_eigenvalue=jnp.minimum(old['min_coarse_eigenvalue'], eigen))
        return jax.lax.fori_loop(0, count, step, state)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def diagnostics(z, d, x, y, arm='full', degree=65, basis=None):
    """Decompose the gradient of THIS arm's loss, retaining original mode labels.

    The instantaneous coarse projection is a diagnostic, never a training update.
    Finite-degree tracking and omitted force are reported separately; exact full-
    complement tracking is also retained for comparison with stepwise integrals.
    """
    a, b, c = z
    m, width = len(x), len(a)
    q = tr.basis(x, degree) if basis is None else basis
    u = x[:, None]*a+b
    h = np.tanh(u)
    exp = np.exp(-2*np.abs(u)); s = 4*exp/(1+exp)**2
    r = h @ c+d-y
    e = q.T @ r/m
    coefficients = np.ones(q.shape[1])
    if arm in ('five_mode', 'ten_mode'):
        coefficients[:] = 0
        coefficients[list(MODES[arm])] = 1
    elif arm == 'remove_lower':
        coefficients[2:9] = 0
    elif arm != 'full':
        raise ValueError(arm)
    ew = coefficients*e
    if arm in ('five_mode', 'ten_mode'):
        p = q @ ew
    elif arm == 'remove_lower':
        p = r-q[:, 2:9] @ e[2:9]
    else:
        p = r
    J = dict(a=(q.T @ (x[:, None]*s)/m)*c, b=(q.T @ s/m)*c,
             c=q.T @ h/m, d=q.mean(axis=0)[:, None])
    gradients = dict(a=c*(x @ (p[:, None]*s))/m, b=c*(p @ s)/m,
                     c=h.T @ p/m, d=np.array([p.mean()]))
    full_gradients = dict(a=c*(x @ (r[:, None]*s))/m, b=c*(r @ s)/m,
                          c=h.T @ r/m, d=np.array([r.mean()]))
    K = sum(j @ j.T for j in J.values())
    C = K[:2, :2]
    eigen = np.linalg.eigvalsh(C)
    if eigen[0] <= 64*np.finfo(float).eps*max(1., eigen[-1]):
        raise ValueError('Unresolved coarse inverse; do not regularize into a regime claim')
    B = np.linalg.solve(C, K[:2, 2:])
    zc = ew[:2]+B @ ew[2:]
    exact_zc = np.linalg.solve(C, sum(j[:2] @ gradients[k] for k, j in J.items()))
    T = {k: j[2:].T-j[:2].T @ B for k, j in J.items()}
    effective = {k: t @ ew[2:] for k, t in T.items()}
    tracking = {k: j[:2].T @ zc for k, j in J.items()}
    exact_tracking = {k: j[:2].T @ exact_zc for k, j in J.items()}
    omitted = {k: gradients[k]-effective[k]-tracking[k] for k in J}
    modal = T['a']*ew[None, 2:]
    forces = dict(generated=modal[:, :7].sum(axis=1), hard=modal[:, 7],
                  higher=modal[:, 8:].sum(axis=1), effective=effective['a'],
                  tracking=tracking['a'], omitted=omitted['a'], actual=gradients['a'])
    outward = {k: -np.sign(a)*v for k, v in forces.items()}
    scalar = dict(half_mse=np.mean(r*r)/2, objective=np.mean(p*p)/2,
        mean_gamma=np.mean(abs(a)), median_gamma=np.median(abs(a)),
        q90_gamma=np.quantile(abs(a), .9), max_gamma=np.max(abs(a)),
        slope_energy=a @ a/2, bias_energy=b @ b/2, readout_l2=np.linalg.norm(c),
        balance=np.sum(a*a+b*b-c*c), coarse_min_eigenvalue=eigen[0],
        coarse_output_0=e[0]+q[:, 0] @ y/m, coarse_output_1=e[1]+q[:, 1] @ y/m,
        hard_residual=e[9], fine_residual_norm=np.linalg.norm(e[2:]),
        tracking_error=np.linalg.norm(zc), exact_tracking_error=np.linalg.norm(exact_zc),
        decomposition_error=np.linalg.norm(gradients['a']-sum(forces[k] for k in ('generated', 'hard', 'higher', 'tracking', 'omitted'))))
    for k in J:
        scalar['own_state_full_gradient_defect_'+k] = np.linalg.norm(gradients[k]-full_gradients[k])
        for name, vectors in [('gradient', gradients), ('effective', effective),
                              ('tracking', tracking), ('omitted', omitted), ('exact_tracking', exact_tracking)]:
            scalar[name+'_'+k+'_norm'] = np.linalg.norm(vectors[k])
    for name, force in forces.items():
        scalar[name+'_outward'] = np.mean(outward[name])
        scalar[name+'_slope_energy_velocity'] = -a @ force
        scalar[name+'_positive_velocity'] = np.maximum(outward[name], 0).mean()
        scalar[name+'_outward_fraction'] = np.mean(outward[name] > 0)
    for i in range(q.shape[1]):
        scalar[f'residual_{i}'] = e[i]
        if i >= 2:
            scalar[f'mode_{i}_outward'] = -np.mean(np.sign(a)*modal[:, i-2])
    for threshold in (.1, .2, 1., 3.2, 16.):
        scalar[f'fraction_{threshold:g}'] = np.mean(abs(a) >= threshold)
    for name, matrix in [('c', J['c'][:2] @ J['c'][:2].T),
                         ('cd', J['c'][:2] @ J['c'][:2].T+J['d'][:2] @ J['d'][:2].T)]:
        values, vectors = np.linalg.eigh(matrix)
        resolved = values[0] > 64*np.finfo(float).eps*max(1., values[-1])
        scalar[f'M_{name}_min_eigenvalue'] = values[0]
        scalar[f'M_{name}_resolved'] = resolved
        invroot = (vectors/np.sqrt(values)) @ vectors.T if resolved else None
        scalar[f'Omega_{name}'] = np.sqrt(np.linalg.eigvalsh(invroot @ C @ invroot)[-1]) if resolved else np.nan
    return scalar, dict(z=z, d=np.array(d), residual_modes=e, gradient=np.stack([gradients[k] for k in 'abc']),
        gradient_d=gradients['d'], effective_a=effective['a'], tracking_a=tracking['a'],
        exact_tracking_a=exact_tracking['a'], generated_a=forces['generated'], hard_a=forces['hard'],
        mode_outward=-np.sign(a)[None, :]*modal.T, T_a=T['a'], J_C_a=J['a'][:2], B=B)


def force_probes(z, d, x, y, degree=65, basis=None):
    """Fixed-parameter penalty/target probes; recompute raw tracking separately."""
    row, arr = diagnostics(z, d, x, y, degree=degree, basis=basis)
    e = arr['residual_modes']; a = z[0]
    outward = -np.sign(a)/len(a)
    lower = arr['generated_a']; other = arr['effective_a']-lower
    lower_v = outward @ lower; other_v = outward @ other
    critical = -other_v/lower_v if lower_v != 0 else np.nan
    probes = []
    for weight in (0., critical/2, critical, 2*critical, 1.):
        if not np.isfinite(weight) or weight < 0:
            continue
        force = other+weight*lower
        # Coarse target and parameters stay fixed; this changes raw disequilibrium.
        change = (weight-1)*(arr['B'][:, :7] @ e[2:9])
        tracking = arr['tracking_a']+arr['J_C_a'].T @ change
        probes.append(dict(probe='lower_penalty', value=weight, critical=critical,
            effective_outward=outward @ force, tracking_outward=outward @ tracking,
            actual_outward=outward @ (force+tracking)+row['omitted_outward']))
    y9 = (tr.basis(x, 9)[:, 9] @ y)/len(x)
    target_derivative = -arr['T_a'][:, 7]
    denom = outward @ target_derivative
    critical_y = y9-row['effective_outward']/denom if denom != 0 else np.nan
    for value in (y9, critical_y/2, critical_y, 2*critical_y):
        if not np.isfinite(value) or value < 0:
            continue
        change = value-y9
        force = arr['effective_a']+change*target_derivative
        tracking = arr['tracking_a']-change*(arr['J_C_a'].T @ arr['B'][:, 7])
        probes.append(dict(probe='hard_target', value=value, critical=critical_y,
            effective_outward=outward @ force, tracking_outward=outward @ tracking,
            actual_outward=outward @ (force+tracking)+row['omitted_outward']))
    return probes
