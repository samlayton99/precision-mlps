"""Width-dependent characteristic transport and exact-tanh modal projection.

Particle weights represent probability mass, independently of physical width.
The characteristic velocity is NOT the Euclidean gradient in quadrature-node
coordinates: that gradient has an extra factor width*weight per particle.
"""
from __future__ import annotations

from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def basis(x, degree):
    """Nested polynomials orthonormal in the empirical probability metric."""
    q, r = np.linalg.qr(legvander(np.asarray(x), degree), mode="reduced")
    return q * np.sign(np.diag(r)) * np.sqrt(len(x))


def law_initial(width, nodes):
    points, mass = leggauss(nodes)
    z = np.array(np.meshgrid(points, points, points, indexing="ij")).reshape(3, -1)
    weights = np.prod(np.array(np.meshgrid(mass/2, mass/2, mass/2, indexing="ij")), axis=0).ravel()
    return z*np.sqrt(6/(width+1)), 0., weights


def field(z, d, x, y, weights, width, modes):
    """Unscaled characteristic negative velocities and both residuals.

An empty mode matrix means the full empirical residual. Otherwise all four
parameter blocks train the squared norm of the projected residual.
"""
    a, b, c = z
    u = x[:, None]*a+b
    h = jnp.tanh(u)
    exp = jnp.exp(-2*jnp.abs(u))
    s = 4*exp/(1+exp)**2
    r = h @ (width*weights*c)+d-y
    projected = modes @ (modes.T @ r/len(x)) if modes.shape[1] else r
    g = jnp.stack((c*(x @ (projected[:, None]*s))/len(x),
                   c*(projected @ s)/len(x), h.T @ projected/len(x)))
    return g, jnp.mean(projected), r, projected, h, s


def initialize(z, d):
    z, d = jnp.asarray(z), jnp.asarray(d)
    return dict(z=z, d=d, positive=jnp.zeros_like(z[:, 0]), negative=jnp.zeros_like(z[:, 0]),
                path=jnp.zeros_like(d), energy=jnp.zeros((len(d), 4)),
                balance_flow=jnp.zeros_like(d), balance_discrete=jnp.zeros_like(d),
                crossing=jnp.zeros_like(d))


@lru_cache(maxsize=64)
def advance_factory(m, width, degree, eta, kappa):
    from .targets import grid
    x = jnp.asarray(grid(m))
    modes = jnp.asarray(basis(np.asarray(x), degree)) if degree >= 0 else jnp.empty((m, 0))
    rates = jnp.array([1., 1., kappa])[:, None]

    def one(state, y, weights, count):
        def step(_, old):
            z, d = old["z"], old["d"]
            g, gd, r, p, h, s = field(z, d, x, y, weights, width, modes)
            zn = z-eta*rates*g
            dn = d-eta*kappa*gd
            delta = jnp.abs(zn[0])-jnp.abs(z[0])
            energy = width*jnp.sum(weights*g*g, axis=1)
            u = x[:, None]*z[0]+z[1]
            balance_flow = 2*width*jnp.sum(weights*z[2]*(p @ (h-u*s))/m)
            correction = energy[0]+energy[1]-kappa*energy[2]
            return dict(z=zn, d=dn,
                positive=old["positive"]+jnp.maximum(delta, 0),
                negative=old["negative"]+jnp.maximum(-delta, 0),
                path=old["path"]+eta*jnp.sqrt(energy[0]),
                energy=old["energy"]+eta*jnp.array([energy[0], energy[1], kappa*energy[2], kappa*gd*gd]),
                balance_flow=old["balance_flow"]+eta*balance_flow,
                balance_discrete=old["balance_discrete"]+eta**2*correction,
                crossing=old["crossing"]+jnp.sum(weights*(delta+eta*jnp.sign(z[0])*g[0])))
        return jax.lax.fori_loop(0, count, step, state)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None)))


TRACE = ("half_mse", "projected_half_mse", "mean_gamma", "max_gamma", "readout_l2",
         "hidden_l2", "xi", "signed_alignment", "coarse_norm", "slope_share",
         "slope_defect_norm", "balance", "balance_velocity", "fraction_gamma1",
         "fraction_gamma3.2", "fraction_gamma16", "signed_velocity",
         "fraction_lambda005", "fraction_lambda025")


@lru_cache(maxsize=64)
def measure_factory(m, width, degree, kappa, n=128):
    from .targets import grid
    x = jnp.asarray(grid(m))
    modes = jnp.asarray(basis(np.asarray(x), degree)) if degree >= 0 else jnp.empty((m, 0))
    def one(z, d, y, weights):
        g, gd, r, p, h, s = field(z, d, x, y, weights, width, modes)
        # Full slope field at the forecast's OWN state; never feed this into training.
        full_ga = z[2]*(x @ (r[:, None]*s))/m
        ga2 = width*jnp.sum(weights*full_ga**2)
        full_gb = z[2]*(r @ s)/m
        full_gc = h.T @ r/m
        all2 = ga2+width*jnp.sum(weights*(full_gb**2+kappa*full_gc**2))+kappa*jnp.mean(r)**2
        r2 = jnp.mean(r*r)
        velocity = -jnp.sum(weights*jnp.sign(z[0])*full_ga)
        coarse2 = jnp.mean(r)**2+(x @ r/m)**2/jnp.mean(x*x)
        u = x[:, None]*z[0]+z[1]
        balance_v = 2*width*jnp.sum(weights*z[2]*(r @ (h-u*s))/m)
        gamma = jnp.abs(z[0])
        return jnp.array([r2/2, jnp.mean(p*p)/2, weights @ gamma, gamma.max(),
            jnp.sqrt(width*jnp.sum(weights*z[2]**2)),
            jnp.sqrt(width*jnp.sum(weights*(z[0]**2+z[1]**2))),
            jnp.sqrt(ga2/r2), jnp.sqrt(width)*velocity/jnp.sqrt(ga2), jnp.sqrt(coarse2), ga2/all2,
            jnp.sqrt(width*jnp.sum(weights*(g[0]-full_ga)**2)),
            width*jnp.sum(weights*(z[0]**2+z[1]**2-z[2]**2/kappa)), balance_v,
            *[weights @ (gamma >= value) for value in (1., 3.2, 16.)], velocity,
            weights @ (gamma >= .05*n/2), weights @ (gamma >= .25*n/2)])
    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0)))


def modal_diagnostics(z, d, x, y, weights, width, degree=65, kappa=1., *, rates=None, modes=None):
    """Exact projected kernel blocks and full-field omitted-mode accounting."""
    a, b, c = np.asarray(z)
    q = basis(x, degree) if modes is None else modes
    ra, rb, rc, rd = (1., 1., kappa, kappa) if rates is None else rates
    u = x[:, None]*a+b
    h = np.tanh(u)
    exp = np.exp(-2*np.abs(u)); s = 4*exp/(1+exp)**2
    r = h @ (width*weights*c)+d-y
    e = q.T @ r/len(x)
    root = np.sqrt(width*weights)
    ja = (q.T @ (x[:, None]*s)/len(x))*(root*c)
    jb = (q.T @ s/len(x))*(root*c)
    jc = (q.T @ h/len(x))*root
    jd = q.mean(axis=0)
    kernels = dict(a=ra*(ja @ ja.T), b=rb*(jb @ jb.T), c=rc*(jc @ jc.T), d=rd*np.outer(jd, jd))
    K = sum(kernels.values())
    rtail = r-q @ e
    ga = c*(x @ (r[:, None]*s))/len(x)
    gb = c*(r @ s)/len(x)
    gc = h.T @ r/len(x)
    full_ga = root*ga
    tail_ga = root*c*(x @ (rtail[:, None]*s))/len(x)
    cc, ch = K[:2, :2], K[:2, 2:]
    eigen = np.linalg.eigvalsh(cc)
    resolved = eigen[0] > 64*np.finfo(float).eps*max(1., eigen[-1])
    udot = -ra*x[:, None]*ga-rb*gb
    sdot = -2*h*s*udot
    jadot = (q.T @ (x[:, None]*sdot)/len(x))*(root*c)-(q.T @ (x[:, None]*s)/len(x))*(root*rc*gc)
    jbdot = (q.T @ sdot/len(x))*(root*c)-(q.T @ s/len(x))*(root*rc*gc)
    jcdot = (q.T @ (s*udot)/len(x))*root
    Kdot = ra*(jadot @ ja.T+ja @ jadot.T)+rb*(jbdot @ jb.T+jb @ jbdot.T)+rc*(jcdot @ jc.T+jc @ jcdot.T)
    edot = -ra*ja @ (root*ga)-rb*jb @ (root*gb)-rc*jc @ (root*gc)-rd*jd*np.mean(r)
    scalar = dict(full_slope_norm=float(np.linalg.norm(full_ga)),
        modal_slope_norm=float(np.linalg.norm(ja.T @ e)),
        omitted_slope_norm=float(np.linalg.norm(tail_ga)),
        gradient_accounting_error=float(np.linalg.norm(full_ga-ja.T @ e-tail_ga)),
        coarse_min_eigenvalue=float(eigen[0]), coarse_inverse_resolved=bool(resolved),
        coarse_regeneration_norm=float(np.linalg.norm(ch @ e[2:])),
        coarse_fitting_norm=float(np.linalg.norm(cc @ e[:2])),
        modal_slope_share=float(e @ kernels['a'] @ e/(e @ K @ e+1e-300)),
        kernel_drift_norm=float(np.linalg.norm(Kdot)),
        modal_velocity_defect=float(np.linalg.norm(edot+K @ e)))
    if resolved:
        closure = np.linalg.solve(cc, ch)
        tracking = e[:2]+closure @ e[2:]
        effective = ja[2:].T-ja[:2].T @ closure
        S = K[2:, 2:]-ch.T @ closure
        slow = effective @ e[2:]
        transient = ja[:2].T @ tracking
        scalar.update(schur_min_eigenvalue=float(np.linalg.eigvalsh(S)[0]),
            coarse_drift_ratio=float(np.linalg.norm(Kdot[:2, :2])/eigen[0]**2),
            tracking_error=float(np.linalg.norm(tracking)),
            effective_slope_norm=float(np.linalg.norm(slow)),
            transient_slope_norm=float(np.linalg.norm(transient)),
            slow_transient_alignment=float(slow @ transient/(np.linalg.norm(slow)*np.linalg.norm(transient)+1e-300)),
            decomposition_error=float(np.linalg.norm(ja.T @ e-slow-transient)))
    return scalar, dict(**{"K_"+key: value for key, value in kernels.items()}, residual_modes=e,
                       J_a=ja, J_b=jb, J_c=jc, J_d=jd, K_dot=Kdot, residual_velocity=edot,
                       J_a_dot=jadot, gradient=np.stack((ga, gb, gc)), residual=r)
