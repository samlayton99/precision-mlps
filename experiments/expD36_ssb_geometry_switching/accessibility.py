"""Smooth finite-budget readout learning, with the reference residual held fixed."""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from experiments.expD35_optimization_exploration import core
from experiments.expD06_fixed_center_scales import higher_order


def filter_values(values, tau):
    """f(x)=(1-exp(-2*tau*x))/x and f'(x), continued at zero."""
    x = jnp.maximum(values, 0.)
    t = 2*tau
    z = t*x
    small = z < 1e-3
    safe = jnp.where(small, 1., x)
    f = jnp.where(small, t*(1-z/2+z*z/6-z**3/24+z**4/120-z**5/720),
                  -jnp.expm1(-z)/safe)
    df = jnp.where(small, t*t*(-.5+z/3-z*z/8+z**3/30-z**4/144),
                   ((z+1)*jnp.exp(-z)-1)/(safe*safe))
    return f, df


@jax.custom_jvp
def matrix_filter(gram, tau):
    values, vectors = jnp.linalg.eigh((gram+gram.T)/2)
    f, _ = filter_values(values, tau)
    return (vectors*f)@vectors.T


@matrix_filter.defjvp
def matrix_filter_jvp(primals, tangents):
    gram, tau = primals
    dgram, dtau = tangents
    values, vectors = jnp.linalg.eigh((gram+gram.T)/2)
    values = jnp.maximum(values, 0.)
    f, _ = filter_values(values, tau)
    gap = values[:, None]-values[None, :]
    midpoint = (values[:, None]+values[None, :])/2
    _, derivative = filter_values(midpoint, tau)
    close = jnp.abs(gap) <= 1e-6*jnp.maximum(1/(2*tau), midpoint)
    divided = (f[:, None]-f[None, :])/jnp.where(close, 1., gap)
    divided = jnp.where(close, derivative, divided)
    transformed = vectors.T@((dgram+dgram.T)/2)@vectors
    change = vectors@(divided*transformed)@vectors.T
    change += ((vectors*(2*jnp.exp(-2*tau*values)))@vectors.T)*dtau
    return (vectors*f)@vectors.T, change


def gain(a, residual, tau):
    """Learnable residual fraction; avoids subtracting nearly equal Q values."""
    b = a.T@residual
    return (b@matrix_filter(a.T@a, tau)@b)/(residual@residual)


def gain_svd(a, residual, tau):
    """Independent value audit; no differentiation of the singular vectors."""
    u, s, _ = jnp.linalg.svd(a, full_matrices=False)
    coefficients = u.T@residual
    return jnp.sum(-jnp.expm1(-2*tau*s*s)*coefficients**2)/(residual@residual)


@lru_cache(maxsize=16)
def problem(n, coordinates, target, samples=16):
    g = core.old.geometry(n)
    x = jnp.linspace(-1, 1, samples*n+1)
    y = core.target(x, target)
    distance = x[:, None]-jnp.asarray(g.centers)
    transform = jnp.asarray(higher_order.readout_map(g, core.ALIASES[coordinates]))
    root = np.sqrt(len(x))

    def readout(lam):
        phi = core.old.tanh(distance*(lam/g.h))
        return jnp.concatenate((jnp.ones((len(x), 1)), phi), axis=1)@transform/root

    def residual(z):
        return readout(z[g.width+1:])@z[:g.width+1]-y/root

    def loss(z):
        r = residual(z)
        return .5*(r@r)

    def fraction(lam, r, tau):
        return gain(readout(lam), r, tau)

    # Values use the same physical summation as the pinned trainer; A only
    # describes counterfactual readout corrections at a held reference residual.
    def physical_residual(z):
        c, gamma = core.physical(z, g, coordinates)
        return (c[0]+core.old.tanh(distance*gamma)@c[1:]-y)/root

    return g, jax.jit(readout), jax.jit(physical_residual), jax.jit(loss), jax.jit(jax.value_and_grad(fraction))


def exposure_beta(en, es, uncertainty, secondary_en, secondary_es, secondary_uncertainty):
    """Minimum resolved-positive mixture; secondary horizon may not deteriorate."""
    margin = 3*uncertainty
    if not np.all(np.isfinite([en, es, uncertainty, secondary_en, secondary_es, secondary_uncertainty])):
        return None
    if en > margin or es <= margin or es <= en:
        return None
    beta = float(np.clip((margin-en)/(es-en), 0., 1.))
    beta = min(1., np.nextafter(beta, 1.))
    if (1-beta)*secondary_en+beta*secondary_es < -3*secondary_uncertainty:
        return None
    return beta


def mix_metric(matrix, gradient, beta):
    norm = jnp.linalg.norm(gradient)
    scale = jnp.linalg.norm(matrix@gradient)/jnp.where(norm > 0, norm, 1.)
    return (1-beta)*matrix+beta*scale*jnp.eye(len(matrix)), scale
