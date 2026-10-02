"""Pure FP64 vector fields and simultaneous physical-coordinate GD."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from . import references


def tanh_field(z, d, x, y, powers):
    a, b, c = z
    pre = x[:, None]*a + b
    h = jnp.tanh(pre)
    exp = jnp.exp(-2*jnp.abs(pre))
    derivative = 4*exp/(1+exp)**2
    e = h @ c + d - y
    m = len(x)
    moments = powers.T @ e/m
    weighted = e[:, None]*derivative
    grad = jnp.stack((c*(x @ weighted)/m, c*jnp.sum(weighted, axis=0)/m, h.T @ e/m))
    ec = moments[0] + moments[1]*x/jnp.mean(x*x)
    ga_coarse = c*(x @ (ec[:, None]*derivative))/m
    return .5*jnp.mean(e*e), moments, grad, moments[0], ga_coarse


def field(z, d, inputs, degree):
    if degree == 0:
        return tanh_field(z, d, inputs["x"], inputs["y"], inputs["powers"])
    return references.polynomial(z, d, degree, inputs["Q"], inputs["ym"], inputs["sy"])


def update(z, d, inputs, degree, eta, kappa):
    loss, moments, grad, gd, coarse = field(z, d, inputs, degree)
    multiplier = jnp.array([eta, eta, eta*kappa])[:, None]
    return z-multiplier*grad, d-eta*kappa*gd


def sample_polynomial(z, d, x, degree):
    pre = z[0]*x[:, None] + z[1]
    activation = sum(references.COEFFICIENTS[k]*pre**k for k in range(1, degree+1, 2))
    return activation @ z[2] + d
