"""Exact polynomial moment gradients and discrete affine Gram closure."""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np

COEFFICIENTS = (0., 1., 0., -1/3, 0., 2/15, 0., -17/315)


def coefficients(z, d, degree):
    """Return the eleven padded coefficients of the polynomial prediction."""
    a, b, c = z
    result = []
    for k in range(degree+1):
        inner = sum(COEFFICIENTS[l]*math.comb(l, k)*b**(l-k)
                    for l in range(k, degree+1) if COEFFICIENTS[l])
        result.append(jnp.sum(c * a**k * inner) + (d if k == 0 else 0.))
    return jnp.pad(jnp.stack(result), (0, 10-degree))


def polynomial(z, d, degree, Q, ym, sy):
    """Own-state reference dynamics; no actual-network quantity is an input."""
    F, pullback = jax.vjp(lambda z, d: coefficients(z, d, degree), z, d)
    moments = Q @ F - ym
    grad, gd = pullback(moments)
    loss = .5*(F @ Q @ F - 2*F @ ym + sy)
    sigma = jnp.sqrt(Q[1, 1])
    coarse_moments = moments[0]*Q[:, 0] + moments[1]/sigma**2*Q[:, 1]
    coarse_grad, _ = pullback(coarse_moments)
    return loss, moments, grad, gd, coarse_grad[0]


def affine_step(G, d, U, beta, sigma, eta, kappa):
    """Seven-scalar recurrence, retaining the exact finite-step quadratic term."""
    m0, m1 = d + G[1, 2] - beta[0], sigma*G[0, 2] - beta[1]
    B = np.array([[0, 0, -sigma*m1], [0, 0, -m0],
                  [-kappa*sigma*m1, -kappa*m0, 0.]])
    T = np.eye(3) + eta*B
    return T @ G @ T.T, d-eta*kappa*m0, T @ U


def slope_interval(z, moments, R, order):
    """State-conditioned analytic remainder bound; not a trajectory radius."""
    from scipy.optimize import brentq
    from numpy.polynomial import Polynomial
    a, b, c = np.asarray(z)
    t = np.tanh(b)
    derivative = Polynomial([1., 0., -1.])
    approximation = np.zeros_like(a)
    for k in range(order):
        approximation += derivative(t)/math.factorial(k)*a**k*moments[k+1]/R
        derivative = derivative.deriv()*Polynomial([1., 0., -1.])
    approximation *= c
    rho = brentq(lambda r: 2*r*np.tan(r)-order, 1e-9, np.pi/2-1e-9)
    return approximation, rho
