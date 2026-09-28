"""Finite tanh differences, Toeplitz symbols, and the unchanged raw metric.

Exploratory exact-arithmetic identities with analytic truncation bounds.
Floating-point outputs are diagnostics, not interval certificates.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular, svd, toeplitz

from .gamma_filter import multiplier


def cumulative_transform(width):
    """w=C z: bias unchanged; hidden weights are backward differences."""
    c = np.eye(width+1)
    c[np.arange(2, width+1), np.arange(1, width)] = -1
    return c, c.T@c


def difference_profile(u, bandwidth):
    """tanh(lambda*u)-tanh(lambda*(u-1)), without saturated subtraction."""
    if bandwidth <= 0:
        raise ValueError('bandwidth must be positive')
    a, b = bandwidth*np.asarray(u), bandwidth*(np.asarray(u)-1)
    log_sinh = bandwidth+np.log(-np.expm1(-2*bandwidth))-np.log(2.)
    return np.exp(log_sinh-(np.logaddexp(a, -a)-np.log(2.))
                  -(np.logaddexp(b, -b)-np.log(2.)))


def finite_geometry(n, q, halo, gamma, length=2.):
    """Endpoint samples and all uniform centers, including both halo sides."""
    if n < 1 or q < 1 or halo < 0 or gamma <= 0 or length <= 0:
        raise ValueError('require positive n,q,gamma,length and nonnegative halo')
    h = length/n
    x = np.arange(q*n+1)*h/q
    centers = np.arange(-halo, n+halo+1)*h
    m, width = len(x), len(centers)
    j = np.column_stack((np.ones(m), np.tanh(gamma*(x[:, None]-centers))))/np.sqrt(m)
    c, metric = cumulative_transform(width)
    z = np.empty_like(j)
    z[:, 0], z[:, -1] = j[:, 0], j[:, -1]
    z[:, 1:-1] = difference_profile((x[:, None]-centers[:-1])/h, gamma*h)/np.sqrt(m)
    return dict(x=x, centers=centers, h=h, n=n, q=q, halo=halo,
                gamma=gamma, length=length, j=j, z=z, transform=c, metric=metric)


def whiten(z, transform):
    """Z C^{-1}; its sample kernel is Z (C* C)^{-1} Z*.

    This avoids forming the inverse metric or solving a squared-condition
    Gram eigenproblem. The bias and terminal feature remain present.
    """
    return solve_triangular(transform.T, np.asarray(z).T, lower=False).T


def spectrum(z, transform):
    """Stable full raw-readout spectrum through feature whitening and SVD."""
    u, singular, vh = svd(whiten(z, transform), full_matrices=False)
    return dict(eigenvalues=singular**2, left=u, right=vh.T)


def transformed_gd_step(z, transform, coefficients, target, step):
    """One GD update in cumulative coordinates with the exact raw metric."""
    gradient = z.T@(z@coefficients-target)
    dual = solve_triangular(transform.T, gradient, lower=False)
    return coefficients-step*solve_triangular(transform, dual, lower=True)


def kernel_error_bound(approximate_z, error_z, transform):
    """Measured error propagated AFTER whitening, not a roundoff certificate."""
    e = np.linalg.norm(whiten(error_z, transform), 2)
    norm = np.linalg.norm(whiten(approximate_z, transform), 2)
    return float((2*norm+e)*e)


def polyphase_symbol(theta, bandwidth, q, aliases=16, samples=1):
    """Difference symbol and energy, including q phases and sampled aliases.

    G_s(theta)=sum_l 2(1-exp(-i theta))/(i xi) M_lambda(xi)
    exp(i xi*s/q), xi=theta+2*pi*l. The omitted-alias bound holds for
    every theta in [-pi,pi], independently of this evaluation grid.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    if q < 1 or aliases < 0 or samples <= 0 or np.any(np.abs(theta) > np.pi):
        raise ValueError('invalid q, aliases, sample normalization, or angle')
    xi = theta[:, None]+2*np.pi*np.arange(-aliases, aliases+1)
    coefficient = np.divide(-2*np.expm1(-1j*theta[:, None]), 1j*xi,
                            out=np.full(xi.shape, 2.+0j), where=xi != 0)
    coefficient *= multiplier(bandwidth, xi)
    values = np.array([np.sum(coefficient*np.exp(1j*xi*s/q), axis=1) for s in range(q)])
    a = np.pi/(2*bandwidth)
    first = np.pi*(2*aliases+1)
    tail = (8*np.pi/bandwidth*np.exp(-a*first)
            /(-np.expm1(-2*a*first)*-np.expm1(-2*np.pi*a)))
    norms = np.linalg.norm(values, axis=0)
    energy_error = (2*norms*np.sqrt(q)*tail+q*tail**2)/samples
    return dict(polyphase=values, energy=norms**2/samples,
                amplitude_tail_bound=float(tail), energy_error_bound=energy_error)


def toeplitz_boundary(geometry, padding):
    """Infinite-lattice Toeplitz Gram minus omitted-row boundary Grams.

    Returns a truncated representation of the LOCALIZED columns only. The
    bias, terminal feature and their cross terms stay in geometry['z'].
    No invariant-subspace claim is made. padding counts fine-grid points.
    tail bounds control exact infinite sums, not floating-point arithmetic.
    """
    if padding < 0:
        raise ValueError('padding must be nonnegative')
    n, q, halo = [geometry[k] for k in ('n', 'q', 'halo')]
    width = len(geometry['centers'])
    dimension, m = width-1, len(geometry['x'])
    bandwidth = geometry['gamma']*geometry['h']
    t = np.arange(-padding, padding+q+1)
    base = difference_profile(t/q, bandwidth)
    correlation = np.array([np.dot(base, difference_profile(t/q+lag, bandwidth))/m
                            for lag in range(dimension)])
    bulk = toeplitz(correlation)
    first, last = halo*q, (halo+n)*q
    left_rows = np.arange(-padding, first)
    right_rows = np.arange(last+1, q*(width-1)+padding+1)

    def rows(indices):
        return difference_profile(indices[:, None]/q-np.arange(dimension), bandwidth)/np.sqrt(m)

    left, right = rows(left_rows), rows(right_rows)
    boundary = left.T@left+right.T@right
    # Correlation tails: |g(t)g(t+lag)|<=2g(t), uniformly in lag.
    correlation_tail = (8*np.exp(-2*bandwidth*(padding+1)/q)
                        /(m*-np.expm1(-2*bandwidth/q)))
    bulk_error = dimension*correlation_tail
    # Missing boundary rows have all columns exponentially small. Their PSD
    # Gram norm is bounded by its trace (squared Frobenius row tail).
    boundary_error = (8*dimension*np.exp(-4*bandwidth*(padding+1)/q)
                      /(m*-np.expm1(-4*bandwidth/q)))
    finite = geometry['z'][:, 1:-1].T@geometry['z'][:, 1:-1]
    return dict(toeplitz=bulk, boundary=boundary, left=left, right=right,
                finite=finite, approximation=bulk-boundary,
                toeplitz_tail_bound=float(bulk_error), boundary_tail_bound=float(boundary_error),
                error_bound=float(bulk_error+boundary_error), padding=int(padding))
