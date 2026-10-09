"""Explicit tanh/Radon construction. No regression or linear-system solve.

The endpoint correction is the scaled partial-fraction formula documented in
the note. It follows experiments/expF19_radon_direct_pde/quill_boundary.py,
with the geometry-only correction matrix cached across profiles.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.special import roots_jacobi, roots_legendre, hyp1f1, expit, gammaln


def sphere_rule(d, q):
    """Folded positive product rule: q**(d-1) directions, normalized weights.

    The first angle is restricted to [0, pi); all Jacobi coordinates retain
    their full symmetric grids. This retains one of every antipodal pair.
    The rule is exact for even spherical polynomials of degree <=2q-1.
    """
    if d == 1:
        return np.ones((1, 1)), np.ones(1)
    phi = np.arange(q)*np.pi/q
    theta = np.column_stack((np.cos(phi), np.sin(phi)))
    weights = np.full(q, 1/q)
    for dim in range(3, d+1):
        t, w = roots_jacobi(q, (dim-3)/2, (dim-3)/2)
        w = w/w.sum()
        theta = np.concatenate((np.sqrt(1-t[:, None, None]**2)*theta[None],
                                np.broadcast_to(t[:, None, None], (q, len(theta), 1))), axis=2).reshape(-1, dim)
        weights = (w[:, None]*weights).ravel()
    return theta, weights


def mixtures(d):
    """Five fixed families. Each tuple is (amplitude, inverse variance, center)."""
    u = np.ones(d)/np.sqrt(d)
    v = np.cos(np.arange(d)+.7)
    v /= np.linalg.norm(v)
    zero = np.zeros(d)
    return [
        [(1., .5, zero)],
        [(1., 3., zero)],
        [(1., 1.5, .35*u)],
        [(1., 2., -.35*u), (.7, 4., .35*v)],
        [(1., 4., -.45*u), (-2., 4., zero), (1., 4., .45*u)],
    ]


def target_values(x):
    return np.column_stack([
        sum(amp*np.exp(-a*np.sum((x-c)**2, axis=1)) for amp, a, c in mix)
        for mix in mixtures(x.shape[1])])


def profiles(theta, z):
    """Canonical Radon profiles, with probability surface measure.

    z's final axis broadcasts with the directions. For a common 1-D array
    use z[:,None]; for projected test points use x@theta.T directly.
    The d-dimensional Gaussian integral gives hyp1f1(d/2,1/2,-a*s*s).
    """
    d = theta.shape[1]
    out = []
    for mix in mixtures(d):
        value = sum(amp*hyp1f1(d/2, .5, -a*(z-theta@c)**2)
                    for amp, a, c in mix)
        out.append(value)
    return np.stack(out, axis=-1)


def test_points(d, count, seed):
    """Fixed independent points, including the origin, boundary, and axes."""
    rng = np.random.default_rng(seed+d)
    direction = rng.normal(size=(count, d))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    radii = np.linspace(.02, 1., count)
    rng.shuffle(radii)
    return np.vstack((np.zeros((1, d)), np.eye(d), -np.eye(d), direction*radii[:, None]))


@lru_cache(maxsize=128)
def geometry(n, r=.5, delta=1., quadrature_factor=1, lam=None):
    h = 2/n
    gamma = 2*np.pi/r if lam is None else float(lam)/h
    tau = np.pi/(2*gamma)
    lam = gamma*h
    halo = math.ceil(r*n/2)
    centers = -1+h*np.arange(-halo, n+halo+1)
    assert n >= max(16, 4*np.pi/r)
    assert 0 < lam <= 1
    assert lam*halo >= 2*np.log(2)
    assert tau <= delta/8*(1+1e-14)
    k = max(1, math.ceil(delta/(8*tau)-1e-14))
    sigma = 2*k*tau
    length = (halo+1)//2
    nq = quadrature_factor*max(96, 8*length*k+32)
    y, w = roots_legendre(nq)
    y, w = (y+1)/2, w/2
    ell = np.arange(1, length+1)
    zeta = np.exp(-2*lam)
    nodes = np.exp(-2*lam*(np.arange(1, length+1)-.5))
    products = np.r_[1., np.cumprod(-np.expm1(-2*lam*np.arange(1, length+1)))]
    correction_map = []
    for i0 in range(length):
        elementary = np.array([1.])
        for j in range(length):
            if j != i0:
                elementary = np.convolve(elementary, [1., nodes[j]])
        i = i0+1
        factor = (-1.)**i*zeta**((i*(i+1)-1)/2)/(products[i-1]*products[length-i])
        correction_map.append(factor*elementary)
    return dict(h=h, gamma=gamma, tau=tau, lam=lam, halo=halo, centers=centers,
                sigma=sigma, length=length, y=y, quad_weights=w, ell=ell,
                correction_map=np.array(correction_map), quadrature_order=nq)


def encode(profile, n, quadrature_factor=1, lam=None):
    g = geometry(n, quadrature_factor=quadrature_factor, lam=lam)
    h, tau, gamma = g['h'], g['tau'], g['gamma']
    def density(z):
        return (profile(z+1j*tau)-profile(z-1j*tau))/(2j*tau)
    base = np.real(h/2*density(g['centers'].astype(complex)))
    weights = base.copy()
    y, qw, ell, sigma = g['y'], g['quad_weights'], g['ell'], g['sigma']
    L = g['length']
    for endpoint, sign in [(g['centers'][0]-h/2, 1), (g['centers'][-1]+h/2, -1)]:
        fy = profile(endpoint+1j*tau*y)-profile(np.asarray(endpoint+0j))
        mu = np.real((np.exp(sign*1j*np.pi*ell[:, None]*y)*qw) @ fy)
        rho = density(endpoint+1j*sigma*y)
        nu = 2*sigma*np.imag((np.exp(sign*2j*ell[:, None]*gamma*sigma*y)*qw*
                              expit(-2*np.pi*sigma*y/h)) @ rho)
        nu *= ((-1.)**ell)[:, None]
        correction = g['correction_map'] @ (mu-nu)
        if sign == 1:
            weights[:L] += correction/2
        else:
            weights[-L:] -= correction[::-1]/2
    return weights, base


def error_bounds(d, q, n):
    """Unfitted, absolute, exact-arithmetic bounds, uniform on the unit ball.

    Angular: positivity/exactness plus Chebyshev interpolation remainder.
    Scalar: the finite-contour certificate, centered at the origin (factor 2).
    Entire Gaussian profiles have |g(s+iy)| <= hyp1f1(d/2,.5,a*y*y).
    """
    g = geometry(n)
    delta = 1.
    lam, R = g['lam'], g['halo']
    D = np.pi/(2*lam)+4*np.log(2)/np.pi
    scalar_factor = (32/delta*np.exp(-np.pi*delta*n/4)
                     +64*np.pi/(lam*delta*n)*np.exp(-np.pi**2/lam)
                     +32*D/(delta*n)*np.exp(-lam*R**2/4))
    angular, scalar = [], []
    for mix in mixtures(d):
        aerr, B = 0., 0.
        for amp, a, c in mix:
            if d > 1:
                # sup |g^(2q)| <= (4a)^q (d/2)_q. The product at 2q
                # Chebyshev roots is bounded by 2^(1-2q). Positivity of
                # quadrature adds factor 2. Charge both x and the anchor 0.
                log_scale = (np.log(4*abs(amp))+q*np.log(a)+gammaln(d/2+q)
                             -gammaln(d/2)-gammaln(2*q+1))
                radius = np.linalg.norm(c)
                aerr += np.exp(log_scale)*( (1+radius)**(2*q)+radius**(2*q) )
            B += abs(amp)*hyp1f1(d/2, .5, a*delta**2)
        angular.append(aerr)
        scalar.append(2*B*scalar_factor)
    return np.asarray(angular), np.asarray(scalar)


def construct_evaluate(d, q, n, x, quadrature_factor=1, return_model=False):
    theta, alpha = sphere_rule(d, q)
    geo = geometry(n, quadrature_factor=quadrature_factor)
    centers, gamma = geo['centers'], geo['gamma']
    accum = np.zeros((len(x), 5))
    baseline = np.zeros_like(accum)
    angular = np.zeros_like(accum)
    anchor = np.zeros(5)
    anchor_base = np.zeros(5)
    mass = np.zeros(5)
    blocks = []
    for start in range(0, len(theta), 32):
        th, al = theta[start:start+32], alpha[start:start+32]
        def profile(z):
            z = np.asarray(z)
            return profiles(th, z[..., None]).reshape(z.shape+(-1,))
        weights, base = encode(profile, n, quadrature_factor)
        weights = weights.reshape(len(centers), len(th), 5)*al[None, :, None]
        base = base.reshape(len(centers), len(th), 5)*al[None, :, None]
        z = x@th.T
        phi = np.tanh(gamma*(z[:, :, None]-centers))
        accum += np.einsum('pmc,cmf->pf', phi, weights, optimize=True)
        baseline += np.einsum('pmc,cmf->pf', phi, base, optimize=True)
        at0 = np.tanh(-gamma*centers)
        anchor += np.einsum('c,cmf->f', at0, weights)
        anchor_base += np.einsum('c,cmf->f', at0, base)
        mass += np.sum(abs(weights), axis=(0, 1))
        angular += np.einsum('pmf,m->pf', profiles(th, z)-profiles(th, np.zeros((1, len(th)))), al)
        if return_model:
            blocks.append((th, weights.copy()))
    f0 = target_values(np.zeros((1, d)))[0]
    bias = f0-anchor
    output = dict(pred=accum+bias, uncorrected=baseline+f0-anchor_base,
                  angular=angular+f0, mass=mass, bias=bias, directions=len(theta),
                  centers_per_direction=len(centers), width=len(theta)*len(centers),
                  quadrature_order=geo['quadrature_order'])
    if return_model:
        output['blocks'] = blocks
        output['centers'] = centers
        output['gamma'] = gamma
    return output


def round_significand(x, bits):
    mantissa, exponent = np.frexp(x)
    return np.ldexp(np.rint(mantissa*2.**bits)/2.**bits, exponent)


def rounded_readout_evaluation(model, x, bits, target_index=1):
    outputs = np.zeros((len(x), len(bits)))
    # Use the same function family as the width/dimension plot.
    for theta, weights in model['blocks']:
        w = weights[:, :, target_index]
        rounded = np.stack([round_significand(w, b) for b in bits], axis=-1)
        phi = np.tanh(model['gamma']*((x@theta.T)[:, :, None]-model['centers']))
        outputs += np.einsum('abc,cbd->ad', phi, rounded, optimize=True)
    return outputs+np.array([round_significand(model['bias'][target_index], b) for b in bits])
