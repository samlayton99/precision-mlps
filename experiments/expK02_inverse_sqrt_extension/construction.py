"""Smooth exact extension and multiscale cardinal-QI for x**(-1/2)."""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, roots_legendre
from src.construction import construct_qi
from src.construction.qi_mpmath import (
    DEFAULT_CACHE_DIR, _cache_key, _load_cached_c_mpmath, _save_cached_c_mpmath,
)

EPS = 1e-4


def switch(u, order):
    """C^order step: zero below 1, one above 2; stable beta polynomial inside."""
    return betainc(order + 1, order + 1, np.clip(np.asarray(u) - 1., 0., 1.))


@dataclass(frozen=True)
class Extension:
    eps: float = EPS
    order: int = 16

    @property
    def radius(self): return (1. - self.eps) / 2.
    @property
    def center(self): return (1. + self.eps) / 2.
    @property
    def scales(self):
        return self.eps / 2. * 2. ** np.arange(math.ceil(math.log2(2. / self.eps)))
    @property
    def outer_scale(self): return 2. * self.scales[-1]

    def centered(self, x):
        # Equivalent to (x-center)/radius; exact -1 at the physical left endpoint.
        return (np.asarray(x) - self.eps) / self.radius - 1.

    def physical(self, z):
        return self.eps + self.radius * (np.asarray(z) + 1.)

    def derivative(self, x):
        x = np.asarray(x, dtype=float)
        safe = np.maximum(x, self.eps / 2.)
        window = switch(x / (self.eps / 2.), self.order) - switch(x / self.outer_scale, self.order)
        return -.5 * safe ** -1.5 * window

    def value(self, x):
        """Bounded C^(order+1) primitive, exactly x^-1/2 on [eps,1]."""
        def scalar(t):
            s0, upper = self.eps / 2., self.outer_scale
            if t < self.eps:
                integ = quad(lambda u: -.5 * u ** -1.5 * float(switch(u, self.order)),
                             2., max(1., t / s0), epsabs=1e-14, epsrel=1e-13)[0]
                return self.eps ** -.5 + s0 ** -.5 * integ
            if t <= upper:
                return t ** -.5
            integ = quad(lambda u: -.5 * u ** -1.5 * (1. - float(switch(u, self.order))),
                         1., min(2., t / upper), epsabs=1e-14, epsrel=1e-13)[0]
            return upper ** -.5 * (1. + integ)
        a = np.asarray(x, dtype=float)
        if a.ndim == 0: return scalar(float(a))
        return np.array([scalar(float(t)) for t in a.ravel()]).reshape(a.shape)

    def prototype_derivative(self, z):
        # u in [0,4]; support [1,4]. The dyadic domain makes the exported anchor exact.
        u = 2. + 2. * np.asarray(z)
        safe = np.maximum(u, 1.)
        return -safe ** -1.5 * (switch(u, self.order) - switch(u / 2., self.order))

    def prototype_value(self, z):
        u = min(4., max(1., 2. + 2. * float(z)))
        return quad(lambda v: -.5 * v ** -1.5 * (float(switch(v, self.order)) - float(switch(v / 2., self.order))),
                    1., u, points=[2.] if u > 2. else None, epsabs=1e-14, epsrel=1e-13)[0]


def prepare_stencil(n, cache_dir, *, kc=160, lam=.25, dps=30):
    """Reuse the same dimensionless cardinal solve: c_N = (128/N) c_128."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(lambda_star=lam, Kc=kc, precision='mpmath', mp_dps=dps)
    source = DEFAULT_CACHE_DIR / _cache_key(N=128, **kwargs)
    if not source.exists():
        construct_qi(lambda x: 0., lambda x: 0., N=128, precision='mpmath',
                     lambda_star=lam, Kc=kc, mp_dps=dps)
    values, info = _load_cached_c_mpmath(source, dps)
    dest = cache_dir / _cache_key(N=n, **kwargs)
    if not dest.exists():
        import mpmath as mp
        with mp.workdps(dps):
            scale = mp.mpf(128) / n
            _save_cached_c_mpmath(dest, [c * scale for c in values],
                                 info['toeplitz_resid_inf'] * float(scale), dps)


@dataclass
class Network:
    weight: np.ndarray
    hidden_bias: np.ndarray
    readout: np.ndarray
    output_bias: float
    eps: float
    method: str

    @property
    def width(self): return len(self.readout)

    def predict(self, x, batch=256):
        z = (np.asarray(x).ravel() - self.eps) / ((1. - self.eps) / 2.) - 1.
        y = np.concatenate([np.tanh(z[i:i+batch, None] * self.weight + self.hidden_bias) @ self.readout
                            + self.output_bias for i in range(0, len(z), batch)])
        return y.reshape(np.asarray(x).shape)

    def save(self, path):
        np.savez(path, weight=self.weight, hidden_bias=self.hidden_bias, readout=self.readout,
                 output_bias=self.output_bias, eps=self.eps, method=self.method,
                 input_center=(1. + self.eps)/2., input_scale=(1. - self.eps)/2.)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls(d['weight'], d['hidden_bias'], d['readout'], float(d['output_bias']),
                       float(d['eps']), str(d['method']))


def build(extension, n, cache_dir, *, method='multiscale', halo=160, kc=160, lam=.25):
    prepare_stencil(n, cache_dir, kc=kc, lam=lam)
    if method == 'multiscale':
        q = construct_qi(extension.prototype_value, extension.prototype_derivative,
                         N=n, halo=halo, Kc=kc, lambda_star=lam, precision='mpmath', cache_dir=cache_dir)
        scales = extension.scales
        # Factor out the dyadic scale ratio before multiplying; this avoids an
        # unnecessary rounding error when a wider sweep uses gamma=12,20,... .
        base_slope = q.gamma * (extension.radius / extension.eps)
        w = np.concatenate([np.full(len(q.centers), base_slope * (extension.eps/(2*s))) for s in scales])
        indices = np.arange(-q.halo, q.N + q.halo + 1)
        # gamma*(1+center_n)=lambda*n. Use the integer index so non-power-of-two
        # widths retain the exact endpoint argument instead of rounding centers twice.
        anchor = np.concatenate([q.gamma * (extension.eps/(2*s)) - lam * indices for s in scales])
        b = w + anchor
        # With dyadic gamma and scale ratios, these differences are exact in fp64.
        if not np.array_equal(b - w, anchor):
            raise ArithmeticError('Centered export lost the left-endpoint argument')
        v = np.concatenate([q.a_coeffs / np.sqrt(s) for s in scales])
    elif method == 'uniform':
        q = construct_qi(lambda z: extension.value(extension.physical(z)),
                         lambda z: extension.radius * extension.derivative(extension.physical(z)),
                         N=n, halo=halo, Kc=kc, lambda_star=lam, precision='mpmath', cache_dir=cache_dir)
        w = np.full(len(q.centers), q.gamma)
        anchor = q.gamma * (-1. - q.centers)
        b, v = w + anchor, q.a_coeffs
    else:
        raise ValueError(method)
    bias = extension.eps ** -.5 - math.fsum(v * np.tanh(anchor))
    return Network(w, b, v, bias, extension.eps, method)


def quadrature(eps, order):
    """Physical dx quadrature, using log-x panels only for numerical integration."""
    edges = np.r_[eps * 2. ** np.arange(math.ceil(math.log2(1./eps))), 1.]
    nodes, weights = roots_legendre(order)
    logx, qw = [], []
    for a, b in zip(np.log(edges[:-1]), np.log(edges[1:])):
        logx.append((a+b)/2 + (b-a)/2 * nodes)
        qw.append((b-a)/2 * weights)
    return np.exp(np.concatenate(logx)), np.concatenate(qw)


def metrics(net, eps, order=256):
    x, w = quadrature(eps, order)
    true = x ** -.5
    relative = (net.predict(x) - true) / true
    # f(x)^2 dx = d(log x), so this is precisely the physical-domain relative L2.
    return float(np.sqrt(np.dot(w, relative ** 2) / np.log(1./eps)))
