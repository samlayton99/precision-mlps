"""Target-specific derivative partitions using the existing C17 QUILL recipe."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad
from experiments.expK02_inverse_sqrt_extension.construction import (
    Extension, Network, prepare_stencil, switch, quadrature,
)
from experiments.expK04_npu_elementary_suite.problem import target
from src.construction import construct_qi

TARGETS = ('root_plus_linear', 'exp_over_root')


def derivative(name, x):
    x = np.asarray(x)
    if name == 'root_plus_linear':
        r = np.sqrt(x)
        return -.5 * x**-1.5 * (1 + 2*r) / (1 + r)**2
    if name == 'exp_over_root':
        return -.5 * x**-1.5 * np.exp(-x) * (1 + 2*x)
    raise ValueError(name)


@dataclass(frozen=True)
class TargetExtension(Extension):
    name: str = 'root_plus_linear'

    def derivative(self, x):
        x = np.asarray(x, dtype=float)
        window = switch(x/(self.eps/2), self.order) - switch(x/self.outer_scale, self.order)
        return derivative(self.name, np.maximum(x, self.eps/2)) * window

    def value(self, x):
        def scalar(t):
            if self.eps <= t <= self.outer_scale:
                return float(target(self.name, t))
            anchor = self.eps if t < self.eps else self.outer_scale
            edge = np.clip(t, self.eps/2, 2*self.outer_scale)
            return float(target(self.name, anchor)) + quad(
                self.derivative, anchor, edge, epsabs=1e-13, epsrel=1e-13)[0]
        a = np.asarray(x, dtype=float)
        return np.array([scalar(float(t)) for t in a.ravel()]).reshape(a.shape)

    def block_derivative(self, z, scale):
        """Derivative of a normalized primitive in z, where x=2s(z+1).

        A block's physical contribution is s^-1/2 Q_s(x/(2s)-1).
        Factoring the leading inverse-root scale keeps derivative samples O(1).
        """
        u = 2 + 2*np.asarray(z)
        safe = np.maximum(u, 1.)
        x = scale*safe
        if self.name == 'root_plus_linear':
            root = np.sqrt(x)
            correction = (1 + 2*root)/(1 + root)**2
        elif self.name == 'exp_over_root':
            correction = np.exp(-x)*(1 + 2*x)
        else:
            raise ValueError(self.name)
        return -safe**-1.5 * correction * (switch(u, self.order)-switch(u/2, self.order))


def build(extension, n, cache_dir, halo=32, kc=160, lam=.25):
    prepare_stencil(n, cache_dir, kc=kc, lam=lam)
    weights, anchors, readouts = [], [], []
    for scale in extension.scales:
        # Q_s(-1)=0: its derivative vanishes until u=1. No primitive quadrature
        # or fitted function values enter these coefficients.
        q = construct_qi(lambda z: 0., lambda z: extension.block_derivative(z, scale),
                         N=n, halo=halo, Kc=kc, lambda_star=lam,
                         precision='mpmath', mp_dps=30, cache_dir=cache_dir)
        slope = (q.gamma*(extension.radius/extension.eps))*(extension.eps/(2*scale))
        weights.append(np.full(len(q.centers), slope))
        anchors.append(q.gamma*(extension.eps/(2*scale))-lam*np.arange(-halo, n+halo+1))
        readouts.append(q.a_coeffs/np.sqrt(scale))
    w, anchor, v = map(np.concatenate, (weights, anchors, readouts))
    b = w + anchor
    if not np.array_equal(b-w, anchor):
        raise ArithmeticError('Centered export lost the left endpoint arguments')
    bias = float(target(extension.name, extension.eps)) - math.fsum(v*np.tanh(anchor))
    return Network(w, b, v, bias, extension.eps, 'target_specific_multiscale_C17')


def metrics(net, name, order=256):
    x, log_weights = quadrature(net.eps, order)
    truth = target(name, x)
    weights = log_weights*x
    return float(np.sqrt(np.dot(weights, (net.predict(x)-truth)**2)/np.dot(weights, truth**2)))
