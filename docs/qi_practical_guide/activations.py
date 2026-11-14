"""Activation values, derivatives, and spectral bandwidth priors for QI banks.

Conventions: erf means erf(z), Gaussian means exp(-z**2), GELU is exact
z*Phi(z), and SiLU/Swish has beta=1. The kernel is the lowest derivative
K=psi**(r) that is an ordinary integrable function with nonzero mass.

The selector controls a normalized kernel Fourier ratio, not total network
error. A custom Activation must supply a justified localized kernel and a
continuous monotone decreasing Fourier tail across the selected branch.
"""
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np
from scipy.special import erf, expit, ndtr

ArrayFunction = Callable[[np.ndarray], np.ndarray]


def _sech2(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-2*np.abs(z))
    return 4*e/(1+e)**2


def _gaussian_product(z, polynomial, denominator=1.0, exponent=.5):
    """Polynomial times a Gaussian, avoiding huge**2 times an underflowed zero."""
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z)
    keep = np.abs(z) < (40.0 if exponent == .5 else 30.0)
    zz = z[keep]
    out[keep] = polynomial(zz)*np.exp(-exponent*zz*zz)/denominator
    return out


def _gaussian(z):
    return _gaussian_product(z, lambda x: np.ones_like(x), exponent=1.)


def _log_sinh(a):
    return a-math.log(2)+math.log(-math.expm1(-2*a))


def _log_cosh(a):
    return a-math.log(2)+math.log1p(math.exp(-2*a))


def _log_ratio_linear(a):
    if a == 0:
        return 0.0
    if a < .01:
        return -a*a/6+a**4/180-a**6/2835
    return math.log(a)-_log_sinh(a)


def _silu_log_transform(xi):
    a = math.pi*abs(xi)
    if a == 0:
        return 0.0
    if a < .001:
        # log(a^2 cosh(a)/sinh(a)^2)
        return a*a/6-13*a**4/180+61*a**6/2835
    return 2*math.log(a)+_log_cosh(a)-2*_log_sinh(a)


@dataclass(frozen=True)
class Activation:
    name: str
    value: ArrayFunction
    first: ArrayFunction
    second: ArrayFunction
    order: int
    mass: float
    log_transform: Callable[[float], float]
    # psi(z)-psi(-z) = affine_gain*z, when this identity holds.
    affine_gain: float | None = None

    def __post_init__(self):
        if self.order not in (0, 1, 2):
            raise ValueError('this compact implementation supports kernel orders 0, 1, 2')
        if not math.isfinite(self.mass) or self.mass == 0:
            raise ValueError('the localized kernel must have finite nonzero mass')

    def kernel(self, z):
        return (self.value, self.first, self.second)[self.order](z)

    def scaled(self, scale):
        """psi_s(z)=psi(z/scale): H_s(xi)=H(scale*xi), lambda_s=scale*lambda."""
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('activation scale must be finite positive')
        return Activation(
            f'{self.name}(z/{scale:g})',
            lambda z: self.value(np.asarray(z)/scale),
            lambda z: self.first(np.asarray(z)/scale)/scale,
            lambda z: self.second(np.asarray(z)/scale)/scale**2,
            self.order, self.mass*scale**(1-self.order),
            lambda xi: self.log_transform(scale*xi),
            None if self.affine_gain is None else self.affine_gain/scale,
        )


SQRT2PI = math.sqrt(2*math.pi)
SQRTPI = math.sqrt(math.pi)
TANH = Activation('tanh', np.tanh, _sech2,
                  lambda z: -2*np.tanh(z)*_sech2(z), 1, 2.,
                  lambda xi: _log_ratio_linear(math.pi*abs(xi)/2))
SIGMOID = Activation('sigmoid', expit,
                     lambda z: .25*_sech2(np.asarray(z)/2),
                     lambda z: .25*_sech2(np.asarray(z)/2)*(1-2*expit(z)),
                     1, 1., lambda xi: _log_ratio_linear(math.pi*abs(xi)))
ERF = Activation('erf', erf,
                 lambda z: 2/SQRTPI*_gaussian(z),
                 lambda z: _gaussian_product(z,lambda x:-4*x/SQRTPI,exponent=1.),
                 1, 2., lambda xi: -float(xi)**2/4)
GAUSSIAN = Activation('gaussian', _gaussian,
                      lambda z: _gaussian_product(z,lambda x:-2*x,exponent=1.),
                      lambda z: _gaussian_product(z,lambda x:4*x*x-2,exponent=1.),
                      0, SQRTPI, lambda xi: -float(xi)**2/4)
GELU = Activation('gelu', lambda z: np.asarray(z)*ndtr(z),
                  lambda z: ndtr(z)+_gaussian_product(z,lambda x:x,SQRT2PI),
                  lambda z: _gaussian_product(z,lambda x:2-x*x,SQRT2PI),
                  2, 1., lambda xi: math.log1p(float(xi)**2)-float(xi)**2/2,
                  affine_gain=1.)
SILU = Activation('silu', lambda z: np.asarray(z)*expit(z),
                  lambda z: expit(z)+np.asarray(z)*.25*_sech2(np.asarray(z)/2),
                  lambda z: .25*_sech2(np.asarray(z)/2)*(2+np.asarray(z)*(1-2*expit(z))),
                  2, 1., _silu_log_transform, affine_gain=1.)
ACTIVATIONS = {a.name:a for a in (TANH,SIGMOID,ERF,GAUSSIAN,GELU,SILU)}
ALIASES = {'swish':'silu','logistic':'sigmoid','exact_gelu':'gelu'}


def get_activation(name):
    """Return the named specification, or pass through a custom Activation."""
    if isinstance(name,Activation):
        return name
    name = ALIASES.get(str(name).lower(),str(name).lower())
    try:
        return ACTIVATIONS[name]
    except KeyError as error:
        raise ValueError(f'unsupported activation {name!r}; choose {tuple(ACTIVATIONS)}') from error


def log_kernel_transform(name, xi):
    """Return scalar log |Khat(xi)/Khat(0)| (not log of unnormalized mass)."""
    return get_activation(name).log_transform(float(xi))


def select_lambda(name, tolerance=2.0**-52):
    """Return scalar lambda solving H(2*pi/lambda)=tolerance on its tail branch.

    Custom spectra with zeros, oscillatory tails or no decaying tail require
    separate branch/alias analysis and are not certified by this function.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('tolerance must be finite and in (0,1)')
    spec = get_activation(name)
    target = math.log(tolerance)
    lo, hi = 1e-6, 1.
    score = lambda lam: spec.log_transform(2*math.pi/lam)
    for _ in range(128):
        if score(lo) < target:
            break
        lo /= 2
    else:
        raise ValueError('no low-bandwidth bracket; check kernel tail/normalization')
    for _ in range(128):
        if score(hi) >= target:
            break
        hi *= 2
    else:
        raise ValueError('no high-bandwidth bracket; check kernel tail/normalization')
    for _ in range(100):
        mid = (lo+hi)/2
        if score(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


TANH_CALIBRATED_BUDGET = math.exp(TANH.log_transform(2*math.pi/.25))


def starting_lambda(name):
    """Practical tanh-calibrated starting value; tanh=.25 exactly by convention."""
    spec = get_activation(name)
    if spec is TANH:
        return .25
    return select_lambda(spec,TANH_CALIBRATED_BUDGET)


def integration_polynomial(spec, centers, slopes, coefficients, anchor, target_jets):
    """Constant/linear anchor correction for an unanchored activation sum.

    Return p where q(x)=sum_j a_j psi(gamma_j*(x-c_j))+p[0]+p[1]*(x-anchor)
    matches supplied f(anchor), f'(anchor) for order 2 (only f for order 1).
    For order 0, no integration polynomial is required. This does not compute
    the deconvolved coefficients; it corrects only integration constants.
    """
    spec = get_activation(spec)
    c=np.asarray(centers);g=np.asarray(slopes);a=np.asarray(coefficients)
    z=g*(anchor-c)
    jets=np.asarray(target_jets)
    if len(jets)<spec.order:
        raise ValueError('supply target derivatives through order r-1')
    if spec.order==0:
        return np.empty(0)
    p=[float(jets[0]-np.sum(a*spec.value(z)))]
    if spec.order==2:
        p.append(float(jets[1]-np.sum(a*g*spec.first(z))))
    return np.array(p)


def affine_pair(spec, slope, pair_gamma=1.):
    """Two ordinary neurons realizing slope*x exactly in real arithmetic.

    Returns hidden weights, hidden biases, and readout coefficients. Available
    for exact GELU/SiLU and their positive argument rescalings.
    """
    spec=get_activation(spec)
    if spec.affine_gain is None:
        raise ValueError('no exact affine-pair identity supplied for this activation')
    if not math.isfinite(pair_gamma) or pair_gamma<=0:
        raise ValueError('pair_gamma must be finite positive')
    a=float(slope)/(spec.affine_gain*pair_gamma)
    return np.array([pair_gamma,-pair_gamma]),np.zeros(2),np.array([a,-a])
