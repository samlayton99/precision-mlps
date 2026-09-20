"""Frozen dictionaries, polynomial-tail measurements, and discrete GD bounds.

Dense factorizations are detached diagnostics. Computed FP64 quantities are
estimates, not interval-certified bounds on the nominal real tanh dictionary.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import get_lapack_funcs, qr, svd
import yaml

from experiments.expD06_fixed_center_scales.core import geometry

HERE = Path(__file__).resolve().parent


def config(path=HERE / 'config.yaml'):
    return yaml.safe_load(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def target(x, name):
    if name == 'sine_mix_2_6_10':
        return np.sin(2*np.pi*x) + .5*np.sin(6*np.pi*x) + .25*np.sin(10*np.pi*x)
    if name == 'quadratic':
        return np.sqrt(5.)*x*x
    raise ValueError(name)


def grid(n):
    return -1 + 2*(np.arange(n) + .5)/n


def design(x, centers, gamma):
    return np.column_stack((np.ones(len(x)), np.tanh(gamma*(x[:, None]-centers)))) / np.sqrt(len(x))


def scales(g, name):
    if name == 'raw':
        return np.ones(g.width + 1)
    if name == 'collective':
        return np.sqrt(g.alpha)
    raise ValueError(name)


def polynomial_transform(x, degree):
    """Raw, unpivoted Householder QR; no loss of complementary sample rows."""
    v = np.polynomial.chebyshev.chebvander(x, degree) / np.sqrt(len(x))
    (reflectors, tau), _ = qr(v, mode='raw', pivoting=False)
    return reflectors, tau


def transform(raw_qr, values, transpose=True):
    reflectors, tau = raw_qr
    c = np.array(values, dtype=np.float64, order='F', copy=True)
    ormqr = get_lapack_funcs('ormqr', (reflectors, c))
    trans = 'T' if transpose else 'N'
    _, workspace, info = ormqr('L', trans, reflectors, tau, c, lwork=-1)
    if info:
        raise RuntimeError(f'ORMQR workspace query failed: {info}')
    result, _, info = ormqr('L', trans, reflectors, tau, c, lwork=int(workspace[0]))
    if info:
        raise RuntimeError(f'ORMQR failed: {info}')
    return result


def discrete_polynomials(x, degree):
    """Independent uniform-grid Gram polynomials, normalized in sample mean.

    The monic recurrence coefficient on the endpoint grid is
    n^2 (m^2-n^2) / ((4n^2-1)(m-1)^2). This independent basis checks QR tails.
    """
    m = len(x)
    p = np.empty((m, degree+1))
    p[:, 0] = 1.
    previous_a = 0.
    for n in range(1, degree+1):
        a = np.sqrt(n*n*(m*m-n*n)/((4*n*n-1)*(m-1)**2))
        p[:, n] = (x*p[:, n-1] - (previous_a*p[:, n-2] if n > 1 else 0)) / a
        previous_a = a
    return p / np.sqrt(m)


def log_feature_envelope(gamma, degrees):
    k = np.asarray(degrees)
    if gamma == 0:
        return np.full(k.shape, -np.inf)
    beta = np.arcsinh(np.pi/(2*abs(gamma)))
    bracket = np.logaddexp(-.5*np.log(gamma*gamma+np.pi**2/4), -np.log(np.pi*(k+1)))
    log_u = np.log(4.) - k*beta - np.log(np.expm1(beta)) + bracket
    return np.minimum(np.log(np.tanh(abs(gamma))), log_u)


def access(y_hat, j_hat, k_max):
    """Backward sums accumulate small deep tails before large leading terms."""
    tail_sq = np.cumsum((y_hat*y_hat)[::-1], axis=0)[::-1]
    gradients = np.cumsum((j_hat[:, :, None]*y_hat[:, None, :])[::-1], axis=0)[::-1]
    frob = np.cumsum(np.sum(j_hat*j_hat, axis=1)[::-1])[::-1]
    ks = np.arange(k_max+1)
    tail = tail_sq[ks+1]
    e = np.sqrt(tail/tail_sq[0])
    mu = np.divide(np.sum(gradients[ks+1]**2, axis=1), tail,
                   out=np.zeros_like(tail), where=tail > 0)
    return e, mu, frob[ks+1]


def bound(e, log_denominator, epsilon, curvature):
    """C2 for eta=.5/L. Return the strongest cutoff and a log10 step bound."""
    e, log_denominator = np.broadcast_arrays(e, log_denominator)
    eligible = e > epsilon
    if not np.any(eligible):
        return dict(k=None, log10_bound=None, bound=0., status='zero_witness')
    logs = np.full(e.shape, -np.inf)
    c = np.log(np.log(1/epsilon)) - 2*np.log1p(-epsilon)
    logs[eligible] = (c + np.log(curvature) - np.log(np.log(2.))
                      + 2*np.log(e[eligible]-epsilon) - log_denominator[eligible])
    k = int(np.argmax(logs))
    if np.isposinf(logs[k]):
        return dict(k=k, log10_bound=None, bound=None, status='zero_access')
    if not np.isfinite(logs[k]):
        return dict(k=None, log10_bound=None, bound=None, status='unresolved')
    value = float(logs[k]/np.log(10.))
    # A decimal log preserves bounds beyond float/integer reporting ranges.
    steps = float(np.ceil(np.exp(logs[k]))) if logs[k] < np.log(2**53) else None
    return dict(k=k, log10_bound=value, bound=steps, status='fp64_estimate')


def spectral_error(step, singular, loadings, floor_sq, norm_y, eta):
    decay = np.ones_like(singular) if step == 0 else np.exp(2*step*np.log1p(-eta*singular**2))
    return np.sqrt(floor_sq + np.sum(loadings*loadings*decay[:, None], axis=0)) / norm_y


def spectral_hit(singular, loading, floor_sq, norm_y, eta, epsilon, cap=10**32):
    def error(n):
        return float(spectral_error(n, singular, loading[:, None], np.array([floor_sq]),
                                    np.array([norm_y]), eta)[0])
    if error(0) <= epsilon:
        return dict(steps=0., log10_steps=None, status='already_reached')
    if np.sqrt(floor_sq)/norm_y >= epsilon:
        return dict(steps=None, log10_steps=None, status='retained_model_unattainable')
    hi = 1
    while hi < cap and error(hi) > epsilon:
        hi *= 2
    hi = min(hi, cap)
    if error(hi) > epsilon:
        return dict(steps=None, log10_steps=None, status='beyond_prediction_cap')
    lo = 0
    while hi-lo > 1:
        mid = (hi+lo)//2
        if error(mid) <= epsilon:
            hi = mid
        else:
            lo = mid
    return dict(steps=float(hi) if hi < 2**53 else None,
                log10_steps=math.log10(hi), status='spectral_extrapolation')


def spectrum(j, y, j_eval, y_eval, cutoffs, tolerances):
    u, s, vh = svd(j, full_matrices=False, lapack_driver='gesdd')
    norm_y = np.linalg.norm(y, axis=0)
    curvature = float(s[0]**2)
    rows, arrays = [], {}
    for cutoff in cutoffs:
        keep = s > cutoff*s[0]
        loading = u[:, keep].T @ y
        residual = y-u[:, keep] @ loading
        floor_sq = np.sum(residual*residual, axis=0)
        theta = vh[keep].T @ (loading/s[keep, None])
        eval_error = np.linalg.norm(j_eval @ theta-y_eval, axis=0)/np.linalg.norm(y_eval, axis=0)
        for t in range(y.shape[1]):
            rows.append(dict(cutoff=float(cutoff), target_index=t, retained_rank=int(keep.sum()),
                train_refit=float(np.sqrt(floor_sq[t])/norm_y[t]), eval_refit=float(eval_error[t]),
                coefficient_l2=float(np.linalg.norm(theta[:, t])),
                predictions=[dict(epsilon=float(eps), **spectral_hit(s[keep], loading[:, t],
                    floor_sq[t], norm_y[t], .5/curvature, eps)) for eps in tolerances]))
        if cutoff == min(cutoffs):
            arrays = dict(singular=s, keep=keep, loadings=loading, floor_sq=floor_sq,
                          norm_y=norm_y, refit_theta=theta)
    return curvature, rows, arrays
