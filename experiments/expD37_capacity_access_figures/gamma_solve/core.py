"""Frozen tanh kernels and spectral counterfactuals; all calculations FP64.

N denotes interior intervals (N+1 centers). H=ceil(sqrt(N)) centers per
side; center integration cells extend half an h beyond the extreme centers.
This is a diagnostic, not a proposed training algorithm.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from scipy import linalg as la
from scipy.optimize import linear_sum_assignment

TARGETS = {
    "mixed": ("Mixed sine", "sin(2πx) + ½ sin(6πx) + ¼ sin(10πx)"),
    "sine": ("Sine", "√2 sin(2πx)"),
    "runge": ("Runge", "1 / (1 + 25x²)"),
    "quadratic": ("Quadratic", "√5 x²"),
    "exp_sine": ("Exponential of sine", "exp(sin(3πx))"),
    "gaussian": ("Gaussian envelope", "exp(−4x²) [sin(2πx) + ½ sin(6πx) + ¼ sin(10πx)]"),
}
MAX_STEPS = 10**15


@dataclass(frozen=True)
class Geometry:
    N: int = 128
    m: int = 263
    center_jitter: float = 0.0
    data_jitter: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if not (8 <= self.N <= 512 and 17 <= self.m <= 1025):
            raise ValueError("Use 8 ≤ N ≤ 512 and 17 ≤ m ≤ 1025.")
        if not (0 <= self.center_jitter <= 1 and 0 <= self.data_jitter <= 1):
            raise ValueError("Jitter must lie in [0,1].")

    @property
    def h(self):
        return 2.0 / self.N

    @property
    def halo(self):
        return int(np.ceil(np.sqrt(self.N)))

    @property
    def bounds(self):
        end = 1 + (self.halo + .5) * self.h
        return -end, end

    def arrays(self):
        c = -1 + np.arange(-self.halo, self.N + self.halo + 1) * self.h
        x = np.linspace(-1, 1, self.m)
        # Fixed offsets: moving a jitter slider interpolates the SAME realization.
        dc = np.random.default_rng(self.seed).uniform(-.49, .49, c.size) * self.h
        dx = np.random.default_rng(self.seed + 1).uniform(-.49, .49, x.size) * (2/(self.m-1))
        # Keep domain boundaries, integration support, and halo extrema fixed.
        dc[[0, self.halo, self.halo + self.N, c.size-1]] = 0
        dx[[0, -1]] = 0
        return x + self.data_jitter * dx, c + self.center_jitter * dc

    def display_x(self, maximum=225):
        x, _ = self.arrays()
        a, b = self.bounds
        dx = 2/(self.m-1)
        left = -1 - dx*np.arange(1, int(np.ceil((-1-a)/dx))+1)
        right = 1 + dx*np.arange(1, int(np.ceil((b-1)/dx))+1)
        all_x = np.unique(np.r_[a, left[left > a], x, right[right < b], b])
        keep = np.linspace(0, len(all_x)-1, min(maximum, len(all_x))).round().astype(int)
        # Always retain the red-box boundaries and both display boundaries.
        return np.unique(np.r_[all_x[keep], -1., 1.])


def target(x, name):
    mixed = np.sin(2*np.pi*x) + .5*np.sin(6*np.pi*x) + .25*np.sin(10*np.pi*x)
    return {"mixed": lambda: mixed, "sine": lambda: np.sqrt(2)*np.sin(2*np.pi*x),
            "runge": lambda: 1/(1+25*x*x), "quadratic": lambda: np.sqrt(5)*x*x,
            "exp_sine": lambda: np.exp(np.sin(3*np.pi*x)),
            "gaussian": lambda: np.exp(-4*x*x)*mixed}[name]()


def features(x, c, gamma):
    return np.column_stack((np.ones(len(x)), np.tanh(gamma*(x[:, None]-c))))


def discrete_kernel(x, c, gamma):
    phi = features(x, c, gamma)
    return phi @ phi.T


def continuum_kernel(x, gamma, a, b, density):
    """1 + density*integral_a^b tanh(gamma*(x-c))*tanh(gamma*(x'-c)) dc.

    Analytic integral; a low-gamma quadrature branch avoids subtracting two
    nearly equal interval lengths. Near the diagonal use a Taylor divided
    difference (fourth-order error in gamma*(x-x')).
    """
    x = np.asarray(x, dtype=np.float64)
    length = b-a
    if gamma*length < 8:
        z, w = np.polynomial.legendre.leggauss(64)
        c = (a+b)/2 + length*z/2
        v = np.tanh(gamma*(x[:, None]-c)) * np.sqrt(w*length/2)
        return 1 + density*(v @ v.T)
    lc = lambda u: np.logaddexp(u, -u) - np.log(2.)
    g = (lc(gamma*(x-a))-lc(gamma*(x-b)))/gamma
    delta = x[None, :]-x[:, None]
    z = gamma*delta
    near = np.abs(z) < 1e-3
    quotient = np.zeros_like(z)
    np.divide(g[None, :]-g[:, None], np.tanh(z), out=quotient, where=~near)
    ii, jj = np.where(near)
    mid = (x[ii]+x[jj])/2
    ta, tb = np.tanh(gamma*(mid-a)), np.tanh(gamma*(mid-b))
    derivative = ta-tb
    third = -2*gamma**2*(ta*(1-ta*ta)-tb*(1-tb*tb))
    zz = z[ii, jj]
    factor = (1 + zz*zz/3 - zz**4/45)/gamma
    quotient[ii, jj] = factor*(derivative + delta[ii, jj]**2*third/24)
    result = 1 + density*(length-quotient)
    return (result+result.T)/2


def decompose(x, c, gamma):
    J = features(x, c, gamma)/np.sqrt(len(x))
    u, s, _ = la.svd(J, full_matrices=False, lapack_driver="gesvd", check_finite=False)
    eigenvalues = s*s
    # SVD rather than eigenvalues of the explicitly formed Gram matrix.
    return u, s, eigenvalues


def weights(u, y):
    y = y/la.norm(y)
    a = u.T@y
    residual = y-u@a
    p = np.r_[a*a, residual@residual]
    return p/p.sum()


def error_squared(rates, p, n):
    """Rates are eta*lambda, including a final zero for the omitted nullspace."""
    return float(np.dot(p, np.exp(2*float(n)*np.log1p(-np.clip(rates, 0, .5)))))


def hitting_time(rates, p, epsilon=.01, maximum=MAX_STEPS):
    threshold = epsilon**2
    if error_squared(rates, p, 0) <= threshold:
        return 0
    if error_squared(rates, p, maximum) > threshold:
        return None
    lo, hi = 0, 1
    while hi < maximum and error_squared(rates, p, hi) > threshold:
        lo, hi = hi, min(hi*2, maximum)
    while hi-lo > 1:
        mid = (lo+hi)//2
        if error_squared(rates, p, mid) <= threshold:
            hi = mid
        else:
            lo = mid
    return hi


@lru_cache(maxsize=8)
def kernel_view(geom, gamma):
    x, c = geom.arrays()
    xd = geom.display_x()
    a, b = geom.bounds
    direct = discrete_kernel(xd, c, gamma)/geom.m
    continuous = continuum_kernel(xd, gamma, a, b, 1/geom.h)/geom.m
    _, s, ev = decompose(x, c, gamma)
    ec = la.eigvalsh(continuum_kernel(x, gamma, a, b, 1/geom.h)/geom.m, check_finite=False)[::-1]
    ec_resolved = ec > 64*np.finfo(float).eps*geom.m*max(ec[0], 1)
    return dict(x=xd, direct=direct, continuous=continuous, eigenvalues=ev,
                continuous_eigenvalues=np.where(ec_resolved, ec, np.nan),
                singular_values=s, centers=c, samples=x,
                meta={**asdict(geom), "h": geom.h, "halo": geom.halo, "width": len(c),
                      "lambda": gamma*geom.h, "gamma": gamma, "bounds": [a,b],
                      "display_points": len(xd),
                      "relative_heatmap_difference": float(la.norm(direct-continuous)/la.norm(direct))})


@lru_cache(maxsize=3)
def sweep(geom, gamma_min=.25, gamma_max=128., count=41, gamma0=8.):
    if not (0 < gamma_min < gamma_max <= 512 and gamma_min <= gamma0 <= gamma_max):
        raise ValueError("Require 0 < gamma min ≤ gamma₀ ≤ gamma max ≤ 512.")
    x, c = geom.arrays()
    gammas = np.unique(np.r_[np.geomspace(gamma_min, gamma_max, count), gamma0])
    us, ss, evals = [], [], []
    for gamma in gammas:
        u, s, ev = decompose(x, c, gamma)
        us.append(u); ss.append(s); evals.append(ev)
    ss, evals = np.asarray(ss), np.asarray(evals)
    ranks = np.sum(ss > 1e-12*ss[:, :1], axis=1)
    rates = np.column_stack((.5*evals/evals[:, :1], np.zeros(len(gammas))))
    ps = {name: np.array([weights(u, target(x, name)) for u in us]) for name in TARGETS}
    ref = int(np.argmin(abs(gammas-gamma0)))
    r = us[0].shape[1]
    permutations = np.tile(np.arange(r), (len(gammas),1))
    adjacent_overlap = np.ones((len(gammas),r))
    for indices in (range(ref+1,len(gammas)), range(ref-1,-1,-1)):
        for i in indices:
            previous = i-1 if i > ref else i+1
            previous_u = us[previous][:, permutations[previous]]
            overlaps = abs(previous_u.T@us[i])**2
            _, permutation = linear_sum_assignment(-overlaps)
            permutations[i] = permutation
            adjacent_overlap[i] = overlaps[np.arange(r), permutation]
    # Check the local eigenvector derivative only on isolated, resolved modes.
    derivative_check = local_derivative_check(x, c, gamma0)
    result = dict(gammas=gammas, lambdas=gammas*geom.h, eigenvalues=evals, rates=rates,
                  p=ps, rank_resolved=ranks, permutations=permutations,
                  adjacent_overlap=adjacent_overlap, reference_index=ref,
                  derivative_check=derivative_check, geometry=asdict(geom))
    return result


def local_derivative_check(x, c, gamma):
    u, s, ev = decompose(x, c, gamma)
    J = features(x, c, gamma)/np.sqrt(len(x))
    t = np.tanh(gamma*(x[:, None]-c))
    jp = np.column_stack((np.zeros(len(x)), (x[:, None]-c)*(1-t*t)))/np.sqrt(len(x))
    kp = jp@J.T + J@jp.T
    delta = gamma*1e-4
    um, _, em = decompose(x, c, gamma-delta)
    up, _, ep = decompose(x, c, gamma+delta)
    records = []
    for i in range(min(12, len(s))):
        gaps = abs(ev[i]-np.delete(ev,i))
        gap = min(gaps.min(initial=np.inf), ev[i] if len(x)>len(s) else np.inf)
        if gap < 1e-8*ev[0] or s[i] < 1e-10*s[0]:
            continue
        jminus = int(np.argmax(abs(um.T@u[:,i])))
        jplus = int(np.argmax(abs(up.T@u[:,i])))
        vm = um[:,jminus]*np.sign(um[:,jminus]@u[:,i])
        vp = up[:,jplus]*np.sign(up[:,jplus]@u[:,i])
        fd = (vp-vm)/(2*delta)
        action = kp@u[:,i]
        coeff = u.T@action
        denominator = ev[i]-ev
        factor = np.divide(coeff, denominator, out=np.zeros_like(coeff), where=np.arange(len(ev))!=i)
        predicted = u@factor
        # Include the entire omitted zero eigenspace without choosing a basis.
        if len(x)>len(s):
            predicted += (action-u@coeff)/ev[i]
        records.append(dict(index=i+1, relative_gap=float(gap/ev[0]),
                            derivative_norm=float(la.norm(predicted)),
                            derivative_relative_error=float(la.norm(fd-predicted)/max(la.norm(predicted),1e-12)),
                            eigenvalue_derivative_relative_error=float(abs((ep[jplus]-em[jminus])/(2*delta)-u[:,i]@action)/max(abs(u[:,i]@action),1e-12))))
    return {"relative_gamma_increment": 1e-4, "isolated_modes": records,
            "meaning": "Local finite-difference check; not a global differentiability assertion at crossings or repeated eigenvalues."}


def matched_arrays(data, name, matching):
    p, rates = data['p'][name].copy(), data['rates'].copy()
    unresolved = np.arange(p.shape[1])[None,:] >= data['rank_resolved'][:,None]
    if matching == 'overlap':
        for i, perm in enumerate(data['permutations']):
            order = np.r_[perm, len(perm)]
            p[i], rates[i], unresolved[i] = p[i,order], rates[i,order], unresolved[i,order]
    elif matching != 'rank':
        raise ValueError('Unknown mode matching.')
    return p, rates, unresolved


def ambiguity_times(rates, p, ambiguous):
    if not np.any(ambiguous):
        t = hitting_time(rates,p)
        return t,t
    pp = p.copy()
    mass = pp[ambiguous].sum()
    pp[ambiguous] = 0
    indices = np.flatnonzero(ambiguous)
    fast, slow = indices[np.argmax(rates[indices])], indices[np.argmin(rates[indices])]
    pp[fast] += mass
    lower = hitting_time(rates,pp)
    pp[fast] -= mass; pp[slow] += mass
    return lower, hitting_time(rates,pp)


def ambiguity_bounds(rates, p, ambiguous, n):
    """Energy in unresolved eigenvectors may be allocated arbitrarily in that block.

    Bound its effect rather than presenting that arbitrary allocation as a
    measured eigenvector change. Structural nullspace is included conservatively.
    """
    factors = np.exp(2*float(n)*np.log1p(-np.clip(rates,0,.5)))
    fixed = float(p[~ambiguous]@factors[~ambiguous])
    mass = p[ambiguous].sum()
    if not np.any(ambiguous):
        return fixed, fixed
    return fixed+mass*factors[ambiguous].min(), fixed+mass*factors[ambiguous].max()


def comparison(data, name='mixed', n=1_000_000, matching='rank'):
    if name not in TARGETS or not 1 <= n <= MAX_STEPS:
        raise ValueError('Invalid target or step count.')
    p, rates, unresolved = matched_arrays(data,name,matching)
    ref = data['reference_index']
    p0, rates0 = p[ref], rates[ref]
    out = {key: [] for key in ['actual','fixed_p','fixed_rates','steps_actual','steps_fixed_p','steps_fixed_rates',
                               'fixed_p_lower','fixed_p_upper','fixed_rates_lower','fixed_rates_upper',
                               'steps_fixed_p_lower','steps_fixed_p_upper','steps_fixed_rates_lower','steps_fixed_rates_upper']}
    for i in range(len(p)):
        for label, pp, rr in [('actual',p[i],rates[i]),('fixed_p',p0,rates[i]),('fixed_rates',p[i],rates0)]:
            out[label].append(error_squared(rr,pp,n))
            out['steps_'+label].append(hitting_time(rr,pp))
        for label, pp, rr, mask in [('fixed_p',p0,rates[i],unresolved[ref]),('fixed_rates',p[i],rates0,unresolved[i])]:
            low, high = ambiguity_bounds(rr,pp,mask,n)
            out[label+'_lower'].append(low); out[label+'_upper'].append(high)
            low, high = ambiguity_times(rr,pp,mask)
            out['steps_'+label+'_lower'].append(low); out['steps_'+label+'_upper'].append(high)
    # Exclude meaningless tiny-weight directions from the displayed scatter.
    # Full arrays, including every tiny value, are kept in the saved data.
    out.update(gammas=data['gammas'], lambdas=data['lambdas'], p=p, rates=rates,
               unresolved=unresolved, reference_index=ref, n=int(n), target=name, matching=matching,
               null_mass=p[:,-1], rank_resolved=data['rank_resolved'],
               derivative_check=data['derivative_check'], adjacent_overlap=data['adjacent_overlap'],
               geometry=data['geometry'], max_steps=MAX_STEPS)
    return out
