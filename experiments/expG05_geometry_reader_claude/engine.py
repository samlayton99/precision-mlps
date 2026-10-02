"""Numerical engine for the geometry reader (Claude version).

Pure numpy, float64 throughout. The model is the single-hidden-layer tanh network

    f(x) = sum_k a_k tanh(g_k (x - c_k)) + b,

with centers c_k, slopes g_k (gamma) and readout (a, b). The local spacing of a
center is the central difference of its sorted neighbors, h_k, and its local
lambda is lambda_k = |g_k| h_k. The rule for tanh is lambda* = 0.25
(src/construction/qi_mpmath.py; expH02 for non-uniform meshes: g_k = lambda*/h_k).

Nothing here imports the repo so the app runs anywhere numpy exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = np.finfo(np.float64).eps

# ----------------------------------------------------------------------------
# targets
# ----------------------------------------------------------------------------
TARGETS = {
    "sine": "sin(2*pi*x)",
    "sqrt2_sine": "sqrt(2)*sin(2*pi*x)",
    "cosine": "cos(2*pi*x)",
    "sine_8pi": "sin(8*pi*x)",
    "sine_mixture": "sin(2*pi*x) + 0.5*sin(6*pi*x) + 0.25*sin(14*pi*x)",
    "runge": "1/(1+25*x**2)",
    "tanh_steep": "tanh(20*x)",
    "exp": "exp(x)",
    "poly5": "x**5 - 3*x**3 + x",
    "abs_cubed": "abs(x)**3",
    "spike": "exp(-((x-0.3)/0.03)**2)",
}

_SAFE = {
    name: getattr(np, name)
    for name in ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
                 "arcsinh", "arccosh", "arctanh", "exp", "log", "log10", "log2", "sqrt",
                 "abs", "sign", "floor", "ceil", "maximum", "minimum", "where", "heaviside"]
}
_SAFE.update({"pi": np.pi, "e": np.e, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
              "ln": np.log, "Abs": np.abs, "max": np.maximum, "min": np.minimum})


def make_target(expr: str):
    """Compile a target expression in x (numpy syntax, '^' allowed). Returns f(x)."""
    src = expr.strip().replace("^", "**")
    code = compile(src, "<target>", "eval")
    for name in code.co_names:
        if name != "x" and name not in _SAFE:
            raise ValueError(f"unknown name in target: {name}")

    def f(x):
        x = np.asarray(x, dtype=np.float64)
        out = eval(code, {"__builtins__": {}}, dict(_SAFE, x=x))  # noqa: S307 -- names whitelisted above
        return np.broadcast_to(np.asarray(out, dtype=np.float64), x.shape).copy()

    f(np.linspace(-1, 1, 5))  # fail early on a bad expression
    return f


def derivative(f, x, h=1e-4):
    """Fourth-order central difference; used only for the readout overlay."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def next_prime(n: int) -> int:
    n = max(2, int(n))
    while True:
        if all(n % p for p in range(2, int(math.isqrt(n)) + 1)):
            return n
        n += 1


def sample_x(n: int, domain, mode: str, seed: int = 0) -> np.ndarray:
    a, b = domain
    if mode == "equispaced":
        return np.linspace(a, b, n)
    if mode == "chebyshev":
        k = np.arange(n)
        return np.sort(a + (b - a) * (1 - np.cos(np.pi * k / max(n - 1, 1))) / 2)
    if mode == "uniform":
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(a, b, n))
        x[0], x[-1] = a, b
        return x
    raise ValueError(mode)


@dataclass
class Problem:
    """Target, data and evaluation grid. Everything the geometry is not."""

    target: str = "sin(2*pi*x)"
    domain: tuple = (-1.0, 1.0)
    n_train: int = 1024
    sampling: str = "equispaced"
    data_seed: int = 0
    noise: float = 0.0
    n_eval: int = 2048
    rcond: float = 1e-13
    lambda_star: float = 0.25

    def build(self):
        self.f = make_target(self.target)
        self.x = sample_x(int(self.n_train), self.domain, self.sampling, self.data_seed)
        self.y = self.f(self.x)
        if self.noise > 0:
            self.y = self.y + np.random.default_rng(self.data_seed + 1).normal(0, self.noise, self.x.shape)
        self.xe = np.linspace(self.domain[0], self.domain[1], next_prime(self.n_eval))
        self.fe = self.f(self.xe)
        return self

    def to_dict(self):
        return {k: getattr(self, k) for k in ["target", "domain", "n_train", "sampling", "data_seed",
                                              "noise", "n_eval", "rcond", "lambda_star"]}


# ----------------------------------------------------------------------------
# model, least squares, gradients
# ----------------------------------------------------------------------------
@dataclass
class Params:
    c: np.ndarray
    g: np.ndarray
    a: np.ndarray
    b: float = 0.0

    def copy(self):
        return Params(self.c.copy(), self.g.copy(), self.a.copy(), float(self.b))

    @property
    def W(self):
        return self.c.size


def features(x, c, g):
    return np.tanh(g[None, :] * (x[:, None] - c[None, :]))


def predict(x, P: Params, chunk=4096):
    out = np.empty(x.size)
    for s in range(0, x.size, chunk):
        out[s:s + chunk] = features(x[s:s + chunk], P.c, P.g) @ P.a + P.b
    return out


def solve_ls(x, y, c, g, rcond):
    """Least-squares readout on [Phi, 1]. rcond <= 0 means machine-precision cutoff."""
    if not (np.isfinite(c).all() and np.isfinite(g).all()):
        return np.full(c.size, np.nan), float("nan"), {"rank": 0, "cond": float("inf")}
    A = np.hstack([features(x, c, g), np.ones((x.size, 1))])
    try:
        sol, _, rank, s = np.linalg.lstsq(A, y, rcond=(rcond if rcond > 0 else None))
    except np.linalg.LinAlgError:
        return np.full(c.size, np.nan), float("nan"), {"rank": 0, "cond": float("inf")}
    cond = float(s[0] / s[-1]) if s.size and s[-1] > 0 else float("inf")
    return sol[:-1].copy(), float(sol[-1]), {"rank": int(rank), "cond": cond}


def loss_and_grads(x, y, P: Params):
    """MSE = mean (f(x_i) - y_i)^2 and its exact gradient."""
    Z = P.g[None, :] * (x[:, None] - P.c[None, :])
    T = np.tanh(Z)
    r = T @ P.a + P.b - y
    n = x.size
    loss = float(r @ r / n)
    M = (1.0 - T * T) * r[:, None]          # dT/dZ * r
    sM = M.sum(axis=0)
    ga = (2.0 / n) * (T.T @ r)
    gb = (2.0 / n) * r.sum()
    gg = (2.0 / n) * P.a * (x @ M - P.c * sM)
    gc = -(2.0 / n) * P.a * P.g * sM
    return loss, {"c": gc, "g": gg, "a": ga, "b": np.array([gb])}


def metrics(P: Params, prob: Problem):
    fe = predict(prob.xe, P)
    rt = predict(prob.x, P) - prob.y
    err = fe - prob.fe
    nf = float(np.linalg.norm(prob.fe)) or 1.0
    return {"train_mse": float(rt @ rt / rt.size),
            "rel_l2": float(np.linalg.norm(err) / nf),
            "linf": float(np.max(np.abs(err)))}, fe


# ----------------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------------
GROUPS = ("c", "g", "a", "b")


@dataclass
class AdamConfig:
    lr: float = 1e-3
    geom_lr_mult: float = 1.0      # multiplies lr for c and g
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-15
    schedule: str = "constant"    # constant | cosine
    gamma_param: str = "linear"   # linear | log  (log trains log|g|)
    batch: int = 0                # 0 = full batch
    batch_seed: int = 0
    train_c: bool = True
    train_g: bool = True
    train_a: bool = True
    train_b: bool = True


class Adam:
    """torch.optim.Adam semantics (bias-corrected, eps outside the sqrt)."""

    def __init__(self, W):
        self.t = 0
        self.m = {k: np.zeros(W if k != "b" else 1) for k in GROUPS}
        self.v = {k: np.zeros(W if k != "b" else 1) for k in GROUPS}

    def reset(self):
        self.t = 0
        for k in GROUPS:
            self.m[k][:] = 0
            self.v[k][:] = 0

    def step(self, P: Params, grads, cfg: AdamConfig, lr: float):
        self.t += 1
        b1, b2 = cfg.beta1, cfg.beta2
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        on = {"c": cfg.train_c, "g": cfg.train_g, "a": cfg.train_a, "b": cfg.train_b}
        for k in GROUPS:
            if not on[k]:
                continue
            gk = grads[k]
            if k == "g" and cfg.gamma_param == "log":
                gk = gk * P.g                    # d/d(log g) = g d/dg
            m, v = self.m[k], self.v[k]
            m *= b1
            m += (1 - b1) * gk
            v *= b2
            v += (1 - b2) * gk * gk
            step_lr = lr * (cfg.geom_lr_mult if k in ("c", "g") else 1.0)
            upd = step_lr * (m / bc1) / (np.sqrt(v / bc2) + cfg.eps)
            if k == "b":
                P.b = float(P.b - upd[0])
            elif k == "g" and cfg.gamma_param == "log":
                P.g *= np.exp(-upd)
            else:
                getattr(P, k)[:] -= upd


def lr_at(cfg: AdamConfig, t: int, total: int) -> float:
    if cfg.schedule == "cosine" and total > 0:
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
    return cfg.lr


# ----------------------------------------------------------------------------
# snapshot schedule
# ----------------------------------------------------------------------------
@dataclass
class SnapConfig:
    mode: str = "geometric"   # every | geometric
    every: int = 10
    first: int = 1
    ratio: float = 1.15
    max_gap: int = 500

    def gap(self, i: int) -> int:
        if self.mode == "every":
            return max(1, int(self.every))
        return int(max(1, min(round(self.first * self.ratio ** i), self.max_gap)))

    def steps(self, start: int, total: int, i0: int = 0):
        """Snapshot steps after `start` up to and including `total` (for previews/tests)."""
        out, s, i = [], start, i0
        while True:
            s += self.gap(i)
            i += 1
            if s >= total:
                out.append(total)
                return out
            out.append(s)


# ----------------------------------------------------------------------------
# geometry operations
# ----------------------------------------------------------------------------
def local_spacing(c):
    """h_k = (c_{k+1} - c_{k-1})/2 in sorted order, one-sided at the ends; original order."""
    W = c.size
    if W == 1:
        return np.ones(1)
    order = np.argsort(c, kind="stable")
    s = c[order]
    h = np.empty(W)
    h[1:-1] = (s[2:] - s[:-2]) / 2
    h[0] = s[1] - s[0]
    h[-1] = s[-1] - s[-2]
    span = max(s[-1] - s[0], 1e-300)
    h = np.maximum(h, 1e-9 * span)
    out = np.empty(W)
    out[order] = h
    return out


def local_lambda(c, g):
    return np.abs(g) * local_spacing(c)


def ideal_gamma(c, lambda_star):
    return lambda_star / local_spacing(c)


def canonical(P: Params):
    """Flip signs so every g >= 0: a tanh(-g(x-c)) = -a tanh(g(x-c))."""
    s = np.where(P.g < 0, -1.0, 1.0)
    return Params(P.c.copy(), P.g * s, P.a * s, P.b)


def resample(P: Params, W_new: int) -> Params:
    """Change the neuron count while keeping the arrangement.

    Sorted centers define a monotone map u in [0,1] -> x; the new centers sample it at
    W_new evenly spaced u. Local lambda is interpolated the same way and converted back
    with the new local spacing, then all gammas are rescaled so the mean lambda is
    exactly preserved. Readout a_k ~ f'(c_k) h_k / 2, so a is interpolated per unit
    spacing and rescaled by the new spacing.
    """
    W_new = int(max(1, W_new))
    order = np.argsort(P.c, kind="stable")
    c, g, a = P.c[order], P.g[order], P.a[order]
    W = c.size
    if W_new == W:
        return P.copy()
    h = local_spacing(c)
    lam = np.abs(g) * h
    sgn = np.where(g < 0, -1.0, 1.0)
    u = np.linspace(0, 1, W) if W > 1 else np.zeros(1)
    un = np.linspace(0, 1, W_new) if W_new > 1 else np.full(1, 0.5)
    if W == 1:
        cn = np.full(W_new, c[0]) + (np.linspace(-0.5, 0.5, W_new) if W_new > 1 else 0)
        lamn = np.full(W_new, lam[0])
        sn = np.full(W_new, sgn[0])
        dens = np.full(W_new, a[0] / h[0])
    else:
        cn = np.interp(un, u, c)
        lamn = np.interp(un, u, lam)
        sn = np.where(np.interp(un, u, sgn) < 0, -1.0, 1.0)
        dens = np.interp(un, u, a / h)
    hn = local_spacing(cn)
    gn = lamn / hn
    target_mean = lam.mean()
    cur_mean = (np.abs(gn) * hn).mean()
    if cur_mean > 0:
        gn *= target_mean / cur_mean
    return Params(cn, gn * sn, dens * hn, P.b)


def clean_target(W: int, domain, halo_mode: str, lambda_star: float, span=None):
    """The standard geometry with W neurons: uniform grid, h = (b-a)/N, halo R, g = lambda*/h.

    halo_mode: 'sqrt' (R = max(10, ceil(sqrt N)), Sam's default 2026-09-26), 'qi'
    (max(ceil(35/(2 lambda*)), floor(0.4 N)), src default_halo), 'none', or 'span'
    (uniform over the current [min c, max c]).
    """
    a, b = domain
    if halo_mode == "span" and span is not None:
        lo, hi = span
        c = np.linspace(lo, hi, W) if W > 1 else np.array([(lo + hi) / 2])
        h = (hi - lo) / max(W - 1, 1) if W > 1 else (b - a)
        return c, np.full(W, lambda_star / h)

    def halo(N):
        if halo_mode == "sqrt":
            return max(10, math.ceil(math.sqrt(N)))
        if halo_mode == "qi":
            return max(math.ceil(35 / (2 * lambda_star)), int(0.4 * N))
        return 0

    N = max(W - 1, 1)
    while N > 1 and N + 1 + 2 * halo(N) > W:
        N -= 1
    R = halo(N)
    extra = W - (N + 1 + 2 * R)            # leftover when the rule cannot fit exactly
    if extra < 0:                           # tiny W with a big halo: fall back to no halo
        R, N, extra = 0, max(W - 1, 1), 0
    h = (b - a) / N
    n = np.arange(-R, N + R + 1 + extra, dtype=np.float64) - extra / 2.0
    c = a + n * h
    return c[:W], np.full(W, lambda_star / h)


def clean(P: Params, p: float, what: str, domain, halo_mode: str, lambda_star: float) -> Params:
    """Interpolate a fraction p in [0,1] of the way to the clean geometry (sorted matching).

    Centers move linearly; |g| moves geometrically toward lambda*/h_k evaluated at the
    (possibly already moved) centers. Signs of g are kept.
    """
    Q = P.copy()
    order = np.argsort(Q.c, kind="stable")
    if what in ("centers", "both"):
        cc, _ = clean_target(Q.W, domain, halo_mode, lambda_star, span=(Q.c.min(), Q.c.max()))
        Q.c[order] = Q.c[order] + p * (cc - Q.c[order])
    if what in ("gamma", "both"):
        gi = ideal_gamma(Q.c, lambda_star)
        sgn = np.where(Q.g < 0, -1.0, 1.0)
        mag = np.maximum(np.abs(Q.g), 1e-300)
        Q.g = sgn * np.exp(np.log(mag) + p * (np.log(gi) - np.log(mag)))
    return Q


def jitter_centers(P: Params, p: float, rng) -> Params:
    """Gaussian center noise; p = 1 gives std one local spacing."""
    Q = P.copy()
    Q.c = Q.c + p * local_spacing(Q.c) * rng.standard_normal(Q.W)
    return Q


def jitter_gamma(P: Params, p: float, rng) -> Params:
    """Log-normal slope noise; p = 1 gives std 1 in log|g| (a factor of e)."""
    Q = P.copy()
    Q.g = Q.g * np.exp(p * rng.standard_normal(Q.W))
    return Q


def xavier_readout(W, seed):
    lim = math.sqrt(6.0 / (W + 1))
    return np.random.default_rng([seed, W, 7]).uniform(-lim, lim, W)


# ----------------------------------------------------------------------------
# geometry generators used by presets
# ----------------------------------------------------------------------------
def uniform_geometry(W, domain, halo_mode, lambda_star):
    c, g = clean_target(W, domain, halo_mode, lambda_star)
    return Params(c, g, np.zeros(W), 0.0)


def xavier_geometry(W, seed):
    """Glorot-uniform inner layer w, b ~ U(+-sqrt(6/(1+W))) read as c = -b/w, g = |w|.

    Same distribution as expD16/expD02 build_model(init='xavier'); the reading follows
    expD05 canonical_semantic. The sign of w is folded into the readout at canonicalization.
    """
    lim = math.sqrt(6.0 / (1 + W))
    rng = np.random.default_rng([seed, W])
    w = rng.uniform(-lim, lim, W)
    bb = rng.uniform(-lim, lim, W)
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return Params(-bb / w, np.abs(w), np.zeros(W), 0.0)


def xavier_rescaled_geometry(W, domain, halo_mode, lambda_star, seed):
    """expD05 scale_center_spread_xavier: uniform centers, Xavier slopes rescaled so the
    median slope sits at lambda*/h (the spread of Xavier slopes is kept)."""
    c, g0 = clean_target(W, domain, halo_mode, lambda_star)
    lim = math.sqrt(6.0 / (1 + W))
    w = np.abs(np.random.default_rng([seed, W, 5]).uniform(-lim, lim, W))
    g = w * (g0[0] / max(np.median(w), 1e-300))
    return Params(c, g, np.zeros(W), 0.0)
