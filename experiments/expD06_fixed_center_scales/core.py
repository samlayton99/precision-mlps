"""Small FP64 JAX model; physical initialization is shared across coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

# This experiment explicitly studies binary64 precision and scale gradients.
jax.config.update("jax_enable_x64", True)

LAMBDA_REF = 0.25


@dataclass(frozen=True)
class Geometry:
    n: int
    h: float
    radius: int
    centers: np.ndarray
    alpha: np.ndarray  # bias first, then all neurons
    core: np.ndarray
    corrected_halo: np.ndarray

    @property
    def width(self):
        return self.centers.size

    @property
    def d(self):
        return np.sqrt(self.alpha)

    @property
    def ordinary_alpha(self):
        return float(self.alpha[1:][self.core][0])

    @property
    def masks(self):
        return {"core": self.core, "halo": ~self.core & ~self.corrected_halo,
                "corrected_halo": self.corrected_halo}


def geometry(n: int) -> Geometry:
    h = 2.0 / n
    radius = math.ceil(math.sqrt(n))
    slots = np.arange(-radius, n + radius + 1)
    delta = 0.25
    pole_distance = math.pi * h / (2 * LAMBDA_REF)
    if pole_distance >= delta:
        raise ValueError("Reference envelopes require pole distance < delta.")
    alpha = np.full(slots.size, h / (2 * (delta - pole_distance)))
    corrected = np.zeros(slots.size, dtype=bool)
    m = (radius + 1) // 2
    zeta = math.exp(-2 * LAMBDA_REF)
    p = np.r_[1.0, np.cumprod(1 - zeta ** np.arange(1, m + 1))]
    d_lambda = math.pi / (2 * LAMBDA_REF) + 4 * math.log(2) / math.pi
    for i in range(1, m + 1):
        li = zeta ** ((i * (i + 1) - 1) / 2) / (p[i - 1] * p[m - i])
        li *= math.prod(1 + zeta ** (j - 0.5) for j in range(1, m + 1) if j != i)
        for slot in (i - 1, slots.size - i):
            alpha[slot] += h * d_lambda * li / (2 * delta)
            corrected[slot] = True
    return Geometry(n, h, radius, -1 + h * slots,
                    np.r_[1 + alpha.sum(), alpha], (slots >= 0) & (slots <= n), corrected)


def target(x, name: str, xp=jnp):
    if name == "sine":
        return math.sqrt(2) * xp.sin(2 * math.pi * x)
    if name == "quadratic":
        return math.sqrt(5) * x**2
    if name == "mixed":
        return (xp.sin(2 * math.pi * x) + 0.1 * xp.sin(20 * math.pi * x)) / math.sqrt(0.505)
    raise ValueError(f"Unknown target {name!r}")


def initial_physical(g: Geometry, seed: int, family: str, halo_init="full"):
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((2, g.width))
    while np.any(draws == 0):
        zero = draws == 0
        draws[zero] = rng.standard_normal(zero.sum())
    gamma_draw, xi = draws
    gamma = np.abs(gamma_draw) * (5 / 3) * math.sqrt(2 / (g.width + 1))
    if family == "xavier":
        weights = math.sqrt(2 / (g.width + 1)) * xi
    elif family in {"xavier_a_uniform", "xavier_a_reference"}:
        scale = math.sqrt(g.h) if family == "xavier_a_uniform" else g.d[1:]
        weights = scale * math.sqrt(2 / (g.width + 1)) * xi
    elif family == "envelope":
        allowances = g.alpha[1:].copy()
        if halo_init == "ordinary":
            allowances[g.corrected_halo] = g.ordinary_alpha
        weights = allowances * np.sign(xi)
    else:
        raise ValueError(f"Unknown initialization {family!r}")
    return np.r_[0.0, weights * np.sign(gamma_draw)], gamma


def coordinate_scales(g: Geometry, arm: str, halo_metric="full", unscaled_bias=False):
    if arm == "raw":
        return np.ones(g.width + 1), 1.0
    if arm == "uniform":
        return np.r_[1., np.full(g.width, math.sqrt(g.h))], 1 / g.h
    if arm != "both":
        raise ValueError(f"Unknown arm {arm!r}")
    d = g.d.copy()
    if unscaled_bias:
        d[0] = 1.
    if halo_metric == "ordinary":
        d[1:][g.corrected_halo] = math.sqrt(g.ordinary_alpha)
    return d, 1 / g.h


def to_params(c, gamma, c_scale, gamma_scale):
    return {"readout": jnp.asarray(c / c_scale), "slope": jnp.asarray(gamma / gamma_scale)}


def physical(params, c_scale, gamma_scale):
    return params["readout"] * c_scale, params["slope"] * gamma_scale


@jax.custom_jvp
def tanh(x):
    return jnp.tanh(x)


def sech_squared(x):
    e = jnp.exp(-2 * jnp.abs(x))
    return 4 * e / (1 + e)**2


@tanh.defjvp
def _tanh_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    # Avoid cancellation in 1 - tanh(x)**2 at moderately large arguments.
    return tanh(x), sech_squared(x) * dx


def predict(params, x, centers, c_scale, gamma_scale):
    c, gamma = physical(params, c_scale, gamma_scale)
    return c[0] + tanh((x[:, None] - centers) * gamma) @ c[1:]


def loss(params, x, y, centers, c_scale, gamma_scale):
    r = predict(params, x, centers, c_scale, gamma_scale) - y
    return 0.5 * jnp.mean(r**2)


def optimizer(name: str, eps=1e-8):
    if name == "gd":
        return optax.sgd(1.0)
    if name == "adam":
        return optax.adam(1.0, b1=0.9, b2=0.999, eps=eps, eps_root=0.0, mu_dtype=jnp.float64)
    raise ValueError(f"Unknown optimizer {name!r}")


def optimizer_direction(name, tx, grads, state, params, epsilon_r=None, epsilon_g=None):
    """Keep Optax's moment state; optionally use explicit per-coordinate epsilons."""
    updates, state = tx.update(grads, state, params)
    if name == "adam" and epsilon_r is not None:
        adam = state[0]
        updates = {}
        for block, epsilon in [("readout", epsilon_r), ("slope", epsilon_g)]:
            mu = adam.mu[block] / (1 - jnp.asarray(.9, dtype=adam.mu[block].dtype)**adam.count)
            nu = adam.nu[block] / (1 - jnp.asarray(.999, dtype=adam.nu[block].dtype)**adam.count)
            updates[block] = -mu / (jnp.sqrt(nu) + epsilon)
    return updates, state


def initial_state(params, tx):
    return {"params": params, "opt": tx.init(params),
            "lambda_travel": jnp.zeros_like(params["slope"]),
            "readout_travel": jnp.zeros_like(params["readout"]),
            "sign_crossings": jnp.zeros(params["slope"].shape, dtype=jnp.int64),
            "gd_lambda_budget": jnp.zeros_like(params["slope"])}


TRACE_COLUMNS = ("loss_before_update", "lambda_update_rms", "readout_update_rms",
                 "lambda_gradient_rms", "readout_gradient_rms")


def make_chunk(g: Geometry, name: str, target_name: str, samples_per_cell=16, steps=100,
               batched=False, eps=1e-8):
    """Compile many steps once; rates/scales are runtime arguments, not static keys."""
    x = jnp.linspace(-1.0, 1.0, samples_per_cell * g.n + 1)
    y = target(x, target_name)
    centers = jnp.asarray(g.centers)
    distance_bound = jnp.maximum(jnp.abs(-1 - centers), jnp.abs(1 - centers))
    tx = optimizer(name, eps)

    def chunk(state, c_scale, gamma_scale, rate_r, rate_g, epsilon_r=None, epsilon_g=None):
        def step(current, _):
            params = current["params"]
            value, grads = jax.value_and_grad(loss)(params, x, y, centers, c_scale, gamma_scale)
            direction, opt_state = optimizer_direction(name, tx, grads, current["opt"], params, epsilon_r, epsilon_g)
            updates = {"readout": rate_r * direction["readout"],
                       "slope": rate_g * direction["slope"]}
            dc = c_scale * updates["readout"]
            dl = g.h * gamma_scale * updates["slope"]
            c, _ = physical(params, c_scale, gamma_scale)
            budget = current["gd_lambda_budget"]
            if name == "gd":
                budget = budget + (g.h * rate_g * gamma_scale**2 * distance_bound
                                   * jnp.abs(c[1:]) * jnp.sqrt(2 * value))
            next_params = optax.apply_updates(params, updates)
            crossings = (params["slope"] * next_params["slope"]) < 0
            next_state = {"params": next_params, "opt": opt_state,
                          "lambda_travel": current["lambda_travel"] + jnp.abs(dl),
                          "readout_travel": current["readout_travel"] + jnp.abs(dc),
                          "sign_crossings": current["sign_crossings"] + crossings,
                          "gd_lambda_budget": budget}
            stats = jnp.array([value, jnp.sqrt(jnp.mean(dl**2)),
                               jnp.sqrt(jnp.mean((dc / jnp.asarray(g.alpha))**2)),
                               jnp.sqrt(jnp.mean((grads["slope"] / (g.h * gamma_scale))**2)),
                               jnp.sqrt(jnp.mean((grads["readout"] / c_scale)**2))])
            return next_state, stats
        return jax.lax.scan(step, state, None, length=steps)

    return jax.jit(jax.vmap(chunk) if batched else chunk)
