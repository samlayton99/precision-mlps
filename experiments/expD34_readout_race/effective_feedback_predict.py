"""Fork-only discrete forecasts and conditional ordinary-GD travel bounds.

No future trajectory enters these predictions. Floating-point evaluations of
analytical inequalities are not directed-rounding certificates. The modified
fields need not descend the original loss; bounds below apply only to joint GD.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigvals

from . import persistence_theory as pt
from . import effective_feedback_kernel as kernel


ARMS = ("joint", "freeze_map", "clamp_residual")
DEFAULT_HORIZONS = (0, 1, 2, 10, 100, 1000, 10000, 50000, 200000,
                    500000, 2000000, 5400000)


def discrete_at(matrix, initial, horizons):
    """Evaluate M**n initial, without diagonalizing a possibly nonnormal M.

Overflow is explicit: affected horizons contain NaN and supported=False.
Earlier powers remain available. No unstable eigenvalue or mode is discarded.
"""
    matrix = np.asarray(matrix, dtype=np.float64)
    initial = np.asarray(initial, dtype=np.float64)
    steps = np.asarray(horizons, dtype=np.int64)
    if steps.ndim != 1 or np.any(steps < 0):
        raise ValueError("Horizons must be a one-dimensional nonnegative array")
    powers = []
    power = matrix.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(steps.max(initial=0)).bit_length()):
            if not np.all(np.isfinite(power)):
                break
            powers.append(power)
            power = power @ power
        values = np.full((len(steps), len(initial)), np.nan)
        supported = np.zeros(len(steps), dtype=bool)
        for row, step in enumerate(steps):
            state = initial.copy()
            remaining = int(step)
            bit = 0
            while remaining:
                if bit >= len(powers):
                    state[:] = np.nan
                    break
                if remaining & 1:
                    state = powers[bit] @ state
                remaining >>= 1
                bit += 1
            if np.all(np.isfinite(state)):
                values[row] = state
                supported[row] = True
    return values, supported


def affine_forecast(p0, gradient, derivative, eta, horizons):
    """Exact real-discrete affine model, evaluated by augmented matrix powers."""
    p0 = np.asarray(p0, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    derivative = np.asarray(derivative, dtype=np.float64)
    dimension = len(p0)
    transition = np.eye(dimension + 1)
    transition[:dimension, :dimension] -= eta * derivative
    transition[:dimension, -1] = -eta * gradient
    seed = np.zeros(dimension + 1)
    seed[-1] = 1.
    values, supported = discrete_at(transition, seed, horizons)
    displacement = values[:, :dimension]
    states = p0 + displacement
    with np.errstate(over="ignore", invalid="ignore"):
        directions = gradient + displacement @ derivative.T
    supported &= np.all(np.isfinite(directions), axis=1)
    states[~supported] = np.nan
    directions[~supported] = np.nan
    return dict(p=states, direction=directions, supported=supported)


def direction_linearization(direction, p0):
    """Differentiate the actual applied direction, including residual curvature."""
    p0 = jnp.asarray(p0)
    return (np.asarray(direction(p0)),
            np.asarray(jax.jit(jax.jacfwd(direction))(p0)))


@partial(jax.jit, static_argnames=("arm",))
def _kernel_linearization(p, context, arm):
    # Context arrays are dynamic arguments: every same-shape checkpoint reuses
    # the compiled derivative, rather than capturing a new fork in a closure.
    direction = lambda point: kernel.field(point, context, arm)[0]
    return direction(p), jax.jacfwd(direction)(p)


def frozen_effective_forecast(p0, gradient, T, J_H, e0, eta, horizons):
    """Freeze T and its Schur coupling; expose the constant remainder separately.

The pure version evolves e_next=e-eta*(J_H T)e. The remainder version
adds J_H R0 to that equation and R0 to the parameter direction. Neither
version silently identifies the retained finite basis with its complement.
"""
    p0, gradient, T, J_H, e0 = map(np.asarray, (p0, gradient, T, J_H, e0))
    h = len(e0)
    remainder = gradient - T @ e0
    coupling = J_H @ T
    result = {"R0": remainder, "S0": coupling}
    for label, residual_force in (("pure", np.zeros_like(remainder)),
                                   ("with_remainder", remainder)):
        # State is (e_H, sum of all previous e_H, constant 1).
        transition = np.eye(2*h + 1)
        transition[:h, :h] -= eta * coupling
        transition[:h, -1] = -eta * (J_H @ residual_force)
        transition[h:2*h, :h] = np.eye(h)
        seed = np.zeros(2*h + 1)
        seed[:h], seed[-1] = e0, 1.
        values, supported = discrete_at(transition, seed, horizons)
        with np.errstate(over="ignore", invalid="ignore"):
            states = p0 - eta * values[:, h:2*h] @ T.T
            states -= eta * np.asarray(horizons)[:, None] * residual_force
            force = values[:, :h] @ T.T
        supported &= np.all(np.isfinite(states), axis=1)
        supported &= np.all(np.isfinite(force), axis=1)
        states[~supported], force[~supported] = np.nan, np.nan
        result[label] = dict(p=states, effective=force, eH=values[:, :h],
                             supported=supported)
    return result


def ordinary_gd_bounds(p0, x, y, eta, horizons, radii=None):
    """First-exit enclosure from uniform tanh derivative bounds at the fork.

For a ball with GD-map Lipschitz bound beta, successive actual step lengths
are at most eta*||g0||*beta**k until first exit. Their geometric sum below
the radius closes the enclosure, including every intervening discrete step.
This bound is deliberately conservative and may be vacuous at long horizons.
"""
    p0, x, y = map(np.asarray, (p0, x, y))
    if np.max(np.abs(x)) > 1:
        raise ValueError("Uniform tanh constants require |x| <= 1")
    steps = np.asarray(horizons, dtype=np.int64)
    state = pt.tensors(p0, x, y)
    gnorm = np.linalg.norm(state["g"])
    radii = np.geomspace(1e-7, 1., 65) if radii is None else np.asarray(radii)
    path = np.full(len(steps), np.inf)
    radius_used = np.full(len(steps), np.nan)
    affine_error = np.full(len(steps), np.inf)
    beta_used = np.full(len(steps), np.nan)
    rho = np.max(np.abs(1 - eta * np.linalg.eigvalsh(state["hessian"])))
    for radius in radii:
        constants = pt.ball_constants(p0, state, float(radius), eta)
        for index, n in enumerate(steps):
            candidate = 0. if gnorm == 0 else eta*gnorm*pt.geometric(constants["beta"], int(n))
            if candidate < radius and candidate < path[index]:
                path[index] = candidate
                radius_used[index] = radius
                beta_used[index] = constants["beta"]
                # Taylor remainder at actual states; propagate with I-eta H_s.
                if candidate == 0:
                    affine_error[index] = 0.
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        affine_error[index] = .5*eta*constants["third"]*candidate**2*pt.geometric(float(rho), int(n))
    closed = np.isfinite(path)
    width = (len(p0)-1)//3
    thresholds = np.array([1., 3.2, 16.])
    counts = np.full((len(steps), len(thresholds)), -1, dtype=np.int64)
    for column, threshold in enumerate(thresholds):
        gaps = np.sort(np.maximum(threshold - abs(p0[:width]), 0.))
        distances = np.sqrt(np.cumsum(gaps*gaps))
        for index in np.flatnonzero(closed):
            # Count is for simultaneous occupancy at any one update, not the
            # number of distinct neurons ever visiting the threshold.
            counts[index, column] = np.count_nonzero(distances <= path[index])
    return dict(closed=closed, parameter_path=path, radius=radius_used,
                beta=beta_used, affine_error=affine_error,
                thresholds=thresholds, maximum_occupancy=counts)


def predict_case(p, x, y, degree=65, eta=.002, horizons=DEFAULT_HORIZONS):
    """Return {'arrays': npz-ready mapping, 'metadata': JSON-ready mapping}."""
    p = np.asarray(p, dtype=np.float64)
    steps = np.asarray(horizons, dtype=np.int64)
    context = kernel.fork_context(p, x, y, degree=degree)
    arrays = {"steps": steps, "p0": p}
    width = (len(p)-1)//3
    metadata = dict(arms=list(ARMS), eta=float(eta), degree=int(degree),
                    forecast="fork-only affine applied-gradient map",
                    effective_force="first-order applied effective force at the affine predicted state",
                    cumulative_travel="not evaluated; endpoint differences are not path lengths",
                    bound_scope="ordinary GD only; analytical inequalities evaluated in FP64, not directed rounding",
                    rates={})
    for arm in ARMS:
        gradient, derivative = map(np.asarray, _kernel_linearization(
            jnp.asarray(p), context, arm))
        forecast = affine_forecast(p, gradient, derivative, eta, steps)
        for name, value in forecast.items():
            arrays[f"{arm}_{name}"] = value
        arrays[f"{arm}_gamma"] = np.abs(forecast["p"][:, :width])
        arrays[f"{arm}_g0"] = gradient
        arrays[f"{arm}_Dg0"] = derivative
        spectrum = eigvals(derivative, check_finite=True)
        metadata["rates"][arm] = dict(
            minimum_real=float(spectrum.real.min()),
            maximum_real=float(spectrum.real.max()),
            maximum_imaginary=float(np.abs(spectrum.imag).max()),
            discrete_spectral_radius=float(np.abs(1-eta*spectrum).max()),
            unsupported_horizons=steps[~forecast["supported"]].tolist())
    # The shared remainder derivative cancels in these differences. Keeping
    # these blocks exposes the two competing fork-only feedback predictions.
    map_derivative = (arrays["joint_Dg0"]-arrays["freeze_map_Dg0"])[:width]
    error_derivative = (arrays["joint_Dg0"]-arrays["clamp_residual_Dg0"])[:width]
    initial_force = np.asarray(context["T_a0"] @ context["eH0"])
    arrays["map_force_derivative0"] = map_derivative
    arrays["error_force_derivative0"] = error_derivative
    arrays["effective_a0"] = initial_force
    for arm, derivative in (("joint", map_derivative+error_derivative),
                             ("freeze_map", error_derivative),
                             ("clamp_residual", map_derivative)):
        with np.errstate(over="ignore", invalid="ignore"):
            arrays[f"{arm}_effective_a"] = initial_force + (arrays[f"{arm}_p"]-p) @ derivative.T
    matrices = kernel.matrices(jnp.asarray(p), context)
    effective = frozen_effective_forecast(
        p, arrays["joint_g0"], matrices["T"], matrices["J_H"],
        context["eH0"], eta, steps)
    for name in ("R0", "S0"):
        arrays[f"effective_{name}"] = effective[name]
    for arm in ("pure", "with_remainder"):
        for name, value in effective[arm].items():
            arrays[f"effective_{arm}_{name}"] = value
    bound = ordinary_gd_bounds(p, x, y, eta, steps)
    for name, value in bound.items():
        arrays[f"bound_{name}"] = value
    return {"arrays": arrays, "metadata": metadata}
