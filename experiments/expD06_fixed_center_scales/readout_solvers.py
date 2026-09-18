"""First-order frozen-dictionary solves, with a fixed neighbor-difference map."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core, diagnostics, ratio, run


def coordinate_map(g, gamma, coordinates):
    """c=Tz. Sorted centers in Geometry; signs canonicalized only in this map."""
    if np.any(np.diff(g.centers) <= 0):
        raise ValueError("Neighbor differences require increasing physical centers")
    if coordinates == "prescribed":
        return np.diag(g.d)
    if coordinates != "differences":
        raise ValueError(coordinates)
    signs = np.where(gamma >= 0, 1., -1.)
    scales = np.sqrt(np.cumsum(g.alpha[1:]))
    transform = np.zeros((g.width + 1, g.width + 1))
    transform[0, 0] = g.d[0]
    for j in range(g.width):
        transform[j + 1, j + 1] = signs[j] * scales[j]
        if j:
            transform[j + 1, j] = -signs[j] * scales[j - 1]
    return transform


def from_physical(c, g, gamma, coordinates):
    if coordinates == "prescribed":
        return c / g.d
    signs = np.where(gamma >= 0, 1., -1.)
    return np.r_[c[0] / g.d[0], np.cumsum(c[1:] * signs) / np.sqrt(np.cumsum(g.alpha[1:]))]


def dictionary(x, g, gamma, coordinates):
    a = diagnostics.features(x, g.centers, gamma)
    if coordinates == "prescribed":
        return a * g.d
    phi = a[:, 1:] * np.where(gamma >= 0, 1., -1.)
    return np.column_stack((a[:, 0] * g.d[0],
                            np.column_stack((phi[:, :-1] - phi[:, 1:], phi[:, -1]))
                            * np.sqrt(np.cumsum(g.alpha[1:]))))


def initial_state(z, algorithm, mu=None, nu=None, count=0):
    return {"z": jnp.asarray(z), "mu": jnp.asarray(np.zeros_like(z) if mu is None else mu),
            "nu": jnp.asarray(np.zeros_like(z) if nu is None else nu), "count": jnp.asarray(count),
            "eta": jnp.asarray(.001 if algorithm >= 2 else .1),
            "physical_travel": jnp.zeros_like(jnp.asarray(z)),
            "gradient_evaluations": jnp.asarray(0), "loss_evaluations": jnp.asarray(0),
            "fallbacks": jnp.asarray(0), "stagnations": jnp.asarray(0)}


TRACE_COLUMNS = ("half_mse", "eta", "gradient_norm", "physical_update_rms", "gradient_step_dot",
                 "loss_evaluations", "fallback", "stagnation")


@lru_cache(maxsize=12)
def linear_chunk(length, capture=False, batched=True):
    """Algorithm IDs: GD=0, momentum=.9=1, Adam=2, scheduled Adam=3, fixed Adam=4."""
    def advance(state, b, y, transform, algorithm, start):
        def step(current, index):
            z, mu, nu = current["z"], current["mu"], current["nu"]
            r = b @ z - y
            loss = .5 * jnp.vdot(r, r)
            grad = b.T @ r
            count = current["count"] + 1
            momentum = .9 * mu + grad
            adam_mu = .9 * mu + .1 * grad
            adam_nu = .999 * nu + .001 * grad**2
            adam = -(adam_mu / (1 - .9**count)) / (jnp.sqrt(adam_nu / (1 - .999**count)) + 1e-8)
            direction = jnp.where(algorithm == 0, -grad, jnp.where(algorithm == 1, -momentum, adam))
            mu = jnp.where(algorithm == 1, momentum, jnp.where(algorithm >= 2, adam_mu, mu))
            nu = jnp.where(algorithm >= 2, adam_nu, nu)
            fallback = (algorithm < 3) & (jnp.vdot(grad, direction) >= 0)
            direction = jnp.where(fallback, -grad, direction)
            directional = jnp.vdot(grad, direction)
            phase = jnp.minimum((start + index) / 80_000, 1.)
            scheduled = 1e-6 + .5 * (.001 - 1e-6) * (1 + jnp.cos(jnp.pi * phase))
            # The first trial is exactly the advertised initial eta; later trials
            # double the last accepted eta, capped at one.
            trial = jnp.where(start + index == 0, current["eta"], jnp.minimum(1., 2 * current["eta"]))
            trial = jnp.where(algorithm == 3, scheduled, jnp.where(algorithm == 4, 1e-6, trial))

            def candidate(eta):
                residual = b @ (z + eta * direction) - y
                return .5 * jnp.vdot(residual, residual)

            first_loss = candidate(trial)
            def keep_searching(search):
                eta, value, attempts = search
                return (algorithm < 3) & (attempts < 40) & (~jnp.isfinite(value) | (value > loss + 1e-4 * eta * directional))

            def halve(search):
                eta, _, attempts = search
                eta = eta * .5
                return eta, candidate(eta), attempts + 1

            eta, after, attempts = jax.lax.while_loop(keep_searching, halve, (trial, first_loss, jnp.asarray(1)))
            accepted = jnp.isfinite(after) & ((algorithm >= 3) | (after <= loss + 1e-4 * eta * directional))
            dz = jnp.where(accepted, eta * direction, jnp.zeros_like(z))
            dc = transform @ dz
            stagnant = (~accepted) | jnp.all(z + dz == z)
            updated = {"z": z + dz, "mu": mu, "nu": nu, "count": count, "eta": eta,
                       "physical_travel": current["physical_travel"] + jnp.abs(dc),
                       "gradient_evaluations": current["gradient_evaluations"] + 1,
                       "loss_evaluations": current["loss_evaluations"] + 1 + attempts,
                       "fallbacks": current["fallbacks"] + fallback,
                       "stagnations": current["stagnations"] + stagnant}
            trace = jnp.array([loss, eta, jnp.linalg.norm(grad), jnp.sqrt(jnp.mean(dc**2)),
                               jnp.vdot(grad, dz), 1 + attempts, fallback, stagnant])
            if capture:
                return updated, (trace, {"c": transform @ z, "delta_c": dc, "gradient_z": grad,
                                         "z": z, "mu": mu, "nu": nu})
            return updated, trace
        return jax.lax.scan(step, state, jnp.arange(length))
    return jax.jit(jax.vmap(advance, in_axes=(0, 0, None, 0, 0, None)) if batched else advance)


def specifications(learned):
    specs = [(coord + "_" + name, coord, algorithm, "zero")
             for coord in ("prescribed", "differences")
             for name, algorithm in (("gd_armijo", 0), ("momentum_armijo", 1), ("adam_armijo", 2))]
    specs.append(("prescribed_adam_scheduled", "prescribed", 3, "zero"))
    if learned:
        specs.extend([("prescribed_adam_warm_" + moments, "prescribed", 4, moments)
                      for moments in ("preserve", "reset")])
    return specs


def solve_group(output, name, case, gamma, source, frontier, deadline):
    folder = output / "solvers" / name
    folder.mkdir(parents=True, exist_ok=True)
    g = core.geometry(case.n)
    x = np.linspace(-1, 1, case.n * case.samples_per_cell + 1)
    y = jnp.asarray(core.target(x, case.target, np) / np.sqrt(len(x)))
    specs = specifications(source is not None)
    transforms = {coord: coordinate_map(g, gamma, coord) for coord in ("prescribed", "differences")}
    dictionaries = {coord: dictionary(x, g, gamma, coord) / np.sqrt(len(x)) for coord in transforms}
    metadata = {"case": case.__dict__, "gamma_sha256": hashlib.sha256(np.asarray(gamma).tobytes()).hexdigest(),
                "source": str(source) if source else "uniform lambda=0.25", "source_step": 160000 if source else 0,
                "specifications": specs, "coordinates": "prescribed c=Dz or fixed invertible neighbor differences",
                "objective": "half-MSE", "line_search": {"armijo": 1e-4, "shrink": .5, "growth": 2., "cap": 1., "maximum_trials": 40},
                "minimum_updates": 20000, "test_evaluation": False}
    run.write_json(folder / "dictionary.json", metadata)
    if not (folder / "reference.npz").exists():
        run.save_arrays(folder / "reference.npz", gamma=gamma, centers=g.centers, d=g.d, h=g.h)
    states, starts = [], []
    warm = None
    if source:
        warm, _ = run.load_state(source / "state_000160000.pkl")
    for label, coord, algorithm, mode in specs:
        path = folder / label
        path.mkdir(exist_ok=True)
        if (path / "latest.json").exists():
            latest = json.loads((path / "latest.json").read_text())
            state, at = run.load_state(path / f"state_{latest['step']:09d}.pkl")
        else:
            z = np.zeros(g.width + 1) if mode == "zero" else np.asarray(warm["params"]["readout"])
            moments = warm["opt"][0] if mode == "preserve" else None
            state = initial_state(z, algorithm, None if moments is None else moments.mu["readout"],
                                  None if moments is None else moments.nu["readout"], 0 if moments is None else int(moments.count))
            at = 0
        states.append(state)
        starts.append(at)
    at = min(starts)
    states = [run.load_state(folder / spec[0] / f"state_{at:09d}.pkl")[0] if s != at else state
              for state, s, spec in zip(states, starts, specs)]
    state = run.stack_states(states)
    b = jnp.asarray(np.stack([dictionaries[coord] for _, coord, _, _ in specs]))
    transform = jnp.asarray(np.stack([transforms[coord] for _, coord, _, _ in specs]))
    algorithm = jnp.asarray([alg for _, _, alg, _ in specs])
    traces, dense = [], []
    written = at
    started = time.monotonic()

    def save():
        nonlocal written
        host = jax.device_get(state)
        joined = np.concatenate(traces, axis=1) if traces else None
        detail = {k: np.concatenate([d[k] for d in dense], axis=1) for k in dense[0]} if dense else None
        val_x = diagnostics.midpoint_grid(case.validation_points)
        val_y = core.target(val_x, case.target, np)
        for i, (label, coord, alg, mode) in enumerate(specs):
            path = folder / label
            if joined is not None:
                run.save_arrays(path / f"trace_{written:09d}_{at:09d}.npz", trace=joined[i], columns=TRACE_COLUMNS)
            if detail is not None:
                run.save_arrays(path / f"dense_{written:09d}_{at:09d}.npz", **{k: v[i] for k, v in detail.items()})
            single = run.unstack_state(host, i)
            c = transforms[coord] @ single["z"]
            residual = dictionaries[coord] @ single["z"] - np.asarray(y)
            val = diagnostics.prediction(val_x, g.centers, c, gamma) - val_y
            run.save_state(path / f"state_{at:09d}.pkl", single, at)
            run.save_arrays(path / f"checkpoint_{at:09d}.npz", c=c, gamma=gamma, **single)
            row = {"step": at, "train_mse": float(residual @ residual), "validation_mse": float(np.mean(val**2)),
                   "gradient_evaluations": int(single["gradient_evaluations"]), "loss_evaluations": int(single["loss_evaluations"]),
                   "fallbacks": int(single["fallbacks"]), "stagnations": int(single["stagnations"]),
                   "eta": float(single["eta"]), "coefficient_l1": float(np.abs(c).sum()),
                   "minimum_completed": at >= 20000, "status": "continuing",
                   "batch_seconds_this_advance": time.monotonic() - started}
            run.write_json(path / f"metrics_{at:09d}.json", row)
            run.write_json(path / "latest.json", row)
        traces.clear()
        dense.clear()
        written = at
        print(json.dumps({"dictionary": name, "additional_steps": at}), flush=True)

    if at == 0:
        save()
    while at < frontier and time.monotonic() < deadline:
        offset = at % ratio.WINDOW
        capture = offset < 2048 or offset >= ratio.WINDOW - 2048
        boundary = 2048 if offset < 2048 else ratio.WINDOW - 2048 if offset < ratio.WINDOW - 2048 else ratio.WINDOW
        length = min(100, boundary - offset, frontier - at)
        state, evidence = linear_chunk(length, capture)(state, b, y, transform, algorithm, at)
        if capture:
            trace, detail = jax.device_get(evidence)
            dense.append({**detail, "step": np.broadcast_to(np.arange(at, at + length), (len(specs), length))})
        else:
            trace = np.asarray(evidence)
        traces.append(trace)
        at += length
        if not np.isfinite(trace).all():
            raise FloatingPointError("Nonfinite first-order solver trace")
        if at % ratio.WINDOW == 0:
            save()
    if at != written:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.root / "runs" / "ratios"
    ratio.gpu_environment(output)
    acquire, _ = ratio.main_groups(args.root, args.seed)
    tasks = []
    for index, group in enumerate(acquire):
        for branch in group:
            source = branch.folder(output)
            with np.load(source / "checkpoint_000160000.npz") as cp:
                gamma = cp["gamma"]
            tasks.append((f"{branch.case.key}_learned", branch.case, gamma, source))
        if index % 2 == args.seed:
            case = group[0].case
            g = core.geometry(case.n)
            tasks.append((f"{case.target}_N{case.n}_uniform", case, np.full(g.width, .25 / g.h), None))
    run.write_json(output / f"manifest_solvers_s{args.seed}.json", [t[0] for t in tasks])
    for end in range(20_000, 320_001, 20_000):
        for task in tasks:
            if time.monotonic() >= started + args.seconds - 120:
                return
            solve_group(output, *task, end, started + args.seconds - 120)


if __name__ == "__main__":
    main()
