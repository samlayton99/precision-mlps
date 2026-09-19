"""Constant shared-rate GD in scaled and scaled neighbor-difference coordinates."""
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

RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
COORDINATES = ("scaled", "scaled_differences")
TRACE_COLUMNS = ("half_mse", "gradient_native_readout_norm", "gradient_lambda_norm",
                 "delta_c_rms", "delta_lambda_rms", "delta_gamma_rms",
                 "gradient_c_norm", "gradient_gamma_norm", "zero_native_step",
                 "zero_physical_step", "sign_crossings", "active", "eta")


def decode(z, g, coordinates, xp=jnp):
    if coordinates == "parameter_scale":
        return z * xp.asarray(g.alpha)
    if coordinates == "scaled":
        return z * xp.asarray(g.d)
    if coordinates != "scaled_differences":
        raise ValueError(coordinates)
    q = z[1:] * xp.asarray(np.sqrt(np.cumsum(g.alpha[1:])))
    return xp.concatenate((z[:1] * g.d[0], q - xp.concatenate((xp.zeros(1), q[:-1]))))


def encode(c, g, coordinates):
    if coordinates == "parameter_scale":
        return c / g.alpha
    if coordinates == "scaled":
        return c / g.d
    if coordinates != "scaled_differences":
        raise ValueError(coordinates)
    return np.r_[c[0] / g.d[0], np.cumsum(c[1:]) / np.sqrt(np.cumsum(g.alpha[1:]))]


def pullback(gc, g, coordinates):
    if coordinates == "parameter_scale":
        return gc * jnp.asarray(g.alpha)
    if coordinates == "scaled":
        return gc * jnp.asarray(g.d)
    s = jnp.asarray(np.sqrt(np.cumsum(g.alpha[1:])))
    return jnp.concatenate((gc[:1] * g.d[0], s * (gc[1:] - jnp.r_[gc[2:], 0.])))


def initial(g, seed, coordinates, optimizer="gd"):
    c, gamma = core.initial_physical(g, seed, "xavier_a_reference")
    state = {"z": jnp.asarray(encode(c, g, coordinates)), "lam": jnp.asarray(g.h * gamma),
            "failed": jnp.asarray(0, dtype=jnp.int64), "travel_c": jnp.zeros_like(jnp.asarray(c)),
            "travel_lambda": jnp.zeros_like(jnp.asarray(gamma))}
    if optimizer == "adam":
        state["opt"] = core.optimizer("adam").init({"readout": state["z"], "slope": state["lam"]})
    elif optimizer != "gd":
        raise ValueError(optimizer)
    return state


def physical_loss(c, lam, x, y, g):
    r = c[0] + core.tanh((x[:, None] - jnp.asarray(g.centers)) * (lam / g.h)) @ c[1:] - y
    return .5 * jnp.mean(r * r)


@lru_cache(maxsize=48)
def chunk(n, coordinates, length, capture=False, samples_per_cell=16, batched=True, optimizer="gd"):
    g = core.geometry(n)
    x = jnp.linspace(-1, 1, samples_per_cell * n + 1)
    y = core.target(x, "sine")
    derivative = jax.value_and_grad(lambda c, lam: physical_loss(c, lam, x, y, g), argnums=(0, 1))
    tx = core.optimizer(optimizer)
    # Preserve the control's physical epsilon=1e-8/D in both diagonal maps.
    epsilon_r = 1e-8 * (g.d if coordinates == "parameter_scale" else np.ones(g.width+1))
    if optimizer == "adam" and coordinates not in ("scaled", "parameter_scale"):
        raise ValueError("Adam comparison supports the two diagonal normalizations")

    def advance(state, eta, start):
        def step(current, offset):
            c = decode(current["z"], g, coordinates)
            loss, (gc, gl) = derivative(c, current["lam"])
            gz = pullback(gc, g, coordinates)
            if optimizer == "adam":
                params = {"readout": current["z"], "slope": current["lam"]}
                direction, opt = core.optimizer_direction("adam", tx, {"readout": gz, "slope": gl},
                                                         current["opt"], params, jnp.asarray(epsilon_r), 1e-8)
                dz, dl_proposed = eta * direction["readout"], eta * direction["slope"]
            else:
                dz, dl_proposed = -eta * gz, -eta * gl
            z1, l1 = current["z"] + dz, current["lam"] + dl_proposed
            c1 = decode(z1, g, coordinates)
            finite_before = jnp.isfinite(loss) & jnp.all(jnp.isfinite(gc)) & jnp.all(jnp.isfinite(gl))
            finite_after = jnp.all(jnp.isfinite(z1)) & jnp.all(jnp.isfinite(l1)) & jnp.all(jnp.isfinite(c1))
            if optimizer == "adam":
                finite_after &= jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(opt)]))
            active = (current["failed"] == 0) & finite_before & finite_after
            # Failed cases retain their state and cannot poison other vmap members.
            z1 = jnp.where(active, z1, current["z"])
            l1 = jnp.where(active, l1, current["lam"])
            dc = decode(z1, g, coordinates) - c
            dl = l1 - current["lam"]
            failed = jnp.where((current["failed"] == 0) & ~active, start + offset + 1, current["failed"])
            next_state = {"z": z1, "lam": l1, "failed": failed,
                          "travel_c": current["travel_c"] + jnp.abs(dc),
                          "travel_lambda": current["travel_lambda"] + jnp.abs(dl)}
            if optimizer == "adam":
                next_state["opt"] = jax.tree.map(lambda new, old: jnp.where(active, new, old), opt, current["opt"])
            trace = jnp.array([jnp.where(active, loss, jnp.nan), jnp.linalg.norm(gz), jnp.linalg.norm(gl),
                               jnp.sqrt(jnp.mean(dc**2)), jnp.sqrt(jnp.mean(dl**2)),
                               jnp.sqrt(jnp.mean((dl/g.h)**2)), jnp.linalg.norm(gc), jnp.linalg.norm(gl*g.h),
                               jnp.all(z1 == current["z"]) & jnp.all(l1 == current["lam"]),
                               jnp.all(dc == 0) & jnp.all(dl == 0),
                               jnp.sum(current["lam"] * l1 < 0), active, eta])
            if capture:
                return next_state, (trace, {"c": c, "gamma": current["lam"] / g.h,
                                           "gradient_c": gc, "gradient_lambda": gl,
                                           "delta_c": dc, "delta_lambda": dl})
            return next_state, trace
        return jax.lax.scan(step, state, jnp.arange(length))
    return jax.jit(jax.vmap(advance, in_axes=(0, 0, None)) if batched else advance)


@lru_cache(maxsize=8)
def evaluator(n, coordinates, samples_per_cell=16, optimizer="gd"):
    g = core.geometry(n)
    x = jnp.linspace(-1, 1, samples_per_cell*n+1)
    xv = jnp.asarray(diagnostics.midpoint_grid(32768))
    y, yv = core.target(x, "sine"), core.target(xv, "sine")
    def evaluate(state):
        c = decode(state["z"], g, coordinates)
        gamma = state["lam"] / g.h
        phi = core.tanh((x[:, None] - jnp.asarray(g.centers)) * gamma)
        pred = c[0] + phi @ c[1:]
        # Bounded validation kernels avoid a giant fused reduction in CUDA compilation.
        def validation_block(xblock):
            return c[0] + core.tanh((xblock[:, None] - jnp.asarray(g.centers)) * gamma) @ c[1:]
        pv = jax.lax.map(validation_block, xv.reshape(-1, 512)).reshape(-1)
        q = jnp.cumsum(c[1:])
        alternate = c[0] + (phi[:, :-1]-phi[:, 1:]) @ q[:-1] + phi[:, -1]*q[-1]
        result = {"c": c, "gamma": gamma, "lambda": state["lam"], "z": state["z"],
                "prediction_train": pred, "prediction_validation": pv, "prediction_alternate": alternate,
                "travel_c": state["travel_c"], "travel_lambda": state["travel_lambda"]}
        if optimizer == "adam":
            adam = state["opt"][0]
            result["adam_count"] = adam.count
            epsilon = 1e-8 * (g.d if coordinates == "parameter_scale" else np.ones(g.width+1))
            for block, eps in (("readout", jnp.asarray(epsilon)), ("slope", 1e-8)):
                result[f"adam_{block}_mu"] = adam.mu[block]
                result[f"adam_{block}_nu"] = adam.nu[block]
                result[f"adam_{block}_sqrt_v_over_epsilon"] = jnp.sqrt(adam.nu[block] / jnp.maximum(1-.999**adam.count, 1e-300))/eps
        return result
    return jax.jit(jax.vmap(evaluate))


def case_key(case):
    prefix = f"{case['optimizer']}_" if case.get("campaign") == "parameter_scale" else ""
    return f"{prefix}N{case['n']}_s{case['seed']}_{case['coordinates']}_eta{case['eta']:.8g}"


def checkpoints(frontier):
    return sorted({s for s in (0, 1, 10, 100, 1000, *range(2000, frontier+1, 2000), frontier) if s <= frontier})


def dense_windows(frontier):
    ends = sorted({20000, 60000, 100000, *range(200000, frontier+1, 100000)})
    return [(end-2048, end) for end in ends if end <= frontier]


def advance_group(output, cases, frontier, deadline=float("inf"), samples_per_cell=16):
    n, coord, seed = (cases[0][k] for k in ("n", "coordinates", "seed"))
    optimizer = cases[0].get("optimizer", "gd")
    parameter_campaign = cases[0].get("campaign") == "parameter_scale"
    if any((c["n"], c["coordinates"], c["seed"]) != (n, coord, seed) for c in cases):
        raise ValueError("A batch must share width, coordinates, and seed")
    if any(c.get("optimizer", "gd") != optimizer for c in cases):
        raise ValueError("A batch must share optimizer")
    if any(c["eta"] <= 0 or not np.isfinite(c["eta"]) for c in cases):
        raise ValueError("Each case needs one positive finite eta")
    g = core.geometry(n)
    paths = [output / case_key(c) for c in cases]
    latest = [json.loads((p/'latest.json').read_text()) if (p/'latest.json').exists() else {"step": 0} for p in paths]
    at = min(r["step"] for r in latest)
    if at >= frontier:
        return
    states = []
    for case, path in zip(cases, paths):
        path.mkdir(parents=True, exist_ok=True)
        metadata = dict(case, optimizer=optimizer, target="sine", initialization="xavier_a_reference",
                        samples_per_cell=samples_per_cell, validation_points=32768, objective="half-MSE",
                        reference_lambda=.25, constant_shared_rate=True)
        if parameter_campaign:
            metadata["epsilon_mode"] = "matched_reference"
        if (path/'case.json').exists() and json.loads((path/'case.json').read_text()) != metadata:
            raise ValueError(f"Changed configuration at {path}")
        run.write_json(path/'case.json', metadata)
        run.save_arrays(path/'reference.npz', centers=g.centers, alpha=g.alpha, d=g.d, h=g.h,
                        core=g.core, corrected_halo=g.corrected_halo)
        states.append(run.load_state(path/f'state_{at:09d}.pkl')[0] if (path/f'state_{at:09d}.pkl').exists()
                      else initial(g, seed, coord, optimizer))
        if at and not (path/f'state_{at:09d}.pkl').exists():
            raise ValueError(f"Missing common resume state at {path}, {at}")
    state = run.stack_states(states)
    eta = jnp.asarray([c['eta'] for c in cases])
    targets = checkpoints(frontier)
    windows = dense_windows(frontier)
    if parameter_campaign:
        targets = sorted(set([s for s in targets if s <= 100000 or s % 10000 == 0 or s == frontier]
                             + [s for s in (40000,80000,160000,320000,640000,1280000,2560000,5120000) if s <= frontier]))
        windows = [(0, min(2048, frontier))] + windows
    boundaries = sorted(set(targets + [v for pair in windows for v in pair]))
    traces, dense, trace_start = [], [], at
    started = time.monotonic()

    def save():
        nonlocal traces, dense, trace_start
        host = jax.device_get(state)
        arrays = jax.device_get(evaluator(n, coord, samples_per_cell, optimizer)(state))
        joined = np.concatenate(traces, axis=1) if traces else None
        detail = {k: np.concatenate([d[k] for d in dense], axis=1) for k in dense[0]} if dense else None
        for i, path in enumerate(paths):
            if joined is not None:
                # Raw NPZ avoids compression stalls inside the GPU allocation.
                np.savez(path/f'trace_{trace_start:09d}_{at:09d}.npz', trace=joined[i], columns=TRACE_COLUMNS)
            if detail is not None:
                np.savez(path/f'dense_{int(detail["step"][i,0]):09d}_{at:09d}.npz', **{k:v[i] for k,v in detail.items()})
            single = run.unstack_state(host, i)
            data = {k:v[i] for k,v in arrays.items()}
            yt = core.target(np.linspace(-1, 1, samples_per_cell*n+1), "sine", np)
            yv = core.target(diagnostics.midpoint_grid(32768), "sine", np)
            data['train_mse'] = np.mean((data['prediction_train']-yt)**2)
            data['validation_mse'] = np.mean((data.pop('prediction_validation')-yv)**2)
            data['alternate_eval_max'] = np.max(np.abs(data['prediction_train']-data.pop('prediction_alternate')))
            run.save_arrays(path/f'checkpoint_{at:09d}.npz', **data)
            run.save_state(path/f'state_{at:09d}.pkl', single, at)
            fail = int(single['failed'])
            row = dict(step=at, completed_updates=fail-1 if fail else at,
                       failed_update=fail or None, status="nonfinite" if fail else "continuing",
                       train_mse=float(data['train_mse']) if np.isfinite(data['train_mse']) else None,
                       validation_mse=float(data['validation_mse']) if np.isfinite(data['validation_mse']) else None,
                       advance_seconds=time.monotonic()-started)
            run.write_json(path/'latest.json', row)
        traces, dense, trace_start = [], [], at

    if at == 0:
        save()
    while at < frontier and time.monotonic() < deadline:
        boundary = min(s for s in boundaries if s > at)
        distance = boundary-at
        length = 1000 if distance >= 1000 else 100 if distance >= 100 else 1
        capture = any(lo <= at < hi for lo, hi in windows)
        state, result = chunk(n, coord, length, capture, samples_per_cell, optimizer=optimizer)(state, eta, at)
        result = jax.device_get(result)
        if capture:
            trace, detail = result
            detail['step'] = np.broadcast_to(np.arange(at, at+length), (len(cases), length))
            dense.append(detail)
        else:
            trace = result
        traces.append(trace)
        at += length
        if at in targets or at == boundary and any(at == hi for _, hi in windows):
            save()
            print(json.dumps(dict(n=n, coordinates=coord, seed=seed, step=at,
                                  elapsed=time.monotonic()-started)), flush=True)
    if at != trace_start:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=[0, 1], required=True)
    parser.add_argument('--frontier', type=int, default=100000)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.require_gpu:
        ratio.gpu_environment(args.output)
    cases = json.loads(args.cases.read_text())
    groups = {}
    for case in cases:
        if case['coordinates'] == COORDINATES[args.worker]:
            groups.setdefault((case['n'], case['seed'], case['coordinates']), []).append(case)
    run.write_json(args.output/f'manifest_worker{args.worker}.json', {
        'cases': cases, 'frontier': args.frontier,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    # Longer selected continuations advance every seed in common 100k blocks.
    frontiers = sorted({*range(100000, args.frontier+1, 100000), args.frontier})
    for frontier in frontiers:
        for group in groups.values():
            if time.monotonic() >= started+args.seconds-30:
                return
            advance_group(args.output, group, frontier, started+args.seconds-30)


if __name__ == '__main__':
    main()
