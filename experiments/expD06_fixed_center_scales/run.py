"""Resumable constant-rate runs. A frontier is progress, never early stopping."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core, diagnostics

MIN_STEPS = 20_000
TERMINAL = {"stationary", "oscillatory", "nonfinite"}


@dataclass(frozen=True)
class Case:
    n: int = 512
    target: str = "sine"
    optimizer: str = "adam"
    arm: str = "both"
    initialization: str = "xavier"
    seed: int = 0
    rate_r: float = 1e-3
    rate_g: float = 1e-3
    samples_per_cell: int = 16
    validation_points: int = 32768
    halo_init: str = "full"
    halo_metric: str = "full"

    @property
    def key(self):
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.target}_N{self.n}_{self.optimizer}_{self.arm}_{self.initialization}_s{self.seed}_{digest}"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_state(path, state, step):
    host = jax.tree.map(lambda x: np.array(x), state)
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump({"step": step, "state": host}, stream, protocol=5)
    temporary.replace(path)


def load_state(path):
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    return jax.tree.map(jnp.asarray, data["state"]), data["step"]


def stack_states(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def unstack_state(state, index):
    return jax.tree.map(lambda x: x[index], state)


@lru_cache(maxsize=32)
def kernels(n, name, target_name, samples_per_cell, validation_points):
    g = core.geometry(n)
    chunks = {length: core.make_chunk(g, name, target_name, samples_per_cell, length, batched=True)
              for length in [1, 100]}
    train_x = jnp.linspace(-1, 1, samples_per_cell * n + 1)
    train_y = core.target(train_x, target_name)
    val_x = jnp.asarray(diagnostics.midpoint_grid(validation_points))
    tx = core.optimizer(name)

    def evaluate(state, cs, gs, rr, rg):
        params = state["params"]
        pred_train = core.predict(params, train_x, g.centers, cs, gs)
        pred_val = core.predict(params, val_x, g.centers, cs, gs)
        grad = jax.grad(core.loss)(params, train_x, train_y, g.centers, cs, gs)
        update, _ = tx.update(grad, state["opt"], params)
        c, gamma = core.physical(params, cs, gs)
        return {"c": c, "gamma": gamma, "lambda": g.h * gamma,
                "prediction_train": pred_train, "prediction_validation": pred_val,
                "gradient_c": grad["readout"] / cs,
                "gradient_lambda": grad["slope"] / (gs * g.h),
                "next_delta_c": rr * cs * update["readout"],
                "next_delta_lambda": rg * gs * g.h * update["slope"]}

    def replay(state, count, cs, gs, rr, rg):
        return jax.lax.fori_loop(0, count, lambda _, current: chunks[1](current, cs, gs, rr, rg)[0], state)

    return chunks, jax.jit(jax.vmap(evaluate)), jax.jit(replay)


def checkpoint_steps(frontier):
    steps = {0, 1, 2, 5, 10, 20, 50, 100}
    scale = 100
    while scale <= frontier:
        steps.update([scale, 2 * scale, 5 * scale])
        scale *= 10
    scale = MIN_STEPS
    while scale <= frontier:
        steps.add(scale)
        scale *= 2
    return sorted(s for s in steps if s <= frontier) + ([] if frontier in steps else [frontier])


def record_checkpoint(root, case, state, step, arrays, window_losses=None):
    folder = root / case.key
    folder.mkdir(parents=True, exist_ok=True)
    g = core.geometry(case.n)
    c, gamma = np.asarray(arrays["c"]), np.asarray(arrays["gamma"])
    with np.errstate(over="ignore", invalid="ignore"):
        finite = (all(np.isfinite(np.linalg.norm(v)) for v in arrays.values())
                  and all(np.all(np.isfinite(v)) for v in jax.tree.leaves(state))
                  and np.isfinite(np.linalg.norm(np.asarray(state["readout_travel"]) / g.alpha)))
    save_state(folder / f"state_{step:09d}.pkl", state, step)
    np.savez_compressed(folder / f"checkpoint_{step:09d}.npz", **arrays,
                        lambda_travel=np.asarray(state["lambda_travel"]),
                        readout_travel=np.asarray(state["readout_travel"]),
                        gd_lambda_budget=np.asarray(state["gd_lambda_budget"]))
    row = {"step": step, "finite": bool(finite)}
    if finite:
        x = np.linspace(-1, 1, case.samples_per_cell * case.n + 1)
        y = core.target(x, case.target, np)
        val_y = core.target(diagnostics.midpoint_grid(case.validation_points), case.target, np)
        residual = arrays["prediction_train"] - y
        _, rb, _ = diagnostics.band_residuals(residual / np.sqrt(len(x)))
        energy = np.sum(rb**2, axis=1)
        row.update({"train": diagnostics.errors(arrays["prediction_train"], y),
                    "validation": diagnostics.errors(arrays["prediction_validation"], val_y),
                    "lambda_quantiles": np.quantile(np.abs(g.h * gamma), [0, .1, .5, .9, 1]).tolist(),
                    "c_l1": float(np.abs(c).sum()), "band_energy": energy.tolist(),
                    "regions": {key: {"lambda_rms": float(np.sqrt(np.mean((g.h * gamma[mask])**2))),
                                      "weight_over_h_rms": float(np.sqrt(np.mean((c[1:][mask] / g.h)**2)))}
                                for key, mask in g.masks.items()},
                    "lambda_travel_rms": float(np.linalg.norm(np.asarray(state["lambda_travel"])) / np.sqrt(g.width)),
                    "readout_travel_scaled_rms": float(np.sqrt(np.mean((np.asarray(state["readout_travel"]) / g.alpha)**2)))})
        if window_losses is not None and len(window_losses):
            row["window_loss"] = {"mean": float(np.mean(window_losses)), "min": float(np.min(window_losses)),
                                  "max": float(np.max(window_losses)), "std": float(np.std(window_losses))}
    write_json(folder / f"metrics_{step:09d}.json", row)
    return row


def convergence_status(history, checkpoint_dir=None, min_steps=MIN_STEPS):
    """Use successively doubled windows; every finite scientific case reaches 20k."""
    if not history[-1]["finite"]:
        return "nonfinite"
    if history[-1]["step"] < min_steps:
        return "continuing"
    windows = [r for r in history if r["step"] >= min_steps and "window_loss" in r]
    if len(windows) < 4:
        return "continuing"
    recent = windows[-4:]
    if any(b["step"] != 2 * a["step"] for a, b in zip(recent, recent[1:])):
        return "continuing"
    stationary = oscillatory = True
    snapshots = None
    if checkpoint_dir is not None:
        snapshots = [np.load(Path(checkpoint_dir) / f"checkpoint_{r['step']:09d}.npz") for r in recent]
        alpha = np.load(Path(checkpoint_dir) / "reference.npz")["alpha"]
    for index, (a, b) in enumerate(zip(recent, recent[1:])):
        floor = 100 * np.finfo(float).eps * max(1, a["c_l1"], b["c_l1"])
        error_stable = abs(b["validation"]["rms"] - a["validation"]["rms"]) <= max(floor, 1e-4 * a["validation"]["rms"])
        lambda_travel = readout_travel = float("inf")
        prediction_stable = False
        if snapshots is not None:
            sa, sb = snapshots[index:index + 2]
            lambda_travel = np.sqrt(np.mean((sb["lambda_travel"] - sa["lambda_travel"])**2))
            readout_travel = np.sqrt(np.mean(((sb["readout_travel"] - sa["readout_travel"]) / alpha)**2))
            prediction_change = np.sqrt(np.mean((sb["prediction_validation"] - sa["prediction_validation"])**2))
            prediction_stable = prediction_change <= max(floor, 1e-4 * a["validation"]["rms"])
        energy_a, energy_b = np.asarray(a["band_energy"]), np.asarray(b["band_energy"])
        energy_stable = np.abs(energy_b - energy_a).sum() <= max(floor**2, 1e-3 * energy_a.sum())
        stationary &= error_stable and prediction_stable and energy_stable and lambda_travel < .25e-4 and readout_travel < 1e-4
        mean_a, mean_b = a["window_loss"]["mean"], b["window_loss"]["mean"]
        quantile_change = np.max(np.abs(np.asarray(a["lambda_quantiles"]) - np.asarray(b["lambda_quantiles"])))
        oscillatory &= (abs(mean_b - mean_a) <= max(floor**2, 1e-3 * mean_a)
                        and quantile_change < .25e-3
                        and b["window_loss"]["std"] > max(floor**2, 1e-4 * mean_b))
    if snapshots is not None:
        for snapshot in snapshots:
            snapshot.close()
    return "stationary" if stationary else "oscillatory" if oscillatory else "continuing"


def run_batch(cases, output, frontier, deadline=float("inf")):
    if frontier < MIN_STEPS:
        raise ValueError("Scientific runs require at least 20,000 updates; use core directly for checks.")
    output = Path(output)
    first = cases[0]
    shape = lambda c: (c.n, c.optimizer, c.target, c.samples_per_cell, c.validation_points)
    if any(shape(c) != shape(first) for c in cases):
        raise ValueError("A compiled batch must share width, optimizer, target, and sample grids.")
    states, starts, scales, histories = [], [], [], []
    for case in cases:
        folder = output / case.key
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "case.json", asdict(case))
        g = core.geometry(case.n)
        cs, gs = core.coordinate_scales(g, case.arm, case.halo_metric)
        scales.append((cs, gs))
        np.savez_compressed(folder / "reference.npz", centers=g.centers, alpha=g.alpha, d=g.d,
                            core=g.core, corrected_halo=g.corrected_halo, c_scale=cs, gamma_scale=gs,
                            h=g.h, radius=g.radius, lambda_reference=core.LAMBDA_REF)
        previous = folder / "latest.json"
        if previous.exists():
            summary = json.loads(previous.read_text())
            state, start = load_state(folder / f"state_{summary['step']:09d}.pkl")
            history = summary["history"]
        else:
            c, gamma = core.initial_physical(g, case.seed, case.initialization, case.halo_init)
            state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer(case.optimizer))
            start, history = 0, []
        states.append(state)
        starts.append(start)
        histories.append(history)
    if len(set(starts)) != 1:
        raise ValueError("Resume together only cases at the same step.")
    start = step = starts[0]
    state = stack_states(states)
    cs = jnp.asarray(np.stack([s[0] for s in scales]))
    gs = jnp.asarray([s[1] for s in scales])
    rr = jnp.asarray([c.rate_r for c in cases])
    rg = jnp.asarray([c.rate_g for c in cases])
    chunks, evaluate, replay = kernels(*shape(first))
    checkpoints = set(checkpoint_steps(frontier))
    all_traces = []
    event_exponents = [set(json.loads((output / c.key / "events.json").read_text())
                           if (output / c.key / "events.json").exists() else []) for c in cases]
    active = np.ones(len(cases), dtype=bool)
    wall_start = time.monotonic()

    def save(current, at_step, indices, losses=None, event=False):
        evaluated = jax.device_get(evaluate(current, cs, gs, rr, rg))
        for i in indices:
            arrays = {key: value[i] for key, value in evaluated.items()}
            row = record_checkpoint(output, cases[i], unstack_state(current, i), at_step, arrays,
                                    None if losses is None else losses[i])
            if not event:
                if not histories[i] or histories[i][-1]["step"] != at_step:
                    histories[i].append(row)
                status = convergence_status(histories[i], output / cases[i].key)
                write_json(output / cases[i].key / "latest.json",
                           {"step": at_step, "status": status, "history": histories[i],
                            "min_steps": MIN_STEPS, "elapsed_this_advance_seconds": time.monotonic() - wall_start})
                if status == "nonfinite":
                    active[i] = False

    if step == 0:
        save(state, step, range(len(cases)))
    initial_rms = np.array([h[0]["train"]["rms"] for h in histories])
    while step < frontier and time.monotonic() < deadline and active.any():
        distance = min([s - step for s in checkpoints if s > step] + [frontier - step])
        length = 100 if distance >= 100 else 1
        before = state
        state, trace = chunks[length](state, cs, gs, rr, rg)
        trace = np.asarray(trace)
        all_traces.append(trace)
        for i in np.flatnonzero(active):
            residuals = np.sqrt(2 * trace[i, :, 0])
            for exponent in range(1, 13):
                if exponent in event_exponents[i]:
                    continue
                hits = np.flatnonzero(residuals <= initial_rms[i] * 10.0**(-exponent))
                if len(hits):
                    offset = int(hits[0])
                    event_state = before if offset == 0 else replay(before, offset, cs, gs, rr, rg)
                    save(event_state, step + offset, [i], event=True)
                    event_exponents[i].add(exponent)
                    write_json(output / cases[i].key / "events.json", sorted(event_exponents[i]))
                    write_json(output / cases[i].key / f"event_reduction_1e-{exponent}.json",
                               {"step": step + offset, "residual_ratio": 10.0**(-exponent)})
        step += length
        nonfinite = ~np.isfinite(trace).all(axis=(1, 2)) & active
        if nonfinite.any():
            save(state, step, np.flatnonzero(nonfinite))
        if step in checkpoints:
            losses = np.concatenate(all_traces, axis=1)[:, :, 0] if step == frontier else None
            save(state, step, np.flatnonzero(active), losses)
    if step not in checkpoints:
        save(state, step, np.flatnonzero(active), np.concatenate(all_traces, axis=1)[:, :, 0] if all_traces else None)
    if all_traces:
        trace_array = np.concatenate(all_traces, axis=1)
        for i, case in enumerate(cases):
            np.savez_compressed(output / case.key / f"trace_{start:09d}_{step:09d}.npz",
                                start_step=start, trace=trace_array[i], columns=np.asarray(core.TRACE_COLUMNS))
    return [json.loads((output / c.key / "latest.json").read_text()) for c in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of Case dictionaries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier", type=int, default=MIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-hours", type=float, default=12.0)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("The remote scientific run requires a GPU backend.")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"environment_{os.getpid()}.json",
               {"jax": jax.__version__, "optax": optax.__version__, "devices": str(jax.devices()),
                "x64": jax.config.x64_enabled, "minimum_steps": MIN_STEPS,
                "pip_freeze": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)})
    cases = [Case(**row) for row in json.loads(args.cases.read_text())]
    deadline = time.monotonic() + args.gpu_hours * 3600
    for i in range(0, len(cases), args.batch_size):
        batch = cases[i:i + args.batch_size]
        result = run_batch(batch, args.output, args.frontier, deadline)
        print(json.dumps({"cases": [c.key for c in batch], "step": [r["step"] for r in result],
                          "status": [r["status"] for r in result]}), flush=True)
        if time.monotonic() >= deadline:
            break


if __name__ == "__main__":
    main()
