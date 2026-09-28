"""Run one independent batch of modal or distributional D34 forecasts.

Degree -1 uses the full residual. --nodes 0 uses the actual finite network;
positive --nodes uses product quadrature of the initialization law. Euler's
method preserves the original GD time discretization and enables paired
half-step checks. No true future trajectory is an input to a forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import targets, transport as tr
from .recovery import PRIMARY, clean
from .run import verify_gpu, write_json


def save_npz(destination, **arrays):
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)


def inputs(args):
    n, halo = next((n, h) for n, h in targets.WIDTHS if n+2*h+1 == args.width)
    x = targets.grid(args.samples)
    records, zs, ds, ys, ws = [], [], [], [], []
    for seed in ([-1] if args.nodes else args.seeds):
        if args.nodes:
            z, d, w = tr.law_initial(args.width, args.nodes)
        else:
            z, d = targets.initial(n, halo, seed)
            w = np.ones(args.width)/args.width
            if args.evidence:
                path = args.evidence/"provenance"/f"core_N{n}_s{seed}"/"manifest.json"
                if path.exists():
                    expected = json.loads(path.read_text())["initial_hash"]
                    if targets.array_hash(z, np.array(d)) != expected:
                        raise ValueError(f"Initialization hash mismatch: {path}")
        for target in args.targets:
            y = targets.data(args.samples, target)["y"]
            records.append(dict(target=target, seed=seed, n=n, width=args.width,
                degree=args.degree, nodes=args.nodes, particles=len(w), eta=args.eta, kappa=args.kappa,
                initial_hash=targets.array_hash(z, np.array(d)), data_hash=targets.array_hash(x, y)))
            zs.append(z.copy()); ds.append(d); ys.append(y); ws.append(w.copy())
    return records, np.stack(zs), np.array(ds), np.stack(ys), np.stack(ws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--width", type=int, choices=[89, 177, 353], default=177)
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--degree", type=int, default=33)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--targets", nargs="+", choices=list(PRIMARY), default=list(PRIMARY))
    parser.add_argument("--eta", type=float, default=.002)
    parser.add_argument("--kappa", type=float, default=1.)
    parser.add_argument("--end-time", type=float, default=1200.)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--max-seconds", type=float, default=1620.)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    if args.degree < -1 or args.nodes < 0 or args.eta <= 0 or args.kappa <= 0 or args.end_time <= 0:
        parser.error("Invalid degree, particle count, rate, or horizon")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        if jax.default_backend() != "cpu":
            parser.error("CPU verification requires JAX_PLATFORMS=cpu")
    else:
        verify_gpu(args.output)
    records, z, d, y, weights = inputs(args)
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ("transport.py", "transport_run.py", "targets.py")}
    config = dict(cases=records, samples=args.samples, end_time=args.end_time,
                  source_hashes=hashes, source_commit=os.environ.get("RACE_SOURCE_COMMIT"),
                  numpy=np.__version__, jax=jax.__version__, trace_columns=tr.TRACE)
    manifest = args.output/"manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["cases"] != records or old["source_hashes"] != hashes or old["end_time"] != args.end_time:
            raise ValueError("Resume protocol or source differs from the immutable manifest")
    else:
        write_json(manifest, config)
    target_step = round(args.end_time/args.eta)
    # No evolving selection rule: fixed dense early and logarithmic later probes.
    nominal = np.unique(np.r_[np.arange(0, 2001, 20), 5000, 20000, 100000, 600000,
                               np.rint(np.geomspace(2020, 600000, 100))]).astype(int)
    probes = {round(s*.002/args.eta) for s in nominal if s*.002 <= args.end_time}
    probes.update((0, target_step))
    milestones = {round(s*.002/args.eta) for s in (0, 2000, 20000, 100000, 600000)} | {target_step}
    state = tr.initialize(z, d)
    curves, saved, step = {}, {}, 0
    restart = args.output/"restart.npz"
    if restart.exists():
        with np.load(restart) as f:
            state = {key: jnp.asarray(f[key]) for key in state}
            step = int(f["step"])
        with np.load(args.output/"curves.npz") as f:
            curves = {int(s): f["trace"][:, j] for j, s in enumerate(f["steps"])}
        with np.load(args.output/"states.npz") as f:
            saved = {int(s): {key: f[key][:, j] for key in state} for j, s in enumerate(f["steps"])}
    yy, ww = jnp.asarray(y), jnp.asarray(weights)
    advance = tr.advance_factory(args.samples, args.width, args.degree, args.eta, args.kappa)
    measure = tr.measure_factory(args.samples, args.width, args.degree, args.kappa, records[0]["n"])
    failed = {}
    prior_status = args.output/"status.json"
    accumulated_seconds = 0.
    if prior_status.exists():
        previous = json.loads(prior_status.read_text())
        failed = previous.get("failed", {})
        accumulated_seconds = previous.get("seconds", 0.)
    begin = time.monotonic()

    def checkpoint():
        host = jax.device_get(state)
        saved[step] = host
        curve_steps, state_steps = sorted(curves), sorted(saved)
        save_npz(args.output/"curves.npz", steps=np.array(curve_steps),
                 trace=np.stack([curves[s] for s in curve_steps], axis=1))
        save_npz(args.output/"states.npz", steps=np.array(state_steps), x=targets.grid(args.samples), y=y, weights=weights,
                 **{key: np.stack([saved[s][key] for s in state_steps], axis=1) for key in state})
        save_npz(restart, step=np.array(step), **host)
        status = dict(step=step, physical_time=step*args.eta, complete=step == target_step,
                      failed=failed, seconds=accumulated_seconds+time.monotonic()-begin)
        write_json(args.output/"status.json", clean(status))
        print(json.dumps(clean(status)), flush=True)

    if step not in curves:
        curves[step] = np.asarray(measure(state["z"], state["d"], yy, ww))
    if step == 0:
        checkpoint()
    schedule = sorted(probes | set(range(0, target_step, 5000)) | {target_step})
    for stop in schedule:
        if stop <= step:
            continue
        state = advance(state, yy, ww, stop-step)
        host = jax.device_get(state)
        step = stop
        finite = np.all(np.isfinite(host["z"]), axis=(1, 2)) & np.isfinite(host["d"])
        for index in np.flatnonzero(~finite):
            failed.setdefault(str(int(index)), dict(detected_step=step, case=records[index]))
        if step in probes:
            curves[step] = np.asarray(measure(state["z"], state["d"], yy, ww))
        if step in milestones:
            checkpoint()
        if time.monotonic()-begin >= args.max_seconds:
            if step not in curves:
                curves[step] = np.asarray(measure(state["z"], state["d"], yy, ww))
            checkpoint()
            if step != target_step:
                raise SystemExit(3)
    if step == target_step and not (args.output/"status.json").exists():
        checkpoint()


if __name__ == "__main__":
    main()
