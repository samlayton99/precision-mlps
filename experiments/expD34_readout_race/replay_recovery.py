"""Replay 15 existing equal-rate cases, saving only compact diagnostic states.

Uses the unchanged D34 vector field and validates initial/data hashes plus
archived scalar observations. Cumulative neuronwise travel is updated at every
step; it is never reconstructed by interpolating sparse checkpoints.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core, targets
from .recovery import PRIMARY, PACKAGES, clean, load_curated


@lru_cache(maxsize=4)
def advance_function(m, eta):
    x = jnp.asarray(targets.grid(m))
    powers = x[:, None] ** jnp.arange(11)

    def one(state, y, count):
        def update(_, old):
            z, d = old["z"], old["d"]
            _, _, g, gd, coarse = core.tanh_field(z, d, x, y, powers)
            zn, dn = z - eta*g, d - eta*gd
            delta = jnp.abs(zn[0]) - jnp.abs(z[0])
            return dict(z=zn, d=dn,
                positive=old["positive"] + jnp.maximum(delta, 0),
                negative=old["negative"] + jnp.maximum(-delta, 0),
                path=old["path"] + eta*jnp.linalg.norm(g[0]),
                coarse=old["coarse"] - eta*jnp.mean(jnp.sign(z[0])*coarse),
                tail=old["tail"] - eta*jnp.mean(jnp.sign(z[0])*(g[0]-coarse)))
        return jax.lax.fori_loop(0, count, update, state)
    return jax.jit(jax.vmap(one, in_axes=(0, 0, None)))


def initial_state(z, d):
    z, d = jnp.asarray(z), jnp.asarray(d)
    return dict(z=z, d=d, positive=jnp.zeros_like(z[:, 0]), negative=jnp.zeros_like(z[:, 0]),
                path=jnp.zeros_like(d), coarse=jnp.zeros_like(d), tail=jnp.zeros_like(d))


def scientific_inputs(evidence, seeds, selected=PRIMARY):
    cases, zz, dd, yy, expected, checks = [], [], [], [], [], []
    schedule = {0, 20000, 100000, 600000}
    for seed in seeds:
        z, d = targets.initial(128, 24, seed)
        source = evidence / "provenance" / f"core_N128_s{seed}" / "manifest.json"
        manifest = json.loads(source.read_text())
        all_data = [targets.data(2048, name) for name in targets.TARGETS]
        ih = targets.array_hash(z, np.array(d))
        # The historical hash includes all seven repeated rate arms in order.
        dh = targets.array_hash(all_data[0]["x"], np.stack([
            all_data[targets.TARGETS.index(case["target"])]["y"] for case in manifest["cases"]]))
        if ih != manifest["initial_hash"] or dh != manifest["data_hash"]:
            raise ValueError(f"Archived initialization or target mismatch: seed {seed}")
        checks.append(dict(seed=seed, initial_hash=ih, data_hash=dh))
        for target in selected:
            cases.append(dict(seed=seed, target=target))
            zz.append(z.copy()); dd.append(d)
            yy.append(all_data[targets.TARGETS.index(target)]["y"])
        for package in PACKAGES:
            path = evidence / package / f"core_N128_s{seed}_curves.npz"
            arrays, cfg, archived_cases, cols = load_curated(path)
            schedule.update(int(step) for step in arrays["p0_steps"])
            for target in selected:
                ci = next(i for i, case in enumerate(archived_cases)
                          if case["target"] == target and case["kappa"] == 1)
                expected.extend(dict(seed=seed, target=target, step=int(step),
                                     half_mse=float(tr[cols["half_mse"]]),
                                     mean_gamma=float(tr[cols["mean_gamma"]]),
                                     grad_a_norm=float(tr[cols["grad_a_norm"]]),
                                     package=package)
                                for step, tr in zip(arrays["p0_steps"], arrays["p0_trace"][ci]))
    schedule.update(range(0, 2001, 20))
    schedule.update(np.rint(np.geomspace(2020, 600000, 100)).astype(int).tolist())
    return cases, np.stack(zz), np.array(dd), np.stack(yy), sorted(schedule), expected, checks


def validate_observations(snapshots, expected, cases, x, y):
    """Require replay agreement before interpreting new diagnostics.

    Absolute tolerances allow backend rounding; the small gradient also gets
    a relative check. No tolerance is adjusted using the observed replay.
    """
    powers = jnp.asarray(x[:, None] ** np.arange(11))
    xx = jnp.asarray(x)
    @jax.jit
    def measure(z, d, yy):
        loss, _, g, _, _ = core.tanh_field(z, d, xx, yy, powers)
        return jnp.array([loss, jnp.mean(jnp.abs(z[0])), jnp.linalg.norm(g[0])])
    lookup = {(case["seed"], case["target"]): i for i, case in enumerate(cases)}
    cache, comparisons = {}, []
    for row in expected:
        if row["step"] not in snapshots:
            continue
        index = lookup[row["seed"], row["target"]]
        key = row["step"], index
        if key not in cache:
            state = snapshots[row["step"]]
            cache[key] = np.asarray(measure(jnp.asarray(state["z"][index]), state["d"][index], jnp.asarray(y[index])))
        truth = np.array([row[name] for name in ("half_mse", "mean_gamma", "grad_a_norm")])
        actual = cache[key]
        tolerances = np.array([1e-10, 1e-9, 1e-12]) + 1e-7 * np.abs(truth)
        error = np.abs(actual - truth)
        comparisons.append(dict(seed=row["seed"], target=row["target"], package=row["package"],
            step=row["step"], max_error_ratio=float(np.max(error/tolerances)),
            loss_error=float(error[0]), mean_gamma_error=float(error[1]), gradient_error=float(error[2])))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--targets", choices=targets.TARGETS, nargs="+", default=PRIMARY)
    parser.add_argument("--frontier", type=int, default=600000)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("Set JAX_ENABLE_X64=true")
    args.output.mkdir(parents=True, exist_ok=True)
    from .run import verify_gpu
    verify_gpu(args.output)
    cases, z, d, y, schedule, expected, checks = scientific_inputs(args.evidence, args.seeds, args.targets)
    schedule = sorted(set(s for s in schedule if s <= args.frontier) | {args.frontier})
    x = targets.grid(2048)
    state = initial_state(z, d)
    snapshots = {0: jax.device_get(state)}
    step, begin = 0, time.monotonic()
    advance = advance_function(2048, .002)
    def save():
        saved_steps = sorted(snapshots)
        arrays = {key: np.stack([snapshots[s][key] for s in saved_steps], axis=1) for key in state}
        np.savez_compressed(args.output / "compact_states.npz", steps=np.array(saved_steps),
                            cases=np.array(json.dumps(cases)), x=x, y=y, **arrays)
    for stop in schedule[1:]:
        state = advance(state, jnp.asarray(y), stop-step)
        host = jax.device_get(state)
        if not all(np.all(np.isfinite(v)) for v in host.values()):
            raise FloatingPointError(f"Nonfinite replay state at {stop}")
        snapshots[stop] = host
        step = stop
        if stop in (20000, 100000, args.frontier):
            comparisons = validate_observations(snapshots, expected, cases, x, y)
            maximum = max(row["max_error_ratio"] for row in comparisons)
            status = dict(step=stop, seconds=time.monotonic()-begin, cases=cases,
                verification_passed=maximum <= 1, maximum_error_ratio=maximum,
                initial_checks=checks, observations=comparisons,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
                       if Path(".git").exists() else os.environ.get("RACE_SOURCE_COMMIT"),
                source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (args.output / "replay_verification.json").write_text(json.dumps(clean(status), indent=2) + "\n")
            save()
            print(json.dumps({k: status[k] for k in ("step", "seconds", "verification_passed", "maximum_error_ratio")}), flush=True)
            if maximum > 1:
                raise ValueError("Replay differs materially from the archived trajectories")


if __name__ == "__main__":
    main()
