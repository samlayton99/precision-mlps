"""Paired rate schedules in fixed theory coordinates; absolute update indices."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
import optax

from . import core, run

WINDOW = 20_000
ARMS = {"shared": (1e-6, 1e-6), "slow_geometry": (1e-6, 1e-7),
        "faster_readout": (1e-5, 1e-6), "both_changes": (1e-5, 1e-7)}


def dense_sample_indices():
    """Fixed stratified sample avoids aliasing a regular optimizer oscillation."""
    offsets = np.random.default_rng(391).integers(0, 32, 64)
    offsets[0] = 0
    return np.arange(64) * 32 + offsets


def schedule(step, knots):
    """Cosine interpolation of [absolute step, eta_a, eta_lambda] rows."""
    knots = jnp.asarray(knots)
    rates = knots[0, 1:]
    for left, right in zip(knots[:-1], knots[1:]):
        phase = jnp.clip((step - left[0]) / (right[0] - left[0]), 0., 1.)
        blend = .5 * (1 - jnp.cos(jnp.pi * phase))
        rates = jnp.where(step >= left[0], left[1:] + blend * (right[1:] - left[1:]), rates)
    return rates


def primary_schedule(acquisition, arm="shared"):
    ra, rg = ARMS[arm]
    return np.array([[0, acquisition, acquisition], [80_000, acquisition, acquisition],
                     [160_000, 1e-6, 1e-6], [180_000, ra, rg]], dtype=float)


def early_schedule():
    return np.array([[0, .01, .01], [20_000, .01, .01], [40_000, .01, .001],
                     [80_000, .01, .001], [160_000, 1e-6, 1e-7]], dtype=float)


@dataclass(frozen=True)
class Branch:
    case: run.Case
    label: str
    knots: tuple
    source: str = ""
    source_step: int = 0
    settle_after: int = 180_000

    def folder(self, root):
        return root / self.label / self.case.key


def make_branch(n, target, seed, rate, label, knots, source="", source_step=0, settle_after=180_000):
    return Branch(run.Case(n=n, target=target, seed=seed, initialization="envelope",
                           rate_r=rate, rate_g=rate), label,
                  tuple(tuple(float(v) for v in row) for row in knots), str(source), source_step, settle_after)


@lru_cache(maxsize=64)
def chunk(n, target, samples, length, capture, batched=True):
    g = core.geometry(n)
    cs, gs = core.coordinate_scales(g, "both")
    one = core.make_chunk(g, "adam", target, samples, 1, capture=capture)

    def advance(state, start, knots):
        def step(current, index):
            rates = schedule(start + index, knots)
            updated, evidence = one(current, cs, gs, rates[0], rates[1])
            return updated, jax.tree.map(lambda v: v[0], evidence)
        return jax.lax.scan(step, state, jnp.arange(length))
    return jax.jit(jax.vmap(advance, in_axes=(0, None, 0)) if batched else advance)


def prepare(root, branch):
    folder = branch.folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    g = core.geometry(branch.case.n)
    cs, gs = core.coordinate_scales(g, "both")
    metadata = asdict(branch)
    metadata.update(step_convention="absolute updates from initialization", coordinates="c=D a; gamma=lambda/h",
                    initialization="physical Xavier slopes; legacy signed reference-envelope readouts",
                    rate_columns=["absolute_step", "eta_a", "eta_lambda"], moment_policy="preserve",
                    physical_adam_factors="eta_a*D; eta_lambda/h; physical epsilon=1e-8/scale",
                    training_loss="half mean squared error", test_evaluation=False)
    if branch.source:
        source_state = Path(branch.source) / f"state_{branch.source_step:09d}.pkl"
        metadata["source_sha256"] = hashlib.sha256(source_state.read_bytes()).hexdigest()
    path = folder / "schedule.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(metadata)):
        raise ValueError(f"Schedule/provenance changed on resume: {folder}")
    run.write_json(path, metadata)
    run.write_json(folder / "case.json", asdict(branch.case))
    run.save_arrays(folder / "reference.npz", centers=g.centers, alpha=g.alpha, d=g.d,
                    core=g.core, corrected_halo=g.corrected_halo, c_scale=cs, gamma_scale=gs,
                    h=g.h, radius=g.radius, lambda_reference=core.LAMBDA_REF)
    if (folder / "latest.json").exists():
        latest = json.loads((folder / "latest.json").read_text())
        state, step = run.load_state(folder / f"state_{latest['step']:09d}.pkl")
        return state, step, latest["history"]
    if branch.source:
        state, step = run.load_state(source_state)
        if step != branch.source_step:
            raise ValueError("Source step mismatch")
    else:
        c, gamma = core.initial_physical(g, branch.case.seed, "envelope")
        state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
        step = 0
    return state, step, []


def advance_group(root, branches, frontier, deadline=float("inf"), runtime_knots=None, settle_after=None):
    """Advance a matched group. Durable traces precede checkpoint promotion."""
    case = branches[0].case
    if any((b.case.n, b.case.target, b.case.samples_per_cell, b.case.validation_points) !=
           (case.n, case.target, case.samples_per_cell, case.validation_points) for b in branches):
        raise ValueError("A batch must share its dimensions and target")
    prepared = [prepare(root, b) for b in branches]
    at = min(p[1] for p in prepared)
    states, histories = [], []
    for branch, (state, step, history) in zip(branches, prepared):
        if step != at:
            state, _ = run.load_state(branch.folder(root) / f"state_{at:09d}.pkl")
        states.append(state)
        histories.append([r for r in history if r["step"] <= at])
    state = run.stack_states(states)
    g = core.geometry(case.n)
    cs, gs = core.coordinate_scales(g, "both")
    cs, gs = jnp.broadcast_to(cs, (len(branches), len(cs))), jnp.full(len(branches), gs)
    knots = jnp.asarray(runtime_knots if runtime_knots is not None else [b.knots for b in branches])
    _, evaluate, _, _ = run.kernels(case.n, "adam", case.target, case.samples_per_cell, case.validation_points)
    traces, dense = [], []
    written = at

    def save():
        nonlocal written
        if traces:
            trace = np.concatenate(traces, axis=1)
            for i, branch in enumerate(branches):
                run.save_arrays(branch.folder(root) / f"trace_{written:09d}_{at:09d}.npz",
                                trace=trace[i], columns=np.asarray(core.TRACE_COLUMNS), start_step=written)
            traces.clear()
        if dense:
            detail = {k: np.concatenate([d[k] for d in dense], axis=1) for k in dense[0]}
            for i, branch in enumerate(branches):
                run.save_arrays(branch.folder(root) / f"dense_{written:09d}_{at:09d}.npz",
                                **{k: v[i] for k, v in detail.items()})
            dense.clear()
        rates = jax.vmap(lambda k: schedule(at, k))(knots)
        evaluated = jax.device_get(evaluate(state, cs, gs, rates[:, 0], rates[:, 1]))
        for i, branch in enumerate(branches):
            folder = branch.folder(root)
            settling_origin = branch.settle_after if settle_after is None else settle_after
            elapsed = at - settling_origin
            multiple = elapsed // WINDOW
            begin = max(branch.source_step, at - WINDOW)
            if elapsed >= WINDOW and elapsed % WINDOW == 0 and multiple & (multiple - 1) == 0:
                begin = settling_origin + (0 if multiple == 1 else elapsed // 2)
            losses = run.trace_window_losses(folder, begin, at) if begin < at else None
            arrays = {k: v[i] for k, v in evaluated.items()}
            arrays.update(eta_a=np.asarray(rates[i, 0]), eta_lambda=np.asarray(rates[i, 1]),
                          physical_readout_rate_factor=np.asarray(rates[i, 0]) * g.d,
                          physical_gamma_rate_factor=np.asarray(rates[i, 1]) / g.h)
            row = run.record_checkpoint(folder.parent, branch.case, run.unstack_state(state, i), at, arrays, losses)
            row.update(eta_a=float(rates[i, 0]), eta_lambda=float(rates[i, 1]))
            histories[i] = [r for r in histories[i] if r["step"] < at] + [row]
            status = run.convergence_status(histories[i], folder, step_offset=settling_origin)
            run.write_json(folder / "latest.json", {"step": at, "status": status, "history": histories[i],
                           "source_step": branch.source_step, "minimum_new_updates": WINDOW,
                           "completed_minimum": at - branch.source_step >= WINDOW})
        print(json.dumps({"labels": [b.label for b in branches], "n": case.n, "target": case.target,
                          "seed": case.seed, "step": at, "validation_mse":
                          [h[-1]["validation"]["rms"]**2 for h in histories]}), flush=True)
        written = at

    if not all(h for h in histories):
        save()
    while at < frontier and time.monotonic() < deadline:
        offset = at % WINDOW
        capture = offset < 2048 or offset >= WINDOW - 2048
        boundary = 2048 if offset < 2048 else WINDOW - 2048 if offset < WINDOW - 2048 else WINDOW
        length = min(100, boundary - offset, frontier - at)
        state, evidence = chunk(case.n, case.target, case.samples_per_cell, length, capture)(state, at, knots)
        if capture:
            trace, detail = jax.device_get(evidence)
            dense.append({**detail, "step": np.broadcast_to(np.arange(at, at + length), (len(branches), length))})
        else:
            trace = np.asarray(evidence)
        traces.append(trace)
        at += length
        if not np.isfinite(trace).all():
            save()
            raise FloatingPointError("Nonfinite batch saved; inspect before continuation")
        if at % WINDOW == 0:
            save()
    if at != written:
        save()
    return at


def cells(seed):
    return [(n, target, seed) for n in (512, 1024) for target in ("sine", "quadratic", "mixed")]


def acquisition_branches(root, n, target, seed):
    result = []
    for rate in (.001, .01):
        b = make_branch(n, target, seed, rate, f"acquire_{rate:g}", primary_schedule(rate))
        source = root / "runs" / "pilot" / b.case.key
        if (source / "state_000080000.pkl").exists():
            b = make_branch(n, target, seed, rate, b.label, b.knots, source, 80_000, 160_000)
        result.append(b)
    return result


def main_groups(root, seed):
    output = root / "runs" / "ratios"
    acquire = [acquisition_branches(root, *cell) for cell in cells(seed)]
    tails = []
    for low, high in acquire:
        n, target = high.case.n, high.case.target
        tails.append([make_branch(n, target, seed, .001, "low_shared", primary_schedule(.001),
                                  low.folder(output), 160_000)] +
                     [make_branch(n, target, seed, .01, "high_" + arm, primary_schedule(.01, arm),
                                  high.folder(output), 160_000) for arm in ARMS])
    return acquire, tails


def gpu_environment(root):
    if not all(os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("Use an allocated Slurm GPU step, preserving its mask")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]], text=True)
    step = subprocess.check_output(["scontrol", "show", "step",
                                   f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"], text=True)
    if "JobState=RUNNING" not in allocation or "State=RUNNING" not in step or "gpu" not in step:
        raise RuntimeError("Scheduler did not confirm a running GPU step")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1 or not jax.config.x64_enabled:
        raise RuntimeError("Exactly one allocated GPU and FP64 required")
    root.mkdir(parents=True, exist_ok=True)
    run.write_json(root / f"environment_{os.environ['SLURM_JOB_ID']}.json", {
        "allocation": allocation, "step": step, "devices": str(jax.devices()),
        "mask": os.environ["CUDA_VISIBLE_DEVICES"], "jax": jax.__version__, "optax": optax.__version__,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--phase", choices=["verify", "primary", "early", "historical"], default="primary")
    args = parser.parse_args()
    started = time.monotonic()
    output = args.root / "runs" / "ratios"
    gpu_environment(output)
    deadline = started + args.seconds - 120
    if args.phase == "verify":
        measurements = []
        for n in (512, 1024):
            g = core.geometry(n)
            cs, gs = core.coordinate_scales(g, "both")
            c, gamma = core.initial_physical(g, 0, "envelope")
            state = core.initial_state(core.to_params(c, gamma, cs, gs), core.optimizer("adam"))
            state = run.stack_states([state] * 5)
            knots = jnp.asarray([primary_schedule(.01, arm) for arm in ARMS] + [primary_schedule(.001)])
            kernel = chunk(n, "mixed", 16, 100, True)
            start = time.monotonic()
            state, evidence = kernel(state, 0, knots)
            jax.block_until_ready(state)
            compile_seconds = time.monotonic() - start
            start = time.monotonic()
            for i in range(10):
                state, evidence = kernel(state, 100 + 100 * i, knots)
            jax.block_until_ready(state)
            measurements.append({"n": n, "batch": 5, "compile_seconds": compile_seconds,
                                 "1000_batched_updates_seconds": time.monotonic() - start,
                                 "finite": bool(np.isfinite(np.asarray(evidence[0])).all())})
        run.write_json(output / "throughput.json", measurements)
        print(json.dumps(measurements), flush=True)
        if not all(m["finite"] for m in measurements):
            raise FloatingPointError("GPU verification failed")
        return
    acquire, tails = main_groups(args.root, args.seed)
    run.write_json(output / f"manifest_acquisition_s{args.seed}.json", [asdict(b) for group in acquire for b in group])
    if args.phase == "primary":
        for end in range(WINDOW, 160_001, WINDOW):
            for group in acquire:
                if time.monotonic() >= deadline:
                    return
                if end >= max(b.source_step for b in group):
                    advance_group(output, group, end, deadline)
        groups, start, stop = tails, 180_000, 1_460_000
    elif args.phase == "early":
        groups = []
        for n, target in [(512, "sine"), (1024, "mixed")]:
            high = next(g[1] for g in acquire if g[1].case.n == n and g[1].case.target == target)
            source = Path(high.source) if high.source else high.folder(output)
            groups.append([make_branch(n, target, args.seed, .01, "early_slow_geometry", early_schedule(),
                                      source, 20_000, 160_000)])
        start, stop = 40_000, 340_000
    else:
        b = make_branch(512, "sine", args.seed, .01, "historical_high_decay",
                        [[320_000, .01, .01], [400_000, 1e-6, 1e-6]],
                        args.root / "runs" / "pilot" / run.Case(seed=args.seed, initialization="envelope",
                                                                  rate_r=.01, rate_g=.01).key,
                        320_000, 400_000)
        groups, start, stop = [[b]], 340_000, 1_680_000
    run.write_json(output / f"manifest_{args.phase}_s{args.seed}.json", [asdict(b) for group in groups for b in group])
    for end in range(start, stop + 1, WINDOW):
        for group in groups:
            if time.monotonic() >= deadline:
                return
            advance_group(output, group, end, deadline)


if __name__ == "__main__":
    main()
