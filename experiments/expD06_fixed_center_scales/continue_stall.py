"""Eight paired Adam continuations: geometry freezing crossed with one LR schedule."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core, run

SOURCE_STEP = 320_000
DECAY_STEPS = 80_000
LABELS = ("joint_constant", "joint_decay", "frozen_constant", "frozen_decay")


def learning_rate(step, decay):
    phase = jnp.minimum(step / DECAY_STEPS, 1.)
    cosine = 1e-6 + .5 * (1e-3 - 1e-6) * (1 + jnp.cos(jnp.pi * phase))
    return jnp.where(decay, cosine, 1e-3)


def dense_step(step):
    return step < 2048 or step % 20000 >= 20000 - 2048


def scheduled_chunk(g, samples_per_cell, length, capture=False, batched=True):
    one = core.make_chunk(g, "adam", "sine", samples_per_cell, 1, capture=capture)
    cs, gs = core.coordinate_scales(g, "both")

    def advance(state, start, frozen, decay):
        def step(current, index):
            eta = learning_rate(start + index, decay)
            updated, evidence = one(current, cs, gs, eta, jnp.where(frozen, 0., eta))
            return updated, jax.tree.map(lambda x: x[0], evidence)
        return jax.lax.scan(step, state, jnp.arange(length))
    return jax.jit(jax.vmap(advance, in_axes=(0, None, 0, 0)) if batched else advance)


def source_case(seed):
    return run.Case(initialization="envelope", seed=seed, rate_r=1e-3, rate_g=1e-3)


def worker(root, seed, deadline):
    case = source_case(seed)
    source = root / "runs" / "pilot" / case.key
    output = root / "runs" / "stall"
    g = core.geometry(case.n)
    cs, gs = core.coordinate_scales(g, "both")
    _, evaluate, _, _ = run.kernels(case.n, "adam", case.target, case.samples_per_cell,
                                  case.validation_points)
    folders = [output / label / case.key for label in LABELS]
    starts, states, histories = [], [], []
    origin, original_step = run.load_state(source / f"state_{SOURCE_STEP:09d}.pkl")
    assert original_step == SOURCE_STEP
    for label, folder in zip(LABELS, folders):
        folder.mkdir(parents=True, exist_ok=True)
        run.write_json(folder / "case.json", asdict(case))
        shutil.copyfile(source / "reference.npz", folder / "reference.npz")
        run.write_json(folder / "continuation.json", {
            "label": label, "source": str(source), "source_step": SOURCE_STEP,
            "source_state_sha256": hashlib.sha256((source / f"state_{SOURCE_STEP:09d}.pkl").read_bytes()).hexdigest(),
            "step_convention": "All filenames and traces use additional updates after source_step",
            "initial_rate": .001, "final_decay_rate": 1e-6, "decay_steps": DECAY_STEPS,
            "frozen_geometry": label.startswith("frozen"), "decay": label.endswith("decay"),
            "convergence_step_offset": DECAY_STEPS,
            "frozen_moments": "Geometry moments evolve but never affect the frozen parameters or readout moments"})
        latest = folder / "latest.json"
        if latest.exists():
            record = json.loads(latest.read_text())
            state, start = run.load_state(folder / f"state_{record['step']:09d}.pkl")
            history = record["history"]
        else:
            state, start, history = origin, 0, []
        states.append(state)
        starts.append(start)
        histories.append(history)
    # A killed save can advance only part of the paired batch. Roll back to the
    # last checkpoint common to all branches; deterministic traces are replaced.
    at = min(starts)
    if len(set(starts)) != 1:
        states = [run.load_state(f / f"state_{at:09d}.pkl")[0] if at else origin for f in folders]
        histories = [[r for r in h if r["step"] <= at] for h in histories]
    state = run.stack_states(states)
    frozen = jnp.asarray([False, False, True, True])
    decay = jnp.asarray([False, True, False, True])
    cs_batch = jnp.broadcast_to(cs, (4, len(cs)))
    gs_batch = jnp.full(4, gs)
    traces, dense = [], []
    trace_start = at
    statuses = [run.convergence_status(h, f, step_offset=DECAY_STEPS) if h else "continuing"
                for h, f in zip(histories, folders)]

    @lru_cache(maxsize=None)
    def kernel(length, capture):
        return scheduled_chunk(g, case.samples_per_cell, length, capture)

    def save():
        nonlocal trace_start
        if traces:
            joined = np.concatenate(traces, axis=1)
            for i, folder in enumerate(folders):
                run.save_arrays(folder / f"trace_{trace_start:09d}_{at:09d}.npz",
                                start_step=trace_start, trace=joined[i], columns=np.asarray(core.TRACE_COLUMNS))
            traces.clear()
        if dense:
            joined = {k: np.concatenate([d[k] for d in dense], axis=1) for k in dense[0]}
            for i, folder in enumerate(folders):
                run.save_arrays(folder / f"dense_{trace_start:09d}_{at:09d}.npz",
                                **{k: value[i] for k, value in joined.items()})
            dense.clear()
        rates = learning_rate(at, decay)
        arrays = jax.device_get(evaluate(state, cs_batch, gs_batch, rates,
                                        jnp.where(frozen, 0., rates)))
        elapsed_after_decay = at - DECAY_STEPS
        ratio = elapsed_after_decay // run.MIN_STEPS
        convergence_frontier = (elapsed_after_decay >= run.MIN_STEPS
                                and elapsed_after_decay % run.MIN_STEPS == 0
                                and ratio & (ratio - 1) == 0)
        for i, folder in enumerate(folders):
            losses = None
            if convergence_frontier:
                begin = DECAY_STEPS + (0 if ratio == 1 else elapsed_after_decay // 2)
                losses = run.trace_window_losses(folder, begin, at)
                if losses is None:
                    raise ValueError(f"Incomplete convergence window: {folder}, {begin}:{at}")
            row = run.record_checkpoint(folder.parent, case, run.unstack_state(state, i), at,
                                        {k: v[i] for k, v in arrays.items()}, losses)
            histories[i] = [r for r in histories[i] if r["step"] < at] + [row]
            statuses[i] = run.convergence_status(histories[i], folder, step_offset=DECAY_STEPS)
            run.write_json(folder / "latest.json", {"step": at, "status": statuses[i],
                            "history": histories[i], "source_step": SOURCE_STEP,
                            "min_steps": run.MIN_STEPS})
        trace_start = at
        if at % 20000 == 0:
            print(json.dumps({"seed": seed, "additional_steps": at, "rates": np.asarray(rates).tolist(),
                              "validation_mse": [h[-1]["validation"]["rms"]**2 for h in histories],
                              "status": statuses}), flush=True)

    if at == 0:
        save()
    while time.monotonic() < deadline and not all(s in run.TERMINAL for s in statuses):
        capture = dense_step(at)
        boundaries = [(at // 1000 + 1) * 1000, (at // 20000 + 1) * 20000]
        if at < 2048:
            boundaries.append(2048)
        end_start = at // 20000 * 20000 + 20000 - 2048
        if end_start > at:
            boundaries.append(end_start)
        length = min(100, min(boundaries) - at)
        state, evidence = kernel(length, capture)(state, at, frozen, decay)
        if capture:
            trace, detail = jax.device_get(evidence)
            dense.append({**detail, "step": np.broadcast_to(np.arange(at, at + length), (4, length))})
        else:
            trace = np.asarray(evidence)
        traces.append(trace)
        at += length
        if at % 1000 == 0 or not np.isfinite(trace).all():
            save()
        if not np.isfinite(trace).all():
            raise FloatingPointError("Nonfinite continuation; saved the failure before stopping the paired batch.")
    if at != trace_start:
        save()
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    parser.add_argument("--limit-seconds", type=float, default=7200., help="Cumulative per-seed GPU-worker cap")
    args = parser.parse_args()
    if not all(os.environ.get(k) for k in ["SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES"]):
        raise RuntimeError("Use an allocated Slurm GPU step with its GPU mask intact.")
    allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]], text=True)
    step = subprocess.check_output(["scontrol", "show", "step",
            f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"], text=True)
    if "JobState=RUNNING" not in allocation or "State=RUNNING" not in step or "gpu" not in step:
        raise RuntimeError("Scheduler did not confirm an active GPU step.")
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1 or not jax.config.x64_enabled:
        raise RuntimeError("Exactly one allocated GPU and FP64 are required.")
    run.write_json(args.root / f"environment_stall_{os.environ['SLURM_JOB_ID']}.json", {
        "jax": jax.__version__, "optax": optax.__version__, "devices": str(jax.devices()),
        "allocation": allocation, "step": step, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob("*.py")}})
    ledger_path = args.root / f"budget_stall_{args.seed}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"seconds_used": 0.}
    if "active_since" in ledger:
        raise RuntimeError("Reconcile the interrupted allocation before resuming its budget ledger.")
    started = time.monotonic()
    deadline = started + max(0., args.limit_seconds - ledger["seconds_used"] - 60.)
    ledger.update(active_since=time.time(), limit_seconds=args.limit_seconds,
                  slurm_job_id=os.environ["SLURM_JOB_ID"])
    run.write_json(ledger_path, ledger)
    try:
        worker(args.root, args.seed, deadline)
    finally:
        ledger["seconds_used"] += time.monotonic() - started
        ledger.pop("active_since", None)
        run.write_json(ledger_path, ledger)


if __name__ == "__main__":
    main()
