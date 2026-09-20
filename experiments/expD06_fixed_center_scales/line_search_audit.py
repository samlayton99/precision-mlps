"""Paired numerical audit of Armijo loss evaluation at saved frozen dictionaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import core, diagnostics, ratio, readout_solvers as solvers, run


METHODS = {"direct_loss": False, "loss_change": True}


def advance(source, output, origin, frontier, deadline):
    meta = json.loads((source / "dictionary.json").read_text())
    case = run.Case(**meta["case"])
    specs = [s for s in meta["specifications"] if s[2] < 3]
    assert len(specs) == 6
    folder = output / source.name
    folder.mkdir(parents=True, exist_ok=True)
    latest = [folder / method / spec[0] / "latest.json" for method in METHODS for spec in specs]
    if all(p.exists() for p in latest) and min(json.loads(p.read_text())["step"] for p in latest) >= frontier:
        return
    g = core.geometry(case.n)
    with np.load(source / "reference.npz") as reference:
        gamma = reference["gamma"]
    x = np.linspace(-1, 1, case.n * case.samples_per_cell + 1)
    y = jnp.asarray(core.target(x, case.target, np) / np.sqrt(len(x)))
    dictionaries = {coord: solvers.dictionary(x, g, gamma, coord) / np.sqrt(len(x))
                    for coord in ("prescribed", "differences")}
    maps = {coord: solvers.coordinate_map(g, gamma, coord) for coord in dictionaries}
    b = jnp.asarray(np.stack([dictionaries[s[1]] for s in specs]))
    transform = jnp.asarray(np.stack([maps[s[1]] for s in specs]))
    algorithms = jnp.asarray([s[2] for s in specs])
    initial, hashes = [], {}
    for label, _, _, _ in specs:
        path = source / label / f"state_{origin:09d}.pkl"
        state, step = run.load_state(path)
        assert step == origin
        initial.append(state)
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {"source": str(source), "source_step": origin, "case": meta["case"],
                "source_state_sha256": hashes, "specifications": specs,
                "methods": {"direct_loss": "original full-loss Armijo evaluation",
                            "loss_change": "r dot B dz + 0.5 ||B dz||^2; restart a zero trial rate; GD trial 0.1 on gradient fallback"},
                "initialization": "identical saved parameters, moments, counters, and accepted rates",
                "objective": "half-MSE", "minimum_new_updates": 20000,
                "test_evaluation": False, "second_order_training": False}
    run.write_json(folder / "protocol.json", protocol)
    run.save_arrays(folder / "reference.npz", gamma=gamma, centers=g.centers, d=g.d)
    states, starts = {}, []
    for method in METHODS:
        members = []
        for i, spec in enumerate(specs):
            path = folder / method / spec[0]
            path.mkdir(parents=True, exist_ok=True)
            if (path / "latest.json").exists():
                at = json.loads((path / "latest.json").read_text())["step"]
                state, _ = run.load_state(path / f"state_{at:09d}.pkl")
            else:
                at, state = 0, initial[i]
            starts.append(at)
            members.append(state)
        states[method] = run.stack_states(members)
    if len(set(starts)) != 1:
        raise ValueError("Audit pairs must resume at the same saved frontier")
    at = starts[0]
    written = at
    traces, dense = {m: [] for m in METHODS}, {m: [] for m in METHODS}

    def save():
        nonlocal written
        val_x = diagnostics.midpoint_grid(case.validation_points)
        val_y = core.target(val_x, case.target, np)
        for method in METHODS:
            host = jax.device_get(states[method])
            trace = np.concatenate(traces[method], axis=1) if traces[method] else None
            detail = {k: np.concatenate([d[k] for d in dense[method]], axis=1)
                      for k in dense[method][0]} if dense[method] else None
            for i, (label, coord, _, _) in enumerate(specs):
                path = folder / method / label
                single = run.unstack_state(host, i)
                if trace is not None:
                    run.save_arrays(path / f"trace_{written:09d}_{at:09d}.npz", trace=trace[i], columns=solvers.TRACE_COLUMNS)
                if detail is not None:
                    ratio.save_dense(path / f"dense_{written:09d}_{at:09d}.npz", **{k: v[i] for k, v in detail.items()})
                c = maps[coord] @ single["z"]
                residual = dictionaries[coord] @ single["z"] - np.asarray(y)
                val = diagnostics.prediction(val_x, g.centers, c, gamma) - val_y
                row = {"step": at, "source_step": origin, "train_mse": float(residual @ residual),
                       "validation_mse": float(np.mean(val**2)), "eta": float(single["eta"]),
                       "coefficient_l1": float(np.abs(c).sum()), "gradient_norm": float(np.linalg.norm(dictionaries[coord].T @ residual)),
                       "minimum_completed": at >= 20000, "status": "continuing",
                       **{k: int(single[k]) - int(initial[i][k]) for k in
                          ("gradient_evaluations", "loss_evaluations", "fallbacks", "stagnations")}}
                run.save_state(path / f"state_{at:09d}.pkl", single, at)
                run.save_arrays(path / f"checkpoint_{at:09d}.npz", c=c, gamma=gamma, **single)
                run.write_json(path / f"metrics_{at:09d}.json", row)
                run.write_json(path / "latest.json", row)
            traces[method].clear()
            dense[method].clear()
        written = at
        print(json.dumps({"dictionary": source.name, "source_step": origin, "additional_steps": at}), flush=True)

    if at == 0:
        save()
    while at < frontier and time.monotonic() < deadline:
        offset = at % ratio.WINDOW
        capture = offset < 2048 or offset >= ratio.WINDOW - 2048
        boundary = 2048 if offset < 2048 else ratio.WINDOW - 2048 if offset < ratio.WINDOW - 2048 else ratio.WINDOW
        length = min(100, boundary - offset, frontier - at)
        for method, stable in METHODS.items():
            states[method], evidence = solvers.linear_chunk(length, capture, True, stable)(
                states[method], b, y, transform, algorithms, origin + at)
            if capture:
                trace, detail = jax.device_get(evidence)
                dense[method].append({**detail, "step": np.broadcast_to(np.arange(at, at + length), (len(specs), length))})
            else:
                trace = np.asarray(evidence)
            if not np.isfinite(trace).all():
                raise FloatingPointError("Nonfinite audit trace")
            traces[method].append(trace)
        at += length
        if at % ratio.WINDOW == 0:
            save()
    if at != written:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True, help="Worker partition, not an additional random seed")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds - 120
    source_root = args.root / "runs" / "ratios" / "solvers"
    output = args.root / "runs" / "ratio_line_search_audit"
    ratio.gpu_environment(output)
    probe = solvers.initial_state(np.zeros(1), 0)
    checked, _ = solvers.linear_chunk(1, False, False, True)(
        probe, jnp.array([[10.], [0.]]), jnp.array([1e-10, 1.]), jnp.eye(1), 0, 0)
    resolved = abs(10 * float(checked["z"][0]) - 1e-10) < 1e-10
    run.write_json(output / f"verification_worker_{args.seed}.json",
                   {"implementation_check_only": True, "hidden_descent_resolved": resolved,
                    "gpu": str(jax.devices()), "fp64": bool(jax.config.x64_enabled)})
    if not resolved:
        raise ArithmeticError("GPU did not resolve the small Armijo loss change")
    dictionaries = sorted(source_root.glob("*/dictionary.json"))
    assert len(dictionaries) == 30
    origin = min(json.loads((p.parent / s[0] / "latest.json").read_text())["step"]
                 for p in dictionaries for s in json.loads(p.read_text())["specifications"]) // 20000 * 20000
    assert origin >= 20000
    # Balance the expensive zero-rate/40-trial case against the other dictionaries.
    names = (["quadratic_N1024_uniform", "sine_N512_uniform", "mixed_N512_uniform"] if args.seed == 0 else
             ["sine_N1024_uniform", "mixed_N1024_uniform", "quadratic_N512_uniform",
              "quadratic_N512_adam_both_envelope_s0_012a20b62876_learned"])
    sources = [source_root / name for name in names]
    run.write_json(output / f"manifest_worker_{args.seed}.json", {"source_step": origin, "sources": list(map(str, sources)),
        "selection": "all six uniform dictionaries plus the learned dictionary with recorded momentum stagnation"})
    for frontier in range(20000, 200001, 20000):
        for source in sources:
            if time.monotonic() >= deadline:
                return
            advance(source, output, origin, frontier, deadline)


if __name__ == "__main__":
    main()
