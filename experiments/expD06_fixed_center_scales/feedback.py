"""One training-only feedback rule; every decision is replayable from saved evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import svd

from . import core, diagnostics, ratio, run


def decide(means, fractions, interventions, step, last_end, counterfactual=None):
    """Operational thresholds, not theoretical constants or validation selection."""
    if interventions >= 2:
        return "hold_limit"
    if interventions and step < last_end + 40_000:
        return "hold_cooldown"
    if len(means) < 3:
        return "hold_insufficient_history"
    if any(b < .95 * a for a, b in zip(means[-3:-1], means[-2:])):
        return "hold_improving"
    if min(fractions) <= .95:
        return "hold_out_of_span"
    if interventions == 0:
        return "slow_geometry"
    if counterfactual is None:
        return "test_readout"
    if counterfactual["mean_delta_mse"] < 0 and counterfactual["decrease_fraction"] >= .95:
        return "increase_readout"
    return "reject_readout"


def span_fractions(folder, step):
    case = run.Case(**json.loads((folder / "case.json").read_text()))
    g = core.geometry(case.n)
    with np.load(folder / f"checkpoint_{step:09d}.npz") as cp:
        x = np.linspace(-1, 1, case.n * case.samples_per_cell + 1)
        a = diagnostics.features(x, g.centers, cp["gamma"]) / np.sqrt(len(x))
        r = a @ cp["c"] - core.target(x, case.target, np) / np.sqrt(len(x))
        u, s, _ = svd(a * g.d, full_matrices=False)
        projection = u.T @ r
    return [float(np.sum(projection[s > tau * s[0]]**2) / max(r @ r, np.finfo(float).tiny))
            for tau in (1e-10, 1e-12)]


def readout_counterfactual(folder, step):
    case = run.Case(**json.loads((folder / "case.json").read_text()))
    g = core.geometry(case.n)
    x = np.linspace(-1, 1, case.n * case.samples_per_cell + 1)
    y = core.target(x, case.target, np)
    changes, states = [], []
    # Predeclared offsets do not depend on losses, updates, or validation.
    wanted = set((step - 2048 + ratio.dense_sample_indices()).tolist())
    for path in sorted(folder.glob("dense_*.npz")):
        _, lo, hi = path.stem.split("_")
        if int(hi) <= step - 2048 or int(lo) >= step:
            continue
        with np.load(path) as dense:
            for i, t in enumerate(dense["step"]):
                if int(t) not in wanted:
                    continue
                a = diagnostics.features(x, g.centers, dense["gamma"][i])
                r = a @ dense["c"][i] - y
                dr = a @ (3 * dense["delta_c"][i])
                changes.append(float(np.mean(2 * r * dr + dr**2)))
                states.append(int(t))
    if set(states) != wanted:
        raise ValueError("Incomplete counterfactual window")
    return {"mean_delta_mse": float(np.mean(changes)), "decrease_fraction": float(np.mean(np.array(changes) < 0)),
            "steps": states, "delta_mse": changes, "multiplier": 3.}


def feedback_step(output, branch, end, deadline):
    folder = branch.folder(output)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "feedback.json"
    ledger = json.loads(path.read_text()) if path.exists() else {
        "knots": [list(k) for k in branch.knots], "events": [], "interventions": 0,
        "last_end": branch.source_step, "policy": "two stalled windows; span>95% at 1e-10 and 1e-12",
        "counterfactual_sampling": "64 strata of 32 states, fixed jitter seed 391; training data only"}
    at = ratio.advance_group(output, [branch], end, deadline, [ledger["knots"]], ledger["last_end"])
    if at != end or at % ratio.WINDOW or any(e["step"] == at for e in ledger["events"]):
        return at
    means = []
    for right in range(max(branch.source_step + ratio.WINDOW, at - 2 * ratio.WINDOW), at + 1, ratio.WINDOW):
        losses = run.trace_window_losses(folder, right - ratio.WINDOW, right)
        if losses is None:
            raise ValueError("Incomplete feedback loss window")
        means.append(float(2 * np.mean(losses)))
    preliminary = decide(means, [1., 1.], ledger["interventions"], at, ledger["last_end"])
    fractions = span_fractions(folder, at) if preliminary in {"slow_geometry", "test_readout"} else None
    action = decide(means, fractions, ledger["interventions"], at, ledger["last_end"]) if fractions else preliminary
    cf = None
    if action == "test_readout":
        cf = readout_counterfactual(folder, at)
        action = decide(means, fractions, ledger["interventions"], at, ledger["last_end"], cf)
    event = {"step": at, "window_mse": means, "span_fractions": fractions,
             "cutoffs": [1e-10, 1e-12], "decision": action, "counterfactual": cf}
    if action in {"slow_geometry", "increase_readout"}:
        old = np.asarray(ratio.schedule(at, ledger["knots"])).tolist()
        new = [old[0], old[1] / 10] if action == "slow_geometry" else [3 * old[0], old[1]]
        ledger["knots"].extend([[at, *old], [at + ratio.WINDOW, *new]])
        ledger["interventions"] += 1
        ledger["last_end"] = at + ratio.WINDOW
    ledger["events"].append(event)
    run.write_json(path, ledger)
    return at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    output = args.root / "runs" / "ratios"
    ratio.gpu_environment(output)
    acquire, _ = ratio.main_groups(args.root, args.seed)
    branches = [ratio.make_branch(high.case.n, high.case.target, args.seed, .01, "feedback",
                                 [[160_000, 1e-6, 1e-6]], high.folder(output), 160_000, 160_000)
                for _, high in acquire]
    run.write_json(output / f"manifest_feedback_s{args.seed}.json", [asdict(b) for b in branches])
    for end in range(180_000, 1_460_001, ratio.WINDOW):
        for branch in branches:
            if time.monotonic() >= start + args.seconds - 120:
                return
            feedback_step(output, branch, end, start + args.seconds - 120)


if __name__ == "__main__":
    main()
