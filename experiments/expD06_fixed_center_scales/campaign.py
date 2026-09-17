"""Two independent GPU workers, broad base/rate-ratio tuning, persistent budgets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import numpy as np
import optax

from . import core
from .run import Case, MIN_STEPS, TERMINAL, run_batch, write_json


def pilot_manifest(optimizer, expanded=False):
    bases = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if expanded else [1e-4, 1e-3, 1e-2]
    ratios = [.01, .1, 1., 10., 100.] if expanded else [.1, 1., 10.]
    g = core.geometry(512)
    records = []
    for initialization in ["xavier", "envelope"]:
        for arm in ["both", "raw"]:
            for base in bases:
                for ratio in ratios:
                    rr, rg = base, base * ratio
                    if arm == "raw":
                        rr *= g.ordinary_alpha if optimizer == "gd" else np.sqrt(g.ordinary_alpha)
                        rg *= g.h**-2 if optimizer == "gd" else g.h**-1
                    for seed in [0, 1]:
                        case = Case(optimizer=optimizer, arm=arm, initialization=initialization,
                                    seed=seed, rate_r=float(rr), rate_g=float(rg))
                        records.append({"case": asdict(case), "key": case.key, "base_lr": base,
                                        "bandwidth_to_readout_ratio": ratio,
                                        "grid": "native" if arm == "both" else "ordinary_update_matched"})
            if arm == "raw":
                # Ordinary numerical-rate baselines remain visible alongside the tuned range.
                for base in [1e-4, 1e-3, 1e-2]:
                    for seed in [0, 1]:
                        case = Case(optimizer=optimizer, arm=arm, initialization=initialization,
                                    seed=seed, rate_r=base, rate_g=base)
                        records.append({"case": asdict(case), "key": case.key, "base_lr": base,
                                        "bandwidth_to_readout_ratio": 1., "grid": "native_control"})
    return records


def selected_manifest(root, optimizer, allow_provisional=False):
    """Select on trained validation RMS only; require two seeds per candidate."""
    groups = defaultdict(list)
    for folder in Path(root).iterdir():
        if not folder.is_dir() or not (folder / "latest.json").exists():
            continue
        case = Case(**json.loads((folder / "case.json").read_text()))
        if case.optimizer != optimizer or case.n != 512 or case.target != "sine" or case.seed not in (0, 1):
            continue
        latest = json.loads((folder / "latest.json").read_text())
        if latest["step"] < MIN_STEPS or latest["status"] == "nonfinite":
            continue
        if not allow_provisional and latest["status"] not in TERMINAL:
            continue
        key = (case.arm, case.initialization, case.rate_r, case.rate_g)
        groups[key].append((case, latest))
    winners = {}
    for key, entries in groups.items():
        if {c.seed for c, _ in entries} != {0, 1}:
            continue
        score = float(np.mean([np.log(max(r["history"][-1]["validation"]["rms"], 1e-300)) for _, r in entries]))
        pair = key[:2]
        if pair not in winners or score < winners[pair][0]:
            winners[pair] = (score, entries[0][0])
    return [asdict(case) for _, case in winners.values()]


def confirmation_manifest(selected):
    records = []
    for config in selected:
        base = Case(**config)
        for n in [512, 1024, 2048]:
            for target in ["sine", "quadratic", "mixed"]:
                for seed in range(2, 7):
                    case = replace(base, n=n, target=target, seed=seed)
                    records.append({"case": asdict(case), "key": case.key, "grid": "width_target_transfer"})
    return records


def control_manifest(selected, mode):
    records = []
    for config in selected:
        base = Case(**config)
        if mode == "halo":
            if (base.arm, base.initialization) != ("both", "envelope"):
                continue
            for n in [512, 1024]:
                for seed in [2, 3, 4]:
                    for halo_init in ["full", "ordinary"]:
                        for halo_metric in ["full", "ordinary"]:
                            case = replace(base, n=n, seed=seed, halo_init=halo_init, halo_metric=halo_metric)
                            records.append({"case": asdict(case), "key": case.key, "grid": "halo_factorial"})
        elif mode == "sampling":
            for n in [512, 1024]:
                for target in ["sine", "mixed"]:
                    for samples in [16, 32]:
                        case = replace(base, n=n, target=target, seed=2, samples_per_cell=samples)
                        records.append({"case": asdict(case), "key": case.key, "grid": "sampling_refinement"})
        else:
            raise ValueError(f"Unknown control mode {mode!r}")
    return records


def pending_groups(records, output, frontier):
    groups = defaultdict(list)
    for record in records:
        case = Case(**record["case"])
        latest_file = output / case.key / "latest.json"
        step = 0
        if latest_file.exists():
            latest = json.loads(latest_file.read_text())
            step = latest["step"]
            if latest["status"] in TERMINAL or step >= frontier:
                continue
        groups[(case.n, case.optimizer, case.target, case.samples_per_cell, case.validation_points, step)].append(case)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--optimizer", choices=["gd", "adam"], required=True)
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--mode", choices=["pilot", "confirmation", "halo", "sampling"], default="pilot")
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--max-frontier", type=int, help="Pause for analysis here; unfinished runs remain continuing.")
    parser.add_argument("--gpu-hours", type=float, default=11.9,
                        help="Cumulative worker limit across resumptions; two workers stay below 24 GPU-hours.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--session-seconds", type=float,
                        help="Checkpoint and exit 75 for reconnection; the scientific trajectory remains continuing.")
    args = parser.parse_args()
    if args.mode != "pilot" and args.selected is None:
        parser.error("Confirmation and controls require a saved selection file.")
    if jax.default_backend() != "gpu":
        raise RuntimeError("The campaign requires one explicitly selected GPU per worker.")
    args.root.mkdir(parents=True, exist_ok=True)
    output = args.root / "runs" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        records = pilot_manifest(args.optimizer, args.expanded)
    else:
        selected = json.loads(args.selected.read_text())
        records = confirmation_manifest(selected) if args.mode == "confirmation" else control_manifest(selected, args.mode)
    records = [r for r in records if r["case"]["optimizer"] == args.optimizer]
    stage = "expanded" if args.expanded else "initial"
    write_json(args.root / f"manifest_{args.mode}_{args.optimizer}_{stage}.json", records)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    write_json(args.root / f"environment_{args.optimizer}_{os.getpid()}.json",
               {"jax": jax.__version__, "optax": optax.__version__, "devices": str(jax.devices()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "x64": jax.config.x64_enabled,
                "source_sha256": source_hashes, "minimum_steps": MIN_STEPS,
                "pip_freeze": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)})
    ledger_path = args.root / f"budget_{args.optimizer}.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"seconds_used": 0.0}
    if "active_since" in ledger:
        ledger["seconds_used"] += max(0, time.time() - ledger["active_since"])
    start = time.monotonic()
    remaining = max(0, args.gpu_hours * 3600 - ledger["seconds_used"])
    deadline = start + remaining
    session_deadline = min(deadline, start + args.session_seconds) if args.session_seconds else deadline
    ledger["active_since"] = time.time()
    write_json(ledger_path, ledger)
    frontier = MIN_STEPS
    last_batch_seconds = 30.0
    try:
        while time.monotonic() < session_deadline:
            groups = pending_groups(records, output, frontier)
            for cases in groups.values():
                for i in range(0, len(cases), args.batch_size):
                    if time.monotonic() >= session_deadline:
                        return 75 if session_deadline < deadline else None
                    # Reserve enough time to finish the 20k minimum for a newly started batch.
                    if deadline - time.monotonic() < max(60, 2 * last_batch_seconds):
                        return
                    batch = cases[i:i + args.batch_size]
                    batch_start = time.monotonic()
                    results = run_batch(batch, output, frontier, session_deadline)
                    last_batch_seconds = time.monotonic() - batch_start
                    print(json.dumps({"frontier": frontier, "seconds": last_batch_seconds,
                                      "cases": [c.key for c in batch],
                                      "status": [r["status"] for r in results],
                                      "validation_rms": [r["history"][-1].get("validation", {}).get("rms") for r in results]}), flush=True)
                    if time.monotonic() >= session_deadline:
                        return 75 if session_deadline < deadline else None
            print(json.dumps({"frontier_complete": frontier, "optimizer": args.optimizer,
                              "worker_seconds": time.monotonic() - start}), flush=True)
            if args.max_frontier and frontier >= args.max_frontier:
                break
            if not pending_groups(records, output, frontier * 2):
                break
            frontier *= 2
        if session_deadline < deadline and time.monotonic() >= session_deadline:
            return 75
    finally:
        ledger["seconds_used"] += time.monotonic() - start
        ledger.pop("active_since", None)
        ledger["limit_seconds"] = args.gpu_hours * 3600
        write_json(ledger_path, ledger)


if __name__ == "__main__":
    raise SystemExit(main())
