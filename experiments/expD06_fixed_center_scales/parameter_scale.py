"""Two-hour paired normalization campaign; one shared constant rate per trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import difference_analysis as analysis, difference_training as training, ratio, run

MAPS = ("scaled", "parameter_scale")
OPTIMIZERS = ("gd", "adam")
RATES = [factor*10.**power for power in range(-5, 0) for factor in (1, 3)]


def case(optimizer, coordinates, eta, n=512, seed=0):
    return dict(campaign="parameter_scale", optimizer=optimizer, coordinates=coordinates,
                eta=eta, n=n, seed=seed)


def pilot():
    return [case(opt, coord, eta) for opt in OPTIMIZERS for coord in MAPS for eta in RATES]


def table(root, end=100000):
    rows = []
    for folder in sorted(root.glob("*_N*_eta*")):
        if not (folder/"latest.json").exists():
            continue
        config = json.loads((folder/"case.json").read_text())
        status = json.loads((folder/"latest.json").read_text())
        row = {k: config[k] for k in ("campaign", "optimizer", "coordinates", "eta", "n", "seed")}
        row.update(key=folder.name, **status, comparison_step=end)
        row["eligible"] = status["completed_updates"] >= end and not status["failed_update"]
        if row["eligible"]:
            trace = analysis.read_trace(folder, end)
            assert np.all(np.isfinite(trace))
            np.testing.assert_array_equal(trace[:, -1], np.full(end, config["eta"]))
            window = 2*trace[-20000:, 0]
            row.update(window_mean_mse=float(window.mean()), window_std_mse=float(window.std()),
                       window_min_mse=float(window.min()), window_max_mse=float(window.max()),
                       previous_mean_mse=float(2*trace[-40000:-20000, 0].mean()))
        rows.append(row)
    return rows


def select(rows):
    selected = []
    for optimizer in OPTIMIZERS:
        for coord in MAPS:
            candidates = [r for r in rows if r["eligible"] and r["n"] == 512 and r["seed"] == 0
                          and (r["optimizer"], r["coordinates"]) == (optimizer, coord)]
            if not candidates:
                raise ValueError(f"No finite 100k pilot for {optimizer}, {coord}")
            best = min(candidates, key=lambda r: (r["window_mean_mse"], r["eta"]))
            selected.append(case(optimizer, coord, best["eta"]))
    return selected


def followups(selected):
    confirmation = []
    for opt in OPTIMIZERS:
        rates = sorted({c["eta"] for c in selected if c["optimizer"] == opt})
        confirmation += [case(opt, coord, eta, n, seed) for n, seed in ((512,1),(1024,0),(1024,1))
                         for coord in MAPS for eta in rates]
    continuation = [dict(c, n=n, seed=seed) for c in selected for n in (512,1024) for seed in (0,1)]
    return confirmation, continuation


def boundary(selected):
    result = []
    for opt in OPTIMIZERS:
        rates = {c["eta"] for c in selected if c["optimizer"] == opt}
        extra = ([1e-6,3e-6] if min(RATES) in rates else []) + ([1.,3.] if max(RATES) in rates else [])
        result += [case(opt, coord, eta) for coord in MAPS for eta in extra]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--worker", type=int, choices=(0,1))
    parser.add_argument("--frontier", type=int, default=100000)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        run.write_json(args.root/"pilot.json", pilot())
        return
    if args.select:
        rows = table(args.root)
        selected = select(rows)
        confirmation, continuation = followups(selected)
        for name, value in (("pilot_summary",rows),("selected",selected),("confirmation",confirmation),
                            ("continuation",continuation),("boundary",boundary(selected))):
            run.write_json(args.root/f"{name}.json", value)
        print(json.dumps(selected), flush=True)
        return
    if args.worker is None or args.manifest is None:
        parser.error("Training requires --worker and --manifest")
    if args.frontier < 100000:
        parser.error("Scientific trials require at least 100k updates")
    started = time.monotonic()
    if args.require_gpu:
        ratio.gpu_environment(args.root)
    cases = json.loads(args.manifest.read_text())
    groups = {}
    for config in cases:
        if config["optimizer"] == OPTIMIZERS[args.worker]:
            groups.setdefault((config["n"], config["seed"], config["coordinates"]), []).append(config)
    run.write_json(args.root/f"worker_{args.worker}_manifest.json", dict(cases=cases, frontier=args.frontier,
                   seconds=args.seconds, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for frontier in range(100000, args.frontier+1, 100000):
        for group in groups.values():
            if time.monotonic() >= started+args.seconds-30:
                return
            training.advance_group(args.root, group, frontier, deadline=started+args.seconds-30)


if __name__ == "__main__":
    main()
